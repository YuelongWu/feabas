from collections import defaultdict
import argparse
import json
import os
import time
from datetime import datetime

from feabas import config, storage
from feabas.concurrent import submit_to_workers
from functools import partial
num_workers = 30

import cv2
import numpy as np
from scipy import sparse
import scipy.sparse.linalg as splinalg
from scipy.ndimage import distance_transform_edt, map_coordinates, grey_dilation, grey_erosion
from math import ceil, floor

from feabas import dal
from feabas.spatial import scale_coordinates
H5File = storage.h5file_class()

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="equalize brightness and contrast")
    parser.add_argument("--mode", metavar="mode", type=str, default='match')
    parser.add_argument("--start", metavar="start", type=int, default=0)
    parser.add_argument("--step", metavar="step", type=int, default=1)
    parser.add_argument("--stop", metavar="stop", type=int)
    parser.add_argument("--postfix", metavar="postfix", type=str, default='')
    parser.add_argument("--clahe", action='store_true')
    return parser.parse_args(args)


def match_main(sel_indx):
    mip = 3
    match_name_file = storage.join_paths(align_dir, 'bc_match_pair.txt')
    storage.makedirs(bc_dir, exist_ok=True)
    with storage.File(ts_spec_file, 'r') as f:
        rendered_mips_spec = json.load(f)
        rendered_mips_spec = {int(m): spec for m, spec in rendered_mips_spec.items()}
    with storage.File(ito_spec_file, 'r') as f:
        ito_mips_spec = json.load(f)
        ito_mips_spec = {int(m): spec for m, spec in ito_mips_spec.items()}
    vol_spec = rendered_mips_spec[mip]
    ito_spec = ito_mips_spec[mip]
    mlist = storage.list_folder_content(storage.join_paths(match_dir, '*.h5'))
    bnames = [os.path.basename(s).replace('.h5','') for s in mlist]
    with storage.File(match_name_file) as f:
        z_pairs = defaultdict(list)
        for line in f:
            if line.strip():
                z0, z1 = line.strip().split('\t')
                z_pairs[int(z0)]
                if '__to__'.join((z0, z1)) in bnames:
                    continue
                z_pairs[int(z0)].append(int(z1))
    z0s = sorted(list(z_pairs.keys()))
    if sel_indx is not None:
        z0s = z0s[sel_indx]
    for z0 in z0s:
        t0 = time.time()
        if len(z_pairs[z0]) == 0:
            continue
        match_sections(vol_spec, ito_spec, z0, z_pairs[z0], match_dir)
        print(f'z {z0} | {(time.time()-t0)/60/len(z_pairs[z0])} min')
    print('finished')


def match_sections(vol_spec, ito_spec, z0, z1s, outdir, **kwargs):
    read_size = kwargs.get('read_size', 2048)
    patch_size = kwargs.get('patch_size', 256)
    vol_loader = dal.TensorStoreLoader.from_json_spec(vol_spec)
    vol_schema = vol_loader.dataset.schema.to_json()
    resolution = vol_schema['dimension_units'][0][0]
    xmin, ymin, xmax, ymax = vol_loader.bounds
    x_sz = ceil((xmax - xmin) / read_size)
    y_sz = ceil((ymax - ymin) / read_size)
    rr = read_size / patch_size
    ind_x, ind_y = np.meshgrid(np.arange(x_sz), np.arange(y_sz))
    ind_x, ind_y = ind_x.ravel(), ind_y.ravel()
    result_shape = (5, int(x_sz * rr), int(y_sz * rr))
    n_blk = ind_x.size
    batch_sz = max(1, floor(n_blk / num_workers))
    arglist = [(ind_x[k:(k+batch_sz)], ind_y[k:(k+batch_sz)]) for k in range(0, n_blk, batch_sz)]
    tfunc = partial(_tfunc_match_2d_bboxes, z0=z0, z1s=z1s, vol_spec=vol_spec, ito_spec=ito_spec, read_size=read_size, patch_size=patch_size)
    out = {(z0, z1): None for z1 in z1s}
    for res in submit_to_workers(tfunc, args=arglist, num_workers=num_workers):
        ind_x_patch, ind_y_patch, stats = res
        for zzs, vals in stats.items():
            if vals is None:
                continue
            if out[zzs] is None:
                out[zzs] = np.zeros(result_shape)
            for k in range(result_shape[0]):
                out[zzs][k, ind_x_patch, ind_y_patch] = vals[k]
    for zz, vals in out.items():
        if vals is None:
            continue
        outname = storage.join_paths(outdir, '__to__'.join(str(z) for z in zz)+'.h5')
        with H5File(outname, 'w') as f:
            f.create_dataset('mm', data=vals[0:2], compression="gzip")
            f.create_dataset('dd', data=vals[2:4], compression="gzip")
            f.create_dataset('wt', data=vals[4], compression="gzip")
            f.create_dataset('resolution', data=resolution)
            f.create_dataset('patch_size', data=patch_size)


def _tfunc_match_2d_bboxes(ind_x, ind_y, z0, z1s, vol_spec, ito_spec, read_size, patch_size, **kwargs):
    t0 = time.time()
    t_read = 0
    t_read1 = 0
    cvg_thresh = kwargs.get('cvg_thresh', 1.8)
    vol_loader = dal.TensorStoreLoader.from_json_spec(vol_spec)
    ito_loader = dal.TensorStoreLoader.from_json_spec(ito_spec)
    out = {(z0, z1): None for z1 in z1s}
    xmin, ymin, _, _ = vol_loader.bounds
    N_subpatch = int(read_size / patch_size)
    N_read = ind_x.size
    N_blk = int(N_read * (N_subpatch**2))
    lxx, lyy = np.meshgrid(np.arange(N_subpatch), np.arange(N_subpatch))
    ind_x_patch = np.repeat(ind_x.ravel()*N_subpatch, N_subpatch**2) + np.tile(lxx.ravel(), N_read)
    ind_y_patch = np.repeat(ind_y.ravel()*N_subpatch, N_subpatch**2) + np.tile(lyy.ravel(), N_read)
    img0 = None
    ito0 = None
    for k, ixy in enumerate(zip(ind_x_patch, ind_y_patch)):
        ix, iy = ixy
        ix_loc, iy_loc = ix % N_subpatch, iy % N_subpatch
        if (ix_loc == 0) and (iy_loc == 0):
            ix0, iy0 = int(ix / N_subpatch), int(iy / N_subpatch)
            bbox_2d_read = (ix0*read_size+xmin, iy0*read_size+ymin, (ix0+1)*read_size+xmin, (iy0+1)*read_size+ymin)
            t1 = time.time()
            img0, img_cache = _read_z_with_cache(vol_loader, bbox_2d_read, z0)
            ito0, ito_cache = _read_z_with_cache(ito_loader, bbox_2d_read, z0)
            t_read += time.time() - t1
        if (img0 is None) or (ito0 is None):
            continue
        loc_indx = (slice(ix_loc * patch_size, (ix_loc+1) * patch_size),
                    slice(iy_loc * patch_size, (iy_loc+1) * patch_size),
                    Ellipsis)
        ito0_t = ito0[loc_indx]
        cvg0 = 0
        for z1 in z1s:
            t1 = time.time()
            ito1, ito_cache = _read_z_with_cache(ito_loader, bbox_2d_read, z1, cache=ito_cache)
            t_read1 += time.time() - t1
            if ito1 is None:
                continue
            ito1_t = ito1[loc_indx]
            valid_t = (ito0_t > 0) & (ito1_t > 0)
            if not np.any(valid_t):
                continue
            img0_t = img0[loc_indx][valid_t]
            if np.all(img0_t == 0, axis=None):
                continue
            t1 = time.time()
            img1, img_cache = _read_z_with_cache(vol_loader, bbox_2d_read, z1, cache=img_cache)
            t_read1 += time.time() - t1
            if img1 is None:
                continue
            img1_t = img1[loc_indx][valid_t]
            if np.all(img1_t == 0, axis=None):
                continue
            m0 = np.mean(img0_t, axis=None)
            m1 = np.mean(img1_t, axis=None)
            tt0 = grey_dilation(img0_t, size=(3,)) - grey_erosion(img0_t, size=(3,))
            tt1 = grey_dilation(img1_t, size=(3,)) - grey_erosion(img1_t, size=(3,))
            idx_tt = (tt0 > np.mean(tt0)) & (tt1 > np.mean(tt1))
            if not np.any(idx_tt):
                continue
            d0 = np.mean(tt0[idx_tt])
            d1 = np.mean(tt1[idx_tt])
            corr = np.sum((img0_t - m0) * (img1_t - m1)) / ((np.std(img0_t)*np.std(img1_t)).clip(1e-3,None))
            wt = np.mean(valid_t, axis=None) * (max(0, corr) ** 0.5)
            if wt == 0:
                continue
            # idx_upp = (img0_t >= m0) & (img1_t >= m1)
            # idx_low = (img0_t <= m0) & (img1_t <= m1)
            # wt = 2 * (np.sum(idx_upp) * np.sum(idx_low))**0.5 / (patch_size**2)
            # if wt == 0:
            #     continue
            # d0 = np.mean(img0_t[idx_upp]) - np.mean(img0_t[idx_low])
            # d1 = np.mean(img1_t[idx_upp]) - np.mean(img1_t[idx_low])
            if out[(z0, z1)] is None:
                out[(z0, z1)] = np.zeros((5, N_blk), dtype=np.float32)
            out[(z0, z1)][:,k] = [m0, m1, d0, d1, wt]
            cvg0 += np.sum(valid_t) / max(1, np.sum(ito0_t > 0))
            if cvg0 > cvg_thresh:
                break
    # print((t_read/(time.time()-t0), t_read1/(time.time()-t0)))
    return ind_x_patch, ind_y_patch, out


def _read_z_with_cache(vol_loader, bbox_2d, z, cache=None):
    if isinstance(vol_loader, dict):
        vol_loader = dal.TensorStoreLoader.from_json_spec(vol_loader)
    shp_z = vol_loader.dataset.schema.chunk_layout.read_chunk.shape[2]
    ori_z = vol_loader.dataset.schema.chunk_layout.grid_origin[2]
    if cache is None:
        cache = {}
    if z not in cache:
        z_idx = floor((z - ori_z) / shp_z)
        z_min, z_max = ori_z + z_idx * shp_z, ori_z + (z_idx + 1) * shp_z
        bbox_3d = (bbox_2d[0], bbox_2d[1], z_min, bbox_2d[2], bbox_2d[3], z_max)
        chunk = vol_loader.get_chunk(bbox_3d)
        if chunk is None:
            cache.update({zz: None for zz in range(z_min, z_max)})
        else:
            chunk = chunk.reshape(chunk.shape[:3])
            for k in range(chunk.shape[2]):
                if np.all(chunk[:,:,k]==0, axis=None):
                    cache[z_min+k] = None
                else:
                    cache[z_min+k] = chunk[:,:,k]
    return cache[z], cache


def optimize_main():
    block_size = 48
    buffer_size = 16
    match_list = storage.list_folder_content(storage.join_paths(match_dir, '*.h5'))
    bnames = [os.path.basename(s).replace('.h5', '') for s in match_list]
    z_all = [[int(z) for z in s.split('__to__')] for s in bnames]
    z_all = np.sort(np.unique(z_all))
    ref_files = sorted(storage.list_folder_content(storage.join_paths(tform_dir, '*.h5')))
    z_ref = [int(os.path.basename(s).replace('.h5', '')) for s in ref_files]
    ref_flag = np.isin(z_all, z_ref)
    if np.all(ref_flag):
        print('nothing to optimize')
        return
    storage.makedirs(tform_dir, exist_ok=True)
    if not np.any(ref_flag):
        ref_flag[-1] = 1
        dis = distance_transform_edt(~ref_flag)
        dis[-1] = 1
    while not np.all(ref_flag):
        to_optimize = (dis > 0) & (dis<=(block_size+buffer_size))
        z_list = z_all[to_optimize]
        if np.all(dis<=(block_size + buffer_size)):
            save_list = None
            ref_flag = np.ones_like(ref_flag)
        else:
            to_save = (dis > 0) & (dis<=block_size)
            save_list = z_all[to_save]
            ref_flag = ref_flag | to_save
        optimize_sections(z_list, tform_dir, match_dir, save_list=save_list)
        dis = distance_transform_edt(~ref_flag)
    print('finished')


def optimize_sections(z_list, outdir, matchdir, save_list=None, **kwargs):
    damp = kwargs.get('damp', 1.0)
    smooth_factor = kwargs.get('smooth_factor', 1.0)
    num_iter = kwargs.get('num_iter', 5)
    ref_files = sorted(storage.list_folder_content(storage.join_paths(outdir, '*.h5')))
    z_ref = {int(os.path.basename(s).replace('.h5', '')):s for s in ref_files}
    z_to_opt = [z for z in z_list if z not in z_ref]
    if save_list is None:
        save_list = z_to_opt
    else:
        save_list = [z for z in save_list if z in z_to_opt]
    z_all = tuple(z_ref.keys()) + tuple(z_to_opt)
    if len(z_to_opt) == 0:
        return
    match_files = sorted(storage.list_folder_content(storage.join_paths(matchdir, '*.h5')))
    matches = {}
    for mname in match_files:
        znames = os.path.basename(mname).replace('.h5', '').split('__to__')
        z0, z1 = int(znames[0]), int(znames[1])
        if ((z0 in z_all) and (z1 in z_all)) and ((z0 in z_to_opt) or (z1 in z_to_opt)):
            matches[(z0, z1)] = mname
    if len(matches) == 0:
        return
    exmp_match = next(iter(matches.values()))
    with H5File(exmp_match, 'r') as f:
        wt = f['wt'][()]
        resolution = f['resolution'][()]
        if isinstance(resolution, np.ndarray):
            resolution = resolution.item()
        patch_size = f['patch_size'][()]
        if isinstance(patch_size, np.ndarray):
            patch_size = patch_size.item()
    dof0 = wt.size
    shp0 = wt.shape[:2]
    # get the matrix smoothness term
    indx_0 = np.arange(dof0)
    indx_t = indx_0.reshape(shp0)
    rpts = np.ones(indx_t.shape[1], dtype=np.uint8)
    rpts[1:-1] = 2
    p_1 = np.repeat(indx_t, repeats=rpts, axis=1).reshape(-1,2)
    rpts = np.ones(indx_t.shape[0], dtype=np.uint8)
    rpts[1:-1] = 2
    p_2 = np.repeat(indx_t.T, repeats=rpts, axis=1).reshape(-1,2)
    p_p = np.concatenate((p_1, p_2), axis=0)
    id0 = np.repeat(np.arange(p_p.shape[0]), 2)
    v = np.tile([1.0, -1.0], p_p.shape[0])
    A_sm0 = sparse.csr_matrix((v, (id0, p_p.ravel())), shape=(p_p.shape[0], dof0), dtype=np.float32)
    A_sm = sparse.kron(sparse.eye(len(z_to_opt), dtype=np.float32), A_sm0, format=A_sm0.format)
    id0_list = []
    id1_list = []
    v_list = []
    dd_list = []
    mm_list = []
    wt_list = []
    local_avg = {}
    ref_bc = {}
    z_lut = defaultdict(lambda: -1)
    z_lut.update({z:k for k, z in enumerate(z_to_opt)})
    crnt_pts = 0
    for zs, mtchname in matches.items():
        z0, z1 = zs
        with H5File(mtchname, 'r') as f:
            mm = f['mm'][()]
            dd = f['dd'][()]
            wt = f['wt'][()]
        vlid_idx = wt > 0
        if not np.any(vlid_idx):
            continue
        loc_idx = indx_0[vlid_idx.ravel()]
        id0 = crnt_pts + np.arange(loc_idx.size)
        crnt_pts += loc_idx.size
        mm_dif = mm[1][vlid_idx] - mm[0][vlid_idx]
        dd_dif = np.log(dd[1][vlid_idx].clip(0.1, None)) - np.log(dd[0][vlid_idx].clip(0.1, None))
        wt_list.append(wt[vlid_idx])
        zk0 = z_lut[z0]
        if zk0 >= 0:
            id0_list.append(id0)
            id1_list.append(zk0 * dof0 + loc_idx)
            v_list.append(np.ones(loc_idx.size, dtype=np.float32))
            if z0 not in local_avg:
                local_avg[z0] = (mm[0], wt)
            else:
                mm0, wt0 = local_avg[z0]
                idx_t = wt > wt0
                mm0[idx_t] = mm[0][idx_t]
                wt0[idx_t] = wt[idx_t]
                local_avg[z0] = (mm0, wt0)
        else:
            if z0 in ref_bc:
                log_a0, b0 = ref_bc[z0]
            else:
                with H5File(z_ref[z0], 'r') as f:
                    log_a0 = f['log_a'][()]
                    b0 = f['b'][()]
                ref_bc[z0] = (log_a0, b0)
            mm_dif = mm_dif - b0[vlid_idx]
            dd_dif = dd_dif - log_a0[vlid_idx]
        zk1 = z_lut[z1]
        if zk1 >= 0:
            id0_list.append(id0)
            id1_list.append(zk1 * dof0 + loc_idx)
            v_list.append(-np.ones(loc_idx.size, dtype=np.float32))
            if z1 not in local_avg:
                local_avg[z1] = (mm[1], wt)
            else:
                mm1, wt1 = local_avg[z1]
                idx_t = wt > wt1
                mm1[idx_t] = mm[1][idx_t]
                wt1[idx_t] = wt[idx_t]
                local_avg[z1] = (mm1, wt1)
        else:
            if z1 in ref_bc:
                log_a1, b1 = ref_bc[z1]
            else:
                with H5File(z_ref[z1], 'r') as f:
                    log_a1 = f['log_a'][()]
                    b1 = f['b'][()]
                ref_bc[z1] = (log_a1, b1)
            mm_dif = mm_dif + b1[vlid_idx]
            dd_dif = dd_dif + log_a1[vlid_idx]
        mm_list.append(mm_dif)
        dd_list.append(dd_dif)
    id0_a = np.concatenate(id0_list)
    id1_a = np.concatenate(id1_list)       
    v_a = np.concatenate(v_list)
    dd_a = np.concatenate(dd_list)
    mm_a = np.concatenate(mm_list)
    wt_a = np.concatenate(wt_list)
    A_data = sparse.csr_matrix((v_a, (id0_a, id1_a)), shape=(crnt_pts, dof0*len(z_to_opt)), dtype=np.float32)
    W = sparse.diags(wt_a**0.5)
    sel_M = sparse.kron(sparse.eye(len(z_to_opt), dtype=np.float32), np.ones((dof0, 1), dtype=np.float32), format='csr')
    WA_data = W @ A_data
    A_comp = sparse.vstack((smooth_factor*A_sm, WA_data), format='csr', dtype=np.float32)
    # contrast
    x_z = splinalg.lsqr(WA_data @ sel_M, W.dot(dd_a), damp=damp*dof0)[0]
    cc = sel_M @ x_z
    for _ in range(num_iter):
        cc = splinalg.lsqr(A_comp, np.concatenate((np.zeros(A_sm.shape[0], dtype=np.float32), W.dot(dd_a))), damp=damp, x0=cc)[0]
    # brightness
    x_z = splinalg.lsqr(WA_data @ sel_M, W.dot(mm_a), damp=damp*dof0**0.5)[0]
    bb = sel_M @ x_z
    for _ in range(num_iter):
        bb = splinalg.lsqr(A_comp, np.concatenate((np.zeros(A_sm.shape[0], dtype=np.float32), W.dot(mm_a))), damp=damp, x0=bb)[0]
    for z in save_list:
        zk = z_lut[z]
        if zk < 0:
            continue
        if z not in local_avg:
            continue
        log_a = cc[(zk*dof0):((zk+1)*dof0)].reshape(shp0)
        b = bb[(zk*dof0):((zk+1)*dof0)].reshape(shp0)
        m = local_avg[z][0]
        with H5File(storage.join_paths(outdir, f'{z}.h5'), 'w') as f:
            f.create_dataset('log_a', data=log_a, compression="gzip")
            f.create_dataset('b', data=b, compression="gzip")
            f.create_dataset('m', data=m, compression="gzip")
            f.create_dataset('resolution', data=resolution)
            f.create_dataset('patch_size', data=patch_size)


def render_main(sel_indx=None, use_clahe=False):
    mip = 0
    with storage.File(ts_spec_file, 'r') as f:
        rendered_mips_spec = json.load(f)
        rendered_mips_spec = {int(m): spec for m, spec in rendered_mips_spec.items()}
    t0 = time.time()
    if use_clahe:
        pstfix = '_CLAHE'
    else:
        pstfix = ''
    src_spec = rendered_mips_spec[mip]
    src_loader = dal.TensorStoreLoader.from_json_spec(src_spec)
    src_data = src_loader.dataset
    src_schema = src_data.schema.to_json()
    if src_spec["kvstore"]["driver"] == "file":
        tensorstore_render_dir = src_spec["kvstore"]["path"]
    elif src_spec["kvstore"]["driver"] == "gcs":
        tensorstore_render_dir = "gs://" + src_spec["kvstore"]["bucket"] + "/" + src_spec["kvstore"]["path"]
    if tensorstore_render_dir.endswith('/'):
        tensorstore_render_dir = tensorstore_render_dir[:-1]
    out_ts_dir = storage.join_paths(os.path.dirname(tensorstore_render_dir), 'hist_eq'+pstfix)
    out_spec = {"driver": "neuroglancer_precomputed", "kvstore": out_ts_dir}
    out_spec["schema"] = src_schema
    out_spec.update({"open": True, "create": True, "delete_existing": False})
    out_writer = dal.TensorStoreWriter.from_json_spec(out_spec)
    out_spec = out_writer.spec
    with storage.File(flag_file, 'w') as f:
        json.dump({mip: out_spec}, f)
    X0, Y0, Z0, X1, Y1, Z1 = out_writer.write_grids
    if sel_indx is not None:
        Z0, Z1 = Z0[sel_indx], Z1[sel_indx]
    flag_list = storage.list_folder_content(storage.join_paths(render_flag_dir, '*.flg'))
    to_skip = [int(os.path.basename(s).replace('.flg', '')) for s in flag_list]
    to_render = ~np.isin(Z0, to_skip)
    if not np.any(to_render):
        print('nothing to render.')
        return
    Z0, Z1 = Z0[to_render], Z1[to_render]
    xm0, ym0, zm0 = np.meshgrid(X0, Y0, Z0)
    xm1, ym1, zm1 = np.meshgrid(X1, Y1, Z1)
    zm0_u, zm0_cnt = np.unique(zm0.ravel(), return_counts=True)
    expected_cnt = {zz:cnt for zz, cnt in zip(zm0_u, zm0_cnt)}
    bboxes = np.stack((xm0.ravel(), ym0.ravel(), zm0.ravel(), xm1.ravel(), ym1.ravel(), zm1.ravel()), axis=-1)
    idx_t = np.argsort(zm0.ravel())
    bboxes = bboxes[idx_t]
    num_chunks = bboxes.shape[0]
    chunk_per_job = min(np.max(zm0_cnt), (num_chunks / num_workers)**0.5)
    N_batch = max(1, round(num_chunks / chunk_per_job))
    bindx = np.unique(np.linspace(0, num_chunks, N_batch+1, endpoint=True).astype(np.uint32))
    args_list = []
    for bidx0, bidx1 in zip(bindx[:-1], bindx[1:]):
        args_list.append((bboxes[bidx0:bidx1],))
    tfunc = partial(render_bboxes, src_spec=src_spec, out_spec=out_spec, tform_dir=tform_dir, use_clahe=use_clahe)
    result_cnt = defaultdict(int)
    storage.makedirs(render_flag_dir)
    for res in submit_to_workers(tfunc, args=args_list, num_workers=num_workers):
        for z, cnt in res.items():
            result_cnt[z] += cnt
        for z in expected_cnt:
            if result_cnt[z] >= expected_cnt[z]:
                with storage.File(storage.join_paths(render_flag_dir, f'{z}.flg'), 'w') as f:
                    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(timestamp_str)
                expected_cnt[z] = np.inf
    print(f'time: {round((time.time() - t0)/60, 2)} min')


def render_bboxes(bboxes, src_spec, out_spec, tform_dir, **kwargs):
    use_clahe = kwargs.get('use_clahe', False)
    clahe_clip_limit = kwargs.get('CLAHE_cliplimit', 3.0)
    ds = kwargs.get('ds', 8)
    if use_clahe:
        clahe_pad = kwargs.get('clahe_pad', 128)
        slc = slice(clahe_pad, -clahe_pad)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8,8))
    else:
        clahe_pad = 0
        slc = slice(None, None)
    z_group = defaultdict(list)
    for bbox in bboxes:
        _, _, z_min, _, _, z_max = bbox
        z_group[(z_min, z_max)].append(bbox)
    src_loader = dal.TensorStoreLoader.from_json_spec(src_spec)
    src_schema = src_loader.dataset.schema.to_json()
    src_resolution = src_schema['dimension_units'][0][0]
    out_writer = dal.TensorStoreWriter.from_json_spec(out_spec)
    map_resoluion = None
    render_cnt = defaultdict(int)
    for zz, bboxes_z in z_group.items():
        z_min, z_max = zz
        bc_maps = {}
        for z in range(z_min, z_max):
            map_name = storage.join_paths(tform_dir, f'{z}.h5')
            if storage.file_exists(map_name):
                with H5File(map_name, 'r') as f:
                    log_a = f['log_a'][()]
                    b = f['b'][()]
                    m = f['m'][()]
                    if np.any(m == 0):
                        m_dlt = cv2.dilate(m, np.ones(3))
                        m[m==0] = m_dlt[m==0]
                    gain = np.exp(log_a)
                    bias = (1 - gain) * m + b
                    bc_maps[z] = (gain, bias)
                    if map_resoluion is None:
                        map_resoluion = f['resolution'][()]
                        if isinstance(map_resoluion, np.ndarray):
                            map_resoluion = map_resoluion.item()
                        map_patch_size = f['patch_size'][()]
                        if isinstance(map_patch_size, np.ndarray):
                            map_patch_size = map_patch_size.item()
        for bbox in bboxes_z:
            try:
                xmin, ymin, zmin, xmax, ymax, zmax = bbox
                xmin_p, ymin_p = xmin - clahe_pad, ymin - clahe_pad
                xmax_p, ymax_p = xmax + clahe_pad, ymax + clahe_pad
                block = src_loader.get_chunk((xmin_p, ymin_p, zmin, xmax_p, ymax_p, zmax))
                if (block is None) or np.all(block[slc, slc, ...]==0):
                    render_cnt[zmin] += 1
                    continue
                if map_resoluion is not None:
                    xx, yy = np.meshgrid(np.arange(xmin_p, xmax_p, ds), np.arange(ymin_p, ymax_p, ds))
                    map_coord = np.array([xx, yy])
                    map_coord = scale_coordinates(map_coord, src_resolution/map_resoluion)/map_patch_size - 0.5
                    for k in range(block.shape[2]):
                        z_k = zmin + k
                        if z_k not in bc_maps:
                            continue
                        gain, bias = bc_maps[z_k]
                        gg = map_coordinates(gain, map_coord, order=1, mode='nearest', prefilter=False).T
                        bb = map_coordinates(bias, map_coord, order=1, mode='nearest', prefilter=False).T
                        img = block[:,:,k].astype(np.float32)
                        shp0 = img.shape
                        if ds != 1:
                            gg = cv2.resize(gg, shp0[:2], interpolation=cv2.INTER_LINEAR)
                            bb = cv2.resize(bb, shp0[:2], interpolation=cv2.INTER_LINEAR)
                        img = img.ravel()
                        msk = (img == 0)
                        img = img.ravel() * gg.ravel() + bb.ravel()
                        img[msk] = 0
                        img = img.clip(0, 255).astype(np.uint8).reshape(shp0)
                        if use_clahe:
                            img = clahe.apply(img)
                        block[:,:,k] = img.reshape(shp0)
                out_writer.write_single_chunk(bbox, block[slc, slc, ...])
                render_cnt[zmin] += 1
            except Exception as err:
                print(f'error: {bbox} | {err}')
    return dict(render_cnt)


if __name__ == '__main__':
    args = parse_args()
    sel_indx = slice(args.start, args.stop, args.step)

    root_dir = config.get_work_dir()
    align_dir = storage.join_paths(root_dir, 'align')
    bc_dir = storage.join_paths(align_dir, 'brightness_contrast'+args.postfix)
    ts_spec_file = storage.join_paths(align_dir, 'ts_spec'+args.postfix+'.json')
    ito_spec_file = storage.join_paths(align_dir, 'ITO_mask'+args.postfix+'.json')
    match_dir = storage.join_paths(bc_dir, 'matches')
    tform_dir = storage.join_paths(bc_dir, 'tforms')
    render_flag_dir = storage.join_paths(bc_dir, 'render_flags')
    if args.mode == 'match':
        match_main(sel_indx=sel_indx)
    elif args.mode == 'opt':
        optimize_main()
    elif args.mode == 'render':
        flag_file = storage.join_paths(align_dir, 'histeq'+args.postfix+'.json')
        render_main(sel_indx, use_clahe=args.clahe)

