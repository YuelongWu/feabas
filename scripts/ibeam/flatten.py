import argparse
import copy
import json
import os
from collections import defaultdict

from feabas import config, storage
from feabas.concurrent import submit_to_workers
from functools import partial
num_workers = 30

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from feabas import dal
from datetime import datetime
import time


def flatten_single_chunks(bboxes_in, zs_out, src_loader, ito_loader, out_writer, **kwargs):
    ito_scale = kwargs.get('ito_scale', None)
    write_batch = kwargs.get('write_batch', 1)
    bite_size = 100000
    smooth_sigma = 2
    if isinstance(src_loader, dict):
        src_loader = dal.TensorStoreLoader.from_json_spec(src_loader)
    if isinstance(ito_loader, dict):
        ito_loader = dal.TensorStoreLoader.from_json_spec(ito_loader, dtype=np.uint8)
    if isinstance(out_writer, dict):
        out_writer = dal.TensorStoreWriter.from_json_spec(out_writer)
    if ito_scale is None:
        src_res = src_loader.dataset.schema.to_json()['dimension_units'][0][0]
        ito_res = ito_loader.dataset.schema.to_json()['dimension_units'][0][0]
        ito_scale = src_res / ito_res
    cnt = defaultdict(int)
    bboxes_out = []
    chunks_out = []
    for bbox_in, z_out in zip(bboxes_in, zs_out):
        xmin, ymin, zmin, xmax, ymax, zmax = bbox_in
        zo_min, zo_max = z_out
        bbox_out = (xmin, ymin, zo_min, xmax, ymax, zo_max)
        chunk_src = src_loader.get_chunk(bbox_in)
        if chunk_src is None:
            cnt[(xmin, ymin)] += 1
            continue
        chunk_src = chunk_src.reshape(chunk_src.shape[:3])
        ito_bbox = (int(xmin*ito_scale), int(ymin*ito_scale), zmin, int(xmax*ito_scale), int(ymax*ito_scale), zmax)
        chunk_ito = ito_loader.get_chunk(ito_bbox)
        if chunk_ito is None:
            cnt[(xmin, ymin)] += 1
            continue
        chunk_ito = (chunk_ito == 1).reshape(chunk_ito.shape[:3]).astype(np.uint8)
        chunk_ito_proj = np.sum(chunk_ito, axis=-1).astype(np.float32)
        chunk_ito_proj = np.minimum(chunk_ito_proj, gaussian_filter(chunk_ito_proj, smooth_sigma, mode='nearest'))
        if ito_scale != 1:
            chunk_ito = cv2.resize(chunk_ito, chunk_src.shape[:2], interpolation=cv2.INTER_NEAREST)
            chunk_ito_proj = cv2.resize(chunk_ito_proj, chunk_src.shape[:2], interpolation=cv2.INTER_LINEAR)
        src_2d = chunk_src.reshape(-1, chunk_src.shape[2])
        ito_2d = chunk_ito.reshape(-1, chunk_ito.shape[2]) > 0
        ito_proj_1d = (chunk_ito_proj.reshape(-1,1) - 1).clip(0,None)
        out_2d = np.zeros_like(src_2d, shape=(src_2d.shape[0], int(zo_max-zo_min)))
        for kb in range(0, src_2d.shape[0], bite_size):
            src_2d_b = src_2d[kb:(kb+bite_size)]
            ito_2d_b = ito_2d[kb:(kb+bite_size)]
            out_2d_b = out_2d[kb:(kb+bite_size)]
            ito_proj_1d_b = ito_proj_1d[kb:(kb+bite_size)]
            milling_cycles = np.sum(ito_2d_b, axis=-1)
            sel_idx = milling_cycles > 0
            if not np.any(sel_idx):
                continue
            indx_2d = np.tile(np.linspace(0, 1, int(zo_max-zo_min)), (np.sum(sel_idx), 1))
            mc_t = milling_cycles[sel_idx].reshape(-1,1)
            indx_2d = indx_2d * ito_proj_1d_b[sel_idx] + np.cumsum(np.insert(mc_t[:-1],0,0)).reshape(-1,1)
            indx_1d = indx_2d.ravel()
            src_1d_b = src_2d_b[ito_2d_b]
            out_1d = np.interp(indx_1d, np.arange(src_1d_b.size), src_1d_b)
            out_2d_b[sel_idx] = out_1d.reshape(-1, out_2d_b.shape[-1])
        out_3d = out_2d.reshape(chunk_src.shape[0], chunk_src.shape[1], -1)
        bboxes_out.append(bbox_out)
        chunks_out.append(out_3d)
        if len(bboxes_out) >= write_batch:
            out_writer.write_chunks_w_transaction(bboxes_out, chunks_out)
            for bb in bboxes_out:
                xm, ym = bb[:2]
                cnt[(xm, ym)] += 1
            bboxes_out = []
            chunks_out = []
    if len(bboxes_out) > 0:
        out_writer.write_chunks_w_transaction(bboxes_out, chunks_out)
        for bb in bboxes_out:
            xm, ym = bb[:2]
            cnt[(xm, ym)] += 1
    return cnt

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="flatten aligned volume")
    parser.add_argument("--start", metavar="start", type=int, default=0)
    parser.add_argument("--step", metavar="step", type=int, default=1)
    parser.add_argument("--stop", metavar="stop", type=int)
    parser.add_argument("--zmax", metavar="zmax", type=int)
    parser.add_argument("--write_batch", metavar="write_batch", type=int, default=1)
    parser.add_argument("--postfix", metavar="postfix", type=str, default='')
    return parser.parse_args(args)


if __name__ == '__main__':
    t0 = time.time()
    args = parse_args()

    src_mip = 0
    ito_mip = 3
    out_z_resolution = 22

    sel_indx = slice(args.start, args.stop, args.step)
    root_dir = config.get_work_dir()
    align_dir = storage.join_paths(root_dir, 'align')
    src_spec_file = storage.join_paths(align_dir, 'histeq'+args.postfix+'.json')
    ito_spec_file = storage.join_paths(align_dir, 'ITO_mask'+args.postfix+'.json')
    z_info_file = storage.join_paths(align_dir, 'flatten_map.txt')
    flag_dir = storage.join_paths(align_dir, 'flattern_flags'+args.postfix)

    with storage.File(src_spec_file, 'r') as f:
        src_mips_spec = json.load(f)
    src_mips_spec = {int(mip): spec for mip, spec in src_mips_spec.items()}
    src_spec = src_mips_spec[src_mip]
    with storage.File(ito_spec_file, 'r') as f:
        ito_mips_spec = json.load(f)
    ito_mips_spec = {int(mip): spec for mip, spec in ito_mips_spec.items()}
    ito_spec = ito_mips_spec[ito_mip]
    ito_scale = 2 ** (src_mip - ito_mip)
    with storage.File(z_info_file, 'r') as f:
        # each line: section_name* \t zmin_src \t zmax_src \t zmin_target \t zmax_target
        z_info = np.array([[int(val) for val in line.strip().split('\t')[1:]] for line in f if line.strip()])
    if args.zmax is None:
        zmax = np.max(z_info[:,-1])
    else:
        zmax = args.zmax
    n_zchunk = z_info.shape[0]
    src_loader = dal.TensorStoreLoader.from_json_spec(src_spec)
    src_spec = src_loader.spec
    img_schema = src_loader.dataset.schema.to_json()
    if src_spec["kvstore"]["driver"] == "file":
        src_img_dir = src_spec["kvstore"]["path"]
    elif src_spec["kvstore"]["driver"] == "gcs":
        src_img_dir = "gs://" + src_spec["kvstore"]["bucket"] + "/" + src_spec["kvstore"]["path"]
    if src_img_dir.endswith('/'):
        src_img_dir = src_img_dir[:-1]
    out_img_dir = src_img_dir + '_flattened'
    out_spec = {"driver": "neuroglancer_precomputed", "kvstore": out_img_dir}
    img_schema['dimension_units'][2][0] = out_z_resolution
    img_schema['domain']['exclusive_max'][2] = zmax
    out_spec['schema'] = img_schema
    out_spec.update({"open": True, "create": True, "delete_existing": False})
    out_writer = dal.TensorStoreWriter.from_json_spec(out_spec)
    with storage.File(storage.join_paths(align_dir, 'flattened'+args.postfix+'.json'), 'w') as f:
        json.dump({src_mip: out_writer.spec}, f)
    mid_x, mid_y = out_writer.morton_xy_grid()
    bboxes_2d = out_writer.grid_indices_to_bboxes(mid_x, mid_y).reshape(-1, 4)
    BBOXES_IN = []
    ZS_OUT = []
    flag_files = storage.list_folder_content(storage.join_paths(flag_dir, '*.flg'))
    for bbox in bboxes_2d[sel_indx]:
        flg_name = storage.join_paths(flag_dir, f'{bbox[0]}_{bbox[1]}.flg')
        if storage.file_exists(flg_name, use_cache=True):
            continue
        for zz in z_info:
            BBOXES_IN.append((bbox[0], bbox[1], zz[0], bbox[2], bbox[3], zz[1]))
            ZS_OUT.append((zz[2], zz[3]))
    num_chunks = len(BBOXES_IN)
    chunk_per_job = (num_chunks / num_workers)
    if chunk_per_job >= n_zchunk:
        chunk_per_job = round((chunk_per_job / n_zchunk)**0.5) * n_zchunk
    else:
        chunk_per_job = max(1, round(chunk_per_job))
    args_list = []
    for k in range(0, num_chunks, chunk_per_job):
        args_list.append([BBOXES_IN[k:(k+chunk_per_job)], ZS_OUT[k:(k+chunk_per_job)]])
    
    cnt = defaultdict(int)
    tfunc = partial(flatten_single_chunks, src_loader=src_spec, ito_loader=ito_spec, out_writer=out_spec, ito_scale=ito_scale, write_batch=args.write_batch)

    storage.makedirs(flag_dir, exist_ok=True)
    for res in submit_to_workers(tfunc, args=args_list, num_workers=num_workers):
        for k, v in res.items():
            cnt[k] += v
        for k, v in cnt.items():
            if v >= n_zchunk:
                with storage.File(storage.join_paths(flag_dir, f'{int(k[0])}_{int(k[1])}.flg'), 'w') as f:
                    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(timestamp_str)
                cnt[k] = 0
    print(f'finished after {(time.time() - t0)/60} min.')