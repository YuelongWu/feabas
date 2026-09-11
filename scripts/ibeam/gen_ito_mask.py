import argparse
import copy
import json
import os

from feabas import config, storage
from feabas.concurrent import submit_to_workers
num_workers = 30

import cv2
from feabas import dal
from functools import partial
from math import floor, ceil
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.morphology import reconstruction, dilation, opening, disk
import time


def get_ito_mask_for_xy_chunk(bbox, z_info, src_spec, out_spec):
    thresholds = (10, 30)
    src_loader = dal.TensorStoreLoader.from_json_spec(src_spec)
    out_writer = dal.TensorStoreWriter.from_json_spec(out_spec)
    resolution0 = src_loader.resolution
    ds = out_writer.resolution / src_loader.resolution
    dimension_cutoff = 1024
    xmin, ymin, xmax, ymax = bbox
    if z_info is None:
        _, _, Z0, _, _, Z1 = out_writer.write_grids
        z_info = [(z0, z1, z1) for z0, z1 in zip(Z0, Z1)]
    for zs in z_info:
        zmin, z_int, zmax = zs
        bbox_3d = (xmin, ymin, zmin, xmax, ymax, zmax)
        bbox_3d_src = (int(xmin*ds), int(ymin*ds), zmin, int(xmax*ds), int(ymax*ds), zmax)
        block = src_loader.get_chunk(bbox_3d_src)
        if (block is None) or (block.size == 0):
            continue
        block = block.reshape(block.shapep[:3])
        out_wd, out_ht, out_dp = xmax - xmin, ymax - ymin, zmax - zmin
        ito_blk = np.zeros_like((out_wd, out_ht, out_dp), dtype=bool)
        previous_mask = np.ones((out_wd, out_ht), dtype=bool)
        z_int = z_int - zmin
        for z in range(out_dp):
            img = block[:,:,z]
            mask_l = img < thresholds[0]
            if not np.any(mask_l):
                ito_blk[:,:,z] = 1
            mask_h = img < thresholds[-1]
            if ds != 1:
                mask_l = cv2.resize(mask_l.astype(np.float32), (out_wd, out_ht), interpolation=cv2.INTER_AREA) > 0
                mask_h = cv2.resize(mask_h.astype(np.float32), (out_wd, out_ht), interpolation=cv2.INTER_AREA) > 0.5
            mask = reconstruction(mask_l & mask_h, mask_h) > 0
            mask = ~mask
            cls_sz = (dimension_cutoff / (resolution0 * ds))
            if cls_sz > 4:
                mask_ds = cv2.resize(mask.astype(np.float32), None, fx=4/cls_sz, fy=4/cls_sz, interpolation=cv2.INTER_AREA) > 0.5
                mask_ds_op = opening(mask_ds, disk(4))
                mask_op = cv2.resize(mask_ds_op.astype(np.float32), mask.shape, interpolation=cv2.INTER_LINEAR) > 0.5
                mask = reconstruction(mask_op & mask, mask) > 0
            elif cls_sz >= 1:
                mask_op = opening(mask, disk(round(cls_sz)))
                mask = reconstruction(mask_op & mask, mask) > 0
            mask = ~dilation(~mask, disk(2))
            if z < z_int:
                ito_blk[:,:,z] = mask
            else:
                ito_blk[:,:,z] = mask & previous_mask
                previous_mask = ito_blk[:,:,z]
        ito_blk[:,:,0] = 0
        out_writer.write_single_chunk(bbox_3d, ito_blk.astype(np.uint8))
    return 0


def threshold_main(sel_indx=None, post_fix=''):
    mip_src = 1
    mip_out = 3
    downsample_factor = 2 ** (mip_out - mip_src)
    root_dir = config.get_work_dir()
    align_dir = storage.join_paths(root_dir, 'align')
    ts_spec_file = storage.join_paths(align_dir, 'ts_spec'+post_fix+'.json')
    with storage.File(ts_spec_file, 'r') as f:
        rendered_mips_spec = json.load(f)
    z_info_file = storage.join_paths(align_dir, 'ito_group.txt')
    with storage.File(z_info_file, 'r') as f:
        z_info = [[int(val) for val in line.strip().split('\t')] for line in f if line.strip()]
    rendered_mips_spec = {int(mip): spec for mip, spec in rendered_mips_spec.items()}
    tmp_spec = next(iter(rendered_mips_spec.values()))
    if tmp_spec["kvstore"]["driver"] == "file":
        tensorstore_render_dir = tmp_spec["kvstore"]["path"]
    elif tmp_spec["kvstore"]["driver"] == "gcs":
        tensorstore_render_dir = "gs://" + tmp_spec["kvstore"]["bucket"] + "/" + tmp_spec["kvstore"]["path"]
    if tensorstore_render_dir.endswith('/'):
        tensorstore_render_dir = tensorstore_render_dir[:-1]
    out_ts_dir = storage.join_paths(os.path.dirname(tensorstore_render_dir), 'ITO_masks')
    flag_dir = storage.join_paths(align_dir, f'ITO_mask{post_fix}.json')

    t0 = time.time()
    src_spec = rendered_mips_spec[mip_src]
    src_loader = dal.TensorStoreLoader.from_json_spec(src_spec)
    src_data = src_loader.dataset
    out_schema = copy.deepcopy(src_data.schema.to_json())
    if downsample_factor != 1:
        dsp_spec = {
            "driver": "downsample",
            "downsample_factors": [downsample_factor, downsample_factor, 1, 1],
            "downsample_method": 'mean',
            "base": src_loader.spec
        }
        dsp_loader = dal.TensorStoreLoader.from_json_spec(dsp_spec)
        dsp_data = dsp_loader.dataset
        dsp_schema = dsp_data.schema.to_json()
        out_schema["dimension_units"] = dsp_schema["dimension_units"]
        out_schema["domain"] = dsp_schema["domain"]
    out_spec = {"driver": "neuroglancer_precomputed", "kvstore": out_ts_dir}
    out_schema["chunk_layout"].update({"codec_chunk": {"shape": [8, 8, 8, 1]}})
    out_schema["codec"] = ({"driver": "neuroglancer_precomputed", "encoding": "compressed_segmentation"})
    out_schema["dtype"] = "uint32"
    out_spec["schema"] = out_schema
    out_spec.update({"open": True, "create": True, "delete_existing": False})

    out_writer = dal.TensorStoreWriter.from_json_spec(out_spec)
    with storage.File(flag_dir, 'w') as f:
        json.dump({mip_out: out_writer.spec},f)

    X0, Y0, _, X1, Y1, _ = out_writer.write_grids
    xm0, ym0 = np.meshgrid(X0, Y0)
    xm1, ym1 = np.meshgrid(X1, Y1)
    xm0, xm1, ym0, ym1 = xm0.ravel(), xm1.ravel(), ym0.ravel(), ym1.ravel()
    if sel_indx is not None:
        xm0, xm1 = xm0[sel_indx], xm1[sel_indx]
        ym0, ym1 = ym0[sel_indx], ym1[sel_indx]
    bboxes = np.stack((xm0, ym0, xm1, ym1), axis=-1)
    Nz = max(1, floor(num_workers / bboxes.shape[0]))
    z_bsz = ceil(len(z_info) / Nz)
    z_info_list = [z_info[k:(k+z_bsz)] for k in range(0, len(z_info), z_bsz)]
    args_list = []
    for zz in z_info_list:
        for bbox in bboxes:
            args_list.append((bbox, zz))
    tfunc = partial(get_ito_mask_for_xy_chunk, src_spec=src_spec, out_spec=out_spec)
    err_cnt = 0
    for res in submit_to_workers(tfunc, args=args_list, num_workers=num_workers):
        err_cnt += res    
    print(f'time: {round((time.time() - t0)/60, 2)} min | {err_cnt} errors')


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="generate ITO masks")
    parser.add_argument("--start", metavar="start", type=int, default=0)
    parser.add_argument("--step", metavar="step", type=int, default=1)
    parser.add_argument("--stop", metavar="stop", type=int)
    parser.add_argument("--postfix", metavar="postfix", type=str, default='')
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args()
    sel_indx = slice(args.start, args.stop, args.step)
    threshold_main(sel_indx=sel_indx, post_fix=args.postfix)