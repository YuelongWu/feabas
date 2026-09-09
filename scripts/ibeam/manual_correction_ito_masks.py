import argparse
import copy
import json
import os

from feabas import config, storage
from feabas.concurrent import submit_to_workers
num_workers = 30
num_workers = config.set_numpy_thread_from_num_workers(num_workers)

import cv2
from feabas import dal
from feabas.common import imread
from functools import partial
from math import log10, ceil
import numpy as np
from skimage.morphology import remove_small_objects, reconstruction, binary_dilation
import time

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="generate ITO masks")
    parser.add_argument("--start", metavar="start", type=int, default=0)
    parser.add_argument("--step", metavar="step", type=int, default=1)
    parser.add_argument("--stop", metavar="stop", type=int)
    return parser.parse_args(args)

def correct_ito_mask_for_bboxes(ind_x, ind_y, ind_z, src_spec, correct_dir, thumb_ds):
    rmv_lable = 255
    add_lable = 100
    src_dataset = dal.TensorStoreWriter.from_json_spec(src_spec)
    imglist = sorted(storage.list_folder_content(storage.join_paths(correct_dir, '*.png')))
    correct_stack = {int(os.path.basename(s).replace('.png', '')): s for s in imglist}
    bboxes = src_dataset.grid_indices_to_bboxes(ind_x, ind_y, ind_z)
    cnt_rm, cnt_add = 0, 0
    for bbox in bboxes:
        xmin, ymin, zmin, xmax, ymax, zmax = bbox
        xmin_ds, ymin_ds, xmax_ds, ymax_ds = int(xmin*thumb_ds), int(ymin*thumb_ds), int(xmax*thumb_ds), int(ymax*thumb_ds)
        to_correct = False
        for zz in range(zmin, zmax):
            if zz not in correct_stack:
                continue
            if isinstance(correct_stack[zz], str):
                correct_stack[zz] = np.swapaxes(imread(correct_stack[zz]), 0, 1)
            to_correct = True
        if not to_correct:
            continue
        block = src_dataset.get_chunk(bbox)
        if (block is None) or (block.size == 0):
            continue
        block_wd, block_ht, block_dp = block.shape[:3]
        block = block.reshape(block_wd, block_ht, block_dp)
        modified = False
        for zz in range(zmin, zmax):
            if zz not in correct_stack:
                continue
            blk_img = block[:,:,zz-zmin]
            correct_tmp = correct_stack[zz][xmin_ds:xmax_ds, ymin_ds:ymax_ds]
            correct_img = cv2.resize(correct_tmp, (block_ht, block_wd), interpolation=cv2.INTER_NEAREST)
            rm_msk = (correct_img == rmv_lable) & (blk_img == 1)
            if np.any(rm_msk):
                blk_img[rm_msk] = 2
                modified = True
                cnt_rm += 1
            ad_msk = (correct_img == add_lable) & (blk_img != 1)
            if np.any(ad_msk):
                blk_img[ad_msk] = 1
                modified = True
                cnt_add += 1
        if modified:
            src_dataset.write_single_chunk(bbox, block)
    return cnt_rm, cnt_add



if __name__ == "__main__":
    args = parse_args()
    sel_indx = slice(args.start, args.stop, args.step)
    mip_high = 3
    mip_thumb = 7

    thumb_ds = 2**(mip_high-mip_thumb)

    root_dir = config.get_work_dir()
    align_dir = storage.join_paths(root_dir, 'align')

    correct_dir = storage.join_paths(align_dir, f'ITO_mask_mip{mip_thumb}_correct')
    ts_spec_file = storage.join_paths(align_dir, f'ITO_mask.json')
    
    t0 = time.time()
    
    with storage.File(ts_spec_file, 'r') as f:
        rendered_mips_spec = json.load(f)

    src_spec = rendered_mips_spec[str(mip_high)]

    src_loader = dal.TensorStoreWriter.from_json_spec(src_spec)
    Nx, Ny, Nz = src_loader.grid_shape
    mid_x, mid_y = src_loader.morton_xy_grid()
    zz = np.arange(Nz)[sel_indx]
    id_x0 = np.tile(mid_x, zz.size).ravel()
    id_y0 = np.tile(mid_y, zz.size).ravel()
    id_z0 = np.repeat(zz, mid_x.size).ravel()

    num_chunks = id_x0.size
    chunk_per_job = (num_chunks / num_workers)**0.5
    N_batch = max(1, round(num_chunks / chunk_per_job))
    bindx = np.unique(np.linspace(0, num_chunks, N_batch+1, endpoint=True).astype(np.uint32))
    args_list = []
    
    for bidx0, bidx1 in zip(bindx[:-1], bindx[1:]):
        args_list.append((id_x0[bidx0:bidx1], id_y0[bidx0:bidx1], id_z0[bidx0:bidx1]))


    tfunc = partial(correct_ito_mask_for_bboxes, src_spec=src_spec, correct_dir=correct_dir, thumb_ds=thumb_ds)

    for res in submit_to_workers(tfunc, args=args_list, num_workers=num_workers):
        pass
    print(f'time: {round((time.time() - t0)/60, 2)} min')