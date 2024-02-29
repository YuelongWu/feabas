import cv2
from collections import defaultdict
import h5py
import numpy as np
import glob
import os
from functools import lru_cache

from feabas.common import numpy_to_str_ascii, imread
from feabas.spatial import scale_coordinates
from feabas import config

@lru_cache(maxsize=10)
def parse_h5_match(match_name, target_resolution=None, delimiter='__to__'):
    with h5py.File(match_name, 'r') as f:
        xy0 = f['xy0'][()]
        xy1 = f['xy1'][()]
        resolution = f['resolution'][()]
        if ('name0' in f) and ('name1' in f):
            name0 = numpy_to_str_ascii(f['name0'][()])
            name1 = numpy_to_str_ascii(f['name1'][()])
        else:
            fname = os.path.splitext(os.path.basename(match_name))[0]
            secnames = fname.split(delimiter)
            name0 = secnames[0]
            name1 = secnames[1]
        if isinstance(resolution, np.ndarray):
            resolution = resolution.item()
        if target_resolution is not None:
            scale = resolution / target_resolution
            xy0 = scale_coordinates(xy0, scale)
            xy1 = scale_coordinates(xy1, scale)
    return {name0: xy0, name1: xy1}


if __name__ == '__main__':
    root_dir = config.get_work_dir()
    thumb_dir = os.path.join(root_dir, 'thumbnail_align', 'thumbnails')
    match_dir = os.path.join(root_dir, 'align', 'matches')
    out_dir = os.path.join(match_dir, 'match_cover')
    ext = '.png'
    thumbnail_configs = config.thumbnail_configs()
    delimiter = thumbnail_configs.get('alignment', {}).get('match_name_delimiter', '__to__')
    thumb_mip = thumbnail_configs.get('thumbnail_mip_level', 6)
    thumb_res = config.montage_resolution() * (2 ** thumb_mip)
    align_configs = config.align_configs()
    align_mip = align_configs.get('matching', {}).get('working_mip_level', 2)
    blksz = align_configs.get('matching', {}).get('matcher_config',{}).get('spacings', [100])[-1]
    blksz = int(np.ceil(blksz * 2 ** (align_mip - thumb_mip)))
    skel = np.ones((blksz, blksz), np.uint8)
    ds = 2

    os.makedirs(out_dir, exist_ok=True)
    
    tlist = sorted(glob.glob(os.path.join(thumb_dir, '*'+ext)))
    mlist = sorted(glob.glob(os.path.join(match_dir, '*.h5')))
    mlist_pairnames = [os.path.splitext(os.path.basename(m))[0].split(delimiter) for m in mlist]
    prev_lut = defaultdict(list)
    post_lut = defaultdict(list)
    for p, mname in zip(mlist_pairnames, mlist):
        post_lut[p[0]].append(mname)
        prev_lut[p[1]].append(mname)
    for tname in tlist:
        tname_noext = os.path.splitext(os.path.basename(tname))[0]
        outname = os.path.join(out_dir, tname_noext+'.jpg')
        if os.path.isfile(outname):
            continue
        if (len(prev_lut[tname_noext]) == 0) and (len(post_lut[tname_noext]) == 0):
            continue
        img = imread(tname, flag=cv2.IMREAD_GRAYSCALE)
        mask0 = np.zeros_like(img)
        mask1 = np.zeros_like(img)
        imght = img.shape[0]
        imgwd = img.shape[1]
        for mname in prev_lut[tname_noext]:
            m = parse_h5_match(mname, target_resolution=thumb_res, delimiter=delimiter)
            pts = m[tname_noext]
            x = np.round(pts[:,0].clip(0, imgwd-1)).astype(np.int32)
            y = np.round(pts[:,1].clip(0, imght-1)).astype(np.int32)
            mask0[y, x] = 1
        for mname in post_lut[tname_noext]:
            m = parse_h5_match(mname, target_resolution=thumb_res, delimiter=delimiter)
            pts = m[tname_noext]
            x = np.round(pts[:,0].clip(0, imgwd-1)).astype(np.int32)
            y = np.round(pts[:,1].clip(0, imght-1)).astype(np.int32)
            mask1[y, x] = 1
        mask0 = cv2.dilate(mask0, skel)
        mask1 = cv2.dilate(mask1, skel)
        img_out = np.stack((img*mask0, img*0.8, img*mask1), axis=-1)
        cv2.imwrite(outname, img_out[::ds,::ds,:])