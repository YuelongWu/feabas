import cv2
from collections import defaultdict
import h5py
import numpy as np
import glob
import os
from functools import lru_cache

from feabas.common import numpy_to_str_ascii
from feabas.spatial import scale_coordinates

@lru_cache(maxsize=20)
def parse_h5_match(match_name, target_resolution=None):
    with h5py.File(match_name, 'r') as f:
        xy0 = f['xy0'][()]
        xy1 = f['xy1'][()]
        resolution = f['resolution'][()]
        name0 = numpy_to_str_ascii(f['name0'][()])
        name1 = numpy_to_str_ascii(f['name1'][()])
        if isinstance(resolution, np.ndarray):
            resolution = resolution.item()
        if target_resolution is not None:
            scale = resolution / target_resolution
            xy0 = scale_coordinates(xy0, scale)
            xy1 = scale_coordinates(xy1, scale)
    return {name0: xy0, name1: xy1}


if __name__ == '__main__':
    root_dir = '/n/boslfs02/LABS/lichtman_lab/yuelong/dce/data/Fish2/alignment'
    thumb_dir = os.path.join(root_dir, 'stitched_256nm')
    match_dir = os.path.join(root_dir, 'match_h5')
    out_dir = os.path.join(root_dir, 'match_cover')
    delimiter = '__to__'
    ext = '.png'
    thumb_res = 256
    ds = 2
    blksz = int(np.ceil(100 * 16 / thumb_res))
    skel = np.ones((blksz, blksz), np.uint8)

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
        img = cv2.imread(tname, cv2.IMREAD_GRAYSCALE)
        mask0 = np.zeros_like(img)
        mask1 = np.zeros_like(img)
        imght = img.shape[0]
        imgwd = img.shape[1]
        for mname in prev_lut[tname_noext]:
            m = parse_h5_match(mname, target_resolution=thumb_res)
            pts = m[tname_noext]
            x = np.round(pts[:,0].clip(0, imgwd-1)).astype(np.int32)
            y = np.round(pts[:,1].clip(0, imght-1)).astype(np.int32)
            mask0[y, x] = 1
        for mname in post_lut[tname_noext]:
            m = parse_h5_match(mname, target_resolution=thumb_res)
            pts = m[tname_noext]
            x = np.round(pts[:,0].clip(0, imgwd-1)).astype(np.int32)
            y = np.round(pts[:,1].clip(0, imght-1)).astype(np.int32)
            mask1[y, x] = 1
        mask0 = cv2.dilate(mask0, skel)
        mask1 = cv2.dilate(mask1, skel)
        img_out = np.stack((img*mask0, img*0.8, img*mask1), axis=-1)
        cv2.imwrite(outname, img_out[::ds,::ds,:])