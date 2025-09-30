import argparse
from collections import defaultdict
import cv2
from collections import defaultdict
import numpy as np
import os
from functools import lru_cache, partial

from feabas.common import numpy_to_str_ascii, imread
from feabas.spatial import scale_coordinates
from feabas import config, storage
from feabas.concurrent import submit_to_workers

H5File = storage.h5file_class()

@lru_cache(maxsize=10)
def parse_h5_match(match_name, target_resolution=None, delimiter='__to__'):
    with H5File(match_name, 'r') as f:
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


def generate_match_images(sec_matches, out_dir, ext_out, blksz, ds, resolution, delimiter):
    for tname, mtches in sec_matches.items():
        tname_noext = os.path.splitext(os.path.basename(tname))[0]
        outname = storage.join_paths(out_dir, tname_noext+ext_out)
        if storage.file_exists(outname, use_cache=True):
            continue
        img = imread(tname, flag=cv2.IMREAD_GRAYSCALE)
        imght, imgwd = img.shape[:2]
        mask0 = np.zeros_like(img)
        mask1 = np.zeros_like(img)
        for mtch in mtches:
            indx, mname = mtch
            m = parse_h5_match(mname, target_resolution=resolution, delimiter=delimiter)
            pts = m[tname_noext]
            x = np.round(pts[:,0].clip(0, imgwd-1)).astype(np.int32)
            y = np.round(pts[:,1].clip(0, imght-1)).astype(np.int32)
            if indx == 0:
                mask0[y, x] = 1
            else:
                mask1[y, x] = 1
        skel = np.ones((blksz, blksz), np.uint8)
        mask0 = cv2.dilate(mask0, skel)
        mask1 = cv2.dilate(mask1, skel)
        img_out = np.stack((img*0.7, img*0.7+100*mask0, img*0.7 + 100*mask1), axis=-1)
        img_out = img_out.clip(0, 255).astype(np.uint8)
        cv2.imwrite(outname, img_out[::ds,::ds,:])



def parse_args(args=None):
    parser = argparse.ArgumentParser(description="visualize distribution of matches found (for quality control)")
    parser.add_argument("--mode", metavar="mode", type=str, default='fine',
                        help="(fine | thumbnail | absolute directory of the match folder)")
    parser.add_argument("--worker", metavar="worker", type=int, default=1)
    parser.add_argument("--spacing", metavar="spacing", type=float)
    parser.add_argument("--ext_out", metavar="ext_out", type=str, default=".png")
    parser.add_argument("--ds", metavar="ds", type=int, default=1, help="downsample ratio")
    return parser.parse_args(args)



if __name__ == '__main__':
    args = parse_args()

    root_dir = config.get_work_dir()
    thumbnail_dir = storage.join_paths(root_dir, 'thumbnail_align')
    thumb_img_dir = storage.join_paths(thumbnail_dir, 'thumbnails')
    thumb_match_dir = storage.join_paths(thumbnail_dir, 'matches')
    thumbnail_configs = config.thumbnail_configs()
    thumbnail_align_configs = thumbnail_configs.get('alignment', {})
    delimiter = thumbnail_align_configs.get('match_name_delimiter', '__to__')
    thumb_mip = thumbnail_configs.get('thumbnail_mip_level', 6)
    thumb_res = config.montage_resolution() * (2 ** thumb_mip)
    ext = '.png'

    match_list0 = []
    if args.mode.lower() == 'fine':
        align_dir = storage.join_paths(root_dir, 'align')
        match_dir = storage.join_paths(align_dir, 'matches')
        align_configs = config.align_configs()
        align_mip = align_configs.get('matching', {}).get('working_mip_level', 2)
        blksz = align_configs.get('matching', {}).get('matcher_config',{}).get('spacings', [100])[-1]
        blksz = int(np.ceil(blksz * 2 ** (align_mip - thumb_mip)))
        match_filename = storage.join_paths(align_dir, 'match_name.txt')
        if storage.file_exists(match_filename):
            with storage.File(match_filename, 'r') as f:
                match_list0 = f.readlines()
            match_list0 = [s.strip() for s in match_list0]
        else:
            match_list0 = sorted(storage.list_folder_content(storage.join_paths(thumb_match_dir, '*.h5')))
    elif args.mode.lower().startswith('thumb'):
        match_dir = storage.join_paths(root_dir, 'thumbnail_align', 'matches')
        blksz = int(thumbnail_align_configs.get('block_matching', {}).get('spacings', [100])[-1])
        match_filename = storage.join_paths(thumbnail_dir, 'match_name.txt')
        if storage.file_exists(match_filename):
            with storage.File(match_filename, 'r') as f:
                match_list0 = f.readlines()
            match_list0 = [s.strip() for s in match_list0]
    else:
        match_dir = args.mode
        blksz = 11
    if args.spacing is not None:
        blksz = int(args.spacing)

    out_dir = storage.join_paths(match_dir, 'match_cover')

    mlist = sorted(storage.list_folder_content(storage.join_paths(match_dir, '*.h5')))
    expected_matches = defaultdict(set)
    done_matches = defaultdict(set)
    for mname in match_list0:
        bname = os.path.basename(mname).replace('.h5', '')
        secnames = bname.split(delimiter)
        expected_matches[secnames[0]].add(bname)
        expected_matches[secnames[-1]].add(bname)
    for mname in mlist:
        bname = os.path.basename(mname).replace('.h5', '')
        secnames = bname.split(delimiter)
        done_matches[secnames[0]].add((0,mname))
        done_matches[secnames[-1]].add((1,mname))
        expected_matches[secnames[0]].add(bname)
        expected_matches[secnames[-1]].add(bname)
    tlist = sorted(storage.list_folder_content(storage.join_paths(thumb_img_dir, '*'+ext)))
    section_match_lut = {}
    for tname in tlist:
        tbname = os.path.basename(tname).replace(ext, '')
        if (tbname not in done_matches) or (len(done_matches[tbname]) < len(expected_matches[tbname])):
            continue
        if storage.file_exists(storage.join_paths(out_dir, tbname + args.ext_out), use_cache=True):
            continue
        section_match_lut[tname] = done_matches[tbname]

    Nsec = len(section_match_lut)
    
    if Nsec > 0:
        storage.makedirs(out_dir, exist_ok=True)
        tfunc = partial(generate_match_images, out_dir=out_dir, ext_out=args.ext_out, 
                        blksz=blksz, ds=args.ds, resolution=thumb_res, delimiter=delimiter)
        num_workers = args.worker
        jobs_per_worker = int(max(1, round(Nsec/num_workers)))
        lut_keys = sorted(section_match_lut.keys())
        key_list = [lut_keys[s:(s+jobs_per_worker)] for s in range(0, Nsec, jobs_per_worker)]
        arg_list = [({ky: section_match_lut[ky] for ky in kl},) for kl in key_list]
        num_workers = len(arg_list)
        for _ in submit_to_workers(tfunc, args=arg_list, num_workers=num_workers):
            pass
