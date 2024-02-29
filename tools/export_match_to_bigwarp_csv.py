import argparse
import h5py
import glob
import numpy as np
import os

from feabas import config
from feabas.spatial import scale_coordinates

def _export_match(mname, outname, target_resolution=None):
    with h5py.File(mname, 'r') as f:
        xy0 = f['xy0'][()]
        xy1 = f['xy1'][()]
        resolution = f['resolution'][()]
    if isinstance(resolution, np.ndarray):
        resolution = resolution.item()
    if target_resolution is not None:
        scale = resolution / target_resolution
        xy0 = scale_coordinates(xy0, scale)
        xy1 = scale_coordinates(xy1, scale)
    xys = np.concatenate((xy0, xy1), axis=-1)
    os.makedirs(os.path.dirname(outname), exist_ok=True)
    with open(outname, 'w') as f:
        for k, xy in enumerate(xys):
            fields = [f'"Pt-{k}"', '"true"'] + [f'"{s}"' for s in xy]
            outstr = ','.join(fields) + '\n'
            f.write(outstr)


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="export matches for Bigwarp")
    parser.add_argument("--match_dir", metavar="match_dir", type=str, default='null')
    parser.add_argument("--out_dir", metavar='out_dir', type=str, default='null')
    parser.add_argument("--resolution", metavar='resolution', type=float, default=-1)
    return parser.parse_args(args)

if __name__ == '__main__':
    args = parse_args()

    match_dir = args.match_dir
    out_dir = args.out_dir
    resolution = args.resolution
    if match_dir.lower() == 'null':
        root_dir = config.get_work_dir()
        thumbnail_dir = os.path.join(root_dir, 'thumbnail_align')
        match_dir = os.path.join(thumbnail_dir, 'matches')
    if out_dir.lower() == 'null':
        root_dir = config.get_work_dir()
        thumbnail_dir = os.path.join(root_dir, 'thumbnail_align')
        out_dir = os.path.join(thumbnail_dir, 'manual_matches')
    if resolution <= 0:
        thumbnail_configs = config.thumbnail_configs()
        thumbnail_mip_lvl = thumbnail_configs.get('thumbnail_mip_level', 6)
        resolution = config.montage_resolution() * (2 ** thumbnail_mip_lvl)
    
    mlist = glob.glob(os.path.join(match_dir, '*.h5'))
    for mname in mlist:
        outname = os.path.join(out_dir, os.path.basename(mname).replace('.h5', '.csv'))
        _export_match(mname, outname, target_resolution=resolution)
