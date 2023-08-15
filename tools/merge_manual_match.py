import h5py
import numpy as np
import glob
import os

import shapely
import shapely.geometry as shpgeo
from feabas import config
from feabas.spatial import scale_coordinates

def _merge_matches(fname0, fname1, outname, clearance=0, weight=1):
    with h5py.File(fname0, 'r') as f:
        xy0 = f['xy0'][()]
        xy1 = f['xy1'][()]
        resolution = f['resolution'][()]
        weight0 = f['weight'][()]
    with h5py.File(fname1, 'r') as f:
        xy0_a = f['xy0'][()]
        xy1_a = f['xy1'][()]
        resolution_a = f['resolution'][()]
        weight_a = f['weight'][()] * weight
    if resolution_a != resolution:
        scale = resolution_a / resolution
        xy0_a = scale_coordinates(xy0_a, scale)
        xy1_a = scale_coordinates(xy1_a, scale)
    if clearance > 0:
        cover_0 = shpgeo.MultiPoint(xy0_a).buffer(clearance)
        cover_1 = shpgeo.MultiPoint(xy1_a).buffer(clearance)
        idx0 = ~shapely.contains_xy(cover_0, xy0)
        idx1 = ~shapely.contains_xy(cover_1, xy1)
        idx = idx0 & idx1
        xy0 = xy0[idx]
        xy1 = xy1[idx]
        weight0 = weight0[idx]
    xy0 = np.concatenate((xy0, xy0_a), axis=0)
    xy1 = np.concatenate((xy1, xy1_a), axis=0)
    weight0 = np.concatenate((weight0, weight_a), axis=0)
    with h5py.File(outname, 'w') as f:
        f.create_dataset('xy0', data=xy0, compression="gzip")
        f.create_dataset('xy1', data=xy1, compression="gzip")
        f.create_dataset('weight', data=weight0, compression="gzip")
        f.create_dataset('resolution', data=resolution)


if __name__ == '__main__':
    root_dir = config.get_work_dir()
    match_dir = os.path.join(root_dir, 'align', 'matches')
    merge_dir = os.path.join(match_dir, 'merge')
    mlist = glob.glob(os.path.join(merge_dir, '*.h5'))
    for mname in mlist:
        fname0 = os.path.join(match_dir, os.path.basename(mname))
        _merge_matches(fname0, mname, mname, clearance=400, weight=5)