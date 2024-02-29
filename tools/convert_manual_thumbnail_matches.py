import h5py
import numpy as np
import glob
import os

from feabas import config

def _parse_bigwarp_csv(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()
    xy0 = []
    xy1 = []
    for line in lines:
        fields = line.split(',')
        fields = [s.strip() for s in fields]
        fields = [s.replace('"', '') for s in fields]
        if fields[1].lower().startswith('t'):
            xy0.append((float(fields[2]), float(fields[3])))
            xy1.append((float(fields[4]), float(fields[5])))
    return np.array(xy0), np.array(xy1)


if __name__ == '__main__':
    root_dir = config.get_work_dir()
    thumbnail_configs = config.thumbnail_configs()
    thumbnail_mip_lvl = thumbnail_configs.get('thumbnail_mip_level', 6)
    resolution = config.montage_resolution() * (2 ** thumbnail_mip_lvl)
    thumbnail_dir = os.path.join(root_dir, 'thumbnail_align')
    manual_dir = os.path.join(thumbnail_dir, 'manual_matches')
    match_dir = os.path.join(thumbnail_dir, 'matches')

    mlist = glob.glob(os.path.join(manual_dir, '*.csv'))
    for mname in mlist:
        outname = os.path.join(match_dir, os.path.basename(mname).replace('.csv', '.h5'))
        xy0, xy1 = _parse_bigwarp_csv(mname)
        weight = np.ones(xy0.shape[0], dtype=np.float32)
        if xy0.size > 0:
           with h5py.File(outname, 'w') as f:
                f.create_dataset('xy0', data=xy0, compression="gzip")
                f.create_dataset('xy1', data=xy1, compression="gzip")
                f.create_dataset('weight', data=weight, compression="gzip")
                f.create_dataset('resolution', data=resolution) 
