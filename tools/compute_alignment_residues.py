import glob
import os
import numpy as np
import argparse
from functools import partial

from feabas import config, storage
from feabas.mesh import Mesh
from feabas.aligner import read_matches_from_h5
from feabas.concurrent import submit_to_workers
import feabas.constant as const

def find_residue_for_one_match(matchname, meshname0, meshname1, **kwargs):
    resolution = kwargs.get('resolution', config.montage_resolution())
    p = kwargs.get('p', 4)
    xy0, xy1, wt, _ = read_matches_from_h5(match_name=matchname, target_resolution=resolution)
    M0 = Mesh.from_h5(meshname0)
    M0.change_resolution(resolution)
    M1 = Mesh.from_h5(meshname1)
    M1.change_resolution(resolution)
    tid0, B0 = M0.cart2bary(xy0, const.MESH_GEAR_INITIAL, tid=None, extrapolate=True)
    xy0t = M0.bary2cart(tid0, B0, const.MESH_GEAR_MOVING, offsetting=True)
    tid1, B1 = M1.cart2bary(xy1, const.MESH_GEAR_INITIAL, tid=None, extrapolate=True)
    xy1t = M1.bary2cart(tid1, B1, const.MESH_GEAR_MOVING, offsetting=True)
    dxy = xy0t - xy1t
    dd = np.sum(dxy**2, axis=-1)**0.5
    wt = wt / np.sum(wt)
    res_avg= np.sum(wt * dd)
    res_high = np.sum(wt * (dd**p)) ** (1/p)
    mname = os.path.splitext(os.path.basename(matchname))[0]
    return mname, res_high, res_avg


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Compute residues")
    parser.add_argument("--worker", metavar="worker", type=int, default=10)
    parser.add_argument("--parent_dir", metavar='parent_dir', type=str)
    parser.add_argument("--resolution", metavar='resolution', type=float)
    parser.add_argument("--power", metavar='power', type=float, default=4)
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args()

    thumbnail_configs = config.thumbnail_configs()
    match_name_delimiter = thumbnail_configs.get('alignment', {}).get('match_name_delimiter', '__to__')
    input_kwargs = {'p': args.power}
    if args.resolution is not None:
        input_kwargs['resolution'] = args.resolution
    tfunc = partial(find_residue_for_one_match, **input_kwargs)
    if args.parent_dir is not None:
        parent_dir = args.parent_dir
    else:
        root_dir = config.get_work_dir()
        parent_dir= storage.join_paths(root_dir, 'align')
    match_list = sorted(glob.glob(storage.join_paths(parent_dir, 'matches', '*.h5')))
    args_list = []
    for mtchname in match_list:
        bname = os.path.splitext(os.path.basename(mtchname))[0]
        snames = bname.split(match_name_delimiter)
        mname0 = storage.join_paths(parent_dir, 'tform', snames[0]+'.h5')
        mname1 = storage.join_paths(parent_dir, 'tform', snames[1]+'.h5')
        if storage.file_exists(mname0, use_cache=True) and storage.file_exists(mname1, use_cache=True):
            args_list.append((mtchname, mname0, mname1))
        else:
            if not storage.file_exists(mname0):
                print(f'mesh not found: {mname0}')
            if not storage.file_exists(mname1):
                print(f'mesh not found: {mname1}')
    out = {}
    for res in submit_to_workers(tfunc, args=args_list, num_workers=args.worker):
        out[res[0]] = res[1:]
    with open(storage.join_paths(parent_dir, 'tform', 'residue.csv'), 'w') as f:
        for mname in sorted(list(out.keys())):
             f.write(', '.join([mname] + [str(s) for s in out[mname]]) + '\n')