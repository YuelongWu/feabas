import argparse
from functools import partial
import json
import numpy as np
import shapely

from feabas.concurrent import submit_to_workers
from feabas.spatial import find_rotation_for_minimum_rectangle
from feabas.aligner import get_convex_hull, apply_transform_normalization
from feabas import config
from feabas.storage import File, join_paths, list_folder_content, makedirs

"""
apply rigid transforms to the meshes in {WORK_DIR}/align/tform/ folder so that
the minimum rotated rectangle of the union of the mesh regions rests its long
side along the x axis and one corner sits at (0,0)
"""


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="rigidly adjust the orientation and position of the aligned meshes.")
    parser.add_argument("--src_dir",  metavar="src_dir", type=str)
    parser.add_argument("--tgt_dir",  metavar="tgt_dir", type=str)
    parser.add_argument("--worker",  metavar="worker", type=int, default=1)
    parser.add_argument("--angle", metavar="angle", type=float)
    parser.add_argument("--offset_x", metavar="offset_x", type=float, default=0.0)
    parser.add_argument("--offset_y", metavar="offset_y", type=float, default=0.0)
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args()

    if args.src_dir is None:
        root_dir =  config.get_work_dir()
        tform_dir = join_paths(root_dir, 'align', 'tform')
    else:
        tform_dir = args.src_dir
    if args.tgt_dir is None:
        out_dir = tform_dir
    else:
        out_dir = args.tgt_dir
    
    canvas_name = join_paths(out_dir, 'canvas.json')
    mip0 = 0

    tlist = sorted(list_folder_content(join_paths(tform_dir, '*.h5')))
    if len(tlist) == 0:
        print(f'No meshes found in {tform_dir}')
        exit()
    makedirs(out_dir, exist_ok=True)
    resolution0 = config.montage_resolution() * 2**(mip0)

    print('finding transformations')
    rfunc = partial(get_convex_hull, resolution=resolution0)
    regions = None
    for wkb in submit_to_workers(rfunc, args=[(s,) for s in tlist], kwargs=[{'wkb': True}], num_workers=args.worker):
        R = shapely.from_wkb(wkb)
        if regions is None:
            regions = R
        else:
            regions = regions.union(R)
    if args.angle is None:
        theta = find_rotation_for_minimum_rectangle(regions)
    else:
        theta = args.angle * np.pi / 180
    corner_xy = np.array(regions.boundary.coords)
    Rt = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    R = np.eye(3)
    R[:2,:2] = Rt
    corner_txy = corner_xy @ Rt
    offset = (args.offset_x, args.offset_y)
    txy = -np.min(corner_txy, axis=0) + np.array(offset)
    xy_min = np.array(offset)
    xy_max = np.max(corner_txy, axis=0) + txy
    bbox = np.concatenate((xy_min, xy_max), axis=None)
    bbox[:2] = np.floor(bbox[:2])
    bbox[-2:] = np.ceil(bbox[-2:]) + 1
    print(f'transformed bbox: {bbox}')
    canvas_bbox = {f'mip{mip0}': [int(s) for s in bbox]}
    with File(canvas_name, 'w') as f:
        json.dump(canvas_bbox, f)
    tfunc = partial(apply_transform_normalization, out_dir=out_dir, R=R, txy=txy,resolution=resolution0)
    print('applying transforms.')
    for _ in submit_to_workers(tfunc, args=[(s,) for s in tlist], num_workers=args.worker):
        pass
    print('finished')