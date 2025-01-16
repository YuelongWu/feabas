import argparse
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures import as_completed
from multiprocessing import get_context
from functools import partial
import numpy as np
import shapely

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
    
    canvas_name = join_paths(out_dir, 'tensorstore_canvas.txt')
    offset_name = join_paths(out_dir, 'offset.txt')

    tlist = sorted(list_folder_content(join_paths(tform_dir, '*.h5')))
    if len(tlist) == 0:
        print(f'No meshes found in {tform_dir}')
        exit()
    makedirs(out_dir, exist_ok=True)
    resolution0 = config.montage_resolution()

    print('finding transformations')
    rfunc = partial(get_convex_hull, resolution=resolution0)
    regions = None
    if args.worker > 1:
        jobs = []
        with ProcessPoolExecutor(max_workers=args.worker, mp_context=get_context('spawn')) as executor:
            for tname in tlist:
                jobs.append(executor.submit(rfunc, tname, wkb=True))
            for job in as_completed(jobs):
                wkb = job.result()
                R = shapely.from_wkb(wkb)
                if regions is None:
                    regions = R
                else:
                    regions = regions.union(R)
    else:
        for tname in tlist:
            R = rfunc(tname, wkb=False)
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
    with File(canvas_name, 'w') as f:
        f.write('\t'.join([str(s) for s in bbox]))
    with File(offset_name, 'w') as f:
        f.write('\t'.join([str(s) for s in -bbox[:2]]))
    tfunc = partial(apply_transform_normalization, out_dir=out_dir, R=R, txy=txy,resolution=resolution0)
    print('applying transforms.')
    if args.worker > 1:
        jobs = []
        with ProcessPoolExecutor(max_workers=args.worker, mp_context=get_context('spawn')) as executor:
            for tname in tlist:
                jobs.append(executor.submit(tfunc, tname))
            for job in as_completed(jobs):
                job.result()
    else:
        for tname in tlist:
            tfunc(tname)
    print('finished')