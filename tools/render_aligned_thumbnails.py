import argparse
import cv2
from functools import partial
import numpy as np
import os
import time

from feabas.concurrent import submit_to_workers
from feabas import common, dal, config, storage
from feabas.mesh import Mesh
import feabas.constant as const
from feabas.renderer import MeshRenderer

default_thumbnail_resolution = config.thumbnail_resolution()

def render_one_thumbnail(thumbnail_path, mesh_path, out_dir, **kwargs):
    thumbnail_resolution = kwargs.get('thumbnail_resolution', default_thumbnail_resolution)
    bbox = kwargs.get('bbox', None)
    out_resolution = kwargs.get('out_resolution', thumbnail_resolution)
    t0 = time.time()
    outname = storage.join_paths(out_dir, os.path.basename(thumbnail_path))
    if storage.file_exists(outname):
        return
    M = Mesh.from_h5(mesh_path)
    M.change_resolution(out_resolution)
    img = common.imread(thumbnail_path)
    if out_resolution > thumbnail_resolution:
        scale = thumbnail_resolution / out_resolution
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    elif out_resolution < thumbnail_resolution:
        raise RuntimeError('not sufficient thumbnail resolution.')
    renderer = MeshRenderer.from_mesh(M)
    renderer.link_image_loader(dal.StreamLoader(img, resolution=out_resolution, fillval=0))
    if bbox is None:
        bbox = M.bbox(gear=const.MESH_GEAR_MOVING, offsetting=True)
        bbox[:2] = 0
    imgt = renderer.crop(bbox, remap_interp=cv2.INTER_LANCZOS4)
    common.imwrite(outname, imgt)
    print(f'{outname}: {time.time()-t0} sec')


def get_bbox_for_one_section(mname, resolution=None):
    M = Mesh.from_h5(mname)
    if resolution is not None:
        M.change_resolution(resolution)
    bbox = M.bbox(gear=const.MESH_GEAR_MOVING, offsetting=True)
    return bbox


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="rendering aligned thumbnails")
    parser.add_argument("--resolution", metavar="resolution", type=float, default=0)
    parser.add_argument("--src_resolution", metavar="src_resolution", type=float, default=0)
    parser.add_argument("--src_dir",  metavar="src_dir", type=str, default='none')
    parser.add_argument("--tgt_dir",  metavar="tgt_dir", type=str, default='none')
    parser.add_argument("--worker",  metavar="worker", type=int, default=1)
    parser.add_argument("--start", metavar="start", type=int, default=0)
    parser.add_argument("--step", metavar="step", type=int, default=1)
    parser.add_argument("--stop", metavar="stop", type=int, default=0)
    parser.add_argument("--xmin", metavar="xmin", type=int)
    parser.add_argument("--ymin", metavar="ymin", type=int)
    parser.add_argument("--xmax", metavar="xmax", type=int)
    parser.add_argument("--ymax", metavar="ymax", type=int)
    parser.add_argument("--ext", metavar="ext", type=str, default='.png')
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args()

    if args.src_resolution <= 0:
        src_resolution = default_thumbnail_resolution
    else:
        src_resolution = args.src_resolution
    if args.resolution <= 0:
        resolution = src_resolution
    else:
        resolution = args.resolution

    root_dir =  config.get_work_dir()

    if args.src_dir.lower() == 'none':
        src_dir = storage.join_paths(root_dir, 'thumbnail_align', 'thumbnails')
    else:
        src_dir = args.src_dir

    if args.tgt_dir.lower() == 'none':
        if resolution % 1 == 0:
            resolution = int(resolution)
        tgt_dir = storage.join_paths(root_dir, 'align', f'aligned_{resolution}nm')
    else:
        tgt_dir = args.tgt_dir
    storage.makedirs(tgt_dir, exist_ok=True)

    tform_dir = storage.join_paths(root_dir, 'align', 'tform')
    tlist = sorted(storage.list_folder_content(storage.join_paths(tform_dir, '*.h5')))

    if len(tlist) == 0:
        print(f'no transformations found in {tform_dir}.')
        exit()

    xmin, ymin, xmax, ymax = args.xmin, args.ymin, args.xmax, args.ymax
    if (xmin is None) or (ymin is None) or (xmax is None) or (ymax is None):
        bbox0 = []
        bfunc = partial(get_bbox_for_one_section, resolution=resolution)
        for bbx in submit_to_workers(bfunc, args=[(s,) for s in tlist], num_workers=args.worker):
            bbox0.append(bbx)
        bbox0 = np.array(bbox0)
        if xmin is None:
            xmin = np.min(bbox0[:,0])
        if ymin is None:
            ymin = np.min(bbox0[:,1])
        if xmax is None:
            xmax = np.max(bbox0[:,2])
        if ymax is None:
            ymax = np.max(bbox0[:,3])
    if (xmin >= xmax) or (ymin >= ymax):
        print(f'invalid bounding box: {(xmin, ymin, xmax, ymax)}')
    bbox = (xmin, ymin, xmax, ymax)
    stt_idx, stp_idx, step = args.start, args.stop, args.step
    if stp_idx == 0:
        stp_idx = None
    indx = slice(stt_idx, stp_idx, step)
    tlist = tlist[indx]

    target_func = partial(render_one_thumbnail, out_dir=tgt_dir,
                          thumbnail_resolution=src_resolution,
                          out_resolution=resolution, bbox=bbox)
    jobs = []
    imglist = [storage.join_paths(src_dir, os.path.basename(s).replace('.h5', args.ext)) for s in tlist]
    args_list = []
    for tname, mname in zip(tlist, imglist):
        if not storage.file_exists(mname):
            continue
        args_list.append((mname, tname))
    for _ in submit_to_workers(target_func, args=args_list, num_workers=args.worker):
        pass
    print('finished.')