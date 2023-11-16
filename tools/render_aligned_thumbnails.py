import argparse
import cv2
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures import as_completed
from multiprocessing import get_context
from functools import partial
import numpy as np
import os
import glob
import time

from feabas import common, dal, config
from feabas.mesh import Mesh
import feabas.constant as const
from feabas.renderer import MeshRenderer

thumbnail_mip_lvl = config.thumbnail_configs().get('thumbnail_mip_level', 6)
default_thumbnail_resolution = config.DEFAULT_RESOLUTION * (2 ** thumbnail_mip_lvl)

def render_one_thumbnail(thumbnail_path, mesh_path, out_dir, **kwargs):
    thumbnail_resolution = kwargs.get('thumbnail_resolution', default_thumbnail_resolution)
    bbox = kwargs.get('bbox', None)
    out_resolution = kwargs.get('out_resolution', thumbnail_resolution)
    t0 = time.time()
    outname = os.path.join(out_dir, os.path.basename(thumbnail_path))
    if os.path.isfile(outname):
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
    parser.add_argument("--xmin", metavar="xmin", type=int, default=0)
    parser.add_argument("--ymin", metavar="ymin", type=int, default=0)
    parser.add_argument("--xmax", metavar="xmax", type=int, default=0)
    parser.add_argument("--ymax", metavar="ymax", type=int, default=0)
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
        src_dir = os.path.join(root_dir, 'thumbnail_align', 'thumbnails')
    else:
        src_dir = args.src_dir

    if args.tgt_dir.lower() == 'none':
        tgt_dir = os.path.join(root_dir, 'align', f'aligned_{resolution}nm')
    else:
        tgt_dir = args.tgt_dir
    os.makedirs(tgt_dir, exist_ok=True)

    tlist = sorted(glob.glob(os.path.join(root_dir, 'align', 'tform', '*.h5')))

    xmin, ymin, xmax, ymax = args.xmin, args.ymin, args.xmax, args.ymax
    if (xmin >= xmax) or (ymin >= ymax):
        initialized = False
        for mname in tlist:
            M = Mesh.from_h5(mname)
            M.change_resolution(resolution)
            bbox0 = M.bbox(gear=const.MESH_GEAR_MOVING, offsetting=True)
            if not initialized:
                initialized = True
                xmin = np.floor(bbox0[0])
                ymin = np.floor(bbox0[1])
                xmax = np.ceil(bbox0[2])
                ymax = np.ceil(bbox0[3])
            else:
                xmin = min(xmin, np.floor(bbox0[0]))
                ymin = min(ymin, np.floor(bbox0[1]))
                xmax = max(xmax, np.ceil(bbox0[2]))
                ymax = max(ymax, np.ceil(bbox0[3]))
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
    imglist = [os.path.join(src_dir, os.path.basename(s).replace('.h5', args.ext)) for s in tlist]
    with ProcessPoolExecutor(max_workers=args.worker, mp_context=get_context('spawn')) as executor:
        for tname, mname in zip(tlist, imglist):
            if not os.path.isfile(mname):
                continue
            jobs.append(executor.submit(target_func, mname, tname))
        for job in as_completed(jobs):
            job.result()
    print('finished.')