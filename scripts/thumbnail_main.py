import argparse
import glob
from functools import partial
from concurrent.futures.process import ProcessPoolExecutor
import math
from multiprocessing import get_context
import os
import time

from feabas import config, logging


def mip_map_one_section(sec_name, img_dir, max_mip, **kwargs):
    ext_out = kwargs.pop('format', 'jpg')
    logger_info = kwargs.get('logger', None)
    logger = logging.get_logger(logger_info)
    t0 = time.time()
    num_tiles = []
    for m in range(max_mip):
        src_dir = os.path.join(img_dir, 'mip'+str(m), sec_name)
        out_dir = os.path.join(img_dir, 'mip'+str(m+1), sec_name)
        n_tile = mipmap.mip_one_level(src_dir, out_dir, output_format=ext_out,
                                      downsample=2, **kwargs)
        num_tiles.append(n_tile)
        if n_tile is None:
            break
    logger.info(f'{sec_name}: number of tiles {num_tiles} | {(time.time()-t0)/60} min')


def generate_stitched_mipmaps(img_dir, max_mip, **kwargs):
    min_mip = kwargs.pop('min_mip', 0)
    num_workers = kwargs.pop('num_workers', 1)
    parallel_within_section = kwargs.pop('parallel_within_section', True)
    logger_info = kwargs.get('logger', None)
    logger= logging.get_logger(logger_info)
    meta_list = sorted(glob.glob(os.path.join(img_dir, 'mip'+str(min_mip), '**', 'metadata.txt'), recursive=True))
    secnames = [os.path.basename(os.path.dirname(s)) for s in meta_list]
    if parallel_within_section or (num_workers == 1):
        for sname in secnames:
            mip_map_one_section(sname, img_dir, max_mip, num_workers=num_workers, **kwargs)
    else:
        target_func = partial(mip_map_one_section, img_dir=img_dir,
                                max_mip=max_mip, num_workers=1, **kwargs)
        jobs = []
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=get_context('spawn')) as executor:
            for sname in secnames:
                job = executor.submit(target_func, sname)
                jobs.append(job)
            for job in jobs:
                job.result()
    logger.info('mipmapping generated.')


def generate_thumbnails(src_dir, out_dir, **kwargs):
    num_workers = kwargs.pop('num_workers', 1)
    logger_info = kwargs.pop('logger', None)
    logger= logging.get_logger(logger_info)
    meta_list = sorted(glob.glob(os.path.join(src_dir, '**', 'metadata.txt'), recursive=True))
    secnames = [os.path.basename(os.path.dirname(s)) for s in meta_list]
    target_func = partial(mipmap.create_thumbnail, **kwargs)
    os.makedirs(out_dir, exist_ok=True)
    if num_workers == 1:
        for sname in secnames:
            outname = os.path.join(out_dir, sname + '.png')
            if os.path.isfile(outname):
                continue
            sdir = os.path.join(src_dir, sname)
            img_out = target_func(sdir)
            common.imwrite(outname, img_out)
    else:
        jobs = []
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=get_context('spawn')) as executor:
            for sname in secnames:
                outname = os.path.join(out_dir, sname + '.png')
                if os.path.isfile(outname):
                    continue
                sdir = os.path.join(src_dir, sname)
                job = executor.submit(target_func, sdir, outname=outname)
                jobs.append(job)
            for job in jobs:
                job.result()
        logger.info('thumbnails generated.')


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Align thumbnails")
    parser.add_argument("--mode", metavar="mode", type=str, default='downsample')
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args()

    root_dir = config.get_work_dir()
    generate_settings = config.general_settings()
    num_cpus = generate_settings['cpu_budget']

    thumbnail_configs = config.thumbnail_configs()
    thumbnail_mip_lvl = thumbnail_configs.get('thumbnail_mip_level', 6)
    if args.mode.lower().startswith('d'):
        thumbnail_configs = thumbnail_configs['downsample']
        mode = 'downsample'

    num_workers = thumbnail_configs.get('num_workers', 1)
    nthreads = max(1, math.floor(num_cpus / num_workers))
    config.limit_numpy_thread(nthreads)

    from feabas import mipmap, common, thumbnail

    thumbnail_dir = os.path.join(root_dir, 'thumbnail_align')
    img_dir = os.path.join(thumbnail_dir, 'thumbnails')
    if mode == 'downsample':
        logger_info = logging.initialize_main_logger(logger_name='stitch_mipmap', mp=num_workers>1)
        thumbnail_configs['logger'] = logger_info[0]
        logger= logging.get_logger(logger_info)
        max_mip = thumbnail_configs.pop('max_mip', thumbnail_mip_lvl-1)
        src_dir0 = config.stitch_render_dir()
        stitch_conf = config.stitch_configs()['rendering']
        pattern = stitch_conf['filename_settings']['pattern']
        one_based = stitch_conf['filename_settings']['one_based']
        fillval = stitch_conf['loader_settings'].get('fillval', 0)
        thumbnail_configs.setdefault('pattern', pattern)
        thumbnail_configs.setdefault('one_based', one_based)
        thumbnail_configs.setdefault('fillval', fillval)
        generate_stitched_mipmaps(src_dir0, max_mip, **thumbnail_configs)
        if thumbnail_configs.get('thumbnail_highpass', True):
            src_dir = os.path.join(src_dir0, 'mip'+str(thumbnail_mip_lvl-2))
            downsample = 4
            highpass = True
        else:
            src_dir = os.path.join(src_dir0, 'mip'+str(thumbnail_mip_lvl-1))
            downsample = 2
            highpass = False
        out_dir = os.path.join()
        thumbnail_configs.setdefault('downsample', downsample)
        thumbnail_configs.setdefault('highpass', highpass)
        generate_thumbnails(src_dir, img_dir, **thumbnail_configs)
        logger.info('finished.')
        logging.terminate_logger(*logger_info)
