import argparse
import glob
from functools import partial
from concurrent.futures.process import ProcessPoolExecutor
from multiprocessing import get_context
import os
import time

from feabas import logging, mipmap, thumbnail, path


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
    logger_info = logging.initialize_main_logger(logger_name='stitch_mipmap', mp=num_workers>1)
    kwargs['logger'] = logger_info[0]
    logger= logging.get_logger(logger_info[0])
    meta_list = glob.glob(os.path.join(img_dir, 'mip'+str(min_mip), '**', 'metadata.txt'), recursive=True)
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
    logger.info('finished.')
    logging.terminate_logger(*logger_info)

