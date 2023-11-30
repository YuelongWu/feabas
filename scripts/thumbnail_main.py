import argparse
import glob
from functools import partial
from concurrent.futures.process import ProcessPoolExecutor
import math
from multiprocessing import get_context
import os

from feabas import config, logging


def generate_stitched_mipmaps(img_dir, max_mip, **kwargs):
    min_mip = kwargs.pop('min_mip', 0)
    num_workers = kwargs.pop('num_workers', 1)
    parallel_within_section = kwargs.pop('parallel_within_section', True)
    logger_info = kwargs.get('logger', None)
    logger = logging.get_logger(logger_info)
    meta_list = sorted(glob.glob(os.path.join(img_dir, 'mip'+str(min_mip), '**', 'metadata.txt'), recursive=True))
    meta_list = meta_list[arg_indx]
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


def generate_stitched_mipmaps_tensorstore(meta_dir, tgt_mips, **kwargs):
    num_workers = kwargs.pop('num_workers', 1)
    parallel_within_section = kwargs.pop('parallel_within_section', True)
    logger_info = kwargs.get('logger', None)
    logger = logging.get_logger(logger_info)
    meta_list = sorted(glob.glob(os.path.join(meta_dir,'*.json')))
    meta_list = meta_list[arg_indx]
    if parallel_within_section or num_workers == 1:
        for metafile in meta_list:
            mipmap.generate_tensorstore_scales(metafile, tgt_mips, num_workers=num_workers, **kwargs)
    else:
        target_func = parallel_within_section(mipmap.generate_tensorstore_scales, mips=tgt_mips, **kwargs)
        jobs = []
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=get_context('spawn')) as executor:
            for metafile in meta_list:
                job = executor.submit(target_func, metafile)
                jobs.append(job)
            for job in jobs:
                job.result()
    logger.info('mipmapping generated.')


def generate_thumbnails(src_dir, out_dir, **kwargs):
    num_workers = kwargs.pop('num_workers', 1)
    logger_info = kwargs.pop('logger', None)
    logger = logging.get_logger(logger_info)
    meta_list = sorted(glob.glob(os.path.join(src_dir, '**', 'metadata.txt'), recursive=True))
    meta_list = meta_list[arg_indx]
    secnames = [os.path.basename(os.path.dirname(s)) for s in meta_list]
    target_func = partial(mipmap.create_thumbnail, **kwargs)
    os.makedirs(out_dir, exist_ok=True)
    updated = []
    if num_workers == 1:
        for sname in secnames:
            outname = os.path.join(out_dir, sname + '.png')
            if os.path.isfile(outname):
                continue
            updated.append(sname)
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
                updated.append(sname)
                sdir = os.path.join(src_dir, sname)
                job = executor.submit(target_func, sdir, outname=outname)
                jobs.append(job)
            for job in jobs:
                job.result()
        logger.info('thumbnails generated.')
    return updated


def generate_thumbnails_tensorstore(src_dir, out_dir, **kwargs):
    num_workers = kwargs.pop('num_workers', 1)
    logger_info = kwargs.pop('logger', None)
    logger = logging.get_logger(logger_info)
    meta_list = sorted(glob.glob(os.path.join(src_dir, '*.json')))
    meta_list = meta_list[arg_indx]
    target_func = partial(mipmap.create_thumbnail_tensorstore, **kwargs)
    os.makedirs(out_dir, exist_ok=True)
    updated = []
    if num_workers == 1:
        for meta_name in meta_list:
            sname = os.path.basename(meta_name).replace('.json', '')
            outname = os.path.join(out_dir, sname + '.png')
            if os.path.isfile(outname):
                continue
            updated.append(sname)
            img_out = target_func(meta_name)
            common.imwrite(outname, img_out)
    else:
        jobs = []
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=get_context('spawn')) as executor:
            for meta_name in meta_list:
                sname = os.path.basename(meta_name).replace('.json', '')
                outname = os.path.join(out_dir, sname + '.png')
                if os.path.isfile(outname):
                    continue
                updated.append(sname)
                job = executor.submit(target_func, meta_name, outname=outname)
                jobs.append(job)
            for job in jobs:
                job.result()
        logger.info('thumbnails generated.')
    return updated


def save_mask_for_one_sections(mesh_file, out_name, scale, **kwargs):
    from feabas.stitcher import MontageRenderer
    import numpy as np
    from feabas import common
    img_dir = kwargs.get('img_dir', None)
    fillval = kwargs.get('fillval', 0)
    mask_erode = kwargs.get('mask_erode', 0)
    rndr = MontageRenderer.from_h5(mesh_file)
    img = 255 - rndr.generate_roi_mask(scale, mask_erode=mask_erode)
    common.imwrite(out_name, img)
    if img_dir is not None:
        thumb_name = os.path.join(img_dir, os.path.basename(out_name))
        if os.path.isfile(thumb_name):
            thumb = common.imread(thumb_name)
            if (thumb.shape[0] != img.shape[0]) or (thumb.shape[1] != img.shape[1]):
                thumb_out_shape = (*img.shape, *thumb.shape[2:])
                thumb_out = np.full_like(thumb, fillval, shape=thumb_out_shape)
                mn_shp = np.minimum(thumb_out.shape[:2], thumb.shape[:2])
                thumb_out[:mn_shp[0], :mn_shp[1], ...] = thumb[:mn_shp[0], :mn_shp[1], ...]
                common.imwrite(thumb_name, thumb_out)


def generate_thumbnail_masks(mesh_dir, out_dir, seclist=None, **kwargs):
    num_workers = kwargs.get('num_workers', 1)
    scale = kwargs.get('scale')
    img_dir = kwargs.get('img_dir', None)
    fillval = kwargs.get('fillval', 0)
    mask_erode = kwargs.get('mask_erode', 0)
    logger_info = kwargs.get('logger', None)
    logger= logging.get_logger(logger_info)
    mesh_list = sorted(glob.glob(os.path.join(mesh_dir, '*.h5')))
    mesh_list = mesh_list[arg_indx]
    target_func = partial(save_mask_for_one_sections, scale=scale, img_dir=img_dir,
                          fillval=fillval, mask_erode=mask_erode)
    os.makedirs(out_dir, exist_ok=True)
    if num_workers == 1:
        for mname in mesh_list:
            sname = os.path.basename(mname).replace('.h5', '')
            outname = os.path.join(out_dir, sname + '.png')
            if seclist is None and os.path.isfile(outname):
                continue
            elif seclist is not None and sname not in seclist:
                continue
            target_func(mname, outname)
    else:
        jobs = []
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=get_context('spawn')) as executor:
            for mname in mesh_list:
                sname = os.path.basename(mname).replace('.h5', '')
                outname = os.path.join(out_dir, sname + '.png')
                if seclist is None and os.path.isfile(outname):
                    continue
                elif seclist is not None and sname not in seclist:
                    continue
                job = executor.submit(target_func, mname, out_name=outname)
                jobs.append(job)
            for job in jobs:
                job.result()
        logger.info('thumbnail masks generated.')


def align_thumbnail_pairs(pairnames, image_dir, out_dir, **kwargs):
    import cv2
    import numpy as np
    from feabas import caching, thumbnail, common
    material_mask_dir = kwargs.pop('material_mask_dir', None)
    region_mask_dir = kwargs.pop('region_mask_dir', None)
    region_labels = kwargs.pop('region_labels', [0])
    match_name_delimiter = kwargs.pop('match_name_delimiter', '__to__')
    cache_size = kwargs.pop('cache_size', 3)
    feature_match_settings = kwargs.get('feature_matching', {})
    logger_info = kwargs.get('logger', None)
    logger = logging.get_logger(logger_info)
    prepared_cache = caching.CacheFIFO(maxlen=cache_size)
    for pname in pairnames:
        try:
            sname0_ext, sname1_ext = pname
            sname0 = os.path.splitext(sname0_ext)[0]
            sname1 = os.path.splitext(sname1_ext)[0]
            outname = os.path.join(out_dir, sname0 + match_name_delimiter + sname1 + '.h5')
            if os.path.isfile(outname):
                continue
            if sname0 in prepared_cache:
                minfo0 = prepared_cache[sname0]
            else:
                img0 = common.imread(os.path.join(image_dir, sname0_ext))
                if (region_mask_dir is not None) and os.path.isfile(os.path.join(region_mask_dir, sname0_ext)):
                    mask0 = common.imread(os.path.join(region_mask_dir, sname0_ext))
                elif (material_mask_dir is not None) and os.path.isfile(os.path.join(material_mask_dir, sname0_ext)):
                    mask_t = common.imread(os.path.join(material_mask_dir, sname0_ext))
                    mask_t = np.isin(mask_t, region_labels).astype(np.uint8)
                    _, mask0 = cv2.connectedComponents(mask_t, connectivity=4, ltype=cv2.CV_16U)
                else:
                    mask0 = None
                if hasattr(mask0, 'shape') and ((mask0.shape[0] != img0.shape[0]) or (mask0.shape[1] != img0.shape[1])):
                    ht0 = min(mask0.shape[0], img0.shape[0])
                    wd0 = min(mask0.shape[1], img0.shape[1])
                    mask_t = np.zeros_like(mask0, shape=img0.shape)
                    mask_t[:ht0, :wd0] = mask0[:ht0, :wd0]
                    mask0 = mask_t
                minfo0 = thumbnail.prepare_image(img0, mask=mask0, **feature_match_settings)
                prepared_cache[sname0] = minfo0
            if sname1 in prepared_cache:
                minfo1 = prepared_cache[sname1]
            else:
                img1 = common.imread(os.path.join(image_dir, sname1_ext))
                if (region_mask_dir is not None) and os.path.isfile(os.path.join(region_mask_dir, sname1_ext)):
                    mask1 = common.imread(os.path.join(region_mask_dir, sname1_ext))
                elif (material_mask_dir is not None) and os.path.isfile(os.path.join(material_mask_dir, sname1_ext)):
                    mask_t = common.imread(os.path.join(material_mask_dir, sname1_ext))
                    mask_t = np.isin(mask_t, region_labels).astype(np.uint8)
                    _, mask1 = cv2.connectedComponents(mask_t, connectivity=4, ltype=cv2.CV_16U)
                else:
                    mask1 = None
                if hasattr(mask1, 'shape') and ((mask1.shape[0] != img1.shape[0]) or (mask1.shape[1] != img1.shape[1])):
                    ht1 = min(mask1.shape[0], img1.shape[0])
                    wd1 = min(mask1.shape[1], img1.shape[1])
                    mask_t = np.zeros_like(mask1, shape=img1.shape)
                    mask_t[:ht1, :wd1] = mask1[:ht1, :wd1]
                    mask1 = mask_t
                minfo1 = thumbnail.prepare_image(img1, mask=mask1, **feature_match_settings)
                prepared_cache[sname1] = minfo1
            thumbnail.align_two_thumbnails(minfo0, minfo1, outname, **kwargs)
        except Exception as err:
            logger.error(f'{pname}: error {err}')



def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Align thumbnails")
    parser.add_argument("--mode", metavar="mode", type=str, default='match')
    parser.add_argument("--start", metavar="start", type=int, default=0)
    parser.add_argument("--step", metavar="step", type=int, default=1)
    parser.add_argument("--stop", metavar="stop", type=int, default=0)
    parser.add_argument("--reverse",  action='store_true')
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
    elif args.mode.lower().startswith('a') or args.mode.lower().startswith('m'):
        thumbnail_configs = thumbnail_configs['alignment']
        mode = 'alignment'
    else:
        raise ValueError

    num_workers = thumbnail_configs.get('num_workers', 1)
    if num_workers > num_cpus:
        num_workers = num_cpus
        thumbnail_configs['num_workers'] = num_workers
    nthreads = max(1, math.floor(num_cpus / num_workers))
    config.limit_numpy_thread(nthreads)

    from feabas import mipmap, common, material
    from feabas.mipmap import mip_map_one_section
    import numpy as np

    stt_idx, stp_idx, step = args.start, args.stop, args.step
    if stp_idx == 0:
        stp_idx = None
    if args.reverse:
        if stt_idx == 0:
            stt_idx = None
        arg_indx = slice(stp_idx, stt_idx, -step)
    else:
        arg_indx = slice(stt_idx, stp_idx, step)

    thumbnail_dir = os.path.join(root_dir, 'thumbnail_align')
    stitch_tform_dir = os.path.join(root_dir, 'stitch', 'tform')
    img_dir = os.path.join(thumbnail_dir, 'thumbnails')
    mat_mask_dir = os.path.join(thumbnail_dir, 'material_masks')
    reg_mask_dir = os.path.join(thumbnail_dir, 'region_masks')
    manual_dir = os.path.join(thumbnail_dir, 'manual_matches')
    match_dir = os.path.join(thumbnail_dir, 'matches')
    feature_match_dir = os.path.join(thumbnail_dir, 'feature_matches')
    if mode == 'downsample':
        logger_info = logging.initialize_main_logger(logger_name='stitch_mipmap', mp=num_workers>1)
        thumbnail_configs['logger'] = logger_info[0]
        logger= logging.get_logger(logger_info[0])
        align_mip = config.align_configs()['matching']['working_mip_level']
        stitch_conf = config.stitch_configs()['rendering']
        driver = stitch_conf.get('driver', 'image')
        if driver == 'image':
            max_mip = thumbnail_configs.pop('max_mip', max(0, thumbnail_mip_lvl-1))
            max_mip = max(align_mip, max_mip)
            src_dir0 = config.stitch_render_dir()
            pattern = stitch_conf['filename_settings']['pattern']
            one_based = stitch_conf['filename_settings']['one_based']
            fillval = stitch_conf['loader_settings'].get('fillval', 0)
            thumbnail_configs.setdefault('pattern', pattern)
            thumbnail_configs.setdefault('one_based', one_based)
            thumbnail_configs.setdefault('fillval', fillval)
            generate_stitched_mipmaps(src_dir0, max_mip, **thumbnail_configs)
            if thumbnail_configs.get('thumbnail_highpass', True):
                src_mip = max(0, thumbnail_mip_lvl-2)
                highpass_inter_mip_lvl = thumbnail_configs.get('highpass_inter_mip_lvl', src_mip)
                assert highpass_inter_mip_lvl < thumbnail_mip_lvl
                src_dir = os.path.join(src_dir0, 'mip'+str(highpass_inter_mip_lvl))
                downsample = 2 ** (thumbnail_mip_lvl - highpass_inter_mip_lvl)
                if downsample >= 4:
                    highpass = True
                else:
                    highpass = False
            else:
                src_mip = max(0, thumbnail_mip_lvl-1)
                src_dir = os.path.join(src_dir0, 'mip'+str(src_mip))
                downsample = 2 ** (thumbnail_mip_lvl - src_mip)
                highpass = False
            thumbnail_configs.setdefault('downsample', downsample)
            thumbnail_configs.setdefault('highpass', highpass)
            slist = generate_thumbnails(src_dir, img_dir, **thumbnail_configs)
        else:
            stitch_dir = os.path.join(root_dir, 'stitch')
            src_dir = os.path.join(stitch_dir, 'ts_specs')
            tgt_mips = [align_mip]
            if thumbnail_configs.get('thumbnail_highpass', True):
                highpass_inter_mip_lvl = thumbnail_configs.pop('highpass_inter_mip_lvl', max(0, thumbnail_mip_lvl-2))
                assert highpass_inter_mip_lvl < thumbnail_mip_lvl
                downsample = 2 ** (thumbnail_mip_lvl - highpass_inter_mip_lvl)
                if downsample >= 4:
                    highpass = True
                    thumbnail_configs.setdefault('highpass_inter_mip_lvl', highpass_inter_mip_lvl)
                    tgt_mips.append(highpass_inter_mip_lvl)
                else:
                    highpass = False
                    tgt_mips.append(thumbnail_mip_lvl)
                    downsample = 1
            else:
                highpass = False
                tgt_mips.append(thumbnail_mip_lvl)
                downsample = 1
            generate_stitched_mipmaps_tensorstore(src_dir, tgt_mips, **thumbnail_configs)
            thumbnail_configs.setdefault('highpass', highpass)
            thumbnail_configs.setdefault('mip', thumbnail_mip_lvl)
            slist = generate_thumbnails_tensorstore(src_dir, img_dir, **thumbnail_configs)
        mask_scale = 1 / (2 ** thumbnail_mip_lvl)
        generate_thumbnail_masks(stitch_tform_dir, mat_mask_dir, seclist=slist, scale=mask_scale,
                                 img_dir=img_dir, **thumbnail_configs)
        generate_thumbnail_masks(stitch_tform_dir, mat_mask_dir, seclist=None, scale=mask_scale,
                                 img_dir=img_dir, **thumbnail_configs)
        logger.info('finished.')
        logging.terminate_logger(*logger_info)
    elif mode == 'alignment':
        os.makedirs(match_dir, exist_ok=True)
        os.makedirs(manual_dir, exist_ok=True)
        compare_distance = thumbnail_configs.pop('compare_distance', 1)
        logger_info = logging.initialize_main_logger(logger_name='thumbnail_align', mp=num_workers>1)
        thumbnail_configs['logger'] = logger_info[0]
        logger= logging.get_logger(logger_info[0])
        resolution = config.DEFAULT_RESOLUTION * (2 ** thumbnail_mip_lvl)
        thumbnail_configs.setdefault('resolution', resolution)
        thumbnail_configs.setdefault('feature_match_dir', feature_match_dir)
        imglist = sorted(glob.glob(os.path.join(img_dir, '*.png')))
        section_order_file = os.path.join(root_dir, 'section_order.txt')
        imglist = common.rearrange_section_order(imglist, section_order_file)[0]
        bname_list = [os.path.basename(s) for s in imglist]
        region_labels = []
        material_table_file = config.material_table_file()
        material_table = material.MaterialTable.from_json(material_table_file, stream=False)
        for _, mat in material_table:
            if mat.enable_mesh and (mat._stiffness_multiplier > 0.1) and (mat.mask_label is not None):
                region_labels.append(mat.mask_label)
        thumbnail_configs.setdefault('region_labels', region_labels)
        pairnames = []
        match_name_delimiter = thumbnail_configs.get('match_name_delimiter', '__to__')
        for stp in range(1, compare_distance+1):
            for k in range(len(bname_list)-stp):
                sname0_ext = bname_list[k]
                sname1_ext = bname_list[k+stp]
                sname0 = os.path.splitext(sname0_ext)[0]
                sname1 = os.path.splitext(sname1_ext)[0]
                outname = os.path.join(match_dir, sname0 + match_name_delimiter + sname1 + '.h5')
                if os.path.isfile(outname):
                    continue
                pairnames.append((sname0_ext, sname1_ext))
        pairnames.sort()
        pairnames = pairnames[arg_indx]
        target_func = partial(align_thumbnail_pairs, image_dir=img_dir, out_dir=match_dir,
                              material_mask_dir=mat_mask_dir, region_mask_dir=reg_mask_dir,
                              **thumbnail_configs)
        if (num_workers == 1) or (len(pairnames) <= 1):
            target_func(pairnames)
        else:
            num_workers = min(num_workers, len(pairnames))
            match_per_job = thumbnail_configs.pop('match_per_job', 15)
            Njobs = max(num_workers, len(pairnames) // match_per_job)
            indx_j = np.linspace(0, len(pairnames), num=Njobs+1, endpoint=True)
            indx_j = np.unique(np.round(indx_j).astype(np.int32))
            jobs = []
            with ProcessPoolExecutor(max_workers=num_workers, mp_context=get_context('spawn')) as executor:
                for idx0, idx1 in zip(indx_j[:-1], indx_j[1:]):
                    prnm = pairnames[idx0:idx1]
                    job = executor.submit(target_func, pairnames=prnm)
                    jobs.append(job)
                for job in jobs:
                    job.result()
        logger.info('finished.')
        logging.terminate_logger(*logger_info)
