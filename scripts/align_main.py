from collections import defaultdict
import argparse
import glob
from functools import partial
from concurrent.futures.process import ProcessPoolExecutor
from multiprocessing import get_context
import math
import os
import time
import gc

from feabas import config, logging
import feabas.constant as const

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40)) # for large masks in meshing

def generate_mesh_from_mask(mask_names, outname, **kwargs):
    if os.path.isfile(outname):
        return
    from feabas import material, dal, spatial, mesh
    material_table = kwargs.get('material_table', material.MaterialTable())
    target_resolution = kwargs.get('target_resolution', config.montage_resolution())
    mesh_size = kwargs.get('mesh_size', 600)
    simplify_tol = kwargs.get('simplify_tol', 2)
    area_thresh = kwargs.get('area_thresh', 0)
    logger_info = kwargs.pop('logger', None)
    logger = logging.get_logger(logger_info)
    if isinstance(simplify_tol, dict):
        region_tols = defaultdict(lambda: 0.1)
        region_tols.update(simplify_tol)
    else:
        region_tols = defaultdict(lambda: simplify_tol)
    loader = None
    if not isinstance(material_table, material.MaterialTable):
        if isinstance(material_table, dict):
            material_table = material.MaterialTable(table=material_table)
        elif isinstance(material_table, str):
            material_table = material.MaterialTable.from_json(material_table, stream=not material_table.endswith('.json'))
        else:
            raise TypeError
    if 'exclude' in material_table.named_table:
        mat = material_table['exclude']
        fillval = mat.mask_label
    else:
        fillval = 255
    for mask_name, resolution in mask_names:
        if not os.path.isfile(mask_name):
            continue
        src_resolution = resolution
        if mask_name.lower().endswith('.json') or mask_name.lower().endswith('.txt'):
            loader = dal.get_loader_from_json(mask_name, resolution=src_resolution, fillval=fillval)
        else:
            loader = mask_name
        break
    secname = os.path.splitext(os.path.basename(outname))[0]
    if loader is None:
        logger.warning(f'{secname}: mask does not exist.')
        return
    mesh_size = mesh_size * config.montage_resolution() / src_resolution
    G = spatial.Geometry.from_image_mosaic(loader, material_table=material_table, resolution=src_resolution)
    PSLG = G.PSLG(region_tol=region_tols,  roi_tol=0, area_thresh=area_thresh)
    M = mesh.Mesh.from_PSLG(**PSLG, material_table=material_table, mesh_size=mesh_size, min_mesh_angle=20)
    M.change_resolution(target_resolution)
    if ('split' in material_table.named_table):
        mid = material_table.named_table['split'].uid
        m_indx = M.material_ids == mid
        M.incise_region(m_indx)
    mshname = os.path.splitext(os.path.basename(mask_name))[0]
    M.save_to_h5(outname, save_material=True, override_dict={'name': mshname})


def generate_mesh_main():
    logger_info = logging.initialize_main_logger(logger_name='mesh_generation', mp=num_workers>1)
    mesh_config['logger'] = logger_info[0]
    logger = logging.get_logger(logger_info[0])
    thumbnail_mip_lvl = thumbnail_configs.get('thumbnail_mip_level', 6)
    thumbnail_resolution = config.montage_resolution() * (2 ** thumbnail_mip_lvl)
    thumbnail_mask_dir = os.path.join(thumbnail_dir, 'material_masks')
    match_list = glob.glob(os.path.join(thumb_match_dir, '*.h5'))
    match_names = [os.path.basename(s).replace('.h5', '').split(match_name_delimiter) for s in match_list]
    secnames = set([s for pp in match_names for s in pp])
    alt_mask_dir = mesh_config.get('mask_dir', None)
    alt_mask_mip_level = mesh_config.get('mask_mip_level', 4)
    alt_mask_resolution = config.montage_resolution() * (2 ** alt_mask_mip_level)
    if alt_mask_dir is None:
        alt_mask_dir = os.path.join(align_dir, 'material_masks')
    material_table_file = config.material_table_file()
    material_table = material.MaterialTable.from_json(material_table_file, stream=False)
    if num_workers == 1:
        for sname in secnames:
            mask_names = [(os.path.join(alt_mask_dir, sname + '.json'), alt_mask_resolution),
                        (os.path.join(alt_mask_dir, sname + '.txt'), alt_mask_resolution),
                        (os.path.join(alt_mask_dir, sname + '.png'), alt_mask_resolution),
                        (os.path.join(thumbnail_mask_dir, sname + '.png'), thumbnail_resolution)]
            outname = os.path.join(mesh_dir, sname + '.h5')
            generate_mesh_from_mask(mask_names, outname, material_table=material_table, **mesh_config)
    else:
        material_table = material_table.save_to_json(jsonname=None)
        target_func = partial(generate_mesh_from_mask, material_table=material_table, **mesh_config)
        jobs = []
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=get_context('spawn')) as executor:
            for sname in secnames:
                mask_names = [(os.path.join(alt_mask_dir, sname + '.json'), alt_mask_resolution),
                              (os.path.join(alt_mask_dir, sname + '.txt'), alt_mask_resolution),
                              (os.path.join(alt_mask_dir, sname + '.png'), alt_mask_resolution),
                              (os.path.join(thumbnail_mask_dir, sname + '.png'), thumbnail_resolution)]
                outname = os.path.join(mesh_dir, sname + '.h5')
                if not os.path.isfile(outname):
                    job = executor.submit(target_func, mask_names=mask_names, outname=outname)
                    jobs.append(job)
            for job in jobs:
                job.result()
    logger.info('meshes generated.')
    logging.terminate_logger(*logger_info)


def match_main(match_list):
    stitch_config = config.stitch_configs().get('rendering', {})
    loader_config = {key: val for key, val in stitch_config.items() if key in ('pattern', 'one_based', 'fillval')}
    working_mip_level = align_config.get('working_mip_level', 2)
    stitch_render_driver = config.stitch_configs().get('rendering', {}).get('driver', 'image')
    if stitch_render_driver == 'image':
        stitch_render_dir = config.stitch_render_dir()
        stitched_image_dir = os.path.join(stitch_render_dir, 'mip'+str(working_mip_level))
    else:
        stitch_dir = os.path.join(root_dir, 'stitch')
        spec_dir = os.path.join(stitch_dir, 'ts_specs')
    logger_info = logging.initialize_main_logger(logger_name='align_matching', mp=False)
    logger = logging.get_logger(logger_info[0])
    if len(match_list) == 0:
        return
    for mname in match_list:
        outname = os.path.join(match_dir, os.path.basename(mname))
        if os.path.isfile(outname):
            continue
        t0 = time.time()
        tname = os.path.basename(mname).replace('.h5', '')
        logger.info(f'start {tname}')
        secnames = os.path.splitext(os.path.basename(mname))[0].split(match_name_delimiter)
        if stitch_render_driver == 'image':
            loaders = [get_image_loader(os.path.join(stitched_image_dir, s), **loader_config) for s in secnames]
        else:
            specs = [dal.get_tensorstore_spec(os.path.join(spec_dir, s+'.json'), mip=working_mip_level) for s in secnames]
            loader0 = {'ImageLoaderType': 'TensorStoreLoader', 'json_spec': specs[0]}
            loader1 = {'ImageLoaderType': 'TensorStoreLoader', 'json_spec': specs[1]}
            loader0.update(loader_config)
            loader1.update(loader_config)
            loaders = [loader0, loader1]
        num_matches = match_section_from_initial_matches(mname, mesh_dir, loaders, match_dir, align_config)
        if num_matches is not None:
            if num_matches > 0:
                logger.info(f'{tname}: {num_matches} matches, {round((time.time()-t0)/60,3)} min.')
            else:
                logger.warning(f'{tname}: {num_matches} matches, {round((time.time()-t0)/60,3)} min.')
        gc.collect()
    logger.info('matching finished.')
    logging.terminate_logger(*logger_info)


def optimize_main(section_list):
    from feabas.aligner import Stack
    stack_config = align_config.get('stack_config', {}).copy()
    slide_window = align_config.get('slide_window', {}).copy()
    logger_info = logging.initialize_main_logger(logger_name='align_optimization', mp=num_workers>1)
    stack_config.setdefault('section_order_file', os.path.join(root_dir, 'section_order.txt'))
    slide_window['logger'] = logger_info[0]
    logger = logging.get_logger(logger_info[0])
    stk = Stack(section_list=section_list, mesh_dir=mesh_dir, match_dir=match_dir, mesh_out_dir=tform_dir, **stack_config)
    section_list = stk.section_list
    stk.update_lock_flags({s: os.path.isfile(os.path.join(tform_dir, s + '.h5')) for s in section_list})
    locked_flags = stk.locked_array
    logger.info(f'{locked_flags.size} images| {np.sum(locked_flags)} references')
    cost = stk.optimize_slide_window(optimize_rigid=True, optimize_elastic=True,
        target_gear=const.MESH_GEAR_MOVING, **slide_window)
    if os.path.isfile(os.path.join(tform_dir, 'residue.csv')):
        cost0 = {}
        with open(os.path.join(tform_dir, 'residue.csv'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                mn, dis0, dis1 = line.split(', ')
                cost0[mn] = (float(dis0), float(dis1))
        cost0.update(cost)
        cost = cost0
    with open(os.path.join(tform_dir, 'residue.csv'), 'w') as f:
        mnames = sorted(list(cost.keys()))
        for key in mnames:
            val = cost[key]
            f.write(f'{key}, {val[0]}, {val[1]}\n')
    logger.info('finished')
    logging.terminate_logger(*logger_info)


def offset_bbox_main():
    logger_info = logging.initialize_main_logger(logger_name='offset_bbox', mp=False)
    logger = logging.get_logger(logger_info[0])
    outname = os.path.join(tform_dir, 'offset.txt')
    tform_list = sorted(glob.glob(os.path.join(tform_dir, '*.h5')))
    if os.path.isfile(outname) or (len(tform_list) == 0):
        return
    secnames = [os.path.splitext(os.path.basename(s))[0] for s in tform_list]
    mip_level = align_config.pop('get', 0)
    outdir = os.path.join(render_dir, 'mip'+str(mip_level))
    for sname in secnames:
        if os.path.isdir(os.path.join(outdir, sname)):
            logger.info(f'section {sname} already rendered: transformation not performed')
            return
    bbox_union = None
    for tname in tform_list:
        M = Mesh.from_h5(tname)
        M.change_resolution(config.montage_resolution())
        bbox = M.bbox(gear=const.MESH_GEAR_MOVING, offsetting=True)
        if bbox_union is None:
            bbox_union = bbox
        else:
            bbox_union = common.bbox_union((bbox_union, bbox))
    offset = -bbox_union[:2]
    bbox_union_new = bbox_union + np.tile(offset, 2)
    if not os.path.isfile(outname):
        with open(outname, 'w') as f:
            f.write('\t'.join([str(s) for s in offset]))
    logger.warning(f'bbox offset: {tuple(bbox_union)} -> {tuple(bbox_union_new)}')
    logging.terminate_logger(*logger_info)


def render_one_section(h5name, z_prefix='', **kwargs):
    logger_info = kwargs.pop('logger', None)
    logger = logging.get_logger(logger_info)
    mip_level = kwargs.pop('mip_level', 0)
    offset = kwargs.pop('offset', None)
    secname = os.path.splitext(os.path.basename(h5name))[0]
    outdir = os.path.join(render_dir, 'mip'+str(mip_level), z_prefix+secname)
    resolution = config.montage_resolution() * (2 ** mip_level)
    meta_name = os.path.join(outdir, 'metadata.txt')
    if os.path.isfile(meta_name):
        return None
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    stitch_config = config.stitch_configs().get('rendering', {})
    loader_config = kwargs.pop('loader_config', {}).copy()
    loader_config.update({key: val for key, val in stitch_config.items() if key in ('pattern', 'one_based', 'fillval')})
    stitch_render_dir = config.stitch_render_dir()
    stitched_image_dir = os.path.join(stitch_render_dir, 'mip'+str(mip_level))
    loader_config['resolution'] = resolution
    if stitch_config.get('driver', 'image') == 'image':
        loader = get_image_loader(os.path.join(stitched_image_dir, secname), **loader_config)
    else:
        stitch_dir = os.path.join(root_dir, 'stitch')
        loader_dir = os.path.join(stitch_dir, 'ts_specs', secname + '.json')
        loader = VolumeRenderer._get_loader(loader_dir, mip=mip_level)
    M = Mesh.from_h5(h5name)
    M.change_resolution(resolution)
    if offset is not None:
        M.apply_translation(offset * config.montage_resolution()/resolution, gear=const.MESH_GEAR_MOVING)
    os.makedirs(outdir, exist_ok=True)
    prefix = os.path.join(outdir, secname)
    rendered = render_whole_mesh(M, loader, prefix, **kwargs)
    fnames = sorted(list(rendered.keys()))
    bboxes = []
    for fname in fnames:
        bboxes.append(rendered[fname])
    out_loader = dal.StaticImageLoader(fnames, bboxes=bboxes, resolution=resolution)
    out_loader.to_coordinate_file(meta_name)
    logger.info(f'{secname}: {len(rendered)} tiles | {time.time()-t0} secs.')
    return len(rendered)


def render_main(tform_list, z_prefix=None):
    logger_info = logging.initialize_main_logger(logger_name='align_render', mp=False)
    align_config['logger'] = logger_info[0]
    logger = logging.get_logger(logger_info[0])
    num_workers = align_config.get('num_workers', 1)
    cache_size = align_config.get('loader_config', {}).get('cache_size', None)
    if (cache_size is not None) and (num_workers > 1):
        align_config.setdefault('loader_config', {})
        align_config['loader_config'].setdefault('cache_size', cache_size // num_workers)
    offset_name = os.path.join(tform_dir, 'offset.txt')
    if os.path.isfile(offset_name):
        with open(offset_name, 'r') as f:
            line = f.readline()
        offset = np.array([float(s) for s in line.strip().split('\t')])
        logger.info(f'use offset {offset}')
    else:
        offset = None
    if z_prefix is None:
        z_prefix = {}
    for tname in tform_list:
        z = z_prefix.get(os.path.basename(tname), '')
        render_one_section(tname, z_prefix=z, offset=offset, **align_config)
    logger.info('finished')
    logging.terminate_logger(*logger_info)


def generate_aligned_mipmaps(render_dir, max_mip, meta_list=None, **kwargs):
    min_mip = kwargs.pop('min_mip', 0)
    num_workers = kwargs.pop('num_workers', 1)
    parallel_within_section = kwargs.pop('parallel_within_section', True)
    logger_info = logging.initialize_main_logger(logger_name='align_mipmap', mp=num_workers>0)
    kwargs['logger'] = logger_info[0]
    logger = logging.get_logger(logger_info[0])
    if meta_list is None:
        meta_list = sorted(glob.glob(os.path.join(render_dir, 'mip'+str(min_mip), '**', 'metadata.txt'), recursive=True))
    secnames = [os.path.basename(os.path.dirname(s)) for s in meta_list]
    if parallel_within_section or (num_workers == 1):
        for sname in secnames:
            mip_map_one_section(sname, render_dir, max_mip, num_workers=num_workers, **kwargs)
    else:
        target_func = partial(mip_map_one_section, img_dir=render_dir,
                                max_mip=max_mip, num_workers=1, **kwargs)
        jobs = []
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=get_context('spawn')) as executor:
            for sname in secnames:
                job = executor.submit(target_func, sname)
                jobs.append(job)
            for job in jobs:
                job.result()
    logger.info('mipmapping generated.')
    logging.terminate_logger(*logger_info)


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Run alignment")
    parser.add_argument("--mode", metavar="mode", type=str, default='matching')
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

    align_config = config.align_configs()
    if args.mode.lower().startswith('r'):
        align_config = align_config['rendering']
        mode = 'rendering'
        num_workers = align_config.get('num_workers', 1)
        if num_workers > num_cpus:
            num_workers = num_cpus
            align_config['num_workers'] = num_workers
    elif args.mode.lower().startswith('o'):
        align_config = align_config['optimization']
        mode = 'optimization'
        start_loc = align_config.get('slide_window', {}).get('start_loc', 'M')
        if start_loc.upper() == 'M':
            num_workers = min(2, align_config.get('slide_window', {}).get('num_workers', 2))
        else:
            num_workers = 1
        if num_workers > num_cpus:
            num_workers = num_cpus
            align_config.setdefault('slide_window', {})
            align_config['slide_window']['num_workers'] = num_workers
    elif args.mode.lower().startswith('ma'):
        mesh_config = align_config['meshing']
        align_config = align_config['matching']
        mode = 'matching'
        num_workers = align_config.get('matcher_config', {}).get('num_workers', 1)
        if num_workers > num_cpus:
            num_workers = num_cpus
            align_config.setdefault('matcher_config', {})
            align_config['matcher_config']['num_workers'] = num_workers
            mesh_config['num_workers'] = min(num_workers, mesh_config.get('num_workers', num_workers))
    elif args.mode.lower().startswith('me'):
        mesh_config = align_config['meshing']
        mode = 'meshing'
        num_workers = mesh_config.get('num_workers', 1)
        if num_workers > num_cpus:
            num_workers = num_cpus
            mesh_config['num_workers'] = num_workers
    elif args.mode.lower().startswith('d'):
        min_mip = align_config.get('rendering', {}).get('mip_level', 0)
        mode = 'downsample'
        render_config = align_config.get('rendering', {})
        filename_config = {key:val for key, val in render_config.items() if key in ('pattern', 'one_based', 'tile_size')}
        if render_config.get('loader_config', {}).get('fillval', None) is not None:
            filename_config['fillval'] = render_config['loader_config']['fillval']
        filename_config.update(align_config['downsample'])
        align_config = filename_config
        num_workers = align_config.get('num_workers', 1)
        if num_workers > num_cpus:
            num_workers = num_cpus
            align_config['num_workers'] = num_workers
    elif args.mode.lower().startswith('tensor'):
        align_config = align_config['tensorstore_rendering']
        mode = 'tensorstore_rendering'
        num_workers = align_config.get('num_workers', 1)
        if num_workers > num_cpus:
            num_workers = num_cpus
            align_config['num_workers'] = num_workers
    else:
        raise RuntimeError(f'{args.mode} not supported mode.')
    nthreads = max(1, math.floor(num_cpus / num_workers))
    config.limit_numpy_thread(nthreads)

    from feabas import material, dal, common
    from feabas.mesh import Mesh
    from feabas.mipmap import get_image_loader, mip_map_one_section
    from feabas.aligner import match_section_from_initial_matches
    from feabas.renderer import render_whole_mesh, VolumeRenderer
    import numpy as np

    align_dir = os.path.join(root_dir, 'align')
    mesh_dir = os.path.join(align_dir, 'mesh')
    match_dir = os.path.join(align_dir, 'matches')
    tform_dir = os.path.join(align_dir, 'tform')
    thumbnail_dir = os.path.join(root_dir, 'thumbnail_align')
    thumb_match_dir = os.path.join(thumbnail_dir, 'matches')
    render_dir = config.align_render_dir()
    tensorstore_render_dir = config.tensorstore_render_dir()
    ts_flag_dir = os.path.join(align_dir, 'ts_spec')
    thumbnail_configs = config.thumbnail_configs()
    match_name_delimiter = thumbnail_configs.get('alignment', {}).get('match_name_delimiter', '__to__')

    stt_idx, stp_idx, step = args.start, args.stop, args.step
    if stp_idx == 0:
        stp_idx = None
    indx = slice(stt_idx, stp_idx, step)

    os.makedirs(mesh_dir, exist_ok=True)
    if mode == 'meshing':
        generate_mesh_main()
    elif mode == 'matching':
        os.makedirs(match_dir, exist_ok=True)
        generate_mesh_main()
        match_list = sorted(glob.glob(os.path.join(thumb_match_dir, '*.h5')))
        match_list = match_list[indx]
        if args.reverse:
            match_list = match_list[::-1]
        align_config.setdefault('match_name_delimiter',  match_name_delimiter)
        match_main(match_list)
    elif mode == 'optimization':
        os.makedirs(tform_dir, exist_ok=True)
        optimize_main(None)
    elif mode == 'rendering':
        if align_config.pop('offset_bbox', True):
            offset_name = os.path.join(tform_dir, 'offset.txt')
            if not os.path.isfile(offset_name):
                time.sleep(0.1 * (1 + (args.start % args.step))) # avoid racing
                offset_bbox_main()
        os.makedirs(render_dir, exist_ok=True)
        tform_list = sorted(glob.glob(os.path.join(tform_dir, '*.h5')))
        tform_list = tform_list[indx]
        if args.reverse:
            tform_list = tform_list[::-1]
        z_prefix = defaultdict(lambda: '')
        if align_config.pop('prefix_z_number', True):
            seclist = sorted(glob.glob(os.path.join(mesh_dir, '*.h5')))
            section_order_file = os.path.join(root_dir, 'section_order.txt')
            seclist, z_indx = common.rearrange_section_order(seclist, section_order_file)
            digit_num = math.ceil(math.log10(len(seclist)))
            z_prefix.update({os.path.basename(s): str(k).rjust(digit_num, '0')+'_'
                             for k, s in zip(z_indx, seclist)})
        render_main(tform_list, z_prefix)
    elif mode == 'downsample':
        max_mip = align_config.pop('max_mip', 8)
        meta_list = sorted(glob.glob(os.path.join(render_dir, 'mip'+str(min_mip), '**', 'metadata.txt'), recursive=True))
        meta_list = meta_list[indx]
        if args.reverse:
            meta_list = meta_list[::-1]
        generate_aligned_mipmaps(render_dir, max_mip=max_mip, meta_list=meta_list, min_mip=min_mip, **align_config)
    elif mode == 'tensorstore_rendering':
        logger_info = logging.initialize_main_logger(logger_name='tensorstore_render', mp=num_workers>1)
        logger = logging.get_logger(logger_info[0])
        mip_level = align_config.pop('mip_level', 0)
        align_config.pop('outdir', None)
        driver = align_config.get('driver', 'neuroglancer_precomputed')
        if driver == 'zarr':
            tensorstore_render_dir = tensorstore_render_dir + '0/'
        elif driver == 'n5':
            tensorstore_render_dir = tensorstore_render_dir + 's0/'
        tform_list = sorted(glob.glob(os.path.join(tform_dir, '*.h5')))
        section_order_file = os.path.join(root_dir, 'section_order.txt')
        tform_list, z_indx = common.rearrange_section_order(tform_list, section_order_file)
        stitch_dir = os.path.join(root_dir, 'stitch')
        loader_dir = os.path.join(stitch_dir, 'ts_specs')
        loader_list = [os.path.join(loader_dir, os.path.basename(s).replace('.h5', '.json')) for s in tform_list]
        resolution = config.montage_resolution() * (2 ** mip_level)
        vol_renderer = VolumeRenderer(tform_list, loader_list, tensorstore_render_dir,
                                      z_indx = z_indx, resolution=resolution,
                                      flag_dir = ts_flag_dir, **align_config)
        vol_renderer.render_volume(skip_indx=indx, logger=logger_info[0], **align_config)
        logger.info('finished')
        logging.terminate_logger(*logger_info)
