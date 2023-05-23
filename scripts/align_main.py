from collections import defaultdict
import argparse
import glob
from functools import partial
from concurrent.futures.process import ProcessPoolExecutor
from multiprocessing import get_context
import math
import os
import time
import yaml
import gc

from feabas import config, logging
import feabas.constant as const


def generate_mesh_from_mask(mask_names, outname, **kwargs):
    if os.path.isfile(outname):
        return
    from feabas import material, dal, spatial, mesh
    material_table = kwargs.get('material_table', material.MaterialTable())
    target_resolution = kwargs.get('target_resolution', config.DEFAULT_RESOLUTION)
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
        logger.warn(f'{secname}: mask does not exist.')
        return
    mesh_size = mesh_size * config.DEFAULT_RESOLUTION / src_resolution
    G = spatial.Geometry.from_image_mosaic(loader, material_table=material_table, resolution=src_resolution)
    PSLG = G.PSLG(region_tol=region_tols,  roi_tol=0, area_thresh=area_thresh)
    M = mesh.Mesh.from_PSLG(**PSLG, material_table=material_table, mesh_size=mesh_size, min_mesh_angle=20)
    M.change_resolution(target_resolution)
    mshname = os.path.splitext(os.path.basename(mask_name))[0]
    M.save_to_h5(outname, save_material=True, override_dict={'name': mshname})


def generate_mesh_main():
    logger_info = logging.initialize_main_logger(logger_name='mesh_generation', mp=num_workers>1)
    mesh_config['logger'] = logger_info[0]
    logger = logging.get_logger(logger_info[0])
    thumbnail_mip_lvl = thumbnail_configs.get('thumbnail_mip_level', 6)
    thumbnail_resolution = config.DEFAULT_RESOLUTION * (2 ** thumbnail_mip_lvl)
    thumbnail_mask_dir = os.path.join(thumbnail_dir, 'material_masks')
    match_list = glob.glob(os.path.join(thumb_match_dir, '*.h5'))
    match_names = [os.path.basename(s).replace('.h5', '').split(match_name_delimiter) for s in match_list]
    secnames = set([s for pp in match_names for s in pp])
    alt_mask_dir = mesh_config.get('mask_dir', None)
    alt_mask_mip_level = mesh_config.get('mask_mip_level', 4)
    alt_mask_resolution = config.DEFAULT_RESOLUTION * (2 ** alt_mask_mip_level)
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
    stitch_render_dir = config.stitch_render_dir()
    stitched_image_dir = os.path.join(stitch_render_dir, 'mip'+str(working_mip_level))
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
        loaders = [get_image_loader(os.path.join(stitched_image_dir, s), **loader_config) for s in secnames]
        num_matches = match_section_from_initial_matches(mname, mesh_dir, loaders, match_dir, align_config)
        if num_matches is not None:
            logger.info(f'{tname}: {num_matches} matches, {round((time.time()-t0)/60,3)} min.')
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
        for key, val in cost.items():
            f.write(f'{key}, {val[0]}, {val[1]}\n')
    logger.info('finished')
    logging.terminate_logger(*logger_info)


def render_one_section(h5name, **kwargs):
    logger_info = kwargs.pop('logger', None)
    logger = logging.get_logger(logger_info)
    mip_level = kwargs.pop('mip_level', 0)
    secname = os.path.splitext(os.path.basename(h5name))[0]
    outdir = os.path.join(render_dir, 'mip'+str(mip_level), secname)
    resolution = config.DEFAULT_RESOLUTION * (2 ** mip_level)
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
    loader = get_image_loader(os.path.join(stitched_image_dir, secname), **loader_config)
    M = Mesh.from_h5(h5name)
    M.change_resolution(resolution)
    prefix = os.path.join(outdir, secname)
    os.makedirs(prefix, exist_ok=True)
    prefix = os.path.join(prefix, secname)
    rendered = render_whole_mesh(M, loader, prefix, **kwargs)
    fnames = sorted(list(rendered.keys()))
    bboxes = []
    for fname in fnames:
        bboxes.append(rendered[fname])
    out_loader = dal.StaticImageLoader(fnames, bboxes=bboxes, resolution=resolution)
    out_loader.to_coordinate_file(meta_name)
    logger.info(f'{secname}: {len(rendered)} tiles | {time.time()-t0} secs.')
    return len(rendered)


def render_main(tform_list):
    logger_info = logging.initialize_main_logger(logger_name='align_render', mp=False)
    align_config['logger'] = logger_info[0]
    logger = logging.get_logger(logger_info[0])
    num_workers = align_config.get('num_workers', 1)
    cache_size = align_config.get('loader_config', {}).get('cache_size', None)
    if (cache_size is not None) and (num_workers > 1):
        align_config.setdefault('loader_config', {})
        align_config['loader_config'].setdefault('cache_size', cache_size // num_workers)
    for tname in tform_list:
        render_one_section(tname, **align_config)
    logger.info('finished')
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
    elif args.mode.lower().startswith('o'):
        align_config = align_config['optimization']
        mode = 'optimization'
        start_loc = align_config.get('slide_window', {}).get('start_loc', 'M')
        if start_loc.upper() == 'M':
            num_workers = align_config.get('slide_window', {}).get('num_workers', 2)
        else:
            num_workers = 1
    elif args.mode.lower().startswith('ma'):
        mesh_config = align_config['meshing']
        align_config = align_config['matching']
        mode = 'matching'
        num_workers = align_config.get('matcher_config', {}).get('num_workers', 1)
    else:
        mesh_config = align_config['meshing']
        mode = 'meshing'
        num_workers = mesh_config.get('num_workers', 1)
    nthreads = max(1, math.floor(num_cpus / num_workers))
    config.limit_numpy_thread(nthreads)

    from feabas import material, dal
    from feabas.mesh import Mesh
    from feabas.mipmap import get_image_loader
    from feabas.aligner import match_section_from_initial_matches
    from feabas.renderer import render_whole_mesh
    import numpy as np

    align_dir = os.path.join(root_dir, 'align')
    mesh_dir = os.path.join(align_dir, 'mesh')
    match_dir = os.path.join(align_dir, 'matches')
    tform_dir = os.path.join(align_dir, 'tform')
    thumbnail_dir = os.path.join(root_dir, 'thumbnail_align')
    thumb_match_dir = os.path.join(thumbnail_dir, 'matches')
    render_dir = config.align_render_dir()
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
        os.makedirs(render_dir, exist_ok=True)
        tform_list = sorted(glob.glob(os.path.join(tform_dir, '*.h5')))
        tform_list = tform_list[indx]
        if args.reverse:
            tform_list = tform_list[::-1]