from collections import defaultdict
import argparse
from functools import partial
import json
import math
import os
import time
import gc

from feabas import config, logging, storage
from feabas.concurrent import submit_to_workers
import feabas.constant as const

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40)) # for large masks in meshing

def generate_mesh_from_mask(mask_names, outname, **kwargs):
    if storage.file_exists(outname):
        return
    from feabas import material, dal, mesh
    material_table = kwargs.pop('material_table', material.MaterialTable())
    target_resolution = kwargs.pop('target_resolution', config.montage_resolution())
    mesh_size = kwargs.pop('mesh_size', 600)
    logger_info = kwargs.pop('logger', None)
    initial_tform = kwargs.pop('initial_tform', None)
    logger = logging.get_logger(logger_info)
    loader = None
    if not isinstance(material_table, material.MaterialTable):
        material_table = material.MaterialTable.from_pickleable(material_table)
    if 'exclude' in material_table.named_table:
        mat = material_table['exclude']
        fillval = mat.mask_label
    else:
        fillval = 255
    for mask_name, resolution in mask_names:
        if not storage.file_exists(mask_name):
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
    M = mesh.mesh_from_mask(loader, mesh_size=mesh_size, material_table=material_table, resolution=src_resolution, **kwargs)
    M.change_resolution(target_resolution)
    if initial_tform is not None:
        Mt = mesh.Mesh.from_h5(initial_tform)
        M = mesh.transform_mesh(M, Mt, gears=(const.MESH_GEAR_INITIAL, const.MESH_GEAR_FIXED), tgears=(const.MESH_GEAR_INITIAL, const.MESH_GEAR_MOVING))
    mshname = os.path.splitext(os.path.basename(mask_name))[0]
    M.save_to_h5(outname, save_material=True, override_dict={'name': mshname})


def generate_mesh_main():
    logger_info = logging.initialize_main_logger(logger_name='mesh_generation', mp=num_workers>1)
    mesh_config['logger'] = logger_info[0]
    logger = logging.get_logger(logger_info[0])
    thumbnail_mip_lvl = thumbnail_configs.get('thumbnail_mip_level', 6)
    thumbnail_resolution = config.montage_resolution() * (2 ** thumbnail_mip_lvl)
    thumbnail_mask_dir = storage.join_paths(thumbnail_dir, 'material_masks')
    match_list = storage.list_folder_content(storage.join_paths(thumb_match_dir, '*.h5'))
    match_names = [os.path.basename(s).replace('.h5', '').split(match_name_delimiter) for s in match_list]
    secnames = set([s for pp in match_names for s in pp])
    alt_mask_dir = mesh_config.get('mask_dir', None)
    alt_mask_mip_level = mesh_config.get('mask_mip_level', 4)
    alt_mask_resolution = config.montage_resolution() * (2 ** alt_mask_mip_level)
    if alt_mask_dir is None:
        alt_mask_dir = storage.join_paths(align_dir, 'material_masks')
    material_table_file = config.material_table_file()
    material_table = material.MaterialTable.from_json(material_table_file, stream=False)
    material_table = material_table.save_to_json(jsonname=None)
    mesh_func = partial(generate_mesh_from_mask, material_table=material_table, **mesh_config)
    kwargs_list = []
    for sname in secnames:
        outname = storage.join_paths(mesh_dir, sname + '.h5')
        if storage.file_exists(outname):
            continue
        mask_names = [(storage.join_paths(alt_mask_dir, sname + '.json'), alt_mask_resolution),
                        (storage.join_paths(alt_mask_dir, sname + '.txt'), alt_mask_resolution),
                        (storage.join_paths(alt_mask_dir, sname + '.png'), alt_mask_resolution),
                        (storage.join_paths(thumbnail_mask_dir, sname + '.png'), thumbnail_resolution)]
        initial_tform = storage.join_paths(initial_tform_dir, sname + '.h5')
        if not storage.file_exists(initial_tform):
            initial_tform = None
        kwargs_list.append({'mask_names': mask_names, 'outname': outname, 'initial_tform': initial_tform})
    for _ in submit_to_workers(mesh_func, kwargs=kwargs_list, num_workers=num_workers):
        pass
    logger.info('meshes generated.')
    logging.terminate_logger(*logger_info)


def match_main(match_list):
    stitch_config = config.stitch_configs().get('rendering', {})
    loader_config = {key: val for key, val in stitch_config.items() if key in ('pattern', 'one_based', 'fillval')}
    working_mip_level = align_config.get('working_mip_level', 2)
    stitch_render_driver = config.stitch_configs().get('rendering', {}).get('driver', 'image')
    if stitch_render_driver == 'image':
        stitch_render_dir = config.stitch_render_dir()
        stitched_image_dir = storage.join_paths(stitch_render_dir, 'mip'+str(working_mip_level))
    else:
        stitch_dir = storage.join_paths(root_dir, 'stitch')
        spec_dir = storage.join_paths(stitch_dir, 'ts_specs')
    logger_info = logging.initialize_main_logger(logger_name='align_matching', mp=False)
    logger = logging.get_logger(logger_info[0])
    if len(match_list) == 0:
        return
    for mname in match_list:
        outname = storage.join_paths(match_dir, os.path.basename(mname))
        if storage.file_exists(outname):
            continue
        t0 = time.time()
        tname = os.path.basename(mname).replace('.h5', '')
        logger.info(f'start {tname}')
        secnames = os.path.splitext(os.path.basename(mname))[0].split(match_name_delimiter)
        if stitch_render_driver == 'image':
            loaders = [get_image_loader(storage.join_paths(stitched_image_dir, s), **loader_config) for s in secnames]
        else:
            specs = [dal.get_tensorstore_spec(storage.join_paths(spec_dir, s+'.json'), mip=working_mip_level) for s in secnames]
            loader0 = {'ImageLoaderType': 'TensorStoreLoader', 'json_spec': specs[0]}
            loader1 = {'ImageLoaderType': 'TensorStoreLoader', 'json_spec': specs[1]}
            loader0.update(loader_config)
            loader1.update(loader_config)
            loaders = [loader0, loader1]
        ignore_initial_match = not storage.file_exists(mname)
        num_matches = match_section_from_initial_matches(mname, mesh_dir, loaders, match_dir, align_config, ignore_initial_match=ignore_initial_match)
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
    stack_config.setdefault('section_order_file', storage.join_paths(root_dir, 'section_order.txt'))
    slide_window['logger'] = logger_info[0]
    logger = logging.get_logger(logger_info[0])
    stk = Stack(section_list=section_list, mesh_dir=mesh_dir, match_dir=match_dir, mesh_out_dir=tform_dir, **stack_config)
    section_list = stk.section_list
    stk.update_lock_flags({s: storage.file_exists(storage.join_paths(tform_dir, s + '.h5')) for s in section_list})
    locked_flags = stk.locked_array
    logger.info(f'{locked_flags.size} images| {np.sum(locked_flags)} references')
    cost = stk.optimize_slide_window(optimize_rigid=True, optimize_elastic=True,
        target_gear=const.MESH_GEAR_MOVING, **slide_window)
    if storage.file_exists(storage.join_paths(tform_dir, 'residue.csv')):
        cost0 = {}
        with storage.File(storage.join_paths(tform_dir, 'residue.csv'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                mn, dis0, dis1 = line.split(', ')
                cost0[mn] = (float(dis0), float(dis1))
        cost0.update(cost)
        cost = cost0
    with storage.File(storage.join_paths(tform_dir, 'residue.csv'), 'w') as f:
        mnames = sorted(list(cost.keys()))
        for key in mnames:
            val = cost[key]
            f.write(f'{key}, {val[0]}, {val[1]}\n')
    logger.info('finished')
    logging.terminate_logger(*logger_info)


def _get_bbox_for_one_section(mname, resolution=None):
    from feabas.mesh import Mesh
    M = Mesh.from_h5(mname)
    if resolution is not None:
        M.change_resolution(resolution)
    bbox = M.bbox(gear=const.MESH_GEAR_MOVING, offsetting=True)
    return bbox


def offset_bbox_main():
    logger_info = logging.initialize_main_logger(logger_name='offset_bbox', mp=False)
    logger = logging.get_logger(logger_info[0])
    outname = storage.join_paths(tform_dir, 'offset.txt')
    tform_list = sorted(storage.list_folder_content(storage.join_paths(tform_dir, '*.h5')))
    if storage.file_exists(outname) or (len(tform_list) == 0):
        return
    num_workers = align_config.get('num_workers', 1)
    secnames = [os.path.splitext(os.path.basename(s))[0] for s in tform_list]
    mip_level = align_config.pop('get', 0)
    outdir = storage.join_paths(render_dir, 'mip'+str(mip_level))
    for sname in secnames:
        if storage.dir_exists(storage.join_paths(outdir, sname)):
            logger.info(f'section {sname} already rendered: transformation not performed')
            return
    bbox_union = None
    bfunc = partial(_get_bbox_for_one_section, resolution=config.montage_resolution())
    for bbox in submit_to_workers(bfunc, args=[(s,) for s in tform_list], num_workers=num_workers):
        if bbox_union is None:
            bbox_union = bbox
        else:
            bbox_union = common.bbox_union((bbox_union, bbox))
    offset = -bbox_union[:2]
    bbox_union_new = bbox_union + np.tile(offset, 2)
    if not storage.file_exists(outname):
        with storage.File(outname, 'w') as f:
            f.write('\t'.join([str(s) for s in offset]))
    logger.warning(f'bbox offsets at mip0: {tuple(bbox_union)} -> {tuple(bbox_union_new)}')
    logging.terminate_logger(*logger_info)


def render_one_section(h5name, z_prefix='', **kwargs):
    logger_info = kwargs.pop('logger', None)
    logger = logging.get_logger(logger_info)
    mip_level = kwargs.pop('mip_level', 0)
    offset = kwargs.pop('offset', None)
    secname = os.path.splitext(os.path.basename(h5name))[0]
    outdir = storage.join_paths(render_dir, 'mip'+str(mip_level), z_prefix+secname)
    resolution = config.montage_resolution() * (2 ** mip_level)
    meta_name = storage.join_paths(outdir, 'metadata.txt')
    if storage.file_exists(meta_name):
        return None
    storage.makedirs(outdir)
    t0 = time.time()
    stitch_config = config.stitch_configs().get('rendering', {})
    loader_config = kwargs.pop('loader_config', {}).copy()
    loader_config.update({key: val for key, val in stitch_config.items() if key in ('pattern', 'one_based', 'fillval')})
    stitch_render_dir = config.stitch_render_dir()
    stitched_image_dir = storage.join_paths(stitch_render_dir, 'mip'+str(mip_level))
    loader_config['resolution'] = resolution
    if stitch_config.get('driver', 'image') == 'image':
        loader = get_image_loader(storage.join_paths(stitched_image_dir, secname), **loader_config)
    else:
        stitch_dir = storage.join_paths(root_dir, 'stitch')
        loader_dir = storage.join_paths(stitch_dir, 'ts_specs', secname + '.json')
        loader = VolumeRenderer._get_loader(loader_dir, mip=mip_level, **loader_config)
    M = Mesh.from_h5(h5name)
    M.change_resolution(resolution)
    if offset is not None:
        M.apply_translation(offset * config.montage_resolution()/resolution, gear=const.MESH_GEAR_MOVING)
    prefix = storage.join_paths(outdir, secname)
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
        align_config['loader_config']['cache_size'] = cache_size // num_workers
    offset_name = storage.join_paths(tform_dir, 'offset.txt')
    if storage.file_exists(offset_name):
        with storage.File(offset_name, 'r') as f:
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
        meta_list = sorted(storage.list_folder_content(storage.join_paths(render_dir, 'mip'+str(min_mip), '**', 'metadata.txt'), recursive=True))
    secnames = [os.path.basename(os.path.dirname(s)) for s in meta_list]
    if parallel_within_section or (num_workers == 1):
        for sname in secnames:
            mip_map_one_section(sname, render_dir, max_mip, num_workers=num_workers, **kwargs)
    else:
        target_func = partial(mip_map_one_section, img_dir=render_dir,
                                max_mip=max_mip, num_workers=1, **kwargs)
        for _ in submit_to_workers(target_func, args=[(s,) for s in secnames], num_workers=num_workers):
            pass
    logger.info('mipmapping generated.')
    logging.terminate_logger(*logger_info)


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Run alignment")
    parser.add_argument("--mode", metavar="mode", type=str, default='matching')
    parser.add_argument("--start", metavar="start", type=int, default=0)
    parser.add_argument("--step", metavar="step", type=int, default=1)
    parser.add_argument("--stop", metavar="stop", type=int)
    parser.add_argument("--reverse",  action='store_true')
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args()

    root_dir = config.get_work_dir()

    align_config = config.align_configs()
    if args.mode.lower().startswith('r'):
        align_config = align_config['rendering']
        mode = 'rendering'
        num_workers = align_config.get('num_workers', 1)
        num_workers = config.set_numpy_thread_from_num_workers(num_workers)
        align_config['num_workers'] = num_workers
    elif args.mode.lower().startswith('o'):
        align_config = align_config['optimization']
        mode = 'optimization'
        start_loc = align_config.get('slide_window', {}).get('start_loc', 'M')
        if start_loc.upper() == 'M':
            num_workers = min(2, align_config.get('slide_window', {}).get('num_workers', 2))
        else:
            num_workers = 1
        num_workers = config.set_numpy_thread_from_num_workers(num_workers)
        align_config.setdefault('slide_window', {})
        align_config['slide_window']['num_workers'] = num_workers
    elif args.mode.lower().startswith('ma'):
        mesh_config = align_config['meshing']
        align_config = align_config['matching']
        mode = 'matching'
        num_workers = align_config.get('matcher_config', {}).get('num_workers', 1)
        num_workers = config.set_numpy_thread_from_num_workers(num_workers)
        align_config.setdefault('matcher_config', {})
        align_config['matcher_config']['num_workers'] = num_workers
        mesh_config['num_workers'] = min(num_workers, mesh_config.get('num_workers', num_workers))
    elif args.mode.lower().startswith('me'):
        mesh_config = align_config['meshing']
        mode = 'meshing'
        num_workers = mesh_config.get('num_workers', 1)
        num_workers = config.set_numpy_thread_from_num_workers(num_workers)
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
        num_workers = config.set_numpy_thread_from_num_workers(num_workers)
        align_config['num_workers'] = num_workers
    elif args.mode.lower().startswith('tensorstore_r') or args.mode.lower().startswith('tsr'):
        align_config = align_config['tensorstore_rendering']
        mode = 'tensorstore_rendering'
        num_workers = align_config.get('num_workers', 1)
        num_workers = config.set_numpy_thread_from_num_workers(num_workers)
        align_config['num_workers'] = num_workers
    elif  args.mode.lower().startswith('tensorstore_d') or args.mode.lower().startswith('tsd'):
        align_config = align_config['tensorstore_downsample']
        mode = 'tensorstore_downsample'
        num_workers = align_config.get('num_workers', 1)
        num_workers = config.set_numpy_thread_from_num_workers(num_workers)
        align_config['num_workers'] = num_workers
    else:
        raise RuntimeError(f'{args.mode} not supported mode.')


    from feabas import material, dal, common
    from feabas.mesh import Mesh
    from feabas.mipmap import get_image_loader, mip_map_one_section, mip_one_level_tensorstore_3d
    from feabas.aligner import match_section_from_initial_matches
    from feabas.renderer import render_whole_mesh, VolumeRenderer
    import numpy as np

    align_dir = storage.join_paths(root_dir, 'align')
    mesh_dir = storage.join_paths(align_dir, 'mesh')
    match_dir = storage.join_paths(align_dir, 'matches')
    tform_dir = storage.join_paths(align_dir, 'tform')
    match_filename = storage.join_paths(align_dir, 'match_name.txt')
    thumbnail_dir = storage.join_paths(root_dir, 'thumbnail_align')
    initial_tform_dir = storage.join_paths(thumbnail_dir, 'tform')
    thumb_match_dir = storage.join_paths(thumbnail_dir, 'matches')
    render_dir = config.align_render_dir()
    tensorstore_render_dir = config.tensorstore_render_dir()
    ts_flag_dir = storage.join_paths(align_dir, 'done_flags')
    ts_spec_file = storage.join_paths(align_dir, 'ts_spec.json')
    ts_mip_flagdir = storage.join_paths(align_dir, 'mipmap_flags')
    thumbnail_configs = config.thumbnail_configs()
    match_name_delimiter = thumbnail_configs.get('alignment', {}).get('match_name_delimiter', '__to__')

    stt_idx, stp_idx, step = args.start, args.stop, args.step
    if (stt_idx in (0, None)) and (stp_idx is None) and (step in (1, None)):
        full_run = True
    else:
        full_run = False
    indx = slice(stt_idx, stp_idx, step)
    storage.makedirs(mesh_dir)
    if mode == 'meshing':
        generate_mesh_main()
    elif mode == 'matching':
        storage.makedirs(match_dir)
        generate_mesh_main()
        if storage.file_exists(match_filename):
            with storage.File(match_filename, 'r') as f:
                match_list0 = f.readlines()
            match_list = []
            for s in match_list0:
                s = s.strip()
                s = s.replace('\t', match_name_delimiter)
                if not s.endswith('.h5'):
                    s = s + '.h5'
                s = storage.join_paths(thumb_match_dir, s)
                match_list.append(s)
        else:
            match_list = sorted(storage.list_folder_content(storage.join_paths(thumb_match_dir, '*.h5')))
        match_list = match_list[indx]
        if args.reverse:
            match_list = match_list[::-1]
        align_config.setdefault('match_name_delimiter',  match_name_delimiter)
        match_main(match_list)
    elif mode == 'optimization':
        storage.makedirs(tform_dir)
        optimize_main(None)
    elif mode == 'rendering':
        if align_config.pop('offset_bbox', True):
            offset_name = storage.join_paths(tform_dir, 'offset.txt')
            if not storage.file_exists(offset_name):
                time.sleep(0.1 * (1 + (args.start % args.step))) # avoid racing
                offset_bbox_main()
        storage.makedirs(render_dir)
        tform_list = sorted(storage.list_folder_content(storage.join_paths(tform_dir, '*.h5')))
        tform_list = tform_list[indx]
        z_prefix = defaultdict(lambda: '')
        if align_config.pop('prefix_z_number', True):
            seclist = sorted(storage.list_folder_content(storage.join_paths(mesh_dir, '*.h5')))
            section_order_file = storage.join_paths(root_dir, 'section_order.txt')
            seclist, z_indx = common.rearrange_section_order(seclist, section_order_file)
            digit_num = math.ceil(math.log10(len(seclist)))
            z_prefix.update({os.path.basename(s): str(k).rjust(digit_num, '0')+'_'
                             for k, s in zip(z_indx, seclist)})
        render_main(tform_list, z_prefix)
    elif mode == 'downsample':
        max_mip = align_config.pop('max_mip', 8)
        meta_list = sorted(storage.list_folder_content(storage.join_paths(render_dir, 'mip'+str(min_mip), '**', 'metadata.txt'), recursive=True))
        meta_list = meta_list[indx]
        if args.reverse:
            meta_list = meta_list[::-1]
        generate_aligned_mipmaps(render_dir, max_mip=max_mip, meta_list=meta_list, min_mip=min_mip, **align_config)
    elif mode == 'tensorstore_rendering':
        logger_info = logging.initialize_main_logger(logger_name='tensorstore_render', mp=num_workers>1)
        logger = logging.get_logger(logger_info[0])
        mip_level = align_config.pop('mip_level', 0)
        align_config.pop('out_dir', None)
        canvas_bbox = align_config.get('canvas_bbox', None)
        if (canvas_bbox is None):
            canvas_file = storage.join_paths(tform_dir, 'tensorstore_canvas.txt')
            if storage.file_exists(canvas_file):
                with storage.File(canvas_file, 'r') as f:
                    line = f.readline()
                canvas_bbox = [float(s) for s in line.strip().split('\t')]
                logger.info(f'use canvas bounding box {canvas_bbox}')
                align_config['canvas_bbox'] = canvas_bbox
        driver = align_config.get('driver', 'neuroglancer_precomputed')
        if driver == 'zarr':
            tensorstore_render_dir = tensorstore_render_dir + '0/'
        elif driver == 'n5':
            tensorstore_render_dir = tensorstore_render_dir + 's0/'
        tform_list = sorted(storage.list_folder_content(storage.join_paths(tform_dir, '*.h5')))
        section_order_file = storage.join_paths(root_dir, 'section_order.txt')
        tform_list, z_indx = common.rearrange_section_order(tform_list, section_order_file)
        stitch_dir = storage.join_paths(root_dir, 'stitch')
        loader_dir = storage.join_paths(stitch_dir, 'ts_specs')
        loader_list = [storage.join_paths(loader_dir, os.path.basename(s).replace('.h5', '.json')) for s in tform_list]
        resolution = config.montage_resolution() * (2 ** mip_level)
        vol_renderer = VolumeRenderer(tform_list, loader_list, tensorstore_render_dir,
                                      z_indx = z_indx, resolution=resolution,
                                      flag_dir = ts_flag_dir, **align_config)
        out_spec = vol_renderer.render_volume(skip_indx=indx, logger=logger_info[0], **align_config)
        with storage.File(ts_spec_file, 'w') as f:
            json.dump({mip_level: out_spec}, f)
        logger.info('finished')
        logging.terminate_logger(*logger_info)
    elif mode == 'tensorstore_downsample':
        logger_info = logging.initialize_main_logger(logger_name='tensorstore_downsample', mp=num_workers>1)
        logger = logging.get_logger(logger_info[0])
        mip_levels = align_config.pop('mip_levels', np.arange(1, 9))
        kvstore_out = align_config.pop('out_dir', None)
        z_range = align_config.pop('z_range', None)
        if storage.file_exists(ts_spec_file):
            with storage.File(ts_spec_file, 'r') as f:
                rendered_mips_spec = json.load(f)
        else:
            raise RuntimeError('no rendered mip0 found, run rendering code first...')
        rendered_mips_spec = {int(mip): spec for mip, spec in rendered_mips_spec.items()}
        rendered_mips = np.array(sorted(list(rendered_mips_spec.keys())))
        if (z_range is not None) or (not full_run):
            mip0_spec = rendered_mips_spec[min(rendered_mips)]
            mip0_writer = dal.TensorStoreWriter.from_json_spec(mip0_spec)
            Z0, Z1 = mip0_writer.write_grids[2], mip0_writer.write_grids[5]
            Z_ptp = Z1.max() - Z0.min()
            stt_idx = indx.start
            stp_idx = indx.stop
            step = indx.step
            if z_range is None:
                z_range = [np.min(Z0), np.max(Z1)]
            if stt_idx is not None:
                z_range[0] = max(z_range[0], Z0[stt_idx])
            else:
                stt_idx = 0
            if stp_idx is not None:
                z_range[1] = min(z_range[-1], Z1[stp_idx-1])
            zr0 = (min(z_range) - Z0.min()) / Z_ptp
            zr1 = (max(z_range) - Z0.min()) / Z_ptp
            if (step is not None) and (step > 1):
                dr = zr1 - zr0
                rr = 1/step
                zr0 = zr0 + (stt_idx % step) * dr * rr
                zr1 = zr0 + dr * rr
            z_range = [zr0, zr1]
        downsample_z = align_config.pop('downsample_z', 'auto')
        if downsample_z == 'auto':
            downsample_z = ['auto'] * len(mip_levels)
        elif not hasattr(downsample_z, '__len__'):
            downsample_z = [downsample_z] * len(mip_levels)
        elif len(downsample_z) == 1:
            downsample_z = list(downsample_z) * len(mip_levels)
        for mip, dsp in zip(sorted(mip_levels), downsample_z):
            if mip in rendered_mips_spec:
                continue
            higher_mips = rendered_mips[rendered_mips < mip]
            if higher_mips.size == 0:
                logger.error(f'No previously rendered volume higer than mip{mip} found in {ts_spec_file}.')
                continue
            src_mip = np.max(higher_mips)
            src_spec = rendered_mips_spec[src_mip]
            if kvstore_out is not None:
                tdriver, kvstore_out = storage.parse_file_driver(kvstore_out)
                if tdriver == 'file':
                    kvstore_out = 'file://' + kvstore_out
            mipup = mip - src_mip
            flag_prefix = storage.join_paths(ts_mip_flagdir, f'mip{mip}_')
            err_raised, out_spec, z_range = mip_one_level_tensorstore_3d(src_spec, 
                                                                         mipup=mipup,
                                                                         kvstore_out=kvstore_out,
                                                                         logger=logger_info,
                                                                         flag_prefix=flag_prefix,
                                                                         full_chunk_only=(not full_run),
                                                                         downsample_z=dsp, 
                                                                         **align_config)
            if err_raised:
                logger.error('failed to generate mip{mip}, abort')
                break
            if (z_range is not None) and len(z_range) == 0:
                logger.warning('no complete chunk to downsample at mip{mip}. skipping...')
                break
            rendered_mips_spec[mip] = out_spec
            align_config['z_range'] = z_range
            if full_run:
                flag_list = storage.list_folder_content(flag_prefix + '*.json')
                flag_out = flag_prefix + 'f.json'
                if len(flag_list > 1):
                    z_rendered = set()
                    for flgfile in flag_list:
                        with storage.File(flgfile, 'r') as f:
                            zidrnd = json.load(f)
                            z_rendered = z_rendered.union(zidrnd)
                    z_rendered = sorted(list(z_rendered))
                    with storage.File(flag_out, 'w') as f:
                        json.dump(z_rendered, f)
                    for flgfile in flag_list:
                        if os.path.basename(flgfile) != os.path.basename(flag_out):
                            storage.remove_file(flgfile)
                with storage.File(ts_spec_file, 'w') as f:
                    json.dump(rendered_mips_spec, f)
            logger.info(f'mip{mip} generated')
            
