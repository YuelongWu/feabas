import argparse
from functools import partial
import os
import time

import feabas
from feabas.concurrent import submit_to_workers
from feabas import config, logging, storage

MAX_CACHE_SCALE_MIP = 3 # stop growing the tile count of the render cache with the mip level beyond this


def match_one_section(coordname, outname, **kwargs):
    logger_info = kwargs.get('logger', None)
    logger = logging.get_logger(logger_info)
    stitcher = Stitcher.from_coordinate_file(coordname)
    if storage.file_exists(outname + '_err'):
        logger.info(f'loading previous results for {os.path.basename(coordname)}')
        stitcher.load_matches_from_h5(outname + '_err', check_order=True)
    if storage.file_exists(outname + '_mesh'):
        msh_loaded = stitcher.load_meshes_from_h5(outname + '_mesh', check_order=True)
        if msh_loaded:
            logger.info(f'loading previous transformation for {os.path.basename(coordname)}')
    _, err = stitcher.dispatch_matchers(verbose=False, **kwargs)
    if err:
        outname = outname + '_err'
    stitcher.save_to_h5(outname, save_matches=True, save_meshes=False)
    return 1


def match_main(coord_list, out_dir, **kwargs):
    num_workers = kwargs.get('num_workers', 1)
    logger_info = logging.initialize_main_logger(logger_name='stitch_matching', mp=num_workers>1)
    kwargs['logger'] = logger_info[0]
    logger= logging.get_logger(logger_info[0])
    for coordname in coord_list:
        t0 = time.time()
        fname = os.path.basename(coordname).replace('.txt', '').replace('.tsv', '')
        outname = storage.join_paths(out_dir, fname + '.h5')
        if storage.file_exists(outname, use_cache=True):
            continue
        logger.info(f'starting matching for {fname}')
        flag = match_one_section(coordname, outname, **kwargs)
        if flag == 1:
            logger.info(f'ending for {fname}: {(time.time()-t0)/60} min')
    logger.info('finished.')
    logging.terminate_logger(*logger_info)


def optimize_one_section(matchname, outname, **kwargs):
    from feabas.stitcher import Stitcher
    from feabas import multisem
    import numpy as np
    if storage.file_exists(outname, use_cache=True):
        return
    use_group = kwargs.get('use_group', True)
    msem = kwargs.get('msem', False)
    mesh_settings = kwargs.get('mesh_settings', {})
    translation_settings = kwargs.get('translation', {})
    affine_settings = kwargs.get('affine', {})
    group_elastic_settings = kwargs.get('group_elastic', {})
    elastic_settings = kwargs.get('final_elastic', {})
    disconnected_settings = kwargs.get('disconnected_assemble', {})
    normalize_setting = kwargs.get('normalize', {})
    minweight = kwargs.get('minweight', None)
    logger_info = kwargs.get('logger', None)
    logger = logging.get_logger(logger_info)
    bname = os.path.basename(matchname).replace('.h5', '')
    mesh_settings = mesh_settings.copy()
    mesh_sizes = mesh_settings.pop('mesh_sizes', [100, 300])
    t0 = time.time()
    stitcher = Stitcher.from_h5(matchname, load_matches=True, load_meshes=False)
    if kwargs.get('equalize_brightness_contrast', False):
        stitcher.equalize_brightness_contrast()
    if stitcher.num_links == 0:
        stitcher.initialize_meshes(mesh_sizes, **mesh_settings)
        stitcher.initialize_optimizer()
        stitcher.normalize_coordinates(**normalize_setting)
        stitcher.save_to_h5(outname, save_matches=False, save_meshes=True)
        logger.warning(f'\t{bname}: no matches found, use stage coordinates.')
        return
    if minweight is not None:
        rejected = stitcher.filter_match_by_weight(minweight)
        if rejected > 0:
            logger.debug(f'{bname}: filtered out {rejected} low-conf matches')
    if use_group:
        if msem:
            groupings, _ = multisem.mfovids_beamids_from_filenames(stitcher.imgrelpaths)
        else:
            groupings = np.zeros(stitcher.num_tiles, dtype=np.int32)
    else:
        groupings = None
    stitcher.set_groupings(groupings)
    stitcher.initialize_meshes(mesh_sizes, **mesh_settings)
    discrd = 0
    if msem:
        discrd0, cost = multisem.filter_links_from_sfov_pattern(stitcher, target_gear=feabas.MESH_GEAR_FIXED, **translation_settings)
        discrd += discrd0
    discrd0, cost = stitcher.optimize_translation(target_gear=feabas.MESH_GEAR_FIXED, **translation_settings)
    discrd += discrd0
    dis = stitcher.match_residues()
    logger.info(f'{bname}: residue after translation {np.nanmean(dis)} | discarded {discrd}')
    if affine_settings.get('maxiter', 0) != 0:
        cost=stitcher.optimize_affine(target_gear=feabas.MESH_GEAR_FIXED, **affine_settings)
        dis = stitcher.match_residues()
        logger.info(f'{bname}: residue after affine {np.nanmean(dis)} | cost {cost}')
    if use_group and (group_elastic_settings.get('maxiter', 0) != 0):
        stitcher.optimize_group_intersection(target_gear=feabas.MESH_GEAR_FIXED, **group_elastic_settings)
        stitcher.optimize_translation(target_gear=feabas.MESH_GEAR_FIXED, **translation_settings)
        cost = stitcher.optimize_elastic(use_groupings=True, target_gear=feabas.MESH_GEAR_FIXED, **group_elastic_settings)
        dis = stitcher.match_residues()
        logger.info(f'{bname}: residue after grouped relaxation {np.nanmean(dis)} | cost {cost}')
    if elastic_settings.get('maxiter', 0) != 0:
        cost = stitcher.optimize_elastic(target_gear=feabas.MESH_GEAR_MOVING, **elastic_settings)
    rot, _ = stitcher.normalize_coordinates(**normalize_setting)
    Ninvld = stitcher.correct_invalid_meshes()
    N_conn = stitcher.connect_isolated_subsystem(**disconnected_settings)
    if N_conn > 1:
        rot_1, _ = stitcher.normalize_coordinates(**normalize_setting)
        rot = max(rot, rot_1)
    if cost[0] is None or cost[1] is None or cost[0] < cost[1]:
        stitcher.save_to_h5(outname.replace('.h5', '.h5_err'), save_matches=False, save_meshes=True)
        logger.error(f'{bname}: failed to converge.')
    else:
        stitcher.save_to_h5(outname, save_matches=False, save_meshes=True)
    lbl_conn = stitcher.connected_subsystem
    _, cnt = np.unique(lbl_conn, return_counts=True)
    ncomp = cnt.size
    ncomp1 = np.sum(cnt>1)
    dis = stitcher.match_residues()
    finish_str = (f'|{bname}| cost: {cost} '
                  + f'| residue: {np.nanmean(dis)} | discarded: {np.sum(np.isnan(dis))} | rotation: {round(rot)} '
                  + f'| {ncomp1}/{ncomp} components '
                  + f'| finished {time.time() - t0} sec ')
    # if abs(rot) > 1.5:
    #     logger.warning(f'{bname}: rotation detected in final transform, potential mesh relaxation issues.')
    if Ninvld > 0:
        logger.warning(f'\t{bname}: corrected invalid {Ninvld} meshes.')
    if ncomp1 > 1:
        logger.warning(f'\t{bname}: {ncomp} disconnected groups found, among which {ncomp1} have more than one tiles.')
    logger.info(finish_str)


def optmization_main(match_list, out_dir, **kwargs):
    num_workers = kwargs.pop('num_workers', 1)
    logger_info = logging.initialize_main_logger(logger_name='stitch_optmization', mp=num_workers>1)
    kwargs['logger'] = logger_info[0]
    logger= logging.get_logger(logger_info[0])
    target_func = partial(optimize_one_section, **kwargs)
    args_list = []
    for matchname in match_list:
        outname = storage.join_paths(out_dir, os.path.basename(matchname))
        if storage.file_exists(outname, use_cache=True):
            continue
        args_list.append((matchname, outname))
    for _ in submit_to_workers(target_func, args=args_list, num_workers=num_workers):
        pass
    logger.info('finished.')
    logging.terminate_logger(*logger_info)


def render_one_section(tform_name, out_prefix, meta_name=None, **kwargs):
    mask_dir = kwargs.pop('mask_dir', None)
    tfname = kwargs.pop('tfname', None)
    mip = kwargs.get('mip', 0)
    loader_settings = kwargs.pop('loader_settings', {})
    if loader_settings.get('cache_size', None) is not None:
        loader_settings = loader_settings.copy()
        # cache_size (per worker) counts source tiles: at mip k a downsampled tile takes
        # 4^-k the memory, while each output tile covers ~4^k more of them.
        cache_scale = 4 ** min(mip, MAX_CACHE_SCALE_MIP)
        loader_settings['cache_size'] = loader_settings['cache_size'] * cache_scale
    kwargs['resolution'] = config.mip0_resolution() * (2 ** mip)
    renderer = MontageRenderer.from_h5(tform_name, loader_settings=loader_settings, **kwargs)
    if tfname is not None:
        renderer.add_transfer_functions(tfname)
    mask_out = storage.join_paths(mask_dir, os.path.basename(tform_name).replace('.h5','.png'))
    num_chunks = renderer.render_one_section(out_prefix, meta_name=meta_name, mask_out=mask_out, **kwargs)
    return num_chunks


def downsample_one_section(sec_name, src_mip, tgt_mip, out_dir, **kwargs):
    """generate stitched mip level tgt_mip from the rendered level src_mip."""
    driver = kwargs.get('driver', 'image')
    num_workers = kwargs.get('num_workers', 1)
    logger_info = kwargs.get('logger', None)
    provenance_file = kwargs.get('provenance_file', None)
    ds_format = config.thumbnail_configs().get('downsample', {}).get('format', 'jpg')
    if driver == 'image':
        filename_settings = kwargs.get('filename_settings', {})
        mip_kwargs = {'pattern': filename_settings.get('pattern', '_tr{ROW_IND}-tc{COL_IND}.png'),
                      'one_based': filename_settings.get('one_based', True),
                      'fillval': kwargs.get('loader_settings', {}).get('fillval', 0)}
        res = mipmap.mip_map_one_section(sec_name, out_dir, tgt_mip, min_mip=src_mip, num_workers=num_workers,
                                         format=ds_format, logger=logger_info, provenance_file=provenance_file,
                                         **mip_kwargs)
    else:
        meta_name = storage.join_paths(kwargs['meta_dir'], sec_name + '.json')
        mask_file = storage.join_paths(kwargs.get('mask_dir', None), sec_name + '.png')
        res = mipmap.generate_tensorstore_scales(meta_name, [tgt_mip], num_workers=num_workers, format=ds_format,
                                                 mask_file=mask_file, logger=logger_info,
                                                 provenance_file=provenance_file)
    return res.get(sec_name, False)


def plan_section_mips(requested, rendered, level_class=None):
    """
    split the requested mip levels not rendered yet into the ones to render
    from the raw tiles, and the ones to downsample (as {mip: source mip}) from
    a finer level, either already rendered or rendered before. If level_class
    is given (mip -> class), only downsample from a level of the same class,
    e.g. keep the matching levels from being derived from the final output
    when they are rendered with different settings.
    """
    if level_class is None:
        level_class = lambda m: None
    available = set(rendered)
    to_render, to_downsample = [], {}
    for m in sorted(set(requested) - available):
        finer = [s for s in available if (s < m) and (level_class(s) == level_class(m))]
        if len(finer) > 0:
            to_downsample[m] = max(finer)
        else:
            to_render.append(m)
        available.add(m)
    return to_render, to_downsample


def render_main(tform_list, out_dir, **kwargs):
    logger_info = logging.initialize_main_logger(logger_name='stitch_rendering', mp=False)
    logger = logger_info[0]
    driver = kwargs.get('driver', 'image')
    histeq_dir = kwargs.pop('histeq_dir', None)
    provenance_dir = kwargs.pop('provenance_dir', None)
    mip_levels = kwargs.pop('mip_levels', [0])
    # settings of the final (full-resolution) output levels, if different
    final_kwargs = kwargs.pop('final_settings', None)
    use_tensorstore = driver != 'image'
    if use_tensorstore:
        meta_dir = kwargs['meta_dir']
        storage.makedirs(meta_dir)
    if final_kwargs is None:
        level_class = None
        settings_of = lambda m: kwargs
    else:
        level_class = config.is_final_stitch_mip
        settings_of = lambda m: final_kwargs if config.is_final_stitch_mip(m) else kwargs
        final_mips = [m for m in mip_levels if config.is_final_stitch_mip(m)]
        if len(final_mips) > 0:
            logger.info(f'final_rendering settings used for mip levels {final_mips}')
    logger.info(f'stitched mip levels to render: {mip_levels}')
    # what is already there, for all the sections at once (concurrent reads)
    sec_names = [os.path.basename(tname).replace('.h5', '') for tname in tform_list]
    if use_tensorstore:
        rendered_list = storage.map_io(partial(mipmap.stitched_mips, meta_dir=meta_dir), sec_names)
    else:
        rendered_list = storage.map_io(partial(mipmap.stitched_mips, img_dir=out_dir), sec_names)
    record_files = [storage.join_paths(provenance_dir, s + '.json') for s in sec_names]
    record_list = provenance.load_all(record_files)
    has_record = [any(record_list[f].get(str(m), None) is not None for m in r)
                  for f, r in zip(record_files, rendered_list)]
    storage.map_io(provenance.tform_fingerprint, [t for t, h in zip(tform_list, has_record) if h])  # cached
    for tname, sec_name, rendered, record_file in zip(tform_list, sec_names, rendered_list, record_files):
        records = record_list[record_file]
        recorded = [m for m in rendered if records.get(str(m), None) is not None]
        if len(recorded) > 0:
            fingerprint = provenance.tform_fingerprint(tname)
            stale = [m for m in recorded if records[str(m)] != fingerprint]
            if len(stale) > 0:
                logger.warning(f'{sec_name}: stitched mip{stale} rendered from a previous version of the '
                               'stitching transformation. Delete them to re-render.')
        to_render, to_downsample = plan_section_mips(mip_levels, rendered, level_class=level_class)
        if (len(to_render) == 0) and (len(to_downsample) == 0):
            continue
        tfname = storage.join_paths(histeq_dir, sec_name+'.json')
        if not storage.file_exists(tfname):
            tfname = None
        for mip in to_render:
            t0 = time.time()
            try:
                if use_tensorstore:
                    meta_name = storage.join_paths(meta_dir, sec_name+'.json')
                    out_prefix = storage.join_paths(out_dir, sec_name)
                else:
                    sec_outdir = storage.join_paths(out_dir, 'mip'+str(mip), sec_name)
                    meta_name = storage.join_paths(sec_outdir, 'metadata.txt')
                    storage.makedirs(sec_outdir)
                    out_prefix = storage.join_paths(sec_outdir, sec_name)
                logger.info(f'{sec_name}: start mip{mip}')
                num_rendered = render_one_section(tname, out_prefix, meta_name=meta_name, tfname=tfname, mip=mip, **settings_of(mip))
                if use_tensorstore:
                    completed = mip in mipmap.stitched_mips(sec_name, meta_dir=meta_dir)
                else:
                    completed = storage.file_exists(meta_name)
                if completed:
                    provenance.update(record_file, {mip: provenance.tform_fingerprint(tname)})
                logger.info(f'{sec_name}: mip{mip} {num_rendered} tiles | {(time.time()-t0)/60} min')
            except TimeoutError:
                logger.error(f'{sec_name}: Tensorstore timed out.')
            except Exception as err:
                logger.error(f'{sec_name}: {err}')
        for mip, src_mip in to_downsample.items():
            t0 = time.time()
            try:
                updated = downsample_one_section(sec_name, src_mip, mip, out_dir, logger=logger,
                                                 provenance_file=record_file, **settings_of(mip))
                if updated:
                    logger.info(f'{sec_name}: mip{mip} downsampled from mip{src_mip} | {(time.time()-t0)/60} min')
            except TimeoutError:
                logger.error(f'{sec_name}: Tensorstore timed out.')
            except Exception as err:
                logger.error(f'{sec_name}: {err}')
    logger.info('finished.')
    logging.terminate_logger(*logger_info)


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Run stitching")
    parser.add_argument("--mode", metavar="mode", type=str, default='opt')
    parser.add_argument("--filter", metavar="filter", type=str, default='')
    parser.add_argument("--start", metavar="start", type=int, default=0)
    parser.add_argument("--step", metavar="step", type=int, default=1)
    parser.add_argument("--stop", metavar="stop", type=int, default=0)
    parser.add_argument("--reverse",  action='store_true')
    parser.add_argument("--mips", metavar="mips", type=str, default=None,
                        help="rendering: comma-separated stitched mip levels to render, overriding "
                             "rendering: mip_levels in the config (default: the levels needed by "
                             "thumbnail generation and alignment matching)")
    parser.add_argument("--full", action='store_true', help="rendering: same as --mips 0")
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args()

    root_dir = config.get_work_dir()

    stitch_configs = config.stitch_configs()
    if args.mode.lower().startswith('r'):
        stitch_configs = config.stitch_render_configs(final=False)
        mode = 'rendering'
        image_outdir = config.stitch_render_dir()
        mip_override = '0' if args.full else args.mips
        mip_levels = config.stitch_render_mips(mip_override)
        if config.separate_final_stitch_render():
            final_settings = config.stitch_render_configs(final=True)
        else:
            final_settings = None
        for conf in (stitch_configs, final_settings):
            if conf is not None:
                conf.pop('out_dir', '')
                conf.pop('mip_levels', None)
        stitch_configs['mip_levels'] = mip_levels
        # Rendering allocates many tile-sized (>32 MB) temporaries. By default glibc mmaps and
        # munmaps each of them, so every allocation page-faults and gets zeroed by the kernel
        # (~18% sys time with 64 workers). Keep them on the heap instead; inherited by the
        # spawned workers. Values already set in the environment take precedence.
        os.environ.setdefault('MALLOC_MMAP_MAX_', '0')
        os.environ.setdefault('MALLOC_TRIM_THRESHOLD_', str(64 * 1024**3))
        os.environ.setdefault('MALLOC_TOP_PAD_', str(256 * 1024**2))
    elif args.mode.lower().startswith('o'):
        stitch_configs = stitch_configs['optimization']
        mode = 'optimization'
    elif args.mode.lower().startswith('m'):
        stitch_configs = stitch_configs['matching']
        mode = 'matching'
    else:
        raise ValueError(f'{args.mode} not supported mode.')
    num_workers = stitch_configs.get('num_workers', 1)
    config.set_numpy_thread_from_num_workers(num_workers)
    stitch_configs['num_workers'] = num_workers

    from feabas.stitcher import Stitcher, MontageRenderer
    from feabas import mipmap, provenance

    stitch_dir = storage.join_paths(root_dir, 'stitch')
    coord_dir = storage.join_paths(stitch_dir, 'stitch_coord')
    match_dir = storage.join_paths(stitch_dir, 'match_h5')
    mesh_dir = storage.join_paths(stitch_dir, 'tform')
    histeq_dir = storage.join_paths(stitch_dir, 'hist_tf')
    render_meta_dir = storage.join_paths(stitch_dir, 'ts_specs')
    render_mask_dir = storage.join_paths(render_meta_dir, 'masks')
    stt_idx, stp_idx, step = args.start, args.stop, args.step
    if stp_idx == 0:
        stp_idx = None
    indx = slice(stt_idx, stp_idx, step)

    if mode == 'rendering':
        tform_list = sorted(storage.list_folder_content(storage.join_paths(mesh_dir, '*.h5')))
        if len(args.filter) > 0:
            tform_list = [s for s in tform_list if args.filter in os.path.basename(s)]
        tform_list = tform_list[indx]
        if args.reverse:
            tform_list = tform_list[::-1]
        stitch_configs.setdefault('meta_dir', render_meta_dir)
        if final_settings is not None:
            final_settings.update({'meta_dir': stitch_configs['meta_dir'], 'mask_dir': render_mask_dir,
                                   'num_workers': final_settings.get('num_workers', num_workers)})
            stitch_configs['final_settings'] = final_settings
        render_main(tform_list, image_outdir, mask_dir=render_mask_dir, histeq_dir=histeq_dir,
                    provenance_dir=storage.join_paths(stitch_dir, 'provenance'), **stitch_configs)
    elif mode == 'optimization':
        match_list = sorted(storage.list_folder_content(storage.join_paths(match_dir, '*.h5')))
        if len(args.filter) > 0:
            match_list = [s for s in match_list if args.filter in os.path.basename(s)]
        match_list = match_list[indx]
        if args.reverse:
            match_list = match_list[::-1]
        storage.makedirs(mesh_dir)
        optmization_main(match_list, mesh_dir, **stitch_configs)
    else:
        coord_list = sorted(storage.list_folder_content(storage.join_paths(coord_dir, '*.txt')))
        if len(coord_list) == 0:
            coord_list = sorted(storage.list_folder_content(storage.join_paths(coord_dir, '*.tsv')))
        if len(args.filter) > 0:
            coord_list = [s for s in coord_list if args.filter in os.path.basename(s)]
        coord_list = coord_list[indx]
        if args.reverse:
            coord_list = coord_list[::-1]
        storage.makedirs(match_dir)
        match_main(coord_list, match_dir, **stitch_configs)
