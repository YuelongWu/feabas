import argparse
from functools import partial
import os
import time

import feabas
from feabas.concurrent import submit_to_workers
from feabas import config, logging, storage


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
        fname = os.path.basename(coordname).replace('.txt', '')
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
    group_elastic_settings.setdefault('continue_on_flip', True)
    elastic_settings.setdefault('continue_on_flip', True)
    bname = os.path.basename(matchname).replace('.h5', '')
    t0 = time.time()
    stitcher = Stitcher.from_h5(matchname, load_matches=True, load_meshes=False)
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
    mesh_settings = mesh_settings.copy()
    mesh_sizes = mesh_settings.pop('mesh_sizes', [100, 300])
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
    if abs(rot) > 1.5:
        logger.warning(f'{bname}: rotation detected in final transform, potential mesh relaxation issues.')
    if ncomp > 1:
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
    num_workers = kwargs.get('num_workers', 1)
    loader_settings = kwargs.get('loader_settings', {})
    if loader_settings.get('cache_size', None) is not None:
        loader_settings = loader_settings.copy()
        loader_settings['cache_size'] = loader_settings['cache_size'] // num_workers
    renderer = MontageRenderer.from_h5(tform_name, loader_settings=loader_settings)
    mask_out = storage.join_paths(mask_dir, os.path.basename(tform_name).replace('.h5','.png'))
    num_chunks = renderer.render_one_section(out_prefix, meta_name=meta_name, mask_out=mask_out, **kwargs)
    return num_chunks


def render_main(tform_list, out_dir, **kwargs):
    logger_info = logging.initialize_main_logger(logger_name='stitch_rendering', mp=False)
    logger = logger_info[0]
    driver = kwargs.get('driver', 'image')
    use_tensorstore = driver != 'image'
    if use_tensorstore:
        meta_dir = kwargs['meta_dir']
        storage.makedirs(meta_dir)
    for tname in tform_list:
        t0 = time.time()
        sec_name = os.path.basename(tname).replace('.h5', '')
        try:
            sec_outdir = storage.join_paths(out_dir, sec_name)
            if use_tensorstore:
                meta_name = storage.join_paths(meta_dir, sec_name+'.json')
            else:
                meta_name = storage.join_paths(sec_outdir, 'metadata.txt')
            if storage.file_exists(meta_name, use_cache=True):
                continue
            logger.info(f'{sec_name}: start')
            if use_tensorstore:
                out_prefix = sec_outdir
            else:
                storage.makedirs(sec_outdir)
                out_prefix = storage.join_paths(sec_outdir, sec_name)
            num_rendered = render_one_section(tname, out_prefix, meta_name=meta_name, **kwargs)
            logger.info(f'{sec_name}: {num_rendered} tiles | {(time.time()-t0)/60} min')
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
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args()

    root_dir = config.get_work_dir()

    stitch_configs = config.stitch_configs()
    if args.mode.lower().startswith('r'):
        stitch_configs = stitch_configs['rendering']
        stitch_configs.pop('out_dir', '')
        mode = 'rendering'
        image_outdir = config.stitch_render_dir()
        driver = stitch_configs.get('driver', 'image')
        if driver == 'image':
            image_outdir = storage.join_paths(image_outdir, 'mip0')
    elif args.mode.lower().startswith('o'):
        stitch_configs = stitch_configs['optimization']
        mode = 'optimization'
    elif args.mode.lower().startswith('m'):
        stitch_configs = stitch_configs['matching']
        mode = 'matching'
    else:
        raise ValueError(f'{args.mode} not supported mode.')
    num_workers = stitch_configs.get('num_workers', 1)
    num_workers = config.set_numpy_thread_from_num_workers(num_workers)
    stitch_configs['num_workers'] = num_workers

    from feabas.stitcher import Stitcher, MontageRenderer

    stitch_dir = storage.join_paths(root_dir, 'stitch')
    coord_dir = storage.join_paths(stitch_dir, 'stitch_coord')
    match_dir = storage.join_paths(stitch_dir, 'match_h5')
    mesh_dir = storage.join_paths(stitch_dir, 'tform')
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
        render_main(tform_list, image_outdir, mask_dir=render_mask_dir, **stitch_configs)
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
        if len(args.filter) > 0:
            coord_list = [s for s in coord_list if args.filter in os.path.basename(s)]
        coord_list = coord_list[indx]
        if args.reverse:
            coord_list = coord_list[::-1]
        storage.makedirs(match_dir)
        match_main(coord_list, match_dir, **stitch_configs)
