import argparse
import glob
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures import as_completed
from multiprocessing import get_context
import numpy as np
from functools import partial
import os
import time
import yaml

import feabas
from feabas.stitcher import Stitcher, MontageRenderer


def match_one_section(coordname, outname, **kwargs):
    num_workers = kwargs.get('num_workers', 1)
    min_width = kwargs.get('min_overlap_width', 25)
    margin = kwargs.get('margin', 200)
    loader_config = kwargs.get('loader_config', {})
    matcher_config = kwargs.get('matcher_config', {})
    stitcher = Stitcher.from_coordinate_file(coordname)
    if os.path.isfile(outname + '_err'):
        print(f'loading previous results for {os.path.basename(coordname)}')
        stitcher.load_matches_from_h5(outname + '_err', check_order=True)
    _, err = stitcher.dispatch_matchers(num_workers=num_workers, min_width=min_width,
        margin=margin, matcher_config=matcher_config, loader_config=loader_config,
        verbose=False)
    if err:
        outname = outname + '_err'
    stitcher.save_to_h5(outname, save_matches=True, save_meshes=False)
    return 1


def match_main(coord_dir, out_dir, stt=0, step=1, stop=None, **conf):
    if stop == 0:
        stop = None
    coord_list = sorted(glob.glob(os.path.join(coord_dir, '*.txt')))
    coord_list = coord_list[slice(stt, stop, step)]
    for coordname in coord_list:
        t0 = time.time()
        fname = os.path.basename(coordname).replace('.txt', '')
        outname = os.path.join(out_dir, fname + '.h5')
        if os.path.isfile(outname):
            continue
        print(f'starting matching for {fname}')
        flag = match_one_section(coordname, outname, **conf)
        if flag == 1:
            print(f'ending for {fname}: {(time.time()-t0)/60} min')
    print('finished.')


def optimize_one_section(matchname, outname, **kwargs):
    use_group = kwargs.get('use_group', True)
    msem = kwargs.get('msem', False)
    mesh_settings = kwargs.get('mesh_settings', {})
    translation_settings = kwargs.get('translation', {})
    group_elastic_settings = kwargs.get('group_elastic', {})
    elastic_settings = kwargs.get('final_elastic', {})
    normalize_setting = kwargs.get('normalize', {})
    bname = os.path.basename(matchname).replace('.h5', '')
    t0 = time.time()
    stitcher = Stitcher.from_h5(matchname, load_matches=True, load_meshes=False)
    if use_group:
        if msem:
            groupings = [int(s.split('/')[0]) for s in stitcher.imgrelpaths]
        else:
            groupings = np.zeros(stitcher.num_tiles, dtype=np.int32)
    else:
        groupings = None
    stitcher.set_groupings(groupings)
    mesh_sizes = mesh_settings.pop('mesh_sizes', [75, 150, 300])
    stitcher.initialize_meshes(mesh_sizes, **mesh_settings)
    discrd =stitcher.optimize_translation(target_gear=feabas.MESH_GEAR_FIXED, **translation_settings)
    dis = stitcher.match_residues()
    print(f'\t{bname}: residue after translation {np.mean(dis)} | discarded {discrd}')
    if use_group:
        stitcher.optimize_group_intersection(target_gear=feabas.MESH_GEAR_FIXED, **group_elastic_settings)
        stitcher.optimize_translation(target_gear=feabas.MESH_GEAR_FIXED, **translation_settings)
        cost = stitcher.optimize_elastic(use_groupings=True, target_gear=feabas.MESH_GEAR_FIXED, **group_elastic_settings)
        dis = stitcher.match_residues()
        print(f'\t{bname}: residue after grouped relaxation {np.mean(dis)} | cost {cost}')
    cost = stitcher.optimize_elastic(target_gear=feabas.MESH_GEAR_MOVING, **elastic_settings)
    rot, _ = stitcher.normalize_coordinates(**normalize_setting)
    N_conn = stitcher.connect_isolated_subsystem()
    if N_conn > 1:
        rot_1, _ = stitcher.normalize_coordinates(**normalize_setting)
        rot = max(rot, rot_1)
    if cost[0] < cost[1]:
        stitcher.save_to_h5(outname.replace('.h5', '.h5_err'), save_matches=False, save_meshes=True)
    else:
        stitcher.save_to_h5(outname, save_matches=False, save_meshes=True)
    lbl_conn = stitcher.connected_subsystem
    _, cnt = np.unique(lbl_conn, return_counts=True)
    ncomp = cnt.size
    ncomp1 = np.sum(cnt>1)
    dis = stitcher.match_residues()
    finish_str = f'\t{bname}: {cost} finished {time.time() - t0} sec | residue: {np.mean(dis)} | rotation: {round(rot)}'
    if ncomp > 1:
        finish_str = finish_str + f' | {ncomp1}/{ncomp} components'
    print(finish_str)


def optmization_main(match_dir, out_dir, stt=0, step=1, stop=None, **conf):
    num_workers = conf.pop('num_workers', 1)
    if stop == 0:
        stop = None
    match_list = sorted(glob.glob(os.path.join(match_dir, '*.h5')))
    match_list = match_list[slice(stt, stop, step)]
    target_func = partial(optimize_one_section, **conf)
    if num_workers <= 1:
        for matchname in match_list:
            outname = os.path.join(out_dir, os.path.basename(matchname))
            if os.path.isfile(outname):
                continue
            target_func(matchname, outname)
    else:
        jobs = []
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=get_context('spawn')) as executor:
            for matchname in match_list:
                outname = os.path.join(out_dir, os.path.basename(matchname))
                if os.path.isfile(outname):
                    continue
                job = executor.submit(target_func, matchname, outname)
                jobs.append(job)
            for job in jobs:
                job.result()
    print('finished.')


def render_one_section(tform_name, out_prefix, meta_name=None, **kwargs):
    num_workers = kwargs.get('num_workers', 1)
    tile_size = kwargs.get('tile_size', [4096, 4096])
    scale = kwargs.get('scale', 1.0)
    loader_settings = kwargs.get('loader_settings', {})
    render_settings = kwargs.get('render_settings', {})
    filename_settings = kwargs.get('filename_settings', {})
    if loader_settings.get('cache_size', None) is not None:
        loader_settings['cache_size'] = loader_settings['cache_size'] // num_workers
    render_settings['scale'] = scale
    if meta_name is not None and os.path.isfile(meta_name):
        return None
    renderer = MontageRenderer.from_h5(tform_name, loader_settings=loader_settings)
    render_series = renderer.plan_render_series(tile_size, prefix=out_prefix,
        scale=scale, **filename_settings)
    if num_workers <= 1:
        bboxes, filenames, _ = render_series
        metadata = renderer.render_series_to_file(bboxes, filenames, **render_settings)
    else:
        bboxes_list, filenames_list, hits_list = renderer.divide_render_jobs(render_series,
            num_workers=num_workers, max_tile_per_job=20)
        metadata = {}
        jobs = []
        target_func = partial(MontageRenderer.subprocess_render_montages, **render_settings)
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=get_context('spawn')) as executor:
            for bboxes, filenames, hits in zip(bboxes_list, filenames_list, hits_list):
                init_args = renderer.init_args(selected=hits)
                job = executor.submit(target_func, init_args, bboxes, filenames)
                jobs.append(job)
            for job in as_completed(jobs):
                metadata.update(job.result())
    if meta_name is not None:
        with open(meta_name, 'w') as f:
            root_dir = os.path.dirname(out_prefix)
            f.write(f'{{ROOT_DIR}}\t{root_dir}\n')
            fnames = sorted(list(metadata.keys()))
            for fname in fnames:
                bbox = metadata[fname]
                relpath = os.path.relpath(fname, root_dir)
                f.write(f'{relpath}\t{bbox[0]}\t{bbox[1]}\t{bbox[2]}\t{bbox[3]}\n')
    return len(metadata)


def render_main(tform_dir, out_dir, meta_dir, stt=0, step=1, stop=None, **conf):
    tform_list = sorted(glob.glob(os.path.join(tform_dir, '*.h5')))
    if stop == 0:
        stop = None
    tform_list = tform_list[slice(stt, stop, step)]
    os.makedirs(meta_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    for tname in tform_list:
        t0 = time.time()
        sec_name = os.path.basename(tname).replace('.h5', '')
        sec_outdir = os.path.join(out_dir, sec_name)
        os.makedirs(sec_outdir, exist_ok=True)
        out_prefix = os.path.join(sec_outdir, sec_name)
        meta_name = os.path.join(meta_dir, sec_name+'.txt')
        num_rendered = render_one_section(tname, out_prefix, meta_name=meta_name, **conf)
        print(f'{sec_name}: {num_rendered} tiles | {(time.time()-t0)/60} min')
    print('finished.')


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Run stitching")
    parser.add_argument("--start", metavar="start", type=int, default=0)
    parser.add_argument("--step", metavar="step", type=int, default=1)
    parser.add_argument("--stop", metavar="stop", type=int, default=0)
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args()
    # root_dir = '/n/boslfs02/LABS/lichtman_lab/yuelong/dce/data/Fish2/stitch'
    root_dir = 'debug/test_stitcher'
    coord_dir = os.path.join(root_dir, 'stitch_coord')
    match_dir = os.path.join(root_dir, 'match_h5')
    tform_dir = os.path.join(root_dir, 'tform')
    meta_dir = os.path.join(root_dir, 'rendered_metadata')
    image_outdir = '/n/boslfs02/LABS/lichtman_lab/Lab/STITCHED/Fish2_0422_03'
    conf_files = os.path.join('configs', 'stitching_configs.yaml')
    mode = 'optimization'

    with open(conf_files, 'r') as f:
        conf = yaml.safe_load(f)

    if mode == 'matching':
        conf_matching = conf['matching']
        os.makedirs(match_dir, exist_ok=True)
        match_main(coord_dir, match_dir, stt=args.start, step=args.step, stop=args.stop, **conf_matching)
    elif mode == 'optimization':
        conf_opt = conf['optimization']
        os.makedirs(tform_dir, exist_ok=True)
        optmization_main(match_dir, tform_dir, stt=args.start, step=args.step, stop=args.stop, **conf_opt)
    elif mode == 'render':
        conf_render = conf['render']
        os.makedirs(image_outdir, exist_ok=True)
        os.makedirs(meta_dir, exist_ok=True)
        render_main(tform_dir, image_outdir, meta_dir, stt=args.start, step=args.step, stop=args.stop, **conf_render)