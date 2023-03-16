import argparse
import glob
import numpy as np
import os
import time
import yaml
import gc

from feabas.aligner import match_section_from_initial_matches, Stack
import feabas.constant as const
from feabas.dal import get_loader_from_json
from feabas.mesh import Mesh
from feabas.renderer import render_whole_mesh


def match_main(match_dir, mesh_dir, loader_dir, out_dir, conf_dir, stt=0, step=1, stop=None, reverse=False):
    match_list = sorted(glob.glob(os.path.join(match_dir, '*.h5')))
    if stop == -1:
        stop = None
    match_list = match_list[slice(stt, stop, step)]
    if len(match_list) == 0:
        return
    for mname in match_list:
        t0 = time.time()
        tname = os.path.basename(mname).replace('.h5', '')
        print(f'start {tname}')
        num_matches = match_section_from_initial_matches(mname, mesh_dir, loader_dir, out_dir, conf=conf_dir)
        if num_matches is not None:
            print(f'{tname}: {num_matches} matches, {round((time.time()-t0)/60,3)} min.')
        gc.collect()
    print('finished.')


def optimize_main(section_list, match_dir, mesh_dir, mesh_out_dir, **conf):
    stack_config = conf.get('stack_config', {})
    slide_window = conf.get('slide_window', {})
    stk = Stack(section_list=section_list, mesh_dir=mesh_dir, match_dir=match_dir, mesh_out_dir=mesh_out_dir, **stack_config)
    stk.update_lock_flags({s: os.path.isfile(os.path.join(mesh_out_dir, s + '.h5')) for s in section_list})  ##############dd###############
    locked_flags = stk.locked_array
    print(f'{locked_flags.size} images| {np.sum(locked_flags)} references')
    cost = stk.optimize_slide_window(optimize_rigid=True, optimize_elastic=True,
        target_gear=const.MESH_GEAR_MOVING, **slide_window)
    # cost = {}
    # for s in stk.section_list:
    #     stk.get_mesh(s)
    # for m in stk.match_list:
    #     links = stk.get_link(m)
    #     dxy = np.concatenate([lnk.dxy(gear=1) for lnk in links], axis=0)
    #     dis = np.sum(dxy ** 2, axis=1)**0.5
    #     cost[m] = (dis.max(), dis.mean())
    with open(os.path.join(mesh_out_dir, 'cost.csv'), 'w') as f:
        for key, val in cost.items():
            f.write(f'{key}, {val[0]}, {val[1]}\n')
    print('finished')


def render_one_section(h5name, loadername, outdir, meta_name=None,  **conf):
    if meta_name is not None and os.path.isfile(meta_name):
        return None
    t0 = time.time()
    loader = get_loader_from_json(loadername, **conf)
    M = Mesh.from_h5(h5name)
    secname = os.path.splitext(os.path.basename(h5name))[0]
    prefix = os.path.join(outdir, secname)
    os.makedirs(prefix, exist_ok=True)
    prefix = os.path.join(prefix, secname)
    rendered = render_whole_mesh(M, loader, prefix, **conf)
    if meta_name is not None:
        with open(meta_name, 'w') as f:
            root_dir = os.path.dirname(prefix)
            f.write(f'{{ROOT_DIR}}\t{root_dir}\n')
            fnames = sorted(list(rendered.keys()))
            for fname in fnames:
                bbox = rendered[fname]
                relpath = os.path.relpath(fname, root_dir)
                f.write(f'{relpath}\t{bbox[0]}\t{bbox[1]}\t{bbox[2]}\t{bbox[3]}\n')
    print(f'{secname}: {len(rendered)} tiles | {time.time()-t0} secs.')
    return len(rendered)


def render_main(mesh_dir, loader_dir, out_dir, out_meta_dir=None, stt=0, step=1, stop=None, reverse=False, **conf):
    num_workers = conf.get('num_workers', 1)
    cache_size = conf.get('cache_size', 0)
    if cache_size is not None:
        conf['cache_size'] = cache_size // num_workers
    h5list = sorted(glob.glob(os.path.join(mesh_dir, '*.h5')))
    if stop == -1:
        stop = None
    h5list = h5list[slice(stt, stop, step)]
    if reverse:
        h5list = h5list[::-1]
    loaderlist = [os.path.join(loader_dir, os.path.basename(s).replace('.h5', '.json')) for s in h5list]
    if out_meta_dir is None:
        metalist = [None]*len(h5list)
    else:
        metalist = [os.path.join(out_meta_dir, os.path.basename(s).replace('.h5', '.txt')) for s in h5list]
    for h5name, loadername, metaname in zip(h5list, loaderlist, metalist):
        render_one_section(h5name, loadername, out_dir, meta_name=metaname, **conf)


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Run stitching")
    parser.add_argument("--start", metavar="start", type=int, default=0)
    parser.add_argument("--step", metavar="step", type=int, default=1)
    parser.add_argument("--stop", metavar="stop", type=int, default=-1)
    parser.add_argument("--reverse",  action='store_true')
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args()
    root_dir = 'F:/Fish2/alignment/test'
    mode = 'matching'
    conf_files = os.path.join('configs', 'alignment_configs.yaml')
    with open(conf_files, 'r') as f:
        conf = yaml.safe_load(f)
    if mode == 'matching':
        conf_matching = conf['matching']
        init_matchdir = os.path.join(root_dir, 'initial_matches')
        match_meshdir = os.path.join(root_dir, 'matching_meshes')
        match_loaderdir = os.path.join(root_dir, 'image_loaders_mip2')
        out_dir = os.path.join(root_dir, 'match_h5')
        match_main(init_matchdir, match_meshdir, match_loaderdir, out_dir, conf, stt=args.start, step=args.step, stop=args.stop, reverse=args.reverse)
    elif mode == 'optimize':
        mesh_dir = os.path.join(root_dir, 'opt_meshes')
        mesh_out_dir = os.path.join(root_dir, 'tforms')
        match_dir = os.path.join(root_dir, 'match_h5')
        conf_optm = conf['optimization']
        slist = sorted(glob.glob(os.path.join(mesh_dir, '*.h5')))
        slist = [os.path.basename(s).replace('.h5', '') for s in slist]
        os.makedirs(mesh_out_dir, exist_ok=True)
        optimize_main(slist, match_dir, mesh_dir, mesh_out_dir, **conf_optm)
    elif mode == 'render':
        conf_render = conf['render']
        render_root_dir = os.path.join(os.path.dirname(root_dir), 'render')
        mesh_dir = os.path.join(render_root_dir, 'tforms')
        loader_dir = os.path.join(render_root_dir, 'image_loaders')
        meta_dir = os.path.join(render_root_dir, 'metadata')
        out_dir = '/n/boslfs02/LABS/lichtman_lab/Lab/ALIGNED_STACKS/Fish2_0422_03/mip0'
        render_main(mesh_dir, loader_dir, out_dir, meta_dir,  stt=args.start, step=args.step, stop=args.stop, reverse=args.reverse, **conf_render)
