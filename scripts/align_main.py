import argparse
import glob
import os
import time
import yaml
import gc

from feabas.aligner import match_section_from_initial_matches, Stack
import feabas.constant as const


def match_main(match_dir, mesh_dir, loader_dir, out_dir, conf_dir, stt=0, step=1, stop=None):
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
    resolution = conf.get('resolution', 4.0)
    slide_window = conf.get('slide_window', {})
    stk = Stack(section_list=section_list, mesh_dir=mesh_dir, match_dir=match_dir, mesh_out_dir=mesh_out_dir, resolution=resolution)
    cost = stk.optimize_slide_window(optimize_rigid=True, optimize_elastic=True,
        target_gear=const.MESH_GEAR_MOVING, **slide_window)
    with open(os.path.join(mesh_out_dir, 'cost.txt'), 'w') as f:
        for key, val in cost.items():
            f.write(f'{key}: {val}\n')
    print('finished')


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Run stitching")
    parser.add_argument("--start", metavar="start", type=int, default=0)
    parser.add_argument("--step", metavar="step", type=int, default=1)
    parser.add_argument("--stop", metavar="stop", type=int, default=-1)
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args()
    root_dir = '/n/boslfs02/LABS/lichtman_lab/yuelong/dce/data/Fish2/alignment'
    mode = 'optimize'
    conf_files = os.path.join('configs', 'alignment_configs.yaml')
    with open(conf_files, 'r') as f:
        conf = yaml.safe_load(f)
    if mode == 'matching':
        conf_matching = conf['matching']
        init_matchdir = os.path.join(root_dir, 'initial_matches')
        match_meshdir = os.path.join(root_dir, 'matching_meshes')
        match_loaderdir = os.path.join(root_dir, 'image_loaders_mip2')
        out_dir = os.path.join(root_dir, 'match_h5')
        match_main(init_matchdir, match_meshdir, match_loaderdir, out_dir, conf, stt=args.start, step=args.step, stop=args.stop)
    elif mode == 'optimize':
        mesh_dir = os.path.join(root_dir, 'opt_meshes')
        mesh_out_dir = os.path.join(root_dir, 'tforms')
        match_dir = os.path.join(root_dir, 'match_h5')
        conf_optm = conf['optimization']
        slist = sorted(glob.glob(os.path.join(mesh_dir, '*.h5')))
        slist = [os.path.basename(s).replace('.h5', '') for s in slist]
        os.makedirs(mesh_out_dir, exist_ok=True)
        optimize_main(slist, match_dir, mesh_dir, mesh_out_dir, **conf_optm)
