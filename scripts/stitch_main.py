import argparse
import glob
import os
import time
import yaml

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
    if stop == -1:
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


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Run stitching")
    parser.add_argument("--start", metavar="start", type=int, default=0)
    parser.add_argument("--step", metavar="step", type=int, default=1)
    parser.add_argument("--stop", metavar="step", type=int, default=-1)
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args()
    root_dir = '/n/boslfs02/LABS/lichtman_lab/yuelong/dce/data/Fish2/stitch'
    coord_dir = os.path.join(root_dir, 'stitch_coord')
    match_dir = os.path.join(root_dir, 'match_h5')

    ############################# matching #####################################
    conf_files = os.path.join('configs', 'example_configs.yaml')
    with open(conf_files, 'r') as f:
        conf = yaml.safe_load(f)
    conf_matching = conf['stitching']['matching']
    os.makedirs(match_dir, exist_ok=True)
    match_main(coord_dir, match_dir, stt=args.start, step=args.step, stop=args.stop, **conf_matching)
