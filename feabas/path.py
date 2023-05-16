import os
import yaml

def get_work_dir():
    config_file = os.path.join('configs', 'general_configs.yaml')
    if os.path.isfile(config_file):
        with open(config_file, 'r') as f:
            conf = yaml.safe_load(f)
    else:
        conf = {}
    work_dir = conf.get('working_directory', './work_dir')
    return work_dir


def get_log_dir():
    config_file = os.path.join('configs', 'general_configs.yaml')
    if os.path.isfile(config_file):
        with open(config_file, 'r') as f:
            conf = yaml.safe_load(f)
    else:
        conf = {}
    log_dir = conf.get('logging_directory', None)
    if log_dir is None:
        work_dir = conf.get('working_directory', './work_dir')
        log_dir = os.path.join(work_dir, 'logs')
    return log_dir


def stitch_config_file():
    work_dir = get_work_dir()
    config_file = os.path.join(work_dir, 'configs', 'stitching_configs.yaml')
    if not os.path.isfile(config_file):
        config_file = os.path.join('configs', 'default_stitching_configs.yaml')
        assert(os.path.isfile(config_file))
    return config_file


def align_config_file():
    work_dir = get_work_dir()
    config_file = os.path.join(work_dir, 'configs', 'alignment_configs.yaml')
    if not os.path.isfile(config_file):
        config_file = os.path.join('configs', 'default_alignment_configs.yaml')
        assert(os.path.isfile(config_file))
    return config_file


def thumbnail_config_file():
    work_dir = get_work_dir()
    config_file = os.path.join(work_dir, 'configs', 'thumbnail_configs.yaml')
    if not os.path.isfile(config_file):
        config_file = os.path.join('configs', 'default_thumbnail_configs.yaml')
        assert(os.path.isfile(config_file))
    return config_file


def stitch_render_dir():
    config_file = stitch_config_file()
    with open(config_file, 'r') as f:        
        stitch_configs = yaml.safe_load(f)
    render_settings = stitch_configs.get('rendering', {})
    outdir = render_settings.get('out_dir', None)
    if outdir is None:
        work_dir = get_work_dir()
        outdir = os.path.join(work_dir, 'stitched_sections')
    return outdir


def align_render_dir():
    config_file = align_config_file()
    with open(config_file, 'r') as f:        
        stitch_configs = yaml.safe_load(f)
    render_settings = stitch_configs.get('rendering', {})
    outdir = render_settings.get('out_dir', None)
    if outdir is None:
        work_dir = get_work_dir()
        outdir = os.path.join(work_dir, 'aligned_stack')
    return outdir
