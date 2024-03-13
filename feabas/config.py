import glob
import math
import os
import yaml
from feabas import constant
from functools import lru_cache
import statistics

if os.path.isfile(os.path.join(os.getcwd(), 'configs', 'general_configs.yaml')):
    _default_configuration_folder = os.path.join(os.getcwd(), 'configs')
elif os.path.isfile(os.path.join(os.path.dirname(os.getcwd()), 'configs', 'general_configs.yaml')):
    _default_configuration_folder = os.path.join(os.path.dirname(os.getcwd()), 'configs')
else:
    _default_configuration_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs')


@lru_cache(maxsize=1)
def general_settings():
    config_file = os.path.join(_default_configuration_folder, 'general_configs.yaml')
    if os.path.isfile(config_file):
        with open(config_file, 'r') as f:
            conf = yaml.safe_load(f)
    else:
        conf = {}
    if conf.get('cpu_budget', None) is None:
        import psutil
        conf['cpu_budget'] = psutil.cpu_count(logical=False)
    return conf


DEFAULT_RESOLUTION = general_settings().get('full_resolution', constant.DEFAULT_RESOLUTION)


@lru_cache(maxsize=1)
def get_work_dir():
    conf = general_settings()
    work_dir = conf.get('working_directory', './work_dir')
    return work_dir


@lru_cache(maxsize=1)
def get_log_dir():
    conf = general_settings()
    log_dir = conf.get('logging_directory', None)
    if log_dir is None:
        work_dir = conf.get('working_directory', './work_dir')
        log_dir = os.path.join(work_dir, 'logs')
    return log_dir


@lru_cache(maxsize=1)
def stitch_config_file():
    work_dir = get_work_dir()
    config_file = os.path.join(work_dir, 'configs', 'stitching_configs.yaml')
    if not os.path.isfile(config_file):
        config_file = os.path.join(_default_configuration_folder, 'default_stitching_configs.yaml')
        assert(os.path.isfile(config_file))
    return config_file


@lru_cache(maxsize=1)
def stitch_configs():
    with open(stitch_config_file(), 'r') as f:
        conf = yaml.safe_load(f)
    return conf


@lru_cache(maxsize=1)
def section_thickness():
    conf = stitch_configs()
    if conf.get('section_thickness', None) is not None:
        return conf['section_thickness']
    else:
        return general_settings().get('section_thickness', constant.DEFAULT_THICKNESS)


@lru_cache(maxsize=1)
def material_table_file():
    work_dir = get_work_dir()
    mt_file = os.path.join(work_dir, 'configs', 'material_table.json')
    if not os.path.isfile(mt_file):
        mt_file = os.path.join(_default_configuration_folder, 'default_material_table.json')
    return mt_file


@lru_cache(maxsize=1)
def align_config_file():
    work_dir = get_work_dir()
    config_file = os.path.join(work_dir, 'configs', 'alignment_configs.yaml')
    if not os.path.isfile(config_file):
        config_file = os.path.join(_default_configuration_folder, 'default_alignment_configs.yaml')
        assert(os.path.isfile(config_file))
    return config_file


@lru_cache(maxsize=1)
def align_configs():
    with open(align_config_file(), 'r') as f:
        conf = yaml.safe_load(f)
    if (SECTION_THICKNESS is not None) and (conf.get('matching', {}).get('working_mip_level', None) is None):
        align_mip = max(0, math.floor(math.log2(SECTION_THICKNESS / montage_resolution())))
        conf.setdefault('matching', {})
        conf['matching'].setdefault('working_mip_level', align_mip)
    return conf


@lru_cache(maxsize=1)
def thumbnail_config_file():
    work_dir = get_work_dir()
    config_file = os.path.join(work_dir, 'configs', 'thumbnail_configs.yaml')
    if not os.path.isfile(config_file):
        config_file = os.path.join(_default_configuration_folder, 'default_thumbnail_configs.yaml')
        assert(os.path.isfile(config_file))
    return config_file


@lru_cache(maxsize=1)
def thumbnail_configs():
    with open(thumbnail_config_file(), 'r') as f:
        conf = yaml.safe_load(f)
    return conf


@lru_cache(maxsize=1)
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


@lru_cache(maxsize=1)
def align_render_dir():
    config_file = align_config_file()
    with open(config_file, 'r') as f:        
        align_configs = yaml.safe_load(f)
    render_settings = align_configs.get('rendering', {})
    outdir = render_settings.get('out_dir', None)
    if outdir is None:
        work_dir = get_work_dir()
        outdir = os.path.join(work_dir, 'aligned_stack')
    return outdir


@lru_cache(maxsize=1)
def tensorstore_render_dir():
    config_file = align_config_file()
    with open(config_file, 'r') as f:        
        align_configs = yaml.safe_load(f)
    render_settings = align_configs.get('tensorstore_rendering', {})
    outdir = render_settings.get('out_dir', None)
    if outdir is None:
        work_dir = get_work_dir()
        outdir = os.path.join(work_dir, 'aligned_tensorstore')
    outdir = outdir.replace('\\', '/')
    if not outdir.endswith('/'):
        outdir = outdir + '/'
    kv_headers = ('gs://', 'http://', 'https://', 'file://', 'memory://', 's3://')
    for kvh in kv_headers:
        if outdir.startswith(kvh):
            break
    else:
        outdir = 'file://' + outdir
    return outdir


lru_cache(maxsize=1)
def data_resolution():
    """
    raw data resolution.
    First check the cached value in {WORKDIR}/configs/resolutions.yaml
    Then check stitch coordinate files. If nothing there, use DEFAULT_RESOLUTION.
    If conflicts in coordinate files, use the mode.
    """
    work_dir = get_work_dir()
    cache_file = os.path.join(work_dir, 'configs', 'resolutions.yaml')
    res = {}
    if os.path.isfile(cache_file):
        with open(cache_file, 'r') as f:
            res = yaml.safe_load(f)
        if 'DATA_RESOLUTION' in res:
            return res['DATA_RESOLUTION']
    # try parse_coordinate_files
    NUM_SAMPLE_FILES = 5
    DELMITER = '\t'
    coord_dir = os.path.join(work_dir, 'stitch', 'stitch_coord')
    coord_list = sorted(glob.glob(os.path.join(coord_dir, '*.txt')))
    coord_list = coord_list[:NUM_SAMPLE_FILES]
    resolution_samples = []
    for coord_file in coord_list:
        try:
            with open(coord_file, 'r') as f:
                for line in f:
                    if '{RESOLUTION}' in line:
                        tlist = line.strip().split(DELMITER)
                        if len(tlist) >= 2:
                            resolution_samples.append(float(tlist[1]))
                        break
                    elif line.startswith('{'):
                        continue
                    else:
                        break
        except Exception:
            pass
    if len(resolution_samples) == 0:
        dt_res = DEFAULT_RESOLUTION
    else:
        dt_res = statistics.mode(resolution_samples)
    # if coordinate list exists, cache data resolution
    if len(coord_list) > 0:
        res.update({'DATA_RESOLUTION': dt_res})
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'w') as f:
            yaml.dump(res, f)
    return dt_res


lru_cache(maxsize=1)
def montage_resolution():
    """
    highest resolution of rendered montage. This will be defined as mip0
    Check stitch config files first, then check the cached value from {WORKDIR}/configs
    Otherwise use data resolution.
    """
    conf = stitch_configs().get('rendering', {})
    mt_res = conf.get('resolution', None)
    if mt_res is None:
        scale = conf.get('scale', 1.0)
        mt_res = data_resolution() / scale
    return mt_res


SECTION_THICKNESS = section_thickness()

def limit_numpy_thread(nthreads):
    nthread_str = str(nthreads)
    os.environ["OMP_NUM_THREADS"] = nthread_str
    os.environ["OPENBLAS_NUM_THREADS"] = nthread_str
    os.environ["MKL_NUM_THREADS"] = nthread_str
    os.environ["VECLIB_MAXIMUM_THREADS"] = nthread_str
    os.environ["NUMEXPR_NUM_THREADS"] = nthread_str