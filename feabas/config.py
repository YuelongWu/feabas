import math
import os
import yaml
from feabas import constant
from feabas import storage
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
    conf = storage.load_yaml(config_file)
    if conf.get('cpu_budget', None) is None:
        import psutil
        conf['cpu_budget'] = psutil.cpu_count(logical=False)
    return conf


DEFAULT_RESOLUTION = general_settings().get('full_resolution', constant.DEFAULT_RESOLUTION)
TS_TIMEOUT = general_settings().get('tensorstore_timeout', None)
TS_RETRY = 2
CHECKPOINT_TIME_INTERVAL = 300 # is seconds
OPT_CHECK_CONVERGENCE = True
DEFAULT_DEFORM_BUDGET = 0.125
MAXIMUM_DEFORM_ALLOWED = 0.5
MATCH_SOFTFACTOR_DOMINANCE = 200 # during matching, assume one mesh is much more rigid than the other so the system will not collapse


@lru_cache(maxsize=1)
def parallel_framework():
    frmwk = general_settings().get('parallel_framework', 'builtin')
    if frmwk.startswith('pr'):
        frmwk = 'process'
    elif frmwk.startswith('th'):
        frmwk = 'thread'
    elif frmwk.startswith('da'):
        frmwk = 'dask'
    else:
        raise ValueError(f'In {_default_configuration_folder}: unsupported parallel framework "{frmwk}"')
    return frmwk


@lru_cache(maxsize=1)
def get_work_dir():
    conf = general_settings()
    work_dir = conf.get('working_directory', './work_dir')
    work_dir = storage.expand_dir(work_dir)
    return work_dir


@lru_cache(maxsize=1)
def get_log_dir():
    conf = general_settings()
    log_dir = conf.get('logging_directory', None)
    if log_dir is None:
        work_dir = get_work_dir()
        log_dir = storage.join_paths(work_dir, 'logs')
    return log_dir


def merge_config(default_config, additional_config):
    for k, v in additional_config.items():
        if isinstance(v, dict) and (k in default_config):
            merge_config(default_config[k], v)
        else:
            default_config[k] = v


def load_yaml_configs(file_default, file_user=None):
    if storage.file_exists(file_default):
        conf = storage.load_yaml(file_default)
    else:
        conf = {}
    if (file_user is not None) and storage.file_exists(file_user):
        conf_usr = storage.load_yaml(file_user)
        merge_config(conf, conf_usr)
    return conf


@lru_cache(maxsize=1)
def stitch_config_file():
    work_dir = get_work_dir()
    config_file_default = storage.join_paths(_default_configuration_folder, 'default_stitching_configs.yaml')
    config_file_user = storage.join_paths(work_dir, 'configs', 'stitching_configs.yaml')
    return config_file_default, config_file_user


@lru_cache(maxsize=1)
def stitch_configs():
    conf = load_yaml_configs(*stitch_config_file())
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
    mt_file_default = storage.join_paths(_default_configuration_folder, 'default_material_table.yaml')
    mt_file_user = storage.join_paths(work_dir, 'configs', 'material_table.yaml')
    if not storage.file_exists(mt_file_default):
        mt_file_default = None
    if not storage.file_exists(mt_file_user):
        mt_file_user = None
    return mt_file_default, mt_file_user


@lru_cache(maxsize=1)
def material_table():
    from feabas.material import MaterialTable
    mt_file_default, mt_file_user = material_table_file()
    if (mt_file_default is None) and (mt_file_user is None):
        mt = MaterialTable()
    elif mt_file_user is None:
        mt = MaterialTable.from_pickleable(mt_file_default)
    else:
        mt = MaterialTable.from_pickleable(mt_file_user)
        if mt_file_default is not None:
            mt0 = MaterialTable.from_pickleable(mt_file_default)
            mt.combine_material_table(mt0, force_update=False, check_label=True)
    return mt


@lru_cache(maxsize=1)
def align_config_file():
    work_dir = get_work_dir()
    config_file_default = storage.join_paths(_default_configuration_folder, 'default_alignment_configs.yaml')
    config_file_user = storage.join_paths(work_dir, 'configs', 'alignment_configs.yaml')
    return config_file_default, config_file_user


@lru_cache(maxsize=1)
def align_configs():
    conf = load_yaml_configs(*align_config_file())
    if (SECTION_THICKNESS is not None) and (conf.get('matching', {}).get('working_mip_level', None) is None):
        align_mip = max(0, math.floor(math.log2(SECTION_THICKNESS / montage_resolution())))
        conf.setdefault('matching', {})
        conf['matching'].setdefault('working_mip_level', align_mip)
    return conf


@lru_cache(maxsize=1)
def thumbnail_config_file():
    work_dir = get_work_dir()
    config_file_default = storage.join_paths(_default_configuration_folder, 'default_thumbnail_configs.yaml')
    config_file_user = storage.join_paths(work_dir, 'configs', 'thumbnail_configs.yaml')
    return config_file_default, config_file_user


@lru_cache(maxsize=1)
def thumbnail_configs():
    conf = load_yaml_configs(*thumbnail_config_file())
    return conf


@lru_cache(maxsize=1)
def stitch_render_dir():
    stitch_conf = stitch_configs()
    render_settings = stitch_conf.get('rendering', {})
    outdir = render_settings.get('out_dir', None)
    if outdir is None:
        work_dir = get_work_dir()
        outdir = storage.join_paths(work_dir, 'stitched_sections')
    return outdir


@lru_cache(maxsize=1)
def align_render_dir():
    align_conf = align_configs()
    render_settings = align_conf.get('rendering', {})
    outdir = render_settings.get('out_dir', None)
    if outdir is None:
        work_dir = get_work_dir()
        outdir = storage.join_paths(work_dir, 'aligned_stack')
    return outdir


@lru_cache(maxsize=1)
def tensorstore_render_dir():
    align_conf = align_configs()
    render_settings = align_conf.get('tensorstore_rendering', {})
    outdir = render_settings.get('out_dir', None)
    if outdir is None:
        work_dir = get_work_dir()
        outdir = storage.join_paths(work_dir, 'aligned_tensorstore')
    outdir = outdir.replace('\\', '/')
    if not outdir.endswith('/'):
        outdir = outdir + '/'
    t_driver, outdir = storage.parse_file_driver(outdir)
    if t_driver == 'file':
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
    cache_file = storage.join_paths(work_dir, 'configs', 'resolutions.yaml')
    res = storage.load_yaml(cache_file)
    if 'DATA_RESOLUTION' in res:
        return res['DATA_RESOLUTION']
    # try parse_coordinate_files
    NUM_SAMPLE_FILES = 5
    DELMITER = '\t'
    coord_dir = storage.join_paths(work_dir, 'stitch', 'stitch_coord')
    coord_list = sorted(storage.list_folder_content(storage.join_paths(coord_dir, '*.txt')))
    coord_list = coord_list[:NUM_SAMPLE_FILES]
    resolution_samples = []
    for coord_file in coord_list:
        try:
            with storage.File(coord_file, 'r') as f:
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
        storage.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with storage.File(cache_file, 'w') as f:
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


lru_cache(maxsize=1)
def thumbnail_resolution():
    thumbnail_mip_lvl = thumbnail_configs().get('thumbnail_mip_level', 6)
    thumbnail_resolution = montage_resolution() * (2 ** thumbnail_mip_lvl)
    return thumbnail_resolution


SECTION_THICKNESS = section_thickness()

def limit_numpy_thread(nthreads):
    nthread_str = str(nthreads)
    os.environ["OMP_NUM_THREADS"] = nthread_str
    os.environ["OPENBLAS_NUM_THREADS"] = nthread_str
    os.environ["MKL_NUM_THREADS"] = nthread_str
    os.environ["VECLIB_MAXIMUM_THREADS"] = nthread_str
    os.environ["NUMEXPR_NUM_THREADS"] = nthread_str


def set_numpy_thread_from_num_workers(num_workers):
    num_cpus = general_settings()['cpu_budget']
    if num_workers > num_cpus:
        num_workers = num_cpus
    if parallel_framework() == 'thread':
        nthreads = num_cpus
    else:
        nthreads = max(1, math.floor(num_cpus / num_workers))
    limit_numpy_thread(nthreads)
    return num_workers