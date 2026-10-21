import copy
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
DEFAULT_AVG_DEFORM = 0.05
MAXIMUM_DEFORM_ALLOWED = 0.35
MATCH_SOFTFACTOR_DOMINANCE = 200 # during matching, assume one mesh is much more rigid than the other so the system will not collapse


@lru_cache(maxsize=1)
def parallel_framework():
    frmwk = general_settings().get('parallel_framework', 'process')
    if frmwk.startswith('pr') or (frmwk == 'builtin'):
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
    # FEABAS_WORKING_DIRECTORY overrides the config file (e.g. set per container/job)
    work_dir = os.environ.get('FEABAS_WORKING_DIRECTORY')
    if not work_dir:
        conf = general_settings()
        work_dir = conf.get('working_directory', './work_dir')
    work_dir = storage.expand_dir(work_dir)
    return work_dir


def default_task_start():
    """default --start: task index of a Google Cloud Batch job, else 0"""
    return int(os.environ.get('BATCH_TASK_INDEX', 0))


def default_task_step():
    """default --step: task count of a Google Cloud Batch job, else 1"""
    return int(os.environ.get('BATCH_TASK_COUNT', 1))


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
        if isinstance(v, dict) and isinstance(default_config.get(k, None), dict):
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
    return _load_align_configs()


@lru_cache(maxsize=1)
def _pristine_align_configs():
    # the scripts modify the dicts returned by align_configs() (pop, setdefault...):
    # helpers reading other stages' settings use this separate copy instead.
    return _load_align_configs()


@lru_cache(maxsize=1)
def _pristine_thumbnail_configs():
    return load_yaml_configs(*thumbnail_config_file())


def _load_align_configs():
    conf = load_yaml_configs(*align_config_file())
    if (SECTION_THICKNESS is not None) and (conf.get('matching', {}).get('working_mip_level', None) is None):
        align_mip = max(0, math.floor(math.log2(SECTION_THICKNESS / mip0_resolution())))
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


_ALIGN_OUTPUTS = ('tiles', 'tensorstore')
_ALIGN_RENDER_OUTPUTS = _ALIGN_OUTPUTS
# section of the alignment configs -> its older tensorstore-only counterpart
_ALIGN_LEGACY_TS_SECTIONS = {'rendering': 'tensorstore_rendering', 'downsample': 'tensorstore_downsample'}


@lru_cache(maxsize=4)
def _align_output_configs(section, output):
    if output not in _ALIGN_OUTPUTS:
        raise ValueError(f'unsupported aligned output {output}')
    layers = []
    for conf_file in align_config_file():
        if (conf_file is None) or (not storage.file_exists(conf_file)):
            continue
        conf = storage.load_yaml(conf_file) or {}
        sec_conf = conf.get(section, None) or {}
        layers.append({k: v for k, v in sec_conf.items() if k not in _ALIGN_OUTPUTS})
        layers.append(sec_conf.get(output, None) or {})
        if output == 'tensorstore':
            layers.append(conf.get(_ALIGN_LEGACY_TS_SECTIONS[section], None) or {})   # older configs
    out = {}
    for layer in layers:
        merge_config(out, copy.deepcopy(layer))
    return out


def _align_render_configs(output):
    return _align_output_configs('rendering', output)


def align_render_configs(output='tiles'):
    """
    a copy of the settings to render the aligned stack, as image tiles
    (output='tiles', align_main.py --mode rendering) or as a tensorstore
    volume (output='tensorstore', --mode tsr): the settings shared at the top
    of the alignment `rendering` section, with those of its `tiles` or
    `tensorstore` subsection on top. Each layer is applied default configs
    first then user configs, so a shared setting given by the user overrides
    a subsection default. For older configs, the flat `rendering` section
    counts as shared settings, and a `tensorstore_rendering` section is
    applied last for tensorstore.
    """
    return copy.deepcopy(_align_render_configs(output))


def align_downsample_configs(output='tiles'):
    """
    a copy of the settings to downsample the aligned stack: image tiles
    (align_main.py --mode downsample) or tensorstore (--mode tsd), layered as
    in align_render_configs from the alignment `downsample` section (and an
    older `tensorstore_downsample` section). For tensorstore, unless given,
    jpeg_compression follows the shared `format`, and mip_levels 'auto' is
    every other level above the rendering mip_level up to max_mip.
    """
    conf = copy.deepcopy(_align_output_configs('downsample', output))
    if output == 'tensorstore':
        conf.setdefault('jpeg_compression', str(conf.get('format', 'jpg')).lower() in ('jpg', 'jpeg'))
        mip_levels = conf.get('mip_levels', 'auto')
        if (mip_levels is None) or (isinstance(mip_levels, str) and mip_levels.lower() == 'auto'):
            src_mip = int(_align_render_configs('tensorstore').get('mip_level', 0))
            max_mip = int(conf.get('max_mip', src_mip + 7))
            mip_levels = list(range(src_mip + 1, max_mip + 1, 2))
            if (len(mip_levels) > 0) and (mip_levels[-1] != max_mip):
                mip_levels.append(max_mip)
            conf['mip_levels'] = mip_levels
    return conf


@lru_cache(maxsize=1)
def align_render_dir():
    outdir = align_render_configs('tiles').get('out_dir', None)
    if outdir is None:
        work_dir = get_work_dir()
        outdir = storage.join_paths(work_dir, 'aligned_stack')
    return outdir


@lru_cache(maxsize=1)
def tensorstore_render_dir():
    outdir = align_render_configs('tensorstore').get('out_dir', None)
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


@lru_cache(maxsize=1)
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


@lru_cache(maxsize=1)
def mip0_resolution():
    """
    resolution (nm) of mip level 0, the grid shared by all the rendered
    outputs (stitched sections at any mip level, alignment, aligned images):
    mip level k is at mip0_resolution() * 2**k, whatever the resolution of
    the source tiles of each section. Defined by the stitching configs
    `mip0_resolution`; for older configs, by rendering: resolution or scale
    (relative to the data resolution); otherwise the data resolution.
    """
    conf = stitch_configs()
    res = conf.get('mip0_resolution', None)
    if res is None:
        render_conf = conf.get('rendering', {})
        res = render_conf.get('resolution', None)
        if res is None:
            res = data_resolution() / render_conf.get('scale', 1.0)
    return res


montage_resolution = mip0_resolution    # deprecated name


@lru_cache(maxsize=1)
def thumbnail_resolution():
    thumbnail_mip_lvl = thumbnail_configs().get('thumbnail_mip_level', 6)
    thumbnail_resolution = mip0_resolution() * (2 ** thumbnail_mip_lvl)
    return thumbnail_resolution


def thumbnail_source_mip():
    """
    mip level of the stitched images the thumbnails are generated from (the
    high-pass intermediate level, or the level right above the thumbnail).
    """
    thumb_conf = _pristine_thumbnail_configs()
    thumbnail_mip_lvl = thumb_conf.get('thumbnail_mip_level', 6)
    ds_conf = thumb_conf.get('downsample', {})
    if ds_conf.get('thumbnail_highpass', True):
        inter_mip = ds_conf.get('highpass_inter_mip_lvl', max(0, thumbnail_mip_lvl-2))
        if 2 ** (thumbnail_mip_lvl - inter_mip) >= 4:
            return inter_mip
    return max(0, thumbnail_mip_lvl-1)


def stitch_render_mips(override=None):
    """
    mip levels the stitching rendering stage should produce. Use override if
    provided, otherwise stitching configs rendering: mip_levels. 'auto' (the
    default) means the levels needed downstream: the alignment matching level
    and the thumbnail source level.
    """
    if override is None:
        override = stitch_configs().get('rendering', {}).get('mip_levels', 'auto')
    if (override is None) or (isinstance(override, str) and override.lower() == 'auto'):
        align_mip = _pristine_align_configs().get('matching', {}).get('working_mip_level', 0)
        mips = {int(align_mip), int(thumbnail_source_mip())}
    elif isinstance(override, str):
        mips = {int(s) for s in override.replace(',', ' ').split()}
    elif hasattr(override, '__iter__'):
        mips = {int(s) for s in override}
    else:
        mips = {int(override)}
    return sorted(mips)


# keys of final_rendering that must stay the same as in rendering: all the
# stitched levels of a section share the same output.
_SHARED_STITCH_RENDER_KEYS = ('driver', 'out_dir', 'mip_levels', 'resolution', 'scale')


def stitch_render_configs(final=False):
    """
    a copy of the stitching rendering settings. With final=True, the settings
    of `final_rendering` (the full-resolution output, also used when the
    aligned sections are rendered directly from the raw tiles) are merged on
    top of `rendering` (the levels used for thumbnails and matching).
    """
    conf = stitch_configs()
    out = copy.deepcopy(conf.get('rendering', {}))
    maskout_val = conf.get('matching', {}).get('maskout_val', None)
    if final:
        final_conf = copy.deepcopy(conf.get('final_rendering', None) or {})
        for key in _SHARED_STITCH_RENDER_KEYS:
            final_conf.pop(key, None)
        merge_config(out, final_conf)
    if maskout_val is not None:
        out.setdefault('render_settings', {})
        out['render_settings'].setdefault('maskout_val', maskout_val)
    return out


def separate_final_stitch_render():
    """whether the final stitched output uses different settings from the matching levels."""
    return stitch_render_configs(final=True) != stitch_render_configs(final=False)


def final_render_mip():
    """
    mip level of the final aligned output: the mip_level of the alignment
    rendering, the coarser one if image tiles and tensorstore differ.
    """
    mips = [_align_render_configs(output).get('mip_level', 0) for output in _ALIGN_RENDER_OUTPUTS]
    return max(int(m) for m in mips)


def is_final_stitch_mip(mip):
    """
    whether a stitched mip level is final output, rendered with the
    final_rendering settings: the levels up to the mip level of the final
    aligned output (even if also used for matching or thumbnails), and any
    level finer than all the levels needed for thumbnails and alignment
    matching.
    """
    return (mip <= final_render_mip()) or (mip < min(stitch_render_mips('auto')))


SECTION_THICKNESS = section_thickness()

def limit_numpy_thread(nthreads):
    nthread_str = str(nthreads)
    os.environ["OMP_NUM_THREADS"] = nthread_str
    os.environ["OPENBLAS_NUM_THREADS"] = nthread_str
    os.environ["MKL_NUM_THREADS"] = nthread_str
    os.environ["VECLIB_MAXIMUM_THREADS"] = nthread_str
    os.environ["NUMEXPR_NUM_THREADS"] = nthread_str


def get_numpy_thread():
    nthread_str = os.getenv("OMP_NUM_THREADS")
    if nthread_str is None:
        return 1
    try:
        return int(nthread_str)
    except ValueError:
        return 1


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