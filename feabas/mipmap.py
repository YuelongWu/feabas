import cv2
import numpy as np
from functools import partial
import json
from scipy.interpolate import interp1d
from scipy.ndimage import binary_dilation
import shapely.geometry as shpgeo
from shapely.ops import unary_union
import os
import tensorstore as ts
import time

from feabas.concurrent import submit_to_workers
from feabas.dal import MosaicLoader, get_tensorstore_spec
from feabas import common, logging, dal, storage
from feabas.spatial import Geometry
from feabas.mesh import Mesh
from feabas.renderer import render_whole_mesh, MeshRenderer
from feabas.config import CHECKPOINT_TIME_INTERVAL

H5File = storage.h5file_class()

def get_image_loader(src_dir, **kwargs):
    ext = kwargs.pop('input_formats', ('png', 'jpg', 'tif', 'bmp'))
    pattern = kwargs.pop('pattern', '_tr{ROW_IND}-tc{COL_IND}.png')
    one_based = kwargs.pop('one_based', True)
    tile_size = kwargs.pop('tile_size', None)
    logger_info = kwargs.pop('logger', None)
    logger = logging.get_logger(logger_info)
    pattern = os.path.splitext(pattern)[0]
    if isinstance(ext, str):
        ext = (ext,)
    for e in ext:
        imgpaths = storage.list_folder_content(storage.join_paths(src_dir, '*.' + e))
        if len(imgpaths) > 0:
            ext = e
            break
    else:
        logger.warning(f'{src_dir}: no image found.')
        return None
    meta_file = storage.join_paths(src_dir, 'metadata.txt')
    if storage.file_exists(meta_file):
        image_loader = MosaicLoader.from_coordinate_file(meta_file, **kwargs)
    else:
        pattern0 = pattern.replace('{', '({').replace('}', r'}\d+)')
        if one_based:
            tile_offset = (-1, -1)
        else:
            tile_offset = (0, 0)
        image_loader = MosaicLoader.from_filepath(imgpaths, pattern=pattern0,
                        tile_size=tile_size, tile_offset=tile_offset, **kwargs)
    return image_loader


def _mesh_from_image_loader(image_loader):
    resolution0 = image_loader.resolution
    bboxes = []
    for bbox in image_loader.file_bboxes(margin=1):
        bboxes.append(shpgeo.box(*bbox))
    covered = unary_union(bboxes)
    n_tiles = len(bboxes)
    mesh_size = (covered.area * 0.5 / n_tiles) ** 0.5
    covered = covered.simplify(0.1)
    G = Geometry(roi=covered, resolution=resolution0)
    M = Mesh.from_PSLG(**G.PSLG(), mesh_size=mesh_size, min_mesh_angle=20)
    return M


def mip_one_level(src_dir, out_dir, **kwargs):
    ext_out = kwargs.pop('output_format', 'png')
    pattern = kwargs.pop('pattern', '_tr{ROW_IND}-tc{COL_IND}.png')
    tile_size = kwargs.pop('tile_size', None)
    downsample = kwargs.pop('downsample', 2)
    logger_info = kwargs.get('logger', None)
    downsample_method = kwargs.get('downsample_method', 'mean')
    remap_interp_lookup = {'mean': cv2.INTER_AREA, 'stride': cv2.INTER_NEAREST,
                           'nearest': cv2.INTER_NEAREST, 'linear': cv2.INTER_LINEAR}
    kwargs.setdefault('remap_interp', remap_interp_lookup.get(downsample_method, cv2.INTER_AREA))
    kwargs.setdefault('cache_type', 'fifo')
    kwargs.setdefault('cache_size', downsample + 2)
    if (kwargs['remap_interp'] != cv2.INTER_NEAREST) and (downsample > 2):
        kwargs.setdefault('preprocess', _smooth_filter_factory)
        kwargs.setdefault('preprocess_params', {'blur': downsample, 'sigma': 0})
    logger = logging.get_logger(logger_info)
    out_meta_file = storage.join_paths(out_dir, 'metadata.txt')
    if storage.file_exists(out_meta_file):
        n_img = len(storage.list_folder_content(storage.join_paths(out_dir, '*.'+ext_out)))
        return -n_img
    rendered = {}
    try:
        if ext_out == 'jpg':
            kwargs.setdefault('dtype', np.uint8)
        image_loader = get_image_loader(src_dir, pattern=pattern, tile_size=tile_size, **kwargs)
        if image_loader is None:
            return 0
        pattern = os.path.splitext(pattern)[0]
        M = _mesh_from_image_loader(image_loader)
        if tile_size is None:
            for bbox in image_loader.file_bboxes(margin=0):
                tile_size = (bbox[3] - bbox[1], bbox[2] - bbox[0])
                break
        prefix0 = os.path.commonprefix(image_loader.imgrelpaths)
        splitter = pattern.split('{')[0]
        if splitter:
            prefix0 = prefix0.split(splitter)[0]
        prefix = storage.join_paths(out_dir, prefix0)
        out_root_dir = os.path.dirname(prefix)
        storage.makedirs(out_root_dir, exist_ok=True)
        kwargs.setdefault('seeds', downsample)
        kwargs.setdefault('mx_dis', (tile_size[0]/2+4, tile_size[-1]/2+4))
        rendered = render_whole_mesh(M, image_loader, prefix, tile_size=tile_size,
                                     pattern=pattern+'.'+ext_out, scale= 1/downsample,
                                     **kwargs)
        if len(rendered) > 0:
            fnames = sorted(list(rendered.keys()))
            bboxes = []
            for fname in fnames:
                bboxes.append(rendered[fname])
            out_loader = dal.StaticImageLoader(fnames, bboxes=bboxes,
                                               resolution=image_loader.resolution*downsample)
            out_loader.to_coordinate_file(out_meta_file)
    except Exception as err:
        logger.error(f'{src_dir}: {err}')
        return None
    return len(rendered)


def mip_map_one_section(sec_name, img_dir, max_mip, **kwargs):
    ext_out = kwargs.pop('format', 'jpg')
    logger_info = kwargs.get('logger', None)
    logger = logging.get_logger(logger_info)
    t0 = time.time()
    num_tiles = []
    updated = False
    try:
        for m in range(max_mip):
            src_dir = storage.join_paths(img_dir, 'mip'+str(m), sec_name)
            out_dir = storage.join_paths(img_dir, 'mip'+str(m+1), sec_name)
            n_tile = mip_one_level(src_dir, out_dir, output_format=ext_out,
                                        downsample=2, **kwargs)
            if n_tile is None:
                updated = False
                break
            if n_tile > 0:
                updated = True
            num_tiles.append(abs(n_tile))
    except TimeoutError:
        logger.error(f'{sec_name}: Tensorstore timed out.')
        updated = False
    except Exception as err:
        logger.error(f'{sec_name}: {err}')
        updated = False
    if updated:
        logger.info(f'{sec_name}: number of tiles {num_tiles} | {(time.time()-t0)/60} min')
    return {sec_name: updated}


def create_thumbnail(src_dir, outname=None, downsample=4, highpass=True, **kwargs):
    normalize_hist = kwargs.get('normalize_hist', True)
    kwargs.setdefault('remap_interp', cv2.INTER_AREA)
    if kwargs['remap_interp'] == cv2.INTER_NEAREST:
        blur = 1
    else:
        blur = downsample
    kwargs.setdefault('dtype', np.float32)
    if highpass:
        kwargs.setdefault('preprocess', _smooth_filter_factory)
        kwargs.setdefault('preprocess_params', {'blur': blur, 'sigma': 0.5})
    else:
        if blur <= 2:
            kwargs.setdefault('preprocess', None)
        else:
            kwargs.setdefault('preprocess', _smooth_filter_factory)
            kwargs.setdefault('preprocess_params', {'blur': downsample, 'sigma': 0})
    kwargs.setdefault('cache_type', 'fifo')
    kwargs.setdefault('cache_size', 8)
    image_loader = get_image_loader(src_dir, **kwargs)
    if image_loader is None:
        return None
    M = _mesh_from_image_loader(image_loader)
    bounds0 = M.bbox()
    for bbox in image_loader.file_bboxes(margin=0):
        tile_size0 = (bbox[3] - bbox[1], bbox[2] - bbox[0])
        break
    n_col = round(bounds0[2] / tile_size0[1])
    n_row = round(bounds0[3] / tile_size0[0])
    kwargs.setdefault('seeds', (n_row, n_col))
    kwargs.setdefault('mx_dis', (tile_size0[0]/2+4, tile_size0[-1]/2+4))
    M.change_resolution(image_loader.resolution * downsample)
    bounds1 = M.bbox()
    out_wd, out_ht = bounds1[2], bounds1[3]
    out_bbox = (0, 0, out_wd, out_ht)
    if highpass:
        rndr = MeshRenderer.from_mesh(M, fillval=0, dtype=np.float32, image_loader=image_loader)
        img = rndr.crop(out_bbox, **kwargs)
        img = 255 - _max_entropy_scaling_one_side(img)
    else:
        rndr = MeshRenderer.from_mesh(M, image_loader=image_loader)
        img = rndr.crop(out_bbox, **kwargs)
        if normalize_hist:
            img = _max_entropy_scaling_both_sides(img)
    if outname is None:
        return img
    else:
        common.imwrite(outname, img)


def generate_target_tensorstore_scale(metafile, mip=None, **kwargs):
    max_tile_per_job = kwargs.pop('max_tile_per_job', 16)
    use_jpeg_compression = kwargs.pop('format', None) in ('jpg', 'jpeg')
    kwargs['use_jpeg_compression'] = use_jpeg_compression
    kwargs['cache_size'] = max_tile_per_job
    kwargs['downsample_z'] = 1
    write_to_file = False
    json_obj, write_to_file = common.parse_json_file(metafile)
    if write_to_file:
        kwargs['checkpoint_prefix'] = os.path.splitext(metafile)[0] + f'_mip{mip}_'
    _, src_mip, mip, mipmaps = get_tensorstore_spec(json_obj, mip=mip, return_mips=True, **kwargs)
    if src_mip == mip:
        return False, None
    src_spec = mipmaps[src_mip]
    err_raised, out_spec, _ = mip_one_level_tensorstore_3d(src_spec, mipup=mip-src_mip, **kwargs)
    mipmaps.update({int(mip): out_spec})
    if (not err_raised) and write_to_file:
        with storage.File(metafile, 'w') as f:
            json.dump(mipmaps, f)
    return err_raised, mipmaps


def generate_tensorstore_scales(metafile, mips, **kwargs):
    logger_info = kwargs.get('logger', None)
    t0 = time.time()
    logger = logging.get_logger(logger_info)
    mips = np.sort(mips)
    sec_name = os.path.basename(metafile).replace('.json', '')
    updated = False
    for mip in mips:
        err_raised, specs = generate_target_tensorstore_scale(metafile, mip=mip, **kwargs)
        if err_raised:
            break
        if specs is not None:
            updated = True
    if updated:
        logger.info(f'{sec_name}: {(time.time()-t0)/60} min')
    return {sec_name: updated}



def create_thumbnail_tensorstore(metafile, mip, outname=None, highpass=True, **kwargs):
    normalize_hist = kwargs.get('normalize_hist', True)
    downsample_method = kwargs.get('downsample_method', 'mean')
    if not highpass:
        ds_spec = get_tensorstore_spec(metafile, mip=mip, downsample_method=downsample_method)
        thumb_loader = dal.TensorStoreLoader.from_json_spec(ds_spec, **kwargs)
        img = thumb_loader.crop(thumb_loader.bounds, **kwargs)
        if normalize_hist:
            img = _max_entropy_scaling_both_sides(img)
    else:
        inter_mip = kwargs.get('highpass_inter_mip_lvl', max(0, mip-2))
        assert mip > inter_mip
        ds_mip = mip - inter_mip
        inter_spec = get_tensorstore_spec(metafile, mip=inter_mip, downsample_method='mean')
        mx_spec = get_tensorstore_spec({0: inter_spec}, 1, downsample_method='max')
        mn_spec = get_tensorstore_spec({0: inter_spec}, 1, downsample_method='min')
        mx_spec = {'driver': 'cast', 'dtype': 'float32', 'base': mx_spec}
        mn_spec = {'driver': 'cast', 'dtype': 'float32', 'base': mn_spec}
        mx_spec = get_tensorstore_spec({1: mx_spec}, ds_mip, downsample_method='mean')
        mn_spec = get_tensorstore_spec({1: mn_spec}, ds_mip, downsample_method='mean')
        mx_loader = dal.TensorStoreLoader.from_json_spec(mx_spec, **kwargs)
        mn_loader = dal.TensorStoreLoader.from_json_spec(mn_spec, **kwargs)
        mx_img = mx_loader.crop(mx_loader.bounds, **kwargs)
        mn_img = mn_loader.crop(mn_loader.bounds, **kwargs)
        img = 255 - _max_entropy_scaling_one_side(mx_img - mn_img)
    if outname is None:
        return img
    else:
        common.imwrite(outname, img)


def _max_entropy_scaling_one_side(img, **kwargs):
    if np.ptp(img, axis=None) == 0:
        return img.astype(np.uint8)
    hist_step = kwargs.get('hist_step', 0.1)
    lower_bound = kwargs.get('lower_bound', 0.7)
    upper_bound = kwargs.get('upper_bound', 1.0)
    gain_step = kwargs.get('gain_step', 1.05)
    data0 = img[img > 0]
    scale0 = 128 / np.mean(data0, axis=None)
    data0 = data0 * scale0
    bin_edges = np.arange(0, max(256, data0.max() + hist_step), hist_step)
    edge_cntrs = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist, _ = np.histogram(data0, bins=bin_edges)
    idxt = hist > 0
    wd = int(round(10 * np.sum(hist[1:-1]) / np.max(hist[1:-1])))
    if wd > 0:
        idxt = idxt | ~binary_dilation(idxt, iterations=wd)
    idxt[0] = 1
    idxt[-1] = 1
    edge_cntrs = edge_cntrs[idxt]
    hist = hist[idxt]
    cdf = np.cumsum(hist, axis=None)
    if upper_bound == 1.0:
        uu = np.max(data0)
    else:
        uu = np.quantile(data0, upper_bound)
    bb = np.quantile(data0, lower_bound)
    if uu == bb:
        return (img * scale0).astype(np.uint8)
    gain = (np.log(uu) - np.log(bb)) / np.log(gain_step)
    trials = bb * (gain_step ** np.linspace(0, gain, num=round(gain)+1, endpoint=True))
    lvls = trials.reshape(-1, 1) * (np.arange(1, 255) / 255)
    cdf_interp = interp1d(edge_cntrs, cdf, kind='linear', bounds_error=False, fill_value=(0, cdf[-1]), assume_sorted=True)
    cdf_matrix = cdf_interp(lvls)
    pdf_matrix = np.diff(cdf_matrix, n=1, axis=-1, prepend=0, append=cdf[-1]) / cdf[-1]
    indx = pdf_matrix > 0
    pp = pdf_matrix[indx]
    pdf_matrix[indx] = pp * np.log(pp)
    entropy = -np.sum(pdf_matrix, axis=-1)
    entropy_wt = np.exp(3*entropy)
    entropy_wt = entropy_wt / np.sum(entropy_wt)
    sel = int(np.round(np.sum(np.arange(entropy.size) * entropy_wt)))
    # sel = np.argmax(entropy)
    scale = 255 / trials[sel]
    img = (img.astype(np.float32) * scale0 * scale).clip(0, 255).astype(np.uint8)
    return img


def _max_entropy_scaling_both_sides(img, **kwargs):
    if np.ptp(img, axis=None) == 0:
        return img.astype(np.uint8)
    hist_num = kwargs.get('hist_num', 3000)
    right_lower_bound = kwargs.get('r_lower_bound', 0.7)
    right_upper_bound = kwargs.get('r_upper_bound', 1.0)
    left_lower_bound = kwargs.get('l_lower_bound', 0.0)
    left_upper_bound = kwargs.get('l_upper_bound', 0.3)
    num_gain = 50
    data0 = img[(img > img.min()) & (img < img.max())]
    bin_edges = np.linspace(data0.min(), data0.max(), num=hist_num, endpoint=True)
    edge_cntrs = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist, _ = np.histogram(data0, bins=bin_edges)
    idxt = hist > 0
    wd = int(round(10 * np.sum(hist[1:-1]) / np.max(hist[1:-1])))
    if wd > 0:
        idxt = idxt | ~binary_dilation(idxt, iterations=wd)
    idxt[0] = 1
    idxt[-1] = 1
    edge_cntrs = edge_cntrs[idxt]
    hist = hist[idxt]
    cdf = np.cumsum(hist, axis=None)
    if right_upper_bound == 1.0:
        ru = np.max(data0)
    else:
        ru = np.quantile(data0, right_upper_bound)
    rl = np.quantile(data0, right_lower_bound)
    if left_lower_bound == 0:
        ll = np.min(data0)
    else:
        ll = np.quantile(data0, left_lower_bound)
    lu = np.quantile(data0, left_upper_bound)
    trials_r = np.linspace(rl, ru, num=num_gain, endpoint=True)
    trials_l = np.linspace(ll, lu, num=num_gain, endpoint=True)
    gain = trials_r.reshape(-1, 1, 1) - trials_l.reshape(1, -1 ,1)
    lvls = gain * (np.arange(1, 255) / 255) + trials_l.reshape(1, -1 ,1)
    cdf_interp = interp1d(edge_cntrs, cdf, kind='linear', bounds_error=False, fill_value=(0, cdf[-1]), assume_sorted=True)
    cdf_matrix = cdf_interp(lvls)
    pdf_matrix = np.diff(cdf_matrix, n=1, axis=-1, prepend=0, append=cdf[-1]) / cdf[-1]
    indx = pdf_matrix > 0
    pp = pdf_matrix[indx]
    pdf_matrix[indx] = pp * np.log(pp)
    entropy = -np.sum(pdf_matrix, axis=-1)
    entropy_wt = np.exp(3*entropy)
    entropy_wt = entropy_wt / np.sum(entropy_wt, axis=None)
    idx0 = int(np.round(np.sum(np.arange(entropy.shape[0]).reshape(-1,1) * entropy_wt, axis=None)))
    idx1 = int(np.round(np.sum(np.arange(entropy.shape[1]).reshape(1,-1) * entropy_wt, axis=None)))
    # idx_mx = np.argmax(entropy, axis=None)
    # idx0, idx1 = np.unravel_index(idx_mx, entropy.shape)
    r = trials_r[idx0]
    l = trials_l[idx1]
    img = (255*(img.astype(np.float32) - l)/(r-l)).clip(0, 255).astype(np.uint8)
    return img


def _smooth_filter_factory(**kwargs):
    blur = kwargs.get('blur', 2)
    sigma = kwargs.get('sigma', 0.0)
    downsample = kwargs.get('downsample_mode', True)
    return partial(_smooth_filter, blur=blur, sigma=sigma, downsample=downsample)


def _smooth_filter(img, blur=2, sigma=0.0, downsample=True):
    if sigma > 0:
        img = common.masked_dog_filter(img, sigma=sigma, signed=False)
    if downsample:
        sz = img.shape
        img = cv2.resize(img, None, fx = 1/blur, fy=1/blur, interpolation=cv2.INTER_AREA)
        img = cv2.resize(img, sz, interpolation=cv2.INTER_NEAREST)
    else:
        if blur > 2:
            if blur % 2 == 1:
                img = cv2.blur(img, (round(blur), round(blur)))
            else:
                img = cv2.blur(img, (round(blur-1), round(blur-1)))
    return img


def _write_ts_block_from_indices(ind_x, ind_y, ind_z, src_spec, out_spec, **kwargs):
    task_id = kwargs.pop('task_id', None)
    src_data = dal.TensorStoreLoader.from_json_spec(src_spec, **kwargs)
    out_data = dal.TensorStoreWriter.from_json_spec(out_spec)
    bboxes = out_data.grid_indices_to_bboxes(ind_x, ind_y, ind_z)
    flag = np.ones(len(bboxes), dtype=bool)
    errmsg = ''
    for k, bbox in enumerate(bboxes):
        try:
            chunk = src_data.get_chunk(bbox)
            if chunk is not None:
                out_data.write_single_chunk(bbox, chunk)
        except Exception as err:
            errmsg = errmsg + f'\t{bbox}: {err}\n'
            break
        else:
            flag[k] = False
    if np.all(flag):
        flag = True
    elif not np.any(flag):
        flag = False
    return task_id, flag, errmsg



def mip_one_level_tensorstore_3d(src_spec, mipup=1, **kwargs):
    num_workers = kwargs.get('num_workers', 1)
    keep_chunk_layout = kwargs.get('keep_chunk_layout', True)
    downsample_z = kwargs.get('downsample_z', 'auto')
    z_range = kwargs.get('z_range', None)
    full_chunk_only = kwargs.get('full_chunk_only', z_range is not None)
    downsample_method = kwargs.get("downsample_method", "mean")
    kvstore_out = kwargs.get('kvstore_out', None)
    use_jpeg_compression = kwargs.get('use_jpeg_compression', True)
    pad_to_tile_size = kwargs.get('pad_to_tile_size', use_jpeg_compression)
    mask_file = kwargs.get('mask_file', None)
    cache_capacity = kwargs.get('cache_capacity', None)
    cache_size = kwargs.get('cache_size', None)
    logger_info = kwargs.get('logger', None)
    logger = logging.get_logger(logger_info)
    flag_prefix = kwargs.get('flag_prefix', None)
    checkpoint_prefix = kwargs.get('checkpoint_prefix', None)
    err_raised = False
    if (flag_prefix is not None) and (checkpoint_prefix is None):
        checkpoint_dir = storage.join_paths(os.path.dirname(flag_prefix), 'checkpoints')
        flag_filename = os.path.basename(flag_prefix)
        checkpoint_prefix = storage.join_paths(checkpoint_dir, flag_filename + '_')
        storage.makedirs(os.path.dirname(checkpoint_prefix))
    src_loader = dal.TensorStoreLoader.from_json_spec(src_spec)
    src_data = src_loader.dataset
    src_spec = src_data.spec(minimal_spec=True).to_json()
    if kvstore_out is None:
        kvstore_out = src_spec["kvstore"]
    scl = 2 ** mipup
    if downsample_z == 'auto':
        resln0, _, thick0 = src_loader.pixel_size
        new_resln = resln0 * scl
        downsample_z = 2**max(0, round(np.log(new_resln/thick0)/np.log(2)))
    downsample_factors = [scl, scl, downsample_z, 1]
    dsp_spec = {
        "driver": "downsample",
        "downsample_factors": downsample_factors,
        "downsample_method": downsample_method,
        "base": src_spec
    }
    dsp_loader = dal.TensorStoreLoader.from_json_spec(dsp_spec)
    dsp_data = dsp_loader.dataset
    out_spec = {"driver": "neuroglancer_precomputed", "kvstore": kvstore_out}
    src_schema = src_data.schema.to_json()
    dsp_schema = dsp_data.schema.to_json()
    out_schema = src_schema.copy()
    out_schema["dimension_units"] = dsp_schema["dimension_units"]
    if keep_chunk_layout:
        chunk_layout = src_schema["chunk_layout"]
    else:
        chunk_layout = dsp_schema["chunk_layout"]
    write_shape = chunk_layout["write_chunk"].pop("shape")
    tile_wd, tile_ht = write_shape[0], write_shape[1]
    chunk_layout["write_chunk"]["shape_soft_constraint"] = write_shape
    read_shape = chunk_layout["read_chunk"].pop("shape")
    read_wd, read_ht = read_shape[0], read_shape[1]
    chunk_layout["read_chunk"]["shape_soft_constraint"] = read_shape
    out_schema["chunk_layout"] = chunk_layout
    out_schema["domain"] = dsp_schema["domain"]
    Xmin, Ymin = dsp_data.domain.inclusive_min[:2]
    Xmax, Ymax = dsp_data.domain.exclusive_max[:2]
    if pad_to_tile_size:
        Xmin = int(np.floor(Xmin / read_wd)) * read_wd
        Ymin = int(np.floor(Ymin / read_ht)) * read_ht
        Xmax = int(np.ceil(Xmax / read_wd)) * read_wd
        Ymax = int(np.ceil(Ymax / read_ht)) * read_ht
        out_schema['domain']['exclusive_max'][:2] = [Xmax, Ymax]
        out_schema['domain']['inclusive_min'][:2] = [Xmin, Ymin]
    if use_jpeg_compression:
        out_schema["codec"] = {"driver": "neuroglancer_precomputed", "encoding": 'jpeg', "jpeg_quality": 95}
        if (read_ht < tile_ht) or (read_wd < tile_wd):
            out_schema["codec"].update({"shard_data_encoding": 'raw'})
    else:
        out_schema["codec"] = {"driver": "neuroglancer_precomputed", "encoding": 'raw'}
        if (read_ht < tile_ht) or (read_wd < tile_wd):
            out_schema["codec"].update({"shard_data_encoding": 'gzip'})
    out_spec["schema"] = out_schema
    if (flag_prefix is None):
        zind_rendered = []
    else:
        zind_rendered = set()
        flag_list = storage.list_folder_content(flag_prefix+'*')
        for flgfile in flag_list:
            with storage.File(flgfile, 'r') as f:
                zidrnd = json.load(f)
                zind_rendered = zind_rendered.union(zidrnd)
        zind_rendered = sorted(list(zind_rendered))
    out_spec.update({"open": True, "create": True, "delete_existing": False})
    out_writer = dal.TensorStoreWriter.from_json_spec(out_spec)
    out_spec = out_writer.spec
    Nx, Ny, Nz = out_writer.grid_shape
    Z0, Z1 = out_writer.write_grids[2], out_writer.write_grids[5]
    if z_range is not None:
        z_ptp = Z1.max() - Z0.min()
        z_min = round(min(z_range) * z_ptp + Z0.min())
        z_max = round(max(z_range) * z_ptp + Z0.min())
        if full_chunk_only:
            idx0 = np.searchsorted(Z0, z_min, side='left')
            idx1 = np.searchsorted(Z1, z_max, side='right') - 1
        else:
            idx0 = np.searchsorted(Z0, z_min, side='right') - 1
            idx1 = np.searchsorted(Z1, z_max, side='left')
        zind_to_render0 = np.unique(np.arange(idx0, idx1+1).clip(0, Nz-1))
        if zind_to_render0.size == 0:
            z_range = []
        else:
            z_range = [(Z0[idx0] - Z0.min())/z_ptp, (Z1[idx1] - Z0.min())/ z_ptp]
    else:
        zind_to_render0 = np.arange(Nz)
    zind_to_render = [int(z) for z in zind_to_render0 if (z not in zind_rendered)]
    if len(zind_to_render) == 0:
        return err_raised, out_writer.spec, z_range
    if flag_prefix is not None:
        flag_file = flag_prefix + f'{Z0[zind_to_render0[0]]}_{Z1[zind_to_render0[-1]]}.json'
        storage.makedirs(os.path.dirname(flag_file))
    else:
        flag_file = None
    chunk_shape = out_writer.write_chunk_shape
    chunk_mb = np.prod(chunk_shape) * np.prod(downsample_factors)/ (1024**2)
    if cache_capacity is not None:
        cache_capacity_per_worker = cache_capacity / num_workers
        chunk_per_job = max(1, cache_capacity_per_worker / chunk_mb)
        if cache_size is not None:
            chunk_per_job = min(cache_size, chunk_per_job)
        max_tasks_per_child = 1
    else:
        cache_capacity_per_worker = None
        chunk_per_job = min(20, max(1, round(Nx*Ny*len(zind_to_render)/num_workers)))
        if cache_size is not None:
            chunk_per_job = min(cache_size, chunk_per_job)
        max_tasks_per_child = None
    z_step = max(1, int(np.floor(chunk_per_job * num_workers / (Nx * Ny))))
    zind_batches = [zind_to_render[k:(k+z_step)] for k in range(0, len(zind_to_render), z_step)]
    mid_x, mid_y = out_writer.morton_xy_grid()
    mask_flag = None
    if (mask_file is not None) and storage.file_exists(mask_file):
        mask = common.imread(mask_file)
        mask = cv2.resize(mask, (Nx, Ny), interpolation=cv2.INTER_AREA)
        mask = cv2.dilate(mask, np.ones((3,3), dtype=np.uint8)) > 0
        mask_flag = mask[mid_y, mid_x]
    for id_zs in zind_batches:
        id_x0 = np.tile(mid_x, len(id_zs))
        id_y0 = np.tile(mid_y, len(id_zs))
        id_z0 = np.repeat(id_zs, mid_x.size)
        if checkpoint_prefix is not None:
            filter_indx = []
            for zz in id_zs:
                checkpoint_file = checkpoint_prefix + str(zz) + '.h5'
                if storage.file_exists(checkpoint_file):
                    with H5File(checkpoint_file, 'r') as f:
                        flg = f[str(zz)][()]
                else:
                    flg = np.ones(mid_x.size, dtype=bool)
                if mask_flag is not None:
                    flg = flg & mask_flag
                filter_indx.append(flg)
            filter_indx = np.concatenate(filter_indx, axis=None)
            id_x0 = id_x0[filter_indx]
            id_y0 = id_y0[filter_indx]
            id_z0 = id_z0[filter_indx]
        else:
            filter_indx = np.ones(id_x0.size, dtype=bool)
        num_chunks = id_x0.size
        N_batch = max(1, round(num_chunks / chunk_per_job))
        bindx = np.unique(np.linspace(0, num_chunks, N_batch+1, endpoint=True).astype(np.uint32))
        tasks = []
        task_lut = {}
        task_id = 0
        for bidx0, bidx1 in zip(bindx[:-1], bindx[1:]):
            task_lut[task_id] = slice(bidx0, bidx1)
            bkwargs = {
                "ind_x": id_x0[bidx0:bidx1],
                "ind_y": id_y0[bidx0:bidx1],
                "ind_z": id_z0[bidx0:bidx1],
                "src_spec": dsp_spec,
                "out_spec": out_spec,
                "task_id": task_id,
                "cache_capacity": cache_capacity_per_worker
            }
            tasks.append(bkwargs)
            task_id += 1
        flag_cut = np.ones(num_chunks, dtype=bool)
        t_check = time.time()
        zind_added = []
        res_cnt = 0
        for res in submit_to_workers(_write_ts_block_from_indices, kwargs=tasks, num_workers=num_workers, max_tasks_per_child=max_tasks_per_child):
            task_id, flg_b, errmsg = res
            res_cnt += 1
            bidx = task_lut[task_id]
            flag_cut[bidx] = flg_b
            if len(errmsg) > 0:
                err_raised = True
                logger.error(errmsg)
            if (checkpoint_prefix is not None) and ((time.time() - t_check) > CHECKPOINT_TIME_INTERVAL) and (res_cnt>=num_workers):
                storage.makedirs(os.path.dirname(checkpoint_prefix))
                res_cnt = 0
                t_check = time.time()
                flag_all = np.zeros_like(filter_indx)
                flag_all[filter_indx] = flag_cut
                flag_all = flag_all.reshape(len(id_zs), -1)
                storage.makedirs(os.path.dirname(checkpoint_prefix))
                newly_finished = False
                for zz, flg_z in zip(id_zs, flag_all):
                    if np.all(flg_z) or (zz in zind_added):
                        continue
                    elif np.any(flg_z):
                        checkpoint_file = checkpoint_prefix + str(zz) + '.h5'
                        with H5File(checkpoint_file, 'w') as f:
                            f.create_dataset(str(zz), data=flg_z, compression="gzip")
                    else:
                        if zz not in zind_added:
                            newly_finished = True
                            zind_added.append(zz)
                if (flag_file is not None) and newly_finished:
                    zind_rendered = sorted(list(set(zind_added).union(zind_rendered)))
                    zind_rendered = [int(zz) for zz in zind_rendered]
                    with storage.File(flag_file, 'w') as f:
                        json.dump(zind_rendered, f)
        if flag_file is not None:
            newly_finished = False
            flag_all = np.zeros_like(filter_indx)
            flag_all[filter_indx] = flag_cut
            flag_all = flag_all.reshape(len(id_zs), -1)
            newly_finished = False
            for zz, flg_z in zip(id_zs, flag_all):
                if np.any(flg_z):
                    err_raised = True
                else:
                    if zz not in zind_added:
                        newly_finished = True
                        zind_added.append(zz)
            if newly_finished:
                zind_rendered = sorted(list(set(zind_added).union(zind_rendered)))
                zind_rendered = [int(zz) for zz in zind_rendered]
                with storage.File(flag_file, 'w') as f:
                    json.dump(zind_rendered, f)
        if (checkpoint_prefix is not None) and (not err_raised):
             for zz in id_zs:
                checkpoint_file = checkpoint_prefix + str(zz) + '.h5'
                storage.remove_file(checkpoint_file)
    return err_raised, out_writer.spec, z_range
