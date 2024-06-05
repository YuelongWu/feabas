from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures import as_completed
import cv2
import glob
import json
from multiprocessing import get_context
import numpy as np
from functools import partial
from scipy.interpolate import interp1d
from scipy.ndimage import binary_dilation
import shapely.geometry as shpgeo
from shapely.ops import unary_union
import os
import tensorstore as ts
import time

from feabas.dal import MosaicLoader, get_tensorstore_spec
from feabas import common, logging, dal, config
from feabas.spatial import Geometry
from feabas.mesh import Mesh
from feabas.renderer import render_whole_mesh, MeshRenderer
from feabas.config import TS_TIMEOUT


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
        imgpaths = glob.glob(os.path.join(src_dir, '*.' + e))
        if len(imgpaths) > 0:
            ext = e
            break
    else:
        logger.warning(f'{src_dir}: no image found.')
        return None
    meta_file = os.path.join(src_dir, 'metadata.txt')
    if os.path.isfile(meta_file):
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
        kwargs.setdefault('preprocess', partial(_smooth_filter, blur=downsample, sigma=0.0))
    logger = logging.get_logger(logger_info)
    out_meta_file = os.path.join(out_dir, 'metadata.txt')
    if os.path.isfile(out_meta_file):
        n_img = len(glob.glob(os.path.join(out_dir, '*.'+ext_out)))
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
        prefix = os.path.join(out_dir, prefix0)
        out_root_dir = os.path.dirname(prefix)
        os.makedirs(out_root_dir, exist_ok=True)
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
            src_dir = os.path.join(img_dir, 'mip'+str(m), sec_name)
            out_dir = os.path.join(img_dir, 'mip'+str(m+1), sec_name)
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
        kwargs.setdefault('preprocess', partial(_smooth_filter, blur=blur, sigma=0.5))
    else:
        if blur <= 2:
            kwargs.setdefault('preprocess', None)
        else:
            kwargs.setdefault('preprocess', partial(_smooth_filter, blur=blur, sigma=0.0))
    kwargs.setdefault('cache_type', 'fifo')
    kwargs.setdefault('cache_size', 8)
    image_loader = get_image_loader(src_dir, **kwargs)
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
    num_workers = kwargs.get('num_workers', 1)
    max_tile_per_job = kwargs.get('max_tile_per_job', 16)
    write_to_file = False
    if isinstance(metafile, str):
        try:
            json_obj = json.loads(metafile)
        except ValueError:
            write_to_file = True
            if metafile.startswith('gs:'):
                json_ts = ts.open({"driver": "json", "kvstore": metafile}).result()
                s = json_ts.read().result()
                json_obj = s.item()
            else:
                with open(metafile, 'r') as f:
                    json_obj = json.load(f)
    elif isinstance(metafile, dict):
        json_obj = metafile
    ds_spec, src_mip, mip, mipmaps = get_tensorstore_spec(json_obj, mip=mip, return_mips=True, **kwargs)
    if src_mip == mip:
        return None
    ts_dsp = ts.open(ds_spec).result()
    ts_src = ts_dsp.base
    src_spec = ts_src.spec(minimal_spec=True).to_json()
    tgt_spec = {}
    driver = src_spec['driver']
    tgt_spec['driver'] = driver
    tgt_spec['kvstore'] = src_spec['kvstore']
    if driver == 'neuroglancer_precomputed':
        pass
    elif driver == 'n5':
        pth = tgt_spec['kvstore']['path']
        pth = pth[::-1].replace(str(src_mip)+'s', str(mip)+'s', 1)[::-1]
        tgt_spec['kvstore']['path'] = pth
    elif driver == 'zarr':
        pth = tgt_spec['kvstore']['path']
        pth = pth[::-1].replace(str(src_mip), str(mip), 1)[::-1]
        tgt_spec['kvstore']['path'] = pth
    else:
        raise ValueError(f'driver type {driver} not supported.')
    tgt_schema = ts_src.schema.to_json()
    ds_schema = ts_dsp.schema.to_json()
    tgt_schema['domain'] = ds_schema['domain']
    tgt_schema['dimension_units'] = ds_schema['dimension_units']
    inclusive_min = ts_dsp.domain.inclusive_min
    exclusive_max = ts_dsp.domain.exclusive_max
    Xmin, Ymin = inclusive_min[0], inclusive_min[1]
    Xmax, Ymax = exclusive_max[0], exclusive_max[1]
    chunk_shape = ts_src.schema.chunk_layout.write_chunk.shape
    tile_wd, tile_ht = chunk_shape[0], chunk_shape[1]
    while tile_wd > (Xmax - Xmin) or tile_ht > (Ymax - Ymin):
        tile_wd = tile_wd // 2
        tile_ht = tile_ht // 2
    tgt_schema['chunk_layout']['write_chunk']['shape'][:2] = [tile_wd, tile_ht]
    read_chunk_shape = ts_src.schema.chunk_layout.read_chunk.shape
    read_wd, read_ht = read_chunk_shape[0], read_chunk_shape[1]
    while read_wd > (Xmax - Xmin) or read_ht > (Ymax - Ymin):
        read_wd = read_wd // 2
        read_ht = read_ht // 2
    tgt_schema['chunk_layout']['read_chunk']['shape'][:2] = [read_wd, read_ht]
    tgt_spec['schema'] = tgt_schema
    x1d = np.arange(Xmin, Xmax, tile_wd, dtype=np.int64)
    y1d = np.arange(Ymin, Ymax, tile_ht, dtype=np.int64)
    xmn, ymn = np.meshgrid(x1d, y1d)
    xmn, ymn = xmn.ravel(), ymn.ravel()
    idxz = common.z_order(np.stack((xmn // tile_wd, ymn // tile_ht), axis=-1))
    xmn, ymn =  xmn[idxz], ymn[idxz]
    xmx = (xmn + tile_wd).clip(Xmin, Xmax)
    ymx = (ymn + tile_ht).clip(Ymin, Ymax)
    bboxes = np.stack((xmn, ymn, xmx, ymx), axis=-1)
    if num_workers == 1:
        out_spec = _write_downsample_tensorstore(ds_spec, tgt_spec, bboxes, **kwargs)
    else:
        n_tiles = xmn.size
        num_tile_per_job = max(1, n_tiles // num_workers)
        if max_tile_per_job is not None:
            num_tile_per_job = min(num_tile_per_job, max_tile_per_job)
        N_jobs = round(n_tiles / num_tile_per_job)
        indices = np.round(np.linspace(0, n_tiles, num=N_jobs+1, endpoint=True))
        indices = np.unique(indices).astype(np.uint32)
        jobs = []
        out_spec = None
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=get_context('spawn')) as executor:
            for idx0, idx1 in zip(indices[:-1], indices[1:]):
                idx0, idx1 = int(idx0), int(idx1)
                bbox_t = bboxes[idx0:idx1]
                job = executor.submit(_write_downsample_tensorstore, ds_spec, tgt_spec, bbox_t, **kwargs)
                jobs.append(job)
            for job in as_completed(jobs):
                out_spec = job.result()
    mipmaps.update({mip: out_spec})
    if write_to_file:
        kv_headers = ('gs://', 'http://', 'https://', 'file://', 'memory://', 's3://')
        for kvh in kv_headers:
            if metafile.startswith(kvh):
                break
        else:
            metafile = 'file://' + metafile
        meta_ts = ts.open({"driver": "json", "kvstore": metafile}).result()
        meta_ts.write(mipmaps).result()
    return mipmaps


def generate_tensorstore_scales(metafile, mips, **kwargs):
    logger_info = kwargs.pop('logger', None)
    t0 = time.time()
    logger = logging.get_logger(logger_info)
    mips = np.sort(mips)
    sec_name = os.path.basename(metafile).replace('.json', '')
    updated = False
    for mip in mips:
        specs = generate_target_tensorstore_scale(metafile, mip=mip, **kwargs)
        if specs is not None:
            updated = True
    if updated:
        logger.info(f'{sec_name}: {(time.time()-t0)/60} min')
    return {sec_name: updated}


def _write_downsample_tensorstore(src_spec, tgt_spec, bboxes, **kwargs):
    src_loader = dal.TensorStoreLoader.from_json_spec(src_spec, **kwargs)
    tgt_spec.update({'open': True, 'create': True, 'delete_existing': False})
    ts_out = ts.open(tgt_spec).result()
    for bbox in bboxes:
        img = src_loader.crop(bbox, return_empty=False, **kwargs)
        if img is None:
            continue
        xmin, ymin, xmax, ymax = bbox
        out_view = ts_out[xmin:xmax, ymin:ymax]
        img = np.swapaxes(img, 0, 1)
        try:
            out_view.write(img.reshape(out_view.shape)).result(timeout=TS_TIMEOUT)
        except TimeoutError:
            ts_out = ts.open(tgt_spec).result(timeout=TS_TIMEOUT)
            out_view = ts_out[xmin:xmax, ymin:ymax]
            out_view.write(img.reshape(out_view.shape)).result(timeout=TS_TIMEOUT)
    return ts_out.spec(minimal_spec=True).to_json()


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


def _smooth_filter(img, blur=2, sigma=0.0):
    if sigma > 0:
        img = common.masked_dog_filter(img, sigma=sigma, signed=False)
    if blur > 2:
        if blur % 2 == 1:
            img = cv2.blur(img, (round(blur), round(blur)))
        else:
            img = cv2.blur(img, (round(blur-1), round(blur-1)))
    return img
