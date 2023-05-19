import cv2
import glob
import numpy as np
from functools import partial
from scipy.interpolate import interp1d
import shapely.geometry as shpgeo
from shapely.ops import unary_union
import os

from feabas.dal import MosaicLoader
from feabas import common, logging, dal
from feabas.spatial import Geometry
from feabas.mesh import Mesh
from feabas.renderer import render_whole_mesh, MeshRenderer


def _get_image_loader(src_dir, **kwargs):
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
        pattern0 = pattern.replace('{', '({').replace('}', '}\d+)')
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
    kwargs.setdefault('remap_interp', cv2.INTER_AREA)
    kwargs.setdefault('cache_type', 'fifo')
    kwargs.setdefault('cache_size', downsample + 2)
    if (kwargs['remap_interp'] != cv2.INTER_NEAREST) and (downsample > 2):
        kwargs.setdefault('preprocess', partial(_smooth_filter, blur=downsample, sigma=0.0))
    logger = logging.get_logger(logger_info)
    out_meta_file = os.path.join(out_dir, 'metadata.txt')
    if os.path.isfile(out_meta_file):
        n_img = len(glob.glob(out_dir, '*.'+ext_out))
        return n_img
    rendered = {}
    try:
        image_loader = _get_image_loader(src_dir, pattern=pattern, tile_size=tile_size, **kwargs)
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
    image_loader = _get_image_loader(src_dir, **kwargs)
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
    scale = 255 / trials[np.argmax(entropy)]
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
    bin_edges = np.linspace(img.min(), img.max(), num=hist_num, endpoint=True)
    edge_cntrs = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist, _ = np.histogram(data0, bins=bin_edges)
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
    idx_mx = np.argmax(entropy, axis=None)
    idx0, idx1 = np.unravel_index(idx_mx, entropy.shape)
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