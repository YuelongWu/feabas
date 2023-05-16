import cv2
import glob
import numpy as np
from functools import partial
import shapely.geometry as shpgeo
from shapely.ops import unary_union
import os

from feabas.dal import MosaicLoader
from feabas import common, logging
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
    num_workers = kwargs.pop('num_workers', 1)
    ext_out = kwargs.pop('output_format', 'png')
    pattern = kwargs.get('pattern', '_tr{ROW_IND}-tc{COL_IND}.png')
    one_based = kwargs.get('one_based', True)
    tile_size = kwargs.get('tile_size', None)
    downsample = kwargs.pop('downsample', 2)
    logger_info = kwargs.get('logger', None)
    logger = logging.get_logger(logger_info)
    out_meta_file = os.path.join(out_dir, 'metadata.txt')
    if os.path.isfile(out_meta_file):
        n_img = len(glob.glob(out_dir, '*.'+ext_out))
        return n_img
    pattern = os.path.splitext(pattern)[0]
    rendered = {}
    try:
        image_loader = _get_image_loader(src_dir, **kwargs)
        if image_loader is None:
            return 0
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
        rendered = render_whole_mesh(M, image_loader, prefix, num_workers=num_workers,
                                    tile_size=tile_size, pattern=pattern+'.'+ext_out,
                                    scale= 1/downsample, one_based=one_based)
        with open(out_meta_file, 'w') as f:
            f.write(f'{{ROOT_DIR}}\t{out_root_dir}\n')
            fnames = sorted(list(rendered.keys()))
            for fname in fnames:
                bbox = rendered[fname]
                f.write(f'{fname}\t{bbox[0]}\t{bbox[1]}\t{bbox[2]}\t{bbox[3]}\n')
    except Exception as err:
        logger.error(f'{src_dir}: {err}')
        return None
    return len(rendered)


def create_thumbnail(src_dir, downsample=4, highpass=True, **kwargs):
    if highpass:
        kwargs['preprocess'] = partial(common.masked_dog_filter, sigma=1, signed=False)
        kwargs['dtype'] = np.float32
    else:
        kwargs['preprocess'] = None
    image_loader = _get_image_loader(src_dir, **kwargs)
    M = _mesh_from_image_loader(image_loader)
    M.change_resolution(image_loader.resolution * downsample)
    bbox = M.bbox()
    xmax, ymax = bbox[2], bbox[3]
    if highpass:
        rndr = MeshRenderer.from_mesh(M, fillval=0, dtype=np.float32, image_loaer=image_loader)

