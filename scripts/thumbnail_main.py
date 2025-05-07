import argparse
from functools import partial
import json
import os
import time

from feabas.concurrent import submit_to_workers
from feabas import config, logging, storage


def generate_stitched_mipmaps(img_dir, max_mip, **kwargs):
    min_mip = kwargs.pop('min_mip', 0)
    num_workers = kwargs.pop('num_workers', 1)
    parallel_within_section = kwargs.pop('parallel_within_section', True)
    logger_info = kwargs.get('logger', None)
    logger = logging.get_logger(logger_info)
    meta_list = sorted(storage.list_folder_content(storage.join_paths(img_dir, 'mip'+str(min_mip), '**', 'metadata.txt'), recursive=True))
    meta_list = meta_list[arg_indx]
    secnames = [os.path.basename(os.path.dirname(s)) for s in meta_list]
    update_status = {}
    if parallel_within_section or (num_workers == 1):
        for sname in secnames:
            res = mip_map_one_section(sname, img_dir, max_mip, num_workers=num_workers, **kwargs)
            update_status.update(res)
    else:
        target_func = partial(mip_map_one_section, img_dir=img_dir,
                                max_mip=max_mip, num_workers=1, **kwargs)
        for res in submit_to_workers(target_func, args=[(s,) for s in secnames], num_workers=num_workers):
            update_status.update(res)
    logger.info('mipmapping generated.')
    return update_status


def generate_stitched_mipmaps_tensorstore(meta_dir, tgt_mips, **kwargs):
    num_workers = kwargs.pop('num_workers', 1)
    parallel_within_section = kwargs.pop('parallel_within_section', True)
    mask_dir = kwargs.pop('mask_dir', None)
    logger_info = kwargs.get('logger', None)
    logger = logging.get_logger(logger_info)
    meta_list = sorted(storage.list_folder_content(storage.join_paths(meta_dir,'*.json')))
    meta_list = meta_list[arg_indx]
    update_status = {}
    if parallel_within_section or num_workers == 1:
        for metafile in meta_list:
            mask_file = storage.join_paths(mask_dir, os.path.basename(metafile).replace('.json','.png'))
            s = mipmap.generate_tensorstore_scales(metafile, tgt_mips, num_workers=num_workers, mask_file=mask_file, **kwargs)
            update_status.update(s)
    else:
        target_func = partial(mipmap.generate_tensorstore_scales, mips=tgt_mips, **kwargs)
        kwarg_list = [{'mask_file': storage.join_paths(mask_dir, os.path.basename(s).replace('.json','.png'))} for s in meta_list]
        for res in submit_to_workers(target_func, args=[(s,) for s in meta_list], kwargs=kwarg_list, num_workers=num_workers):
            update_status.update(res)
    logger.info('mipmapping generated.')
    return update_status


def generate_thumbnails(src_dir, out_dir, seclist=None, **kwargs):
    num_workers = kwargs.pop('num_workers', 1)
    logger_info = kwargs.pop('logger', None)
    logger = logging.get_logger(logger_info)
    meta_list = sorted(storage.list_folder_content(storage.join_paths(src_dir, '**', 'metadata.txt'), recursive=True))
    meta_list = meta_list[arg_indx]
    secnames = [os.path.basename(os.path.dirname(s)) for s in meta_list]
    target_func = partial(mipmap.create_thumbnail, **kwargs)
    storage.makedirs(out_dir)
    updated = {}
    args_list = []
    kwargs_list = []
    for sname in secnames:
        outname = storage.join_paths(out_dir, sname + '.png')
        if seclist is None:
            if storage.file_exists(outname, use_cache=True):
                continue
            else:
                updated[sname] = True
        else:
            if sname not in seclist:
                continue
            elif (seclist[sname]) or (not storage.file_exists(outname, use_cache=True)):
                updated[sname] = True
            else:
                updated[sname] = False
                continue
        sdir = storage.join_paths(src_dir, sname)
        args_list.append((sdir,))
        kwargs_list.append({'outname': outname})
    for _ in submit_to_workers(target_func, args=args_list, kwargs=kwargs_list, num_workers=num_workers):
        pass         
    logger.info('thumbnails generated.')
    return updated


def generate_thumbnails_tensorstore(src_dir, out_dir, seclist=None, **kwargs):
    num_workers = kwargs.pop('num_workers', 1)
    logger_info = kwargs.pop('logger', None)
    logger = logging.get_logger(logger_info)
    meta_list = sorted(storage.list_folder_content(storage.join_paths(src_dir, '*.json')))
    meta_list = meta_list[arg_indx]
    target_func = partial(mipmap.create_thumbnail_tensorstore, **kwargs)
    storage.makedirs(out_dir)
    updated = {}
    args_list = []
    kwargs_list = []
    for meta_name in meta_list:
        sname = os.path.basename(meta_name).replace('.json', '')
        outname = storage.join_paths(out_dir, sname + '.png')
        if seclist is None:
            if storage.file_exists(outname, use_cache=True):
                continue
            else:
                updated[sname] = True
        else:
            if sname not in seclist:
                continue
            elif (seclist[sname]) or (not storage.file_exists(outname, use_cache=True)):
                updated[sname] = True
            else:
                updated[sname] = False
                continue
        args_list.append((meta_name,))
        kwargs_list.append({'outname': outname})
    for _ in submit_to_workers(target_func, args=args_list, kwargs=kwargs_list, num_workers=num_workers):
        pass
    logger.info('thumbnails generated.')
    return updated


def save_mask_for_one_sections(mesh_file, out_name, resolution, **kwargs):
    from feabas.stitcher import MontageRenderer
    import numpy as np
    from feabas import common
    img_dir = kwargs.get('img_dir', None)
    fillval = kwargs.get('fillval', 0)
    mask_erode = kwargs.get('mask_erode', 0)
    rndr = MontageRenderer.from_h5(mesh_file)
    roi_mask = rndr.generate_roi_mask(resolution, mask_erode=mask_erode)
    material_table = config.material_table()
    lbl_d = material_table['default'].mask_label
    lbl_e = material_table['exclude'].mask_label
    img = np.full_like(roi_mask, lbl_e)
    img[roi_mask > 0] = lbl_d
    common.imwrite(out_name, img)
    if img_dir is not None:
        thumb_name = storage.join_paths(img_dir, os.path.basename(out_name))
        if storage.file_exists(thumb_name):
            thumb = common.imread(thumb_name)
            if (thumb.shape[0] != img.shape[0]) or (thumb.shape[1] != img.shape[1]):
                thumb_out_shape = (*img.shape, *thumb.shape[2:])
                thumb_out = np.full_like(thumb, fillval, shape=thumb_out_shape)
                mn_shp = np.minimum(thumb_out.shape[:2], thumb.shape[:2])
                thumb_out[:mn_shp[0], :mn_shp[1], ...] = thumb[:mn_shp[0], :mn_shp[1], ...]
                common.imwrite(thumb_name, thumb_out)


def generate_thumbnail_masks(mesh_dir, out_dir, seclist=None, **kwargs):
    num_workers = kwargs.get('num_workers', 1)
    resolution = kwargs.get('resolution')
    img_dir = kwargs.get('img_dir', None)
    fillval = kwargs.get('fillval', 0)
    mask_erode = kwargs.get('mask_erode', 0)
    logger_info = kwargs.get('logger', None)
    logger= logging.get_logger(logger_info)
    mesh_list = sorted(storage.list_folder_content(storage.join_paths(mesh_dir, '*.h5')))
    mesh_list = mesh_list[arg_indx]
    target_func = partial(save_mask_for_one_sections, resolution=resolution, img_dir=img_dir,
                          fillval=fillval, mask_erode=mask_erode)
    storage.makedirs(out_dir)
    args_list = []
    for mname in mesh_list:
        sname = os.path.basename(mname).replace('.h5', '')
        outname = storage.join_paths(out_dir, sname + '.png')
        if seclist is None:
            if storage.file_exists(outname, use_cache=True):
                continue
        else:
            if sname not in seclist:
                continue
            elif (not seclist[sname]) and (storage.file_exists(outname, use_cache=True)):
                continue
        args_list.append((mname, outname))
    for _ in submit_to_workers(target_func, args=args_list, num_workers=num_workers):
        pass
    logger.info('thumbnail masks generated.')


def align_thumbnail_pairs(pairnames, image_dir, out_dir, **kwargs):
    import cv2
    import numpy as np
    from feabas import caching, thumbnail, common
    material_mask_dir = kwargs.pop('material_mask_dir', None)
    region_mask_dir = kwargs.pop('region_mask_dir', None)
    region_labels = kwargs.pop('region_labels', None)
    match_name_delimiter = kwargs.pop('match_name_delimiter', '__to__')
    cache_size = kwargs.pop('cache_size', 3)
    feature_match_settings = kwargs.get('feature_matching', {})
    logger_info = kwargs.get('logger', None)
    logger = logging.get_logger(logger_info)
    prepared_cache = caching.CacheFIFO(maxlen=cache_size)
    if region_labels is None:
        material_table = config.material_table()
        default_mat = material_table['default']
        region_labels = [default_mat.mask_label]
    for pname in pairnames:
        try:
            sname0_ext, sname1_ext = pname
            sname0 = os.path.splitext(sname0_ext)[0]
            sname1 = os.path.splitext(sname1_ext)[0]
            outname = storage.join_paths(out_dir, sname0 + match_name_delimiter + sname1 + '.h5')
            if storage.file_exists(outname, use_cache=True):
                continue
            if sname0 in prepared_cache:
                minfo0 = prepared_cache[sname0]
            else:
                img0 = common.imread(storage.join_paths(image_dir, sname0_ext))
                if (region_mask_dir is not None) and storage.file_exists(storage.join_paths(region_mask_dir, sname0_ext)):
                    mask0 = common.imread(storage.join_paths(region_mask_dir, sname0_ext))
                elif (material_mask_dir is not None) and storage.file_exists(storage.join_paths(material_mask_dir, sname0_ext)):
                    mask_t = common.imread(storage.join_paths(material_mask_dir, sname0_ext))
                    mask_t = np.isin(mask_t, region_labels).astype(np.uint8)
                    _, mask0 = cv2.connectedComponents(mask_t, connectivity=4, ltype=cv2.CV_16U)
                else:
                    mask0 = None
                if hasattr(mask0, 'shape') and ((mask0.shape[0] != img0.shape[0]) or (mask0.shape[1] != img0.shape[1])):
                    ht0 = min(mask0.shape[0], img0.shape[0])
                    wd0 = min(mask0.shape[1], img0.shape[1])
                    mask_t = np.zeros_like(mask0, shape=img0.shape)
                    mask_t[:ht0, :wd0] = mask0[:ht0, :wd0]
                    mask0 = mask_t
                minfo0 = thumbnail.prepare_image(img0, mask=mask0, **feature_match_settings)
                prepared_cache[sname0] = minfo0
            if sname1 in prepared_cache:
                minfo1 = prepared_cache[sname1]
            else:
                img1 = common.imread(storage.join_paths(image_dir, sname1_ext))
                if (region_mask_dir is not None) and storage.file_exists(storage.join_paths(region_mask_dir, sname1_ext)):
                    mask1 = common.imread(storage.join_paths(region_mask_dir, sname1_ext))
                elif (material_mask_dir is not None) and storage.file_exists(storage.join_paths(material_mask_dir, sname1_ext)):
                    mask_t = common.imread(storage.join_paths(material_mask_dir, sname1_ext))
                    mask_t = np.isin(mask_t, region_labels).astype(np.uint8)
                    _, mask1 = cv2.connectedComponents(mask_t, connectivity=4, ltype=cv2.CV_16U)
                else:
                    mask1 = None
                if hasattr(mask1, 'shape') and ((mask1.shape[0] != img1.shape[0]) or (mask1.shape[1] != img1.shape[1])):
                    ht1 = min(mask1.shape[0], img1.shape[0])
                    wd1 = min(mask1.shape[1], img1.shape[1])
                    mask_t = np.zeros_like(mask1, shape=img1.shape)
                    mask_t[:ht1, :wd1] = mask1[:ht1, :wd1]
                    mask1 = mask_t
                minfo1 = thumbnail.prepare_image(img1, mask=mask1, **feature_match_settings)
                prepared_cache[sname1] = minfo1
            thumbnail.align_two_thumbnails(minfo0, minfo1, outname, **kwargs)
        except Exception as err:
            logger.error(f'{pname}: error {err}')


def generate_mesh_from_mask(secname, **kwargs):
    from feabas import mesh, storage, common
    mask_name = kwargs.pop('mask_name', None)
    mask_size = kwargs.pop('mask_size', None)
    out_dir = kwargs.pop('out_dir', None)
    logger_info = kwargs.pop('logger', None)
    logger = logging.get_logger(logger_info)
    kwargs.setdefault('name', secname)
    if (mask_name is not None) and storage.file_exists(mask_name):
        M = mesh.mesh_from_mask(mask_name, **kwargs)
    elif mask_size is not None:
        if isinstance(mask_size, str):
            img = common.imread(mask_size)
            mask_size = img.shape
        ymax, xmax = mask_size[0], mask_size[1]
        M = mesh.Mesh.from_polygon_equilateral((0, 0, xmax, ymax), **kwargs)
    else:
        logger.error(f'{secname}: mask not found for meshing')
        return False
    if out_dir is None:
        return M
    else:
        outname = storage.join_paths(out_dir, secname+'.h5')
        M.save_to_h5(outname, save_material=True)
        return True


def normalize_transforms(tlist, angle=0.0, offset=(0,0), **kwargs):
    num_workers = kwargs.get('num_workers', 1)
    resolution = kwargs.get('resolution', thumbnail_resolution)
    if (angle == 0) and (offset is None):
        modify_tform = False
    else:
        modify_tform = True
    rfunc = partial(get_convex_hull, resolution=resolution)
    regions = shapely.Polygon()
    for wkb in submit_to_workers(rfunc, args=[(s,) for s in tlist], kwargs=[{'wkb': True}], num_workers=num_workers):
        R = shapely.from_wkb(wkb)
        regions = regions.union(R)
    if angle is None:
        theta = find_rotation_for_minimum_rectangle(regions)
    else:
        theta = angle * np.pi / 180
    corner_xy = np.array(regions.boundary.coords)
    Rt = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    R = np.eye(3)
    R[:2,:2] = Rt
    corner_txy = corner_xy @ Rt
    corner_min = np.min(corner_txy, axis=0)
    corner_max = np.max(corner_txy, axis=0)
    if offset is None:
        centr = np.array(regions.centroid.coords).ravel()
        txy = centr - centr @ Rt
    else:
        txy = np.array(offset).ravel() - corner_min
    xy_max = np.ceil((corner_max + txy) + (corner_min + txy).clip(0, None))
    bbox_out = (0, 0, int(xy_max[0]), int(xy_max[1]))
    if modify_tform:
        tfunc = partial(apply_transform_normalization, out_dir=None, R=R, txy=txy, resolution=resolution)
        for _ in submit_to_workers(tfunc, args=[(s,) for s in tlist], num_workers=num_workers):
            pass
    return bbox_out


def render_one_thumbnail(tform_name, thumbnail_dir, out_dir, **kwargs):
    import cv2
    from feabas.mesh import Mesh
    from feabas.renderer import MeshRenderer
    from feabas import common
    from feabas.dal import StreamLoader
    src_resolution = kwargs.get('src_resolution', None)
    if src_resolution is None:
        thumbnail_configs = config.thumbnail_configs()
        thumbnail_mip_lvl = thumbnail_configs.get('thumbnail_mip_level', 6)
        src_resolution = config.montage_resolution() * (2 ** thumbnail_mip_lvl)
    out_resolution = kwargs.get('out_resolution', src_resolution)
    bbox = kwargs.get('bbox', None)
    logger_info = kwargs.get('logger', None)
    logger = logging.get_logger(logger_info)
    t0 = time.time()
    secname = os.path.basename(tform_name).replace('.h5', '.png')
    thumbnail_name = storage.join_paths(thumbnail_dir, secname)
    outname = storage.join_paths(out_dir, secname)
    if storage.file_exists(outname):
        return
    M = Mesh.from_h5(tform_name)
    M.change_resolution(out_resolution)
    img = common.imread(thumbnail_name)
    if out_resolution > src_resolution:
        scale = src_resolution / out_resolution
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    elif out_resolution < src_resolution:
        raise RuntimeError('not sufficient thumbnail resolution.')
    renderer = MeshRenderer.from_mesh(M)
    renderer.link_image_loader(StreamLoader(img, resolution=out_resolution, fillval=0))
    if bbox is None:
        bbox = M.bbox(gear=constant.MESH_GEAR_MOVING, offsetting=True)
        bbox[:2] = 0
    imgt = renderer.crop(bbox, remap_interp=cv2.INTER_LANCZOS4)
    common.imwrite(outname, imgt)
    logger.debug(f'{outname}: {time.time()-t0} sec')


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Align thumbnails")
    parser.add_argument("--mode", metavar="mode", type=str, default='match')
    parser.add_argument("--start", metavar="start", type=int, default=0)
    parser.add_argument("--step", metavar="step", type=int, default=1)
    parser.add_argument("--stop", metavar="stop", type=int, default=0)
    parser.add_argument("--reverse",  action='store_true')
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args()

    root_dir = config.get_work_dir()

    thumbnail_configs = config.thumbnail_configs()
    thumbnail_mip_lvl = thumbnail_configs.get('thumbnail_mip_level', 6)
    thumbnail_resolution = config.thumbnail_resolution()
    if args.mode.lower().startswith('d'):
        thumbnail_configs = thumbnail_configs['downsample']
        mode = 'downsample'
    else:
        thumbnail_configs = thumbnail_configs['alignment']
        if args.mode.lower().startswith('a'):
            mode = 'alignment'
        elif args.mode.lower().startswith('m'):
            mode = 'matching'
        elif args.mode.lower().startswith('o'):
            mode = 'optimization'
        elif args.mode.lower().startswith('r'):
            mode = 'render'
        else:
            raise ValueError(f'{args.mode} not supported mode.')

    num_workers = thumbnail_configs.get('num_workers', 1)
    num_workers = config.set_numpy_thread_from_num_workers(num_workers)
    thumbnail_configs['num_workers'] = num_workers


    from feabas import mipmap, common, constant
    from feabas.aligner import apply_transform_normalization, get_convex_hull, Aligner
    from feabas.mipmap import mip_map_one_section
    from feabas.spatial import find_rotation_for_minimum_rectangle
    import numpy as np
    import shapely

    stt_idx, stp_idx, step = args.start, args.stop, args.step
    if stp_idx == 0:
        stp_idx = None
    if args.reverse:
        if stt_idx == 0:
            stt_idx = None
        arg_indx = slice(stp_idx, stt_idx, -step)
    else:
        arg_indx = slice(stt_idx, stp_idx, step)

    tdriver, root_dir = storage.parse_file_driver(root_dir)
    thumbnail_dir = storage.join_paths(root_dir, 'thumbnail_align')
    match_filename = storage.join_paths(thumbnail_dir, 'match_name.txt')
    stitch_tform_dir = storage.join_paths(root_dir, 'stitch', 'tform')
    img_dir = storage.join_paths(thumbnail_dir, 'thumbnails')
    mat_mask_dir = storage.join_paths(thumbnail_dir, 'material_masks')
    reg_mask_dir = storage.join_paths(thumbnail_dir, 'region_masks')
    manual_dir = storage.join_paths(thumbnail_dir, 'manual_matches')
    match_dir = storage.join_paths(thumbnail_dir, 'matches')
    feature_match_dir = storage.join_paths(thumbnail_dir, 'feature_matches')
    tform_dir = storage.join_paths(thumbnail_dir, 'tform')
    canvas_file = storage.join_paths(tform_dir, 'canvas.json')
    render_prefix = storage.join_paths(thumbnail_dir, 'aligned_thumbnails_')
    section_order_file = storage.join_paths(root_dir, 'section_order.txt')
    chunk_map_file = storage.join_paths(thumbnail_dir, 'chunk_map.json')
    if mode == 'downsample':
        logger_info = logging.initialize_main_logger(logger_name='stitch_mipmap', mp=num_workers>1)
        thumbnail_configs['logger'] = logger_info[0]
        logger= logging.get_logger(logger_info[0])
        align_mip = config.align_configs()['matching']['working_mip_level']
        stitch_conf = config.stitch_configs()['rendering']
        driver = stitch_conf.get('driver', 'image')
        if driver == 'image':
            max_mip = thumbnail_configs.pop('max_mip', max(0, thumbnail_mip_lvl-1))
            max_mip = max(align_mip, max_mip)
            src_dir0 = config.stitch_render_dir()
            pattern = stitch_conf['filename_settings']['pattern']
            one_based = stitch_conf['filename_settings']['one_based']
            fillval = stitch_conf['loader_settings'].get('fillval', 0)
            thumbnail_configs.setdefault('pattern', pattern)
            thumbnail_configs.setdefault('one_based', one_based)
            thumbnail_configs.setdefault('fillval', fillval)
            slist = generate_stitched_mipmaps(src_dir0, max_mip, **thumbnail_configs)
            if thumbnail_configs.get('thumbnail_highpass', True):
                src_mip = max(0, thumbnail_mip_lvl-2)
                highpass_inter_mip_lvl = thumbnail_configs.get('highpass_inter_mip_lvl', src_mip)
                assert highpass_inter_mip_lvl < thumbnail_mip_lvl
                src_dir = storage.join_paths(src_dir0, 'mip'+str(highpass_inter_mip_lvl))
                downsample = 2 ** (thumbnail_mip_lvl - highpass_inter_mip_lvl)
                if downsample >= 4:
                    highpass = True
                else:
                    highpass = False
            else:
                src_mip = max(0, thumbnail_mip_lvl-1)
                src_dir = storage.join_paths(src_dir0, 'mip'+str(src_mip))
                downsample = 2 ** (thumbnail_mip_lvl - src_mip)
                highpass = False
            thumbnail_configs.setdefault('downsample', downsample)
            thumbnail_configs.setdefault('highpass', highpass)
            slist = generate_thumbnails(src_dir, img_dir, seclist=slist, **thumbnail_configs)
        else:
            stitch_dir = storage.join_paths(root_dir, 'stitch')
            src_dir = storage.join_paths(stitch_dir, 'ts_specs')
            render_mask_dir = storage.join_paths(src_dir, 'masks')
            tgt_mips = [align_mip]
            if thumbnail_configs.get('thumbnail_highpass', True):
                highpass_inter_mip_lvl = thumbnail_configs.pop('highpass_inter_mip_lvl', max(0, thumbnail_mip_lvl-2))
                assert highpass_inter_mip_lvl < thumbnail_mip_lvl
                downsample = 2 ** (thumbnail_mip_lvl - highpass_inter_mip_lvl)
                if downsample >= 4:
                    highpass = True
                    thumbnail_configs.setdefault('highpass_inter_mip_lvl', highpass_inter_mip_lvl)
                    tgt_mips.append(highpass_inter_mip_lvl)
                else:
                    highpass = False
                    tgt_mips.append(thumbnail_mip_lvl)
                    downsample = 1
            else:
                highpass = False
                tgt_mips.append(thumbnail_mip_lvl)
                downsample = 1
            slist = generate_stitched_mipmaps_tensorstore(src_dir, tgt_mips, mask_dir=render_mask_dir, **thumbnail_configs)
            thumbnail_configs.setdefault('highpass', highpass)
            thumbnail_configs.setdefault('mip', thumbnail_mip_lvl)
            slist = generate_thumbnails_tensorstore(src_dir, img_dir, seclist=slist, **thumbnail_configs)
        generate_thumbnail_masks(stitch_tform_dir, mat_mask_dir, seclist=slist, resolution=thumbnail_resolution,
                                 img_dir=img_dir, **thumbnail_configs)
        logger.info('finished.')
        logging.terminate_logger(*logger_info)
    else:
        imglist = sorted(storage.list_folder_content(storage.join_paths(img_dir, '*.png')))
        imglist = common.rearrange_section_order(imglist, section_order_file)[0]
        bname_list = [os.path.basename(s) for s in imglist]
        secname_list = [os.path.splitext(s)[0] for s in bname_list]
        logger_info = logging.initialize_main_logger(logger_name='thumbnail_align', mp=num_workers>1)
        logger= logging.get_logger(logger_info[0])
        match_name_delimiter = thumbnail_configs.get('match_name_delimiter', '__to__')
        material_table = config.material_table()
        if (mode == 'matching') or (mode == 'alignment'):
            storage.makedirs(match_dir)
            storage.makedirs(manual_dir)
            thumbnail_configs['logger'] = logger_info[0]
            thumbnail_configs.setdefault('resolution', thumbnail_resolution)
            thumbnail_configs.setdefault('feature_match_dir', feature_match_dir)
            region_labels = []
            for _, mat in material_table:
                if mat.enable_mesh and (mat._stiffness_multiplier > 0.1) and (mat.mask_label is not None):
                    region_labels.append(mat.mask_label)
            thumbnail_configs.setdefault('region_labels', region_labels)
            pairnames = []
            processed = []
            if storage.file_exists(match_filename):
                with storage.File(match_filename, 'r') as f:
                    match_list0 = f.readlines()
                for s in match_list0:
                    s = s.strip()
                    s = s.replace('.h5', '').replace(match_name_delimiter, '\t')
                    snames = s.split('\t')
                    sname0, sname1 = snames[0], snames[1]
                    pairnames.append((sname0 + '.png', sname1 + '.png'))
                    outname = storage.join_paths(match_dir, sname0 + match_name_delimiter + sname1 + '.h5')
                    if storage.file_exists(outname, use_cache=True):
                        processed.append(True)
                    else:
                        processed.append(False)
            else:
                compare_distance = thumbnail_configs.pop('compare_distance', 1)
                if not hasattr(compare_distance, '__iter__'):
                    compare_distance = range(1, compare_distance+1)
                for stp in compare_distance:
                    for k in range(len(bname_list)-stp):
                        sname0_ext = bname_list[k]
                        sname1_ext = bname_list[k+stp]
                        sname0 = os.path.splitext(sname0_ext)[0]
                        sname1 = os.path.splitext(sname1_ext)[0]
                        outname = storage.join_paths(match_dir, sname0 + match_name_delimiter + sname1 + '.h5')
                        if storage.file_exists(outname, use_cache=True):
                            processed.append(True)
                        else:
                            processed.append(False)
                        pairnames.append((sname0_ext, sname1_ext))
            if len(pairnames) == len(pairnames[arg_indx]):
                pairnames = [s for p, s in zip(processed, pairnames) if not p]
            pairnames.sort()
            pairnames = pairnames[arg_indx]
            target_func = partial(align_thumbnail_pairs, image_dir=img_dir, out_dir=match_dir,
                                material_mask_dir=mat_mask_dir, region_mask_dir=reg_mask_dir,
                                **thumbnail_configs)
            if (num_workers == 1) or (len(pairnames) <= 1):
                target_func(pairnames)
            else:
                num_workers = min(num_workers, len(pairnames))
                match_per_job = thumbnail_configs.pop('match_per_job', 15)
                Njobs = max(num_workers, len(pairnames) // match_per_job)
                indx_j = np.linspace(0, len(pairnames), num=Njobs+1, endpoint=True)
                indx_j = np.unique(np.round(indx_j).astype(np.int32))
                kwargs_list = []
                for idx0, idx1 in zip(indx_j[:-1], indx_j[1:]):
                    kwargs_list.append({'pairnames': pairnames[idx0:idx1]})
                for _ in submit_to_workers(target_func, kwargs=kwargs_list, num_workers=num_workers):
                    pass
        if (mode == 'optimization') or (mode == 'alignment'):
            tmp_mesh_dir = storage.join_paths(thumbnail_dir, 'mesh')
            storage.makedirs(tform_dir, exist_ok=True)
            storage.makedirs(tmp_mesh_dir, exist_ok=True)
            # meshing
            opt_configs = thumbnail_configs.get('optimization', {})
            mesh_configs = opt_configs.get('meshing_config', {})
            mfunc = partial(generate_mesh_from_mask, out_dir=tmp_mesh_dir, material_table=material_table.save_to_json(),
                            resolution=thumbnail_resolution, logger=logger_info[0], **mesh_configs)
            tasks = []
            for sname in secname_list:
                tform_name = storage.join_paths(tform_dir, sname+'.h5')
                mask_name = storage.join_paths(mat_mask_dir, sname+'.png')
                img_name = storage.join_paths(img_dir, sname+'.png')
                if storage.file_exists(tform_name, use_cache=True):
                    continue  
                if storage.file_exists(mask_name, use_cache=True):
                    tasks.append({'secname': sname, 'mask_name': mask_name})
                elif storage.file_exists(img_name, use_cache=True):
                    logger.warning(f'{sname} meshing: {mask_name} not found, use rectangular mesh.')
                    tasks.append({'secname': sname, 'mask_size': img_name})
                else:
                    logger.error(f'{sname} meshing: {mask_name} not found.')
            if len(tasks) > 0:
                logger.info('generating meshes for thumbnails')
                for _ in submit_to_workers(mfunc, kwargs=tasks, num_workers=num_workers):
                    pass
            # optimization
                logger.info('optimizing...')
                chunk_settings = opt_configs.get('chunk_settings', {'chunked_to_depth': 0})
                stack_config = opt_configs.get('stack_config', {})
                slide_window = opt_configs.get('slide_window', {})
                worker_settings = opt_configs.get('worker_settings', {})
                chunked_to_depth = chunk_settings.pop('chunked_to_depth', 0)
                chunk_settings.setdefault('match_name_delimiter', match_name_delimiter)
                chunk_settings.setdefault('section_list', secname_list)
                chunk_settings.setdefault('chunk_map', chunk_map_file)
                chunk_settings.setdefault('mip_level', thumbnail_mip_lvl)
                chunk_settings['logger'] = logger_info[0]
                algnr = Aligner(tmp_mesh_dir, tform_dir, match_dir, **chunk_settings)
                algnr.run(num_workers=num_workers, chunked_to_depth=chunked_to_depth,
                          stack_config=stack_config, slide_window=slide_window,
                          worker_settings=worker_settings)
        if (mode == 'render') or (mode == 'alignment'):
            render_configs = thumbnail_configs.get('render', {})
            render_scale = render_configs.get('scale', None)
            if render_scale is None:
                to_render = False
            else:
                to_render = True
                render_resolution = thumbnail_resolution / render_scale
                target_mip = thumbnail_mip_lvl - np.log2(render_scale)
                render_dir = render_prefix + str(int(render_resolution)) + 'nm'
                storage.makedirs(render_dir, exist_ok=True)
                tform_list0 = sorted(storage.list_folder_content(storage.join_paths(tform_dir, '*.h5')))
                tform_list = []
                for tname in tform_list0:
                    secname = os.path.basename(tname).replace('.h5', '.png')
                    outname = storage.join_paths(render_dir, secname)
                    if not storage.file_exists(outname, use_cache=True):
                        tform_list.append(tname)
                if storage.file_exists(canvas_file):
                    bbox_render = common.get_canvas_bbox(canvas_file, target_mip=target_mip)
                else:
                    angle = render_configs.get('rotation_angle', None)
                    offset = render_configs.get('bbox_offset', [0,0])
                    bbox_render = normalize_transforms(tform_list, angle=angle, offset=offset, num_workers=num_workers, resolution=render_resolution)
                    canvas_bbox = {f'mip{target_mip}': [int(s) for s in bbox_render]}
                    with storage.File(canvas_file, 'w') as f:
                        json.dump(canvas_bbox, f)
                rfunc = partial(render_one_thumbnail, thumbnail_dir=img_dir, out_dir=render_dir, src_resolution=thumbnail_resolution,
                                out_resolution=render_resolution, bbox=bbox_render, logger=logger_info[0])
                for _ in submit_to_workers(rfunc, args=[(s,) for s in tform_list], num_workers=num_workers):
                    pass
        logger.info('finished.')
        logging.terminate_logger(*logger_info)
