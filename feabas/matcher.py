import cv2
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures import as_completed
from functools import partial
from multiprocessing import get_context
import numpy as np
from scipy import fft
from scipy.fftpack import next_fast_len

from feabas.mesh import Mesh
from feabas.renderer import MeshRenderer
from feabas import optimizer, dal, common, spatial
import feabas.constant as const



def xcorr_fft(img0, img1, conf_mode=const.FFT_CONF_MIRROR, **kwargs):
    """
    find the displacements between two image(-stack)s from the Fourier based
    cross-correlation.
    Args:
        img0(ndarray): the first image, N x H0 x W0.
        img1(ndarray): the second image, N x H1 x W1.
    Kwargs:
        sigma: if set to larger than 0, will apply masked Difference of Gassian
            filter to the images before cross-correlation
        mask0: mask for DoG filter for the first image.
        mask1: mask for DoG filter for the second image.
        normalize (bool): whether to normalize the cross-correlation. The inputs
            of the function are expected to be band-pass filtered, therefore
            normalization is not that necessary.
        pad (bool): whether to zero-pad the images so that the peak position is
            not ambiguous.
    Return:
        dx, dy: the displacement of the peak of the cross-correlation, so that
            the center of img1 + (dx, dy) corresponds to the center of img0.
        conf: the confidence value of the cross-correlation between 0 and 1.
    """
    sigma = kwargs.get('sigma', 0)
    mask0 = kwargs.get('mask0', None)
    mask1 = kwargs.get('mask1', None)
    normalize = kwargs.get('normalize', False)
    pad = kwargs.get('pad', True)
    if sigma > 0:
        img0 = common.masked_dog_filter(img0, sigma, mask=mask0)
        img1 = common.masked_dog_filter(img1, sigma, mask=mask1)
    imgshp0 = img0.shape[-2:]
    imgshp1 = img1.shape[-2:]
    if pad:
        fftshp = [next_fast_len(s0 + s1 - 1) for s0, s1 in zip(imgshp0, imgshp1)]
    else:
        fftshp = [next_fast_len(max(s0, s1)) for s0, s1 in zip(imgshp0, imgshp1)]
    F0 = fft.rfft2(img0, s=fftshp)
    F1 = fft.rfft2(img1, s=fftshp)
    C = fft.irfft2(np.conj(F0) * F1, s=fftshp)
    C = C.reshape(-1, np.prod(fftshp))
    if normalize:
        if mask0 is None:
            mask0 = np.ones_like(img0)
        if mask1 is None:
            mask1 = np.ones_like(img1)
        M0 = fft.rfft2(mask0, s=fftshp)
        M1 = fft.rfft2(mask1, s=fftshp)
        NC = fft.irfft2(np.conj(M0) * M1, s=fftshp)
        NC = NC.reshape(-1, np.prod(fftshp))
        NC = (NC / (NC.max(axis=-1, keepdims=True).clip(1, None))).clip(0.1, None)
        C = C / NC
    indx = np.argmax(C, axis=-1)
    dy, dx = np.unravel_index(indx, fftshp)
    dy = dy + (imgshp0[0] - imgshp1[0]) / 2
    dx = dx + (imgshp0[1] - imgshp1[1]) / 2
    dy = dy - np.round(dy / fftshp[0]) * fftshp[0]
    dx = dx - np.round(dx / fftshp[1]) * fftshp[1]
    if conf_mode == const.FFT_CONF_NONE:
        conf = np.ones_like(dx, dtype=np.float32)
    elif conf_mode == const.FFT_CONF_MIRROR:
        C_mirror = np.abs(fft.irfft2(F0 * F1, s=fftshp))
        C_mirror = C_mirror.reshape(-1, np.prod(fftshp))
        if normalize:
            NC = fft.irfft2(M0 * M1, s=fftshp)
            NC = NC.reshape(-1, np.prod(fftshp))
            NC = (NC / (NC.max(axis=-1, keepdims=True).clip(1, None))).clip(0.1, None)
            C_mirror = C_mirror / NC
        mx_rl = C.max(axis=-1)
        mx_mr = C_mirror.max(axis=-1)
        conf = np.zeros_like(dx, dtype=np.float32)
        conf[mx_rl>0] = 1 -  mx_mr[mx_rl>0]/mx_rl[mx_rl>0]
        conf = conf.clip(0, 1)
    elif conf_mode == const.FFT_CONF_STD:
        C_std = C.std(axis=-1)
        C_max = C.max(axis=-1)
        # assuming exponential distribution
        conf = (1 - np.exp(-C_max / C_std)) ** np.prod(fftshp)
        conf = conf.clip(0, 1)
    return dx, dy, conf


def global_translation_matcher(img0, img1, **kwargs):
    sigma = kwargs.get('sigma', 0.0)
    mask0 = kwargs.get('mask0', None)
    mask1 = kwargs.get('mask1', None)
    conf_mode = kwargs.get('conf_mode', const.FFT_CONF_MIRROR)
    conf_thresh = kwargs.get('conf_thresh', 0.3)
    divide_factor = kwargs.get('divide_factor', 6)
    if sigma > 0:
        img0 = common.masked_dog_filter(img0, sigma, mask=mask0)
        img1 = common.masked_dog_filter(img1, sigma, mask=mask1)
    tx, ty, conf = xcorr_fft(img0, img1, conf_mode=conf_mode, pad=True)
    tx, ty, conf = tx.item(), ty.item(), conf.item()
    if conf > conf_thresh:
        return tx, ty, conf
    # divide the image for another shot (avoid local artifacts)
    imgshp0 = img0.shape[-2:]
    imgshp1 = img1.shape[-2:]
    imgshp = np.minimum(imgshp0, imgshp1)
    # find the division that results in most moderate aspect ratio
    if hasattr(divide_factor, '__len__'):
        divide_N = divide_factor[:2]
    else:
        ratio0 = imgshp[0]/imgshp[1]
        ratio = np.inf
        for r0 in range(1, int(divide_factor**0.5) + 1):
            if divide_factor % r0 != 0:
                continue
            rr = np.abs(np.log(ratio0 * (r0 ** 2 / divide_factor)))
            if rr < ratio:
                ratio = rr
                divide_N = (int(divide_factor/r0), int(r0))
            rr = np.abs(np.log(ratio0 / (r0 ** 2 / divide_factor)))
            if rr < ratio:
                ratio = rr
                divide_N = (int(r0), int(divide_factor/r0))
    xmin0, ymin0, xmax0, ymax0 = common.divide_bbox((0, 0, imgshp0[1], imgshp0[0]), min_num_blocks=divide_N)
    xmin1, ymin1, xmax1, ymax1 = common.divide_bbox((0, 0, imgshp1[1], imgshp1[0]), min_num_blocks=divide_N)
    stack0 = []
    stack1 = []
    for k in range(xmin0.size):
        stack0.append(img0[ymin0[k]:ymax0[k], xmin0[k]:xmax0[k]])
        stack1.append(img1[ymin1[k]:ymax1[k], xmin1[k]:xmax1[k]])
    btx, bty, bconf = xcorr_fft(np.stack(stack0, axis=0), np.stack(stack1, axis=0), conf_mode=conf_mode, pad=True)
    btx = btx + (xmin1 + xmax1 - xmin0 - xmax0 + imgshp0[1] - imgshp1[1])/2
    bty = bty + (ymin1 + ymax1 - ymin0 - ymax0 + imgshp0[0] - imgshp1[0])/2
    k_best = np.argmax(bconf)
    if bconf[k_best] <= conf:
        tx = btx[k_best]
        ty = bty[k_best]
        conf = bconf[k_best]
    return tx, ty, conf


def stitching_matcher(img0, img1, **kwargs):
    """
    given two images with rectangular shape, return the displacement vectors on a
    grid of sample points. Mostly for stitching matching.
    """
    sigma = kwargs.get('sigma', 2.5)
    mask0 = kwargs.get('mask0', None)
    mask1 = kwargs.get('mask1', None)
    conf_mode = kwargs.get('conf_mode', const.FFT_CONF_MIRROR)
    conf_thresh = kwargs.get('conf_thresh', 0.3)
    err_thresh = kwargs.get('err_thresh', 5)
    opt_tol = kwargs.get('opt_tol', None)
    coarse_downsample = kwargs.get('coarse_downsample', 1)
    fine_downsample = kwargs.get('fine_downsample', 1)
    spacings = kwargs.get('spacings', None)
    min_num_blocks = kwargs.get('min_num_blocks', 2)
    if spacings is None:
        imgshp = np.minimum(img0.shape, img1.shape)
        smx = max(imgshp) * 0.25
        smn = max(min(75, min(imgshp)/3), 25)
        if smn > smx:
            spacings = np.array([smn])
        else:
            Nsp = max(1, round(np.log(smx/smn)/np.log(4)))
            spacings = np.exp(np.linspace(np.log(smn), np.log(smx), num=Nsp, endpoint=True))
    else:
        spacings = np.array(spacings, copy=False)
    if coarse_downsample != 1:
        img0_g = cv2.resize(img0, None, fx=coarse_downsample, fy=coarse_downsample, interpolation=cv2.INTER_AREA)
        img1_g = cv2.resize(img1, None, fx=coarse_downsample, fy=coarse_downsample, interpolation=cv2.INTER_AREA)
        if mask0 is not None:
            mask0_g = cv2.resize(mask0, None, fx=coarse_downsample, fy=coarse_downsample, interpolation=cv2.INTER_NEAREST)
        else:
            mask0_g = None
        if mask1 is not None:
            mask1_g = cv2.resize(mask1, None, fx=coarse_downsample, fy=coarse_downsample, interpolation=cv2.INTER_NEAREST)
        else:
            mask1_g = None
    else:
        img0_g = img0
        img1_g = img1
        mask0_g = mask0
        mask1_g = mask1
    if sigma > 0:
        img0_g = common.masked_dog_filter(img0_g, sigma*coarse_downsample, mask=mask0_g)
        img1_g = common.masked_dog_filter(img1_g, sigma*coarse_downsample, mask=mask1_g)
    tx0, ty0, conf0 = global_translation_matcher(img0_g, img1_g, conf_mode=conf_mode,
        conf_thresh=conf_thresh)
    if conf0 < conf_thresh:
        return conf_thresh, None, None, None
    if fine_downsample == coarse_downsample:
        img0_f = img0_g
        img1_f = img1_g
    else:
        if fine_downsample != 1:
            img0_f = cv2.resize(img0, None, fx=fine_downsample, fy=fine_downsample, interpolation=cv2.INTER_AREA)
            img1_f = cv2.resize(img1, None, fx=fine_downsample, fy=fine_downsample, interpolation=cv2.INTER_AREA)
            if mask0 is not None:
                mask0_f = cv2.resize(mask0, None, fx=fine_downsample, fy=fine_downsample, interpolation=cv2.INTER_NEAREST)
            else:
                mask0_f = None
            if mask1 is not None:
                mask1_f = cv2.resize(mask1, None, fx=fine_downsample, fy=fine_downsample, interpolation=cv2.INTER_NEAREST)
            else:
                mask1_f = None
        else:
            img0_f = img0
            img1_f = img1
            mask0_f = mask0
            mask1_f = mask1
        if sigma > 0:
            img0_f = common.masked_dog_filter(img0_f, sigma*fine_downsample, mask=mask0_f)
            img1_f = common.masked_dog_filter(img1_f, sigma*fine_downsample, mask=mask1_f)
    tx0 = tx0 * fine_downsample / coarse_downsample
    ty0 = ty0 * fine_downsample / coarse_downsample
    err_thresh = err_thresh * fine_downsample
    img_loader0 = dal.StreamLoader(img0_f, fillval=0)
    img_loader1 = dal.StreamLoader(img1_f, fillval=0)
    if np.any(spacings < 1):
        bbox0 = np.array(img_loader0.bounds) + np.tile((tx0, ty0), 2)
        bbox1 = img_loader1.bounds
        bbox, _ = common.intersect_bbox(bbox0, bbox1)
        wd0 = bbox[2] - bbox[0]
        ht0 = bbox[3] - bbox[1]
        lside = max(wd0, ht0)
        spacings[spacings < 1] *= lside
    spacings = spacings * fine_downsample
    min_spacing = np.min(spacings)
    mesh0 = Mesh.from_bbox(img_loader0.bounds, cartesian=True,
        mesh_size=min_spacing, min_num_blocks=min_num_blocks, uid=0)
    mesh1 = Mesh.from_bbox(img_loader1.bounds, cartesian=True,
        mesh_size=min_spacing, min_num_blocks=min_num_blocks, uid=1)
    mesh0.apply_translation((tx0, ty0), const.MESH_GEAR_FIXED)
    mesh0.lock()
    weight, xy0, xy1, strain = iterative_xcorr_matcher_w_mesh(mesh0, mesh1, img_loader0, img_loader1,
        conf_mode=conf_mode, conf_thresh=conf_thresh, err_method='huber', 
        err_thresh=err_thresh, opt_tol=opt_tol, spacings=spacings,
        distributor=const.BLOCKDIST_CART_BBOX, min_num_blocks=min_num_blocks)
    if (fine_downsample != 1) and (xy0 is not None):
        xy0 = spatial.scale_coordinates(xy0, 1/fine_downsample)
        xy1 = spatial.scale_coordinates(xy1, 1/fine_downsample)
    return weight, xy0, xy1, strain


def iterative_xcorr_matcher_w_mesh(mesh0, mesh1, image_loader0, image_loader1, spacings, **kwargs):
    """
    find the corresponding points by alternatively performing template matching
    and mesh relaxation.
    Args:
        mesh0, mesh1 (feabas.mesh.Mesh): the mesh objects of the two sections to
            be matched. Note that the matching is operated at the resolution of
            the meshes, so it is important to adjust the resoltion of the meshes
            to allow desired scaling for template matching.
        image_loader0, image_loader1 (feabas.dal.MosaicLoader or StreamLoader):
            image loaders of the sections.
    Kwargs:
        sigma(float): if larger than 0, the cropped images for template matching
            will be pre-processed with DoG filter with sigma.
        conf_mode: the method to compute the confidence value for template
            matching. check feabas.constant for options.
        conf_thresh: the threshold of template matching confidence values below
            which the match will be discarded.
        err_thresh: the threshold of the residue error distance between matching
            points after mesh relaxation. Matches larger than this will be cut.
        opt_tol: the stopping tolerance for the mesh relaxation steps.
        spacings(list): the distances between the neighboring sample blocks for
            template matcing. The function will follow a coarse to fine regiment
            by starting from the largest spacing and working towards the smaller
            ends. The largest displacement found in the template matching step
            will be used to determine the next spacing: it will select the
            smallest spacing that is larger than a multiple of the largest
            displacement. At the first scale, the spacing can go backwards (
            moving to a larger value); but after that it can only go smaller and
            can skip values.
        distributor: the method to distribute the center of the blocks for
            template-matching.
        min_num_blocks: parameter fed into the distributor function. will only
            be enforced at the smallest spacing.
        shrink_factor: parameter fed into the distributor function. Determines
            the template-matching block size compared to the spacing.
        render_mode: the rendering mode when generating the blocks for template-
            matching.
        allow_dwell(int): number of times each spacing settings can be repeated.
        compute_strain(bool): whether to caculate the largest strain in
            the final relaxed meshes. Could be an indicator of how "crazy" the
            matching points are
    Return:
        weight: weight of each mathing point pairs.
        xy0, xy1: xy coordinates of the matching points in the images before any
            transformations.
        strain: the largest strain of the mesh. 
    """
    num_workers = kwargs.get('num_workers', 1)
    conf_thresh = kwargs.get('conf_thresh', 0.3)
    err_method = kwargs.get('err_method', 'huber')
    err_thresh = kwargs.get('err_thresh', 0)
    opt_tol = kwargs.get('opt_tol', None)
    distributor = kwargs.get('distributor', const.BLOCKDIST_CART_BBOX)
    min_num_blocks = kwargs.get('min_num_blocks', 2)
    shrink_factor = kwargs.get('shrink_factor', 1)
    allow_dwell = kwargs.get('allow_dwell', 0)
    compute_strain = kwargs.get('compute_strain', True)
    batch_size = kwargs.pop('batch_size', None)
    if num_workers > 1 and batch_size is not None:
        batch_size = max(1, batch_size / num_workers)
        loader_dict0 = image_loader0.init_dict()
        loader_dict1 = image_loader1.init_dict()
    # if any spacing value smaller than 1, means they are relative to longer side
    spacings = np.array(spacings, copy=False)
    min_block_size_multiplier = 4
    strain = 0.0
    invalid_output = (0, None, None, strain)
    if np.any(spacings < 1):
        bbox0 = mesh0.bbox(gear=const.MESH_GEAR_MOVING)
        bbox1 = mesh1.bbox(gear=const.MESH_GEAR_MOVING)
        bbox, valid = common.intersect_bbox(bbox0, bbox1)
        if not valid:
            return invalid_output
        wd0 = bbox[2] - bbox[0]
        ht0 = bbox[3] - bbox[1]
        lside = max(wd0, ht0)
        spacings[spacings < 1] *= lside
    opt = optimizer.SLM([mesh0, mesh1])
    spacings = np.sort(spacings)[::-1]
    sp = np.max(spacings)
    sp_indx = 0
    initialized = False
    spacing_enlarged = False
    dwelled = 0
    while sp_indx < spacings.size:
        if sp == spacings[-1]:
            mnb = min_num_blocks
        else:
            mnb = 1
        if distributor == const.BLOCKDIST_CART_BBOX:
            bboxes0 = distributor_cartesian_bbox(mesh0, mesh1, sp,
                min_num_blocks=mnb, shrink_factor=shrink_factor, zorder=(num_workers>1))
        else:
            raise ValueError
        if bboxes0 is None:
            return invalid_output
        num_blocks = bboxes0.shape[0]
        if num_workers > 1:
            if batch_size is None:
                batch_size = max(1, num_blocks/num_workers)
            else:
                batch_size = min(max(1, num_blocks/num_workers), batch_size)
            num_batchs = int(np.ceil(num_blocks / batch_size))
            if len(num_batchs) == 1:
                xy0, xy1, conf = bboxes_mesh_renderer_matcher(mesh0, mesh1,
                    image_loader0, image_loader1, bboxes0,
                    batch_size=batch_size, pad=(not initialized), **kwargs)
            else:
                batch_indices = np.linspace(0, num_blocks, num=num_batchs+1, endpoint=True)
                batch_indices = np.unique(batch_indices.astype(np.int32))
                batched_bboxes = []
                batched_bboxes_union = []
                for bidx0, bidx1 in zip(batch_indices[:-1], batch_indices[1:]):
                    batched_bboxes.append(bboxes0[bidx0:bidx1])
                    batched_bboxes_union.append(common.bbox_union(bboxes0[bidx0:bidx1]))
                target_func = partial(bboxes_mesh_renderer_matcher, pad=(not initialized), **kwargs)
                submeshes0 = mesh0.submeshes_from_bboxes(batched_bboxes_union)
                submeshes1 = mesh1.submeshes_from_bboxes(batched_bboxes_union)
                jobs = []
                xy0 = []
                xy1 = []
                conf = []
                with ProcessPoolExecutor(max_workers=num_workers, mp_context=get_context('spawn')) as executor:
                    for m0_p, m1_p, bboxes_p in zip(submeshes0, submeshes1, batched_bboxes):
                        m0dict = m0_p.get_init_dict(vertex_flags=(const.MESH_GEAR_INITIAL, const.MESH_GEAR_MOVING))
                        m1dict = m1_p.get_init_dict(vertex_flags=(const.MESH_GEAR_INITIAL, const.MESH_GEAR_MOVING))
                        job = executor.submit(target_func, m0dict, m1dict, loader_dict0, loader_dict1, bboxes_p)
                        jobs.append(job)
                    for job in as_completed(jobs):
                        pt0, pt1, cnf = job.result()
                        xy0.append(pt0)
                        xy1.append(pt1)
                        conf.append(cnf)
                xy0 = np.concatenate(xy0, axis=0)
                xy1 = np.concatenate(xy1, axis=0)
                conf = np.concatenate(conf, axis=0)
        else:
            xy0, xy1, conf = bboxes_mesh_renderer_matcher(mesh0, mesh1,
                image_loader0, image_loader1, bboxes0,
                batch_size=batch_size, pad=(not initialized), **kwargs)
        if np.all(conf <= conf_thresh):
            if not initialized:
                return invalid_output
            else:
                break
        opt.clear_links()
        xy0 = xy0[conf > conf_thresh]
        xy1 = xy1[conf > conf_thresh]
        wt = conf[conf > conf_thresh]
        max_dis = np.max(np.sum((xy0 - xy1) ** 2, axis=-1)) ** 0.5
        if opt_tol is None:
            opt_tol_t = 0.01 / max(1, max_dis)
        else:
            opt_tol_t = opt_tol
        min_block_size = min_block_size_multiplier * max_dis
        next_pos = np.searchsorted(-spacings, -min_block_size) - 1
        if (not spacing_enlarged) and (next_pos < 0):
            spacing_enlarged = True
            sp = np.ceil(min_block_size)
            continue
        spacing_enlarged = True
        if next_pos > sp_indx:
            sp_indx = next_pos
            dwelled = 0
        elif dwelled >= allow_dwell:
            sp_indx += 1
            dwelled = 0
        else:
            dwelled += 1
        opt.add_link_from_coordinates(mesh0.uid, mesh1.uid, xy0, xy1,
                        gear=(const.MESH_GEAR_MOVING, const.MESH_GEAR_MOVING), weight=wt,
                        check_duplicates=False)
        if max_dis > 0.1:
            opt.optimize_linear(tol=opt_tol_t, batch_num_matches=np.inf)
            if err_thresh > 0:
                if err_method == 'huber':
                    opt.set_link_residue_huber(err_thresh)
                elif err_method == 'threshold':
                    opt.set_link_residue_threshold(err_thresh)
                else:
                    raise ValueError
                weight_modified, _ = opt.adjust_link_weight_by_residue()
                if weight_modified and (sp_indx < spacings.size):
                    opt.optimize_linear(tol=opt_tol_t, batch_num_matches=np.inf)
        initialized = True
        if sp_indx < spacings.size:
            sp = spacings[sp_indx]
    link = opt.links[0]
    # # Debug:
    # import matplotlib.pyplot as plt
    # bbox0 = mesh0.bbox(gear=MESH_GEAR_MOVING)
    # bbox1 = mesh1.bbox(gear=MESH_GEAR_MOVING)
    # bbox, valid = common.intersect_bbox(bbox0, bbox1)
    # render0 = MeshRenderer.from_mesh(mesh0, image_loader=image_loader0)
    # render1 = MeshRenderer.from_mesh(mesh1, image_loader=image_loader1)
    # img0t = render0.crop(bbox)
    # img1t = render1.crop(bbox)
    # imgt = np.stack((img0t, img1t, img0t), axis=-1)
    # plt.imshow(imgt/10)
    # plt.plot(link.xy0(gear=MESH_GEAR_MOVING, use_mask=True)[:,0] - bbox[0], link.xy0(gear=MESH_GEAR_MOVING,  use_mask=True)[:,1] - bbox[1], 'r.')
    # plt.plot(link.xy1(gear=MESH_GEAR_MOVING, use_mask=True)[:,0] - bbox[0], link.xy1(gear=MESH_GEAR_MOVING,  use_mask=True)[:,1] - bbox[1], 'g.')
    # plt.show()
    xy0 = link.xy0(gear=const.MESH_GEAR_INITIAL, use_mask=True, combine=True)
    xy1 = link.xy1(gear=const.MESH_GEAR_INITIAL, use_mask=True, combine=True)
    weight = link.weight(use_mask=True)
    if compute_strain:
        for m in opt.meshes:
            if not m.locked:
                ss = np.exp(np.abs(np.log(m.triangle_tform_svd().clip(1e-3, None)))) - 1
                smx = np.quantile(ss, 0.9, axis=None)
                strain = max(strain, smx)
    return weight, xy0, xy1, strain


def bboxes_mesh_renderer_matcher(mesh0, mesh1, image_loader0, image_loader1, bboxes, **kwargs):
    batch_size = kwargs.get('batch_size', None)
    sigma = kwargs.get('sigma', 0.0)
    render_mode = kwargs.get('render_mode', const.RENDER_FULL)
    conf_mode = kwargs.get('conf_mode', const.FFT_CONF_MIRROR)
    pad = kwargs.get('pad', True)
    if isinstance(mesh0, dict):
        mesh0 = Mesh(**mesh0)
    elif isinstance(mesh0, str):
        mesh0 = Mesh.from_h5(mesh0)
    if isinstance(mesh1, dict):
        mesh1 = Mesh(**mesh1)
    elif isinstance(mesh1, str):
        mesh1 = Mesh.from_h5(mesh1)
    if isinstance(image_loader0, (str, dict)):
        image_loader0 = dal.get_loader_from_json(image_loader0)
    if isinstance(image_loader1, (str, dict)):
        image_loader1 = dal.get_loader_from_json(image_loader1)
    num_blocks = bboxes.shape[0]
    if batch_size is None or batch_size >= num_blocks:
        batched_block_indices = [bboxes]
    else:
        num_batchs = int(np.ceil(num_blocks / batch_size))
        batch_indices = np.linspace(0, num_blocks, num=num_batchs+1, endpoint=True)
        batch_indices = np.unique(batch_indices.astype(np.int32))
        batched_block_indices = []
        for bidx0, bidx1 in zip(batch_indices[:-1], batch_indices[1:]):
            batched_block_indices.append(bboxes[bidx0:bidx1])
    render0 = MeshRenderer.from_mesh(mesh0, image_loader=image_loader0)
    render1 = MeshRenderer.from_mesh(mesh1, image_loader=image_loader1)
    xy0 = []
    xy1 = []
    conf = []
    for block_indices in batched_block_indices:
        stack0 = []
        stack1 = []
        xy_ctr = []
        for x0, y0, x1, y1 in block_indices:
            img0 = render0.crop((x0, y0, x1, y1), mode=render_mode, log_sigma=sigma, remap_interp=cv2.INTER_LINEAR)
            if img0 is None:
                continue
            img1 = render1.crop((x0, y0, x1, y1), mode=render_mode, log_sigma=sigma, remap_interp=cv2.INTER_LINEAR)
            if img1 is None:
                continue
            stack0.append(img0)
            stack1.append(img1)
            xy_ctr.append(((x0+x1-1)/2, (y0+y1-1)/2))
        dx, dy, conf_b = xcorr_fft(np.stack(stack0, axis=0), np.stack(stack1, axis=0),
            conf_mode=conf_mode, pad=pad)
        xy_ctr = np.array(xy_ctr)
        dxy = np.stack((dx, dy), axis=-1)
        xy0_b = xy_ctr - dxy/2
        xy1_b = xy_ctr + dxy/2
        xy0.append(xy0_b)
        xy1.append(xy1_b)
        conf.append(conf_b)
    xy0 = np.concatenate(xy0, axis=0)
    xy1 = np.concatenate(xy1, axis=0)
    conf = np.concatenate(conf, axis=0)
    return xy0, xy1, conf


## ----------------- matching block distributors --------------------------- ##
def distributor_cartesian_bbox(mesh0, mesh1, spacing, **kwargs):
    gear = kwargs.get('gear', const.MESH_GEAR_MOVING)
    min_num_blocks = kwargs.get('min_num_blocks', 1)
    shrink_factor = kwargs.get('shrink_factor', 1)
    zorder = kwargs.get('zorder', False)
    bbox0 = mesh0.bbox(gear=gear)
    bbox1 = mesh1.bbox(gear=gear)
    bbox, valid = common.intersect_bbox(bbox0, bbox1)
    if not valid:
        return None
    xstt, ystt, xend, yend = common.divide_bbox(bbox, block_size=spacing,
        min_num_blocks=min_num_blocks, shrink_factor=shrink_factor)
    if zorder:
        x_rnd = np.round((xstt - xstt.min()) / spacing)
        y_rnd = np.round((ystt - ystt.min()) / spacing)
        idx = common.z_order(np.stack((x_rnd, y_rnd), axis=-1))
        xstt, ystt, xend, yend = xstt[idx], ystt[idx], xend[idx], yend[idx]
    return np.stack((xstt, ystt, xend, yend), axis=-1)
