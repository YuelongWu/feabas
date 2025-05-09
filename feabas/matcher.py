from collections import defaultdict
import cv2
from functools import partial
import numpy as np
from scipy import fft
from scipy.fftpack import next_fast_len
import shapely.geometry as shpgeo
from shapely.ops import unary_union

from feabas.config import DEFAULT_DEFORM_BUDGET, data_resolution, section_thickness
from feabas.concurrent import submit_to_workers
from feabas.mesh import Mesh
from feabas.renderer import MeshRenderer
from feabas import optimizer, dal, common, spatial
import feabas.constant as const



def xcorr_fft(img0, img1, conf_mode=const.FFT_CONF_MIRROR, **kwargs):
    """
    find the displacements between two image(-stack)s from the Fourier based
    cross-correlation.
    Args:
        img0(ndarray): the first image, N x H0 x W0 (x C).
        img1(ndarray): the second image, N x H1 x W1 (x C).
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
    subpixel = kwargs.get('subpixel', False)
    pad = kwargs.get('pad', True)
    if len(img0.shape) > 3:
        img0 = np.moveaxis(img0, -1, 1)
    if len(img1.shape) > 3:
        img1 = np.moveaxis(img1, -1, 1)
    if sigma > 0:
        img0 = common.masked_dog_filter(img0, sigma, mask=mask0)
        img1 = common.masked_dog_filter(img1, sigma, mask=mask1)
    imgshp0 = img0.shape[-2:]
    imgshp1 = img1.shape[-2:]
    if pad:
        fftshp = [next_fast_len(s0 + s1 - 1) for s0, s1 in zip(imgshp0, imgshp1)]
    else:
        fftshp = [next_fast_len(max(s0, s1)) for s0, s1 in zip(imgshp0, imgshp1)]
    F0 = fft.rfft2(img0, s=fftshp, axes=(-2,-1))
    F1 = fft.rfft2(img1, s=fftshp, axes=(-2,-1))
    FF = np.conj(F0) * F1
    if len(FF.shape) > 3:
        FF = FF.mean(axis=1)
    C = fft.irfft2(FF, s=fftshp, axes=(-2,-1))
    Nimg = C.shape[0]
    C = C.reshape(Nimg, -1)
    if normalize:
        if mask0 is None:
            mask0 = np.ones_like(img0, shape=img0.shape[-2:])
        if mask1 is None:
            mask1 = np.ones_like(img1, shape=img1.shape[-2:])
        M0 = fft.rfft2(mask0, s=fftshp)
        M1 = fft.rfft2(mask1, s=fftshp)
        NC = fft.irfft2(np.conj(M0) * M1, s=fftshp)
        NC = NC.reshape(-1, np.prod(fftshp))
        NC = (NC / (NC.max(axis=-1, keepdims=True).clip(1, None))).clip(0.1, None)
        C = C / NC
    indx = np.argmax(C, axis=-1)
    dy, dx = np.unravel_index(indx, fftshp)
    if subpixel:
        ddx, ddy = np.meshgrid((-1,0,1),(-1,0,1))
        cy = (dy.reshape(-1,1) + ddy.reshape(1,-1)) % fftshp[0]
        cx = (dx.reshape(-1,1) + ddx.reshape(1,-1)) % fftshp[1]
        indx2 = np.ravel_multi_index((cy, cx), fftshp)
        z_ind = np.tile(np.arange(indx2.shape[0]).reshape(-1,1),(1, indx2.shape[1]))
        Ct = C[z_ind, indx2]
        tx = (Ct[:,5] - Ct[:,3]) / 2
        ty = (Ct[:,7] - Ct[:,1]) / 2
        txx = Ct[:,3] + Ct[:,5] - 2 * Ct[:,4]
        tyy = Ct[:,7] + Ct[:,1] - 2 * Ct[:,4]
        txy = (Ct[:,0] + Ct[:,8] - Ct[:,2] - Ct[:,6]) / 4
        det = txx * tyy - txy * txy
        ox = np.zeros_like(dx, dtype=np.float32)
        oy = np.zeros_like(dy, dtype=np.float32)
        nz_idx = det > 0
        ixx = tyy[nz_idx] / det[nz_idx]
        ixy = -txy[nz_idx] / det[nz_idx]
        iyy =  txx[nz_idx] / det[nz_idx]
        ox[nz_idx] = -ixx * tx[nz_idx] - ixy * ty[nz_idx]
        oy[nz_idx] = -ixy * tx[nz_idx] - iyy * ty[nz_idx]
        dx = dx + ox.clip(-0.5,0.5)
        dy = dy + oy.clip(-0.5,0.5)
    dy = dy + (imgshp0[0] - imgshp1[0]) / 2
    dx = dx + (imgshp0[1] - imgshp1[1]) / 2
    dy = dy - np.round(dy / fftshp[0]) * fftshp[0]
    dx = dx - np.round(dx / fftshp[1]) * fftshp[1]
    if conf_mode == const.FFT_CONF_NONE:
        conf = np.ones_like(dx, dtype=np.float32)
    elif conf_mode == const.FFT_CONF_MIRROR:
        FF = F0 * F1
        if len(FF.shape) > 3:
            FF = FF.mean(axis=1)
        C_mirror = np.abs(fft.irfft2(FF, s=fftshp, axes=(-2,-1)))
        C_mirror = C_mirror.reshape(Nimg, -1)
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
    img0_t = np.expand_dims(img0, 0)
    img1_t = np.expand_dims(img1, 0)
    tx, ty, conf = xcorr_fft(img0_t, img1_t, conf_mode=conf_mode, pad=True)
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
    sigma = kwargs.pop('sigma', 2.5)
    mask0 = kwargs.pop('mask0', None)
    mask1 = kwargs.pop('mask1', None)
    coarse_downsample = kwargs.pop('coarse_downsample', 1)
    fine_downsample = kwargs.pop('fine_downsample', 1)
    spacings = kwargs.pop('spacings', None)
    residue_len = kwargs.pop('residue_len', 5)
    conf_mode = kwargs.get('conf_mode', const.FFT_CONF_MIRROR)
    conf_thresh = kwargs.get('conf_thresh', 0.3)
    min_num_blocks = kwargs.get('min_num_blocks', 2)
    kwargs.setdefault('residue_mode', 'huber')
    kwargs.setdefault('opt_tol', None)

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
            mask0_g = cv2.resize(mask0.astype(np.uint8), None, fx=coarse_downsample, fy=coarse_downsample, interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            mask0_g = None
        if mask1 is not None:
            mask1_g = cv2.resize(mask1.astype(np.uint8), None, fx=coarse_downsample, fy=coarse_downsample, interpolation=cv2.INTER_NEAREST).astype(bool)
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
        return None, None, conf_thresh, None
    if fine_downsample == coarse_downsample:
        img0_f = img0_g
        img1_f = img1_g
    else:
        if fine_downsample != 1:
            img0_f = cv2.resize(img0, None, fx=fine_downsample, fy=fine_downsample, interpolation=cv2.INTER_AREA)
            img1_f = cv2.resize(img1, None, fx=fine_downsample, fy=fine_downsample, interpolation=cv2.INTER_AREA)
            if mask0 is not None:
                mask0_f = cv2.resize(mask0.astype(np.uint8), None, fx=fine_downsample, fy=fine_downsample, interpolation=cv2.INTER_NEAREST).astype(bool)
            else:
                mask0_f = None
            if mask1 is not None:
                mask1_f = cv2.resize(mask1.astype(np.uint8), None, fx=fine_downsample, fy=fine_downsample, interpolation=cv2.INTER_NEAREST).astype(bool)
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
    working_resolution = data_resolution() / fine_downsample
    residue_len = residue_len * fine_downsample
    img_loader0 = dal.StreamLoader(img0_f, fillval=0, resolution=working_resolution)
    img_loader1 = dal.StreamLoader(img1_f, fillval=0, resolution=working_resolution)
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
        mesh_size=min_spacing, min_num_blocks=min_num_blocks, uid=0,
        resolution=img_loader0.resolution)
    mesh1 = Mesh.from_bbox(img_loader1.bounds, cartesian=True,
        mesh_size=min_spacing, min_num_blocks=min_num_blocks, uid=1,
        resolution=img_loader1.resolution)
    mesh0.apply_translation((tx0, ty0), const.MESH_GEAR_FIXED)
    mesh0.lock()
    xy0, xy1, weight, strain = iterative_xcorr_matcher_w_mesh(mesh0, mesh1, img_loader0, img_loader1,
        spacings=spacings, distributor='cartesian_bbox', residue_len=residue_len, **kwargs)
    if (fine_downsample != 1) and (xy0 is not None):
        xy0 = spatial.scale_coordinates(xy0, 1/fine_downsample)
        xy1 = spatial.scale_coordinates(xy1, 1/fine_downsample)
    return xy0, xy1, weight, strain


def section_matcher(mesh0, mesh1, image_loader0, image_loader1, **kwargs):
    """
    matching two sections. When initial_matches is givenm handle disconnected
    subregions if necessary. If no initial_matches are provided, assume the two
    meshes are roughly aligned in MESH_GEAR_MOVING gears.
    """
    initial_matches = kwargs.pop('initial_matches', None)
    spacings = kwargs.pop('spacings', [100])
    kwargs.setdefault('sigma', 2.5)
    kwargs.setdefault('batch_size', 100)
    kwargs.setdefault('continue_on_flip', True)
    kwargs.setdefault('distributor', 'cartesian_region')
    kwargs.setdefault('link_weight_decay', 0.0)
    stiffness_multiplier_threshold = kwargs.get('stiffness_multiplier_threshold', 0.1)
    kwargs.setdefault('render_weight_threshold', 0.1)
    stiffness_lambda = kwargs.setdefault('stiffness_lambda', 0.5)
    if stiffness_multiplier_threshold > 0:
        idx0 = mesh0.triangle_mask_for_stiffness(stiffness_multiplier_threshold=stiffness_multiplier_threshold)
        mesh0 = mesh0.submesh(idx0)
        idx1 = mesh1.triangle_mask_for_stiffness(stiffness_multiplier_threshold=stiffness_multiplier_threshold)
        mesh1 = mesh1.submesh(idx1)
    if (initial_matches is None) or (mesh0.connected_triangles()[0] == 1 and mesh1.connected_triangles()[0] == 1):
        xy0, xy1, weight, strain = iterative_xcorr_matcher_w_mesh(mesh0, mesh1, image_loader0, image_loader1,
            spacings=spacings, initial_matches=initial_matches, compute_strain=True,
            **kwargs)
    else:
        opt = optimizer.SLM([mesh0, mesh1], stiffness_lambda=stiffness_lambda)
        xy0, xy1, weight = initial_matches.xy0, initial_matches.xy1, initial_matches.weight
        opt.add_link_from_coordinates(mesh0.uid, mesh1.uid, xy0, xy1,
            gear=(const.MESH_GEAR_INITIAL, const.MESH_GEAR_INITIAL), weight=weight,
            check_duplicates=False)
        opt.divide_disconnected_submeshes(prune_links=True)
        xy0 = []
        xy1 = []
        weight = []
        for lnk in opt.links:
            msh0_t, msh1_t = lnk.meshes
            ini_xy0_t = lnk.xy0(gear=const.MESH_GEAR_INITIAL, use_mask=False, combine=True)
            ini_xy1_t = lnk.xy1(gear=const.MESH_GEAR_INITIAL, use_mask=False, combine=True)
            ini_wt_t = lnk.weight(use_mask=False)
            ini_mtch_t = common.Match(ini_xy0_t, ini_xy1_t, ini_wt_t)
            xy0_t, xy1_t, wt_t, strain = iterative_xcorr_matcher_w_mesh(msh0_t.copy(), msh1_t.copy(),
                image_loader0, image_loader1, spacings=spacings, compute_strain=True,
                initial_matches=ini_mtch_t, **kwargs)
            if xy0_t is not None:
                if (msh0_t.uid - msh1_t.uid) * (mesh0.uid - mesh1.uid) > 0:
                    xy0.append(xy0_t)
                    xy1.append(xy1_t)
                else:
                    xy0.append(xy1_t)
                    xy1.append(xy0_t)
                weight.append(wt_t)
        if len(xy0) == 0:
            return None, None, 0, DEFAULT_DEFORM_BUDGET
        xy0 = np.concatenate(xy0, axis=0)
        xy1 = np.concatenate(xy1, axis=0)
        weight = np.concatenate(weight, axis=0)
    return xy0, xy1, weight, strain


def iterative_xcorr_matcher_w_mesh(mesh0, mesh1, image_loader0, image_loader1, spacings, **kwargs):
    """
    find the corresponding points by alternatively performing template matching
    and mesh relaxation.
    Args:
        mesh0, mesh1 (feabas.mesh.Mesh): the mesh objects of the two sections to
            be matched. Note that the matching is operated at the resolution of
            the meshes, so it is important to adjust the resoltion of the meshes
            to allow desired scaling for template matching.
        image_loader0, image_loader1 (feabas.dal.MosaicLoader or StreamLoader
            or TensorStoreLoader):image loaders of the sections.
    Kwargs:
        sigma(float): if larger than 0, the cropped images for template matching
            will be pre-processed with DoG filter with sigma.
        conf_mode: the method to compute the confidence value for template
            matching. check feabas.constant for options.
        conf_thresh: the threshold of template matching confidence values below
            which the match will be discarded.
        residue_len: the threshold of the residue error distance between matching
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
        allow_enlarge(bool): whether to allow the spacing to increase to value
            larger than the largest among the preset values, if the first-round
            residue is very large after mesh relaxation indicating excessive
            error in initial estimation.
        link_weight_decay(float): the weight applied to the links belong to
            older spacing setting for each step. Set to 0.0 to start fresh for
            each new spacing setting.
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
    residue_mode = kwargs.get('residue_mode', 'huber')
    residue_len = kwargs.get('residue_len', 0)
    opt_tol = kwargs.get('opt_tol', None)
    distributor = kwargs.get('distributor', 'cartesian_bbox')
    min_num_blocks = kwargs.get('min_num_blocks', 2)
    min_boundary_distance = kwargs.get('min_boundary_distance', 0)
    shrink_factor = kwargs.get('shrink_factor', 1)
    allow_dwell = kwargs.get('allow_dwell', 0)
    allow_enlarge = kwargs.get('allow_enlarge', False)
    link_weight_decay = kwargs.get('link_weight_decay', 0.0)
    compute_strain = kwargs.get('compute_strain', True)
    batch_size = kwargs.pop('batch_size', None)
    initial_matches = kwargs.get('initial_matches', None)
    to_pad = kwargs.pop('pad', None)
    refine_mode = kwargs.get('refine_mode', 2)
    do_subpixel = kwargs.pop('subpixel', None)
    max_spacing_skip = kwargs.get('max_spacing_skip', 0)
    continue_on_flip = kwargs.get('continue_on_flip', True)
    callback_settings = kwargs.get('callback_settings', {'early_stop_thresh': 0.1, 'chances':10, 'eval_step': 5})
    render_weight_threshold = kwargs.get('render_weight_threshold', 0)
    stiffness_lambda = kwargs.pop('stiffness_lambda', 1)
    if num_workers > 1 and batch_size is not None:
        batch_size = max(1, batch_size / num_workers)
    if isinstance(image_loader0, dal.AbstractImageLoader):
        loader_dict0 = image_loader0.init_dict()
    else:
        loader_dict0 = image_loader0
    if isinstance(image_loader1, dal.AbstractImageLoader):
        loader_dict1 = image_loader1.init_dict()
    else:
        loader_dict1 = image_loader1
    if residue_len < 0:
        aspect_ratio = section_thickness() / mesh0.resolution
        residue_len = max(1, abs(residue_len) * aspect_ratio)
    if isinstance(refine_mode, str):
        if refine_mode.lower() == 'none':
            refine_mode = 0
        elif 'only' in refine_mode.lower():
            refine_mode = 1
        else:
            refine_mode = 2
    # if any spacing value smaller than 1, means they are relative to longer side
    spacings = np.array(spacings, copy=False)
    kwargs_opt = {
        "batch_num_matches": np.inf,
        "continue_on_flip": continue_on_flip,
        "callback_settings": callback_settings,
        "tolerated_perturbation": 0.5,
        "check_converge": True
    }
    linear_system = mesh0.is_linear and mesh1.is_linear
    one_locked = mesh0.locked or mesh1.locked
    min_block_size_multiplier = 4
    strain = DEFAULT_DEFORM_BUDGET
    invalid_output = (None, None, 0, strain)
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
    if compute_strain:
        mesh0_ori, mesh1_ori = mesh0.copy(), mesh1.copy()
    opt = optimizer.SLM([mesh0, mesh1], stiffness_lambda=stiffness_lambda)
    if initial_matches is not None:
        xy0, xy1, weight = initial_matches.xy0, initial_matches.xy1, initial_matches.weight
        opt.add_link_from_coordinates(mesh0.uid, mesh1.uid, xy0, xy1,
            gear=(const.MESH_GEAR_INITIAL, const.MESH_GEAR_INITIAL), weight=weight,
            check_duplicates=False, render_weight_threshold=render_weight_threshold)
        opt.optimize_affine_cascade(start_gear=const.MESH_GEAR_INITIAL, target_gear=const.MESH_GEAR_FIXED, svd_clip=(1,1))
        opt.anneal(gear=(const.MESH_GEAR_FIXED, const.MESH_GEAR_MOVING), mode=const.ANNEAL_COPY_EXACT)
        if linear_system:
            opt.optimize_linear(tol=1e-6, **kwargs_opt)
        else:
            opt.optimize_Newton_Raphson(max_newtonstep=5, tol=1e-4, **kwargs_opt)
    else:
        mesh0.anneal(gear=(const.MESH_GEAR_MOVING, const.MESH_GEAR_FIXED), mode=const.ANNEAL_COPY_EXACT)
        mesh1.anneal(gear=(const.MESH_GEAR_MOVING, const.MESH_GEAR_FIXED), mode=const.ANNEAL_COPY_EXACT)
    spacings = np.sort(spacings)[::-1]
    sp = np.max(spacings)
    sp_indx = 0
    initialized = False
    spacing_enlarged = not allow_enlarge
    dwelled = 0
    if to_pad is None:
        pad = True
    else:
        pad = to_pad
    while sp_indx < spacings.size:
        if sp == spacings[-1]:
            mnb = min_num_blocks
            rfm = refine_mode
            if do_subpixel is None:
                subpixel = True
            else:
                subpixel = do_subpixel
        else:
            mnb = 1
            if refine_mode == 2:
                rfm = 0
            else:
                rfm = refine_mode
            if do_subpixel is None:
                subpixel = False
            else:
                subpixel = do_subpixel
        if distributor == 'cartesian_bbox':
            bboxes0, bboxes1 = distributor_cartesian_bbox(mesh0, mesh1, sp,
                min_num_blocks=mnb, shrink_factor=shrink_factor, zorder=True)
        else:
            bboxes0, bboxes1 = distribute_matching_blocks(mesh0, mesh1, sp,
                dfunc=distributor, refine_mode=rfm,
                min_boundary_distance=min_boundary_distance, shrink_factor=shrink_factor,
                zorder=True, render_weight_threshold=render_weight_threshold)
        if bboxes0 is None:
            return invalid_output
        num_blocks = bboxes0.shape[0]
        if batch_size is not None:
            batch_size_s = max(1, np.round(batch_size * (np.max(spacings) / sp) ** 2))
        else:
            batch_size_s = None
        if num_workers > 1:
            if batch_size_s is None:
                batch_size_s = max(1, num_blocks/num_workers)
            else:
                batch_size_s = min(max(1, num_blocks/num_workers), batch_size_s)
            num_batchs = int(np.ceil(num_blocks / batch_size_s))
            if num_batchs == 1:
                xy0, xy1, conf = bboxes_mesh_renderer_matcher(mesh0, mesh1,
                    image_loader0, image_loader1, bboxes0, bboxes1,
                    batch_size=batch_size, pad=pad, subpixel=subpixel, **kwargs)
            else:
                batch_indices = np.linspace(0, num_blocks, num=num_batchs+1, endpoint=True)
                batch_indices = np.unique(batch_indices.astype(np.int32))
                batched_bboxes0 = []
                batched_bboxes1 = []
                batched_bboxes_union0 = []
                batched_bboxes_union1 = []
                for bidx0, bidx1 in zip(batch_indices[:-1], batch_indices[1:]):
                    batched_bboxes0.append(bboxes0[bidx0:bidx1])
                    batched_bboxes1.append(bboxes1[bidx0:bidx1])
                    batched_bboxes_union0.append(common.bbox_union(bboxes0[bidx0:bidx1]))
                    batched_bboxes_union1.append(common.bbox_union(bboxes1[bidx0:bidx1]))
                target_func = partial(bboxes_mesh_renderer_matcher, pad=pad, subpixel=subpixel, **kwargs)
                submeshes0 = mesh0.submeshes_from_bboxes(batched_bboxes_union0)
                submeshes1 = mesh1.submeshes_from_bboxes(batched_bboxes_union1)
                xy0 = []
                xy1 = []
                conf = []
                args_list = []
                for m0_p, m1_p, bboxes0_p, bboxes1_p in zip(submeshes0, submeshes1, batched_bboxes0, batched_bboxes1):
                    if (m0_p is None) or (m1_p is None):
                        continue
                    m0dict = m0_p.get_init_dict()
                    m1dict = m1_p.get_init_dict()
                    args_list.append((m0dict, m1dict, loader_dict0, loader_dict1, bboxes0_p, bboxes1_p))
                for res in submit_to_workers(target_func, args=args_list, num_workers=num_workers):
                    pt0, pt1, cnf = res
                    xy0.append(pt0)
                    xy1.append(pt1)
                    conf.append(cnf)
                if len(xy0) == 0:
                    return invalid_output
                xy0 = np.concatenate(xy0, axis=0)
                xy1 = np.concatenate(xy1, axis=0)
                conf = np.concatenate(conf, axis=0)
        else:
            xy0, xy1, conf = bboxes_mesh_renderer_matcher(mesh0, mesh1,
                image_loader0, image_loader1, bboxes0, bboxes1,
                batch_size=batch_size, pad=pad, subpixel=subpixel, **kwargs)
        if np.all(conf <= conf_thresh):
            if not initialized:
                return invalid_output
            else:
                break
        if link_weight_decay == 0:
            opt.clear_links()
        else:
            for lnk in opt.links:
                lnk._weight = lnk._weight * link_weight_decay
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
            sp_indx = -1
            spacing_enlarged = True
            sp = np.ceil(min_block_size)
            if to_pad is None:
                pad = True
            continue
        spacing_enlarged = True
        if next_pos > sp_indx:
            next_pos = min(next_pos, sp_indx + 1 + max_spacing_skip)
            if to_pad is None:
                if next_pos > sp_indx + 1:
                    pad = True
                else:
                    pad = False
            sp_indx = next_pos
            dwelled = 0
        elif dwelled >= allow_dwell:
            if to_pad is None:
                pad = True
            sp_indx += 1
            dwelled = 0
        else:
            if to_pad is None:
                pad = True
            dwelled += 1
        opt.add_link_from_coordinates(mesh0.uid, mesh1.uid, xy0, xy1,
                        gear=(const.MESH_GEAR_MOVING, const.MESH_GEAR_MOVING), weight=wt,
                        check_duplicates=False, render_weight_threshold=render_weight_threshold)
        if max_dis > 0.1:
            if linear_system:
                opt.optimize_linear(tol=opt_tol_t, **kwargs_opt)
            else:
                opt.optimize_Newton_Raphson(max_newtonstep=3, tol=opt_tol_t, **kwargs_opt)
            if residue_len > 0:
                if residue_mode == 'huber':
                    opt.set_link_residue_huber(residue_len)
                elif residue_mode == 'threshold':
                    opt.set_link_residue_threshold(residue_len)
                else:
                    raise ValueError
                weight_modified, _ = opt.adjust_link_weight_by_residue(relax_first=True)
                if weight_modified and (sp_indx < spacings.size):
                    if linear_system:
                        opt.optimize_linear(tol=opt_tol_t, **kwargs_opt)
                    else:
                        opt.optimize_Newton_Raphson(max_newtonstep=3, tol=opt_tol_t, **kwargs_opt)
        initialized = True
        if (sp_indx < spacings.size) and (sp_indx >= 0):
            sp = spacings[sp_indx]
    if len(opt.links) == 0:
        return invalid_output
    link = opt.links[-1]
    xy0 = link.xy0(gear=const.MESH_GEAR_INITIAL, use_mask=True, combine=True)
    xy1 = link.xy1(gear=const.MESH_GEAR_INITIAL, use_mask=True, combine=True)
    weight = link.weight(use_mask=True)
    if compute_strain:
        opt = optimizer.SLM([mesh0_ori, mesh1_ori], stiffness_lambda=stiffness_lambda, assert_dominance=(not one_locked))
        opt.add_link_from_coordinates(mesh0_ori.uid, mesh1_ori.uid, xy0, xy1,
            gear=(const.MESH_GEAR_INITIAL, const.MESH_GEAR_INITIAL), weight=weight,
            check_duplicates=False, render_weight_threshold=render_weight_threshold)
        opt.optimize_affine_cascade(start_gear=const.MESH_GEAR_INITIAL, target_gear=const.MESH_GEAR_FIXED, svd_clip=(1,1))
        opt.anneal(gear=(const.MESH_GEAR_FIXED, const.MESH_GEAR_MOVING), mode=const.ANNEAL_COPY_EXACT)
        if linear_system:
            opt.optimize_linear(tol=1e-6, **kwargs_opt)
        else:
            opt.optimize_Newton_Raphson(max_newtonstep=5, tol=1e-4, **kwargs_opt)
        Es0 = 0
        Es = 0
        soft_factor_avg = np.mean([m.soft_factor for m in opt.meshes])
        for m in opt.meshes:
            to_include = (one_locked and (not m.locked)) or ((not one_locked) and (m.soft_factor<=soft_factor_avg))
            if to_include:
                v0 = m.vertices(gear=const.MESH_GEAR_FIXED)
                v1 = m.vertices(gear=const.MESH_GEAR_MOVING)
                dv = v1 - v0
                v0 = v0 - np.mean(v0, axis=0, keepdims=True)
                dv = dv - np.mean(dv, axis=0, keepdims=True)
                St, _ = m.stiffness_matrix()
                Es += max(0, St.dot(dv.ravel()).dot(dv.ravel()))
                Es0 += max(0, St.dot(v0.ravel()).dot(v0.ravel()))
        strain = (Es / Es0) ** 0.5
    return xy0, xy1, weight, strain


def bboxes_mesh_renderer_matcher(mesh0, mesh1, image_loader0, image_loader1, bboxes0, bboxes1, **kwargs):
    batch_size = kwargs.get('batch_size', None)
    sigma = kwargs.get('sigma', 0.0)
    render_mode = kwargs.get('render_mode', const.RENDER_FULL)
    geodesic_mask = kwargs.get('geodesic_mask', False)
    conf_mode = kwargs.get('conf_mode', const.FFT_CONF_MIRROR)
    pad = kwargs.get('pad', True)
    subpixel = kwargs.get('subpixel', False)
    render_weight_threshold = kwargs.get('render_weight_threshold', 0)
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
    num_blocks = bboxes0.shape[0]
    blksz0 = np.round(common.bbox_sizes(bboxes0))
    blksz1 = np.round(common.bbox_sizes(bboxes1))
    szchg = np.nonzero(np.any(np.diff(blksz0, axis=0), axis=-1) | np.any(np.diff(blksz1, axis=0), axis=-1))[0]
    szchg_id = np.concatenate(([0], szchg+1, [num_blocks]), axis=None)
    if batch_size is None or batch_size >= num_blocks:
        batch_indices = szchg_id
    else:
        batch_indices = []
        for bid0, bid1 in zip(szchg_id[:-1], szchg_id[1:]):
            num_batchs = max(1, int(np.ceil(bid1-bid0 / batch_size)))
            batch_indices.append(np.linspace(bid0, bid1, num=num_batchs+1, endpoint=True))
        batch_indices = np.round(np.concatenate(batch_indices, axis=-1))
        batch_indices = np.unique(batch_indices.astype(np.int32))
    batched_block_indices0 = []
    batched_block_indices1 = []
    for bidx0, bidx1 in zip(batch_indices[:-1], batch_indices[1:]):
        batched_block_indices0.append(bboxes0[bidx0:bidx1])
        batched_block_indices1.append(bboxes1[bidx0:bidx1])
    render0 = MeshRenderer.from_mesh(mesh0, image_loader=image_loader0, geodesic_mask=geodesic_mask, render_weight_threshold=render_weight_threshold)
    render1 = MeshRenderer.from_mesh(mesh1, image_loader=image_loader1, geodesic_mask=geodesic_mask, render_weight_threshold=render_weight_threshold)
    if (render0 is None) or (render1 is None):
        xy0 = np.empty((0,2))
        xy1 = np.empty((0,2))
        conf = np.empty(0)
        return xy0, xy1, conf
    xy0 = []
    xy1 = []
    conf = []
    for block_indices0, block_indices1 in zip(batched_block_indices0, batched_block_indices1):
        stack0 = []
        stack1 = []
        xy_ctr0 = []
        xy_ctr1 = []
        wt_ratio = []
        for bbox0, bbox1 in zip(block_indices0, block_indices1):
            img0 = render0.crop(bbox0, mode=render_mode, log_sigma=sigma, remap_interp=cv2.INTER_LINEAR)
            if img0 is None:
                continue
            img1 = render1.crop(bbox1, mode=render_mode, log_sigma=sigma, remap_interp=cv2.INTER_LINEAR)
            if img1 is None:
                continue
            stack0.append(img0)
            stack1.append(img1)
            xy_ctr0.append(((bbox0[0]+bbox0[2]-1)/2, (bbox0[1]+bbox0[3]-1)/2))
            xy_ctr1.append(((bbox1[0]+bbox1[2]-1)/2, (bbox1[1]+bbox1[3]-1)/2))
            wd0, ht0 = bbox0[2] - bbox0[0], bbox0[3] - bbox0[1]
            wd1, ht1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
            wt_ratio.append((wd0 / (wd0 + wd1), ht0 / (ht0 + ht1)))
        if (len(stack0) == 0) or (len(stack1) == 0):
            continue
        dx, dy, conf_b = xcorr_fft(np.stack(stack0, axis=0), np.stack(stack1, axis=0),
            conf_mode=conf_mode, pad=pad, subpixel=subpixel)
        xy_ctr0 = np.array(xy_ctr0)
        xy_ctr1 = np.array(xy_ctr1)
        wt_ratio = np.array(wt_ratio)
        dxy = np.stack((dx, dy), axis=-1)
        xy0_b = xy_ctr0 - dxy * wt_ratio
        xy1_b = xy_ctr1 + dxy * (1-wt_ratio)
        xy0.append(xy0_b)
        xy1.append(xy1_b)
        conf.append(conf_b)
    if len(xy0) > 0:
        xy0 = np.concatenate(xy0, axis=0)
        xy1 = np.concatenate(xy1, axis=0)
        conf = np.concatenate(conf, axis=0)
    else:
        xy0 = np.empty((0,2))
        xy1 = np.empty((0,2))
        conf = np.empty(0)
    return xy0, xy1, conf


## ----------------- matching block distributors --------------------------- ##
def distributor_cartesian_bbox(mesh0, mesh1, spacing, **kwargs):
    gear = kwargs.get('gear', const.MESH_GEAR_MOVING)
    min_num_blocks = kwargs.get('min_num_blocks', 1)
    shrink_factor = kwargs.get('shrink_factor', 1)
    zorder = kwargs.get('zorder', False)
    if not hasattr(shrink_factor, '__len__'):
        shrink_factor = (shrink_factor, shrink_factor)
    bbox0 = mesh0.bbox(gear=gear)
    bbox1 = mesh1.bbox(gear=gear)
    bbox, valid = common.intersect_bbox(bbox0, bbox1)
    if not valid:
        return None, None
    bbox0 = common.divide_bbox(bbox, block_size=spacing,
        min_num_blocks=min_num_blocks, shrink_factor=shrink_factor[0])
    bbox1 = common.divide_bbox(bbox, block_size=spacing,
        min_num_blocks=min_num_blocks, shrink_factor=shrink_factor[1])
    bbox0 = np.stack(bbox0, axis=-1)
    bbox1 = np.stack(bbox1, axis=-1)
    if zorder:
        xstt = bbox0[:, 0]
        ystt = bbox0[:, 1]
        x_rnd = np.round((xstt - xstt.min()) / spacing)
        y_rnd = np.round((ystt - ystt.min()) / spacing)
        idx = common.z_order(np.stack((x_rnd, y_rnd), axis=-1))
        bbox0 = bbox0[idx]
        bbox1 = bbox1[idx]
    return bbox0, bbox1


def distribute_matching_blocks(mesh0, mesh1, spacing, dfunc='cartesian_region', **kwargs):
    """
    Distribute rectangular blocks for local template matching.
    Args:
        mesh0, mesh1 (feabas.mesh.Mesh): the mesh objects of the two sections to
            be matched.
        spacing (float): the distances between the neighboring sample blocks for
            template matcing.
        dfunc (callable): function that takes in shapely regions and spacing, and
            produces points for matching.
    Kwargs:
        refine_mode: 0-ignore refine flags; 1-refine flags only; 2-composite.
        gear: gear of the meshes to use.
        shrink_factor (float): parameter fed into the distributor function. Determines
            the template-matching block size compared to the spacing.
        refine_box_exp (float): box size for refinement = spacing * (area_constraint
            of refine material)^refine_box_exp
        min_boundary_distance (float): inimum distance allowed from a matching point
            to the boundary of the meshes
    """
    refine_mode = kwargs.get('refine_mode', 2)
    gear = kwargs.get('gear', const.MESH_GEAR_MOVING)
    shrink_factor = kwargs.get('shrink_factor', 1)
    refine_box_exp = kwargs.get('refine_box_exp', 0.5)
    min_box_side = kwargs.get('min_box_side', 5)
    max_box_side = kwargs.get('max_box_side', np.inf)
    min_boundary_distance = kwargs.get('min_boundary_distance', 0)
    zorder = kwargs.get('zorder', True)
    render_weight_threshold = kwargs.get('render_weight_threshold', 0)
    if isinstance(dfunc, str):
        if dfunc.lower() == 'cartesian_region':
            dfunc = _region2grid_cartesian
        elif dfunc.lower() == 'intersect_triangulation':
            dfunc = _region2grid_triang
        else:
            raise ValueError(f'unsupported distributor type {dfunc}')
    if isinstance(refine_mode, str):
        if refine_mode.lower() == 'none':
            refine_mode = 0
        elif 'only' in refine_mode.lower():
            refine_mode = 1
        else:
            refine_mode = 2
    bboxes0 = []
    bboxes1 = []
    if render_weight_threshold > 0:
        idx0 = mesh0.triangle_mask_for_render(render_weight_threshold=render_weight_threshold)
        mesh0 = mesh0.submesh(idx0)
        idx1 = mesh1.triangle_mask_for_render(render_weight_threshold=render_weight_threshold)
        mesh1 = mesh1.submesh(idx1)
    regs = defaultdict(list)
    region0 = mesh0.shapely_regions(gear=gear)
    region1 = mesh1.shapely_regions(gear=gear)
    reg_crx0 = region0.intersection(region1)
    if reg_crx0.area == 0:
        bboxes0, bboxes1 = np.empty((0,4)), np.empty((0,4))
        return bboxes0, bboxes1
    if not hasattr(shrink_factor, '__len__'):
        shrink_factor = (shrink_factor, shrink_factor)
    else:
        if (region0.area / mesh0.num_triangles) > (region1.area / mesh1.num_triangles):
            shrink_factor = (max(shrink_factor), min(shrink_factor))
        else:
            shrink_factor = (min(shrink_factor), max(shrink_factor))
    if refine_mode == 0:
        if reg_crx0.area > 0:
            regs[1.0].append(reg_crx0)
    else:
        if refine_mode == 2:
            regs[1.0].append(reg_crx0)
        meshes  = (mesh0, mesh1)
        for msh in meshes:
            mtb = msh.named_material_table
            mid = msh.material_ids
            for mnm in mtb:
                area_factor = mtb[mnm].area_constraint
                if ('refine' not in mnm) and ((area_factor == 0) or (area_factor >= 1)):
                    continue
                tidx = mid == mtb[mnm].uid
                if np.any(tidx):
                    region = msh.shapely_regions(gear=gear, tri_mask=tidx)
                    region = region.intersection(reg_crx0)
                    if region.area > 0:
                        regs[area_factor].append(region)
    covered = shpgeo.Polygon()
    for area_factor in sorted(list(regs.keys())):
        spc = spacing * area_factor
        shrnk0 = area_factor ** (refine_box_exp - 1)
        region = unary_union(regs[area_factor])
        area_r = region.area
        bound_coeff = 1.0
        if min_boundary_distance > 0:
            while True:
                adjust_dis = min_boundary_distance * shrnk0 * bound_coeff
                reg_crx = reg_crx0.buffer(-adjust_dis).difference(covered)
                region_crx = reg_crx.intersection(region)
                if region_crx.area >= 0.5 * area_r:
                    break
                bound_coeff *= 0.3 / (1 - region_crx.area/area_r)
                if bound_coeff < 0.1:
                    region_crx = reg_crx0.difference(covered).intersection(region)
        else:
            region_crx = reg_crx0.difference(covered).intersection(region)
        covered = covered.union(region)
        cntrs = dfunc(region_crx, spc, **kwargs)
        if cntrs is None:
            continue
        sides_L = (spc * shrnk0 * np.array(shrink_factor)).clip(min_box_side, max_box_side)
        blk_hfsz0 = np.ceil(sides_L[0] / 2)
        blk_hfsz1 = np.ceil(sides_L[1] / 2)
        bboxes0_a = np.concatenate((cntrs-blk_hfsz0, cntrs+blk_hfsz0), axis=-1)
        bboxes1_a = np.concatenate((cntrs-blk_hfsz1, cntrs+blk_hfsz1), axis=-1)
        if zorder:
            x_rnd = np.round((cntrs[:,0] - cntrs[:,0].min()) / spc)
            y_rnd = np.round((cntrs[:,1] - cntrs[:,1].min()) / spc)
            idx = common.z_order(np.stack((x_rnd, y_rnd), axis=-1))
            bboxes0_a = bboxes0_a[idx]
            bboxes1_a = bboxes1_a[idx]
        bboxes0.append(bboxes0_a)
        bboxes1.append(bboxes1_a)
    bboxes0 = np.concatenate(bboxes0, axis=0)
    bboxes1 = np.concatenate(bboxes1, axis=0)
    return bboxes0, bboxes1


def _region2grid_cartesian(region, spacing, **kwargs):
    if hasattr(region, 'geoms'):
        regions = list(region.geoms)
    else:
        regions = [region]
    cntrs = []
    for reg in regions:
        if reg.area == 0:
            continue
        rx_mn, ry_mn, rx_mx, ry_mx = reg.bounds
        rpts = reg.representative_point()
        rx, ry = rpts.x, rpts.y
        rx_mn = rx - ((rx - rx_mn) // spacing) * spacing
        ry_mn = ry - ((ry - ry_mn) // spacing) * spacing
        rxx, ryy = np.meshgrid(np.arange(rx_mn, rx_mx, spacing), np.arange(ry_mn, ry_mx, spacing))
        rv = np.stack((rxx.ravel(), ryy.ravel()), axis=-1)
        cntrs.append(shpgeo.MultiPoint(rv).intersection(reg))
    if len(cntrs) == 0:
        return None
    cntrs = unary_union(cntrs)
    if hasattr(cntrs, 'geoms'):
        cntrs_np = np.array([(p.x, p.y) for p in cntrs.geoms])
    else:
        cntrs_np = np.array([(cntrs.x, cntrs.y)])
    return cntrs_np


def _region2grid_triang(region, spacing, **kwargs):
    epsilon = kwargs.get('epsilon', 0.01 * spacing)
    region = region.simplify(epsilon/2, preserve_topology=True)
    if region.area == 0:
        return None
    # region_b = region.buffer(-epsilon, join_style=2)
    # region_b = region_b.buffer(epsilon, join_style=3)
    # if region_b.area > 0.5 * region.area:
    #     region = region_b
    G = spatial.Geometry(roi=region, regions={'default': region})
    M = Mesh.from_PSLG(**G.PSLG(), mesh_size=spacing)
    cntrs_np = M.initial_vertices
    return cntrs_np
    

def distributor_cartesian_region(mesh0, mesh1, spacing, **kwargs):
    gear = kwargs.get('gear', const.MESH_GEAR_MOVING)
    shrink_factor = kwargs.get('shrink_factor', 1)
    min_boundary_distance = kwargs.get('min_boundary_distance', 0)
    zorder = kwargs.get('zorder', False)
    render_weight_threshold = kwargs.get('render_weight_threshold', 0)
    if render_weight_threshold > 0:
        idx0 = mesh0.triangle_mask_for_render(render_weight_threshold=render_weight_threshold)
        mesh0 = mesh0.submesh(idx0)
        idx1 = mesh1.triangle_mask_for_render(render_weight_threshold=render_weight_threshold)
        mesh1 = mesh1.submesh(idx1)
    region0 = mesh0.shapely_regions(gear=gear)
    region1 = mesh1.shapely_regions(gear=gear)
    if not hasattr(shrink_factor, '__len__'):
        shrink_factor = (shrink_factor, shrink_factor)
    else:
        if (region0.area / mesh0.num_triangles) > (region1.area / mesh1.num_triangles):
            shrink_factor = (max(shrink_factor), min(shrink_factor))
        else:
            shrink_factor = (min(shrink_factor), max(shrink_factor))
    reg_crx = region0.intersection(region1)
    if min_boundary_distance > 0:
        reg_crx = reg_crx.buffer(-min_boundary_distance)
    if reg_crx.area == 0:
        return None, None
    blk_hfsz0 = np.ceil(spacing * shrink_factor[0] / 2)
    blk_hfsz1 = np.ceil(spacing * shrink_factor[1] / 2)
    bcnters = _region2grid_cartesian(reg_crx, spacing)
    bboxes0 = np.concatenate((bcnters-blk_hfsz0, bcnters+blk_hfsz0), axis=-1)
    bboxes1 = np.concatenate((bcnters-blk_hfsz1, bcnters+blk_hfsz1), axis=-1)
    if zorder:
        x_rnd = np.round((bcnters[:,0] - bcnters[:,0].min()) / spacing)
        y_rnd = np.round((bcnters[:,1] - bcnters[:,1].min()) / spacing)
        idx = common.z_order(np.stack((x_rnd, y_rnd), axis=-1))
        bboxes0 = bboxes0[idx]
        bboxes1 = bboxes1[idx]
    return bboxes0, bboxes1


def distributor_intersect_triangulation(mesh0, mesh1, spacing, **kwargs):
    gear = kwargs.get('gear', const.MESH_GEAR_MOVING)
    shrink_factor = kwargs.get('shrink_factor', 1)
    min_boundary_distance = kwargs.get('min_boundary_distance', 0)
    zorder = kwargs.get('zorder', False)
    render_weight_threshold = kwargs.get('render_weight_threshold', 0)
    if render_weight_threshold > 0:
        idx0 = mesh0.triangle_mask_for_render(render_weight_threshold=render_weight_threshold)
        mesh0 = mesh0.submesh(idx0)
        idx1 = mesh1.triangle_mask_for_render(render_weight_threshold=render_weight_threshold)
        mesh1 = mesh1.submesh(idx1)
    region0 = mesh0.shapely_regions(gear=gear)
    region1 = mesh1.shapely_regions(gear=gear)
    reg_crx = region0.intersection(region1)
    if min_boundary_distance > 0:
        reg_crx = reg_crx.simplify(min_boundary_distance/3, preserve_topology=True)
        reg_crx = reg_crx.buffer(-min_boundary_distance, join_style=2)
        reg_crx = reg_crx.buffer(min_boundary_distance*0.2, join_style=3)
    if reg_crx.area == 0:
        return None, None
    bcnters = _region2grid_triang(reg_crx, spacing)
    if not hasattr(shrink_factor, '__len__'):
        shrink_factor = (shrink_factor, shrink_factor)
    else:
        if (region0.area / mesh0.num_triangles) > (region1.area / mesh1.num_triangles):
            shrink_factor = (max(shrink_factor), min(shrink_factor))
        else:
            shrink_factor = (min(shrink_factor), max(shrink_factor))
    blk_hfsz0 = np.ceil(spacing * shrink_factor[0] / 2)
    blk_hfsz1 = np.ceil(spacing * shrink_factor[1] / 2)
    bboxes0 = np.concatenate((bcnters-blk_hfsz0, bcnters+blk_hfsz0), axis=-1)
    bboxes1 = np.concatenate((bcnters-blk_hfsz1, bcnters+blk_hfsz1), axis=-1)
    if zorder:
        x_rnd = np.round((bcnters[:,0] - bcnters[:,0].min()) / spacing)
        y_rnd = np.round((bcnters[:,1] - bcnters[:,1].min()) / spacing)
        idx = common.z_order(np.stack((x_rnd, y_rnd), axis=-1))
        bboxes0 = bboxes0[idx]
        bboxes1 = bboxes1[idx]
    return bboxes0, bboxes1
