import cv2
import numpy as np
from scipy import fft, ndimage
from scipy.fftpack import next_fast_len

from feabas import optimizer, dal, miscs, mesh, renderer
from feabas.constant import *



def xcorr_fft(img0, img1, conf_mode=FFT_CONF_MIRROR, **kwargs):
    """
    find the displacements between two image(-stack)s from the Fourier based
    cross-correlation .
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
        img0 = miscs.masked_dog_filter(img0, sigma, mask=mask0)
        img1 = miscs.masked_dog_filter(img1, sigma, mask=mask1)
    imgshp0 = img0.shape[-2:]
    imgshp1 = img1.shape[-2:]
    if pad:
        fftshp = [next_fast_len(s0 + s1 - 1) for s0, s1 in zip(imgshp0, imgshp1)]
    else:
        fftshp = [next_fast_len(max(s0, s1)) for s0, s1 in zip(imgshp0, imgshp1)]
    F0 = fft.rfft2(img0, s=fftshp)
    F1 = fft.rfft2(img1, s=fftshp)
    C = fft.irfft2(np.conj(F0) * F1)
    C = C.reshape(-1, np.prod(fftshp))
    if normalize:
        if mask0 is None:
            mask0 = np.ones_like(img0)
        if mask1 is None:
            mask1 = np.ones_like(img1)
        M0 = fft.rfft2(mask0, s=fftshp)
        M1 = fft.rfft2(mask1, s=fftshp)
        NC = fft.irfft2(np.conj(M0) * M1)
        NC = NC.reshape(-1, np.prod(fftshp))
        NC = (NC / (NC.max(axis=-1, keepdims=True).clip(1, None))).clip(0.1, None)
        C = C / NC
    indx = np.argmax(C, axis=-1)
    dy, dx = np.unravel_index(indx, fftshp)
    dy = dy + (imgshp0[0] - imgshp1[0]) / 2
    dx = dx + (imgshp0[1] - imgshp1[1]) / 2
    dy = dy - np.round(dy / fftshp[0]) * fftshp[0]
    dx = dx - np.round(dx / fftshp[1]) * fftshp[1]
    if conf_mode == FFT_CONF_NONE:
        conf = np.ones_like(dx, dtype=np.float32)
    elif conf_mode == FFT_CONF_MIRROR:
        C_mirror = np.abs(fft.irfft2(F0 * F1))
        C_mirror = C_mirror.reshape(-1, np.prod(fftshp))
        if normalize:
            NC = fft.irfft2(M0 * M1)
            NC = NC.reshape(-1, np.prod(fftshp))
            NC = (NC / (NC.max(axis=-1, keepdims=True).clip(1, None))).clip(0.1, None)
            C_mirror = C_mirror / NC
        conf = 1 - C_mirror.max(axis=-1) / C.max(axis=-1)
        conf = conf.clip(0, 1)
    elif conf_mode == FFT_CONF_STD:
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
    conf_mode = kwargs.get('conf_mode', FFT_CONF_MIRROR)
    conf_thresh = kwargs.get('conf_thresh', 0.3)
    divide_factor = kwargs.get('divide_factor', 6)
    if sigma > 0:
        img0 = miscs.masked_dog_filter(img0, sigma, mask=mask0)
        img1 = miscs.masked_dog_filter(img1, sigma, mask=mask1)
    tx, ty, conf = xcorr_fft(img0, img1, conf_mode=conf_mode, pad=True)
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
    xmin0, ymin0, xmax0, ymax0 = miscs.divide_bbox((0, 0, imgshp0[1], imgshp0[0]), min_num_blocks=divide_N)
    xmin1, ymin1, xmax1, ymax1 = miscs.divide_bbox((0, 0, imgshp1[1], imgshp1[0]), min_num_blocks=divide_N)
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
    given two images with rectangular mask, return the displacement vectors on a
    grid of sample points. Mostly for stitching matching.
    """
    sigma = kwargs.get('sigma', 0.0)
    mask0 = kwargs.get('mask0', None)
    mask1 = kwargs.get('mask1', None)
    conf_mode = kwargs.get('conf_mode', FFT_CONF_MIRROR)
    conf_thresh = kwargs.get('conf_thresh', 0.3)
    err_thresh = kwargs.get('err_thresh', 5)
    err_tol = kwargs.get('err_tol', 2)
    coarse_downsample = kwargs.get('coarse_downsample', 1)
    fine_downsample = kwargs.get('fine_downsample', 1)
    spacing = np.sort(kwargs.get('spacing', [100]))
    min_num_blocks = kwargs.get('min_num_blocks', 2)
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
        img0_g = miscs.masked_dog_filter(img0_g, sigma*coarse_downsample, mask=mask0_g)
        img1_g = miscs.masked_dog_filter(img1_g, sigma*coarse_downsample, mask=mask1_g)
    tx0, ty0, conf0 = global_translation_matcher(img0_g, img1_g, conf_mode=conf_mode,
        conf_thresh=conf_thresh)
    if conf0 < conf_thresh:
        return conf_thresh, None, None
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
            img0_f = miscs.masked_dog_filter(img0_f, sigma*fine_downsample, mask=mask0_f)
            img1_f = miscs.masked_dog_filter(img1_f, sigma*fine_downsample, mask=mask1_f)
    tx0 = tx0 * fine_downsample / coarse_downsample
    ty0 = ty0 * fine_downsample / coarse_downsample
    err_thresh = err_thresh * fine_downsample
    err_tol = err_tol * fine_downsample
    img_loader0 = dal.StreamImageLoader(img0_f)
    img_loader1 = dal.StreamImageLoader(img1_f)
    min_spacing = np.min(spacing)
    mesh0 = mesh.Mesh.from_bbox(img_loader0.bounds, cartesian=False,
        mesh_size=min_spacing, min_num_blocks=min_num_blocks, uid=0)
    mesh1 = mesh.Mesh.from_bbox(img_loader1.bounds, cartesian=False,
        mesh_size=min_spacing, min_num_blocks=min_num_blocks, uid=1)
    mesh0.apply_translation((tx0, ty0), MESH_GEAR_FIXED)
    current_spacing = np.max(spacing)
    initialized = False
    opt = optimizer.SLM([mesh0, mesh1])
    for sp in spacing[::-1]:
        if sp > current_spacing:
            continue
        bbox0 = mesh0.bbox(gear=MESH_GEAR_MOVING)
        bbox1 = mesh1.bbox(gear=MESH_GEAR_MOVING)
        bbox, valid = miscs.intersect_bbox(bbox0, bbox1)
        if not valid:
            return 0, None, None
        if sp == min_spacing:
            mnb = min_num_blocks
        else:
            mnb = 1
        xstt, ystt, xend, yend = miscs.divide_bbox(bbox, block_size=sp, min_num_blocks=mnb)
        render0 = renderer.MeshRenderer.from_mesh(mesh0, image_loader=img_loader0)
        render1 = renderer.MeshRenderer.from_mesh(mesh1, image_loader=img_loader1)
        stack0 = []
        stack1 = []
        for x0, y0, x1, y1 in zip(xstt, ystt, xend, yend):
            stack0.append(render0.crop((x0, y0, x1, y1)))
            stack1.append(render1.crop((x0, y0, x1, y1)))
        dx, dy, conf = xcorr_fft(np.stack(stack0, axis=0), np.stack(stack1, axis=0),
            conf_mode=conf_mode, pad=(not initialized))
        xy0 = np.stack(((xstt+xend-1-dx)/2, (ystt+yend-1-dy)/2), axis=-1)
        xy1 = np.stack(((xstt+xend-1+dx)/2, (ystt+yend-1+dy)/2), axis=-1)
        if np.all(conf <= conf_thresh):
            if not initialized:
                return 0, None, None
            else:
                break
        opt.clear_links()
        xy0 = xy0[conf > conf_thresh]
        xy1 = xy1[conf > conf_thresh]
        wt = conf[conf > conf_thresh]
        opt.add_link_from_coordinates(0, 1, xy0, xy1,
                        gear=(MESH_GEAR_MOVING, MESH_GEAR_MOVING), weight=wt,
                        check_duplicates=False)
        opt.optimize_linear(tol=1e-6)
        if err_thresh > 0:
            opt.set_link_residue_threshold(err_thresh)


def iterative_mesh_matcher(mesh0, mesh1, image_loader0, image_loader1, **kwargs):
    sigma = kwargs.get('sigma', 0.0)
    conf_mode = kwargs.get('conf_mode', FFT_CONF_MIRROR)
    conf_thresh = kwargs.get('conf_thresh', 0.3)
    err_thresh = kwargs.get('err_thresh', 0)
    err_tol = kwargs.get('err_tol', 1e-5)
    spacing = np.array(kwargs.get('spacing', [100]), copy=False)
    min_num_blocks = kwargs.get('min_num_blocks', 2)
    shrink_factor = kwargs.get('shrink_factor', 1)
    distributor = kwargs.get('distributor', BLOCKDIST_CART_BBOX)
    render_mode = kwargs.get('render_mode', RENDER_FULL)
    # if any spacing value smaller than 1, means they are relative to longer side
    if np.any(spacing < 1):
        bbox0 = mesh0.bbox(gear=MESH_GEAR_MOVING)
        bbox1 = mesh1.bbox(gear=MESH_GEAR_MOVING)
        bbox, valid = miscs.intersect_bbox(bbox0, bbox1)
        if not valid:
            return None
        wd0 = bbox[2] - bbox[0]
        ht0 = bbox[3] - bbox[1]
        lside = max(wd0, ht0)
        spacing[spacing < 1] *= lside
    current_spacing = np.max(spacing)
    initialized = False
    opt = optimizer.SLM([mesh0, mesh1])
    c_pointer = 0
    sp = np.max(spacing)
    min_spacing = np.min(spacing)
    while c_pointer < spacing.size:
        if sp == min_spacing:
            mnb = min_num_blocks
        else:
            mnb = 1
        if distributor == BLOCKDIST_CART_BBOX:
            block_indices = distributor_cartesian_bbox(mesh0, mesh1, sp,
                min_num_blocks=mnb, shrink_factor=shrink_factor)
        else:
            raise ValueError
        if block_indices is None:
            return None
        render0 = renderer.MeshRenderer.from_mesh(mesh0, image_loader=image_loader0)
        render1 = renderer.MeshRenderer.from_mesh(mesh1, image_loader=image_loader1)
        stack0 = []
        stack1 = []
        xy_ctr = []
        for x0, y0, x1, y1 in zip(*block_indices):
            img0 = render0.crop((x0, y0, x1, y1), mode=render_mode, log_sigma=sigma)
            if img0 is None:
                continue
            img1 = render1.crop((x0, y0, x1, y1), mode=render_mode, log_sigma=sigma)
            if img1 is None:
                continue
            stack0.append(img0)
            stack1.append(img1)
            xy_ctr.append(((x0+x1-1)/2, (y0+y1-1)/2))
        dx, dy, conf = xcorr_fft(np.stack(stack0, axis=0), np.stack(stack1, axis=0),
            conf_mode=conf_mode, pad=(not initialized))
        xy_ctr = np.array(xy_ctr)
        dxy = np.stack((dx, dy), axis=-1)
        xy0 = xy_ctr - dxy/2
        xy1 = xy_ctr + dxy/2
        if np.all(conf <= conf_thresh):
            if not initialized:
                return None
            else:
                break
        opt.clear_links()
        xy0 = xy0[conf > conf_thresh]
        xy1 = xy1[conf > conf_thresh]
        wt = conf[conf > conf_thresh]
        max_dis = np.max(np.sum((xy0 - xy1) ** 2, axis=-1)) ** 0.5
        min_block_size = 4 * max_dis
        next_pos = np.searchsorted(spacing, min_block_size)
        opt.add_link_from_coordinates(mesh0.uid, mesh1.uid, xy0, xy1,
                        gear=(MESH_GEAR_MOVING, MESH_GEAR_MOVING), weight=wt,
                        check_duplicates=False)
        opt.optimize_linear(tol=err_tol, batch_num_matches=np.inf)
        if err_thresh > 0:
            opt.set_link_residue_threshold(err_thresh)
            weight_modified, _ = opt.adjust_link_weight_by_residue()
            opt.optimize_linear(tol=err_tol, batch_num_matches=np.inf)
        


## ----------------- matching block distributors --------------------------- ##
def distributor_cartesian_bbox(mesh0, mesh1, spacing, **kwargs):
    gear = kwargs.get('gear', MESH_GEAR_MOVING)
    min_num_blocks = kwargs.get('min_num_blocks', 1)
    shrink_factor = kwargs.get('shrink_factor', 1)
    bbox0 = mesh0.bbox(gear=gear)
    bbox1 = mesh1.bbox(gear=gear)
    bbox, valid = miscs.intersect_bbox(bbox0, bbox1)
    if not valid:
        return None
    xstt, ystt, xend, yend = miscs.divide_bbox(bbox, block_size=spacing,
        min_num_blocks=min_num_blocks, shrink_factor=shrink_factor)
    return xstt, ystt, xend, yend
    