import numpy as np
from scipy import fft, ndimage
from scipy.fftpack import next_fast_len

from feabas import optimizer, dal, miscs
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
    ratio = imgshp[0]/imgshp[1]
    divide_sel = np.argmin(np.abs([ratio*4, ratio, ratio/4]))
    if divide_sel == 0:
        divide_N = (1, 4)
    elif divide_sel == 1:
        divide_N = (2, 2)
    else:
        divide_N = (4, 1)
    dx0 = int(np.ceil(imgshp0[1] / divide_N[1]))
    dy0 = int(np.ceil(imgshp0[0] / divide_N[0]))
    dx1 = int(np.ceil(imgshp1[1] / divide_N[1]))
    dy1 = int(np.ceil(imgshp1[0] / divide_N[0]))
    x0 = np.round(np.linspace(0, imgshp0[1] - dx0, num=divide_N[1], endpoint=True)).astype(np.int32)
    y0 = np.round(np.linspace(0, imgshp0[0] - dy0, num=divide_N[0], endpoint=True)).astype(np.int32)
    x1 = np.round(np.linspace(0, imgshp1[1] - dx1, num=divide_N[1], endpoint=True)).astype(np.int32)
    y1 = np.round(np.linspace(0, imgshp1[0] - dy1, num=divide_N[0], endpoint=True)).astype(np.int32)
    xx0, yy0 = np.meshgrid(x0, y0)
    xx1, yy1 = np.meshgrid(x1, y1)
    xx0, yy0, xx1, yy1 = xx0.ravel(), yy0.ravel(), xx1.ravel(), yy1.ravel()
    stack0 = []
    stack1 = []
    for k in range(xx0.size):
        stack0.append(img0[yy0[k]:(yy0[k]+dy0), xx0[k]:(xx0[k]+dx0)])
        stack1.append(img1[yy1[k]:(yy1[k]+dy1), xx1[k]:(xx1[k]+dx1)])
    btx, bty, bconf = xcorr_fft(np.stack(stack0, axis=0), np.stack(stack1, axis=0), conf_mode=conf_mode, pad=True)
    k_best = np.argmax(bconf)
    if bconf[k_best] <= conf:
        tx = btx[k_best] + xx1[k_best] + dx1/2 - xx0[k_best] - dx0/2 - (imgshp1[1] - imgshp0[1]) / 2
        ty = bty[k_best] + yy1[k_best] + dy1/2 - yy0[k_best] - dy0/2 - (imgshp1[0] - imgshp0[0]) / 2
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
    spacing = kwargs.get('spacing', [100])
    if sigma > 0:
        img0 = miscs.masked_dog_filter(img0, sigma, mask=mask0)
        img1 = miscs.masked_dog_filter(img1, sigma, mask=mask1)