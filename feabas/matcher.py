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
        normalize (bool): whether to normalize the cross-correlation.
    Return:
        dy, dx: the displacement of the peak of the cross-correlation, so that
            the center of img1 + (dx, dy) corresponds to the center of img0.
        conf: the confidence value of the cross-correlation between 0 and 1.
    """
    sigma = kwargs.get('sigma', 0)
    mask0 = kwargs.get('mask0', None)
    mask1 = kwargs.get('mask1', None)
    normalize = kwargs.get('normalize', False)
    if sigma > 0:
        img0 = miscs.masked_dog_filter(img0, sigma, mask=mask0)
        img1 = miscs.masked_dog_filter(img1, sigma, mask=mask1)
    imgshp0 = img0.shape[-2:]
    imgshp1 = img1.shape[-2:]
    fftshp = [next_fast_len(s0 + s1 - 1) for s0, s1 in zip(imgshp0, imgshp1)]
    F0 = fft.rfft2(img0, s=fftshp)
    F1 = fft.rfft2(img1, s=fftshp)
    C = fft.irfft2(np.conj(F0) * F1)
    C = C.reshape(-1, np.prod(fftshp))
    if normalize:
        if mask0 is None:
            mask0 = np.ones_like(img0)
        else:
            mask0 = 1 - mask0
        if mask1 is None:
            mask1 = np.ones_like(img1)
        else:
            mask1 = 1 - mask1
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
    return dy, dx, conf
