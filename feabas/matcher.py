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
    """
    mask0 = kwargs.get('mask0', None)
    mask1 = kwargs.get('mask1', None)
    sigma = kwargs.get('sigma', 0)
    if sigma > 0:
        img0 = miscs.masked_dog_filter(img0, sigma, mask=mask0)
        img1 = miscs.masked_dog_filter(img1, sigma, mask=mask1)
    imgshp0 = img0.shape[-2:]
    imgshp1 = img1.shape[-2:]
    fftshp = [next_fast_len(s0 + s1 - 1) for s0, s1 in zip(imgshp0, imgshp1)]
    F0 = fft.rfft2(img0, s=fftshp)
    F1 = fft.rfft2(img1, s=fftshp)
    F = np.conj(F0) * F1
    C = fft.irfft2(F)