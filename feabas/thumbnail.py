import numpy as np
from scipy.spatial import KDTree
from skimage.feature import peak_local_max

from feabas import common


def detect_extrema_log(img, mask=None, offset=(0,0), **kwargs):
    sigma = kwargs.get('sigma', 3.5)
    min_spacing = kwargs.get('min_spacing', 15)
    intensity_thresh = kwargs.get('intensity_thresh', 0.05)
    num_features = kwargs.get('num_features', np.inf)
    kps = {}
    if mask is None:
        mask = np.ones_like(img, dtype=np.uint8)
    elif not np.issubdtype(mask.dtype, np.integer):
        mask = mask.astype(np.int16, copy=False)
    if sigma > 0:
        img = common.masked_dog_filter(img, sigma, mask=(mask>0))
    if np.ptp(img, axis=None) == 0:
        return None
    xy = peak_local_max(np.abs(img), min_distance=min_spacing,
                        threshold_rel=intensity_thresh, labels=mask,
                        num_peaks=num_features)[:,::-1]
    response = img[xy[:,1], xy[:,0]]
    sidx = np.argsort(np.abs(response))[::-1]
    kps['xy'] = xy[sidx] + np.array(offset)
    kps['response'] = response[sidx]
    kps['label'] = mask[xy[:,1], xy[:,0]]
    return kps


def extract_LRadon_feature(img, kps, **kwargs):
    if 'descriptor' in kps:
        return kps
    

