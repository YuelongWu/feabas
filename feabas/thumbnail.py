import cv2
import numpy as np
from scipy.spatial import KDTree
from skimage.feature import peak_local_max

from feabas import common


class KeyPoints:
    """
    class to represents keypoints in feature matching.
    """
    def __init__(self, xy=None, response=None, class_id=None, offset=(0,0), descriptor=None):
        if xy is None:
            self.xy = np.empty((0, 2), dtype=np.float32)
        else:
            self.xy = np.array(xy)
            if not np.issubdtype(xy.dtype, np.floating):
                self.xy = self.xy.astype(np.float32)
        self.offset = np.array(offset)
        self._response = response
        self._class_id = class_id
        self.des = descriptor


    @classmethod
    def combine_keypoints(cls, kps):
        if len(kps) == 1:
            return kps[0]
        computed = np.any([kp.des is not None for kp in kps])
        offset0 = kps[0].offset
        xy_list = []
        response_list = []
        class_id_list = []
        descriptor_list = []
        for kp in kps:
            if computed and (kp.des is None):
                continue
            xy_list.append(kp.xy + kp.offset - offset0)
            response_list.append(kp.response)
            class_id_list.append(kp.class_id)
            descriptor_list.append(kp.des)
        xy = np.concatenate(xy_list, axis=0)
        response = np.concatenate(response_list, axis=0)
        class_id = np.concatenate(class_id_list, axis=0)
        if computed:
            descriptor = np.concatenate(descriptor_list, axis=0)
        else:
            descriptor = None
        return cls(xy=xy, response=response, class_id=class_id, offset=offset0, descriptor=descriptor)


    def filter_keypoints(self, indx, inplace=True):
        xy = self.xy[indx]
        if self._response is not None:
            response = self.response[indx]
        else:
            response = None
        if self._class_id is not None:
            class_id = self.class_id[indx]
        else:
            class_id = None
        if self.des is not None:
            descriptor = self.des[indx]
        else:
            descriptor = None
        if inplace:
            self.xy = xy
            self._response = response
            self._class_id = class_id
            self.des = descriptor
            return self
        else:
            return self.__class__(xy=xy, response=response, class_id=class_id, offset=self.offset, descriptor=descriptor)


    @property
    def num_points(self):
        return self.xy.shape[0]


    @property
    def response(self):
        if self._response is None:
            return np.zeros(self.num_points, dtype=np.float32)
        else:
            return self._response


    @property
    def class_id(self):
        if self._class_id is None:
            return np.ones(self.num_points, dtype=np.int16)
        else:
            return self._class_id



def detect_extrema_log(img, mask=None, offset=(0,0), **kwargs):
    sigma = kwargs.get('sigma', 3.5)
    min_spacing = kwargs.get('min_spacing', 15)
    intensity_thresh = kwargs.get('intensity_thresh', 0.05)
    num_features = kwargs.get('num_features', np.inf)
    if mask is None:
        mask = np.ones_like(img, dtype=np.uint8)
    elif not np.issubdtype(mask.dtype, np.integer):
        mask = mask.astype(np.int16, copy=False)
    if sigma > 0:
        img = common.masked_dog_filter(img, sigma, mask=(mask>0))
    if np.ptp(img, axis=None) == 0:
        return KeyPoints()
    xy = peak_local_max(np.abs(img), min_distance=min_spacing,
                        threshold_rel=intensity_thresh, labels=mask,
                        num_peaks=num_features)[:,::-1]
    response = img[xy[:,1], xy[:,0]]
    sidx = np.argsort(np.abs(response))[::-1]
    xy = xy[sidx] + np.array(offset)
    response = response[sidx]
    class_id = mask[xy[:,1], xy[:,0]]
    return KeyPoints(xy=xy, response=response, class_id=class_id, offset=offset)


def extract_LRadon_feature(img, kps, offset=None, **kwargs):
    proj_num = kwargs.get('proj_num', 6)
    beam_num = kwargs.get('beam_num', 8)
    beam_wd = kwargs.get('beam_wd', 3)
    beam_radius = kwargs.get('beam_radius', 40)
    max_batchsz = 16300
    if kps.des is not None:
        return kps
    if kps.num_points == 0:
        kps.des = np.empty((0, beam_num, proj_num*2), dtype=np.float32)
        return kps
    if not np.issubdtype(img.dtype, np.floating):
        img = img.astype(np.float32)
    imgf = cv2.GaussianBlur(img, (0,0), beam_wd/1.18)
    xy = kps.xy
    if offset is None:
        xy0 = xy - kps.offset
    else:
        xy0 = xy - np.array(offset)
    descriptor0 = []    # 0-pi
    descriptor1 = []    # pi-2pi
    for theta in np.linspace(0, np.pi, num=proj_num, endpoint=False):
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, s], [-s, c]])
        xy1 = xy0 @ R
        xy1_min = np.floor(xy1.min(axis=0) - beam_radius) - 4
        xy1_max = np.ceil(xy1.max(axis=0) + beam_radius) + 4
        dst_sz = (xy1_max - xy1_min).astype(np.int32)
        A = np.concatenate((R, -xy1_min.reshape(-1,2)), axis=0)
        img_r = cv2.warpAffine(imgf, A.T, (dst_sz[0], dst_sz[1]))
        img_rf = cv2.boxFilter(img_r, -1, (1, beam_radius))
        x1, y1 = xy1[:,0] - xy1_min[0], xy1[:,1] - xy1_min[1]
        dx = np.linspace(0, beam_num*beam_wd*2, num=beam_num*2, endpoint=False)
        dx = dx - dx.mean()
        xx = (x1.reshape(-1,1) + dx).astype(np.float32)
        yy = (y1.reshape(-1,1) + np.zeros_like(dx)).astype(np.float32)
        if xx.shape[0] < max_batchsz:
            des0 = cv2.remap(img_rf, xx, yy, interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            des0_list = []
            for stt_idx in np.arange(0, xx.shape[0], max_batchsz):
                xx_b = xx[stt_idx:(stt_idx+max_batchsz)]
                yy_b = yy[stt_idx:(stt_idx+max_batchsz)]
                des0_b = cv2.remap(img_rf, xx_b, yy_b, interpolation=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                des0_list.append(des0_b)
            des0 = np.concatenate(des0_list, axis=0)
        descriptor0.append(des0[:, :beam_num])
        descriptor1.append(des0[:, -1:(-beam_num-1):-1])
    descriptor = np.concatenate((np.stack(descriptor0, axis=-1),
                                 np.stack(descriptor1, axis=-1)), axis=-1)
    kps.des = descriptor
    return kps
