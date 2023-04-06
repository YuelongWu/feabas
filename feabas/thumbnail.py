import cv2
import numpy as np
from scipy.spatial import KDTree
from scipy.fft import rfft, irfft
from skimage.feature import peak_local_max

from feabas import common


class KeyPoints:
    """
    class to represents keypoints in feature matching.
    """
    def __init__(self, xy=None, response=None, class_id=None, offset=(0,0), descriptor=None, angle=None, angle_aligned=False):
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
        self._angle = angle
        self.angle_aligned = angle_aligned


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
        angle_list = []
        angle_aligned = None
        for kp in kps:
            if computed and (kp.des is None):
                continue
            if angle_aligned is None:
                angle_aligned = kp.angle_aligned
            xy_list.append(kp.xy + kp.offset - offset0)
            response_list.append(kp.response)
            class_id_list.append(kp.class_id)
            if angle_aligned:
                kp.align_angle()
            else:
                kp.reset_angle()
            descriptor_list.append(kp.des)
            angle_list.append(kp._angle)
        xy = np.concatenate(xy_list, axis=0)
        response = np.concatenate(response_list, axis=0)
        class_id = np.concatenate(class_id_list, axis=0)
        if computed:
            descriptor = np.concatenate(descriptor_list, axis=0)
            angle = np.concatenate(angle_list, axis=0)
        else:
            descriptor = None
            angle = None
        return cls(xy=xy, response=response, class_id=class_id, offset=offset0,
                   descriptor=descriptor, angle=angle, angle_aligned=angle_aligned)


    def filter_keypoints(self, indx, include_descriptor=True, inplace=True):
        xy = self.xy[indx]
        if self._response is not None:
            response = self.response[indx]
        else:
            response = None
        if self._class_id is not None:
            class_id = self.class_id[indx]
        else:
            class_id = None
        if include_descriptor and (self.des is not None):
            descriptor = self.des[indx]
        else:
            descriptor = None
        if self._angle is not None:
            angle = self.angle[indx]
        else:
            angle = None
        if inplace:
            self.xy = xy
            self._response = response
            self._class_id = class_id
            self.des = descriptor
            self._angle = angle
            return self
        else:
            return self.__class__(xy=xy, response=response, class_id=class_id,
                                  offset=self.offset, descriptor=descriptor,
                                  angle=angle, angle_aligned=self.angle_aligned)


    def align_angle(self):
        if (self.angle_aligned) or (self.des is None):
            return self.des
        proj_num = self.des.shape[-1]
        F = rfft(self.des, n=proj_num, axis=-1)
        omega = np.linspace(0, proj_num, num=proj_num, endpoint=False)
        angle_offset = self._angle.reshape(-1,1) * omega *1j
        F = F * np.exp(angle_offset.reshape(-1, 1, proj_num))[:,:,:F.shape[-1]]
        self.des = irfft(F, n=proj_num, axis=-1)
        self.angle_aligned = True
        return self.des


    def reset_angle(self):
        if (self.angle_aligned) or (self.des is None):
            return self.des
        proj_num = self.des.shape[-1]
        F = rfft(self.des, n=proj_num, axis=-1)
        omega = np.linspace(0, proj_num, num=proj_num, endpoint=False)
        angle_offset = -self._angle.reshape(-1,1) * omega *1j
        F = F * np.exp(angle_offset.reshape(-1, 1, proj_num))[:,:,:F.shape[-1]]
        self.des = irfft(F, n=proj_num, axis=-1)
        self.angle_aligned = False
        return self.des


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


    @property
    def angle(self):
        if self._angle is None:
            return np.zeros(self.num_points, dtype=np.float32)
        else:
            return self._angle



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
    beam_radius = kwargs.get('beam_radius', 15)
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
    dx = np.linspace(0, beam_num*beam_wd*2, num=beam_num*2, endpoint=False)
    dx = dx - dx.mean()
    angle_wt = dx[:beam_num] ** 2
    angle_wt = angle_wt / np.sum(angle_wt)
    angle_vec = np.zeros((kps.num_points, 2), dtype=np.float32)
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
        xx = (x1.reshape(-1,1) + dx).astype(np.float32)
        yy = (y1.reshape(-1,1) + np.zeros_like(dx)).astype(np.float32)
        if xx.shape[0] < max_batchsz:
            des_t = cv2.remap(img_rf, xx, yy, interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            des_t_list = []
            for stt_idx in np.arange(0, xx.shape[0], max_batchsz):
                xx_b = xx[stt_idx:(stt_idx+max_batchsz)]
                yy_b = yy[stt_idx:(stt_idx+max_batchsz)]
                des_t_b = cv2.remap(img_rf, xx_b, yy_b, interpolation=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                des_t_list.append(des_t_b)
            des_t = np.concatenate(des_t_list, axis=0)
        des0 = des_t[:, :beam_num]
        des1 = des_t[:, -1:(-beam_num-1):-1]
        angle_vec = angle_vec + np.sum((des0-des1) * angle_wt, axis=-1).reshape(-1,1) * np.array([s, c])
        descriptor0.append(des0)
        descriptor1.append(des1)
    angle_in_rad = np.arctan2(angle_vec[:,0], angle_vec[:,1])
    descriptor = np.concatenate((np.stack(descriptor0, axis=-1),
                                 np.stack(descriptor1, axis=-1)), axis=-1)
    mm = np.nanmean(descriptor, axis=(1,2), keepdims=True)
    ss = np.nanstd(descriptor, axis=(1,2), keepdims=True).clip(1e-6, None)
    descriptor = (descriptor - mm) / ss
    kps.des = descriptor
    kps._angle = angle_in_rad
    kps.angle_aligned = False
    return kps



def match_LRadon_feature(kps0, kps1, **kwargs):
    exhaustive = kwargs.get('exhaustive', False)
    conf_thresh = kwargs.get('conf_thresh', 0.5)
    if kps0.num_points > kps1.num_points:
        kps0, kps1 = kps1, kps0
        flipped = True
    else:
        flipped = False
    if exhaustive:
        des0 = kps0.reset_angle()
        des1 = kps1.reset_angle()
        norm_fact = 1 / (des0.shape[-1] * des0.shape[-2])
        F0 = rfft(des0, n=des0.shape[-1], axis=-1)
        F1 = rfft(des1, n=des1.shape[-1], axis=-1)
        F = np.einsum('mjk,njk->mnk', F0, np.conj(F1), optimize=True)
        C0 = irfft(F, n=des0.shape[-1], axis=-1)
        C = norm_fact * C0.max(axis=-1)
    else:
        des0 = kps0.align_angle()
        des1 = kps1.align_angle()
        norm_fact = 1 / (des0.shape[-1] * des0.shape[-2])
        C = des0.reshape(des0.shape[0], -1) @ des1.reshape(des1.shape[0], -1).T
        C = norm_fact * C
    idx0 = np.arange(C.shape[0])
    idx1 = np.argmax(C, axis=-1)
    conf0 = C[idx0, idx1]
    C[idx0, idx1] = -1
    conf1 = np.max(C, axis=-1)
    ROD = 1 - (conf1 / conf0)
    conf = ROD * conf0 ** 2
    if flipped:
        idx0, idx1 = idx1, idx0,
        kps0, kps1 = kps1, kps0
    idx0 = idx0[conf0 > conf_thresh]
    idx1 = idx1[conf0 > conf_thresh]
    conf = conf[conf0 > conf_thresh]
    kps0_out = kps0.filter_keypoints(idx0, include_descriptor=False, inplace=False)
    kps1_out = kps1.filter_keypoints(idx1, include_descriptor=False, inplace=False)
    return kps0_out, kps1_out, conf
