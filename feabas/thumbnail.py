import cv2
import numpy as np
import h5py
import os
from scipy.fft import rfft, irfft
from shapely import concave_hull, MultiPoint, intersects_xy
import shapely.geometry as shpgeo
from shapely.affinity import affine_transform
from skimage.feature import peak_local_max
from itertools import combinations

from feabas import common, logging
from feabas.spatial import fit_affine, Geometry, scale_coordinates
from feabas.mesh import Mesh
from feabas.optimizer import SLM
import feabas.constant as const
from feabas.dal import StreamLoader
from feabas.matcher import section_matcher
from feabas.aligner import read_matches_from_h5


DEFAULT_FEATURE_SPACING = 15


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
        angle_offset = self.angle.reshape(-1,1) * omega *1j
        F = F * np.exp(angle_offset.reshape(-1, 1, proj_num))[:,:,:F.shape[-1]]
        self.des = irfft(F, n=proj_num, axis=-1)
        self.angle_aligned = True
        return self.des


    def reset_angle(self):
        if (not self.angle_aligned) or (self.des is None):
            return self.des
        proj_num = self.des.shape[-1]
        F = rfft(self.des, n=proj_num, axis=-1)
        omega = np.linspace(0, proj_num, num=proj_num, endpoint=False)
        angle_offset = -self.angle.reshape(-1,1) * omega *1j
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


class KeyPointMatches:
    def __init__(self, xy0, xy1, weight=None, class_id0=None, class_id1=None, angle0=None, angle1=None):
        assert xy0.shape[0] == xy1.shape[0]
        self.xy0 = xy0
        self.xy1 = xy1
        self._weight = weight
        self._class_id0 = class_id0
        self._class_id1 = class_id1
        self._angle0 = angle0
        self._angle1 = angle1

    def copy(self):
        xy0 = self.xy0
        xy1 = self.xy1
        weight = self._weight
        class_id0 = self._class_id0
        class_id1 = self._class_id1
        angle0 = self._angle0
        angle1 = self._angle1
        return self.__class__(xy0, xy1, weight, class_id0=class_id0, class_id1=class_id1, angle0=angle0, angle1=angle1)

    @classmethod
    def from_keypoints(cls, kps0, kps1, weight=None):
        xy0 = kps0.xy + kps0.offset
        xy1 = kps1.xy + kps1.offset
        class_id0 = kps0._class_id
        class_id1 = kps1._class_id
        angle0 = kps0._angle
        angle1 = kps1._angle
        return cls(xy0, xy1, weight, class_id0=class_id0, class_id1=class_id1, angle0=angle0, angle1=angle1)

    def filter_match(self, indx, inplace=True):
        if inplace:
            mtch = self
        else:
            mtch = self.copy()
        if indx is None:
            return mtch
        mtch.xy0 = mtch.xy0[indx]
        mtch.xy1 = mtch.xy1[indx]
        if mtch._weight is not None:
            mtch._weight = mtch._weight[indx]
        if mtch._class_id0 is not None:
            mtch._class_id0 = mtch._class_id0[indx]
        if mtch._class_id1 is not None:
            mtch._class_id1 = mtch._class_id1[indx]
        if mtch._angle0 is not None:
            mtch._angle0 = mtch._angle0[indx]
        if mtch._angle1 is not None:
            mtch._angle1 = mtch._angle1[indx]
        return mtch

    def sort_match_by_weight(self):
        if self._weight is None:
            return self
        indx = np.argsort(self._weight, kind='stable')
        if not np.all(indx == np.arange(indx.size)):
            self.filter_match(indx[::-1])
        return self

    def reset_weight(self, val=None):
        if val is not None:
            self._weight = np.full(self.num_points, val, dtype=np.float32)
        else:
            self._weight = None

    @property
    def num_points(self):
        return self.xy0.shape[0]

    @property
    def weight(self):
        if self._weight is None:
            return np.ones(self.num_points, dtype=np.float32)
        else:
            return self._weight

    @property
    def class_id0(self):
        if self._class_id0 is None:
            return np.ones(self.num_points, dtype=np.int16)
        else:
            return self._class_id0
        
    @property
    def class_id1(self):
        if self._class_id1 is None:
            return np.ones(self.num_points, dtype=np.int16)
        else:
            return self._class_id1

    @property
    def angle0(self):
        if self._angle0 is None:
            return np.zeros(self.num_points, dtype=np.float32)
        else:
            return self._angle0

    @property
    def angle1(self):
        if self._angle1 is None:
            return np.zeros(self.num_points, dtype=np.float32)
        else:
            return self._angle1


def prepare_image(img, mask=None, **kwargs):
    uid = kwargs.get('uid', None)
    compute_keypoints = kwargs.get('compute_keypoints', True)
    detect_settings = kwargs.get('detect_settings', {}).copy()
    extract_settings = kwargs.get('extract_settings', {}).copy()
    mesh_size = kwargs.get('mesh_size', DEFAULT_FEATURE_SPACING * 2)
    scale = kwargs.get('scale', 1.0)
    if isinstance(img, dict):
        out = img
        img = out['image']
        mask = out['mask']
    else:
        out = {'image': img, 'mask': mask}
    if scale not in out:
        if scale != 1:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if mask is not None:
                mask_dtype = mask.dtype
                if mask_dtype == np.dtype('bool'):
                    mask = mask.astype(np.uint8)
                mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                mask = mask.astype(mask_dtype, copy=False)
        scale_info = {'image': img, 'mask': mask}
        regions = {}
        if mask is None:
            imght, imgwd = img.shape[:2]
            regions[1] = shpgeo.box(0,0,imgwd, imght)
            mesh = Mesh.from_bbox((0,0,imgwd, imght), cartesian=True,
                                    mesh_size=mesh_size, uid=uid)
        else:
            meshes_stg = []
            for lb in np.unique(mask[mask>0]):
                G0 = Geometry.from_image_mosaic(255 - 255 * (mask == lb).astype(np.uint8),
                                                region_names={'default':0, 'exclude': 255})
                G0.simplify(region_tol={'exclude':1.5}, inplace=True)
                regions[lb] = G0.region_default
                meshes_stg.append(Mesh.from_PSLG(**G0.PSLG(), mesh_size=mesh_size,
                                                min_mesh_angle=20, uid=-1.0))
            mesh = Mesh.combine_mesh(meshes_stg, uid=uid)
        scale_info['regions'] = regions
        scale_info['mesh'] = mesh
        out[scale] = scale_info
    if compute_keypoints and ('kps' not in out[scale]):
        img = out[scale]['image']
        mask = out[scale]['mask']
        if (mask is not None) and not (np.all(mask>0, axis=None)):
            mm = np.mean(img[mask>0])
            img = (img * (mask > 0) + mm * (mask == 0)).astype(img.dtype)
        kps = detect_extrema_log(img, mask=mask, **detect_settings)
        kps = extract_LRadon_feature(img, kps, **extract_settings)
        out[scale]['kps'] = kps
    return out
            


def match_two_thumbnails_LRadon(img0, img1, mask0=None, mask1=None, **kwargs):
    affine_only = kwargs.get('affine_only', False)
    scale = kwargs.get('scale', 1.0)
    detect_settings = kwargs.get('detect_settings', {}).copy()
    matching_settings = kwargs.get('matching_settings', {}).copy()
    feature_spacing = detect_settings.get('min_spacing', DEFAULT_FEATURE_SPACING)
    strain_filter_settings = kwargs.get('strain_filter_settings', {}).copy()
    ransac_filter_settings = kwargs.get('ransac_filter_settings', {}).copy()
    matchnum_thresh = kwargs.get('matchnum_thresh', 64)
    elastic_dis_tol = kwargs.get('elastic_dis_tol', feature_spacing / 5)
    ransac_filter_settings.setdefault('dis_tol',feature_spacing / 3)
    info0 = prepare_image(img0, mask=mask0, **kwargs)[scale]
    mesh0 = info0['mesh'].copy()
    info1 = prepare_image(img1, mask=mask1, **kwargs)[scale]
    mesh1 = info1['mesh'].copy()
    kps0, kps1 = info0['kps'], info1['kps']
    mesh0.uid = 0.0
    mesh1.uid = 1.0
    optm = SLM([mesh0, mesh1], stiffness_lambda=1.0)
    optm.divide_disconnected_submeshes(prune_links=False)
    xy0 = []
    xy1 = []
    settled_link = {}
    D0 = None
    exclude_class = []
    if affine_only:
        clsid0 = np.unique(kps0.class_id)
        clsid1 = np.unique(kps1.class_id)
        num_cls_cmb = clsid0.size * clsid1.size
    filter_in_place = False
    while True:
        modified = False
        mtch, D0 = match_LRadon_feature(kps0, kps1, D=D0, exclude_class=exclude_class, **matching_settings)
        mtch, _ = filter_match_pairwise_strain(mtch, **strain_filter_settings)
        if affine_only:
            mtch, _ = filter_match_global_ransac(mtch, **ransac_filter_settings)
            if (mtch is None) or (mtch.num_points == 0):
                match_list = None
            else:
                match_list = [mtch]
                clsids = (mtch.class_id0[0], mtch.class_id1[0])
                exclude_class.append(clsids)
        else:
            match_list, _ = filter_match_sequential_ransac(mtch, **ransac_filter_settings)
        used_matches = []
        if match_list is None:
            break
        for mtch in match_list:
            if mtch.num_points == 0:
                continue
            optm.clear_links()
            optm.add_link_from_coordinates(0.0, 1.0, mtch.xy0, mtch.xy1,
                                           check_duplicates=False,
                                           weight=np.full(mtch.num_points, 0.05),
                                           name='staging')
            staging_link = optm.links.copy()
            if (len(settled_link) > 0) and (mtch.num_points < matchnum_thresh):
                for lnk in settled_link.values():
                    optm.add_link(lnk, check_relevance=False, check_duplicates=False)
                optm.optimize_affine_cascade(target_gear=const.MESH_GEAR_FIXED)
                optm.anneal(gear=(const.MESH_GEAR_FIXED, const.MESH_GEAR_MOVING), mode=const.ANNEAL_COPY_EXACT)
                optm.clear_equation_terms()
                optm.optimize_linear(tol=1.0e-5, targt_gear=const.MESH_GEAR_MOVING)
                valid_num = 0
                xy0_t_list = []
                xy1_t_list = []
                for lnk in optm.links:
                    if lnk.name == 'staging':
                        lnk.set_hard_residue_filter(elastic_dis_tol)
                        lnk.adjust_weight_from_residue()
                        inlier_indx = lnk.mask
                        valid_num += np.sum(inlier_indx)
                        xy0_t = lnk.xy0(gear=const.MESH_GEAR_INITIAL, use_mask=True, combine=True)
                        xy1_t = lnk.xy1(gear=const.MESH_GEAR_INITIAL, use_mask=True, combine=True)
                        xy0_t_list.append(xy0_t)
                        xy1_t_list.append(xy1_t)
                        lnk.eliminate_zero_weight()
                if ((valid_num / mtch.num_points) < 0.5) or (valid_num < 3):
                    break
            else:
                xy0_t_list = [mtch.xy0]
                xy1_t_list = [mtch.xy1]
            xy0.extend(xy0_t_list)
            xy1.extend(xy1_t_list)
            for lnk in staging_link:
                lnk.name = 'settled'
                lnk.reset_weight()
                lnk_uids = tuple(lnk.uids)
                if lnk_uids in settled_link:
                    settled_link[lnk_uids].combine_link(lnk)
                else:
                    settled_link[lnk_uids] = lnk
            used_matches.append(mtch)
            modified = True
        if not modified:
            break
        if affine_only and (len(exclude_class) >= num_cls_cmb):
            break
        covered_region0 = {}
        covered_region1 = {}
        for mtch in used_matches:
            cid0 = mtch.class_id0[0]
            cid1 = mtch.class_id1[0]
            if affine_only:
                A0 = fit_affine(mtch.xy0, mtch.xy1, avoid_flip=True)
                A0_m = np.concatenate((A0[:2,:2].T, A0[-1,:2]), axis=None)
                cg0 = affine_transform(info1['regions'][cid1], A0_m)
                A1 = np.linalg.inv(A0)
                A1_m = np.concatenate((A1[:2,:2].T, A1[-1,:2]), axis=None)
                cg1 = affine_transform(info0['regions'][cid0], A1_m)
            else:
                cg0 = concave_hull(MultiPoint(mtch.xy0), ratio=0.25).buffer(0.5*feature_spacing)
                cg1 = concave_hull(MultiPoint(mtch.xy1), ratio=0.25).buffer(0.5*feature_spacing)
                # cg0 = MultiPoint(mtch.xy0).buffer(2*feature_spacing)
                # cg1 = MultiPoint(mtch.xy1).buffer(2*feature_spacing)
            if cid0 not in covered_region0:
                covered_region0[cid0] = cg0
            else:
                covered_region0[cid0] = covered_region0[cid0].union(cg0)
            if cid1 not in covered_region1:
                covered_region1[cid1] = cg1
            else:
                covered_region1[cid1] = covered_region1[cid1].union(cg1)
        fidx = np.ones(kps0.num_points, dtype=bool)
        for cid0, cg0 in covered_region0.items():
            cidx_t = kps0.class_id == cid0
            xy_t = kps0.xy[cidx_t]
            to_keep = ~intersects_xy(cg0, xy_t[:,0], xy_t[:,1])
            fidx[cidx_t] = to_keep
        if np.sum(fidx) < 3:
            break
        D0 = D0[fidx]
        kps0 = kps0.filter_keypoints(fidx, include_descriptor=True, inplace=filter_in_place)
        fidx = np.ones(kps1.num_points, dtype=bool)
        for cid1, cg1 in covered_region1.items():
            cidx_t = kps1.class_id == cid1
            xy_t = kps1.xy[cidx_t]
            to_keep = ~intersects_xy(cg1, xy_t[:,0], xy_t[:,1])
            fidx[cidx_t] = to_keep
        if np.sum(fidx) < 3:
            break
        D0 = D0[:, fidx]
        kps1 = kps1.filter_keypoints(fidx, include_descriptor=True, inplace=filter_in_place)
        filter_in_place = True
    if len(xy0) == 0:
        return None
    else:
        xy0 = scale_coordinates(np.concatenate(xy0, axis=0), 1 / scale)
        xy1 = scale_coordinates(np.concatenate(xy1, axis=0), 1 / scale)
        weight = np.ones(xy0.shape[0], dtype=np.float32)
        return common.Match(xy0, xy1, weight)



def match_two_thumbnails_pmcc(img0, img1, mask0=None, mask1=None, **kwargs):
    sigma = kwargs.pop('sigma', 3)
    scale = kwargs.get('scale', 1.0)
    info0 = prepare_image(img0, mask=mask0, compute_keypoints=False, **kwargs)[scale]
    mesh0 = info0['mesh'].copy()
    info1 = prepare_image(img1, mask=mask1, compute_keypoints=False, **kwargs)[scale]
    mesh1 = info1['mesh'].copy()
    mesh0.uid = 0.0
    mesh1.uid = 1.0
    if not 'dog_image' in info0:
        info0['dog_image'] = common.masked_dog_filter(info0['image'], sigma=sigma, mask=info0['mask'])
    if not 'dog_image' in info1:
        info1['dog_image'] = common.masked_dog_filter(info1['image'], sigma=sigma, mask=info1['mask'])
    loader0 = StreamLoader(info0['dog_image'])
    loader1 = StreamLoader(info1['dog_image'])
    xy0, xy1, weight = section_matcher(mesh0, mesh1, loader0, loader1, **kwargs)
    if xy0 is None:
        return None
    xy0 = scale_coordinates(xy0, 1 / scale)
    xy1 = scale_coordinates(xy1, 1 / scale)
    return common.Match(xy0, xy1, weight)



def _scale_matches(match, scale):
    scale = np.atleast_1d(scale)
    if (match is None) or np.all(scale == 1):
        return match
    xy0, xy1, weight = match
    xy0 = scale_coordinates(xy0, scale[0])
    xy1 = scale_coordinates(xy1, scale[-1])
    return common.Match(xy0, xy1, weight)



def align_two_thumbnails(img0, img1, outname, mask0=None, mask1=None, **kwargs):
    if os.path.isfile(outname):
        return
    resolution = kwargs.get('resolution')
    feature_match_settings = kwargs.get('feature_matching', {}).copy()
    block_match_settings = kwargs.get('block_matching', {}).copy()
    feature_match_dir = kwargs.get('feature_match_dir', None)
    save_feature_match = kwargs.get('save_feature_match', False)
    logger_info = kwargs.get('logger', None)
    logger= logging.get_logger(logger_info)
    bname_ext = os.path.basename(outname)
    bname = bname_ext.replace('.h5', '')
    if feature_match_dir is not None:
        feature_matchname = os.path.join(feature_match_dir, bname_ext)
    else:
        feature_matchname = None
    if (feature_matchname is not None) and (os.path.isfile(feature_matchname)):
        mtch0 = read_matches_from_h5(feature_matchname, target_resolution=resolution)
    else:
        mtch0 = match_two_thumbnails_LRadon(img0, img1, mask0=mask0, mask1=mask1,
                                            **feature_match_settings)
        if mtch0 is None:     
            logger.warning(f'{bname}: fail to find matches.')
            return 0
        if save_feature_match:
            xy0, xy1, weight = mtch0
            os.makedirs(feature_match_dir, exist_ok=True)
            with h5py.File(feature_matchname, 'w') as f:
                f.create_dataset('xy0', data=xy0, compression="gzip")
                f.create_dataset('xy1', data=xy1, compression="gzip")
                f.create_dataset('weight', data=weight, compression="gzip")
                f.create_dataset('resolution', data=resolution)
    pmcc_scale = np.full(2, block_match_settings.get('scale', 1.0))
    mtch0 = _scale_matches(mtch0, pmcc_scale)
    mtch1 = match_two_thumbnails_pmcc(img0, img1, mask0=mask0, mask1=mask1,
                                      initial_matches=mtch0, **block_match_settings)
    if mtch1 is None:
        logger.warning(f'{bname}: fail to find matches.')
        return 0
    else:
        xy0, xy1, weight = mtch1
        os.makedirs(os.path.dirname(outname), exist_ok=True)
        with h5py.File(outname, 'w') as f:
            f.create_dataset('xy0', data=xy0, compression="gzip")
            f.create_dataset('xy1', data=xy1, compression="gzip")
            f.create_dataset('weight', data=weight, compression="gzip")
            f.create_dataset('resolution', data=resolution)
        return xy0.shape[0]



def detect_extrema_log(img, mask=None, offset=(0,0), **kwargs):
    sigma = kwargs.get('sigma', 3.5)
    min_spacing = kwargs.get('min_spacing', DEFAULT_FEATURE_SPACING)
    intensity_thresh = kwargs.get('intensity_thresh', 0.05)
    num_features = kwargs.get('num_features', np.inf)
    if isinstance(img, KeyPoints):
        return img
    if mask is None:
        mask = np.ones_like(img, dtype=np.uint8)
    elif not np.issubdtype(mask.dtype, np.integer):
        mask = mask.astype(np.int16, copy=False)
    if sigma > 0:
        img = common.masked_dog_filter(img, sigma, mask=(mask>0))
    if np.ptp(img, axis=None) == 0:
        return KeyPoints()
    if num_features <= 0:
        num_features = np.inf
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



def match_LRadon_feature(kps0, kps1, D=None, exclude_class=None, **kwargs):
    exhaustive = kwargs.get('exhaustive', False)
    conf_thresh = kwargs.get('conf_thresh', 0.5)
    to_exclude_class = (exclude_class is not None) and (len(exclude_class) > 0)
    if to_exclude_class:
        exclude_class = np.array(exclude_class, copy=False).reshape(-1,2)
    if kps0.num_points > kps1.num_points:
        kps0, kps1 = kps1, kps0
        if D is not None:
            D = D.T
        flipped = True
        if to_exclude_class:
            exclude_class = exclude_class[:,::-1]
    else:
        flipped = False
    if D is None:
        if exhaustive:
            des0 = kps0.reset_angle()
            des1 = kps1.reset_angle()
            norm_fact = 1 / (des0.shape[-1] * des0.shape[-2])
            D = np.full_like(des0, fill_value=-1, shape=(des0.shape[0], des1.shape[0]))
            des1_t = des1
            for _ in range(des1.shape[-1]):
                des1_t = np.roll(des1_t, 1, axis=-1)
                D_t = des0.reshape(des0.shape[0], -1) @ des1_t.reshape(des1.shape[0], -1).T
                D = np.maximum(D, D_t)
            D = norm_fact * D
        else:
            des0 = kps0.align_angle()
            des1 = kps1.align_angle()
            norm_fact = 1 / (des0.shape[-1] * des0.shape[-2])
            D = des0.reshape(des0.shape[0], -1) @ des1.reshape(des1.shape[0], -1).T
            D = norm_fact * D
    C = D.copy()
    if to_exclude_class:
        class_id0 = kps0.class_id
        class_id1 = kps1.class_id
        class_ids = class_id0.reshape(-1,1) + class_id1.reshape(1, -1) * 1j
        excluded_ids = exclude_class[:,0] + exclude_class[:,1] * 1j
        mask = np.isin(class_ids, excluded_ids)
        C[mask] = -1
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
        D = D.T
    idx0 = idx0[conf0 > conf_thresh]
    idx1 = idx1[conf0 > conf_thresh]
    conf = conf[conf0 > conf_thresh]
    kps0_out = kps0.filter_keypoints(idx0, include_descriptor=False, inplace=False)
    kps1_out = kps1.filter_keypoints(idx1, include_descriptor=False, inplace=False)
    mtch = KeyPointMatches.from_keypoints(kps0_out, kps1_out, weight=conf)
    mtch.sort_match_by_weight()
    return mtch, D



def filter_match_pairwise_strain(matches, **kwargs):
    strain_limit = kwargs.get('strain_limit', 0.2)
    shear_limit = kwargs.get('shear_limit', 45)
    sample_ratio = kwargs.get('sample_ratio', 0.05)
    inlier_thresh = kwargs.get('inlier_thresh', 0.5)
    min_output_num = kwargs.get('min_output_num', None)
    if (strain_limit is None) and (shear_limit is None):
        return matches, None
    if min_output_num is None:
        if inlier_thresh < 1:
            min_output_num = 20
        else:
            min_output_num = inlier_thresh
    if matches.num_points < min_output_num:
        return matches, None
    num_samples = max(25, int(matches.num_points*sample_ratio))
    dis0, rot0 = _pairwise_distance_rotation(matches.xy0)
    dis1, rot1 = _pairwise_distance_rotation(matches.xy1)
    sindx = np.argsort(dis0, axis=-1)[:, 1:num_samples]
    dis0 = np.take_along_axis(dis0, sindx, axis=-1)
    dis1 = np.take_along_axis(dis1, sindx, axis=-1)
    rot0 = np.take_along_axis(rot0, sindx, axis=-1)
    rot1 = np.take_along_axis(rot1, sindx, axis=-1)
    valid_pairs = np.ones_like(sindx, dtype=bool)
    strain = np.abs(np.log(dis0.clip(1.0e-9, None) / dis1.clip(1.0e-9, None)))
    if strain_limit is not None:
        valid_pairs = (strain < strain_limit) & valid_pairs
    if shear_limit is not None:
        shear_limit = shear_limit * np.pi / 180
        rot = rot0 - rot1
        not_all_bad = np.any(valid_pairs, axis=-1, keepdims=True)
        rot[(~valid_pairs) & (not_all_bad)] = np.nan
        crot = np.cos(rot)
        srot = np.sin(rot)
        rot_med = np.arctan2(np.nanmedian(srot, axis=-1), np.nanmedian(crot, axis=-1))
        rot_err = rot - rot_med.reshape(-1,1)
        rot_err_wrap = rot_err - np.round(rot_err / (2*np.pi)) * (2*np.pi)
        valid_pairs = (np.abs(rot_err_wrap) < shear_limit) & valid_pairs
    valid_num = np.sum(valid_pairs, axis=-1)
    if inlier_thresh < 1:
        kindx = valid_num > (inlier_thresh * np.max(valid_num))
    else:
        qt = np.quantile(valid_num, max(0, (valid_num.size - inlier_thresh) / valid_num.size))
        kindx = valid_num >= qt
    if np.sum(kindx) < min_output_num:
        accum_strain = 1 / np.sum(1/strain.clip(1e-3, None)**2, axis=-1)
        accum_strain[kindx] = 0
        tindx = np.argsort(accum_strain)
        kindx[tindx[:min_output_num]] = 1
    if np.all(kindx):
        discarded = None
    else:
        discarded = matches.filter_match(~kindx, inplace=False)
    matches.filter_match(kindx)
    return matches, discarded



def _pairwise_distance_rotation(xy):
    x = xy[:,0]
    y = xy[:,1]
    dx = x.reshape(-1, 1) - x
    dy = y.reshape(-1, 1) - y
    dis = (dx ** 2 + dy ** 2) ** 0.5
    rot = np.arctan2(dy, dx)
    return dis, rot



def filter_match_global_ransac(matches, **kwargs):
    maxiter = kwargs.get('maxiter', 5000)
    dis_tol = kwargs.get('dis_tol', DEFAULT_FEATURE_SPACING / 3)
    early_stop_num = kwargs.get('early_stop_num', 150)
    early_stop_ratio = kwargs.get('early_stop_ratio', 0.75)
    mixed_class = kwargs.get('mixed_class', False)
    num_points = matches.num_points
    if num_points < 3:
        return matches, None
    deform_thresh = 0.5
    early_stop_num = max(25, min(early_stop_num, num_points*early_stop_ratio))
    iternum = 0
    hit = False
    countdown = 20
    inlier_indx = None
    current_score = 0
    xy0 = matches.xy0
    xy1 = matches.xy1
    _, cnt0 = np.unique(matches.class_id0, return_counts=True)
    _, cnt1 = np.unique(matches.class_id1, return_counts=True)
    cnt_mx = max(cnt0.max(), cnt1.max())
    early_stop_num = early_stop_num * cnt_mx / matches.num_points
    early_stop = False
    for indx2 in range(2, num_points):
        if early_stop:
            break
        for indx01 in combinations(np.arange(indx2), 2):
            iternum += 1
            if (maxiter is not None) and (iternum > maxiter):
                early_stop = True
                break
            smpl_indices = np.concatenate((indx01, (indx2,)), axis=None)
            xy0_s = xy0[smpl_indices]
            xy1_s = xy1[smpl_indices]
            if (np.unique(xy0_s, axis=0).size != xy0_s.size) or (np.unique(xy1_s, axis=0).size != xy1_s.size):
                continue
            A = fit_affine(xy0_s, xy1_s, return_rigid=False)
            if np.linalg.det(A[:2,:2]) <= 0:
                continue
            sv = np.linalg.svd(A[:2,:2], full_matrices=False, compute_uv=False)
            deform = np.min(np.exp(-np.abs(np.log(sv))))
            if deform < deform_thresh:
                continue
            dxy = xy1 @ A[:2,:2] + A[-1,:2] - xy0
            inliers = np.sum(dxy**2, axis=-1) <= dis_tol ** 2
            inlier_cnt = np.sum(inliers)
            score = inlier_cnt * (deform - deform_thresh)
            if score > current_score:
                current_score = score
                inlier_indx = inliers
                if (not hit) and (inlier_cnt > early_stop_num):
                    hit = True
            if hit:
                countdown -= 1
                if countdown == 0:
                    early_stop = True
                    break
    if inlier_indx is not None:
        A = fit_affine(xy0[inlier_indx,:], xy1[inlier_indx,:], return_rigid=False)
        dxy = xy1 @ A[:2,:2] + A[-1,:2] - xy0
        inlier_indx = np.sum(dxy**2, axis=-1) <= dis_tol ** 2
    else:
        return None, matches
    if not mixed_class:
        class_id0 = matches.class_id0
        class_id1 = matches.class_id1
        class_id = class_id0 + class_id1 * 1j
        cls_val, cls_cnt = np.unique(class_id[inlier_indx], return_counts=True)
        class_indx = class_id == cls_val[np.argmax(cls_cnt)]
        inlier_indx = inlier_indx & class_indx
    if np.all(inlier_indx):
        discarded = None
    else:
        discarded = matches.filter_match(~inlier_indx, inplace=False)
    matches.filter_match(inlier_indx)
    return matches, discarded



def filter_match_sequential_ransac(matches, **kwargs):
    min_features_ratio = kwargs.pop('min_features_ratio', 0.1)
    kwargs.setdefault('mixed_class', False)
    max_rounds = kwargs.pop('max_rounds', np.inf)
    match_list = []
    cnt = 0
    min_features = None
    while True:
        hit, matches = filter_match_global_ransac(matches, **kwargs)
        if hit is None:
            break
        if min_features is None:
            min_features = max(5, hit.num_points * min_features_ratio)
        match_list.append(hit)
        if (matches is None) or (matches.num_points < min_features):
            break
        cnt += 1
        if cnt > max_rounds:
            break
    return match_list, matches
