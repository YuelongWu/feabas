import cv2
from collections import defaultdict, namedtuple
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures import as_completed
from functools import partial
import gc
import h5py
import matplotlib.tri
from multiprocessing import get_context
import numpy as np
import os
from rtree import index
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter, binary_dilation
from scipy import sparse
import scipy.sparse.csgraph as csgraph
import tensorstore as ts

from feabas.dal import StaticImageLoader
from feabas import logging
from feabas.matcher import stitching_matcher
from feabas.mesh import Mesh
from feabas.optimizer import SLM
from feabas import common, caching
from feabas.spatial import scale_coordinates
import feabas.constant as const
from feabas.config import DEFAULT_RESOLUTION, general_settings

class Stitcher:
    """
    Class to handle everything related to 2d stitching.

    Args:
        imgpaths(list): fullpaths of the image file list
        bboxes(N x 4 ndarray): initial estimation of the bounding boxes for each
            image tiles. Could be from the microscope stage coordinates or rough
            stitching results. (xmin, ymin, xmax, ymax), supposed integers.
    Kwargs:
        root_dir(str): if provided, imgpaths arg is considered to be relative
            paths and root_dir will be prepended to generate the fullpaths.
    """
  ## ------------------------- initialization & IO ------------------------- ##
    def __init__(self, imgpaths, bboxes, **kwargs):
        root_dir = kwargs.get('root_dir', None)
        groupings = kwargs.get('groupings', None)
        self._connected_subsystem = kwargs.get('connected_subsystem', None)
        self.resolution = kwargs.get('resolution', DEFAULT_RESOLUTION)
        if bool(root_dir):
            self.imgrootdir = root_dir
            self.imgrelpaths = imgpaths
        else:
            self.imgrootdir = os.path.dirname(os.path.commonprefix(imgpaths))
            self.imgrelpaths = [os.path.relpath(s, self.imgrootdir) for s in imgpaths]
        bboxes = np.round(bboxes).astype(np.int32)
        self.init_bboxes = bboxes
        self.tile_sizes = common.bbox_sizes(bboxes)
        self.average_tile_size = np.median(self.tile_sizes, axis=0)
        init_offset = bboxes[...,:2]
        self._init_offset = init_offset - init_offset.min(axis=0)
        self.matches = {}
        self.match_strains = {}
        self.meshes = None
        self.mesh_sharing = np.arange(self.num_tiles)
        self._optimizer = None
        self._default_mesh_cache = None
        self.set_groupings(groupings)


    @classmethod
    def from_coordinate_file(cls, filename, **kwargs):
        imgpaths, bboxes, root_dir, resolution = common.parse_coordinate_files(filename, **kwargs)
        return cls(imgpaths, bboxes, root_dir=root_dir, resolution=resolution)


    @classmethod
    def from_h5(cls, filename, load_matches=True, load_meshes=True, selected=None):
        """
        selected (ndarray of int): if provided, only load tiles whose indices
        are in selected.
        """
        with h5py.File(filename, 'r') as f:
            root_dir = common.numpy_to_str_ascii(f['imgrootdir'][()])
            imgpaths = common.numpy_to_str_ascii(f['imgrelpaths'][()]).split('\n')
            bboxes = f['init_bboxes'][()]
            if 'groupings' in f:
                groupings = f['groupings'][()]
            else:
                groupings = None
            if 'connected_subsystem' in f:
                connected_subsystem = f['connected_subsystem'][()]
            else:
                connected_subsystem = None
            if 'resolution' in f:
                resolution = f['resolution'][()]
            else:
                resolution = DEFAULT_RESOLUTION
        if selected is not None:
            imgpaths = [s for k, s in enumerate(imgpaths) if k in selected]
            bboxes = bboxes[selected]
            groupings = groupings[selected]
            check_order = True
        else:
            check_order = False
        obj = cls(imgpaths, bboxes, root_dir=root_dir, groupings=groupings,
            connected_subsystem=connected_subsystem, resolution=resolution)
        if load_matches:
            obj.load_matches_from_h5(filename, check_order=check_order)
        if load_meshes:
            obj.load_meshes_from_h5(filename, check_order=check_order)
        return obj


    def save_to_h5(self, fname, **kwargs):
        save_matches = kwargs.get('save_matches', True)
        save_meshes = kwargs.get('save_meshes', True)
        compression = kwargs.get('compression', True)
        with h5py.File(fname, 'w') as f:
            if compression:
                create_dataset = partial(f.create_dataset, compression='gzip')
            else:
                create_dataset = f.create_dataset
            f.create_dataset('imgrootdir', data=common.str_to_numpy_ascii(self.imgrootdir))
            f.create_dataset('resolution', data=self.resolution)
            imgnames_encoded = common.str_to_numpy_ascii('\n'.join(self.imgrelpaths))
            create_dataset('imgrelpaths', data=imgnames_encoded)
            create_dataset('init_bboxes', data=self.init_bboxes)
            if self._groupings is not None:
                create_dataset('groupings', data=self._groupings)
            if self.connected_subsystem is not None:
                create_dataset('connected_subsystem', data=self._connected_subsystem)
            if save_matches and (len(self.matches) > 0):
                for uids, mtch in self.matches.items():
                    prefix = 'matches/' + '_'.join(str(int(s)) for s in uids)
                    xy0, xy1, weight = mtch
                    strain = self.match_strains[uids]
                    data = np.concatenate((xy0, xy1, weight, strain), axis=None)
                    data = data.astype(np.float32, copy=False)
                    create_dataset(prefix, data=data)
            if save_meshes and (self.meshes is not None):
                soft_factors = np.array([m.soft_factor for m in self.meshes], dtype=np.float32)
                create_dataset('mesh_soft_factors', data=soft_factors)
                create_dataset('mesh_sharing_indx', data=self.mesh_sharing)
                mesh_share_u, mesh_share_sample = np.unique(self.mesh_sharing, return_index=True)
                for indx_u, indx_m in zip(mesh_share_u, mesh_share_sample):
                    M0 = self.meshes[indx_m]
                    prefix = 'master_meshes/' + str(int(indx_u))
                    M0.save_to_h5(f, vertex_flags=(const.MESH_GEAR_INITIAL,), prefix=prefix,
                        save_material=False, compression=compression)
                moving_offsets = np.array([m.offset(gear=const.MESH_GEAR_MOVING) for m in self.meshes])
                create_dataset('moving_offsets', data=moving_offsets)
                f.create_group('moving_vertices')
                for k, m in enumerate(self.meshes):
                    if m.vertices_initialized(gear=const.MESH_GEAR_MOVING):
                        prefix = 'moving_vertices/' + str(k)
                        v = m.vertices(gear=const.MESH_GEAR_MOVING)
                        create_dataset(prefix, data=v)


    def load_matches_from_h5(self, fname, check_order=False):
        with h5py.File(fname, 'r') as f:
            match_cnt = 0
            if 'matches' not in f:
                return match_cnt
            if check_order:
                imgnames = common.numpy_to_str_ascii(f['imgrelpaths'][()]).split('\n')
                name_lut = {name: k for k, name in enumerate(self.imgrelpaths)}
                indx_mapper = [name_lut.get(s, -1) for s in imgnames]
            else:
                indx_mapper = None
            matches = f['matches']
            for key in matches:
                uid0, uid1 = [int(s) for s in key.split('_')]
                if indx_mapper is not None:
                    uid0 = indx_mapper[uid0]
                    uid1 = indx_mapper[uid1]
                if (uid0 < 0) or (uid1 < 0):
                    continue
                data = matches[key][()]
                Npt = int((data.size - 1)/5)
                xy0 = data[0:(2*Npt)].reshape(-1, 2)
                xy1 = data[(2*Npt):(4*Npt)].reshape(-1, 2)
                weight = data[(4*Npt):(5*Npt)]
                strain = data[-1]
                self.matches[(uid0, uid1)] = (xy0, xy1, weight)
                self.match_strains[(uid0, uid1)] = strain
                match_cnt += 1
        return match_cnt


    def load_meshes_from_h5(self, fname, check_order=False, force_update=False):
        mesh_loaded = False
        if (not force_update) and (self.meshes is not None):
            return mesh_loaded
        with h5py.File(fname, 'r') as f:
            if 'mesh_sharing_indx' not in f:
                return mesh_loaded
            mesh_sharing_indx = f['mesh_sharing_indx'][()]
            if check_order:
                imgnames = common.numpy_to_str_ascii(f['imgrelpaths'][()]).split('\n')
                if len(imgnames) < len(self.imgrelpaths):
                    # mesh not complete
                    return mesh_loaded
                name_lut = {name: k for k, name in enumerate(imgnames)}
                indx_mapper = np.array([name_lut.get(s, -1) for s in self.imgrelpaths])
                if np.any(indx_mapper < 0):
                    # mesh not complete
                    return mesh_loaded
                mesh_sharing_u, mesh_sharing = np.unique(mesh_sharing_indx[indx_mapper], return_inverse=True)
            else:
                indx_mapper = None
                mesh_sharing = mesh_sharing_indx
            mesh_soft_factors = f['mesh_soft_factors'][()]
            moving_offsets = f['moving_offsets'][()]
            master_meshes = {}
            for uid_src in f['master_meshes']:
                if (indx_mapper is not None) and (uid_src not in mesh_sharing_u):
                    continue
                prefix = 'master_meshes/' + uid_src
                M0 = Mesh.from_h5(f, prefix=prefix)
                M0.unlock()
                master_meshes[int(uid_src)] = M0
            self.meshes = []
            for uid in range(self.num_tiles):
                if indx_mapper is None:
                    uid_src = uid
                else:
                    uid_src = indx_mapper[uid]
                tile_sft_factor = mesh_soft_factors[uid_src]
                tile_mstr_indx = mesh_sharing_indx[uid_src]
                M0 = master_meshes[tile_mstr_indx]
                M = M0.copy(save_material=False,
                    override_dict={'uid':uid, 'soft_factor':tile_sft_factor})
                if str(uid_src) in f['moving_vertices']:
                    prefix = 'moving_vertices/' + str(uid_src)
                    v = f[prefix][()]
                    M.set_vertices(v, const.MESH_GEAR_MOVING)
                    M.set_offset(moving_offsets[uid_src], const.MESH_GEAR_MOVING)
                self.meshes.append(M)
            self.mesh_sharing = mesh_sharing
            mesh_loaded = True
        return mesh_loaded


    def set_groupings(self, groupings):
        """
        groupings can be used to indicate a subset of tiles should share the
        same mechanical properties and meshing options. Their fixed-pattern
        deformation will also be compensated identically (as necessary). This
        also means the tiles within the same group should share the same tile
        sizes.
        """
        if groupings is not None:
            groupings = np.array(groupings, copy=False)
            assert len(groupings) == len(self.imgrelpaths)
            tile_ht = self.tile_sizes[:,0]
            tile_wd = self.tile_sizes[:,1]
            grp_u, cnt = np.unique(groupings, return_counts=True)
            grp_u = grp_u[cnt > 1]
            if tile_ht.ptp() > 0:
                for g in grp_u:
                    idx = groupings == g
                    if tile_ht[idx].ptp() > 0:
                        raise RuntimeError(f'tile size in group {g} not consistent')
            if tile_wd.ptp() > 0:
                for g in grp_u:
                    idx = groupings == g
                    if tile_ht[idx].ptp() > 0:
                        raise RuntimeError(f'tile size in group {g} not consistent')
        self._groupings = groupings


  ## ------------------------------ matching ------------------------------- ##
    def dispatch_matchers(self, **kwargs):
        """
        run matching between overlapping tiles.
        """
        overwrite = kwargs.pop('overwrite', False)
        num_workers = kwargs.pop('num_workers', 1)
        num_overlaps_per_job = kwargs.get('num_overlaps_per_job', 180) # 180 roughly number of overlaps in an MultiSEM mFoV
        loader_config = kwargs.pop('loader_config', {}).copy()
        logger_info = kwargs.get('logger', None)
        logger = logging.get_logger(logger_info)
        if bool(loader_config.get('cache_size', None)) and (num_workers > 1):
            loader_config['cache_size'] = int(np.ceil(loader_config['cache_size'] / num_workers))
        if bool(loader_config.get('cache_capacity', None)) and (num_workers > 1):
            loader_config['cache_capacity'] = loader_config['cache_capacity'] / num_workers
        loader_config['number_of_channels'] = 1 # only gray-scale matching are supported
        target_func = partial(Stitcher.subprocess_match_list_of_overlaps,
                              root_dir=self.imgrootdir, loader_config=loader_config,
                              **kwargs)
        if overwrite:
            self.matches = {}
            self.match_strains = {}
            overlaps = self.overlaps
        else:
            overlaps = self.overlaps_without_matches
        num_overlaps = len(overlaps)
        if ((num_workers is not None) and (num_workers <= 1)) or (num_overlaps <= 1):
            new_matches, match_strains, err_raised = target_func(overlaps, self.imgrelpaths, self.init_bboxes)
            self.matches.update(new_matches)
            self.match_strains.update(match_strains)
            return len(new_matches), err_raised
        num_workers = min(num_workers, num_overlaps)
        num_overlaps_per_job = min(num_overlaps//num_workers, num_overlaps_per_job)
        N_jobs = round(num_overlaps / num_overlaps_per_job)
        indx_j = np.linspace(0, num_overlaps, num=N_jobs+1, endpoint=True)
        indx_j = np.unique(np.round(indx_j).astype(np.int32))
        # divide works
        jobs = []
        num_new_matches = 0
        err_raised = False
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=get_context('spawn')) as executor:
            for idx0, idx1 in zip(indx_j[:-1], indx_j[1:]):
                ovlp_g = overlaps[idx0:idx1] # global indices of overlaps
                mapper, ovlp = np.unique(ovlp_g, return_inverse=True, axis=None)
                ovlp = ovlp.reshape(ovlp_g.shape)
                bboxes = self.init_bboxes[mapper]
                imgpaths = [self.imgrelpaths[s] for s in mapper]
                job = executor.submit(target_func, ovlp, imgpaths, bboxes, index_mapper=mapper)
                jobs.append(job)
            matched_counter = 0
            for job in as_completed(jobs):
                matches, match_strains, ouch = job.result()
                err_raised = err_raised or ouch
                num_new_matches += len(matches)
                self.matches.update(matches)
                self.match_strains.update(match_strains)
                matched_counter += len(matches)
                if matched_counter > (len(overlaps)/10):
                    logger.debug(f'matching in progress: {num_new_matches}/{len(overlaps)}')
                    matched_counter = 0
        return num_new_matches, err_raised


    def find_overlaps(self):
        overlaps = []
        tree = index.Index(interleaved=True)
        for k, bbox in enumerate(self.init_bboxes):
            hits = list(tree.intersection(bbox, objects=False))
            tree.insert(k, bbox, obj=None)
            if bool(hits):
                overlaps.extend([(k, hit) for hit in hits])
        overlaps = np.array(overlaps)
        bboxes0 = self.init_bboxes[overlaps[:,0]]
        bboxes1 = self.init_bboxes[overlaps[:,1]]
        bbox_ov, _ = common.bbox_intersections(bboxes0, bboxes1)
        ov_cntr = common.bbox_centers(bbox_ov)
        average_step_size = self.average_tile_size[::-1] / 2
        ov_indices = np.round((ov_cntr - ov_cntr.min(axis=0))/average_step_size)
        z_order = common.z_order(ov_indices)
        return overlaps[z_order]


    @staticmethod
    def subprocess_match_list_of_overlaps(overlaps, imgpaths, bboxes, **kwargs):
        """
        matching function used as the target function in Stitcher.dispatch_matchers.
        Args:
            overlaps(Kx2 ndarray): pairs of indices of the images to match.
            imgpaths(list of str): the paths to the image files.
            bboxes(Nx4 ndarray): the estimated bounding boxes of each image.
        Kwargs:
            root_dir(str): if provided, the paths in imgpaths are relative paths to
                this root directory.
            min_overlap_width: minimum amount of overlapping width (in pixels)
                to use. overlaps below this threshold will be skipped.
            index_mapper(N ndarray): the mapper to map the local image indices to
                the global ones, in case the imgpaths fed to this function is only
                a subset of all the images from the dispatcher.
            margin: extra image to include for matching around the overlapping
                regions to accommodate for inaccurate starting location. If
                smaller than 3, multiply it with the shorter edges of the
                overlapping bboxes before use.
            image_to_mask_path (list of str): this provide a way to input masks
                for matching. The file structure of the masks should mirror
                mirror that of the images, so that
                  imgpath.replace(image_to_mask_path[0], image_to_mask_path[1])
                direct to the mask of the corresponding images. In the mask,
                pixels with 0 values are not used. If no mask image are found,
                default to keep.
            loader_config(dict): key-word arguments passed to configure ImageLoader.
            matcher_config(dict): key-word arguments passed to configure matching.
        Return:
            matches(dict): M[(global_index0, global_index1)] = (xy0, xy1, conf).
            match_strains(dict): D[(global_index0, global_index1)] = strain.
        """
        root_dir = kwargs.get('root_dir', None)
        min_width = kwargs.get('min_overlap_width', 0)
        maskout_val = kwargs.get('maskout_val', None)
        index_mapper = kwargs.get('index_mapper', None)
        margin = kwargs.get('margin', 1.0)
        image_to_mask_path = kwargs.get('image_to_mask_path', None)
        loader_config = kwargs.get('loader_config', {}).copy()
        matcher_config = kwargs.get('matcher_config', {}).copy()
        instant_gc = kwargs.get('instant_gc', False)
        logger_info = kwargs.get('logger', None)
        logger = logging.get_logger(logger_info)
        margin_ratio_switch = 2
        err_raised = False
        if len(overlaps) == 0:
            return {}, {}, err_raised
        bboxes_overlap, wds = common.bbox_intersections(bboxes[overlaps[:,0]], bboxes[overlaps[:,1]])
        if 'cache_border_margin' not in loader_config:
            loader_config = loader_config.copy()
            overlap_wd0 = 6 * np.median(np.abs(wds - np.median(wds))) + np.median(wds) + 1
            if margin < margin_ratio_switch:
                loader_config['cache_border_margin'] = int(overlap_wd0 * (1 + margin))
            else:
                loader_config['cache_border_margin'] = int(overlap_wd0 + margin)
        image_loader = StaticImageLoader(imgpaths, bboxes, root_dir=root_dir, **loader_config)
        if image_to_mask_path is not None:
            if isinstance(image_to_mask_path[0], str):
                mask_paths = [s.replace(image_to_mask_path[0], image_to_mask_path[1])
                    for s in image_loader.filepaths_generator]
            else:
                mask_paths = [s for s in image_loader.filepaths_generator]
                for sub0, sub1 in zip(image_to_mask_path[0], image_to_mask_path[1]):
                    mask_paths = [s.replace(sub0, sub1) for s in mask_paths]
            mask_exist = np.array([os.path.isfile(s) for s in mask_paths])
            loader_config = loader_config.copy()
            loader_config.update({'apply_CLAHE': False,
                                  'inverse': False,
                                  'number_of_channels': None,
                                  'preprocess': None,
                                  'fillval': 1})
            mask_loader = StaticImageLoader(mask_paths, bboxes, root_dir=None, **loader_config)
        else:
            mask_exist = np.zeros(len(imgpaths), dtype=bool)
        matches = {}
        strains = {}
        error_messages = []
        for indices, bbox_ov, wd in zip(overlaps, bboxes_overlap, wds):
            if wd <= min_width:
                continue
            if margin < margin_ratio_switch:
                real_margin = int(margin * wd)
            else:
                real_margin = int(margin)
            bbox_ov = common.bbox_enlarge(bbox_ov, real_margin)
            idx0, idx1 = indices
            try:
                bbox0 = bboxes[idx0]
                bbox1 = bboxes[idx1]
                bbox_ov0 = common.bbox_intersections(bbox_ov, bbox0)[0]
                bbox_ov1 = common.bbox_intersections(bbox_ov, bbox1)[0]
                img0 = image_loader.crop(bbox_ov0, idx0, return_index=False)
                img1 = image_loader.crop(bbox_ov1, idx1, return_index=False)
                if mask_exist[idx0]:
                    mask0 = mask_loader.crop(bbox_ov0, idx0, return_index=False) > 0
                elif maskout_val is not None:
                    mask0t = img0 == maskout_val
                    if np.any(mask0t):
                        mask0 = ~binary_dilation(mask0t, iterations=2)
                    else:
                        mask0 = None
                else:
                    mask0 = None
                if mask_exist[idx1]:
                    mask1 = mask_loader.crop(bbox_ov1, idx1, return_index=False) > 0
                elif maskout_val is not None:
                    mask1t = img1 == maskout_val
                    if np.any(mask1t):
                        mask1 = ~binary_dilation(mask1t, iterations=2)
                    else:
                        mask1 = None
                else:
                    mask1 = None
                xy0, xy1, weight, strain = stitching_matcher(img0, img1, mask0=mask0, mask1=mask1, **matcher_config)
                if xy0 is None:
                    continue
                offset0 = bbox_ov0[:2] - bbox0[:2]
                offset1 = bbox_ov1[:2] - bbox1[:2]
                xy0 = xy0 + offset0
                xy1 = xy1 + offset1
                if index_mapper is not None:
                    idx0 = index_mapper[idx0]
                    idx1 = index_mapper[idx1]
                matches[(idx0, idx1)] = (xy0, xy1, weight)
                strains[(idx0, idx1)] = strain
            except Exception as err:
                if not err_raised:
                    error_messages.append(f'{image_loader.imgrootdir}: {err}')
                    err_raised = True
                error_messages.append(f'\t{image_loader.imgrelpaths[idx0]} <-> {image_loader.imgrelpaths[idx1]}')
        if err_raised:
            msg = '\n'.join(error_messages)
            logger.error(msg)
        image_loader.clear_cache()
        if instant_gc:
            gc.collect()
        return matches, strains, err_raised


  ## -------------------------- mesh relaxation ---------------------------- ##
    def initialize_meshes(self, mesh_sizes, **kwargs):
        """
        initialize meshes of each tile for mesh relaxation.
        Args:
            mesh_sizes(list): a list of options for mesh size to choose from.
                the actual mesh sizes for each tile will be determined based on
                how much the tile needs to be deformed during the matching.
                Provide only one value to fixed mesh size.
        Kwargs:
            border_width(float): the width of the overlap in the ratio to the
                tile size, so that mesh triangles away from the border can be
                coarser to save computation. Set to None to automatically
                determine.
            interior_growth(float): increase of the mesh size in the interior
                region.
            match_soften(tuple): If set, the meshes will be soften based on
                the strain during matching step. the tuple has two elements:
                (upper_strain, lower_soft_factor), so that the soften factor
                linearly changes from 1 to lower_soft_factor with a strain
                from 0 to upper_strain.
            soft_top(float): the multiplier to apply to the stiffness to the top
                part of the tile (there might be more distorsion at the beginning
                of the scan, so make it weight less).
            soft_top_width(float): the width of the soft top region with regard
                to the tile height.
            soft_left, soft_left_width: same as the "soft top" but in horizontal
                direction.
            cache_size: size of cache that save the intermediate results of
                the meshes.
        """
        border_width = kwargs.get('border_width', None)
        interior_growth = kwargs.get('interior_growth', 3.0)
        match_soften = kwargs.get('match_soften', None)
        soft_top = kwargs.get('soft_top', 0.2)
        soft_top_width = kwargs.get('soft_top_width', 0.0)
        soft_left = kwargs.get('soft_left', 0.2)
        soft_left_width = kwargs.get('soft_left_width', 0.0)
        cache_size = kwargs.get('cache_size', None)
        groupings = self.groupings(normalize=True)
        if (not kwargs.get('force_update', False)) and (self.meshes is not None):
            return
        if (not hasattr(mesh_sizes, '__len__')) or (len(mesh_sizes) == 1) or (len(self.matches) == 0):
            tile_mesh_sizes = np.full(self.num_tiles, np.max(mesh_sizes))
        else:
            # determine the mesh size based on the deformation during mathcing.
            err_contant = 5
            mesh_sizes = np.array(mesh_sizes, copy=False)
            mx_meshsz = np.max(mesh_sizes)
            lower_strain = 0.5 * err_contant / mx_meshsz
            strain_list = list(self.match_strains.items())
            overlap_indices = np.array([s[0] for s in strain_list])
            strain_vals = np.array([(s[1], s[1]) for s in strain_list])
            tile_strains = np.zeros(self.num_tiles, dtype=np.float32)
            np.maximum.at(tile_strains, overlap_indices, strain_vals)
            targt_mesh_size = err_contant / (tile_strains.clip(lower_strain, None))
            mesh_size_diff = np.abs(mesh_sizes.reshape(-1,1) - targt_mesh_size)
            mesh_size_idx = np.argmin(mesh_size_diff, axis=0)
            tile_mesh_sizes = mesh_sizes[mesh_size_idx]
            grp_u, cnt = np.unique(groupings, return_counts=True)
            grp_u = grp_u[cnt>1]
            for g in grp_u:
                idx = groupings == g
                tile_mesh_sizes[idx] = np.min(tile_mesh_sizes[idx])
        if border_width is None:
            indx = self.overlaps
            bboxes = self.init_bboxes
            _, ovlp_wds = common.bbox_intersections(bboxes[indx[:,0]], bboxes[indx[:,1]])
            tile_border_widths = np.zeros(self.num_tiles, dtype=np.float32)
            np.maximum.at(tile_border_widths, indx, np.stack((ovlp_wds, ovlp_wds), axis=-1))
            tile_border_widths = tile_border_widths / np.min(self.tile_sizes, axis=-1)
            # tiles in a group share mesh
            grp_u, cnt = np.unique(groupings, return_counts=True)
            grp_u = grp_u[cnt>1]
            for g in grp_u:
                idx = groupings == g
                tile_border_widths[idx] = np.median(tile_border_widths[idx])
            # rounding to make mesh reuse more likely
            rfct = np.median(tile_border_widths) * 1.5
            tile_border_widths = np.maximum(np.round(tile_border_widths/rfct), 1.0) * rfct
        elif hasattr(border_width,'__len__'):
            tile_border_widths = border_width
        else:
            tile_border_widths = np.full(self.num_tiles, border_width, dtype=np.float32)
        # make starting portion of the tile softer to make for distortion
        if soft_top != 1 and soft_top_width > 0:
            stf_y = interp1d([0, 0.99*soft_top_width, soft_top_width,1],
                             [soft_top, soft_top, 1, 1], kind='linear',
                             bounds_error=False, fill_value=(soft_top, 1))
        else:
            stf_y = None
        if soft_left != 1 and soft_left_width > 0:
            stf_x = interp1d([0, 0.99*soft_left_width, soft_left_width,1],
                             [soft_left, soft_left, 1, 1], kind='linear',
                             bounds_error=False, fill_value=(soft_left, 1))
        else:
            stf_x = None
        # soften the mesh with the strain from matching
        if (match_soften is None) or (match_soften[1] == 1):
            tile_soft_factors = np.ones(self.num_tiles, dtype=np.float32)
        else:
            strain_list = list(self.match_strains.items())
            overlap_indices = np.array([s[0] for s in strain_list])
            strain_vals = np.array([(s[1], s[1]) for s in strain_list])
            groupov_indices = groupings[overlap_indices]
            # only probe the interfaces between groups
            idxt = groupov_indices[:,0] != groupov_indices[:,1]
            groupov_indices = groupov_indices[idxt].ravel()
            strain_vals = strain_vals[idxt].ravel()
            avg_strain = np.zeros(self.num_tiles, dtype=np.float32)
            for g in np.unique(groupov_indices):
                idxt = groupov_indices == g
                strn = np.median(strain_vals[idxt])
                avg_strain[groupings == g] = strn
            upper_strain, lower_soft_factor = match_soften
            soften_func = interp1d([0, upper_strain], [1, lower_soft_factor],
                kind='linear', bounds_error=False, fill_value=(1, lower_soft_factor))
            tile_soft_factors = soften_func(avg_strain)
        meshes = []
        mesh_indx = np.full(self.num_tiles, -1)
        mesh_params_ptr = {} # map the parameters of the mesh to
        if self._default_mesh_cache is None:
            default_caches = {}
            for gear in const.MESH_GEARS:
                default_caches[gear] = defaultdict(lambda: caching.CacheFIFO(maxlen=cache_size))
            self._default_mesh_cache = default_caches
        else:
            default_caches = self._default_mesh_cache
        for k, tile_size in enumerate(self.tile_sizes):
            tmsz = tile_mesh_sizes[k]
            tbwd = tile_border_widths[k]
            tsf = tile_soft_factors[k]
            key = (*tile_size, tmsz, tbwd)
            if key in mesh_params_ptr:
                midx = mesh_params_ptr[key]
                mesh_indx[k] = midx
                M0 = meshes[midx].copy(override_dict={'uid':k, 'soft_factor':tsf})
            else:
                tileht, tilewd = tile_size
                M0 = Mesh.from_boarder_bbox((0,0,tilewd,tileht), bd_width=tbwd,
                    mesh_growth=interior_growth, mesh_size=tmsz, uid=k,
                    soft_factor=tsf)
                M0.set_stiffness_multiplier_from_interp(xinterp=stf_x, yinterp=stf_y)
                M0.center_meshes_w_offsets(gear=const.MESH_GEAR_FIXED)
                mesh_params_ptr[key] = k
                mesh_indx[k] = k
            for gear in const.MESH_GEARS:
                M0.set_default_cache(cache=default_caches[gear], gear=gear)
            meshes.append(M0)
        for M, offset in zip(meshes, self._init_offset):
            M.apply_translation(offset, gear=const.MESH_GEAR_FIXED)
        self.meshes = meshes
        _, mesh_indx_nm = np.unique(mesh_indx, return_inverse=True)
        self.mesh_sharing = mesh_indx_nm


    def filter_match_by_weight(self, minweight):
        rejected = 0
        if minweight is not None:
            to_pop = []
            for uids, mtch in self.matches.items():
                xy0, xy1, weight = mtch
                sel = weight >= minweight
                if not np.any(sel):
                    to_pop.append(uids)
                elif not np.all(sel):
                    xy0 = xy0[sel]
                    xy1 = xy1[sel]
                    weight = weight[sel]
                    rejected += np.sum(~sel)
                    self.matches[uids] = (xy0, xy1, weight)
            for uids in to_pop:
                self.matches.pop(uids)
                self.match_strains.pop(uids)
        return rejected                    


    def initialize_optimizer(self, **kwargs):
        if (not kwargs.get('force_update', False)) and (self._optimizer is not None):
            return False
        if (self.meshes is None) or (self.num_links == 0):
            raise RuntimeError('meshes and matches not initialized for Stitcher.')
        self._optimizer = SLM(self.meshes, **kwargs)
        for key, mtch in self.matches.items():
            uid0, uid1 = key
            xy0, xy1, weight = mtch
            self._optimizer.add_link_from_coordinates(uid0, uid1, xy0, xy1,
                weight=weight, check_duplicates=False)
        return True


    def optimize_translation(self, **kwargs):
        """
        optimize the translation of tiles according to the matches for a specific
        gear.
        Kwargs:
            maxiter: maximum number of iterations in LSQR. None if no limit.
            tol: the stopping tolerance of the least-square iterations.
            start_gear: gear that associated with the vertices before applying
                the translation.
            target_gear: gear that associated with the vertices at the final
                postions for locked meshes. Also the results are saved to this
                gear as well.
            residue_threshold: if set, links with average error larger than this
                at the end of the optimization will be removed one at a time.

        """
        maxiter = kwargs.get('maxiter', None)
        tol = kwargs.get('tol', 1e-07)
        target_gear = kwargs.get('target_gear', const.MESH_GEAR_FIXED)
        start_gear = kwargs.get('start_gear', target_gear)
        residue_threshold = kwargs.get('residue_threshold', None)
        if self._optimizer is None:
            self.initialize_optimizer()
        self._optimizer.optimize_translation_lsqr(maxiter=maxiter, tol=tol,
            start_gear=start_gear, target_gear=target_gear)
        num_disabled = 0
        if (residue_threshold is not None) and (residue_threshold > 0):
            while True:
                lnks_w_large_dis = []
                mxdis = 0
                for k, lnk in enumerate(self._optimizer.links):
                    if not lnk.relevant:
                        continue
                    dxy = lnk.dxy(gear=(target_gear, target_gear), use_mask=True)
                    dxy_m = np.median(dxy, axis=0)
                    dis = np.sqrt(np.sum(dxy_m**2))
                    mxdis = max(mxdis, dis)
                    if dis > residue_threshold:
                        lnks_w_large_dis.append((dis, lnk.uids, k))
                if not lnks_w_large_dis:
                    break
                else:
                    lnks_w_large_dis.sort(reverse=True)
                    uid_record = set()
                    for lnk in lnks_w_large_dis:
                        dis, lnk_uids, lnk_k = lnk
                        if uid_record.isdisjoint(lnk_uids):
                            self._optimizer.links[lnk_k].disable()
                            num_disabled += 1
                        uid_record.update(lnk_uids)
                    self._optimizer.optimize_translation_lsqr(maxiter=maxiter,
                        tol=tol,start_gear=start_gear, target_gear=target_gear)
        return num_disabled


    def optimize_group_intersection(self, **kwargs):
        """
        initialize the mesh transformation based only on the matches between
        tiles from different groups. tiles within each group will have the same
        transformation.
        Kwargs: refer to the input of feabas.optimizer.SLM.optimize_linear.
        """
        cache_size = kwargs.get('cache_size', None)
        target_gear = kwargs.setdefault('target_gear', const.MESH_GEAR_FIXED)
        if not self.has_groupings:
            return 0, 0
        groupings = self.groupings(normalize=True)
        match_uids = np.array(list(self.matches.keys()))
        match_grps = groupings[match_uids]
        idxt = match_grps[:,0] != match_grps[:,1]
        if not np.any(idxt):
            return 0, 0
        match_uids = match_uids[idxt]
        sel_uids, sel_match_uids = np.unique(match_uids, return_inverse=True)
        sel_match_uids = sel_match_uids.reshape(-1,2)
        sel_grps = groupings[sel_uids]
        sel_meshes = [self.meshes[s].copy(override_dict={'uid':k}) for k,s in enumerate(sel_uids)]
        if self._default_mesh_cache is None:
            default_caches = {}
            for gear in const.MESH_GEARS:
                default_caches[gear] = defaultdict(lambda: caching.CacheFIFO(maxlen=cache_size))
            self._default_mesh_cache = default_caches
        else:
            default_caches = self._default_mesh_cache
        for M0 in sel_meshes:
            for gear in const.MESH_GEARS:
                M0.set_default_cache(cache=default_caches[gear], gear=gear)
        opt = SLM(sel_meshes)
        for uids0, uids1 in zip(match_uids, sel_match_uids):
            mtch = self.matches[tuple(uids0)]
            xy0, xy1, weight = mtch
            opt.add_link_from_coordinates(uids1[0], uids1[1], xy0, xy1,
                weight=weight, check_duplicates=False)
        cost = opt.optimize_linear(groupings=sel_grps, **kwargs)
        for g, m in zip(groupings, self.meshes):
            sel_idx = np.nonzero(sel_grps == g)[0]
            if sel_idx.size == 0:
                continue
            else:
                m_border = opt.meshes[sel_idx[0]]
                v = m_border.vertices(gear=target_gear)
                m.set_vertices(v, target_gear)
        return cost


    def optimize_elastic(self, **kwargs):
        """
        elastically optimize the spring mesh system.
        Kwargs:
            use_groupings(bool): whether to bundle the meshes within the same
                group when applying the transformation.
            residue_len(float): if set, the link weight will be adjusted
                according to the length before attempting another optimization
                routine.
            residue_mode: the mode to adjust residue. Could be 'huber' or
                'threshold'.
        other kwargs refer to the input of feabas.optimizer.SLM.optimize_linear.
        """
        use_groupings = kwargs.get('use_groupings', False) and self.has_groupings
        residue_len = kwargs.get('residue_len', 0)
        residue_mode = kwargs.get('residue_mode', None)
        target_gear = kwargs.setdefault('target_gear', const.MESH_GEAR_MOVING)
        if use_groupings:
            groupings = self.groupings(normalize=True)
        else:
            groupings = None
        if self._optimizer is None:
            self.initialize_optimizer()
        cost = self._optimizer.optimize_linear(groupings=groupings, **kwargs)
        if (residue_mode is not None) and (residue_len > 0):
            if residue_mode == 'huber':
                self._optimizer.set_link_residue_huber(residue_len)
            else:
                self._optimizer.set_link_residue_threshold(residue_len)
            weight_modified, _ = self._optimizer.adjust_link_weight_by_residue(gear=(target_gear, target_gear))
            if weight_modified:
                if kwargs.get('tol', None) is not None:
                    kwargs['tol'] = max(1.0e-3, kwargs['tol'])
                if kwargs.get('atol', None) is not None:
                    kwargs['atol'] = 0.1 * kwargs['atol']
                cost1 = self._optimizer.optimize_linear(groupings=groupings, **kwargs)
                cost = (cost[0], cost1[-1])
        return cost


    def connect_isolated_subsystem(self, **kwargs):
        """
        translate blocks of tiles that are connected by links as a whole so that
        disconnected blocks roughly maintain the initial relative positions.
        """
        gear = (const.MESH_GEAR_INITIAL, const.MESH_GEAR_MOVING)
        explode_factor = kwargs.get('explode_factor', 1.0)
        maxiter = kwargs.get('maxiter', None)
        tol = kwargs.get('tol', 1e-07)
        if self._optimizer is None:
            raise RuntimeError('optimizer of the stitcher not initialized.')
        Lbls, N_conn = self._optimizer.connected_subsystems
        self._connected_subsystem = Lbls
        if N_conn <= 1:
            return N_conn
        cnt = np.zeros(N_conn)
        np.add.at(cnt, Lbls, 1)
        overlaps = self.overlaps
        L_ov = Lbls[overlaps]
        idxt = L_ov[:,0] != L_ov[:,1]
        L_ov = L_ov[idxt]
        overlaps = overlaps[idxt]
        Neq = overlaps.shape[0]
        idx0 = np.stack((np.arange(Neq), np.arange(Neq)), axis=-1).ravel()
        idx1 = L_ov.ravel()
        v = (np.ones_like(L_ov) * np.array([1, -1])).ravel()
        idx0 = np.append(idx0, Neq)
        idx1 = np.append(idx1, np.argmax(cnt))
        v = np.append(v, 1)
        A = sparse.csr_matrix((v, (idx0, idx1)), shape=(Neq+1, N_conn))
        overlaps_u, overlaps_inverse = np.unique(overlaps, return_inverse=True)
        overlaps_inverse = overlaps_inverse.reshape(overlaps.shape)
        txy = np.array([self.meshes[s].estimate_translation(gear=gear) for s in overlaps_u]).reshape(-1, 2)
        offset1 = txy[overlaps_inverse[:,0]] - txy[overlaps_inverse[:,1]]
        offset0 = self._init_offset[overlaps[:,0]] - self._init_offset[overlaps[:,1]]
        bxy = np.append(explode_factor * offset0 - offset1, [[0, 0]], axis=0)
        Tx = sparse.linalg.lsqr(A, bxy[:,0], atol=tol, btol=tol, iter_lim=maxiter)[0]
        Ty = sparse.linalg.lsqr(A, bxy[:,1], atol=tol, btol=tol, iter_lim=maxiter)[0]
        if np.any(Tx != 0) or np.any(Ty != 0):
            for m, lbl in zip(self.meshes, Lbls):
                tx = Tx[lbl]
                ty = Ty[lbl]
                m.unlock()
                m.apply_translation((tx, ty), gear[-1])
        # if still not fully connected, align average point
        A_blk = sparse.csr_matrix((np.ones_like(L_ov[:,0]), (L_ov[:,0], L_ov[:,1])), shape=(N_conn, N_conn))
        N_blk, L_blk = csgraph.connected_components(A_blk, directed=False, return_labels=True)
        if N_blk > 1:
            offset1 = np.array([m.estimate_translation(gear=gear) for m in self.meshes])
            dxy = explode_factor * (self._init_offset) - offset1
            Ls = L_blk[Lbls]
            for lbl in range(N_blk):
                txy = np.mean(dxy[Ls == lbl], axis = 0)
                for m, l in zip(self.meshes, Ls):
                    if l != lbl:
                        continue
                    m.unlock()
                    m.apply_translation(txy, gear[-1])
        return N_conn


    def normalize_coordinates(self, **kwargs):
        """
        apply uniform rigid transforms to the meshes so that the rotation is
        less than rotation_threshold and the upper-left corner is aligned to
        the offset.
        """
        gear = (const.MESH_GEAR_INITIAL, const.MESH_GEAR_MOVING)
        rotation_threshold = kwargs.get('rotation_threshold', None)
        offset = kwargs.get('offset', None)
        theta0 = 0
        offset0 = None
        if self._optimizer is None:
            raise RuntimeError('optimizer of the stitcher not initialized.')
        if rotation_threshold is not None:
            rotations = []
            for m in self.meshes:
                R = m.estimate_affine(gear=gear, svd_clip=(1,1))
                rotations.append(np.arctan2(R[0,1], R[0,0]))
            rotations = np.array(rotations, copy=False)
            L_ss, N_ss = self._optimizer.connected_subsystems
            theta = {}
            for lbl in range(N_ss):
                theta = np.median(rotations[L_ss == lbl])
                theta_d = np.abs(theta * 180 / np.pi)
                theta0 = max(theta0, theta_d)
                if theta_d > rotation_threshold:
                    c, s = np.cos(theta), np.sin(theta)
                    R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=np.float32)
                    for m, l in zip(self.meshes, L_ss):
                        if l != lbl:
                            continue
                        m.unlock()
                        m.apply_affine(R, gear[-1])
        if offset is not None:
            offset0 = np.array([np.inf, np.inf])
            for m in self.meshes:
                xy_min = m.bbox(gear=gear[-1], offsetting=True)[:2]
                offset0 = np.minimum(offset0, xy_min)
            txy = np.array(offset) - offset0
            if np.any(txy != 0):
                for m in self.meshes:
                    m.unlock()
                    m.apply_translation(txy, gear[-1])
        return theta0, offset0


    def clear_mesh_cache(self, gear=None, instant_gc=True):
        if self._default_mesh_cache is not None:
            if gear is None:
                for g in const.MESH_GEARS:
                    self.clear_mesh_cache(gear=g)
            else:
                cache = self._default_mesh_cache[gear]
                if isinstance(cache, caching.CacheNull):
                    cache.clear()
                elif isinstance(cache, dict):
                    for c in cache.values():
                        c.clear()
        if instant_gc:
            gc.collect()


  ## ----------------------------- properties ------------------------------ ##
    def groupings(self, normalize=False):
        if self._groupings is None:
            return np.arange(self.num_tiles)
        if normalize:
            if not hasattr(self, '_normalized_groupings') or self._normalized_groupings is None:
                _, indx = np.unique(self._groupings, return_inverse=True)
                self._normalized_groupings = indx
            return self._normalized_groupings
        else:
            return self._groupings


    @property
    def connected_subsystem(self):
        if (len(self.matches) > 0) and (self._connected_subsystem is None):
            edges = np.array([uids for uids in self.matches])
            V = np.ones(edges.shape[0], dtype=bool)
            A = sparse.csr_matrix((V, (edges[:,0], edges[:,1])), shape=(self.num_tiles, self.num_tiles))
            _, V_conn = csgraph.connected_components(A, directed=False, return_labels=True)
            self._connected_subsystem = V_conn
        return self._connected_subsystem


    @property
    def has_groupings(self):
        if self._groupings is None:
            return False
        groupings = self.groupings(normalize=True)
        if groupings.size == np.max(groupings):
            return False
        else:
            return True


    @property
    def overlaps_without_matches(self):
        overlaps = self.overlaps
        return np.array([s for s in overlaps if (tuple(s) not in self.matches)])


    @property
    def overlaps(self):
        if (not hasattr(self, '_overlaps')) or (self._overlaps is None):
            self._overlaps = self.find_overlaps()
        return self._overlaps


    @property
    def num_tiles(self):
        return len(self.imgrelpaths)


    @property
    def num_links(self):
        return len(self.matches)


    def match_residues(self, quantile=1):
        return self._optimizer.match_residues(gear=const.MESH_GEAR_MOVING, use_mask=True, quantile=quantile)



Mesh_Info = namedtuple('Mesh_Info', ['moving_vertices', 'moving_offsets', 'triangles', 'fixed_verticess'])


class MontageRenderer:
    """
    A class to render Montage with overlapping tiles.
    Here I won't consider flipped triangle cases as in feabas.renderer.MeshRenderer,
    it would be too messed up for a stitching problem and need human intervention
    anyway.
    Args:
        imgpaths(list): list of paths of the image tile.
        mesh_info(list): list of mesh information with each element as
            (moving_vertices, moving_offsets, triangles, fixed_verticess).
        tile_sizes(N x 2 ndarray): tile sizes of each tile.
    Kwargs:
        loader_settings(dict): settings to initialize the image loader, refer to
            feabas.dal.AbstractImageLoade
        root_dir(str): if set, the paths in filepaths are relative path to this
            root directory.
    """
    def __init__(self, imgpaths, mesh_info, tile_sizes, **kwargs):
        self.resolution = kwargs.get('resolution', DEFAULT_RESOLUTION)
        self._loader_settings = kwargs.get('loader_settings', {}).copy()
        self._connected_subsystem = kwargs.get('connected_subsystem', None)
        if bool(kwargs.get('root_dir', None)):
            self.imgrootdir = kwargs['root_dir']
            self.imgrelpaths = imgpaths
        else:
            self.imgrootdir = os.path.dirname(os.path.commonprefix(imgpaths))
            self.imgrelpaths = [os.path.relpath(s, self.imgrootdir) for s in imgpaths]
        self._tile_sizes = tile_sizes.reshape(-1,2)
        self._identical_tile_size = np.all(np.ptp(self._tile_sizes, axis=0) == 0)
        self._mesh_info = mesh_info
        self._interpolators = None
        self._rtree = None
        self._image_loader = None


    @classmethod
    def from_stitcher(cls, stitcher, gear=(const.MESH_GEAR_INITIAL, const.MESH_GEAR_MOVING), **kwargs):
        if stitcher.meshes is None:
            raise RuntimeError('stitcher meshes not initializad.')
        root_dir = stitcher.imgrootdir
        imgpaths = stitcher.imgrelpaths
        tile_sizes = stitcher.tile_sizes
        connected_subsystem = stitcher.connected_subsystem
        resolution = stitcher.resolution
        mesh_info = []
        for M in stitcher.meshes:
            v0 = M.vertices_w_offset(gear=gear[0])
            v1 = M.vertices(gear=gear[1])
            offset = M.offset(gear=gear[1])
            T = M.triangles
            mesh_info.append(Mesh_Info(v1, offset, T, v0))
        return cls(imgpaths, mesh_info, tile_sizes, root_dir=root_dir,
                   connected_subsystem=connected_subsystem, resolution=resolution, **kwargs)


    @classmethod
    def from_h5(cls, fname, selected=None, gear=(const.MESH_GEAR_INITIAL, const.MESH_GEAR_MOVING), **kwargs):
        stitcher = Stitcher.from_h5(fname, load_matches=False, load_meshes=True, selected=selected)
        return cls.from_stitcher(stitcher, gear=gear, **kwargs)


    def init_args(self, selected=None):
        if selected is None:
            imgpaths = self.imgrelpaths
            mesh_info = self._mesh_info
            tile_size = self._tile_sizes
        else:
            imgpaths = [self.imgrelpaths[s] for s in selected]
            mesh_info = [self._mesh_info[s] for s in selected]
            if not self._identical_tile_size:
                tile_size = self._tile_sizes[selected]
            else:
                tile_size = self._tile_sizes[0]
            args = [imgpaths, mesh_info, tile_size]
            kwargs = {}
            kwargs['root_dir'] = self.imgrootdir
            kwargs['loader_settings'] = self._loader_settings
            kwargs['resolution'] = self.resolution
        return args, kwargs


    def clear_cache(self, instant_gc=False):
        self._interpolants = None
        self._rtree = None
        if self._image_loader is not None:
            self._image_loader.clear_cache(instant_gc=False)
            self._image_loader = None
        if instant_gc:
            gc.collect()


    def CLAHE_off(self):
        self.image_loader.CLAHE_off()


    def CLAHE_on(self):
        self.image_loader.CLAHE_on()


    def inverse_off(self):
        self.image_loader.inverse_off()


    def inverse_on(self):
        self.image_loader.inverse_on()


    def crop(self, bbox, **kwargs):
        """
        crop out the tile defined by the bounding box in output space, blend the
        source tiles if necessary.
        blend(str):
            LINEAR: the weight is proportional to the distance to the tile edge.
            NEAREST: find the pixel innermost to any tiles.
            PYRAMID: two-level pyramid where hp part use NEAREST and lp part use
                LINEAR.
            MAX: find the maximum value
            MIN: find the minimum value
            NONE: no blending, simply place the src tiles.
        kwargs:
            clip_lrtb (tuple): the left, right, top, bottom pixels in a source
                tile to be excluded (in case there are e.g consistent distortion
                at the begininig of the scan).
            scale (float): scale factor of the output image.
        refer to feabas.common.render_by_subregions for other kwargs
        """
        blend = kwargs.pop('blend', 'LINEAR')
        clip_ltrb = kwargs.pop('clip_lrtb', (0,0,0,0))
        scale = kwargs.pop('scale', 1)
        fillval = kwargs.get('fillval', self.image_loader.default_fillval)
        dtype_out = kwargs.get('dtype_out', self.image_loader.dtype)
        sigma = 2.5 # sigma for pyramid generation.
        weight_eps = 1e-3
        bbox = scale_coordinates(bbox, 1/scale)
        hits = list(self.mesh_tree.intersection(bbox, objects=True))
        if len(hits) == 0:
            return None
        elif len(hits) == 1:
            blend = None
        x_min0 = bbox[0]
        y_min0 = bbox[1]
        ht0 = round(bbox[3] - y_min0)
        wd0 = round(bbox[2] - x_min0)
        x0 = np.arange(x_min0, x_min0+wd0, 1/scale)
        y0 = np.arange(y_min0, y_min0+ht0, 1/scale)
        image_hp = None
        image_lp = None
        weight_sum = None
        weight_max = None
        for hit in hits:
            indx = hit.id
            bbox_mesh = common.bbox_enlarge(hit.bbox, 2)
            bbox_overlap, _ = common.bbox_intersections(bbox, bbox_mesh)
            msk_x = np.nonzero((x0 >= bbox_overlap[0]) & (x0 <= bbox_overlap[2]))[0]
            msk_y = np.nonzero((y0 >= bbox_overlap[1]) & (y0 <= bbox_overlap[3]))[0]
            slc_x = slice(np.min(msk_x), np.max(msk_x)+1, None)
            slc_y = slice(np.min(msk_y), np.max(msk_y)+1, None)
            x_msh = x0[slc_x]
            y_msh = y0[slc_y]
            xx, yy = np.meshgrid(x_msh, y_msh)
            x_interp, y_interp = self.interpolators[indx]
            offset = self._mesh_info[indx].moving_offsets.ravel()
            xxt = xx - offset[0]
            yyt = yy - offset[1]
            map_x = x_interp(xxt, yyt)
            mask =map_x.mask
            if np.all(mask, axis=None):
                continue
            map_y = y_interp(xxt, yyt)
            x_field = np.nan_to_num(map_x.data, nan=-1, copy=False)
            y_field = np.nan_to_num(map_y.data, nan=-1, copy=False)
            tile_ht, tile_wd = self.tile_size(indx)
            weight = np.minimum.reduce([x_field - clip_ltrb[0] + 0.5,
                                    - x_field + tile_wd - clip_ltrb[2] - 0.5,
                                    y_field - clip_ltrb[1] + 0.5,
                                    - y_field + tile_ht - clip_ltrb[3] - 0.5])
            mask = weight > 0
            if not np.any(mask, axis=None):
                continue
            expand_image = partial(common.expand_image, target_size=(y0.size, x0.size), slices=(slc_y, slc_x))
            imgt = common.render_by_subregions(x_field, y_field, mask, self.image_loader, fileid=indx, **kwargs)
            if blend is None:
                image_hp = expand_image(imgt)
                weight_sum = expand_image(weight)
                break
            if imgt is None:
                continue
            if not np.issubdtype(imgt.dtype, np.floating):
                imgt = imgt.astype(np.float32, copy=False)
            imgshp = imgt.shape
            weight = weight.clip(0, None).astype(np.float32)
            if len(imgshp) > 2:
                weight = np.stack((weight, )*imgt.shape[-1], axis=-1)
            if image_hp is None:
                image_hp = np.zeros_like(expand_image(imgt))
                image_lp = np.zeros_like(image_hp)
                weight_sum = np.zeros_like(expand_image(weight))
                weight_max = np.zeros_like(weight_sum)
                if blend == 'MAX':
                    image_hp = image_hp - np.inf
                elif blend == 'MIN':
                    image_hp = image_hp + np.inf
            weight_sum += expand_image(weight)
            if blend == 'LINEAR':
                image_hp = expand_image(imgt * weight) + image_hp
            elif blend == 'NEAREST':
                maskb = expand_image(weight) > weight_max
                image_hp[maskb] = expand_image(imgt)[maskb]
                weight_max[maskb] = expand_image(weight)[maskb]
            elif blend == 'PYRAMID':
                sigmas = [sigma, sigma] + [0]*(len(imgshp)-2)
                imgt_f = gaussian_filter(imgt, sigma=sigmas)
                imgt_h = imgt - imgt_f
                image_lp = expand_image(imgt_f * weight) + image_lp
                maskb = expand_image(weight) > weight_max
                image_hp[maskb] = expand_image(imgt_h)[maskb]
                weight_max[maskb] = expand_image(weight)[maskb]
            elif blend == 'MAX':
                maskb = expand_image(weight > 0)
                image_hp[maskb] = np.maximum(image_hp[maskb], expand_image(imgt)[maskb])
            elif blend == 'MIN':
                maskb = expand_image(weight > 0)
                image_hp[maskb] = np.minimum(image_hp[maskb], expand_image(imgt)[maskb])
            elif blend == 'NONE':
                maskb = expand_image(weight > 0)
                image_hp[maskb] = expand_image(imgt)[maskb]
            else:
                raise ValueError(f'unsupported blending mode {blend}')
        if image_hp is None:
            return None
        if blend == 'LINEAR':
            img_out = image_hp / weight_sum.clip(weight_eps, None)
        elif blend == 'PYRAMID':
            image_lp = image_lp / weight_sum.clip(weight_eps, None)
            img_out = image_lp + image_hp
        else:
            img_out = image_hp
        img_out[weight_sum <= weight_eps] = fillval
        if np.issubdtype(dtype_out, np.integer):
            iinfo = np.iinfo(dtype_out)
            img_out = img_out.clip(iinfo.min, iinfo.max)
        img_out = img_out.astype(dtype_out, copy=False)
        if np.all(img_out == fillval, axis=None):
            return None
        return img_out


    def render_series_to_file(self, bboxes, filenames, **kwargs):
        if isinstance(filenames, (dict, ts.TensorStore)):
            use_tensorstore = True
            rendered = []
        else:
            use_tensorstore = False
            rendered = {}
        scale = kwargs.get('scale', 1.0)
        if scale > 0.33:
            self.image_loader._preprocess = None
        else:
            ksz = round(0.5/scale) * 2 - 1
            self.image_loader._preprocess = partial(cv2.blur, ksize=(ksz, ksz))
        if not use_tensorstore: # render as image tiles
            for bbox, filename in zip(bboxes, filenames):
                if os.path.isfile(filename):
                    rendered[filename] = bbox
                    continue
                imgt = self.crop(bbox, **kwargs)
                if imgt is None:
                    continue
                common.imwrite(filename, imgt)
                rendered[filename] = bbox
        else: # use tensorstore
            if not isinstance(filenames, ts.TensorStore):
                dataset = ts.open(filenames).result()
            else:
                dataset = filenames
            driver = dataset.spec().to_json()['driver']
            if driver in ('neuroglancer_precomputed', 'n5'):
                kwargs['fillval'] = 0
            for bbox in bboxes:
                imgt = self.crop(bbox, **kwargs)
                if imgt is None:
                    continue
                xmin, ymin, _, _ = bbox
                shp = imgt.shape
                imght, imgwd = shp[0], shp[1]
                data_view = dataset[xmin:(xmin+imgwd), ymin:(ymin+imght)]
                data_view.write(imgt.T.reshape(data_view.shape)).result()
                rendered.append(tuple(bbox))
        self.image_loader.clear_cache()
        return rendered


    def plan_render_series(self, tile_size, prefix='', **kwargs):
        """
        Return the output filenames and bboxes that cover the entire montage in
        z-order. Set pattern to provide filename formatting appended to the
        prefix. Keywords in the formatting include:
            {ROW_IND}: row_index
            {COL_IND}: col_index
            {X_MIN}: minimum of x coordinates
            {Y_MIN}: minimum of y coordinates
            {X_MAX}: maximum of x coordinates
            {Y_MAX}: maximum of y coordinates
        e.g. pattern: Section001_tr{ROW_IND}_tc{COL_IND}.png
        """
        driver = kwargs.get('driver', 'image')
        scale = kwargs.get('scale', 1)
        resolution = self.resolution / scale
        if not hasattr(tile_size, '__len__'):
            tile_ht, tile_wd = tile_size, tile_size
        else:
            tile_ht, tile_wd = tile_size[0], tile_size[-1]
        bounds = self.bounds
        if scale != 1:
            bounds = scale_coordinates(bounds, scale)
        montage_wd = int(np.ceil(bounds[2]))
        montage_ht = int(np.ceil(bounds[3]))
        if driver != 'image':
            while tile_ht > montage_ht or tile_wd > montage_wd:
                tile_ht = tile_ht // 2
                tile_wd = tile_wd // 2
        Ncol = int(np.ceil(montage_wd / tile_wd))
        Nrow = int(np.ceil(montage_ht / tile_ht))
        cols, rows = np.meshgrid(np.arange(Ncol), np.arange(Nrow))
        cols, rows = cols.ravel(), rows.ravel()
        idxz = common.z_order(np.stack((rows, cols), axis=-1))
        cols, rows =  cols[idxz], rows[idxz]
        if driver == 'image':
            montage_ht, montage_wd = Nrow * tile_ht, Ncol * tile_wd
            filename_settings = kwargs.get('filename_settings', {})
            pattern = filename_settings.get('pattern', 'tr{ROW_IND}_tc{COL_IND}.png')
            one_based = filename_settings.get('one_based', False)
            keywords = ['{ROW_IND}', '{COL_IND}', '{X_MIN}', '{Y_MIN}', '{X_MAX}', '{Y_MAX}']
            filenames = []
        else:
            if not prefix.endswith('/'):
                prefix = prefix + '/'
            kv_headers = ('gs://', 'http://', 'https://', 'file://', 'memory://')
            for kvh in kv_headers:
                if prefix.startswith(kvh):
                    break
            else:
                prefix = 'file://' + prefix
            number_of_channels = self.number_of_channels
            dtype = self.dtype
            fillval = self.default_fillval
            schema = {
                "chunk_layout":{
                    "grid_origin": [0, 0, 0, 0],
                    "inner_order": [3, 2, 1, 0],
                    "read_chunk": {"shape": [tile_wd, tile_ht, 1, number_of_channels]},
                    "write_chunk": {"shape": [tile_wd, tile_ht, 1, number_of_channels]},
                },
                "domain":{
                    "exclusive_max": [montage_wd, montage_ht, 1, number_of_channels],
                    "inclusive_min": [0, 0, 0, 0],
                    "labels": ["x", "y", "z", "channel"]
                },
                "dimension_units": [[resolution, "nm"], [resolution, "nm"], [general_settings().get('section_thickness', 30), "nm"], None],
                "dtype": np.dtype(dtype).name,
                "rank" : 4
            }
            if driver == 'zarr':
                filenames = {
                    "driver": "zarr",
                    "kvstore": prefix + '0/',
                    "key_encoding": ".",
                    "metadata": {
                        "zarr_format": 2,
                        "fill_value": fillval,
                        "compressor": {"id": "gzip", "level": 6}
                    },
                    "schema": schema,
                    "open": True,
                    "create": True,
                    "delete_existing": False
                }
            elif driver == 'n5':
                filenames = {
                    "driver": "n5",
                    "kvstore": prefix + 's0/',
                    "metadata": {
                        "compression": {"type": "gzip"}
                    },
                    "schema": schema,
                    "open": True,
                    "create": True,
                    "delete_existing": False
                }
            elif driver == 'neuroglancer_precomputed':
                if tile_ht % 256 == 0:
                    read_ht = 256
                else:
                    read_ht = tile_ht
                if tile_wd % 256 == 0:
                    read_wd = 256
                else:
                    read_wd = tile_wd
                schema["codec"]= {
                    "driver": "neuroglancer_precomputed",
                    "encoding": "raw",
                    "shard_data_encoding": "gzip"
                }
                schema['chunk_layout']["read_chunk"]["shape"] = [read_wd, read_ht, 1, number_of_channels]
                filenames = {
                    "driver": "neuroglancer_precomputed",
                    "kvstore": prefix,
                    "schema": schema,
                    "open": True,
                    "create": True,
                    "delete_existing": False
                }
            else:
                raise ValueError(f'{driver} not supported')
        hits = []
        bboxes = []
        for r, c in zip(rows, cols):
            bbox = (c*tile_wd, r*tile_ht, min((c+1)*tile_wd, montage_wd), min((r+1)*tile_ht, montage_ht))
            if scale != 1:
                bbox_hit = scale_coordinates(bbox, 1/scale)
            else:
                bbox_hit = bbox
            hit = list(self.mesh_tree.intersection(bbox_hit, objects=False))
            if len(hit) == 0:
                continue
            hits.append(hit)
            bboxes.append(bbox)
            if driver == 'image':
                xmin, ymin, xmax, ymax = bbox
                keyword_replaces = [str(r+one_based), str(c+one_based), str(xmin), str(ymin), str(xmax), str(ymax)]
                fname = pattern
                for kw, kwr in zip(keywords, keyword_replaces):
                    fname = fname.replace(kw, kwr)
                filenames.append(prefix + fname)
        return bboxes, filenames, hits


    def divide_render_jobs(self, render_series, num_workers=1, **kwargs):
        max_tile_per_job = kwargs.get('max_tile_per_job', None)
        bboxes, filenames, hits = render_series
        num_tiles = len(bboxes)
        num_tile_per_job = max(1, num_tiles // num_workers)
        if max_tile_per_job is not None:
            num_tile_per_job = min(num_tile_per_job, max_tile_per_job)
        N_jobs = round(num_tiles / num_tile_per_job)
        indices = np.round(np.linspace(0, num_tiles, num=N_jobs+1, endpoint=True))
        indices = np.unique(indices).astype(np.uint32)
        bboxes_list = []
        filenames_list = []
        hits_list = []
        for idx0, idx1 in zip(indices[:-1], indices[1:]):
            idx0, idx1 = int(idx0), int(idx1)
            bboxes_list.append(bboxes[idx0:idx1])
            if isinstance(filenames, dict):
                filenames_list.append(filenames)
            else:
                filenames_list.append(filenames[idx0:idx1])
            hits_list.append(set(s for hit in hits[idx0:idx1] for s in hit))
        return bboxes_list, filenames_list, hits_list


    def tile_size(self, indx):
        if self._identical_tile_size:
            return self._tile_sizes[0]
        else:
            return self._tile_sizes[indx]


    def generate_roi_mask(self, scale, show_conn=False, mask_erode=0):
        """
        generate low resolution roi mask that can fit in a single image.
        """
        bboxes0 = []
        for msh in self._mesh_rtree_generator():
            _, bbox, _ = msh
            bboxes0.append(bbox)
        bboxes = scale_coordinates(np.array(bboxes0), scale).clip(0, None)
        bboxes = np.round(bboxes).astype(np.int32)
        imgwd, imght = np.max(bboxes[:,-2:], axis=0) + 2
        imgout = np.zeros((imght, imgwd), dtype=np.uint8)
        if show_conn and self._connected_subsystem is not None:
            lbls = np.maximum(1, 255 - self._connected_subsystem).astype(np.uint8)
        else:
            lbls = np.full(len(bboxes), 255, dtype=np.uint8)
        for bbox, lb in zip(bboxes, lbls):
            xmin, ymin, xmax, ymax = bbox
            imgout[ymin:ymax, xmin:xmax] = lb
        if mask_erode > 0:
            mask = (imgout > 0).astype(np.uint8)
            mask = cv2.erode(mask, np.ones((3,3), dtype=np.uint8), iterations=mask_erode)
            mask[:mask_erode, :] = 0
            mask[-mask_erode:, :] = 0
            mask[:, :mask_erode] = 0
            mask[:, -mask_erode:] = 0
            imgout[mask == 0] = 0
        if mask_erode < 0:
            mask = (imgout == 0).astype(np.uint8)
            D, L = cv2.distanceTransformWithLabels(mask,distanceType=cv2.DIST_L2,
                                                   maskSize=5, labelType=cv2.DIST_LABEL_PIXEL)
            M = imgout[mask==0]
            imgout = M[L - 1]
            imgout[D > -mask_erode] = 0
        return imgout


    def _mesh_rtree_generator(self):
        for k, msh in enumerate(self._mesh_info):
            vertices, offset = msh.moving_vertices, msh.moving_offsets.ravel()
            xy_min = vertices.min(axis=0)
            xy_max = vertices.max(axis=0)
            bbox = (xy_min[0] + offset[0], xy_min[1] + offset[1],
                xy_max[0] + offset[0], xy_max[1] + offset[1])
            yield (k, bbox, None)


    @property
    def mesh_tree(self):
        if self._rtree is None:
            self._rtree = index.Index(self._mesh_rtree_generator())
        return self._rtree


    @property
    def interpolators(self):
        if self._interpolators is None:
            self._interpolators = []
            for msh in self._mesh_info:
                v1, T, v0 = msh.moving_vertices, msh.triangles, msh.fixed_verticess
                mattri = matplotlib.tri.Triangulation(v1[:,0], v1[:,1], triangles=T)
                xinterp = matplotlib.tri.LinearTriInterpolator(mattri, v0[:,0])
                yinterp = matplotlib.tri.LinearTriInterpolator(mattri, v0[:,1])
                self._interpolators.append((xinterp, yinterp))
        return self._interpolators


    @property
    def image_loader(self):
        if self._image_loader is None:
            if self._identical_tile_size:
                tile_size = self._tile_sizes[0]
                self._image_loader = StaticImageLoader(self.imgrelpaths,
                    root_dir=self.imgrootdir, tile_size=tile_size,
                    **self._loader_settings)
            else:
                xy_min = np.zeros_like(self._tile_sizes)
                xy_max = self._tile_sizes[:,::-1]
                bboxes = np.concatenate((xy_min, xy_max), axis=-1)
                self._image_loader = StaticImageLoader(self.imgrelpaths,
                    bboxes=bboxes, root_dir=self.imgrootdir,
                    **self._loader_settings)
        return self._image_loader


    @property
    def bounds(self):
        return self.mesh_tree.bounds


    @property
    def number_of_channels(self):
        return self.image_loader.number_of_channels


    @property
    def dtype(self):
        return self.image_loader.dtype


    @property
    def default_fillval(self):
        return self.image_loader.default_fillval


    @staticmethod
    def subprocess_render_montages(montage, bboxes, outnames, **kwargs):
        selected = kwargs.pop('selected', None)
        if isinstance(montage, str):
            M = MontageRenderer.from_h5(montage, selected=selected, **kwargs)
        elif isinstance(montage, (list, tuple)):
            margs = montage[0]
            mkwargs = montage[1]
            M = MontageRenderer(*margs, **mkwargs)
        elif isinstance(montage, Stitcher):
            M = MontageRenderer.from_stitcher(montage, **kwargs)
        elif isinstance(montage, MontageRenderer):
            M = montage
        else:
            raise TypeError
        return M.render_series_to_file(bboxes, outnames, **kwargs)
