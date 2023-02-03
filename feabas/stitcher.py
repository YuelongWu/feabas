from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures import as_completed
import cv2
from functools import partial
from multiprocessing import get_context
import numpy as np
import os
from rtree import index

from feabas.matcher import stitching_matcher
from feabas.dal import StaticImageLoader
from feabas import optimizer, miscs
from feabas.constant import *


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
    def __init__(self, imgpaths, bboxes, **kwargs):
        root_dir = kwargs.get('root_dir', None)
        if bool(root_dir):
            self.imgrootdir = root_dir
            self.imgrelpaths = imgpaths
        else:
            self.imgrootdir = os.path.dirname(os.path.commonprefix(imgpaths))
            self.imgrelpaths = [os.path.relpath(s, self.imgrootdir) for s in imgpaths]
        bboxes = np.round(bboxes).astype(np.int32)
        self.init_bboxes = bboxes
        self.tile_sizes = miscs.bbox_sizes(bboxes)
        self.average_tile_size = np.median(self.tile_sizes, axis=0)
        init_offset = miscs.bbox_centers(bboxes)
        self._init_offset = init_offset - init_offset.min(axis=0)
        self.matches = {}


    @classmethod
    def from_coordinate_file(cls, filename, **kwargs):
        """
        initialize Stitcher from a coordinate txt file. Each row in the file
        follows the pattern:
            image_path  x_min  y_min  x_max(optional)  y_max(optional)
        if x_max and y_max is not provided, they are inferred from tile_size.
        If relative path is provided in the image_path colume, at the first line
        of the file, the root_dir can be defined as:
            {ROOT_DIR}  rootdir_to_the_path

        Args:
            filename(str): full path to the coordinate file.
        Kwargs:
            rootdir: if the imgpaths colume in the file is relative paths, can
                use this to prepend the paths. Set to None to disable.
            tile_size: the tile size used to compute the bounding boxes in the
                absense of x_max and y_max in the file. If None, will read an
                image file to figure out
            delimiter: the delimiter to separate each colume in the file. If set
                to None, any whitespace will be considered.
        """
        root_dir = kwargs.get('root_dir', None)
        tile_size = kwargs.get('tile_size', None)
        delimiter = kwargs.get('delimiter', '\t') # None for any whitespace
        imgpaths = []
        bboxes = []
        with open(filename, 'r') as f:
            lines = f.readlines()
        if len(lines) == 0:
            raise RuntimeError(f'empty file: {filename}')
        start_line = 0
        if '{ROOT_DIR}' in lines[0]:
            start_line += 1
            tlist = lines[0].strip().split(delimiter)
            if len(tlist) > 1:
                root_dir = tlist[1]
        relpath = bool(root_dir)
        for line in lines[start_line:]:
            line = line.strip()
            tlist = line.split(delimiter)
            if len(tlist) < 3:
                raise RuntimeError(f'corrupted coordinate file: {filename}')
            mpath = tlist[0]
            x_min = float(tlist[1])
            y_min = float(tlist[2])
            if len(tlist) >= 5:
                x_max = float(tlist[3])
                y_max = float(tlist[4])
            else:
                if tile_size is None:
                    if relpath:
                        mpath_f = os.path.join(root_dir, mpath)
                    else:
                        mpath_f = mpath
                    img = cv2.imread(mpath_f,cv2.IMREAD_GRAYSCALE)
                    tile_size = img.shape
                x_max = x_min + tile_size[-1]
                y_max = y_min + tile_size[0]
            imgpaths.append(mpath)
            bboxes.append((x_min, y_min, x_max, y_max))
        return cls(imgpaths, bboxes, root_dir=root_dir)


    def dispatch_matchers(self, **kwargs):
        """
        run matching between overlapping tiles.
        """
        over_write = kwargs.get('over_write', False)
        num_workers = kwargs.get('num_workers', 1)
        min_width = kwargs.get('min_width', 0)
        margin = kwargs.get('margin', 1.0)
        image_to_mask_path = kwargs.get('image_to_mask_path', None)
        matcher_config = kwargs.get('matcher_config', {})
        loader_config = kwargs.get('loader_config', {})
        if bool(loader_config.get('cache_size', None)) and (num_workers > 1):
            loader_config['cache_size'] = int(np.ceil(loader_config['cache_size'] / num_workers))
        loader_config['number_of_channels'] = 1 # only gray-scale matching are supported
        target_func = partial(Stitcher.match_list_of_overlaps,
                              root_dir=self.imgrootdir,
                              min_width=min_width,
                              margin=margin,
                              image_to_mask_path=image_to_mask_path,
                              loader_config=loader_config,
                              matcher_config=matcher_config)
        
        if over_write:
            self.matches = {}
            overlaps = self.overlaps
        else:
            overlaps = self.overlaps_without_matches
        num_overlaps = len(overlaps)
        if ((num_workers is not None) and (num_workers <= 1)) or (num_overlaps <= 1):
            new_matches = target_func(overlaps, self.imgrelpaths, self.init_bboxes)
            self.matches.update(new_matches)
            return len(new_matches)
        num_workers = min(num_workers, num_overlaps)
        # 180 roughly number of overlaps in an MultiSEM mFoV
        num_overlaps_per_job = min(num_overlaps//num_workers, 180)
        N_jobs = round(num_overlaps / num_overlaps_per_job)
        indx_j = np.linspace(0, num_overlaps, num=N_jobs+1, endpoint=True)
        indx_j = np.unique(np.round(indx_j).astype(np.int32))
        # divide works
        jobs = []
        num_new_matches = 0
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=get_context('spawn')) as executor:
            for idx0, idx1 in zip(indx_j[:-1], indx_j[1:]):
                ovlp_g = overlaps[idx0:idx1] # global indices of overlaps
                mapper, ovlp = np.unique(ovlp_g, return_inverse=True, axis=None)
                ovlp = ovlp.reshape(ovlp_g.shape)
                bboxes = self.init_bboxes[mapper]
                imgpaths = [self.imgrelpaths[s] for s in mapper]
                job = executor.submit(target_func, ovlp, imgpaths, bboxes, index_mapper=mapper)
                jobs.append(job)
            for job in as_completed(jobs):
                matches = job.result()
                num_new_matches += len(matches)
                self.matches.update(matches)
        return num_new_matches


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
        bbox_ov, _ = miscs.bbox_intersections(bboxes0, bboxes1)
        ov_cntr = miscs.bbox_centers(bbox_ov)
        average_step_size = self.average_tile_size[::-1] / 2
        ov_indices = np.round((ov_cntr - ov_cntr.min(axis=0))/average_step_size)
        z_order = miscs.z_order(ov_indices)
        return overlaps[z_order]


    @property
    def overlaps_without_matches(self):
        overlaps = self.overlaps
        return np.array([s for s in overlaps if (tuple(s) not in self.matches)])


    @property
    def overlaps(self):
        if (not hasattr(self, '_overlaps')) or (self._overlaps is None):
            self._overlaps = self.find_overlaps()
        return self._overlaps


    @staticmethod
    def match_list_of_overlaps(overlaps, imgpaths, bboxes, **kwargs):
        """
        matching function used as the target function in Stitcher.dispatch_matchers.
        Args:
            overlaps(Kx2 ndarray): pairs of indices of the images to match.
            imgpaths(list of str): the paths to the image files.
            bboxes(Nx4 ndarray): the estimated bounding boxes of each image.
        Kwargs:
            root_dir(str): if provided, the paths in imgpaths are relative paths to
                this root directory.
            min_width: minimum amount of overlapping width (in pixels) to use.
                overlaps below this threshold will be skipped.
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
            matches(dict): res[(global_index0, global_index1)] = (xy0, xy1, conf).
        """
        root_dir = kwargs.get('root_dir', None)
        min_width = kwargs.get('min_width', 0)
        index_mapper = kwargs.get('index_mapper', None)
        margin = kwargs.get('margin', 1.0)
        image_to_mask_path = kwargs.get('image_to_mask_path', None)
        loader_config = kwargs.get('loader_config', {})
        matcher_config = kwargs.get('matcher_config', {})
        margin_ratio_switch = 2
        if len(overlaps) == 0:
            return {}
        bboxes_overlap, wds = miscs.bbox_intersections(bboxes[overlaps[:,0]], bboxes[overlaps[:,1]])
        if 'cache_border_margin' not in loader_config:
            overlap_wd0 = 6 * np.median(np.abs(wds - np.median(wds))) + np.median(wds) + 1
            if margin < margin_ratio_switch:
                loader_config['cache_border_margin'] = int(overlap_wd0 * (1 + margin))
            else:
                loader_config['cache_border_margin'] = int(overlap_wd0 + margin)
        image_loader = StaticImageLoader(imgpaths, bboxes, root_dir=root_dir, **loader_config)
        if image_to_mask_path is not None:
            mask_paths = [s.replace(image_to_mask_path[0], image_to_mask_path[1])
                for s in image_loader.filepaths_generator]
            mask_exist = np.array([os.path.isfile(s) for s in mask_paths])
            loader_config.update({'apply_CLAHE': False,
                                  'inverse': False,
                                  'number_of_channels': None,
                                  'preprocess': None})
            mask_loader = StaticImageLoader(mask_paths, bboxes, root_dir=None, **loader_config)
        else:
            mask_exist = np.zeros(len(imgpaths), dtype=bool)
        matches = {}
        for indices, bbox_ov, wd in zip(overlaps, bboxes_overlap, wds):
            if wd <= min_width:
                continue
            if margin < margin_ratio_switch:
                real_margin = int(margin * wd)
            else:
                real_margin = int(margin)
            bbox_ov = miscs.bbox_enlarge(bbox_ov, real_margin)
            idx0, idx1 = indices
            bbox0 = bboxes[idx0]
            bbox1 = bboxes[idx1]
            bbox_ov0 = miscs.bbox_intersections(bbox_ov, bbox0)
            bbox_ov1 = miscs.bbox_intersections(bbox_ov, bbox1)
            img0 = image_loader.crop(bbox_ov0, idx0, return_index=False)
            img1 = image_loader.crop(bbox_ov1, idx1, return_index=False)
            if mask_exist[idx0]:
                mask0 = mask_loader.crop(bbox_ov0, idx0, return_index=False)
            else:
                mask0 = None
            if mask_exist[idx1]:
                mask1 = mask_loader.crop(bbox_ov1, idx1, return_index=False)
            else:
                mask1 = None
            weight, xy0, xy1 = stitching_matcher(img0, img1, mask0=mask0, mask1=mask1, **matcher_config)
            if xy0 is not None:
                offset0 = bbox_ov0[:2] - bbox0[:2]
                offset1 = bbox_ov1[:2] - bbox1[:2]
                xy0 = xy0 + offset0
                xy1 = xy1 + offset1
            if index_mapper is not None:
                idx0 = index_mapper[idx0]
                idx1 = index_mapper[idx1]
            matches[(idx0, idx1)] = (xy0, xy1, weight)
        return matches
