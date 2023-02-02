from concurrent.futures.process import ProcessPoolExecutor
import cv2
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
            stitching results. (xmin, ymin, xmax, ymax)
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
        bboxes = np.array(bboxes, copy=False)
        self.init_bboxes = bboxes
        self.tile_sizes = Stitcher.bbox_sizes(bboxes)
        self.average_tile_size = np.median(self.tile_sizes, axis=0)
        init_offset = Stitcher.bbox_centers(bboxes)
        self._init_offset = init_offset - init_offset.min(axis=0)


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
        NotImplemented


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
        bbox_ov, _ = Stitcher.bbox_intersections(bboxes0, bboxes1)
        ov_cntr = Stitcher.bbox_centers(bbox_ov)
        average_step_size = self.average_tile_size[::-1] / 2
        ov_indices = np.round((ov_cntr - ov_cntr.min(axis=0))/average_step_size)
        z_order = miscs.z_order(ov_indices)
        return overlaps[z_order]


    @property
    def overlaps(self):
        if (not hasattr(self, '_overlaps')) or (self._overlaps is None):
            self._overlaps = self.find_overlaps()
        return self._overlaps


    @staticmethod
    def bbox_centers(bboxes):
        bboxes = np.array(bboxes, copy=False)
        cntr = 0.5 * bboxes @ np.array([[1,0],[0,1],[1,0],[0,1]]) - 0.5
        return cntr


    @staticmethod
    def bbox_sizes(bboxes):
        bboxes = np.array(bboxes, copy=False)
        szs = bboxes @ np.array([[0,-1],[-1,0],[0,1],[1,0]])
        return szs.clip(0, None)


    @staticmethod
    def bbox_intersections(bboxes0, bboxes1):
        xy_min = np.maximum(bboxes0[...,:2], bboxes1[...,:2])
        xy_max = np.minimum(bboxes0[...,-2:], bboxes1[...,-2:])
        bbox_int = np.concatenate((xy_min, xy_max), axis=-1)
        valid = np.all((xy_max - xy_min) > 0, axis=-1)
        return bbox_int, valid



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
            a subset of all the images from the dispatcher
        loader_config(dict): key-word arguments passed to configure ImageLoader.
        matcher_config(dict): key-word arguments passed to configure matching.
    Return:
        matches(dict): res[(global_index0, global_index1)] = (xy0, xy1, conf).
    """
    root_dir = kwargs.get('root_dir', None)
    min_width = kwargs.get('min_width', 0)
    index_mapper = kwargs.get('index_mapper', None)
    loader_config = kwargs.get('loader_config', {})
    matcher_config = kwargs.get('matcher_config', {})
    NotImplemented
    