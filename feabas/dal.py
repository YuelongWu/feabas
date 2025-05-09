from abc import ABC
from collections import OrderedDict
from functools import partial
import gc
import json
import os
import re

import cv2
import numpy as np
from rtree import index
import tensorstore as ts

from feabas import common, caching
from feabas.storage import File, join_paths, list_folder_content, file_exists
from feabas.config import DEFAULT_RESOLUTION, TS_TIMEOUT, TS_RETRY


# bbox :int: [xmin, ymin, xmax, ymax]

##------------------------------ tile dividers -------------------------------##
# tile divider functions to divide an image tile to smaller block for caching
# Args:
#    imght, imgwd (int): height/width of image bounding box
#    x0, y0 (int): x, y coordinates of top-left conrer of image bounding box
# Returns:
#    divider: dict[block_id] = [xmin, ymin, xmax, ymax]
# Tiles with tile_id <=0 will not be cached. (DynamicImageLoader._cache_block)

def _tile_divider_blank(imght, imgwd, x0=0, y0=0):
    """Cache image as a whole without division."""
    divider = OrderedDict()
    divider[1] = (x0, y0, x0+imgwd, y0+imght)
    return divider


def _tile_divider_border(imght, imgwd, x0=0, y0=0, cache_border_margin=10):
    """Divide image borders to cache separately. Interior is not cached."""
    divider = OrderedDict()
    if (cache_border_margin == 0) or (cache_border_margin >= min(imght//2, imgwd//2)):
        divider[1] = (x0, y0, x0+imgwd, y0+imght)
    else:
        border_ht = min(cache_border_margin, imght // 2)
        border_wd = min(cache_border_margin, imgwd // 2)
        x1, x2, x3, x4 = 0, border_wd, (imgwd - border_wd), imgwd
        y1, y2, y3, y4 = 0, border_ht, (imght - border_ht), imght
        # start from the interior (not cached) so it appears earlier in rtree
        if (x2 < x3) and (y2 < y3):
            divider[0] = (x2 + x0, y2 + y0, x3 + x0, y3 + y0)
        divider.update({
            1: (x1 + x0, y1 + y0, x3 + x0, y2 + y0),
            2: (x1 + x0, y2 + y0, x2 + x0, y4 + y0),
            3: (x2 + x0, y3 + y0, x4 + x0, y4 + y0),
            4: (x3 + x0, y1 + y0, x4 + x0, y3 + y0),
        })
    return divider


def _tile_divider_block(imght, imgwd, x0=0, y0=0, cache_block_size=0):
    """Divide image to equal(ish) square(ish) blocks to cache separately."""
    divider = OrderedDict()
    if cache_block_size == 0:
        divider[1] = (x0, y0, x0+imgwd, y0+imght)
    else:
        Nx = max(round(imgwd / cache_block_size), 1)
        Ny = max(round(imght / cache_block_size), 1)
        xx = np.round(np.linspace(x0, x0+imgwd, Nx+1)).astype(int)
        yy = np.round(np.linspace(y0, y0+imght, Ny+1)).astype(int)
        xe0, ye0 = np.meshgrid(xx[:-1], yy[:-1])
        xe1, ye1 = np.meshgrid(xx[1:], yy[1:])
        xe0 = xe0.ravel()
        ye0 = ye0.ravel()
        xe1 = xe1.ravel()
        ye1 = ye1.ravel()
        indx = np.argsort(xe0 + ye0 + xe1 + ye1, axis=None)
        key = 1
        for xmin, ymin, xmax, ymax in zip(xe0[indx], ye0[indx], xe1[indx], ye1[indx]):
            divider[key] = (xmin, ymin, xmax, ymax)
            key += 1
    return divider


##------------------------------ image loaders -------------------------------##

def get_loader_from_json(json_info, loader_type=None, **kwargs):
    if isinstance(json_info, str) and json_info.endswith('.txt'):
        if loader_type == 'StaticImageLoader':
            loader = StaticImageLoader.from_coordinate_file(json_info)
        else:
            loader = MosaicLoader.from_coordinate_file(json_info)
        json_obj = loader.init_dict()
    else:
        json_obj, _ = common.parse_json_file(json_info)
    if kwargs.get('mip', None) is not None:
        if str(kwargs['mip']) in json_obj:
             json_obj = json_obj[str(kwargs['mip'])]
        elif kwargs['mip'] in json_obj:
             json_obj = json_obj[kwargs['mip']]
    if ('kvstore' in json_obj) or ('base' in json_obj):
        return TensorStoreLoader.from_json_spec(json_obj, **kwargs)
    json_obj.update(kwargs)
    if loader_type is None:
        loader_type = json_obj['ImageLoaderType']
    if loader_type == 'DynamicImageLoader':
        return DynamicImageLoader.from_json(json_obj)
    elif loader_type == 'StaticImageLoader':
        return StaticImageLoader.from_json(json_obj)
    elif loader_type == 'MosaicLoader':
        return MosaicLoader.from_json(json_obj)
    elif loader_type == 'StreamLoader':
        return StreamLoader.from_init_dict(json_obj)
    elif loader_type == 'TensorStoreLoader':
        return TensorStoreLoader.from_json(json_obj)
    else:
        raise ValueError


class AbstractImageLoader(ABC):
    """
    Abstract class for image loader.
    Kwargs:
        dtype: datatype of the output images. Default to None same as input.
        number_of_channels: # of channels of the output images. Default to
            None same as input.
        apply_CLAHE(bool): whether to apply CLAHE on output images.
        inverse(bool): whether to invert images.
        fillval(scalar): fill value for missing data.
        cache_size(int): length of image cache.
        cache_capacity(float): capacity of image cache in MiB.
        cache_border_margin(int)/cache_block_size(int): the border width
            for _tile_divider_border caching, or the square block size for
            _tile_divider_block caching. If neither is set, cache
            the entile image with _tile_divider_blank.
        resolution(float): resolution of the images. default to 4nm
    """
    def __init__(self, **kwargs):
        self._dtype = kwargs.get('dtype', None)
        self._number_of_channels = kwargs.get('number_of_channels', None)
        self._apply_CLAHE = kwargs.get('apply_CLAHE', False)
        clahe_clip_limit = kwargs.get('CLAHE_cliplimit', 2.0)
        self._clahe_clip_limit = clahe_clip_limit
        self._CLAHE = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8,8))
        self._inverse = kwargs.get('inverse', False)
        self._default_fillval = kwargs.get('fillval', 0)
        self._cache_size = kwargs.get('cache_size', 0)
        self._cache_capacity = kwargs.get('cache_capacity', None)
        self._use_cache = (self._cache_size is None) or (self._cache_size > 0)
        self._init_tile_divider(**kwargs)
        self._cache_type = kwargs.get('cache_type', 'mfu')
        self._cache = caching.generate_cache(self._cache_type, maxlen=self._cache_size, maxbytes=self._cache_capacity)
        preprocess_factory = kwargs.get('preprocess', None)
        preprocess_parames = kwargs.get('preprocess_params', {})
        self.update_preprocess_function(preprocess_factory, **preprocess_parames)
        self.resolution = kwargs.get('resolution', DEFAULT_RESOLUTION)
        self._read_counter = 0


    def clear_cache(self, instant_gc=False):
        # instant_gc: when True, instantly call garbage collection
        self._cache.clear(instant_gc)


    def update_preprocess_function(self, preprocess_factory, **preprocess_parames):
        self._preprocess_factory = preprocess_factory
        self._preprocess_parames = preprocess_parames
        self._preprocess = common.str_to_func(preprocess_factory, **preprocess_parames)


    def CLAHE_off(self):
        if self._apply_CLAHE:
            self.clear_cache()
            self._apply_CLAHE = False


    def CLAHE_on(self):
        if not self._apply_CLAHE:
            self.clear_cache()
            self._apply_CLAHE = True


    def inverse_off(self):
        if self._inverse:
            self.clear_cache()
            self._inverse = False


    def inverse_on(self):
        if not self._inverse:
            self.clear_cache()
            self._inverse = True


    def init_dict(self, **kwargs):
        out = {'ImageLoaderType': self.__class__.__name__}
        out.update(self._export_dict(**kwargs))
        return out


    def save_to_json(self, jsonname=None, **kwargs):
        out = self.init_dict(**kwargs)
        if jsonname is None:
            return json.dumps(out, indent=2)
        else:
            with File(jsonname, 'w') as f:
                json.dump(out, f, indent=2)


    def _cached_block_rtree_generator(self, bbox):
        x0 = bbox[0]
        y0 = bbox[1]
        imght = bbox[3] - bbox[1]
        imgwd = bbox[2] - bbox[0]
        for blkid, blkbbox in self._tile_divider(imght, imgwd, x0=x0, y0=y0).items():
            yield (blkid, blkbbox, None)


    def _crop_from_one_image(self, bbox, imgpath, return_empty=False, return_index=False, **kwargs):
        fillval = kwargs.get('fillval', self._default_fillval)
        dtype = kwargs.get('dtype', self.dtype)
        number_of_channels = kwargs.get('number_of_channels', self.number_of_channels)
        if (not self._use_cache) or (imgpath not in self._cache):
            imgout = self._crop_from_one_image_without_cache(bbox, imgpath, return_empty=return_empty, return_index=return_index, **kwargs)
        else:
            # find the intersection between crop bbox and cached block bboxes
            bbox_img = self._get_image_bbox(imgpath)
            hits = self._get_image_hits(imgpath, bbox)
            if not hits:
                # no overlap, return trivial results based on output control
                if return_empty and not return_index:
                    if dtype is None or number_of_channels is None:
                      # not sufficient info to generate empty tile, read an image to get info
                        img = self._read_image(imgpath, **kwargs)
                        imgout = common.crop_image_from_bbox(img, bbox_img, bbox, return_index=False,
                            return_empty=True, fillval=fillval)
                    else:
                        outht = bbox[3] - bbox[1]
                        outwd = bbox[2] - bbox[0]
                        if number_of_channels <= 1:
                            outsz = (outht, outwd)
                        else:
                            outsz = (outht, outwd, number_of_channels)
                        imgout = np.full(outsz, fillval, dtype=dtype)
                elif return_index:
                    imgout = None, None
                else:
                    imgout = None
            elif any(item not in self._get_cached_dict(imgpath) for item in hits):
                # has missing blocks from cache, need to read image anyway
                imgout = self._crop_from_one_image_without_cache(bbox, imgpath, return_empty=return_empty, return_index=return_index, **kwargs)
            else:
                # read from cache
                if return_index:
                    px_max, py_max, px_min, py_min  = bbox_img
                    for bbox_blk in hits.values():
                        tx_min, ty_min, tx_max, ty_max = [int(s) for s in bbox_blk]
                        px_min = min(px_min, tx_min)
                        py_min = min(py_min, ty_min)
                        px_max = max(px_max, tx_max)
                        py_max = max(py_max, ty_max)
                    bbox_partial = (px_min, py_min, px_max, py_max)
                else:
                    bbox_partial = bbox
                initialized = False
                cache_dict = self._get_cached_dict(imgpath)
                for blkid, blkbbox in hits.items():
                    blkbbox = [int(s) for s in blkbbox]
                    blk = cache_dict[blkid]
                    if blk is None:
                        continue
                    if not initialized:
                        imgp = common.crop_image_from_bbox(blk, blkbbox, bbox_partial,
                            return_index=False, return_empty=True, fillval=fillval)
                        initialized = True
                    else:
                        blkt, indx =  common.crop_image_from_bbox(blk, blkbbox, bbox_partial,
                            return_index=True, return_empty=False, fillval=fillval)
                        if indx is not None and blkt is not None:
                            imgp[indx] = blkt
                if return_index:
                    imgout = common.crop_image_from_bbox(imgp, bbox_partial, bbox, return_index=True,
                        return_empty=return_empty, fillval=fillval)
                else:
                    imgout = imgp
        return imgout


    def _crop_from_one_image_without_cache(self, bbox, imgpath, return_empty=False, return_index=False, **kwargs):
        # directly crop the image without checking the cache first
        fillval = kwargs.get('fillval', self._default_fillval)
        img = self._read_image(imgpath, **kwargs)
        bbox_img = self._get_image_bbox(imgpath)
        imgout = common.crop_image_from_bbox(img, bbox_img, bbox, return_index=return_index,
            return_empty=return_empty, fillval=fillval)
        if self._use_cache:
            self._cache_image(imgpath, img=img, **kwargs)
        return imgout


    def _settings_dict(self, **kwargs):
        output_controls = kwargs.get('output_controls', True)
        cache_settings = kwargs.get('cache_settings', True)
        out = {}
        out['resolution'] = self.resolution
        if output_controls:
            if self._dtype is not None:
                out['dtype'] = np.dtype(self._dtype).str
            if self._number_of_channels is not None:
                out['number_of_channels'] = self._number_of_channels
            out['fillval'] = self._default_fillval
            out['apply_CLAHE'] = self._apply_CLAHE
            if hasattr(self, '_clahe_clip_limit'):
                out['CLAHE_cliplimit'] = self._clahe_clip_limit
            out['inverse'] = self._inverse
            if self._preprocess_factory is not None:
                out['preprocess'] = common.func_to_str(self._preprocess_factory)
                preprocess_params = {}
                for key, val in self._preprocess_parames.items():
                    if isinstance(val, np.ndarray):
                        val = val.tolist()
                    preprocess_params[key] = val
                if preprocess_params:
                    out['preprocess_params'] = preprocess_params
        if cache_settings:
            out['cache_size'] = self._cache_size
            out['cache_type'] = self._cache_type
            if self._tile_divider_type == 'border':
                out['cache_border_margin'] = self._cache_border_margin
            elif self._tile_divider_type == 'block':
                out['cache_block_size'] = self.cache_block_size
        return out


    def _init_tile_divider(self, **kwargs):
        if not self._use_cache:
            self._tile_divider = _tile_divider_blank
            self._tile_divider_type = 'blank'
        elif 'cache_border_margin' in kwargs:
            self._tile_divider = partial(_tile_divider_border, cache_border_margin=kwargs['cache_border_margin'])
            self._tile_divider_type = 'border'
        elif 'cache_block_size' in kwargs:
            self._tile_divider = partial(_tile_divider_block, cache_block_size=kwargs['cache_block_size'])
            self._tile_divider_type = 'block'
        else:
            self._tile_divider = _tile_divider_blank
            self._tile_divider_type = 'blank'


    @staticmethod
    def _load_settings_from_json(jsonname):
        json_obj, _ = common.parse_json_file(jsonname)
        settings = {}
        if 'resolution' in json_obj:
            settings['resolution'] = json_obj['resolution']
        if 'dtype' in json_obj:
            settings['dtype'] = np.dtype(json_obj['dtype'])
        if 'number_of_channels' in json_obj:
            settings['number_of_channels'] = int(json_obj['number_of_channels'])
        if 'fillval' in json_obj:
            settings['fillval'] = json_obj['fillval']
        if 'apply_CLAHE' in json_obj:
            settings['apply_CLAHE'] = json_obj['apply_CLAHE']
        if 'CLAHE_cliplimit' in json_obj:
            settings['CLAHE_cliplimit'] = json_obj['CLAHE_cliplimit']
        if 'inverse' in json_obj:
            settings['inverse'] = json_obj['inverse']
        if 'preprocess' in json_obj:
            settings['preprocess'] = json_obj['preprocess']
        if 'preprocess_params' in json_obj:
            settings['preprocess_params'] = json_obj['preprocess_params']
        if 'cache_size' in json_obj:
            settings['cache_size'] = json_obj['cache_size']
        if 'cache_type' in json_obj:
            settings['cache_type'] = json_obj['cache_type']
        if 'cache_border_margin' in json_obj:
            settings['cache_border_margin'] = json_obj['cache_border_margin']
        elif 'cache_block_size' in json_obj:
            settings['cache_block_size'] = json_obj['cache_block_size']
        return settings, json_obj


    def _read_image(self, imgpath, **kwargs):
        number_of_channels = kwargs.get('number_of_channels', self._number_of_channels)
        dtype = kwargs.get('dtype', self._dtype)
        apply_CLAHE = kwargs.get('apply_CLAHE', self._apply_CLAHE)
        inverse = kwargs.get('inverse', self._inverse)
        if (number_of_channels == 3) and (np.dtype(dtype) == np.uint8):
            img = common.imread(imgpath, flag=cv2.IMREAD_COLOR)
        elif (number_of_channels == 1) and np.dtype(dtype) == np.uint8:
            img = common.imread(imgpath, flag=cv2.IMREAD_GRAYSCALE)
        else:
            img = common.imread(imgpath, flag=cv2.IMREAD_UNCHANGED)
        self._read_counter += 1
        if img is None:
            raise RuntimeError(f'Image file {imgpath} not valid!')
        if dtype is None:
            dtype = img.dtype
        while (len(img.shape) > 2) and (img.shape[-1] == 1):
            img = img[..., 0]
        if (number_of_channels == 1):
            while (len(img.shape) > 2):
                img = img.mean(axis=-1).astype(dtype)
        if apply_CLAHE:
            if (len(img.shape) > 2) and (img.shape[-1] == 3):
                img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
                img[:,:,0] = self._CLAHE.apply(img[:,:,0])
                img = cv2.cvtColor(img, cv2.COLOR_Lab2RGB)
            else:
                img = self._CLAHE.apply(img)
        if self._preprocess is not None:
            img = self._preprocess(img)
        if inverse:
            img = common.inverse_image(img, dtype)
        if (img.dtype == np.uint16) and (np.dtype(dtype) == np.uint8):
            img = img / 255
        return img.astype(dtype, copy=False)


    def _export_dict(self, **kwargs):
        """turn the loader settings to a dict used in self.save_to_json"""
        return self._settings_dict(**kwargs)


    def _get_cached_dict(self, imgpath):
        """cached_dict get function used in self._crop_from_one_image"""
        pass


    def _get_image_bbox(self, imgpath):
        """image_bbox get function used in self._crop_from_one_image"""
        pass


    def _get_image_cached_block_rtree(self, imgpath):
        """image_cached_block_rtree get function used in self._crop_from_one_image"""
        pass


    def _get_image_hits(self, imgpath, bbox):
        # hits[blkid] = bbox
        cached_block_rtree = self._get_image_cached_block_rtree(imgpath)
        hit_list = list(cached_block_rtree.intersection(bbox, objects=True))
        hits ={item.id: item.bbox for item in hit_list}
        return hits


    @property
    def number_of_channels(self):
        return self._number_of_channels


    @property
    def dtype(self):
        return self._dtype


    @property
    def default_fillval(self):
        return self._default_fillval



class DynamicImageLoader(AbstractImageLoader):
    """
    Class for image loader without predetermined image list.
    Mostly for caching & output formate control.
    caching format:
        self._cache[imgpath] = ((0,0,imgwd, imght), cache_dict{blkid: tile})
        self._cached_block_rtree[(imght, imgwd)] = rtree
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cached_block_rtree = {}


    def crop(self, bbox, imgpath, return_empty=False, return_index=False, **kwargs):
        return super()._crop_from_one_image(bbox, imgpath, return_empty=return_empty, return_index=return_index, **kwargs)


    @classmethod
    def from_json(cls, jsonname, **kwargs):
        settings, _ = cls._load_settings_from_json(jsonname)
        settings.update(kwargs)
        return cls(**settings)


    def _cache_image(self, imgpath, img=None, **kwargs):
        # self._cache[imgpath] = ((0, 0, imgwd, imght), cache_dict{blkid: tile})
        if not self._use_cache:
            return
        if imgpath in self._cache:
            cache_dict = self._cache[imgpath]
            bbox_img = cache_dict[0]
            x0, y0, x1, y1 = bbox_img
            imgwd = x1 - x0
            imght = y1 - y0
        else:
            cache_dict = {}
            if img is None:
                img = self._read_image(imgpath, **kwargs)
            imght = img.shape[0]
            imgwd = img.shape[1]
            x0 = 0
            y0 = 0
            bbox_img = (x0, y0, x0+imgwd, y0+imght)
        if bbox_img not in self._cached_block_rtree:
            self._cached_block_rtree[bbox_img] = index.Index(
                self._cached_block_rtree_generator(bbox_img), interleaved=True)
        divider = self._tile_divider(imght, imgwd, x0=x0, y0=y0)
        new_cache = False
        for bid, blkbbox in divider.items():
            if (bid > 0) and (bid not in cache_dict):
                if img is None:
                    img = self._read_image(imgpath, **kwargs)
                blk = img[blkbbox[1]:blkbbox[3], blkbbox[0]:blkbbox[2],...]
                if blk.size != img.size:
                    blk = blk.copy()
                cache_dict[bid] = blk
                new_cache = True
        if new_cache:
            self._cache[imgpath] = ((0, 0, imgwd,imght), cache_dict)


    def _get_cached_dict(self, imgpath):
        if imgpath not in self._cache:
            image_not_in_cache = '{} not in the image loader cache'.format(imgpath)
            raise KeyError(image_not_in_cache)
        else:
            return self._cache[imgpath][1]


    def _get_image_bbox(self, imgpath):
        if imgpath in self._cache:
            return self._cache[imgpath][0]
        else:
            img = self._read_image(imgpath)
            imght = img.shape[0]
            imgwd = img.shape[1]
            return (0,0, imgwd, imght)


    def _get_image_cached_block_rtree(self, imgpath):
        bbox_img = self._get_image_bbox(imgpath)
        if bbox_img not in self._cached_block_rtree:
            self._cached_block_rtree[bbox_img] = index.Index(
                    self._cached_block_rtree_generator(bbox_img), interleaved=True)
        return self._cached_block_rtree[bbox_img]



class StaticImageLoader(AbstractImageLoader):
    """
    Class for image loader with predetermined image list.
    Assuming all the images are of the same dtype and bitdepth.

    Args:
        filepaths(list): fullpaths of the image file list.
        bboxes(ndarray of int): bounding boxes of each file. If set to None,
            assume all the tiles have the same tile size (either by reading in
            the first image or defined by the key-word argument) and all the
            bboxes are default to (0, 0, imgwd, imght).
    Kwargs:
        root_dir(str): if set, the paths in filepaths are relative path to this
            root directory.
        tile_size(tuple): the size of tiles. If None(default), infer from the
            first image encountered.
        The rest of the kwargs are passed to AbstractImageLoader constructor.
    caching format:
        self._cache[imgpath] = cache_dict{blkid: tile}
    """
    def __init__(self, filepaths, bboxes=None, **kwargs):
        super().__init__(**kwargs)
        if bboxes is None:
            bboxes = []
        if bool(kwargs.get('root_dir', None)):
            self.imgrootdir = kwargs['root_dir']
            self.imgrelpaths = filepaths
        else:
            self.imgrootdir = os.path.dirname(os.path.commonprefix(filepaths))
            self.imgrelpaths = [os.path.relpath(s, self.imgrootdir) for s in filepaths]
        self.check_filename_uniqueness()
        if len(bboxes) == 0:
            tile_size = kwargs.get('tile_size', None)
            if tile_size is None:
                imgpath = join_paths(self.imgrootdir, self.imgrelpaths[0])
                img = super()._read_image(imgpath, number_of_channels=1, apply_CLAHE=False, inverse=False)
                tile_size = img.shape[:2]
            bboxes = np.tile((0, 0, tile_size[-1], tile_size[0]), (len(filepaths), 1))
        self._file_bboxes = np.round(bboxes).astype(np.int32)
        self._cached_block_rtree = {}
        self._divider = {}


    def check_filename_uniqueness(self):
        assert(len(self.imgrelpaths) > 0), 'empty file list'
        assert len(set(self.imgrelpaths)) == len(self.imgrelpaths), 'duplicated filenames'


    def crop(self, bbox, fileid, return_empty=False, return_index=False, **kwargs):
        if isinstance(fileid, str):
            filepath = fileid
            fileid = self.fileid_lookup(filepath)
        return super()._crop_from_one_image(bbox, fileid, return_empty=return_empty, return_index=return_index, **kwargs)


    @classmethod
    def from_json(cls, jsonname, **kwargs):
        settings, json_obj = cls._load_settings_from_json(jsonname)
        filepaths = []
        if 'root_dir' in json_obj:
            settings['root_dir'] = json_obj['root_dir']
        if 'tile_size' in json_obj:
            settings['tile_size'] = json_obj['tile_size']
        settings.update(kwargs)
        filepaths = []
        bboxes = []
        bboxes_included = True
        for f in json_obj['images']:
            filepaths.append(f['filepath'])
            if bboxes_included:
                if 'bbox' in f:
                    bboxes.append(f['bbox'])
                else:
                    bboxes = False
        return cls(filepaths=filepaths, bboxes=bboxes, **settings)


    @classmethod
    def from_coordinate_file(cls, filename, **kwargs):
        imgpaths, bboxes, root_dir, resolution = common.parse_coordinate_files(filename, **kwargs)
        kwargs.setdefault('root_dir', root_dir)
        if resolution is not None:
            kwargs.setdefault('resolution', resolution)
        return cls(filepaths=imgpaths, bboxes=bboxes, **kwargs)


    def to_coordinate_file(self, filename, **kwargs):
        delimiter = kwargs.get('delimiter', '\t')
        with File(filename, 'w') as f:
            f.write(f'{{ROOT_DIR}}{delimiter}{self.imgrootdir}\n')
            f.write(f'{{RESOLUTION}}{delimiter}{self.resolution}\n')
            for imgpath, bbox in zip(self.imgrelpaths, self._file_bboxes):
                bbox_str = [str(s) for s in bbox]
                line = delimiter.join((imgpath, *bbox_str))
                f.write(line+'\n')


    def _cache_image(self, fileid, img=None, **kwargs):
        if not self._use_cache:
            return
        if isinstance(fileid, str):
            filepath = fileid
            fileid = self.fileid_lookup(filepath)
            if fileid == -1:
                image_not_in_list = '{} not in the image loader list'.format(filepath)
                raise KeyError(image_not_in_list)
        if fileid in self._cache:
            cache_dict = self._cache[fileid]
        else:
            cache_dict = {}
        new_cache = False
        divider = self.divider(fileid)
        for bid, blkbbox in divider.items():
            if (bid > 0) and (bid not in cache_dict):
                if img is None:
                    img = self._read_image(fileid, **kwargs)
                blk = img[blkbbox[1]:blkbbox[3], blkbbox[0]:blkbbox[2],...]
                if blk.size != img.size:
                    blk = blk.copy()
                cache_dict[bid] = blk
                new_cache = True
        if new_cache:
            self._cache[fileid] = cache_dict


    def _export_dict(self, output_controls=True, cache_settings=True, image_list=True):
        out = super()._settings_dict(output_controls=output_controls, cache_settings=cache_settings)
        out['root_dir'] = self.imgrootdir
        if image_list:
            out['images'] = [{'filepath':p, 'bbox':b.tolist()} for p, b in zip(self.imgrelpaths, self._file_bboxes)]
        return out


    def _get_cached_dict(self, fileid):
        if fileid not in self._cache:
            if isinstance(fileid, str):
                imgpath = fileid
            else:
                imgpath = join_paths(self.imgrootdir, self.imgrelpaths[fileid])
            raise KeyError(f'{imgpath} not in the image loader cache')
        else:
            return self._cache[fileid]


    def _get_image_bbox(self, fileid):
        if isinstance(fileid, str):
            filepath = fileid
            fileid = self.fileid_lookup(filepath)
            if fileid == -1:
                raise KeyError(f'{filepath} not in the image loader list')
        return self._file_bboxes[fileid]


    def file_bboxes(self, margin=0):
        for bbox in self._file_bboxes:
            bbox_m = [bbox[0]-margin, bbox[1]-margin, bbox[2]+margin, bbox[3]+margin]
            yield bbox_m


    def _get_image_hits(self, fileid, bbox):
        # hits[blkid] = bbox
        imgbbox = self._get_image_bbox(fileid)
        xmin_img, ymin_img, _, _ = imgbbox
        bbox_normalized = (bbox[0]-xmin_img, bbox[1]-ymin_img,
            bbox[2]-xmin_img, bbox[3]-ymin_img)
        cached_block_rtree_normalized = self._get_image_cached_block_rtree(fileid)
        hit_list = list(cached_block_rtree_normalized.intersection(bbox_normalized, objects=True))
        hits = {}
        for item in hit_list:
            bbox_hit = item.bbox
            bbox_out = (bbox_hit[0]+xmin_img, bbox_hit[1]+ymin_img,
                bbox_hit[2]+xmin_img, bbox_hit[3]+ymin_img)
            hits[item.id] = bbox_out
        return hits


    def _get_image_cached_block_rtree(self, fileid):
        bbox_img = self._get_image_bbox(fileid)
        xmin, ymin, xmax, ymax = bbox_img
        # normalize bbox to reduce number of rtrees needed
        imgwd, imght = round(xmax - xmin), round(ymax - ymin)
        bbox_normalized = (0, 0, imgwd, imght)
        if bbox_normalized not in self._cached_block_rtree:
            self._cached_block_rtree[bbox_normalized] = index.Index(
                    self._cached_block_rtree_generator(bbox_normalized), interleaved=True)
        return self._cached_block_rtree[bbox_normalized]


    def _read_image(self, fileid, **kwargs):
        if isinstance(fileid, str):
            imgpath = fileid
        else:
            imgpath = join_paths(self.imgrootdir, self.imgrelpaths[fileid])
        img = super()._read_image(imgpath, **kwargs)
        if self._dtype is None:
            self._dtype = img.dtype
        if self._number_of_channels is None:
            if len(img.shape) <= 2:
                self._number_of_channels = 1
            else:
                self._number_of_channels = img.shape[-1]
        return img


    def fileid_lookup(self, fname):
        if (not hasattr(self, '_fileid_LUT')) or (self._fileid_LUT is None):
            self._fileid_LUT = {}
            for idx, fnm in enumerate(self.filepaths_generator):
                self._fileid_LUT[fnm] = idx
        if fname in self._fileid_LUT:
            return self._fileid_LUT[fname]
        else:
            return -1


    def divider(self, fileid):
        bbox_img = self._get_image_bbox(fileid)
        xmin, ymin, xmax, ymax = bbox_img
        # normalize bbox to reduce number of rtrees needed
        imgwd, imght = round(xmax - xmin), round(ymax - ymin)
        bbox_normalized = (0, 0, imgwd, imght)
        if bbox_normalized not in self._divider:
            self._divider[bbox_normalized] = self._tile_divider(imght, imgwd, x0=0, y0=0)
        return self._divider[bbox_normalized]


    @property
    def dtype(self):
        if self._dtype is None:
            self._read_image(0)
        return self._dtype


    @property
    def filepaths_generator(self):
        for fname in self.imgrelpaths:
            yield join_paths(self.imgrootdir, fname)


    @property
    def filepaths(self):
        return list(self.filepaths_generator)


    @property
    def number_of_channels(self):
        if self._number_of_channels is None:
            self._read_image(0)
        return self._number_of_channels



class MosaicLoader(StaticImageLoader):
    """
    Image loader to render cropped images from non-overlapping tiles
        bbox :int: [xmin, ymin, xmax, ymax]
    """
    def __init__(self, filepaths, bboxes, **kwargs):
        assert len(filepaths) == len(bboxes)
        super().__init__(filepaths, bboxes, **kwargs)
        self._init_rtrees()


    @classmethod
    def from_filepath(cls, imgpaths, pattern=r'_tr({ROW_IND}\d+)-tc({COL_IND}\d+)', **kwargs):
        """
        pattern: See MosaicLoader._filename_parser
        tile_size: Size of the tile. If not provide, read-in the first image.
        tile_offset: x-y offset added to the bboxes in the unit of tile size.
        pixel_offset: x-y offset added to the bboxes in the unit of pixel.
        """
        tile_size = kwargs.get('tile_size', None)   # if None, read in first image
        tile_offset = kwargs.get('tile_offset', None)
        pixel_offset = kwargs.get('pixel_offset', None)
        if isinstance(imgpaths, str):
            if '*' in imgpaths:
                imgpaths = list_folder_content(imgpaths)
                assert bool(imgpaths), 'No image found: {}'.format(imgpaths)
                imgpaths.sort()
            else:
                imgpaths = [imgpaths]
        if tile_size is None:
            img = common.imread(imgpaths[0], flag=cv2.IMREAD_UNCHANGED)
            imght, imgwd = img.shape[0], img.shape[1]
            tile_size = (imght, imgwd)
        bboxes = []
        for fname in imgpaths:
            bbox = MosaicLoader._filename_parser(fname, pattern, tile_size)
            if tile_offset is not None:
                dx = tile_offset[0] * tile_size[-1]
                dy = tile_offset[-1] * tile_size[0]
                bbox = (bbox[0]+dx, bbox[1]+dy, bbox[2]+dx, bbox[3]+dy)
            if pixel_offset is not None:
                dx, dy = pixel_offset
                bbox = (bbox[0]+dx, bbox[1]+dy, bbox[2]+dx, bbox[3]+dy)
            bboxes.append(bbox)
        return cls(filepaths=imgpaths, bboxes=bboxes, **kwargs)


    def _file_rtree_generator(self):
        """
        generator function to build rtree of image bounding boxes.
        """
        for fileid, bbox in enumerate(self._file_bboxes):
            yield (fileid, bbox, None)


    def crop(self, bbox, return_empty=False, **kwargs):
        hits = self._file_rtree.intersection(bbox, objects=False)
        initialized = False
        if hits:
            for fileid in hits:
                if not initialized:
                    out = super().crop(bbox, fileid, return_empty=return_empty, return_index=False, **kwargs)
                    if out is not None:
                        initialized = True
                else:
                    blk, indx = super().crop(bbox, fileid, return_empty=return_empty, return_index=True, **kwargs)
                    if blk is not None:
                        out[indx] = blk
        if not initialized:
            if return_empty:
                out = super().crop(bbox, 0, return_empty=True, return_index=False, **kwargs)
            else:
                out = None
        return out


    def _init_rtrees(self):
        self._file_rtree = index.Index(self._file_rtree_generator(), interleaved=True)
        self._cache_block_rtree = {}
        for bbox in self._file_bboxes:
            # normalize file bounding box to reduce number of trees needed
            imgwd = bbox[2] - bbox[0]
            imght = bbox[3] - bbox[1]
            box_normalized = (0, 0, imgwd, imght)
            if box_normalized not in self._cache_block_rtree:
                self._cache_block_rtree[box_normalized] = index.Index(
                    self._cached_block_rtree_generator(box_normalized), interleaved=True)


    @ staticmethod
    def _filename_parser(fname, pattern, tile_size):
        """
        given a filename and pattern, return the bboxes. pattern str follows
        regular expression, expect the following reserved keywords, which are
        used to indicate the nature of the field and the will be removed during
        parsing.
            {ROW_IND}: row_index
            {COL_IND}: col_index
            {X_MIN}: minimum of x coordinates
            {Y_MIN}: minimum of y coordinates
            {X_MAX}: maximum of x coordinates
            {Y_MAX}: maximum of y coordinates
        each keyword should only appear once and has one-to-one correspondance
        with a group in regexp.
        e.g. feabas_tr1_is_not_a_pokemon_tc2.png can be parsed with pattern:
            _tr({ROW_IND}\\d+)\\w+_tc({COL_IND}\\d+)
        """
        keywords = ['{ROW_IND}','{COL_IND}','{X_MIN}','{Y_MIN}','{X_MAX}','{Y_MAX}']
        pos = np.array([pattern.find(s) for s in keywords])
        kw_pos = sorted([(p, s) for p, s in zip(pos, keywords) if p >= 0])
        used_keywords = [s[1] for s in kw_pos]
        for kw in used_keywords:
            pattern = pattern.replace(kw, '')
        m = re.findall(pattern, fname)
        grps = {}
        for k, kw in enumerate(used_keywords):
            grps[kw] = int(m[0][k])
        if ('{X_MIN}' in grps) and ('{X_MAX}' in grps):
            xmin, xmax = grps['{X_MIN}'], grps['{X_MAX}']
        elif '{X_MIN}' in grps:
            xmin = grps['{X_MIN}']
            xmax = xmin + tile_size[-1]
        elif '{X_MAX}' in grps:
            xmax = grps['{X_MAX}']
            xmin = xmax - tile_size[-1]
        elif '{COL_IND}' in grps:
            xmin = grps['{COL_IND}'] * tile_size[-1]
            xmax = xmin + tile_size[-1]
        else:
            raise RuntimeError(f'x position of tile not defined in filename {fname}')
        if ('{Y_MIN}' in grps) and ('{Y_MAX}' in grps):
            ymin, ymax = grps['{Y_MIN}'], grps['{Y_MAX}']
        elif '{Y_MIN}' in grps:
            ymin = grps['{Y_MIN}']
            ymax = ymin + tile_size[0]
        elif '{Y_MAX}' in grps:
            ymax = grps['{Y_MAX}']
            ymin = ymax - tile_size[0]
        elif '{ROW_IND}' in grps:
            ymin = grps['{ROW_IND}'] * tile_size[0]
            ymax = ymin + tile_size[0]
        else:
            raise RuntimeError(f'y position of tile not defined in filename {fname}')
        return (xmin, ymin, xmax, ymax)


    @property
    def bounds(self):
        return self._file_rtree.bounds



class StreamLoader(AbstractImageLoader):
    """
    Loader class for images already in RAM. Should mimic the interface of that
    of MosaicLoader.
    """
    def __init__(self, img, **kwargs):
        self._img = img
        self._dtype = kwargs.get('dtype', img.dtype)
        self._apply_CLAHE = kwargs.get('apply_CLAHE', False)
        if len(img.shape) < 3:
            self._src_number_of_channels = 0
        else:
            self._src_number_of_channels = img.shape[-1]
        self._number_of_channels = kwargs.get('number_of_channels', max(1, self._src_number_of_channels))
        self._clahe_img = None
        self._preprocess = kwargs.get('preprocess', None)
        self._inverse = kwargs.get('inverse', False)
        self._default_fillval = kwargs.get('fillval', 0)
        self.resolution = kwargs.get('resolution', DEFAULT_RESOLUTION)
        self.update_preprocess_function(None)
        self.x0 = kwargs.get('x0', 0)
        self.y0 = kwargs.get('y0', 0)


    @classmethod
    def from_filepath(cls, imgpath, **kwargs):
        img = common.imread(imgpath, flag=cv2.IMREAD_UNCHANGED)
        return cls(img, **kwargs)


    @classmethod
    def from_init_dict(cls, init_dict):
        img = init_dict.pop('img')
        return cls(img, **init_dict)


    def crop(self, bbox, return_empty=False, **kwargs):
        fillval = kwargs.get('fillval', self._default_fillval)
        bbox_img = self.bounds
        img = self._read_image(**kwargs)
        return common.crop_image_from_bbox(img, bbox_img, bbox,
            return_empty=return_empty, return_index=False, fillval=fillval)


    @property
    def clahe_img(self):
        if self._clahe_img is None:
            clahe_filter = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            if len(self.img.shape) > 2:
                img = cv2.cvtColor(self.img, cv2.COLOR_RGB2Lab)
                img[:,:,0] = clahe_filter.apply(img[:,:,0])
                self._clahe_img = cv2.cvtColor(img, cv2.COLOR_Lab2RGB)
            else:
                self._clahe_img = clahe_filter.apply(self.img)
        return self._clahe_img


    def _read_image(self, **kwargs):
        number_of_channels = kwargs.get('number_of_channels', self._number_of_channels)
        if number_of_channels is None:
            number_of_channels = self._src_number_of_channels
        dtype = kwargs.get('dtype', self._dtype)
        apply_CLAHE = kwargs.get('apply_CLAHE', self._apply_CLAHE)
        inverse = kwargs.get('inverse', self._inverse)
        if apply_CLAHE:
            img = self.clahe_img
        else:
            img = self._img
        if (self._src_number_of_channels <= 1) and (number_of_channels > 1):
            img = np.tile(img.reshape(img.shape[0], img.shape[1], 1), (1,1,number_of_channels))
        elif (self._src_number_of_channels > 1) and (number_of_channels == 1):
            img = img.mean(axis=-1)
        if self._preprocess is not None:
            img = self._preprocess(img)
        if (dtype is not None) and (np.dtype(dtype) != img.dtype):
            img = img.astype(dtype)
        if inverse:
            img = common.inverse_image(img, dtype)
        return img


    def clear_cache(self, instant_gc=False):
        self._clahe_img = None
        if instant_gc:
            gc.collect()


    def init_dict(self, **kwargs):
        return super().init_dict(**kwargs)


    def save_to_json(self, jsonname, **kwargs):
        raise NotImplementedError


    def _export_dict(self):
        out = super()._export_dict(output_controls=True, cache_settings=False)
        out['img'] = self._img
        out['x0'] = self.x0
        out['y0'] = self.y0
        return out


    def file_bboxes(self, margin=0):
        bbox = [self.x0 - margin,
                self.y0 - margin,
                self.x0 + self._img.shape[1] + margin,
                self.y0 + self._img.shape[0] + margin]
        return [bbox]


    @property
    def bounds(self):
        return self.file_bboxes(margin=0)[0]


def get_tensorstore_spec(metafile, mip=None, **kwargs):
    downsample_method = kwargs.get('downsample_method', 'mean')
    return_mips = kwargs.get('return_mips', False)
    json_obj, _ = common.parse_json_file(metafile)
    try:
        mipmaps = {int(m): json_spec for m, json_spec in json_obj.items()}
    except ValueError:
        mipmaps = {0: json_obj}
    rendered_mips = np.array([m for m in mipmaps])
    if mip is None:
        mip = rendered_mips.max() + 1
    src_mip = rendered_mips[rendered_mips <= mip].max()
    src_spec = mipmaps[src_mip]
    if src_mip == mip:
        ds_spec = src_spec
    else:
        ts_src = ts.open(src_spec).result()
        src_spec = ts_src.spec(minimal_spec=True).to_json()
        downsample_factors = [2**(mip - src_mip), 2**(mip - src_mip)] + ([1] * (ts_src.rank - 2))
        ds_spec = {
            "driver": "downsample",
            "downsample_factors": downsample_factors,
            "downsample_method": downsample_method,
            "base": src_spec
        }
    if return_mips:
        return ds_spec, src_mip, mip, mipmaps
    else:
        return ds_spec


class TensorStoreLoader(AbstractImageLoader):
    """
    Loader class for image saved by tensorstore. Mirrors the APIs of MosaicLoader.
    """
    def __init__(self, dataset=None, **kwargs):
        super().__init__(**kwargs)
        if dataset is None:
            self._spec = kwargs.get('spec')
            self.reconnect()
        else:
            self.dataset = dataset
            self._spec = self.dataset.spec(minimal_spec=True).to_json()
        self.resolution = self.dataset.schema.dimension_units[0].multiplier
        self._z = kwargs.get('z', 0)


    @classmethod
    def from_json(cls, jsonname, **kwargs):
        settings, json_obj = cls._load_settings_from_json(jsonname)
        settings.update(kwargs)
        json_spec = json_obj['json_spec']
        return cls.from_json_spec(json_spec, **settings)


    @classmethod
    def from_json_spec(cls, js_spec, **kwargs):
        if kwargs.get('cache_capacity', None) is not None:
            total_bytes_limit = round(kwargs['cache_capacity'] * 1_000_000)
        elif kwargs.get('cache_size', None) is not None:
            # assume 4k tiles
            total_bytes_limit = round(kwargs['cache_size'] * 4096 * 4096)
        else:
            total_bytes_limit = -1
        if total_bytes_limit >= 0:
            cntx = {'cache_pool': {'total_bytes_limit': total_bytes_limit}}
            js_spec = js_spec.copy()
            crnt_lvl = js_spec
            while 'base' in crnt_lvl:
                crnt_lvl = crnt_lvl['base']
            crnt_lvl.update({'context': cntx, 'recheck_cached_data': 'open'})
        if js_spec['driver'] in ('neuroglancer_precomputed', 'n5'):
            kwargs['fillval'] = 0
        return cls(spec=js_spec)


    def _export_dict(self, **kwargs):
        out = super()._settings_dict(**kwargs)
        out['json_spec'] = self._spec
        return out


    def reconnect(self):
        for _ in range(TS_RETRY+1):
            try:
                self.dataset = ts.open(self._spec).result(timeout=TS_TIMEOUT)
                break
            except TimeoutError:
                pass
        else:
            raise TimeoutError


    def crop(self, bbox, return_empty=False, **kwargs):
        fillval = kwargs.get('fillval', self._default_fillval)
        apply_CLAHE = kwargs.get('apply_CLAHE', self._apply_CLAHE)
        dtype = kwargs.get('dtype', self.dtype)
        number_of_channels = kwargs.get('number_of_channels', self.number_of_channels)
        inverse = kwargs.get('inverse', self._inverse)
        rnk = self.dataset.rank
        for _ in range(TS_RETRY+1):
            try:
                if rnk > 2:
                    slc = self.dataset[:, :, self._z, ...]
                else:
                    slc = self.dataset
                while slc.rank > 2 and slc.shape[-1] == 1:
                    slc = slc[..., 0]
                bbox_img = self.bounds
                slc_crp, indx = common.crop_image_from_bbox(slc, bbox_img, bbox,
                                                            return_index=True, flip_indx=True)
                if (slc_crp is None) and (not return_empty):
                    return None
                img_crp = slc_crp.read().result(timeout=TS_TIMEOUT)
                break
            except TimeoutError:
                self.reconnect()
        else:
            raise TimeoutError
        if np.all(img_crp == fillval) and (not return_empty):
            return None
        if dtype is None:
            dtype = img_crp.dtype
        if number_of_channels is not None:
            if (number_of_channels > 1) and len(img_crp.shape) < 3:
                img_crp = np.tile(img_crp[..., np.newaxis], (1,1,number_of_channels))
            elif (number_of_channels == 1):
                while len(img_crp.shape) > 2:
                    img_crp = img_crp.mean(axis=-1).astype(img_crp.dtype)
        if apply_CLAHE:
            if (len(img_crp.shape) > 2) and (img_crp.shape[-1] == 3):
                img_crp = cv2.cvtColor(img_crp, cv2.COLOR_RGB2Lab)
                img_crp[:,:,0] = self._CLAHE.apply(img_crp[:,:,0])
                img_crp = cv2.cvtColor(img_crp, cv2.COLOR_Lab2RGB)
            else:
                img_crp = self._CLAHE.apply(img_crp)
        if self._preprocess is not None:
            img_crp = self._preprocess(img_crp)
        if inverse:
            img = common.inverse_image(img, dtype)
        outht = bbox[3] - bbox[1]
        outwd = bbox[2] - bbox[0]
        outsz = [outht, outwd] + list(img_crp.shape)[2:]
        imgout = np.full_like(img_crp, fillval, shape=outsz)
        imgout[indx] = img_crp
        return imgout.astype(dtype, copy=False)


    def get_chunk(self, bbox, return_empty=False, **kwargs):
        # bbox [xmin, ymin, zmin, xmax, ymax, zmax]
        fillval = kwargs.get('fillval', self._default_fillval)
        bbox = np.array(bbox)
        b_min, b_max = bbox[:3], bbox[3:]
        bounds = np.array(self.bounds_3d)
        D_min, D_max = bounds[:3], bounds[3:]
        for _ in range(TS_RETRY+1):
            try:
                if np.all(b_min >= D_min) and np.all(b_max <= D_max):
                    block = self.dataset[b_min[0]:b_max[0], b_min[1]:b_max[1], b_min[2]:b_max[2], ...].read().result(timeout=TS_TIMEOUT)
                else:
                    block = np.full([*(b_max - b_min).clip(0,None)] + [self.number_of_channels], fillval, dtype=self.dtype)
                    glb0 = np.maximum(b_min, D_min)
                    glb1 = np.minimum(b_max, D_max)
                    if np.all(glb1-glb0 > 0):
                        loc0 = glb0 - b_min
                        loc1 = glb1 - b_min
                        ptch = self.dataset[glb0[0]:glb1[0], glb0[1]:glb1[1], glb0[2]:glb1[2], ...].read().result(timeout=TS_TIMEOUT)
                        while len(ptch.shape) < len(block.shape):
                            ptch = ptch[..., np.newaxis]
                        block[loc0[0]:loc1[0], loc0[1]:loc1[1], loc0[2]:loc1[2], ...] = ptch
                break
            except TimeoutError:
                self.reconnect()
        else:
            raise TimeoutError
        if (not return_empty) and ((block.size == 0) or np.all(block == fillval)):
            return None
        while len(block.shape) > self.dataset.rank:
            block = block[...,0]
        while len(block.shape) < self.dataset.rank:
            block = block[..., np.newaxis]
        return block


    @property
    def dtype(self):
        if self._dtype is None:
            self._dtype = np.dtype(self.dataset.dtype.name)
        return self._dtype


    @property
    def number_of_channels(self):
        if self._number_of_channels is None:
            shp = self.dataset.shape
            if len(shp) < 4:
                self._number_of_channels = 1
            else:
                self._number_of_channels = shp[-1]
        return self._number_of_channels


    @property
    def pixel_size(self):
        res_x = self.dataset.dimension_units[0].multiplier
        res_y = self.dataset.dimension_units[1].multiplier
        res_z = self.dataset.dimension_units[2].multiplier
        return res_x, res_y, res_z


    @property
    def bounds(self):
        domain = self.dataset.domain
        inclusive_min = domain.inclusive_min
        exclusive_max = domain.exclusive_max
        xmin, ymin = inclusive_min[0], inclusive_min[1]
        xmax, ymax = exclusive_max[0], exclusive_max[1]
        return (xmin, ymin, xmax, ymax)


    @property
    def bounds_3d(self):
        domain = self.dataset.domain
        inclusive_min = domain.inclusive_min
        exclusive_max = domain.exclusive_max
        xmin, ymin, zmin = inclusive_min[:3]
        xmax, ymax, zmax = exclusive_max[:3]
        return (xmin, ymin, zmin, xmax, ymax, zmax)


    @property
    def spec(self):
        if (not hasattr(self, '_min_spec')) or (self._min_spec is None):
            self._min_spec = self.dataset.spec(minimal_spec=True).to_json()
            if self._min_spec['driver'] == 'neuroglancer_precomputed':
                self._min_spec.pop('scale_index', None)
                self._min_spec.setdefault('scale_metadata', {})
                self._min_spec['scale_metadata'].setdefault('resolution', list(self.pixel_size))
        return self._min_spec



class TensorStoreWriter(TensorStoreLoader):
    """
    Writer class for TensorStore Volume. Handles cropping/padding when the
    bounding box not aligned to the grid. Also handles timeout retry,
    """
    def write_single_chunk(self, bbox, img, **kwargs):
        ts_retry = kwargs.get('ts_retry', TS_RETRY)
        txn = kwargs.get('txn', None)
        updated = False
        for _ in range(ts_retry+1):
            try:
                if len(bbox) == 4:
                    if self.dataset.rank > 2:
                        dataset = self.dataset[:, :, self._z, ...]
                    else:
                        dataset = self.dataset
                    D_min, D_max = np.array(self.bounds[:2]), np.array(self.bounds[2:])
                    b_min, b_max = np.array(bbox[:2]), np.array(bbox[2:])
                    imgshp = img.shape[:2]
                elif len(bbox) == 6:
                    dataset = self.dataset
                    D_min, D_max = np.array(self.bounds_3d[:3]), np.array(self.bounds_3d[3:])
                    b_min, b_max = np.array(bbox[:3]), np.array(bbox[3:])
                    imgshp = img.shape[:3]
                    if ((b_max[-1] - b_min[-1]) == 1) and len(img.shape) < len(dataset.shape):
                        img = img[:,:,np.newaxis,...]
                        imgshp = img.shape[:3]
                else:
                    raise TypeError
                target_empty = False
                glb_indices = []
                loc_indices = []
                for d0, d1, b0, b1, ss in zip(D_min, D_max, b_min, b_max, imgshp):
                    glb0 = max(b0, d0)
                    glb1 = min(d1, b1, b0 + ss)
                    loc0 = glb0 - b0
                    loc1 = glb1 - b0
                    if loc0 >= loc1:
                        target_empty = True
                        break
                    glb_indices.append(slice(glb0, glb1, 1))
                    loc_indices.append(slice(loc0, loc1, 1))
                if not target_empty:
                    data_view = dataset[tuple(glb_indices)]
                    img_crp = img[tuple(loc_indices)]
                    while len(img_crp.shape) > len(data_view.shape):
                        img_crp = img_crp[..., 0]
                    while len(img_crp.shape) < len(data_view.shape):
                        img_crp = img_crp[..., np.newaxis]
                    if txn is None:
                        data_view.write(img_crp).result(timeout=TS_TIMEOUT)
                    else:
                        data_view.with_transaction(txn).write(img_crp).result(timeout=TS_TIMEOUT)
                    updated = True
                break
            except TimeoutError:
                self.reconnect()
        else:
            raise TimeoutError
        return updated


    def write_chunks_w_transaction(self, bboxes, imgs):
        if (len(bboxes) == 1) and (len(imgs) == 1):
            self.write_single_chunk(bboxes[0], imgs[0])
        else:
            for _ in range(TS_RETRY+1):
                try:
                    txn = ts.Transaction()
                    for bbox, img in zip(bboxes, imgs):
                        self.write_single_chunk(bbox, img, txn=txn, ts_retry=0)
                    txn.commit_async().result(timeout=TS_TIMEOUT)
                    if not txn.aborted:
                        break
                except TimeoutError:
                    self.reconnect()
            else:
                raise TimeoutError


    @property
    def write_grids(self):
        if (not hasattr(self, '_write_grids')) or (self._write_grids is None):
            shp_x, shp_y, shp_z = self.write_chunk_shape[:3]
            ori_x, ori_y, ori_z = self.dataset.schema.chunk_layout.grid_origin[:3]
            Dx0, Dy0, Dz0, Dx1, Dy1, Dz1 = self.bounds_3d
            start_x = int(np.floor((Dx0 - ori_x) / shp_x)) * shp_x + ori_x
            start_y = int(np.floor((Dy0 - ori_y) / shp_y)) * shp_y + ori_y
            start_z = int(np.floor((Dz0 - ori_z) / shp_z)) * shp_z + ori_z
            X0 = np.arange(start_x, Dx1, shp_x)
            Y0 = np.arange(start_y, Dy1, shp_y)
            Z0 = np.arange(start_z, Dz1, shp_z)
            X1, Y1, Z1 = X0 + shp_x, Y0 + shp_y, Z0 + shp_z
            X0, Y0, Z0 = X0.clip(Dx0, Dx1), Y0.clip(Dy0, Dy1), Z0.clip(Dz0, Dz1)
            X1, Y1, Z1 = X1.clip(Dx0, Dx1), Y1.clip(Dy0, Dy1), Z1.clip(Dz0, Dz1)
            self._write_grids = (X0, Y0, Z0, X1, Y1, Z1)
        return self._write_grids


    def grid_indices_to_bboxes(self, g_x, g_y, g_z=None):
        if g_z is None:
            X0, Y0, _, X1, Y1, _ = self.write_grids
            g_x, g_y = np.array(g_x).ravel(), np.array(g_y).ravel()
            Nb = max(g_x.size, g_y.size)
            if Nb > 1:
                if g_x.size == 1:
                    g_x = np.repeat(g_x, Nb)
                if g_y.size == 1:
                    g_y = np.repeat(g_y, Nb)
            return np.stack((X0[g_x], Y0[g_y], X1[g_x], Y1[g_y]), axis=-1)
        else:
            X0, Y0, Z0, X1, Y1, Z1 = self.write_grids
            g_x, g_y, g_z = np.array(g_x).ravel(), np.array(g_y).ravel(), np.array(g_z).ravel()
            Nb = max(g_x.size, g_y.size, g_z.size)
            if Nb > 1:
                if g_x.size == 1:
                    g_x = np.repeat(g_x, Nb)
                if g_y.size == 1:
                    g_y = np.repeat(g_y, Nb)
                if g_z.size == 1:
                    g_z = np.repeat(g_z, Nb)
            return np.stack((X0[g_x], Y0[g_y], Z0[g_z], X1[g_x], Y1[g_y], Z1[g_z]), axis=-1)


    def morton_xy_grid(self, indx=None):
        if (not hasattr(self, '_morton_xy')) or (self._morton_xy is None):
            N_x, N_y = self.grid_shape[:2]
            id_x, id_y = np.meshgrid(np.arange(N_x), np.arange(N_y))
            id_x, id_y = id_x.ravel(), id_y.ravel()
            zdr = common.z_order(np.stack((id_x, id_y),axis=-1))
            self._morton_xy = (id_x[zdr], id_y[zdr])
        if indx is None:
            return self._morton_xy
        else:
            id_x, id_y = self._morton_xy
            return id_x[indx], id_y[indx]


    @property
    def grid_shape(self):
        if (not hasattr(self, '_grid_shape')) or (self._grid_shape is None):
            X0, Y0, Z0, _, _, _ = self.write_grids
            self._grid_shape = (X0.size, Y0.size, Z0.size)
        return self._grid_shape


    @property
    def write_chunk_shape(self):
        return self.dataset.schema.chunk_layout.write_chunk.shape


    def sort_precomputed_scale(self):
        num_scales = 0
        url = self.dataset.kvstore.url
        if self.spec["driver"] == "neuroglancer_precomputed":
            info_file = join_paths(url, 'info')
            if file_exists(info_file):
                with File(info_file, 'r') as f:
                    spec = json.load(f)
                scales = spec.pop("scales")
                num_scales = len(scales)
                resolutions = np.zeros(num_scales)
                for m, scale in enumerate(scales):
                    resolutions[m] = scale["resolution"][0]
                idx = np.argsort(resolutions)
                new_scales = [scales[m] for m in idx]
                spec["scales"] = new_scales
                with File(info_file, 'w') as f:
                    json.dump(spec, f, indent=2)
        return num_scales



class MultiResolutionImageLoader:
    """
    A collection of image loaders with different resolution. Override __getitem__
    method to automatically select the right loader for the right resolution.
    """
    def __init__(self, loaders, overkill=True):
        self._loaders = loaders
        self._resolution = np.array([ldr.resolution for ldr in loaders])
        self._overkill = overkill # whether to prioritize higher resolution


    def __getitem__(self, resolution):
        dis = resolution - self._resolution
        if self._overkill:
            dis[dis<0] = np.abs(dis[dis<0]) + dis.max()
        else:
            dis = np.abs(dis)
        indx = np.argmin(dis)
        return self._loaders[indx]
