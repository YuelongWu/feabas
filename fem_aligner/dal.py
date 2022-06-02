from collections import OrderedDict, defaultdict
from functools import partial
import glob
import json
import os
import re

import cv2
import numpy as np
from rtree import index

from fem_aligner.miscs import CacheFIFO, crop_image_from_bbox


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
    if cache_border_margin == 0:
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


def _tile_divider_border_block(imght, imgwd, x0=0, y0=0, cache_block_size=0):
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

class DynamicImageLoader:
    """
    Class for image loader without predetermined image list.
    Mostly for caching & output formate control.

    Kwargs:
        dtype: datatype of the output images. Default to None same as input.
        number_of_channels: # of channels of the output images. Default to
            None same as input.
        apply_CLAHE(bool): whether to apply CLAHE on output images.
        inverse(bool): whether to invert images.
        fillval(scalar): fill value for missing data.
        cache_size(int): length of image cache (FIFO).
            self._cache: dict[(imgpath, block_id)] = blk
        cache_border_margin(int)/cache_block_size(int): the border width
            for _tile_divider_border caching, or the square block size for
            _tile_divider_border_block caching. If neither is set, cache
            the entile image with _tile_divider_blank.
    """
    def __init__(self, **kwargs):
        self.dtype = kwargs.get('dtype', None)
        self._number_of_channels = kwargs.get('number_of_channels', None)
        self._apply_CLAHE = kwargs.get('apply_CLAHE', False)
        self._CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self._inverse = kwargs.get('inverse', False)
        self._default_fillval = kwargs.get('fillval', 0)
        self._cache_size = kwargs.get('cache_size', 0)
        self._init_tile_divider(**kwargs)
        self._cache = CacheFIFO(maxlen=self._cache_size)
        self._file_bboxes = {}
        self._cached_block_rtree = {}


    def CLAHE_off(self):
        if self._apply_CLAHE:
            self.clear_cache()
            self._apply_CLAHE = False


    def CLAHE_on(self):
        if not self._apply_CLAHE:
            self.clear_cache()
            self._apply_CLAHE = True


    def clear_cache(self):
        self._cache.clear()


    def crop(self, bbox, imgpath, return_empty=False, **kwargs):
        fillval = kwargs.get('fillval', self._default_fillval)
        if (self._cache_size <= 0) or (imgpath not in self._file_bboxes):
            imgout = self._crop_without_cache(bbox, imgpath, return_empty=return_empty, **kwargs)
        else:
            # once chached before, try to read the cach first
            bbox_img = self._file_bboxes[imgpath]
            if bbox_img not in self._cached_block_rtree:
                    self._cached_block_rtree[bbox_img] = index.Index(
                        self._cached_block_rtree_generator(bbox_img), interleaved=True)
            hits = list(self.cached_block_rtree[bbox_img].intersection(bbox, objects=True))
            if not hits:
                if return_empty:
                    if self.dtype is None or self._number_of_channels is None:
                      # not sufficient info to generate empty tile, read an image to get info
                        img = self._read_image(imgpath, **kwargs)
                        imgout = crop_image_from_bbox(img, bbox_img, bbox, return_index=False,
                            return_empty=True, fillval=fillval)
                    else:
                        outht = bbox[3] - bbox[1]
                        outwd = bbox[2] - bbox[0]
                        if self._number_of_channels <= 1:
                            outsz = (outht, outwd)
                        else:
                            outsz = (outht, outwd, self._number_of_channels)
                        imgout = np.full(outsz, fillval, dtype=self.dtype)
                else:
                    imgout = None
            elif any((imgpath, item.object) not in self._cache for item in hits):
                # has missing blocks from cache, need to read image anyway
                imgout = self._crop_without_cache(bbox, imgpath, return_empty=return_empty, **kwargs)
            else:
                # read from cache
                initialized = False
                for item in hits:
                    blkid = item.object
                    blkbbox = [int(s) for s in item.bbox]
                    blk = self._get_block(imgpath, blkid, **kwargs)
                    if not initialized:
                        imgout = crop_image_from_bbox(blk, blkbbox, bbox,
                            return_index=False, return_empty=True, fillval=fillval)
                        initialized = True
                    else:
                        blkt, indx =  crop_image_from_bbox(blk, blkbbox, bbox,
                            return_index=True, return_empty=False, fillval=fillval)
                        if indx is not None and blkt is not None:
                            imgout[indx] = blkt
        return imgout


    def inverse_off(self):
        if self._inverse:
            self.clear_cache()
            self._inverse = False


    def inverse_on(self):
        if not self._inverse:
            self.clear_cache()
            self._inverse = True


    def _cache_block(self, imgpath, blkid, blk):
        if self._cache_size > 0 and blkid > 0:
            self._cache[(imgpath, blkid)] = blk


    def _cached_block_rtree_generator(self, bbox):
        x0 = bbox[0]
        y0 = bbox[1]
        imght = bbox[3] - bbox[1]
        imgwd = bbox[2] - bbox[0]
        for indx, divider in enumerate(self._tile_divider(imght, imgwd, x0=x0, y0=y0).items()):
            blkid, blkbbox = divider
            yield (indx, blkbbox, blkid)


    def _cache_image(self, imgpath, img=None, blkid=None, **kwargs):
        if img is None:
            img = self._read_image(imgpath, **kwargs)
        imght = img.shape[0]
        imgwd = img.shape[1]
        divider = self._tile_divider(imght, imgwd)
        if blkid is not None:
            divider.move_to_end(blkid, last=False)
        blkout = None
        for bid, blkbbox in divider.items():
            if self._cache_size > 0:
                if (imgpath, bid) not in self._cache:
                    blk = img[blkbbox[1]:blkbbox[3], blkbbox[0]:blkbbox[2],...]
                    self._cache_block(imgpath, bid, blk)
                if bid == blkid:
                    blkout = self._cache[(imgpath, bid)]
            else:
                if bid == blkid:
                    blkout = img[blkbbox[1]:blkbbox[3], blkbbox[0]:blkbbox[2],...]
        return blkout


    def _crop_without_cache(self,bbox, imgpath, return_empty=False, **kwargs):
        # directly crop the image without checking the cache first
        fillval = kwargs.get('fillval', self._default_fillval)
        img = self._read_image(imgpath, **kwargs)
        imght, imgwd = img.shape[0], img.shape[1]
        bbox_img = (0, 0, imgwd, imght)
        imgout = crop_image_from_bbox(img, bbox_img, bbox, return_index=False,
            return_empty=return_empty, fillval=fillval)
        if self._cache_size > 0:
            self._file_bboxes[imgpath] = bbox_img
            if bbox_img not in self._cached_block_rtree:
                self._cached_block_rtree[bbox_img] = index.Index(
                    self._cached_block_rtree_generator(bbox_img), interleaved=True)
            self._cache_image(imgpath, img=img, **kwargs)
        return imgout



    def _get_block(self, imgpath, blkid, **kwargs):
        if (self._cache_size > 0) and (imgpath, blkid) in self._cache:
            blkout = self._cache[(imgpath, blkid)]
        else:
            blkout = self._cache_image(imgpath, blkid=blkid, **kwargs)
        return blkout


    def _init_tile_divider(self, **kwargs):
        if self._cache_size <= 0:
            self._tile_divider = _tile_divider_blank
        elif 'cache_border_margin' in kwargs:
            self._tile_divider = partial(_tile_divider_border, cache_border_margin=kwargs['cache_border_margin'])
        elif 'cache_block_size' in kwargs:
            self._tile_divider = partial(_tile_divider_border_block, cache_block_size=kwargs['cache_block_size'])
        else:
            self._tile_divider = _tile_divider_blank


    def _read_image(self, imgpath, **kwargs):
        number_of_channels = kwargs.get('number_of_channels', self._number_of_channels)
        dtype = kwargs.get('dtype', self.dtype)
        apply_CLAHE = kwargs.get('apply_CLAHE', self._apply_CLAHE)
        inverse = kwargs.get('inverse', self._inverse)
        if number_of_channels == 3:
            img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(imgpath, cv2.IMREAD_UNCHANGED)
        if img is None:
            invalid_image_file_error = 'Image file {} not valid!'.format(imgpath)
            raise RuntimeError(invalid_image_file_error)
        if (number_of_channels == 1) and (len(img.shape) > 2) and (img.shape[-1] > 1):
            img = img.mean(axis=-1)
        if dtype is not None:
            img = img.astype(dtype, copy=False)
        if apply_CLAHE:
            img = self._CLAHE.apply(img)
        if inverse:
            if np.dtype(dtype) == np.dtype('uint8'):
                img = 255 - img
            elif np.dtype(dtype) == np.dtype('uint16'):
                img = 65535 - img
            else:
                img = img.max() - img
        return img



class StaticImageLoader(DynamicImageLoader):
    """
    Class for image loader with predetermined image list.
    Assuming all the images are of the same dimensions(default), dtype and bitdepth.

    Args:
        filepaths(list): fullpaths of the image file list
    Kwargs:
        identical_tiles(bool): if all the source images have identical dimensions,
            default True.
        tile_size(tuple): the size of tiles if identical_tiles is True.
            If None(default), infer from the first image encountered.
    """
    def __init__(self, filepaths, **kwargs):
        self.imgrootdir = os.path.dirname(os.path.commonprefix(filepaths))
        self.imgrelpaths = [os.path.relpath(s, self.imgrootdir) for s in filepaths]
        self.check_filename_uniqueness()
        super().__init__(**kwargs)
        self._identical_tiles = kwargs.get('identical_tiles', True)
        self._tile_size = kwargs.get('tile_size', None)


    def check_filename_uniqueness(self):
        assert len(set(self.imgrelpaths)) == len(self.imgrelpaths), 'duplicated filenames'


    def crop(self, bbox, fileid, return_empty=False, **kwargs):
        if isinstance(fileid, str):
            imgpath = fileid
        else:
            imgpath = os.path.join(self.imgrootdir, self.imgrelpaths[fileid])
        return super().crop(bbox, imgpath, return_empty=return_empty, **kwargs)


    def _cache_image(self, fileid, img=None, blkid=None, **kwargs):
        if isinstance(fileid, str):
            imgpath = fileid
        else:
            imgpath = os.path.join(self.imgrootdir, self.imgrelpaths[fileid])
        return super()._cache_image(imgpath, img, blkid, **kwargs)


    def _get_block(self, fileid, blkid, **kwargs):
        if isinstance(fileid, str):
            imgpath = fileid
        else:
            imgpath = os.path.join(self.imgrootdir, self.imgrelpaths[fileid])
        return super()._get_block(imgpath, blkid, **kwargs)


    def _read_image(self, fileid, **kwargs):
        imgpath = os.path.join(self.imgrootdir, self.imgrelpaths[fileid])
        img = super()._read_image(imgpath, **kwargs)
        if self._identical_tiles:
            if self._tile_size is None:
                self._tile_size = img.shape[:2]
                tile_ht, tile_wd = self._tile_size
                bbox_img = (0, 0, tile_wd, tile_ht)
                self._cached_block_rtree[bbox_img] = index.Index(
                    self._cached_block_rtree_generator(bbox_img), interleaved=True)
        if self.dtype is None:
            self.dtype = img.dtype
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


    @property
    def filepaths_generator(self):
        for fname in self.imgrelpaths:
            yield os.path.join(self.imgrootdir, fname)


    @property
    def filepaths(self):
        return list(self.filepaths_generator)



class MosaicLoader(StaticImageLoader):
    """
    Image loader to render cropped images from non-overlapping tiles
        bbox :int: [xmin, ymin, xmax, ymax]
    """
    def __init__(self, filepaths=[], bboxes=[], **kwargs):
        assert len(filepaths) == len(bboxes)
        super().__init__(filepaths, **kwargs)
        self._file_bboxes = bboxes
        self._cached_block_rtree = None


    @classmethod
    def from_filepaths(cls, imgpaths, pattern='_tr(\d+)-tc(\d+)', rc_offset=[1, 1], rc_order=True, **kwargs):
        """
        pattern:    regexp pattern to parse row-column pattern
        rc_offset:  if None, normalize
        rc_order:   True:row-colume or False:colume-row
        """
        tile_size = kwargs.get('tile_size', None)   # if None, read in first image
        if isinstance(imgpaths, str):
            if '*' in imgpaths:
                imgpaths = glob.glob(imgpaths)
                assert bool(imgpaths), 'No image found: {}'.format(imgpaths)
                imgpaths.sort()
            else:
                imgpaths = [imgpaths]
        if len(imgpaths) == 1:
            r_c = np.array([(0, 0)])
        else:
            r_c = np.array([MosaicLoader._filename_parser(s, pattern, rc_order=rc_order) for s in imgpaths])
            if rc_offset is None:
                r_c -= r_c.min(axis=0)
            else:
                r_c -= np.array(rc_offset).reshape(1,2)
        if tile_size is None:
            img = cv2.imread(imgpaths[0], cv2.IMREAD_UNCHANGED)
            imght, imgwd = img.shape[0], img.shape[1]
            tile_size = (imgwd, imght)
        bboxes = []
        for rc in r_c:
            r = rc[0]
            c = rc[1]
            bbox = [c*tile_size[0], r*tile_size[-1], (c+1)*tile_size[0], (r+1)*tile_size[-1]]
            bboxes.append(bbox)
        return cls(filepaths=imgpaths, bboxes=bboxes, **kwargs)


    @ staticmethod
    def _filename_parser(fname, pattern, rc_order=True):
        m = re.findall(pattern, fname)
        if rc_order:
            r = int(m[0][0])
            c = int(m[0][1])
        else:
            c = int(m[0][0])
            r = int(m[0][1])
        return r, c


    @classmethod
    def from_json(cls, jsonname, **kwargs):
        with open(jsonname, 'r') as f:
            json_obj = json.load(f)
        kwargs_new = {}
        if 'dtype' in json_obj:
            kwargs_new['dtype'] = np.dtype(json_obj['dtype'])
        if 'number_of_channels' in json_obj:
            kwargs_new['number_of_channels'] = int(json_obj['number_of_channels'])
        if 'fillval' in json_obj:
            kwargs_new['fillval'] = json_obj['fillval']
        if 'apply_CLAHE' in json_obj:
            kwargs_new['apply_CLAHE'] = json_obj['apply_CLAHE']
        if 'inverse' in json_obj:
            kwargs_new['inverse'] = json_obj['inverse']
        if 'cache_size' in json_obj:
            kwargs_new['cache_size'] = json_obj['cache_size']
        if 'cache_border_margin' in json_obj:
            kwargs_new['cache_border_margin'] = json_obj['cache_border_margin']
        kwargs_new.update(kwargs)
        filepaths = []
        bboxes = []
        for f in json_obj['images']:
            filepaths.append(f['filepath'])
            bboxes.append(f['bbox'])
        return cls(filepaths=filepaths, bboxes=bboxes, **kwargs_new)


    def save_to_json(self, jsonname, output_control=True, cache_status=True):
        out = {}
        if output_control:
            out['dtype'] = np.dtype(self.dtype).str
            out['number_of_channels'] = self._number_of_channels
            out['fillval'] = self._default_fillval
            out['apply_CLAHE'] = self._apply_CLAHE
            out['inverse'] = self._inverse
        if cache_status:
            out['cache_size'] = self._cache_size
            out['cache_border_margin'] = self._cache_border_margin
        images = []
        for fname, bbox in zip(self.filepaths_generator, self._file_bboxes):
            images.append({'filepath': fname, 'bbox': [int(s) for s in bbox]})
        out['images'] = images
        with open(jsonname, 'w') as f:
            json.dump(out, f, indent=2)


    def _cache_block(self, fileid, blkid, blk):
        if blkid > 0:  # only cache border blocks
            super()._cache_block(fileid, blkid, blk)


    def cached_block_rtree(self, to_cache=True):
        if self._cached_block_rtree is not None:
            return self._cached_block_rtree
        cached_block_rtree = index.Index(self._cached_block_rtree_generator(), interleaved=True)
        if to_cache:
            self._cached_block_rtree = cached_block_rtree
        return cached_block_rtree


    def _cached_block_rtree_generator(self):
        indx = -1
        for fileid, bbox in enumerate(self._file_bboxes):
            x0 = bbox[0]
            y0 = bbox[1]
            imght = bbox[3] - bbox[1]
            imgwd = bbox[2] - bbox[0]
            for blkid, blkbbox in self._tile_divider(imght, imgwd, x0=x0, y0=y0).items():
                indx += 1
                yield (indx, blkbbox, (fileid, blkid))


    def crop(self, bbox, return_empty=False, **kwargs):
        fillval = kwargs.get('fillval', self._default_fillval)
        dtype = kwargs.get('dtype', self.dtype)
        imght = bbox[3] - bbox[1]
        imgwd = bbox[2] - bbox[0]
        hits = list(self.cached_block_rtree().intersection(bbox, objects=True))
        if not hits:
            if return_empty:
                if self._number_of_channels > 1:
                    return fillval * np.ones((imght, imgwd, self._number_of_channels), dtype=dtype)
                else:
                    return fillval * np.ones((imght, imgwd), dtype=dtype)
            else:
                return None
        hits = sorted(hits, key=lambda obj: obj.id)
        if self._number_of_channels > 1:
            out = fillval * np.ones((imght, imgwd, self._number_of_channels), dtype=dtype)
        else:
            out = fillval * np.ones((imght, imgwd), dtype=dtype)
        for item in hits:
            fileid, blkid = item.object
            blkbbox = [int(s) for s in item.bbox]
            blk = self._get_block(fileid, blkid)
            if blk.size == 0:
                continue
            cropped_blk, indx = self._crop_block_roi(blk, blkbbox, bbox)
            if cropped_blk is None or indx is None:
                continue
            out[indx] = cropped_blk
        return out


    def file_bboxes(self, margin=0):
        for bbox in self._file_bboxes:
            bbox_m = [bbox[0]-margin, bbox[1]-margin, bbox[2]+margin, bbox[3]+margin]
            yield bbox_m



    @staticmethod
    def _crop_block_roi(blk, blkbbox, roibbox):
        x0 = blkbbox[0]
        y0 = blkbbox[1]
        blkht = min(blkbbox[3]-blkbbox[1], blk.shape[0])
        blkwd = min(blkbbox[2]-blkbbox[0], blk.shape[1])
        xmin = max(x0, roibbox[0])
        xmax = min(x0 + blkwd, roibbox[2])
        ymin = max(y0, roibbox[1])
        ymax = min(y0 + blkht, roibbox[3])
        if xmin >= xmax or ymin >= ymax:
            return None, None
        cropped = blk[(ymin-y0):(ymax-y0), (xmin-x0):(xmax-x0), ...]
        dimpad = len(blk.shape) - 2
        indx = tuple([slice(ymin-roibbox[1], ymax-roibbox[1]), slice(xmin-roibbox[0],xmax-roibbox[0])] +
            [slice(0, None)] * dimpad)
        return cropped, indx


    @property
    def bounds(self):
        return self.cached_block_rtree().bounds
