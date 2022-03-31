from abc import ABC
from collections import OrderedDict
from functools import partial
import glob
import json
import os
import re

import cv2
import numpy as np
from rtree import index

from fem_aligner.miscs import DequeDict


# bbox :int: [xmin, ymin, xmax, ymax]

def _tile_divider_blank(imght, imgwd, x0=0, y0=0):
    divider = OrderedDict()
    divider[1] = (x0, y0, x0+imgwd, y0+imght)
    return divider


def _tile_divider_border(imght, imgwd, x0=0, y0=0, cache_border_margin=0):
    # devide image borders to cache separately
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
    # devide image to blocks to cache separately
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
        key = 1
        for xmin, ymin, xmax, ymax in zip(xe0.ravel(), ye0.ravel(), xe1.ravel(), ye1.ravel()):
            divider[key] = (xmin, ymin, xmax, ymax)
            key += 1
    return divider



class AbstractImageLoader(ABC):
    """
    Abstract class for image loader
    """
    def __init__(self, filepaths, **kwargs):
        self.imgrootdir = os.path.dirname(os.path.commonprefix(filepaths))
        self.imgrelpaths = [os.path.relpath(s, self.imgrootdir) for s in filepaths]
        self.check_filename_uniqueness()
      # output control
        if not kwargs.get('load_img', False):
            self.dtype = kwargs.get('dtype', np.uint8)
            self._number_of_channels = kwargs.get('number_of_channels', 1)
        else:
            # if load_img, read in the first image to complete missing meta info
            number_of_channels = kwargs.get('number_of_channels', None)
            dtype = kwargs.get('dtype', None)
            if (dtype is None) or (number_of_channels is None):
                img = cv2.imread(filepaths[0], cv2.IMREAD_UNCHANGED)
                assert img is not None
                if number_of_channels is None:
                    if len(img.shape) < 3:
                        self._number_of_channels = 1
                    else:
                        self._number_of_channels = img.shape[-1]
                else:
                    self._number_of_channels = number_of_channels
                if dtype is None:
                    self.dtype = img.dtype
                else:
                    self.dtype = dtype
        self._apply_clahe = kwargs.get('apply_clahe', False)
        if self._apply_clahe:
            self.clahe_on()
        self._inverse = kwargs.get('inverse', False)
        self._default_fillval = kwargs.get('fillval', 0)
      # caching
        self._cache_size = kwargs.get('cache_size', 0)
        self._init_tile_divider(**kwargs)
        self._cache = DequeDict(maxlen=self._cache_size)


    def _cache_block(self, fileid, blkid, blk):
        self._cache[(fileid, blkid)] = blk


    def check_filename_uniqueness(self):
        assert len(set(self.imgrelpaths)) == len(self.imgrelpaths), 'duplicated filenames'


    def clahe_off(self):
        if self._apply_clahe:
            self.clear_cache()
            self._apply_clahe = False


    def clahe_on(self):
        if not self._apply_clahe:
            self.clear_cache()
            self._apply_clahe = True
            self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))


    def clear_cache(self):
        self._cache.clear()


    def _get_block(self, fileid, blkid):
        if (self._cache_size > 0) and (fileid, blkid) in self._cache:
            return self._cache[(fileid, blkid)]
        img = self._read_image(fileid)
        imght = img.shape[0]
        imgwd = img.shape[1]
        divider = self._tile_divider(imght, imgwd)
        divider.move_to_end(blkid, last=False) # FIFO queue, discard queried first 
        if self._cache_size > 0:
            for bid, blkbbox in divider.items():
                if (fileid, bid) not in self._cache:
                    blk = img[blkbbox[1]:blkbbox[3], blkbbox[0]:blkbbox[2]]
                    self._cache_block(fileid, bid, blk)
                if bid == blkid:
                    blkout = blk
            return blkout
        else:
            blkbbox = divider[blkid]
            return img[blkbbox[1]:blkbbox[3], blkbbox[0]:blkbbox[2]]


    def _init_tile_divider(self, **kwargs):
        if 'cache_border_margin' in kwargs:
            self._tile_divider = partial(_tile_divider_border, cache_border_margin=kwargs['cache_border_margin'])
        elif 'cache_block_size' in kwargs:
            self._tile_divider = partial(_tile_divider_border_block, cache_block_size=kwargs['cache_block_size'])
        else:
            self._tile_divider = _tile_divider_blank


    def inverse_off(self):
        if self._inverse:
            self.clear_cache()
            self._inverse = False


    def inverse_on(self):
        if not self._inverse:
            self.clear_cache()
            self._inverse = True


    def _read_image(self, fileid):
        imgpath = os.path.join(self.imgrootdir, self.imgrelpaths[fileid])
        if self._number_of_channels == 3:
            img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(imgpath, cv2.IMREAD_UNCHANGED)
        if (self._number_of_channels == 1) and (len(img.shape) > 2) and (img.shape[-1] > 1):
            img = img.mean(axis=-1)
        if self._apply_clahe:
            img = self._clahe.apply(img)
        if self._inverse:
            if self.dtype == np.dtype('uint8'):
                img = 255 - img
            elif self.dtype == np.dtype('uint16'):
                img = 65535 - img
            else:
                img = img.max() - img
        return img


    @property
    def fileid_LUT(self):
        if (not hasattr(self, '_fileid_LUT')) or (self._fileid_LUT is None):
            # truncate filenames from right to left until unique
            self._fileid_LUT = {}
            inverse_fname = [os.path.splitext(s)[0][::-1] for s in self.imgrelpaths]



    @property
    def filepaths_generator(self):
        for fname in self.imgrelpaths:
            yield os.path.join(self.imgrootdir, fname)


    @property
    def filepaths(self):
        return list(self.filepaths_generator)


    @property
    def margin(self):
        return self._cache_border_margin



class ImageLoader(AbstractImageLoader):
    """
    Image loader to crop partial image from a collection of images
        bbox :int: [xmin, ymin, xmax, ymax]
    """
    def __init__(self, filepaths=[], **kwargs):
        super().__init__(filepaths, **kwargs)


    @classmethod
    def from_filepaths(cls, foldername, **kwargs):
        if '*' in foldername:
            imgpaths = glob.glob(foldername)
            assert bool(imgpaths), 'No image found: {}'.format(imgpaths)
            imgpaths.sort()
        else:
            imgpaths = [foldername]
        return cls(filepaths=imgpaths, **kwargs)


class MontageLoader(AbstractImageLoader):
    """
    Image loader to render cropped images from non-overlapping montage tiles
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
            r_c = np.array([MontageLoader._filename_parser(s, pattern, rc_order=rc_order) for s in imgpaths])
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
        if 'apply_clahe' in json_obj:
            kwargs_new['apply_clahe'] = json_obj['apply_clahe']
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
            out['apply_clahe'] = self._apply_clahe
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
