"""
Background shading (flat-field / dark-field) correction of the raw image tiles,
following BaSiC (Peng et al., Nat. Commun. 2017; BaSiCPy).

Each tile is modelled as
    I_i(x) = b_i * F(x) + D(x) + S_i(x)
with F the flat-field (mean 1), D the dark-field, b_i the baseline of the tile
and S_i a sparse residue (the actual image contents). F and D are estimated
from a stack of downsampled tiles, and the correction applied to a tile is
    I_corrected = (I - D) / F + mean(D)
(mean(D) added back to keep the overall grayscale level).

The profiles of a section are estimated during the stitching matching step
and saved to an h5 file (stitch/shading/<section>.h5). For MultiSEM data, the
tiles taken by the same beam share one profile; otherwise one profile is shared
by all the tiles. The image loaders (feabas.dal) apply the correction right
after reading the raw tiles, before the per-tile brightness/contrast transfer
functions.
"""
from collections import OrderedDict
from functools import lru_cache
import os

import cv2
import numpy as np
from scipy import fft

from feabas import common, logging, storage


SHADING_FILE_VERSION = 2
POOLED_GROUP_KEY = -1   # group key of the profile estimated from all the sampled tiles
SMALL_CACHE_SIZE = 512  # number of blended (working size) profiles cached by ShadingProfiles


##----------------------------- BaSiC estimation -----------------------------##

def _dct2(x):
    return fft.dctn(x, norm='ortho')


def _idct2(x):
    return fft.idctn(x, norm='ortho')


def _shrink(x, thresh):
    """soft thresholding"""
    return x - np.clip(x, -thresh, thresh)


def _inexact_alm_rspca_l1(D, shape, lambda_flat, lambda_dark, weight, get_darkfield=True,
                          max_iterations=500, tol=1e-6):
    """
    inexact augmented Lagrange multiplier solver of the BaSiC decomposition
    (ent1 = 1 so that the update of the sparse residue simplifies).
    Args:
        D (m x n ndarray): the stack of n images with m pixels each.
        shape (tuple): (height, width) of the images, height * width = m.
        lambda_flat, lambda_dark (float): sparsity in the DCT domain (i.e.
            smoothness) of the flat-field and the dark-field.
        weight (m x n ndarray): weights of the sparse residue.
    Return:
        A (m x n): the low-rank part (b_i * F + D).
        E (m x n): the sparse residue.
        offset (m): the dark-field.
    """
    dtype = D.dtype
    m, n = D.shape
    ent2 = 10.0
    norm_two = float(np.sqrt(np.max(np.linalg.eigvalsh((D.T @ D).astype(np.float64))).clip(1e-12, None)))
    mu = 12.5 / norm_two
    mu_bar = mu * 1e7
    rho = 1.5
    d_norm = max(float(np.linalg.norm(D)), 1e-12)
    D_rowmean = D.mean(axis=1)
    neg_weight = -weight
    A1 = np.empty_like(D)   # low-rank part
    E1 = np.zeros_like(D)   # sparse part
    Y1 = np.zeros_like(D)   # Lagrange multipliers
    R1 = np.empty_like(D)
    buf = np.empty_like(D)
    W_hat = np.zeros(shape, dtype=np.float64)
    W_idct = np.zeros(m, dtype=np.float64)
    A1_coeff = np.ones(n, dtype=np.float64)
    A_offset = np.zeros(m, dtype=np.float64)
    B1_uplimit = float(D.min())
    B1_offset = 0.0
    for _ in range(max_iterations):
        W_idct = _idct2(W_hat).ravel()
        # row mean of (D - A1 - E1 + Y1/mu), with A1 = W_idct x A1_coeff + A_offset
        temp_W = D_rowmean - (W_idct * A1_coeff.mean() + A_offset) - E1.mean(axis=1) + Y1.mean(axis=1) / mu
        W_hat = _shrink(W_hat + _dct2(temp_W.reshape(shape)), lambda_flat / mu)
        W_idct = _idct2(W_hat).ravel()
        np.multiply.outer(W_idct.astype(dtype), A1_coeff.astype(dtype), out=A1)
        A1 += A_offset.astype(dtype)[:, None]
        # E1 = shrink(D - A1 + Y1/mu, weight/mu)
        np.subtract(D, A1, out=E1)
        np.multiply(Y1, 1.0 / mu, out=buf)
        E1 += buf
        np.multiply(E1, mu, out=buf)
        np.clip(buf, neg_weight, weight, out=buf)
        buf *= 1.0 / mu
        E1 -= buf
        np.subtract(D, E1, out=R1)
        R1_colmean = R1.mean(axis=0, dtype=np.float64)
        R1_mean = R1_colmean.mean()
        if R1_mean == 0:
            break
        A1_coeff = (R1_colmean / R1_mean).clip(0, None)
        if get_darkfield:
            valid = A1_coeff < 1
            if np.any(valid):
                W_mean = W_idct.mean()
                hi = (W_idct > (W_mean - 1e-6)).astype(dtype)
                lo = (W_idct < (W_mean + 1e-6)).astype(dtype)
                hi_mean = (hi @ R1).astype(np.float64) / max(hi.sum(), 1)
                lo_mean = (lo @ R1).astype(np.float64) / max(lo.sum(), 1)
                B1_coeff = (hi_mean[valid] - lo_mean[valid]) / R1_mean
                k = np.sum(valid)
                a = A1_coeff[valid]
                temp1 = np.sum(a ** 2)
                temp2 = np.sum(a)
                temp3 = np.sum(B1_coeff)
                temp4 = np.sum(a * B1_coeff)
                temp5 = temp2 * temp3 - k * temp4
                if temp5 == 0:
                    B1_offset = 0.0
                else:
                    B1_offset = (temp1 * temp3 - temp2 * temp4) / temp5
                B1_offset = max(B1_offset, 0.0)
                if W_mean > 0:
                    B1_offset = min(B1_offset, B1_uplimit / W_mean)
                B_offset = B1_offset * W_mean - B1_offset * W_idct
                R1_validmean = (R1 @ valid.astype(dtype)).astype(np.float64) / k
                A1_offset = R1_validmean - a.mean() * W_idct
                A1_offset = A1_offset - A1_offset.mean()
                A_offset = A1_offset - A1_offset.mean() - B_offset
                # smooth A_offset
                W_offset = _shrink(_dct2(A_offset.reshape(shape)), lambda_dark / (ent2 * mu))
                A_offset = _idct2(W_offset).ravel()
                # encourage sparse A_offset
                A_offset = _shrink(A_offset, lambda_dark / (ent2 * mu))
                A_offset = A_offset + B_offset
        # Z1 = D - A1 - E1 = R1 - A1
        np.subtract(R1, A1, out=buf)
        stop_criterion = float(np.linalg.norm(buf)) / d_norm
        buf *= mu
        Y1 += buf
        mu = min(mu * rho, mu_bar)
        if stop_criterion < tol:
            break
    A_offset = A_offset + B1_offset * W_idct
    return A1, E1, A_offset


def basic_fit(stack, **kwargs):
    """
    estimate the flat-field and dark-field from a stack of images with BaSiC.
    Args:
        stack (N x H x W ndarray): the images, usually downsampled to ~128px.
    Kwargs:
        get_darkfield (bool): whether to estimate the dark-field.
        smoothness_flatfield, smoothness_darkfield (float): the regularization
            weights of the flat-field / dark-field. None to decide from the
            data.
        sort_intensity (bool): sort the intensities of each pixel along the
            image axis before fitting (as the original BaSiC implementation).
        max_iterations (int): maximum iterations of the inner solver.
        max_reweight_iterations (int): maximum iterations of reweighting.
        optimization_tol (float): convergence tolerance of the inner solver.
        reweight_tol (float): convergence tolerance of reweighting.
    Return:
        flatfield (H x W float32): with mean 1.
        darkfield (H x W float32): in the same unit as the input images.
    """
    get_darkfield = kwargs.get('get_darkfield', True)
    lambda_flat = kwargs.get('smoothness_flatfield', None)
    lambda_dark = kwargs.get('smoothness_darkfield', None)
    sort_intensity = kwargs.get('sort_intensity', True)
    max_iterations = kwargs.get('max_iterations', 500)
    max_reweight_iterations = kwargs.get('max_reweight_iterations', 10)
    optimization_tol = kwargs.get('optimization_tol', 1e-6)
    reweight_tol = kwargs.get('reweight_tol', 1e-3)
    epsilon = 0.1
    stack = np.asarray(stack, dtype=np.float32)
    if stack.ndim != 3:
        raise ValueError('stack should be N x H x W')
    N, H, W = stack.shape
    scale = float(stack.mean(dtype=np.float64))
    if scale <= 0:
        return np.ones((H, W), dtype=np.float32), np.zeros((H, W), dtype=np.float32)
    stack = stack / scale
    mean_img = stack.mean(axis=0, dtype=np.float64)
    mean_img = mean_img / mean_img.mean()
    dct_sum = np.sum(np.abs(_dct2(mean_img)))
    if lambda_flat is None:
        lambda_flat = dct_sum / 400 * 0.5
    if lambda_dark is None:
        lambda_dark = dct_sum / 400 * 0.2
    if sort_intensity:
        stack = np.sort(stack, axis=0)
    D = np.ascontiguousarray(stack.reshape(N, H * W).T)   # m x n
    del stack
    weight = np.ones_like(D)
    flatfield_last = np.ones(H * W)
    darkfield_last = None
    flatfield = flatfield_last
    darkfield = np.zeros(H * W)
    for _ in range(max(1, max_reweight_iterations)):
        XA, XE, Xoffset = _inexact_alm_rspca_l1(D, (H, W), lambda_flat, lambda_dark, weight,
                                                get_darkfield=get_darkfield,
                                                max_iterations=max_iterations,
                                                tol=optimization_tol)
        XA_mean = XA.mean(axis=1, dtype=np.float64)
        # weight = 1 / (|XE / mean(XA)| + epsilon), normalized to mean 1
        weight = XE * (1.0 / (np.abs(XA_mean) + 1e-6)).astype(np.float32)[:, None]
        np.abs(weight, out=weight)
        weight += epsilon
        np.reciprocal(weight, out=weight)
        weight *= weight.size / np.sum(weight, dtype=np.float64)
        del XA, XE
        temp = XA_mean - Xoffset
        temp_mean = temp.mean()
        if temp_mean <= 0:
            break
        flatfield = temp / temp_mean
        darkfield = Xoffset if get_darkfield else np.zeros(H * W)
        mad_flat = np.sum(np.abs(flatfield - flatfield_last)) / max(np.sum(np.abs(flatfield_last)), 1e-12)
        if darkfield_last is None:
            mad_dark = 0 if not get_darkfield else np.inf
        else:
            diff_dark = np.sum(np.abs(darkfield - darkfield_last))
            if diff_dark < 1e-7:
                mad_dark = 0
            else:
                mad_dark = diff_dark / max(np.sum(np.abs(darkfield_last)), 1e-6)
        flatfield_last = flatfield
        darkfield_last = darkfield
        if max(mad_flat, mad_dark) <= reweight_tol:
            break
    flatfield = flatfield.reshape(H, W)
    if (not np.all(np.isfinite(flatfield))) or np.any(flatfield <= 0):
        flatfield = np.where(np.isfinite(flatfield), flatfield, 1).clip(1e-3, None)
        flatfield = flatfield / flatfield.mean()
    darkfield = np.nan_to_num(darkfield.reshape(H, W)) * scale
    return flatfield.astype(np.float32), darkfield.astype(np.float32)


##---------------------------- section estimation ----------------------------##

def tile_groups(imgrelpaths, grouping='none'):
    """
    group id of each tile sharing one shading profile.
    grouping:
        'beam': MultiSEM beam number from the file names.
        'none': all the tiles in one group (0).
    """
    grouping = str(grouping).lower()
    if grouping == 'beam':
        from feabas.multisem import mfovids_beamids_from_filenames
        _, beams = mfovids_beamids_from_filenames(imgrelpaths)
        return np.asarray(beams, dtype=np.int32)
    elif grouping in ('none', 'single', 'all'):
        return np.zeros(len(imgrelpaths), dtype=np.int32)
    else:
        raise ValueError(f'unsupported shading correction grouping {grouping}')


def _evenly_spaced(indices, maxnum):
    indices = np.asarray(indices)
    if (maxnum is None) or (indices.size <= maxnum):
        return indices
    sel = np.unique(np.round(np.linspace(0, indices.size - 1, num=maxnum)).astype(np.int64))
    return indices[sel]


def downsample_tile(img, working_size):
    """
    grayscale copy of a raw tile downsampled to working_size x working_size,
    keeping uint8/uint16 dtypes (float32 otherwise).
    """
    while (img.ndim > 2) and (img.shape[-1] == 1):
        img = img[..., 0]
    if img.ndim > 2:
        img = img.mean(axis=-1)
    if img.dtype not in (np.uint8, np.uint16):
        img = img.astype(np.float32)
    return cv2.resize(img, (working_size, working_size), interpolation=cv2.INTER_AREA)


def _read_downsampled_tile(imgpath, working_size, keep_dtype=False):
    img = common.imread(imgpath, flag=cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f'Image file {imgpath} not valid!')
    img = downsample_tile(img, working_size)
    return img if keep_dtype else img.astype(np.float32, copy=False)


def _subprocess_read_tiles(imgpaths, working_size, index=None, num_threads=4):
    """read and downsample a list of tiles (native dtype). failed reads are returned as None."""
    def _reader(pth):
        try:
            return _read_downsampled_tile(pth, working_size, keep_dtype=True)
        except Exception:
            return None
    return index, storage.map_io(_reader, imgpaths, num_threads=num_threads)


class TileStore:
    """
    downsampled copies of the tiles of a section, kept in a memory-mapped
    temporary file on the local disk (not in RAM). Picklable: workers on the
    same machine can read it, or write their copies directly with
    write_shared (after the file is created with create) so that the copies
    do not travel through (and accumulate in) the main process; the main
    process then only marks the indices as filled.
    """
    def __init__(self, num_tiles, working_size, filename=None):
        self.num_tiles = int(num_tiles)
        self.working_size = int(working_size)
        if filename is None:
            filename = storage.local_temp_file(suffix='.shading_tiles')
        self.filename = filename
        self.dtype = None
        self.filled = np.zeros(self.num_tiles, dtype=bool)
        self._mm = None
        self._writable = True


    def __getstate__(self):
        state = self.__dict__.copy()
        state['_mm'] = None
        state['_mm_shared'] = None
        state['_writable'] = False
        return state


    def _memmap(self, mode):
        shape = (self.num_tiles, self.working_size, self.working_size)
        return np.memmap(self.filename, dtype=self.dtype, mode=mode, shape=shape)


    def create(self, dtype):
        """create the (zero-filled, sparse) file for tiles of the given dtype"""
        if not self._writable:
            raise RuntimeError('read-only TileStore')
        if self._mm is None:
            self.dtype = np.dtype(dtype)
            self._mm = self._memmap('w+')
            self._mm.flush()


    def add(self, index, tile):
        if not self._writable:
            raise RuntimeError('read-only TileStore')
        if self._mm is None:
            self.create(tile.dtype)
        self._mm[int(index)] = tile
        self.filled[int(index)] = True


    def write_shared(self, index, tile):
        """write a copy from a worker process (the file must exist); filled is not updated here"""
        if self.dtype is None:
            raise RuntimeError('TileStore.create must be called first')
        if getattr(self, '_mm_shared', None) is None:
            self._mm_shared = self._memmap('r+')
        self._mm_shared[int(index)] = tile


    def flush_shared(self):
        if getattr(self, '_mm_shared', None) is not None:
            self._mm_shared.flush()


    def mark(self, index, tile=None):
        """mark a tile as filled (written by a worker), or add it if the copy is given"""
        if tile is not None:
            self.add(index, tile)
        else:
            self.filled[int(index)] = True


    def get(self, indices):
        """copies of the tiles at the given indices (n x ws x ws)"""
        indices = np.asarray(indices, dtype=np.int64)
        if self._mm is None:
            if self.dtype is None:
                return np.empty((0, self.working_size, self.working_size), dtype=np.float32)
            self._mm = self._memmap('r')
        elif self._writable:
            self._mm.flush()
        return np.array(self._mm[indices])


    def missing(self, indices=None):
        """the indices (among the given ones, default all) without a copy"""
        if indices is None:
            return np.nonzero(~self.filled)[0]
        indices = np.asarray(indices, dtype=np.int64)
        return indices[~self.filled[indices]]


    def close(self):
        self._mm = None
        try:
            os.remove(self.filename)
        except OSError:
            pass


def fill_tile_store(store, imgrootdir, imgrelpaths, indices, num_workers=1):
    """read (and downsample) the tiles at the given indices into a TileStore. Return the number of failed reads."""
    from feabas.concurrent import submit_to_workers
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        return 0
    imgpaths = [storage.join_paths(imgrootdir, imgrelpaths[s]) if imgrootdir else imgrelpaths[s] for s in indices]
    num_jobs = max(1, min(num_workers, int(np.ceil(len(imgpaths) / 16))))
    chunk_idx = np.array_split(np.arange(len(imgpaths)), num_jobs)
    args_list = [([imgpaths[s] for s in cidx], store.working_size) for cidx in chunk_idx]
    kwargs_list = [{'index': k} for k in range(len(chunk_idx))]
    num_failed = 0
    for k, imgs in submit_to_workers(_subprocess_read_tiles, args=args_list, kwargs=kwargs_list, num_workers=num_jobs):
        for s, img in zip(chunk_idx[k], imgs):
            if img is None:
                num_failed += 1
            else:
                store.add(indices[s], img)
    return num_failed


def _store_or_tiles(store, indices, local_workers):
    """what to send to a worker for the tiles at indices: the store itself if the workers are local, otherwise the tiles"""
    return (store, np.asarray(indices)) if local_workers else (None, store.get(indices))


def _get_tiles(store, data):
    return store.get(data) if store is not None else data


def _subprocess_fit(key, store, data, **kwargs):
    flatfield, darkfield = basic_fit(_get_tiles(store, data), **kwargs)
    return key, flatfield, darkfield


_FIT_KWARGS = ('get_darkfield', 'smoothness_flatfield', 'smoothness_darkfield', 'sort_intensity',
               'max_iterations', 'max_reweight_iterations', 'optimization_tol', 'reweight_tol')


def section_anchors(locations, num_anchors='auto', max_images=100, seed=0):
    """
    anchor positions (centres of the neighbourhoods fitted separately) for the
    localized estimation.
    Args:
        locations (N x 2): the location of each tile (e.g. mFoV centres).
        num_anchors: number of anchors, or 'auto' for about
            2 * (number of unique locations) / max_images, so that the
            neighbourhoods of max_images locations overlap ~2x.
    Return:
        anchors (K x 2), neighbourhoods (K x M int array of indices into the
        unique locations, the M nearest ones to each anchor), unique
        locations (U x 2), location index of each tile (N,), and the blending
        sigma (median distance between neighbouring anchors).
    """
    from scipy.cluster.vq import kmeans2
    from scipy.spatial import cKDTree
    locations = np.asarray(locations, dtype=np.float64).reshape(-1, 2)
    uloc, loc_idx = np.unique(locations, axis=0, return_inverse=True)
    loc_idx = loc_idx.ravel()
    U = uloc.shape[0]
    if (num_anchors is None) or (str(num_anchors).lower() == 'auto'):
        num_anchors = max(1, int(round(2 * U / max(1, max_images))))
    num_anchors = int(max(1, min(int(num_anchors), U)))
    if num_anchors == 1:
        anchors = uloc.mean(axis=0, keepdims=True)
    else:
        try:
            anchors, _ = kmeans2(uloc, num_anchors, minit='++', rng=seed)
        except TypeError:
            anchors, _ = kmeans2(uloc, num_anchors, minit='++', seed=seed)
        anchors = anchors[np.all(np.isfinite(anchors), axis=1)]
    M = int(min(U, max(1, max_images)))
    _, nbs = cKDTree(uloc).query(anchors, k=M)
    nbs = np.asarray(nbs).reshape(anchors.shape[0], M)
    if anchors.shape[0] > 1:
        dd, _ = cKDTree(anchors).query(anchors, k=2)
        sigma = float(np.median(dd[:, 1]))
    else:
        sigma = np.inf
    return anchors, nbs, uloc, loc_idx, sigma


def blending_weights(xy, anchors, valid, sigma, num_blend=4):
    """
    indices and weights of the anchors blended for points xy: the num_blend
    nearest valid anchors, Gaussian weights of the distance (sigma), summing
    to 1. Return (N x num_blend int, -1 for unused) and (N x num_blend float).
    """
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    vidx = np.nonzero(valid)[0]
    N = xy.shape[0]
    nb = int(max(1, min(num_blend, vidx.size)))
    idx_out = np.full((N, nb), -1, dtype=np.int32)
    wt_out = np.zeros((N, nb), dtype=np.float32)
    if vidx.size == 0:
        return idx_out, wt_out
    d2 = np.sum((xy[:, None, :] - anchors[vidx][None, :, :]) ** 2, axis=-1)
    order = np.argsort(d2, axis=1)[:, :nb]
    d2s = np.take_along_axis(d2, order, axis=1)
    if np.isfinite(sigma) and sigma > 0:
        w = np.exp(-(d2s - d2s[:, :1]) / (2 * sigma ** 2))
    else:
        w = np.zeros_like(d2s)
        w[:, 0] = 1
    w = w / w.sum(axis=1, keepdims=True)
    idx_out[:] = vidx[order]
    wt_out[:] = w
    return idx_out, wt_out


def _subprocess_fit_group_local(key, store, data, tile_loc, neighbourhoods, min_images, num_samples=0,
                                anchor_ids=None, **fit_kwargs):
    """
    fit the profiles of one group (beam) at each given anchor neighbourhood.
    Args:
        key: group key.
        store, data: a TileStore and the indices of the group's (valid) tiles
            in it, or None and the tiles themselves.
        tile_loc (ndarray): the unique location index of each of these tiles.
        neighbourhoods (k x M): unique location indices of each neighbourhood.
        anchor_ids (list): the anchor index of each neighbourhood (default
            0...k-1).
    Return:
        key, {anchor index: (flatfield, darkfield)}, sampled tiles (for the
        pooled profile).
    """
    tiles = _get_tiles(store, data)
    tile_loc = np.asarray(tile_loc)
    if anchor_ids is None:
        anchor_ids = range(len(neighbourhoods))
    profiles = {}
    for a, nb in zip(anchor_ids, neighbourhoods):
        sel = np.nonzero(np.isin(tile_loc, nb))[0]
        if sel.size >= min_images:
            profiles[a] = basic_fit(tiles[sel], **fit_kwargs)
    samples = tiles[_evenly_spaced(np.arange(len(tiles)), num_samples)] if num_samples > 0 else tiles[:0]
    return key, profiles, samples


def _estimate_global(store, group_keys, tile_group, max_images, min_images, num_workers, fit_kwargs,
                     logger, secname, local_workers=True):
    """one profile per group, from the valid tiles in the store."""
    from feabas.concurrent import submit_to_workers
    valid = store.filled
    stacks_idx = {}
    for gidx, gkey in enumerate(group_keys):
        sel = _evenly_spaced(np.nonzero((tile_group == gidx) & valid)[0], max_images)
        if sel.size >= min_images:
            stacks_idx[int(gkey)] = sel
    pooled_idx = _evenly_spaced(np.nonzero(valid)[0], max_images)
    if pooled_idx.size < min_images:
        logger.warning(f'{secname}: only {pooled_idx.size} valid tiles, skip shading correction.')
        return None
    if (len(group_keys) > 1) or (len(stacks_idx) == 0):
        stacks_idx[POOLED_GROUP_KEY] = pooled_idx
    fit_keys = sorted(stacks_idx.keys())
    args_list = [(k, *_store_or_tiles(store, stacks_idx[k], local_workers)) for k in fit_keys]
    profiles = {}
    for key, flatfield, darkfield in submit_to_workers(_subprocess_fit, args=args_list, kwargs=[fit_kwargs],
                                                       num_workers=min(num_workers, len(args_list))):
        profiles[key] = (flatfield, darkfield)
    key_to_idx = {k: i for i, k in enumerate(fit_keys)}
    default_profile = key_to_idx.get(POOLED_GROUP_KEY, 0)
    tile_profiles = np.array([key_to_idx.get(int(group_keys[g]), default_profile) for g in tile_group],
                             dtype=np.int32).reshape(-1, 1)
    tile_weights = np.ones(tile_profiles.shape, dtype=np.float32)
    num_fallback = sum(int(k) not in stacks_idx for k in group_keys)
    if (num_fallback > 0) and (len(group_keys) > 1):
        logger.info(f'{secname}: {num_fallback}/{len(group_keys)} groups with too few tiles use the pooled shading profile.')
    return {
        'flatfield': np.stack([profiles[k][0] for k in fit_keys], axis=0),
        'darkfield': np.stack([profiles[k][1] for k in fit_keys], axis=0),
        'profile_group': np.array(fit_keys, dtype=np.int32),
        'profile_anchor': np.full(len(fit_keys), -1, dtype=np.int32),
        'tile_profiles': tile_profiles, 'tile_weights': tile_weights,
        'default_profile': default_profile,
    }


def _estimate_local(store, group_keys, tile_group, locations, max_images, min_images, num_workers, fit_kwargs,
                    logger, secname, num_anchors='auto', num_blend=4, neighbourhood=50, local_workers=True):
    """profiles per group and per anchor neighbourhood, from the valid tiles in the store, blended per tile."""
    from feabas.concurrent import submit_to_workers
    valid = store.filled
    anchors, nbs, uloc, loc_idx, sigma = section_anchors(locations, num_anchors=num_anchors, max_images=neighbourhood)
    K = anchors.shape[0]
    G = len(group_keys)
    num_samples = int(np.ceil(max_images / G)) if G > 1 else 0
    # one job per group and chunk of anchors: about 4 jobs per worker for an even load
    num_chunks = int(min(K, max(1, np.ceil(4 * num_workers / max(1, G)))))
    anchor_chunks = [c for c in np.array_split(np.arange(K), num_chunks) if c.size > 0]
    args_list, kwargs_list = [], []
    for gidx, gkey in enumerate(group_keys):
        tidx = np.nonzero((tile_group == gidx) & valid)[0]
        if tidx.size == 0:
            continue
        for c, aids in enumerate(anchor_chunks):
            tsel = tidx[np.isin(loc_idx[tidx], np.unique(nbs[aids]))]
            if tsel.size == 0:
                continue
            args_list.append((int(gkey), *_store_or_tiles(store, tsel, local_workers), loc_idx[tsel], nbs[aids], min_images))
            kwargs_list.append(dict(fit_kwargs, num_samples=(num_samples if c == 0 else 0), anchor_ids=list(aids)))
    group_profiles, samples = {}, []
    for key, profiles, smp in submit_to_workers(_subprocess_fit_group_local, args=args_list, kwargs=kwargs_list,
                                                num_workers=min(num_workers, max(1, len(args_list)))):
        group_profiles.setdefault(key, {}).update(profiles)
        samples.extend(list(smp))
    flats, darks, pgroup, panchor = [], [], [], []
    index_of = {}
    for gkey in sorted(group_profiles.keys()):
        for a in sorted(group_profiles[gkey].keys()):
            index_of[(gkey, a)] = len(flats)
            f, d = group_profiles[gkey][a]
            flats.append(f); darks.append(d); pgroup.append(gkey); panchor.append(a)
    groups_wo = [int(k) for k in group_keys if len(group_profiles.get(int(k), {})) == 0]
    default_profile = 0
    if (len(groups_wo) > 0) or (len(flats) == 0):
        if G > 1 and len(samples) >= min_images:
            f, d = basic_fit(np.stack(samples, axis=0), **fit_kwargs)
            default_profile = len(flats)
            flats.append(f); darks.append(d); pgroup.append(POOLED_GROUP_KEY); panchor.append(-1)
            if len(groups_wo) > 0:
                logger.info(f'{secname}: {len(groups_wo)}/{G} groups with too few tiles use the pooled shading profile.')
        elif len(flats) == 0:
            logger.warning(f'{secname}: too few valid tiles, skip shading correction.')
            return None
    num_blend = int(max(1, min(num_blend, K)))
    N = len(tile_group)
    tile_profiles = np.full((N, num_blend), -1, dtype=np.int32)
    tile_weights = np.zeros((N, num_blend), dtype=np.float32)
    tile_xy = uloc[loc_idx]
    for gidx, gkey in enumerate(group_keys):
        tidx = np.nonzero(tile_group == gidx)[0]
        gp = group_profiles.get(int(gkey), {})
        if len(gp) == 0:
            tile_profiles[tidx, 0] = default_profile
            tile_weights[tidx, 0] = 1
            continue
        valid_a = np.zeros(K, dtype=bool)
        valid_a[list(gp.keys())] = True
        aidx, wt = blending_weights(tile_xy[tidx], anchors, valid_a, sigma, num_blend=num_blend)
        lut = np.full(K, -1, dtype=np.int32)
        for a in gp.keys():
            lut[a] = index_of[(int(gkey), a)]
        pidx = np.where(aidx >= 0, lut[np.maximum(aidx, 0)], -1)
        tile_profiles[tidx, :pidx.shape[1]] = pidx
        tile_weights[tidx, :wt.shape[1]] = wt
    logger.info(f'{secname}: {K} shading anchors, {len(flats)} profiles.')
    return {
        'flatfield': np.stack(flats, axis=0), 'darkfield': np.stack(darks, axis=0),
        'profile_group': np.array(pgroup, dtype=np.int32), 'profile_anchor': np.array(panchor, dtype=np.int32),
        'tile_profiles': tile_profiles, 'tile_weights': tile_weights,
        'default_profile': default_profile, 'anchors': anchors, 'blend_sigma': sigma,
    }


def tiles_needed(imgrelpaths, grouping='none', local=False, max_images=100):
    """indices of the tiles used for the estimation: all of them in local mode, else up to max_images per group"""
    if local:
        return np.arange(len(imgrelpaths))
    groups = tile_groups(imgrelpaths, grouping)
    sel = [_evenly_spaced(np.nonzero(groups == g)[0], max_images) for g in np.unique(groups)]
    return np.unique(np.concatenate(sel, axis=None)).astype(np.int64)


def estimate_section_shading(imgrootdir, imgrelpaths, outname, grouping='none', **kwargs):
    """
    estimate the shading profiles of the tiles of one section and save them.
    Args:
        imgrootdir (str): the root directory of the image tiles.
        imgrelpaths (list of str): the tile paths relative to imgrootdir.
        outname (str): the output h5 file.
        grouping (str): 'beam' or 'none', see tile_groups.
    Kwargs:
        tile_store (TileStore): downsampled copies of (some of) the tiles,
            e.g. collected while matching. The missing ones still needed are
            read. If not given, the needed tiles are read into a temporary one.
        working_size (int): tiles are downsampled to this size for fitting.
        max_images_per_group (int): maximum number of tiles per group to use
            (global mode, and for the pooled profile).
        min_images_per_group (int): groups with fewer tiles use the profile
            pooled from all the groups.
        local (bool): fit profiles on neighbourhoods of the section and blend
            them per tile, to follow shading drift. Needs locations.
        locations (N x 2): position of each tile (e.g. its mFoV centre).
        local_neighbourhood (int): number of nearest positions (mFoVs/tiles)
            in each neighbourhood, i.e. the tiles per profile in local mode.
        local_anchors: number of neighbourhoods, or 'auto' for about
            2 x positions / local_neighbourhood.
        local_blend (int): number of nearest neighbourhoods blended per tile.
        num_workers (int): number of workers.
        logger: logger info.
        the rest is passed to basic_fit.
    Return:
        outname if saved, otherwise None (too few tiles).
    """
    from feabas.concurrent import REMOTE_FRAMEWORKS, DEFAUL_FRAMEWORK
    tile_store = kwargs.pop('tile_store', None)
    working_size = int(kwargs.pop('working_size', 128))
    max_images = kwargs.pop('max_images_per_group', 100)
    min_images = max(2, kwargs.pop('min_images_per_group', 16))
    num_workers = kwargs.pop('num_workers', 1) or 1
    logger_info = kwargs.pop('logger', None)
    local = kwargs.pop('local', False)
    locations = kwargs.pop('locations', None)
    num_anchors = kwargs.pop('local_anchors', 'auto')
    num_blend = kwargs.pop('local_blend', 4)
    neighbourhood = kwargs.pop('local_neighbourhood', 50)
    fit_kwargs = {k: v for k, v in kwargs.items() if k in _FIT_KWARGS}
    logger = logging.get_logger(logger_info)
    secname = os.path.splitext(os.path.basename(outname))[0]
    groups = tile_groups(imgrelpaths, grouping)
    group_keys, tile_group = np.unique(groups, return_inverse=True)
    tile_group = tile_group.ravel()
    if local and (locations is None):
        logger.warning(f'{secname}: no tile locations given, fall back to global shading profiles.')
        local = False
    own_store = tile_store is None
    if own_store:
        tile_store = TileStore(len(imgrelpaths), working_size)
    elif tile_store.working_size != working_size:
        raise ValueError('working size of the tile store does not match')
    try:
        missing = tile_store.missing(tiles_needed(imgrelpaths, grouping, local, max_images))
        if not own_store:
            logger.info(f'{secname}: {int(np.sum(tile_store.filled))} tile copies from matching, {missing.size} more to read for shading estimation.')
        num_failed = fill_tile_store(tile_store, imgrootdir, imgrelpaths, missing, num_workers=num_workers)
        if num_failed > 0:
            logger.warning(f'{secname}: failed to read {num_failed} tiles for shading estimation.')
        local_workers = DEFAUL_FRAMEWORK not in REMOTE_FRAMEWORKS
        if local:
            res = _estimate_local(tile_store, group_keys, tile_group, locations, max_images, min_images, num_workers,
                                  fit_kwargs, logger, secname, num_anchors=num_anchors, num_blend=num_blend,
                                  neighbourhood=neighbourhood, local_workers=local_workers)
        else:
            res = _estimate_global(tile_store, group_keys, tile_group, max_images, min_images, num_workers,
                                   fit_kwargs, logger, secname, local_workers=local_workers)
    finally:
        if own_store:
            tile_store.close()
    if res is None:
        return None
    imgnames = [os.path.basename(s) for s in imgrelpaths]
    H5File = storage.h5file_class()
    storage.makedirs(os.path.dirname(outname))
    with H5File(outname, 'w') as f:
        f.create_dataset('flatfield', data=res['flatfield'].astype(np.float16), compression='gzip')
        f.create_dataset('darkfield', data=res['darkfield'].astype(np.float16), compression='gzip')
        f.create_dataset('profile_group', data=res['profile_group'])
        f.create_dataset('profile_anchor', data=res['profile_anchor'])
        f.create_dataset('imgnames', data=common.str_to_numpy_ascii('\n'.join(imgnames)), compression='gzip')
        f.create_dataset('tile_profiles', data=res['tile_profiles'], compression='gzip')
        f.create_dataset('tile_weights', data=res['tile_weights'], compression='gzip')
        if 'anchors' in res:
            f.create_dataset('anchors', data=res['anchors'])
            f.attrs['local_neighbourhood'] = int(neighbourhood)
            f.attrs['blend_sigma'] = float(res['blend_sigma'])
        f.attrs['default_profile'] = int(res['default_profile'])
        f.attrs['working_size'] = working_size
        f.attrs['grouping'] = str(grouping)
        f.attrs['local'] = bool(local)
        f.attrs['get_darkfield'] = bool(fit_kwargs.get('get_darkfield', True))
        f.attrs['version'] = SHADING_FILE_VERSION
    return outname


def correct_photometric(profiles, imgname, tile_size, box, mean, std, blocks=None):
    """
    overlap statistics of a raw tile converted to those of the shading
    corrected tile. With the raw tile I = c * F + D (c the content, roughly
    independent of F), the corrected one is c + mean(D), so over a region
    mean' = (mean(I) - mean(D)) / mean(F) + mean(D) and, for the detrended
    std, std' = std(I) / sqrt(mean(F^2)).
    With block statistics (see matcher._block_photometric), the conversion is
    done block by block along the overlap, otherwise with F and D averaged
    over the whole overlapping region.
    Args:
        profiles (ShadingProfiles): the shading profiles of the section.
        imgname (str): the tile.
        tile_size: (height, width) of the tile.
        box: (xmin, ymin, xmax, ymax) of the overlapping region in the tile.
        mean, std: the statistics measured on the raw tile (std detrended).
        blocks: (n_blocks, axis, length, block means, block detrended
            variances, block weights) of this tile, or None.
    """
    inv_flat, dark, dark_mean = profiles.small_fields(imgname)
    gh, gw = inv_flat.shape
    ht, wd = float(tile_size[0]), float(tile_size[1])
    def region_means(bx):
        x0 = int(np.clip(np.floor(bx[0] / wd * gw), 0, gw - 1))
        x1 = int(np.clip(np.ceil(bx[2] / wd * gw), x0 + 1, gw))
        y0 = int(np.clip(np.floor(bx[1] / ht * gh), 0, gh - 1))
        y1 = int(np.clip(np.ceil(bx[3] / ht * gh), y0 + 1, gh))
        flat = 1.0 / inv_flat[y0:y1, x0:x1]
        d = float(dark[y0:y1, x0:x1].mean()) if dark is not None else 0.0
        return float(flat.mean()), float(np.mean(flat ** 2)), d
    if blocks is not None:
        nb, axis, L, mb, vb, wb = blocks
        nb, axis, L = int(nb), int(axis), int(L)
        mb, vb, wb = np.asarray(mb)[:nb], np.asarray(vb)[:nb], np.asarray(wb)[:nb]
        ok = np.isfinite(mb) & np.isfinite(vb) & (wb > 0)
        if np.any(ok):
            lo, hi = (box[0], box[2]) if axis == 1 else (box[1], box[3])
            m_out, v_out, w_out = 0.0, 0.0, 0.0
            for b, idx in enumerate(np.array_split(np.arange(L), nb)):
                if not ok[b]:
                    continue
                f0 = lo + (hi - lo) * idx[0] / L
                f1 = lo + (hi - lo) * (idx[-1] + 1) / L
                bx = (f0, box[1], f1, box[3]) if axis == 1 else (box[0], f0, box[2], f1)
                f1m, f2m, d = region_means(bx)
                m_out += wb[b] * ((mb[b] - dark_mean - d) / f1m + dark_mean)
                v_out += wb[b] * vb[b] / f2m
                w_out += wb[b]
            return m_out / w_out, (v_out / w_out) ** 0.5
    f1m, f2m, d = region_means(box)
    return (mean - dark_mean - d) / f1m + dark_mean, std / f2m ** 0.5


##------------------------------- application --------------------------------##

class ShadingProfiles:
    """
    shading profiles of the tiles of one section, loaded from the file saved
    by estimate_section_shading. Each tile uses a weighted blend of one or
    more profiles (more than one in local mode).
    """
    def __init__(self, flatfields, darkfields, imgnames, tile_profiles, tile_weights=None, default_profile=0):
        # kept as float16 (as saved): local mode can have thousands of profiles
        self._flatfields = np.asarray(flatfields, dtype=np.float16)
        self._darkfields = np.asarray(darkfields, dtype=np.float16)
        tile_profiles = np.asarray(tile_profiles, dtype=np.int32)
        if tile_profiles.ndim == 1:
            tile_profiles = tile_profiles.reshape(-1, 1)
        if tile_weights is None:
            tile_weights = (tile_profiles >= 0).astype(np.float32)
        self._tile_profiles = tile_profiles
        self._tile_weights = np.asarray(tile_weights, dtype=np.float32).reshape(tile_profiles.shape)
        self._lut = {nm: k for k, nm in enumerate(imgnames)}
        self._default_profile = int(default_profile)
        self._small_cache = OrderedDict()  # blended profiles at the working size (LRU)


    @classmethod
    def from_h5(cls, filename):
        H5File = storage.h5file_class()
        with H5File(filename, 'r') as f:
            flatfields = f['flatfield'][()]
            darkfields = f['darkfield'][()]
            imgnames = common.numpy_to_str_ascii(f['imgnames'][()]).split('\n')
            if 'tile_profiles' in f:
                tile_profiles = f['tile_profiles'][()]
                tile_weights = f['tile_weights'][()]
                default_profile = int(f.attrs.get('default_profile', 0))
            else:   # version 1: one profile per tile
                tile_profiles = f['tile_group'][()]
                tile_weights = None
                default_profile = int(f.attrs.get('default_group', 0))
        return cls(flatfields, darkfields, imgnames, tile_profiles, tile_weights, default_profile=default_profile)


    def profiles_of(self, imgname):
        """indices and weights of the profiles used for a tile"""
        k = self._lut.get(os.path.basename(imgname), None)
        if k is None:
            return (self._default_profile,), (1.0,)
        idx = self._tile_profiles[k]
        wt = self._tile_weights[k]
        valid = (idx >= 0) & (wt > 0)
        if not np.any(valid):
            return (self._default_profile,), (1.0,)
        return tuple(int(s) for s in idx[valid]), tuple(float(s) for s in wt[valid])


    def group_of(self, imgname):
        """the main (largest weight) profile of a tile"""
        idx, wt = self.profiles_of(imgname)
        return idx[int(np.argmax(wt))]


    def small_fields(self, imgname):
        """
        (1/F, D - mean(D), mean(D)) of a tile at the working size, with F and
        D blended from the tile's profiles (F renormalized to mean 1).
        """
        idx, wt = self.profiles_of(imgname)
        key = (idx, tuple(round(w, 4) for w in wt))
        if key in self._small_cache:
            self._small_cache.move_to_end(key)
        else:
            if len(idx) == 1:
                flat = self._flatfields[idx[0]].astype(np.float32)
                dark = self._darkfields[idx[0]].astype(np.float32)
            else:
                w = np.array(wt, dtype=np.float32)
                w = w / w.sum()
                flat = np.tensordot(w, self._flatfields[list(idx)].astype(np.float32), axes=1)
                flat = flat / flat.mean()
                dark = np.tensordot(w, self._darkfields[list(idx)].astype(np.float32), axes=1)
            dark_mean = float(dark.mean())
            dark = dark - dark_mean
            inv_flat = 1.0 / flat.clip(1e-3, None)
            if not np.any(dark):
                dark = None
            self._small_cache[key] = (inv_flat.astype(np.float32),
                                      None if dark is None else dark.astype(np.float32), dark_mean)
            while len(self._small_cache) > SMALL_CACHE_SIZE:
                self._small_cache.popitem(last=False)
        return self._small_cache[key]


    def correct(self, img, imgname):
        """(img - D) / F + mean(D), keeping the dtype"""
        inv_flat, dark, dark_mean = self.small_fields(imgname)
        ht, wd = img.shape[:2]
        inv_flat = cv2.resize(inv_flat, (wd, ht), interpolation=cv2.INTER_LINEAR)
        if dark is not None:
            dark = cv2.resize(dark, (wd, ht), interpolation=cv2.INTER_LINEAR)
        if img.ndim > 2:
            inv_flat = inv_flat[..., None]
            dark = dark[..., None] if dark is not None else None
        dtype = img.dtype
        imgf = img.astype(np.float32)
        # (img - D)/F + mean(D) = (img - mean(D) - (D - mean(D)))/F + mean(D)
        imgf -= dark_mean
        if dark is not None:
            imgf -= dark
        imgf *= inv_flat
        imgf += dark_mean
        if np.issubdtype(dtype, np.integer):
            imgf = np.round(imgf)
            return preserve_extremes(img, imgf, dtype)
        return imgf.astype(dtype, copy=False)


def preserve_extremes(img, img_new, dtype=None):
    """
    for integer dtypes, keep the pixels of img at the extremes of the dtype
    (with a 3x3 closing) unchanged in img_new, and clip the others within the
    extremes (e.g. [1, 254] for uint8).
    """
    if dtype is None:
        dtype = img.dtype
    dtype = np.dtype(dtype)
    if not np.issubdtype(dtype, np.integer):
        return np.asarray(img_new).astype(dtype, copy=False)
    vmin, vmax = np.iinfo(dtype).min, np.iinfo(dtype).max
    mask_t = (img == vmin) | (img == vmax)
    if np.any(mask_t):
        if mask_t.ndim > 2:
            mask_t = np.any(mask_t, axis=-1)
        mask_t = cv2.dilate(mask_t.astype(np.uint8), np.ones((3,3), dtype=np.uint8))
        mask_t = cv2.erode(mask_t, np.ones((3,3), dtype=np.uint8)) > 0
    out = np.asarray(img_new).clip(vmin + 1, vmax - 1).astype(dtype)
    if np.any(mask_t):
        out[mask_t] = img[mask_t]
    return out


@lru_cache(maxsize=8)
def load_profiles(filename):
    return ShadingProfiles.from_h5(filename)
