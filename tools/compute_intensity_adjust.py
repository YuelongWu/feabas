import argparse
import cv2
from functools import partial
import json
import numpy as np
import os

from feabas.concurrent import submit_to_workers
from feabas import common, config, storage
"""
per-section linear intensity adjustment (y = b + a * x) that normalizes the
brightness and contrast of the aligned stack, from the foreground/background
masks of tools/intensity_masks.py (100 foreground, 255 background, measured on
the unaligned thumbnails).
Methods:
  zmatch (default): match each section to its z-neighbors. For the pairs (z,
      z+1) and (z, z+2), a robust fit y = c x + d between co-located pixels of
      the aligned thumbnails (tools/render_aligned_thumbnails.py); the gains
      are solved in log (log a_i - log a_j = log c_ij) and then the offsets,
      over the whole stack, each pair weighted by its information (pixel count
      and correlation). A weak prior pulls every section to background ->
      bg_target and foreground -> fg_target, so the targets only set the trend
      on scales longer than ~smooth_length sections; sections with little
      tissue follow their neighbors.
  stats: from the masked statistics of each section alone. background ->
      bg_target, and foreground ->
      smooth: the z-smoothed foreground trend (smoothing the contrast,
          background - foreground mean), the stack average scaled to fg_target.
      fixed: fg_target for every section.
Output, one file per section in the transfer function format of
dal.AbstractImageLoader.tf_lut:
    {"__TYPE__": "BRIGHTNESS_CONTRAST_ADJUST", "": [b, a]}
default folder align/intensity_adjust, read by the aligned rendering if
rendering: intensity_adjust_dir is set, and applied on top of the per-tile
stitching adjustment. A summary table goes to intensity_adjust.csv.
"""

FOREGROUND_VAL = 100
BACKGROUND_VAL = 255
BG_TARGET = 230.0
FG_TARGET = 160.0
MIN_PIXELS = 500        # sections with fewer foreground pixels have no reliable statistics
MIN_DIFF = 5.0          # sections with background - foreground mean smaller than this: same
# stats method
MEDIAN_WINDOW = 15      # smooth: median filter window (sections) of the contrast, against outliers
SMOOTH_SIGMA = 50.0     # smooth: gaussian sigma (sections) of the contrast trend
GAIN_RANGE = (0.5, 2.0) # smooth: per-section contrast corrections (before the global enhancement) are clipped to this range
# zmatch method
NB = 128                # joint histogram bins
STEPS = (1, 2)          # z-distances of the matched pairs
STEP_W = {1: 1.0, 2: 0.5}   # weight of the pairs by z-distance
BLUR = 1.5              # gaussian blur (thumbnail pixels) against misalignment and noise
ERODE = 4               # pixels at the edge of the sections left out
N0 = 20000              # pixel count giving a pair unit weight
HUBER = 12.0            # gray levels: robust weighting of the pair fits
INFO_CAP = 50.0         # cap of the pair information rho^2 / (1 - rho^2)
PRIOR_FLOOR = 0.2       # minimum prior reliability, for the sections without usable statistics
SMOOTH_LENGTH = 30.0    # sections: the targets set the trend on scales longer than this
PAIR_CHUNK = 512        # pairs fitted at once (bounds the temporary memory)


def section_stats(secname, thumb_dir, mask_dir, ext='.png'):
    """
    Return:
        (secname, bg_mean, fg_mean, bg_count, fg_count, hist): hist is the
        (256,) grayscale histogram of the imaged pixels.
    """
    img = common.imread(storage.join_paths(thumb_dir, secname + ext), flag=cv2.IMREAD_GRAYSCALE)
    msk = common.imread(storage.join_paths(mask_dir, secname + '.png'), flag=cv2.IMREAD_GRAYSCALE)
    bg = msk == BACKGROUND_VAL
    fg = msk == FOREGROUND_VAL
    nb, nf = int(np.count_nonzero(bg)), int(np.count_nonzero(fg))
    bmean = float(img[bg].mean()) if nb > 0 else np.nan
    fmean = float(img[fg].mean()) if nf > 0 else np.nan
    hist = np.bincount(img[msk > 0], minlength=256)
    return secname, bmean, fmean, nb, nf, hist


def _interp_invalid(v, valid):
    """fill the invalid entries of v by linear interpolation of the valid ones (by index)."""
    v = v.copy()
    idx = np.arange(v.size)
    if np.any(valid) and np.any(~valid):
        v[~valid] = np.interp(idx[~valid], idx[valid], v[valid])
    return v


def _smooth(v, median_window, sigma):
    from scipy.ndimage import median_filter, gaussian_filter1d
    if median_window > 1:
        v = median_filter(v, size=median_window, mode='nearest')
    if sigma > 0:
        v = gaussian_filter1d(v, sigma, mode='nearest')
    return v


def reliable_sections(bmean, fmean, nf, **kwargs):
    """sections with reliable foreground statistics."""
    min_pixels = kwargs.get('min_pixels', MIN_PIXELS)
    min_diff = kwargs.get('min_diff', MIN_DIFF)
    return np.isfinite(bmean) & np.isfinite(fmean) & (nf >= min_pixels) & (np.nan_to_num(bmean - fmean) >= min_diff)


def stats_adjust(bmean, fmean, nf, **kwargs):
    """
    Args:
        bmean, fmean (N ndarray): background / foreground means in z order.
        nf (N ndarray): foreground pixel counts.
    Kwargs:
        mode (str): 'smooth' or 'fixed'.
        bg_target (float): background value after adjustment.
        fg_target (float): smooth: average over the sections of the
            foreground mean after adjustment; fixed: foreground mean of every
            section after adjustment.
    Return:
        a, b (N ndarray): contrast gains and brightness offsets (y = b + a * x).
        valid (N bool ndarray): sections with reliable foreground statistics.
    """
    mode = kwargs.get('mode', 'smooth')
    bg_target = kwargs.get('bg_target', BG_TARGET)
    fg_target = kwargs.get('fg_target', FG_TARGET)
    median_window = kwargs.get('median_window', MEDIAN_WINDOW)
    smooth_sigma = kwargs.get('smooth_sigma', SMOOTH_SIGMA)
    gain_range = kwargs.get('gain_range', GAIN_RANGE)
    bvalid = np.isfinite(bmean)
    if not np.any(bvalid):
        raise RuntimeError('no background found in any section.')
    bmean = _interp_invalid(bmean, bvalid)
    contrast = bmean - fmean
    valid = reliable_sections(bmean, fmean, nf, **kwargs)
    if not np.any(valid):
        raise RuntimeError('no reliable foreground found in any section.')
    a = np.full(bmean.size, np.nan)
    if mode == 'fixed':
        a[valid] = (bg_target - fg_target) / contrast[valid]
        a = np.exp(_interp_invalid(np.log(a), valid))
    elif mode == 'smooth':
        # smooth the foreground mean once the background is constant, i.e. the contrast bg - fg
        contrast_smooth = _smooth(_interp_invalid(contrast, valid), median_window, smooth_sigma)
        a[valid] = contrast_smooth[valid] / contrast[valid]
        a = np.clip(a, *gain_range)
        # unreliable foreground: contrast of the neighbors (smoothed gains)
        a_fill = _smooth(_interp_invalid(a, valid), median_window, max(1.0, smooth_sigma / 10))
        a[~valid] = a_fill[~valid]
        a = a * (bg_target - fg_target) / np.mean(contrast_smooth)
    else:
        raise ValueError(f'unsupported stats mode {mode}')
    b = bg_target - a * bmean
    return a, b, valid


def pair_hists(k, secnames, src_dir, ext='.png', nb=NB, blur=BLUR, erode=ERODE):
    """
    joint histograms of the co-located pixels of secnames[0] and each of the
    following sections in the aligned thumbnail folder src_dir.
    Return:
        k, and the list of (nb x nb float32 ndarray).
    """
    def load(s):
        img = common.imread(storage.join_paths(src_dir, s + ext), flag=cv2.IMREAD_GRAYSCALE)
        m = img > 0
        if erode > 0:
            m = cv2.erode(m.astype(np.uint8), np.ones((2 * erode + 1, 2 * erode + 1), np.uint8)) > 0
        if blur > 0:
            img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), blur)
        return img.astype(np.float32), m
    i0, m0 = load(secnames[0])
    out = []
    for s in secnames[1:]:
        i1, m1 = load(s)
        m = m0 & m1
        x = np.clip(i0[m] * (nb / 256.0), 0, nb - 1).astype(np.int32)
        y = np.clip(i1[m] * (nb / 256.0), 0, nb - 1).astype(np.int32)
        out.append(np.bincount(x * nb + y, minlength=nb * nb).astype(np.float32).reshape(nb, nb))
    return k, out


def build_pair_hists(secnames, src_dir, ext='.png', steps=STEPS, num_workers=1, **kwargs):
    """
    Return:
        I, J (P ndarray): z indices of the pairs; H (P x nb x nb float32 ndarray).
    """
    n = len(secnames)
    nb = kwargs.get('nb', NB)
    tfunc = partial(pair_hists, src_dir=src_dir, ext=ext, **kwargs)
    jobs = [(k, [secnames[k + d] for d in (0,) + tuple(steps) if k + d < n]) for k in range(n - 1)]
    I = np.array([k for k in range(n - 1) for d in steps if k + d < n])
    J = np.array([k + d for k in range(n - 1) for d in steps if k + d < n])
    pair_id = {(i, j): p for p, (i, j) in enumerate(zip(I, J))}
    H = np.zeros((I.size, nb, nb), dtype=np.float32)
    # the workers may return out of order
    for k, hs in submit_to_workers(tfunc, args=jobs, num_workers=num_workers):
        for d, h in zip(steps, hs):
            H[pair_id[(k, k + d)]] = h
    return I, J, H


def pair_fits(H, huber=HUBER, n_iter=5, chunk=PAIR_CHUNK):
    """
    robust reduced-major-axis fits y = c x + d of the pairs from their joint
    histograms (Huber weights on the orthogonal residuals).
    Return:
        c, d, corr, count, mx (P ndarray): slope, intercept, correlation,
            pixel count and mean x of each pair.
    """
    nb = H.shape[-1]
    v = ((np.arange(nb) + 0.5) * (256.0 / nb)).astype(np.float32)
    X, Y = v[None, :, None], v[None, None, :]
    outs = []
    for s0 in range(0, H.shape[0], chunk):
        h = H[s0:(s0 + chunk)]
        cnt = h.sum(axis=(1, 2))
        w = h
        for it in range(n_iter + 1):
            W = np.maximum(w.sum(axis=(1, 2)), 1e-6)
            mx = (w * X).sum(axis=(1, 2)) / W
            my = (w * Y).sum(axis=(1, 2)) / W
            dx = X - mx[:, None, None]
            dy = Y - my[:, None, None]
            vx = (w * dx * dx).sum(axis=(1, 2)) / W
            vy = (w * dy * dy).sum(axis=(1, 2)) / W
            cxy = (w * dx * dy).sum(axis=(1, 2)) / W
            corr = cxy / np.sqrt(np.maximum(vx * vy, 1e-12))
            c = np.sqrt(np.maximum(vy, 1e-12) / np.maximum(vx, 1e-12))
            d = my - c * mx
            if it == n_iter:
                break
            res = np.abs(Y - c[:, None, None] * X - d[:, None, None]) / np.sqrt(1 + c[:, None, None] ** 2)
            w = h * np.minimum(1, huber / np.maximum(res, 1e-6))
        mx_all = (h.sum(axis=2) * v).sum(axis=1) / np.maximum(cnt, 1)
        outs.append((c, d, corr, cnt, mx_all))
    return [np.concatenate([o[k] for o in outs]) for k in range(5)]


def _lsq_diff(n, I, J, rhs_pair, wpair, prior_val, prior_w):
    """min sum_p wpair (u_i - u_j - rhs_pair)^2 + sum_i prior_w (u_i - prior_val)^2"""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
    rows = np.concatenate([I, J, I, J, np.arange(n)])
    cols = np.concatenate([I, J, J, I, np.arange(n)])
    vals = np.concatenate([wpair, wpair, -wpair, -wpair, prior_w])
    A = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsc()
    rhs = prior_w * prior_val
    np.add.at(rhs, I, wpair * rhs_pair)
    np.add.at(rhs, J, -wpair * rhs_pair)
    return spsolve(A, rhs)


def zmatch_adjust(I, J, H, bmean, fmean, nb, nf, **kwargs):
    """
    Args:
        I, J, H: the pairs from build_pair_hists.
        bmean, fmean, nb, nf (N ndarray): masked statistics in z order.
    Return:
        a, b (N ndarray), valid (N bool ndarray), and a dict of the pair statistics.
    """
    bg_target = kwargs.get('bg_target', BG_TARGET)
    fg_target = kwargs.get('fg_target', FG_TARGET)
    smooth_length = kwargs.get('smooth_length', SMOOTH_LENGTH)
    n = bmean.size
    c, d, corr, cnt, mx = pair_fits(H, huber=kwargs.get('huber', HUBER))
    # precision of log c ~ n rho^2 / (1 - rho^2): resin-dominated pairs carry little weight
    rho2 = np.clip(corr, 0, 0.999) ** 2
    info = np.minimum(rho2 / (1 - rho2), kwargs.get('info_cap', INFO_CAP))
    step_w = np.array([STEP_W.get(int(j - i), 1.0) for i, j in zip(I, J)])
    ok = (cnt > 0) & (c > 0) & np.isfinite(c)
    wp = np.where(ok, step_w * (cnt / N0) * info, 0)
    if not np.any(wp > 0):
        raise RuntimeError('no usable section pairs.')
    lam = np.median(wp[wp > 0]) / smooth_length ** 2
    prior_floor = kwargs.get('prior_floor', PRIOR_FLOOR)
    valid = reliable_sections(bmean, fmean, nf, **kwargs)
    if not np.any(valid):
        raise RuntimeError('no reliable foreground found in any section.')
    bvalid = np.isfinite(bmean)
    bmean_f = _interp_invalid(np.nan_to_num(bmean), bvalid)
    # gains: a_i x + b_i = a_j y + b_j with y = c x + d  ->  a_i = a_j c
    loga_anchor = np.full(n, np.nan)
    loga_anchor[valid] = np.log((bg_target - fg_target) / (bmean[valid] - fmean[valid]))
    loga_anchor = _interp_invalid(loga_anchor, valid)
    rf = np.where(valid, np.minimum(1, nf / 2000), 0)
    loga = _lsq_diff(n, I, J, np.log(np.where(ok, c, 1)), wp, loga_anchor, lam * np.maximum(rf, prior_floor))
    a = np.exp(loga)
    # offsets given the gains: b_i - b_j = a_j (c mx + d) - a_i mx, at the mean x of the pair
    rhs_b = a[J] * (c * mx + d) - a[I] * mx
    rb = np.where(bvalid, np.minimum(1, np.nan_to_num(nb) / 2000), 0)
    b = _lsq_diff(n, I, J, rhs_b, wp, bg_target - a * bmean_f, lam * np.maximum(rb, prior_floor))
    pstats = {'corr_p1': np.percentile(corr[ok], 1), 'corr_p50': np.median(corr[ok]),
              'weight_p1': np.percentile(wp[ok], 1), 'weight_p50': np.median(wp[ok])}
    return a, b, valid, pstats


def adjust_lut(b, a, dtype=np.uint8):
    """lookup table of y = b + a x, the same as the rendering: the extreme values are kept, the others clipped to [min+1, max-1]."""
    lo, hi = np.iinfo(dtype).min, np.iinfo(dtype).max
    lut = np.clip(b + a * np.arange(lo, hi + 1, dtype=np.float64), lo + 1, hi - 1).astype(dtype)
    lut[0], lut[-1] = lo, hi
    return lut


def preview_one_section(secname, src_dir, out_dir, b, a, fillval=255, ext='.png'):
    img = common.imread(storage.join_paths(src_dir, secname + ext), flag=cv2.IMREAD_GRAYSCALE)
    out = adjust_lut(b, a)[img]
    out[img == 0] = fillval
    common.imwrite(storage.join_paths(out_dir, secname + ext), out)


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="per-section intensity adjustment of the aligned stack")
    parser.add_argument("--worker", metavar="worker", type=int, default=1)
    parser.add_argument("--method", metavar="method", type=str, default='zmatch', help="zmatch | stats")
    parser.add_argument("--stats_mode", metavar="stats_mode", type=str, default='smooth', help="stats method: smooth | fixed")
    parser.add_argument("--mask_dir", metavar="mask_dir", type=str, default='default', help="default: align/intensity_masks")
    parser.add_argument("--out_dir", metavar="out_dir", type=str, default='default', help="default: align/intensity_adjust")
    parser.add_argument("--aligned_dir", metavar="aligned_dir", type=str, default='default',
                        help="aligned thumbnails (zmatch and preview), default: align/aligned_<thumbnail resolution>nm")
    parser.add_argument("--key", metavar="key", type=str, default='', help="file name substring the adjustment applies to")
    parser.add_argument("--bg_target", metavar="bg_target", type=float, default=BG_TARGET, help="background value after adjustment")
    parser.add_argument("--fg_target", metavar="fg_target", type=float, default=FG_TARGET,
                        help="foreground value after adjustment (stats smooth: average over the sections)")
    parser.add_argument("--min_pixels", metavar="min_pixels", type=int, default=MIN_PIXELS)
    parser.add_argument("--min_diff", metavar="min_diff", type=float, default=MIN_DIFF)
    parser.add_argument("--median_window", metavar="median_window", type=int, default=MEDIAN_WINDOW)
    parser.add_argument("--smooth_sigma", metavar="smooth_sigma", type=float, default=SMOOTH_SIGMA)
    parser.add_argument("--gain_min", metavar="gain_min", type=float, default=GAIN_RANGE[0])
    parser.add_argument("--gain_max", metavar="gain_max", type=float, default=GAIN_RANGE[1])
    parser.add_argument("--smooth_length", metavar="smooth_length", type=float, default=SMOOTH_LENGTH)
    parser.add_argument("--preview", action='store_true', help="also write the adjusted aligned thumbnails to <aligned_dir>_adjusted")
    parser.add_argument("--preview_fill", metavar="preview_fill", type=int, default=255, help="value outside the sections in the preview")
    parser.add_argument("--ext", metavar="ext", type=str, default='.png')
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args()
    root_dir = config.get_work_dir()
    align_dir = storage.join_paths(root_dir, 'align')
    thumb_dir = storage.join_paths(root_dir, 'thumbnail_align', 'thumbnails')
    mask_dir = storage.join_paths(align_dir, 'intensity_masks') if args.mask_dir.lower() == 'default' else args.mask_dir
    out_dir = storage.join_paths(align_dir, 'intensity_adjust') if args.out_dir.lower() == 'default' else args.out_dir
    if args.aligned_dir.lower() == 'default':
        thumb_res = config.thumbnail_resolution()
        if thumb_res % 1 == 0:
            thumb_res = int(thumb_res)
        aligned_dir = storage.join_paths(align_dir, f'aligned_{thumb_res}nm')
    else:
        aligned_dir = args.aligned_dir
    method = args.method.lower()
    if method not in ('zmatch', 'stats'):
        raise ValueError(f'unsupported method {args.method}')
    # z order as in the aligned rendering
    tform_dir = storage.join_paths(align_dir, 'tform')
    tlist = sorted(storage.list_folder_content(storage.join_paths(tform_dir, '*.h5')))
    if len(tlist) == 0:
        tlist = sorted(storage.list_folder_content(storage.join_paths(mask_dir, '*.png')))
    tlist, _ = common.rearrange_section_order(tlist, storage.join_paths(root_dir, 'section_order.txt'))
    secnames = [os.path.splitext(os.path.basename(s))[0] for s in tlist]
    missing = [s for s in secnames if not storage.file_exists(storage.join_paths(mask_dir, s + '.png'))]
    if len(missing) > 0:
        print(f'{len(missing)} sections without masks in {mask_dir} (e.g. {missing[:3]}): run tools/intensity_masks.py first.')
        exit(1)
    if (method == 'zmatch') or args.preview:
        missing = [s for s in secnames if not storage.file_exists(storage.join_paths(aligned_dir, s + args.ext))]
        if len(missing) > 0:
            print(f'{len(missing)} sections without aligned thumbnails in {aligned_dir} (e.g. {missing[:3]}): run\n'
                  '    python tools/render_aligned_thumbnails.py --worker <N> --xmin 0 --ymin 0 --xmax <W> --ymax <H>\n'
                  'with the canvas at the thumbnail resolution, or set --aligned_dir.')
            exit(1)
    storage.makedirs(out_dir, exist_ok=True)
    tfunc = partial(section_stats, thumb_dir=thumb_dir, mask_dir=mask_dir, ext=args.ext)
    stats = {}
    for res in submit_to_workers(tfunc, args=[(s,) for s in secnames], num_workers=args.worker):
        stats[res[0]] = res[1:]
    bmean, fmean, nb, nf = [np.array([stats[s][k] for s in secnames], dtype=np.float64) for k in range(4)]
    hists = np.stack([stats[s][4] for s in secnames], axis=0)
    kwargs = {'bg_target': args.bg_target, 'fg_target': args.fg_target,
              'min_pixels': args.min_pixels, 'min_diff': args.min_diff}
    if method == 'stats':
        a, b, valid = stats_adjust(bmean, fmean, nf, mode=args.stats_mode.lower(),
                                   median_window=args.median_window, smooth_sigma=args.smooth_sigma,
                                   gain_range=(args.gain_min, args.gain_max), **kwargs)
    else:
        I, J, H = build_pair_hists(secnames, aligned_dir, ext=args.ext, num_workers=args.worker)
        a, b, valid, pstats = zmatch_adjust(I, J, H, bmean, fmean, nb, nf, smooth_length=args.smooth_length, **kwargs)
        del H
        print('section pairs: ' + ', '.join(f'{k} {v:.3g}' for k, v in pstats.items()))
    # fraction of the imaged thumbnail pixels pushed out of [1, 254] (full resolution saturates more)
    gs = np.arange(256)
    mapped = b[:, None] + a[:, None] * gs[None, :]
    npx = np.maximum(hists.sum(axis=1), 1)
    sat_lo = (hists * (mapped < 1)).sum(axis=1) / npx
    sat_hi = (hists * (mapped > 254)).sum(axis=1) / npx
    bg_adj = b + a * bmean
    fg_adj = b + a * fmean
    for k, secname in enumerate(secnames):
        with storage.File(storage.join_paths(out_dir, secname + '.json'), 'w') as f:
            json.dump({'__TYPE__': 'BRIGHTNESS_CONTRAST_ADJUST', args.key: [float(b[k]), float(a[k])]}, f)
    with storage.File(storage.join_paths(out_dir, 'intensity_adjust.csv'), 'w') as f:
        f.write('section,bg_mean,fg_mean,bg_pixels,fg_pixels,fg_valid,contrast,brightness,bg_adjusted,fg_adjusted,saturated_low,saturated_high\n')
        for k, secname in enumerate(secnames):
            f.write(f'{secname},{bmean[k]:.3f},{fmean[k]:.3f},{int(nb[k])},{int(nf[k])},{int(valid[k])},'
                    f'{a[k]:.5f},{b[k]:.3f},{bg_adj[k]:.2f},{fg_adj[k]:.2f},{sat_lo[k]:.5f},{sat_hi[k]:.5f}\n')
    print(f'{len(secnames)} sections ({args.method}); {np.count_nonzero(~valid)} without reliable foreground; '
          f'contrast p0/p50/p100 {np.min(a):.3f} / {np.median(a):.3f} / {np.max(a):.3f}')
    print('adjusted background p1/p50/p99: ' + ' / '.join(f'{v:.1f}' for v in np.nanpercentile(bg_adj, [1, 50, 99])) +
          '; foreground: ' + ' / '.join(f'{v:.1f}' for v in np.nanpercentile(fg_adj, [1, 50, 99])))
    print('saturated thumbnail pixels (low / high), median: '
          f'{np.median(sat_lo):.4f} / {np.median(sat_hi):.4f}, 99th pct: '
          f'{np.percentile(sat_lo, 99):.4f} / {np.percentile(sat_hi, 99):.4f}')
    if args.preview:
        preview_dir = aligned_dir.rstrip('/') + '_adjusted'
        storage.makedirs(preview_dir, exist_ok=True)
        jobs = [(s, aligned_dir, preview_dir, float(b[k]), float(a[k])) for k, s in enumerate(secnames)]
        for _ in submit_to_workers(preview_one_section, args=jobs,
                                   kwargs=[{'fillval': args.preview_fill, 'ext': args.ext}] * len(jobs),
                                   num_workers=args.worker):
            pass
        print(f'preview written to {preview_dir}')
    print('finished.')
