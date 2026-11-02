import argparse
import cv2
from functools import partial
import numpy as np
import os
import time

from feabas.concurrent import submit_to_workers
from feabas import common, config, storage
"""
foreground (tissue) / background (resin) masks of the unaligned thumbnails, to
sample the brightness and contrast of each section for intensity normalization
(tools/compute_intensity_adjust.py).
Output: one uint8 image per section, same size as the thumbnail, default
folder align/intensity_masks:
    OUTSIDE_VAL (0): outside the imaged area
    FOREGROUND_VAL (100): (solid) tissue
    UNCERTAIN_VAL (200): neither
    BACKGROUND_VAL (255): resin
Only the foreground and background pixels are used downstream, so a mask can
be edited by hand, e.g. by painting UNCERTAIN_VAL over doubtful areas. Existing
masks are not overwritten unless --overwrite.
Thresholds are learned per section from two weakly labeled regions:
  background: a ring of RING_WIDTH pixels inside the edge of the imaged area
      (material mask != 255), which is mostly resin.
  foreground: the areas covered by fine alignment matches. Read from
      align/matches/match_cover (visualize_align_match_coverage.py) if there,
      otherwise rasterized from the match files of align/matches (or of
      thumbnail_align/matches if there are no fine matches).
Pixels are described by their smoothed intensity and their local standard
deviation (texture). Resin is flat and bright, tissue is textured or darker.
The texture threshold is the geometric mean of the ring and cover median local
std, the intensity threshold the mean of the ring and cover median intensities.
Textured/dark pixels are closed into the foreground, pruned of the parts away
from the matches (resin streaks, debris, seams); flat/bright pixels away from
the foreground are the background. Holes in the tissue (vacuoles) are not
filled. Finally only the "solid" tissue is kept as foreground: the pixels
darker than the background mean by more than solid_k background stds; the
rest of the tissue (resin-like pixels in vacuoles, between cells, in the
margin of the match windows) becomes uncertain.
"""

OUTSIDE_VAL = 0
FOREGROUND_VAL = 100
UNCERTAIN_VAL = 200
BACKGROUND_VAL = 255

RING_WIDTH = 8          # width of the border ring of the imaged area used as background sample
EDGE_SKIP = 2           # pixels right at the imaged edge left out of the ring (stitching seams / zero fill)
STD_WINDOW = 5          # window size of the local standard deviation
SMOOTH_SIGMA = 1.0      # gaussian sigma of the smoothed intensity
STD_WEIGHT = 0.5        # texture threshold: ring_std^(1-w) * cover_std^w (geometric interpolation)
INTENSITY_WEIGHT = 0.5  # intensity threshold: ring_I + w * (cover_I - ring_I)
CLOSE_RADIUS = 3        # closing radius joining textured pixels into tissue
COVER_RADIUS = 15       # with a match cover: drop foreground farther than this from any match (0: off)
MIN_COMPONENT_COVER = 0.1   # with a match cover: drop foreground components less covered than this
MIN_FG_AREA = 20        # smallest foreground component kept (thumbnail pixels)
OPEN_RADIUS = 2         # without a match cover: opening radius removing scratches and debris
MIN_FG_AREA_NO_COVER = 400  # without a match cover: smallest foreground component kept
BG_MARGIN = 3           # gap between the background and the foreground
MIN_BG_AREA = 100       # smallest background component kept
SOLID_K = 3.0           # solid tissue: darker than background mean - SOLID_K * background std (< 0: keep all the tissue)
MIN_BG_PIXELS = 500     # fewer background pixels than this: solid threshold from the border ring instead
COVER_DIFF = 40         # channel difference in the match cover image marking a matched pixel
MIN_COVER_PIXELS = 500  # fewer cover pixels than this: thresholds from FALLBACK_* instead
FALLBACK_STD_RATIO = 4.0         # texture threshold = ring_std * ratio
FALLBACK_INTENSITY_DROP = 15.0   # intensity threshold = ring_I - drop


def _disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _remove_small(mask, min_area):
    if min_area <= 0:
        return mask
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    return keep[lbl]


def load_section(thumb_path, mask_path):
    img = common.imread(thumb_path, flag=cv2.IMREAD_GRAYSCALE)
    H, W = img.shape
    imaged = img > 0
    if storage.file_exists(mask_path):
        msk = common.imread(mask_path, flag=cv2.IMREAD_GRAYSCALE)
        # material masks may be padded beyond the thumbnail: crop top-left
        h, w = min(H, msk.shape[0]), min(W, msk.shape[1])
        m = np.full((H, W), 255, dtype=np.uint8)
        m[:h, :w] = msk[:h, :w]
        imaged &= (m != 255)
    return img, imaged


def _match_points(match_name, secname, target_resolution, delimiter):
    """matching points of section secname in a match file, at target_resolution."""
    from feabas.spatial import scale_coordinates
    H5File = storage.h5file_class()
    with H5File(match_name, 'r') as f:
        xy0 = f['xy0'][()]
        xy1 = f['xy1'][()]
        resolution = f['resolution'][()]
        if ('name0' in f) and ('name1' in f):
            name0 = common.numpy_to_str_ascii(f['name0'][()])
        else:
            name0 = os.path.splitext(os.path.basename(match_name))[0].split(delimiter)[0]
    if isinstance(resolution, np.ndarray):
        resolution = resolution.item()
    xy = xy0 if name0 == secname else xy1
    return scale_coordinates(xy, resolution / target_resolution)


def load_cover(secname, shape, imaged, cover_path=None, match_list=None, **kwargs):
    """
    pixels covered by matches: from the match cover image if it exists,
    otherwise rasterized from the match files in match_list.
    Kwargs:
        resolution (float): resolution of the thumbnail.
        blksz (int): size of the square drawn around each matching point.
        delimiter (str): section name delimiter in the match file names.
    Return:
        cover (H x W bool ndarray), or None.
    """
    H, W = shape
    cover = None
    if (cover_path is not None) and storage.file_exists(cover_path):
        cov = common.imread(cover_path, flag=cv2.IMREAD_COLOR).astype(np.int16)
        if cov.shape[:2] == (H, W):
            # match cover: base = 0.7 * img, the other two channels + 100 where matched
            cover = ((cov[..., 1] - cov[..., 0]) > COVER_DIFF) | ((cov[..., 2] - cov[..., 0]) > COVER_DIFF)
    if (cover is None) and match_list:
        resolution = kwargs['resolution']
        blksz = max(1, int(kwargs.get('blksz', 1)))
        delimiter = kwargs.get('delimiter', '__to__')
        cover = np.zeros((H, W), dtype=np.uint8)
        for mname in match_list:
            pts = _match_points(mname, secname, resolution, delimiter)
            if pts.size == 0:
                continue
            x = np.round(pts[:, 0].clip(0, W - 1)).astype(np.int32)
            y = np.round(pts[:, 1].clip(0, H - 1)).astype(np.int32)
            cover[y, x] = 1
        cover = cv2.dilate(cover, np.ones((blksz, blksz), np.uint8)) > 0
    if cover is not None:
        cover &= imaged
    return cover


def pixel_features(img, **kwargs):
    std_window = kwargs.get('std_window', STD_WINDOW)
    smooth_sigma = kwargs.get('smooth_sigma', SMOOTH_SIGMA)
    f = img.astype(np.float32)
    Is = cv2.GaussianBlur(f, (0, 0), smooth_sigma)
    mu = cv2.blur(f, (std_window, std_window))
    S = np.sqrt(np.maximum(cv2.blur(f * f, (std_window, std_window)) - mu * mu, 0))
    return Is, S


def border_ring(imaged, **kwargs):
    ring_width = kwargs.get('ring_width', RING_WIDTH)
    edge_skip = kwargs.get('edge_skip', EDGE_SKIP)
    outside = (~imaged).astype(np.uint8)
    dil = lambda r: cv2.dilate(outside, _disk(r), borderType=cv2.BORDER_CONSTANT, borderValue=1) > 0
    ring = dil(ring_width) & imaged
    if edge_skip > 0:
        ring &= ~dil(edge_skip)
    return ring


def section_thresholds(Is, S, ring, cover, **kwargs):
    """
    Return:
        std_thresh, intensity_thresh (float), and a dict of the statistics.
    """
    std_weight = kwargs.get('std_weight', STD_WEIGHT)
    intensity_weight = kwargs.get('intensity_weight', INTENSITY_WEIGHT)
    ring_S = max(float(np.median(S[ring])), 0.1)
    ring_I = float(np.median(Is[ring]))
    stats = {'ring_std': ring_S, 'ring_I': ring_I}
    if (cover is not None) and (np.count_nonzero(cover) >= kwargs.get('min_cover_pixels', MIN_COVER_PIXELS)):
        cover_S = max(float(np.median(S[cover])), ring_S)
        cover_I = min(float(np.median(Is[cover])), ring_I)
        stats.update({'cover_std': cover_S, 'cover_I': cover_I})
        std_thresh = ring_S ** (1 - std_weight) * cover_S ** std_weight
        intensity_thresh = ring_I + intensity_weight * (cover_I - ring_I)
    else:
        std_thresh = ring_S * kwargs.get('fallback_std_ratio', FALLBACK_STD_RATIO)
        intensity_thresh = ring_I - kwargs.get('fallback_intensity_drop', FALLBACK_INTENSITY_DROP)
    return std_thresh, intensity_thresh, stats


def fg_bg_masks(img, imaged, cover=None, **kwargs):
    """
    Args:
        img (H x W uint8 ndarray): thumbnail.
        imaged (H x W bool ndarray): imaged area.
        cover (H x W bool ndarray): pixels covered by matches, or None.
    Return:
        fg, bg, ring (H x W bool ndarray), and a dict of the thresholds and statistics.
    """
    close_radius = kwargs.get('close_radius', CLOSE_RADIUS)
    bg_margin = kwargs.get('bg_margin', BG_MARGIN)
    Is, S = pixel_features(img, **kwargs)
    ring = border_ring(imaged, **kwargs)
    if not np.any(ring):
        ring = imaged
    std_thresh, intensity_thresh, stats = section_thresholds(Is, S, ring, cover, **kwargs)
    stats.update({'std_thresh': std_thresh, 'intensity_thresh': intensity_thresh})
    fg = (((S > std_thresh) | (Is < intensity_thresh)) & imaged).astype(np.uint8)
    use_cover = (cover is not None) and np.any(cover)
    if not use_cover:
        # no matches to tell tissue from scratches and debris: rely on shape and size
        open_radius = kwargs.get('open_radius', OPEN_RADIUS)
        if open_radius > 0:
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, _disk(open_radius))
    if close_radius > 0:
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, _disk(close_radius), borderType=cv2.BORDER_REPLICATE)
    fg = (fg > 0) & imaged
    if use_cover:
        cover_radius = kwargs.get('cover_radius', COVER_RADIUS)
        if cover_radius > 0:
            # cut resin streaks and debris attached to the tissue
            fg &= cv2.dilate(cover.astype(np.uint8), _disk(cover_radius)) > 0
        n, lbl, cstats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
        area = cstats[:, cv2.CC_STAT_AREA]
        covered = np.bincount(lbl[cover], minlength=n)
        keep = (covered >= kwargs.get('min_component_cover', MIN_COMPONENT_COVER) * area) & (area >= kwargs.get('min_fg_area', MIN_FG_AREA))
        keep[0] = False
        fg = keep[lbl]
    else:
        fg = _remove_small(fg, kwargs.get('min_fg_area_no_cover', MIN_FG_AREA_NO_COVER))
    bg = (S <= std_thresh) & (Is >= intensity_thresh) & imaged
    if bg_margin > 0:
        bg &= ~(cv2.dilate(fg.astype(np.uint8), _disk(bg_margin)) > 0)
    else:
        bg &= ~fg
    bg = _remove_small(bg, kwargs.get('min_bg_area', MIN_BG_AREA))
    if cover is not None:
        stats['cover_in_fg'] = float(np.count_nonzero(cover & fg)) / max(1, np.count_nonzero(cover))
        stats['cover_in_bg'] = float(np.count_nonzero(cover & bg)) / max(1, np.count_nonzero(cover))
    stats['ring_in_fg'] = float(np.count_nonzero(ring & fg)) / max(1, np.count_nonzero(ring))
    return fg, bg, ring, stats


def solid_tissue(img, fg, bg, ring, **kwargs):
    """
    the part of the foreground clearly darker than the resin: below the
    background mean by more than solid_k background stds (from the border
    ring if there are too few background pixels).
    Return:
        solid (H x W bool ndarray), and a dict of the statistics.
    """
    solid_k = kwargs.get('solid_k', SOLID_K)
    if solid_k < 0:
        return fg, {}
    ref = bg if np.count_nonzero(bg) >= kwargs.get('min_bg_pixels', MIN_BG_PIXELS) else ring
    v = img[ref].astype(np.float32)
    if v.size == 0:
        return fg, {}
    bg_mean, bg_std = float(v.mean()), float(v.std())
    solid = fg & (img < bg_mean - solid_k * bg_std)
    stats = {'bg_mean': bg_mean, 'bg_std': bg_std,
             'solid_frac': float(np.count_nonzero(solid)) / max(1, np.count_nonzero(fg))}
    return solid, stats


def process_one_section(secname, thumb_dir, mask_dir, out_dir, cover_path=None, match_list=None,
                        cover_kwargs=None, ext='.png', overlay_dir=None, overwrite=False, **kwargs):
    t0 = time.time()
    outname = storage.join_paths(out_dir, secname + '.png')
    if storage.file_exists(outname) and (not overwrite):
        return None
    img, imaged = load_section(storage.join_paths(thumb_dir, secname + ext),
                               storage.join_paths(mask_dir, secname + '.png'))
    cover = load_cover(secname, img.shape, imaged, cover_path=cover_path, match_list=match_list,
                       **({} if cover_kwargs is None else cover_kwargs))
    fg, bg, ring, stats = fg_bg_masks(img, imaged, cover=cover, **kwargs)
    solid, solid_stats = solid_tissue(img, fg, bg, ring, **kwargs)
    stats.update(solid_stats)
    out = np.full(img.shape, OUTSIDE_VAL, dtype=np.uint8)
    out[imaged] = UNCERTAIN_VAL
    out[solid] = FOREGROUND_VAL
    out[bg] = BACKGROUND_VAL
    common.imwrite(outname, out)
    if overlay_dir is not None:
        # gray thumbnail: foreground tinted red, background tinted blue, match cover outlined green
        base = (img.astype(np.float32) * 0.7)
        vis = np.stack((base, base, base), axis=-1)   # BGR
        vis[solid, 2] += 80
        vis[bg, 0] += 80
        if cover is not None:
            edge = cover & ~(cv2.erode(cover.astype(np.uint8), _disk(1)) > 0)
            vis[edge, 1] = 255
        vis[~imaged] = 0
        common.imwrite(storage.join_paths(overlay_dir, secname + '.png'), vis.clip(0, 255).astype(np.uint8))
    stats['time'] = time.time() - t0
    return secname, stats


def match_index(match_dir, delimiter):
    """section name -> list of its match files in match_dir."""
    out = {}
    for mname in storage.list_folder_content(storage.join_paths(match_dir, '*.h5')):
        bname = os.path.splitext(os.path.basename(mname))[0]
        secnames = bname.split(delimiter)
        for s in (secnames[0], secnames[-1]):
            out.setdefault(s, []).append(mname)
    return out


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="foreground / background masks of the unaligned thumbnails for intensity normalization")
    parser.add_argument("--worker", metavar="worker", type=int, default=1)
    parser.add_argument("--start", metavar="start", type=int, default=config.default_task_start())
    parser.add_argument("--step", metavar="step", type=int, default=config.default_task_step())
    parser.add_argument("--stop", metavar="stop", type=int, default=0)
    parser.add_argument("--sections", metavar="sections", type=str, nargs='*', help="section names (default: all thumbnails)")
    parser.add_argument("--cover_dir", metavar="cover_dir", type=str, default='default',
                        help="match cover image folder (default: align/matches/match_cover; sections without one are rasterized from the match files), 'none' to not use matches")
    parser.add_argument("--out_dir", metavar="out_dir", type=str, default='default', help="output folder (default: align/intensity_masks)")
    parser.add_argument("--overlay", action='store_true', help="also write overlays to <out_dir>_overlay")
    parser.add_argument("--overwrite", action='store_true', help="overwrite existing masks (by default kept, e.g. after manual edits)")
    parser.add_argument("--ext", metavar="ext", type=str, default='.png')
    for name, v in (('ring_width', RING_WIDTH), ('edge_skip', EDGE_SKIP), ('std_window', STD_WINDOW),
                    ('smooth_sigma', SMOOTH_SIGMA), ('std_weight', STD_WEIGHT), ('intensity_weight', INTENSITY_WEIGHT),
                    ('close_radius', CLOSE_RADIUS), ('cover_radius', COVER_RADIUS),
                    ('min_component_cover', MIN_COMPONENT_COVER), ('min_fg_area', MIN_FG_AREA),
                    ('open_radius', OPEN_RADIUS), ('min_fg_area_no_cover', MIN_FG_AREA_NO_COVER),
                    ('bg_margin', BG_MARGIN), ('min_bg_area', MIN_BG_AREA), ('solid_k', SOLID_K)):
        parser.add_argument('--' + name, metavar=name, type=type(v), default=v)
    return parser.parse_args(args)


KWARG_NAMES = ('ring_width', 'edge_skip', 'std_window', 'smooth_sigma', 'std_weight', 'intensity_weight',
               'close_radius', 'cover_radius', 'min_component_cover', 'min_fg_area', 'open_radius',
               'min_fg_area_no_cover', 'bg_margin', 'min_bg_area', 'solid_k')


if __name__ == '__main__':
    args = parse_args()
    Nthreads = config.get_numpy_thread()
    cv2.setNumThreads(Nthreads)
    root_dir = config.get_work_dir()
    thumbnail_dir = storage.join_paths(root_dir, 'thumbnail_align')
    thumb_dir = storage.join_paths(thumbnail_dir, 'thumbnails')
    mask_dir = storage.join_paths(thumbnail_dir, 'material_masks')
    align_dir = storage.join_paths(root_dir, 'align')
    thumbnail_configs = config.thumbnail_configs()
    delimiter = thumbnail_configs.get('alignment', {}).get('match_name_delimiter', '__to__')
    thumb_mip = thumbnail_configs.get('thumbnail_mip_level', 6)
    thumb_res = config.mip0_resolution() * (2 ** thumb_mip)
    use_matches = args.cover_dir.lower() != 'none'
    if args.cover_dir.lower() in ('default', 'none'):
        cover_dir = storage.join_paths(align_dir, 'matches', 'match_cover')
    else:
        cover_dir = args.cover_dir
    if args.out_dir.lower() == 'default':
        out_dir = storage.join_paths(align_dir, 'intensity_masks')
    else:
        out_dir = args.out_dir
    storage.makedirs(out_dir, exist_ok=True)
    overlay_dir = None
    if args.overlay:
        overlay_dir = out_dir.rstrip('/') + '_overlay'
        storage.makedirs(overlay_dir, exist_ok=True)
    if args.sections:
        secnames = sorted(args.sections)
    else:
        tlist = sorted(storage.list_folder_content(storage.join_paths(thumb_dir, '*' + args.ext)))
        secnames = [os.path.basename(s)[:-len(args.ext)] for s in tlist]
        stp_idx = args.stop if args.stop != 0 else None
        secnames = secnames[slice(args.start, stp_idx, args.step)]
    # matches to rasterize the cover of the sections without a match cover image
    matches, cover_kwargs = {}, {}
    if use_matches:
        matches = match_index(storage.join_paths(align_dir, 'matches'), delimiter)
        if len(matches) > 0:
            align_configs = config.align_configs()
            align_mip = align_configs.get('matching', {}).get('working_mip_level', 2)
            spacing = align_configs.get('matching', {}).get('matcher_config', {}).get('spacings', [100])[-1]
            blksz = int(np.ceil(spacing * 2 ** (align_mip - thumb_mip)))
        else:
            matches = match_index(storage.join_paths(thumbnail_dir, 'matches'), delimiter)
            blksz = int(thumbnail_configs.get('alignment', {}).get('block_matching', {}).get('spacings', [100])[-1])
            if len(matches) > 0:
                print('no fine alignment matches: use the thumbnail matches.')
        cover_kwargs = {'resolution': thumb_res, 'blksz': blksz, 'delimiter': delimiter}
    kwargs = {k: getattr(args, k) for k in KWARG_NAMES}
    tfunc = partial(process_one_section, thumb_dir=thumb_dir, mask_dir=mask_dir, out_dir=out_dir,
                    cover_kwargs=cover_kwargs, ext=args.ext, overlay_dir=overlay_dir,
                    overwrite=args.overwrite, **kwargs)
    tasks = [{'secname': s, 'cover_path': (storage.join_paths(cover_dir, s + '.png') if use_matches else None),
              'match_list': matches.get(s, None)} for s in secnames]
    for res in submit_to_workers(tfunc, kwargs=tasks, num_workers=args.worker):
        if res is None:
            continue
        secname, stats = res
        print(secname + ': ' + ', '.join(f'{k} {v:.3g}' for k, v in stats.items()))
    print('finished.')
