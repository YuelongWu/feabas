"""
functions specific to Zeiss MultiSEM data.
"""
from collections import defaultdict
from functools import lru_cache
import os
import numpy as np

from feabas import constant as const
from feabas import caching, mesh, optimizer

def mfovids_from_relpaths(relpaths):
    mfovs = [int(s.split('/')[0]) for s in relpaths]
    return mfovs


def mfovids_beamids_from_filenames(filenames):
    # file names follow pattern e.g.
    #   001_000001_001_2022-04-26T1308251544560.bmp
    #   convert beam ids to 0-index
    mfovs = []
    beams = []
    for fname in filenames:
        bname = os.path.basename(fname)
        substrs = bname.split('_')
        mfovs.append(int(substrs[1]))
        beams.append(int(substrs[2])-1)
    return np.array(mfovs), np.array(beams)


@lru_cache(maxsize=2)
def beam_coordinate_vectors(beam_num=91):
    # the coordinates of each beam in each mFov when the unit base vectors are:
    #   e0: beam 1 -> beam 2; e1:  beam 1 -> beam3;
    beam_coord = np.zeros((beam_num, 2))
    ring_id = np.ceil((-3 + (12*(np.arange(1, beam_num+1))-3)**0.5)/6)
    for b in range(1, beam_num):
        if ring_id[b] != ring_id[b-1]:
            crnt_pt = np.array([ring_id[b], 0])
            beam_coord[b,:] = crnt_pt
            t = 0
            continue
        tn = np.floor(t / ring_id[b])
        if tn == 0:
            vec = np.array([-1, 1])
        elif tn == 1:
            vec = np.array([-1, 0])
        elif tn == 2:
            vec = np.array([0, -1])
        elif tn == 3:
            vec = np.array([1, -1])
        elif tn == 4:
            vec = np.array([1, 0])
        else:
            vec = np.array([0, 1])
        crnt_pt = crnt_pt + vec
        beam_coord[b,:] = crnt_pt
        t += 1
    return beam_coord


@lru_cache(maxsize=2)
def beam_neighbors(beam_num=91):
    beam_coord = beam_coordinate_vectors(beam_num=beam_num)
    coord_diff0 = beam_coord[:, 0].reshape(1, -1) - beam_coord[:,0].reshape(-1,1)
    coord_diff1 = beam_coord[:, 1].reshape(1, -1) - beam_coord[:,1].reshape(-1,1)
    dis = coord_diff0 ** 2 + coord_diff1 ** 2 + coord_diff0 * coord_diff1
    nb_indx = (dis == 1) & ((coord_diff0 * 0.5 + coord_diff1) > 0)
    beam_id0, beam_id1 = np.nonzero(nb_indx)
    diff0 = coord_diff0[beam_id0, beam_id1]
    diff1 = coord_diff1[beam_id0, beam_id1]
    nbs = {(b0,b1): (d0,d1) for b0, b1, d0, d1 in zip(beam_id0, beam_id1, diff0, diff1)}
    return nbs


def estimate_beam_pattern(matches):
    """
    matches (list): [(beamid0, beamid1), (dxy, weight)]. only contains intra-mFoV matches
    """
    regularization_weight = 0.1
    beam_ids = np.array([s[0] for s in matches])
    if beam_ids.max(axis=None) < 61:
        beam_num = 61
    else:
        beam_num = 91
    neighbors = beam_neighbors(beam_num=beam_num)
    sfov_dxy_depo = defaultdict(list)
    base_vec_depo = defaultdict(list)
    for mm in matches:
        bids, mtches = mm
        bid0, bid1 = bids
        dxy, wt = mtches
        if (bid1, bid0) in neighbors:
            bid0, bid1 = bid1, bid0
            dxy = -dxy
        elif (bid0, bid1) not in neighbors:
            continue
        dxy = dxy.ravel()
        sfov_dxy_depo[(bid0, bid1)].append([dxy[0], dxy[1], wt])
        base_vec_depo[neighbors[(bid0, bid1)]].append([dxy[0], dxy[1], wt])
    if len(base_vec_depo) < 2: # not enough matches to estimate a pattern
        return None
    base_lhs = []
    base_rhs = []
    for key, val in base_vec_depo.items():
        val = np.array(val)
        idx = val[:, -1] >= np.median(val[:, -1])
        val = val[idx]
        dxy = val[:,:2]
        wt = val[:,-1].reshape(-1, 1)
        base_rhs.append(np.sum(dxy * wt, axis=0))
        base_lhs.append(np.sum(wt) * np.array(key))
    base_vecs = np.linalg.lstsq(np.array(base_lhs), base_rhs, rcond=None)[0]
    sfov_lhs = []
    sfov_rhs = []
    min_wt = None
    for bids, val in sfov_dxy_depo.items():
        bid0, bid1 = bids
        val = np.array(val)
        idx = val[:, -1] >= np.median(val[:, -1])
        if (np.sum(idx) < 3) or (np.sum(val[:, -1]) == 0):
            continue
        val = val[idx]
        dxy = val[:,:2]
        wt = val[:,-1].reshape(-1, 1)
        sfov_rhs.append(np.sum(dxy * wt, axis=0, keepdims=True))
        cf = np.zeros(beam_num)
        swt = np.sum(wt)
        cf[bid0] = -swt
        cf[bid1] = swt
        sfov_lhs.append(cf.reshape(1,-1))
        if (min_wt is None) or (min_wt > swt):
            min_wt = swt
    if min_wt is None:
        return None
    coord_from_base = beam_coordinate_vectors(beam_num=beam_num) @ base_vecs
    reg_wt = min_wt * regularization_weight
    sfov_lhs.append(reg_wt * np.eye(beam_num))
    sfov_rhs.append(reg_wt * coord_from_base)
    A = np.concatenate(sfov_lhs, axis=0)
    b = np.concatenate(sfov_rhs, axis=0)
    sfov_pattern = np.linalg.lstsq(A, b, rcond=None)[0]
    sfov_pattern = sfov_pattern - np.mean(sfov_pattern, axis=0)
    return sfov_pattern


def filter_links_from_sfov_pattern(stitcher, **kwargs):
    target_gear = kwargs.get('target_gear', const.MESH_GEAR_FIXED)
    residue_threshold = kwargs.get('residue_threshold', None)
    num_disabled = 0
    cost0 = (0, 0)
    if residue_threshold is None:
        return num_disabled, cost0
    elif residue_threshold <= 1:
        overlap_width = np.median(stitcher.overlap_widths)
        residue_threshold = residue_threshold * overlap_width
        kwargs['residue_threshold'] = residue_threshold
    mfovids, beamids = mfovids_beamids_from_filenames(stitcher.imgrelpaths)
    _, mfovids = np.unique(mfovids, return_inverse=True)
    if stitcher._optimizer is None:
        stitcher.initialize_optimizer()
    sfov_matches = []
    mfov_matches = {}
    dmax = np.zeros(2)
    for lnk in stitcher._optimizer.links:
        wtsum = lnk.weight_sum
        if wtsum == 0:
            continue
        uid0, uid1 = lnk.uids
        mid0, mid1 = mfovids[int(uid0)], mfovids[int(uid1)]
        bid0, bid1 = beamids[int(uid0)], beamids[int(uid1)]
        dxy = -np.median(lnk.dxy(gear=(const.MESH_GEAR_INITIAL, const.MESH_GEAR_INITIAL), use_mask=True), axis=0)
        if mid0 == mid1:
            sfov_matches.append(((bid0, bid1), (dxy, wtsum)))
        else:
            mfov_matches[((mid0, mid1), (bid0, bid1))] = (dxy, wtsum)
            dmax = np.maximum(dmax, np.abs(dxy))
    init_offset = stitcher._init_offset
    sfov_pattern = estimate_beam_pattern(sfov_matches)
    if sfov_pattern is None:
        return num_disabled, cost0
    mxy_min = sfov_pattern.min(axis=0) - dmax - 100
    mxy_max = sfov_pattern.max(axis=0) + dmax + 100
    mbbox = np.concatenate((mxy_min, mxy_max), axis=None)
    meshsz = np.max(mxy_max-mxy_min, axis=None) + 10
    M0 = mesh.Mesh.from_bbox(mbbox, cartesian=True, mesh_size=meshsz,
                             resolution=stitcher.resolution, max_aspect_ratio=1)
    mfov_meshes = []
    default_caches = {}
    for gear in const.MESH_GEARS:
        default_caches[gear] = defaultdict(lambda: caching.CacheFIFO(maxlen=None))
    for mid in range(np.max(mfovids)+1):
        idx = mfovids == mid
        xy_avg_local = np.mean(sfov_pattern[beamids[idx]], axis=0)
        xy_avg_global = np.mean(init_offset[idx], axis=0)
        Mm = M0.copy(override_dict={'uid':mid})
        mfov_offset = xy_avg_global-xy_avg_local
        Mm.apply_translation(mfov_offset, gear=const.MESH_GEAR_FIXED)
        for gear in const.MESH_GEARS:
            M0.set_default_cache(cache=default_caches[gear], gear=gear)
        mfov_meshes.append(Mm)
    mfov_optimizer = optimizer.SLM(mfov_meshes)
    for key, mtch in mfov_matches.items():
        mids, bids = key
        dxy_0, wt = mtch
        xy0 = (sfov_pattern[bids[0]] + dxy_0).reshape(-1,2)
        xy1 = (sfov_pattern[bids[1]]).reshape(-1,2)
        wt = np.array(wt).ravel()
        mfov_optimizer.add_link_from_coordinates(mids[0], mids[1], xy0, xy1,
                                                 weight=wt, check_duplicates=False)
    _, cost0 = mfov_optimizer.optimize_translation_w_filtering(**kwargs)
    
    mfov_offsets = [M.estimate_translation(gear=(const.MESH_GEAR_INITIAL, const.MESH_GEAR_FIXED)) for M in mfov_optimizer.meshes]
    for mid, bid, M in zip(mfovids, beamids, stitcher.meshes):
        txy = mfov_offsets[mid] + sfov_pattern[bid]
        M.set_translation(txy, gear=(const.MESH_GEAR_INITIAL, target_gear))
    for lnk in stitcher._optimizer.links:
        wtsum = lnk.weight_sum
        if wtsum == 0:
            continue
        dxy = lnk.dxy(gear=(target_gear, target_gear), use_mask=True)
        dxy_m = np.median(dxy, axis=0)
        dis = np.sqrt(np.sum(dxy_m**2))
        if dis > residue_threshold:
            lnk.disable()
            num_disabled += 1
    return num_disabled, cost0
