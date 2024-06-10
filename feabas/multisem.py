"""
functions specific to Zeiss MultiSEM data.
"""
from collections import defaultdict
from functools import lru_cache
import os
import numpy as np


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
    return mfovs, beams


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
    base_vecs = np.linalg.lstsq(np.array(base_lhs), base_rhs)[0]
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
    sfov_pattern = np.linalg.lstsq(A, b)[0]
    return sfov_pattern