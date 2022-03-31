import numpy as np


def fit_affine(pts0, pts1, return_rigid=False):
    # pts0 = pts1 @ A
    pts0 = pts0.reshape(-1,2)
    pts1 = pts1.reshape(-1,2)
    assert pts0.shape[0] == pts1.shape[0]
    mm0 = pts0.mean(axis=0)
    mm1 = pts1.mean(axis=0)
    pts0 = pts0 - mm0
    pts1 = pts1 - mm1
    pts0_pad = np.insert(pts0, 2, 1, axis=-1)
    pts1_pad = np.insert(pts1, 2, 1, axis=-1)
    res = np.linalg.lstsq(pts1_pad, pts0_pad, rcond=None)
    r1 = np.linalg.matrix_rank(pts0_pad)
    A = res[0]
    r = min(res[2], r1)
    if r == 1:
        A = np.eye(3)
        A[-1,:2] = mm0 - mm1
    elif r == 2:
        pts0_rot90 = pts0[:,::-1] * np.array([1,-1])
        pts1_rot90 = pts1[:,::-1] * np.array([1,-1])
        pts0 = np.concatenate((pts0, pts0_rot90), axis=0)
        pts1 = np.concatenate((pts1, pts1_rot90), axis=0)
        pts0_pad = np.insert(pts0, 2, 1, axis=-1)
        pts1_pad = np.insert(pts1, 2, 1, axis=-1)
        res = np.linalg.lstsq(pts1_pad, pts0_pad, rcond=None)
        A = res[0]
    if return_rigid:
        u, _, vh = np.linalg.svd(A[:2,:2], compute_uv=True)
        R = A.copy()
        R[:2,:2] = u @ vh
        R[-1,:2] = R[-1,:2] + mm0 - mm1 @ R[:2,:2]
        R[:,-1] = np.array([0,0,1])
    A[-1,:2] = A[-1,:2] + mm0 - mm1 @ A[:2,:2]
    A[:,-1] = np.array([0,0,1])
    if return_rigid:
        return A, R
    else:
        return A