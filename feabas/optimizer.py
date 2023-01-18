import numpy as np

from feabas import spatial
from feabas.constant import *


class Link:
    """
    class to represent the corresponding points between two meshes.
    """
    def __init__(self, mesh0, mesh1, tid0, tid1, B0, B1, weight=None):
        self.uids = [mesh0.uid, mesh1.uid]
        self.meshes = [mesh0, mesh1]
        self._tid0 = tid0
        self._B0 = B0
        self._tid1 = tid1
        self._B1 = B1
        self._weight = ((tid0 >= 0) & (tid1 >= 0)).astype(np.float32)
        if weight is not None:
            self._weight *= weight
        # weight that defines nonlinear behaviour
        self._residue_weight = np.ones_like(self._weight)
        self._weight_func = None
        self._mask = None
        self._disabled = False


    @classmethod
    def from_coordinates(cls, mesh0, mesh1, xy0, xy1, gear=(MESH_GEAR_INITIAL, MESH_GEAR_INITIAL), weight=None):
        tid0, B0 = mesh0.cart2bary(xy0, gear[0], tid=None)
        tid1, B1 = mesh1.cart2bary(xy1, gear[1], tid=None)
        indx = (tid0 >= 0) & (tid1 >= 0)
        if not np.all(indx):
            tid0 = tid0[indx]
            tid1 = tid1[indx]
            B0 = B0[indx]
            B1 = B1[indx]
        return cls(mesh0, mesh1, tid0, tid1, B0, B1, weight=weight)


    def equation_contrib(self, index_offsets, **kwargs):
        """computing the contribution needed to add to the FEM assembled matrix."""
        min_match_num = kwargs.get('min_match_num', 0)
        if not self.relevant(min_match_num):
            return None, None, None, None
        start_gear = kwargs.get('start_gear', MESH_GEAR_MOVING)
        targt_gear = kwargs.get('target_gear', MESH_GEAR_MOVING)
        num_matches = self.num_matches[-1]
        gears = [targt_gear if m.locked else start_gear for m in self.meshes]
        m_rht = self.dxy(gear=gears, use_mask=True)
        L_b = []  # barycentric coordinate matrices
        L_indx = [] # indices in the assembled system matrix
        if not self.meshes[0].locked:
            L_b.append(self.B0(use_mask=True))
            vidx = self.meshes[0].triangles[self.tid0(use_mask=True)]
            L_indx.append(2 * vidx + index_offsets[0]) # multiply by 2 for x, y
        if not self.meshes[1].locked:
            L_b.append(-self.B1(use_mask=True))
            vidx = self.meshes[1].triangles[self.tid1(use_mask=True)]
            L_indx.append(2 * vidx + index_offsets[1])
        B = np.concatenate(L_b, axis=-1)
        indx = np.concatenate(L_indx, axis=-1).reshape(num_matches, -1, 1)
        C = B.reshape(num_matches, -1, 1) @ B.reshape(num_matches, 1, -1)
        indx0 = np.tile(indx, (1,1, indx.shape[1]))
        indx1 = np.swapaxes(indx0, 1, 2)
        rht_x = B * (m_rht[:,0].reshape(-1,1))
        rht_y = B * (m_rht[:,1].reshape(-1,1))
        # left-hand side:
        indx0_lft = np.concatenate((indx0.ravel(), indx0.ravel()+1))
        indx1_lft = np.concatenate((indx1.ravel(), indx1.ravel()+1))
        wt  = self.weight(use_mask=True)
        if np.any(wt != 1):
            C = wt.reshape(-1,1,1) * C
            rht_x = wt.reshape(-1,1) * rht_x
            rht_y = wt.reshape(-1,1) * rht_y
        V_lft = np.concatenate((C.ravel(), C.ravel()))
        # right-hand side
        indx_rht = np.concatenate((indx.ravel(), indx.ravel()+1))
        V_rht = np.concatenate((rht_x.ravel(), rht_y.ravel()))
        return  V_lft, (indx0_lft, indx1_lft), V_rht, indx_rht


    def adjust_weight_from_residue(self, gear=(MESH_GEAR_MOVING, MESH_GEAR_MOVING)):
        """adjust residue_weight to define nonlinear behaviour of the link."""
        if self._weight_func is None:
            return
        dxy = self.dxy(gear=gear, use_mask=False)
        dis = np.sum(dxy ** 2, axis=-1) ** 0.5
        self._residue_weight = self._weight_func(dis).astype(np.float32)
        self._mask = None


    def set_hard_residue_filter(self, residue_len):
        """set a hard residue threshold above which match points will be cut."""
        self._weight_func = lambda x: x <= residue_len


    def set_huber_residue_filter(self, residue_len):
        """use huber loss as the link energy function."""
        self._weight_func = lambda x: residue_len / np.maximum(x, residue_len)


    def disable(self):
        self._disabled = True


    def enable(self):
        self._disabled = False


    def reset_mask(self):
        self._residue_weight = np.ones_like(self._weight)


    def reset_weight(self, weight=1):
        self._weight = np.full_like(self._weight, weight)


    def xy0(self, gear=MESH_GEAR_MOVING, use_mask=True, combine=True):
        tid = self.tid0(use_mask=use_mask)
        B = self.B0(use_mask=use_mask)
        xy = self.meshes[0].bary2cart(tid, B, gear, offsetting=False)
        offset = self.meshes[0].offset(gear)
        if combine:
            return xy + offset
        else:
            return xy, offset


    def xy1(self, gear=MESH_GEAR_MOVING, use_mask=True, combine=True):
        tid = self.tid1(use_mask=use_mask)
        B = self.B1(use_mask=use_mask)
        xy = self.meshes[1].bary2cart(tid, B, gear, offsetting=False)
        offset = self.meshes[1].offset(gear)
        if combine:
            return xy + offset
        else:
            return xy, offset


    def dxy(self, gear=(MESH_GEAR_MOVING, MESH_GEAR_MOVING), use_mask=False):
        if not hasattr(gear, '__len__'):
            gear = (gear, gear)
        xy0, offset0 = self.xy0(gear=gear[0], use_mask=use_mask, combine=False)
        xy1, offset1 = self.xy1(gear=gear[1], use_mask=use_mask, combine=False)
        dxy = xy1 - xy0
        dof = offset1 - offset0
        return dxy + dof


    def singular_vals(self, gear=(MESH_GEAR_FIXED, MESH_GEAR_FIXED), use_mask=True):
        if not hasattr(gear, '__len__'):
            gear = (gear, gear)
        xy0 = self.xy0(gear=gear[0], use_mask=use_mask, combine=False)[0]
        xy1 = self.xy1(gear=gear[1], use_mask=use_mask, combine=False)[0]
        A = spatial.fit_affine(xy0, xy1, return_rigid=False)[:2,:2]
        u, ss, vh = np.linalg.svd(A, compute_uv=True)
        ss = ss * np.sign(np.linalg.det(A))
        R = u @ vh
        rot = np.arctan2(R[0,0], R[0,1]) * 180 / np.pi
        return ss, rot


    def eliminate_zero_weight(self):
        """remove matches with zero weights."""
        keep_idx = self._weight > 0
        if np.all(keep_idx):
            return
        self._tid0 = self._tid0[keep_idx]
        self._tid1 = self._tid1[keep_idx]
        self._B0 = self._B0[keep_idx]
        self._B1 = self._B1[keep_idx]
        self._weight = self._weight[keep_idx]
        self._residue_weight = self._residue_weight[keep_idx]
        self._mask = None


    def tid0(self, use_mask=False):
        if use_mask:
            return self._tid0[self.mask]
        else:
            return self._tid0


    def tid1(self, use_mask=False):
        if use_mask:
            return self._tid1[self.mask]
        else:
            return self._tid1


    def B0(self, use_mask=False):
        if use_mask:
            return self._B0[self.mask]
        else:
            return self._B0


    def B1(self, use_mask=False):
        if use_mask:
            return self._B1[self.mask]
        else:
            return self._B1


    def weight(self, use_mask=False):
        if use_mask:
            return self._weight[self.mask] * self._residue_weight[self.mask]
        else:
            return self._weight * self._residue_weight


    @property
    def locked(self):
        return [self.meshes[0].locked, self.meshes[1].locked]


    def relevant(self, min_match_num=0):
        if self._disabled or np.all(self.locked):
            return False
        else:
            return np.sum(self.mask) > min_match_num


    @property
    def num_matches(self):
        if self._disabled:
            return (self._weight.size, 0)
        else:
            return (self._weight.size, np.sum(self.mask))


    @property
    def mask(self):
        if self._mask is None:
            self._mask = (self._weight * self._residue_weight) > 0
        return self._mask



class SpringLinkedMeshes:
    pass
