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
        self.weight = ((tid0 >= 0) & (tid1 >= 0)).astype(np.float32)
        if weight is not None:
            self.weight *= weight
        self._mask = self.weight > 0
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
        min_match_num = kwargs.get('min_match_num', 0)
        if not self.relevant(min_match_num):
            return None, None, None, None


    def disable(self):
        self._disabled = True


    def enable(self):
        self._disabled = False


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
        xy0, offset0 = self.xy0(gear=gear[0], use_mask=use_mask, combine=False)
        xy1, offset1 = self.xy1(gear=gear[1], use_mask=use_mask, combine=False)
        dxy = xy1 - xy0
        dof = offset1 - offset0
        return dxy + dof


    def singular_vals(self, gear=(MESH_GEAR_FIXED, MESH_GEAR_FIXED), use_mask=True):
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
        keep_idx = self.weight > 0
        if np.all(keep_idx):
            return
        self._tid0 = self._tid0[keep_idx]
        self._tid1 = self._tid1[keep_idx]
        self._B0 = self._B0[keep_idx]
        self._B1 = self._B1[keep_idx]
        self.weight = self.weight[keep_idx]
        self._mask = self._mask[keep_idx]


    def tid0(self, use_mask=False):
        if use_mask:
            return self._tid0[self._mask]
        else:
            return self._tid0


    def tid1(self, use_mask=False):
        if use_mask:
            return self._tid1[self._mask]
        else:
            return self._tid1


    def B0(self, use_mask=False):
        if use_mask:
            return self._B0[self._mask]
        else:
            return self._B0


    def B1(self, use_mask=False):
        if use_mask:
            return self._B1[self._mask]
        else:
            return self._B1


    @property
    def locked(self):
        return [self.meshes[0].locked, self.meshes[1].locked]


    def relevant(self, min_match_num=0):
        if self._disabled or np.all(self.locked):
            return False
        else:
            return np.sum((self.weight > 0) & self._mask) > min_match_num


    @property
    def num_matches(self):
        if self._disabled:
            return (self._mask.size, 0)
        else:
            return (self._mask.size, np.sum(self._mask))



class SpringLinkedMeshes:
    pass
