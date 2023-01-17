import numpy as np

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



class SpringLinkedMeshes:
    pass
