import gc
import numpy as np

from feabas import spatial
from feabas.constant import *


class Link:
    """
    class to represent the corresponding points between two meshes.
    """
    def __init__(self, mesh0, mesh1, tid0, tid1, B0, B1, **kwargs):
        weight = kwargs.get('weight', None)
        self.name = kwargs.get('name', '')
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
    def from_coordinates(cls, mesh0, mesh1, xy0, xy1, gear=(MESH_GEAR_INITIAL, MESH_GEAR_INITIAL), **kwargs):
        tid0, B0 = mesh0.cart2bary(xy0, gear[0], tid=None)
        indx0 = tid0 >= 0
        if not np.any(indx0):
            return None, None
        elif not np.all(indx0):
            tid0 = tid0[indx0]
            B0 = B0[indx0]
            xy1 = xy1[indx0]
            if 'weight' in kwargs and isinstance(kwargs['weight'], np.ndarray):
                kwargs['weight'] = kwargs['weight'][indx0]
        tid1, B1 = mesh1.cart2bary(xy1, gear[1], tid=None)
        indx1 = tid1 >= 0
        if not np.any(indx1):
            return None, None
        if not np.all(indx1):
            tid0 = tid0[indx1]
            tid1 = tid1[indx1]
            B0 = B0[indx1]
            B1 = B1[indx1]
            if 'weight' in kwargs and isinstance(kwargs['weight'], np.ndarray):
                kwargs['weight'] = kwargs['weight'][indx1]
            indx0[indx0] = indx1
        return cls(mesh0, mesh1, tid0, tid1, B0, B1, **kwargs), indx0


    def combine_link(self, other):
        if other is None:
            return
        assert np.all(np.sort(self.uids) == np.sort(other.uids))
        flipped = self.uids[0] != other.uids[0]
        if flipped:
            aB0 = other._B1
            aB1 = other._B0
            atid0 = other._tid1
            atid1 = other._tid0
        else:
            aB0 = other._B0
            aB1 = other._B1
            atid0 = other._tid0
            atid1 = other._tid1
        self._B0 = np.concatenate((self._B0, aB0), axis=0)
        self._B1 = np.concatenate((self._B1, aB1), axis=0)
        self._tid0 = np.concatenate((self._tid0, atid0), axis=0)
        self._tid1 = np.concatenate((self._tid1, atid1), axis=0)
        self._weight = np.concatenate((self._weight, other._weight), axis=0)
        self._residue_weight = np.concatenate((self._residue_weight, other._residue_weight), axis=0)
        self._mask = None


    def equation_contrib(self, index_offsets, **kwargs):
        """computing the contribution needed to add to the FEM assembled matrix."""
        if not self.relevant:
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
        self._mask = None


    def reset_weight(self, weight=1):
        self._weight = np.full_like(self._weight, weight)
        self._mask = None


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


    @property
    def relevant(self):
        if self._disabled or np.all(self.locked):
            return False
        else:
            return np.sum(self.mask) > 0


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
    """
    A spring connected mesh system used for optimization.
    """
    def __init__(self, meshes, links=[], **kwargs):
        self.meshes = meshes
        self.links = links
        self._stiffness_lambda = kwargs.get('stiffness_lambda', 1.0)
        self._crosslink_lambda = kwargs.get('crosslink_lambda', 1.0)


    def link_changed(self, gc_now=False):
        NotImplemented
        if gc_now:
            gc.collect()


    def mesh_changed(self, gc_now=False):
        NotImplemented
        if gc_now:
            gc.collect()


    def add_link(self, link, check_relevance=True, check_duplicates=True, **kwargs):
        """
        add a link to the system
            Args:
                link (Link): link to add.
            Kwargs:
                check_revelvance (bool): whether to check whether the link is
                    useful in solving the optimization problem, i.e. if the
                    meshes it connects are locked or not included (based on
                    uids), or the link itself has no activated matches.
                check_duplicates (bool): if to check the link is already loaded
                    in the system (based on link names)
                working gear: the gear that used to reinitiate the link if there
                    are separated or combined mesh in the system.

        """
        working_gear = kwargs.get('working_gear', MESH_GEAR_INITIAL)
        if link is None:
            return False
        if check_duplicates and (link.name in self.link_names):
            return False
        if check_relevance:
            relevance = self.link_relevant(link)
            if (not relevance):
                return False
            need_reinit = relevance == 2
        else:
            self.links.append(link)
            self.link_names.append(link.name)
            self.link_changed(gc_now=False)
            return True
        if need_reinit:
            meshlist0, _ = self.select_mesh_from_uid(link.uids[0])
            meshlist1, _ = self.select_mesh_from_uid(link.uids[1])
            re_links = SpringLinkedMeshes.reinitialize_link(meshlist0, meshlist1, link, working_gear=working_gear)
            if len(re_links) > 0:
                self.links.extend(re_links)
                self.link_names.extend([lnk.name for lnk in re_links])
                self.link_changed(gc_now=False)
                return True
            else:
                return False
        else:
            self.links.append(link)
            self.link_names.append(link.name)
            self.link_changed(gc_now=False)
            return True


    def link_relevant(self, link):
        """
        check if a link is useful for optimization.
        Return:
            0 - not useful; 1 - useful; 2 - useful but need to reinitialized.
        """
        if link is None:
            return 0
        if not link.relevant:
            return 0
        mesh_uids = self.mesh_uids
        link_uids = link.uids
        for lid in link_uids:
            sel_mesh, exact = self.select_mesh_from_uid(lid)
            if len(sel_mesh) == 0:
                return 0
            elif not exact:
                return 2
        return 1


    def select_mesh_from_uid(self, uid):
        """
        given an uid of a mesh, return all the meshes in the system that have
        that uid. Also return a flag indicating whether the uid is exactly the
        same, or just within 0.5 distance (meaning it's a submesh)
        """
        if self.num_meshes == 0:
            return [], False
        uid = float(uid)
        mesh_uids = self.mesh_uids
        dis = np.abs(mesh_uids - uid)
        if np.min(dis) == 0:
            # the exact mesh found
            indx = np.nonzero(dis == 0)[0][0]
            return [self.meshes[indx]], True
        elif uid.is_integer():
            # probing uid is a complete mesh, but the corresponding mesh inside
            #   the system is subdivided already 
            indx = np.nonzero(dis < 0.5)[0]
            return [self.meshes[s] for s in indx], False
        else:
            # probing uid is a submesh. only return if the mesh in the system is
            #   a complete mesh or exact the same (as in the first condition)
            uid_r = np.floor(uid)
            dis_r = np.abs(mesh_uids - uid_r)
            if np.min(dis_r) == 0:
                indx = np.nonzero(dis_r == 0)[0][0]
                return [self.meshes[indx]], False
            else:
                return [], False


    @property
    def link_names(self):
        if (not hasattr(self, '_link_names')) or (not bool(self._link_names)):
            self._link_names = [l.name for l in self._links]
        return self._link_names


    @property
    def num_meshes(self):
        return len(self.meshes)


    @property
    def num_links(self):
        return len(self.links)


    @property
    def mesh_uids(self):
        if (not hasattr(self, '_mesh_uids') or self._mesh_uids is None):
            self._mesh_uids = np.array([m.uid for m in self.meshes])
        return self._mesh_uids


    @staticmethod
    def reinitialize_link(mesh0_list, mesh1_list, link, working_gear=MESH_GEAR_INITIAL):
        xy0 = link.xy0(gear=working_gear, use_mask=False, combine=True)
        xy1 = link.xy1(gear=working_gear, use_mask=False, combine=True)
        weight = link._weight
        name = link._name
        out_links = []
        for m0 in mesh0_list:
            for m1 in mesh1_list:
                lnk, mask = Link.from_coordinates(m0, m1, xy0, xy1, gear=(working_gear, working_gear), weight=weight, name=name)
                if lnk is not None:
                    out_links.append(lnk)
                xy0 = xy0[~mask]
                xy1 = xy1[~mask]
                weight = weight[~mask]
        return out_links
