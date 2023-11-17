from collections import defaultdict
import gc
import h5py
import numpy as np
from scipy import sparse
import time

from feabas import spatial, common
import feabas.constant as const
from feabas.mesh import Mesh


class Link:
    """
    class to represent the corresponding points between two meshes.
    """
    def __init__(self, mesh0, mesh1, tid0, tid1, B0, B1, weight=None, **kwargs):
        name = kwargs.get('name',  None)
        self.uids = [mesh0.uid, mesh1.uid]
        if name is None:
            self.name = self.default_name
        else:
            self.name = name
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
    def from_coordinates(cls, mesh0, mesh1, xy0, xy1,
                         gear=(const.MESH_GEAR_INITIAL, const.MESH_GEAR_INITIAL),
                         weight=None,
                         **kwargs):
        if xy0.size == 0:
            return None, None
        kwargs.setdefault('render_weight_threshold', 0.1)
        tid0, B0 = mesh0.cart2bary(xy0, gear[0], tid=None, **kwargs)
        indx0 = tid0 >= 0
        if not np.any(indx0):
            return None, None
        elif not np.all(indx0):
            tid0 = tid0[indx0]
            B0 = B0[indx0]
            xy1 = xy1[indx0]
            if isinstance(weight, np.ndarray):
                weight = weight[indx0]
        tid1, B1 = mesh1.cart2bary(xy1, gear[1], tid=None, **kwargs)
        indx1 = tid1 >= 0
        if not np.any(indx1):
            return None, None
        if not np.all(indx1):
            tid0 = tid0[indx1]
            tid1 = tid1[indx1]
            B0 = B0[indx1]
            B1 = B1[indx1]
            if isinstance(weight, np.ndarray):
                weight = weight[indx1]
            indx0[indx0] = indx1
        return cls(mesh0, mesh1, tid0, tid1, B0, B1, weight=weight, **kwargs), indx0


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
        if (not self.relevant) or (self.num_matches == 0) or ((index_offsets[0] < 0) and (index_offsets[1] < 0)):
            return None, None, None, None
        start_gear = kwargs.get('start_gear', const.MESH_GEAR_MOVING)
        targt_gear = kwargs.get('target_gear', const.MESH_GEAR_MOVING)
        num_matches = self.num_matches
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
        return V_lft, (indx0_lft, indx1_lft), V_rht, indx_rht


    def adjust_weight_from_residue(self, gear=(const.MESH_GEAR_MOVING, const.MESH_GEAR_MOVING)):
        """adjust residue_weight to define nonlinear behaviour of the link."""
        weight_modified = False
        connection_modified = False
        if self._weight_func is None:
            return weight_modified, connection_modified
        previous_weight = self._residue_weight
        previous_connection = self.num_matches > 0
        dxy = self.dxy(gear=gear, use_mask=False)
        dis = np.sum(dxy ** 2, axis=-1) ** 0.5
        residue_weight = self._weight_func(dis).astype(np.float32)
        if np.any(residue_weight != previous_weight):
            self._residue_weight = residue_weight
            weight_modified = True
            self._mask = None
            connection_modified = (self.num_matches > 0) != previous_connection
        return weight_modified, connection_modified


    def duplicate_weight_func(self, other):
        self._weight_func = other._weight_func


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


    def xy0(self, gear=const.MESH_GEAR_MOVING, use_mask=True, combine=True):
        tid = self.tid0(use_mask=use_mask)
        B = self.B0(use_mask=use_mask)
        xy = self.meshes[0].bary2cart(tid, B, gear, offsetting=False)
        offset = self.meshes[0].offset(gear)
        if combine:
            return xy + offset
        else:
            return xy, offset


    def xy1(self, gear=const.MESH_GEAR_MOVING, use_mask=True, combine=True):
        tid = self.tid1(use_mask=use_mask)
        B = self.B1(use_mask=use_mask)
        xy = self.meshes[1].bary2cart(tid, B, gear, offsetting=False)
        offset = self.meshes[1].offset(gear)
        if combine:
            return xy + offset
        else:
            return xy, offset


    def dxy(self, gear=(const.MESH_GEAR_MOVING, const.MESH_GEAR_MOVING), use_mask=False):
        if not hasattr(gear, '__len__'):
            gear = (gear, gear)
        xy0, offset0 = self.xy0(gear=gear[0], use_mask=use_mask, combine=False)
        xy1, offset1 = self.xy1(gear=gear[1], use_mask=use_mask, combine=False)
        dxy = xy1 - xy0
        dof = offset1 - offset0
        return dxy + dof


    def singular_vals(self, gear=(const.MESH_GEAR_FIXED, const.MESH_GEAR_FIXED), use_mask=True):
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
    def default_name(self):
        return '_'.join(str(s) for s in self.uids)


    @property
    def locked(self):
        return [self.meshes[0].locked, self.meshes[1].locked]


    @property
    def relevant(self):
        return not (self._disabled or np.all(self.locked))


    @property
    def num_matches(self):
        if self._disabled:
            return 0
        else:
            return np.sum(self.mask)


    @property
    def weight_sum(self):
        if self._disabled:
            return 0
        else:
            return np.sum(self.weight(use_mask=True))


    @property
    def mask(self):
        if self._mask is None:
            self._mask = (self._weight * self._residue_weight) > 0
        return self._mask



class SLM:
    """
    Spring Linked Meshes: spring connected mesh system used for optimization.
    """
  ## --------------------------- initialization  --------------------------- ##
    def __init__(self, meshes, links=None, **kwargs):
        if links is None:
            links = []
        self.meshes = meshes
        self.links = links
        self._stiffness_lambda = kwargs.get('stiffness_lambda', 1.0)
        self._crosslink_lambda = kwargs.get('crosslink_lambda', -1.0)
        self._shared_cache = kwargs.get('shared_cache', None)
        self.clear_cached_attr()


    def clear_cached_attr(self, instant_gc=False):
        self._mesh_uids = None
        self._link_uids = None
        self._link_names = []
        self._linkage_adjacency = None
        self._connected_subsystems = None
        self._stiffness_matrix = None
        self._crosslink_terms = None
        if instant_gc:
            gc.collect()


    def link_changed(self, instant_gc=False):
        """
        flag the link list has changed that will affect the system connectivity
        graph. Note that only changing weight is not considered link change.
        """
        self._link_uids = None
        self._linkage_adjacency = None
        self._connected_subsystems = None
        self._crosslink_terms = None
        if instant_gc:
            gc.collect()


    def mesh_changed(self, instant_gc=False):
        """
        flag the mesh list has changed that will affect the system connectivity
        graph. Note that only changing vertices is not considered mesh change.
        """
        self._mesh_uids = None
        self._linkage_adjacency = None
        self._connected_subsystems = None
        self._stiffness_matrix = None
        self._crosslink_terms = None
        if instant_gc:
            gc.collect()


    def clear_equation_terms(self, instant_gc=False):
        self._stiffness_matrix = None
        self._crosslink_terms = None
        if instant_gc:
            gc.collect()


  ## -------------------------- system manipulation ------------------------ ##
    def add_meshes(self, meshes):
        if not bool(meshes):
            return
        if isinstance(meshes, (tuple, list)):
            self.meshes.extend(meshes)
        else:
            self.meshes.append(meshes)
        self.mesh_changed()


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
        working_gear = kwargs.get('working_gear', const.MESH_GEAR_INITIAL)
        submesh_exclusive = kwargs.get('submesh_exclusive', True)
        if link is None:
            return False
        if check_duplicates and (link.name in self.link_names):
            return False
        if check_relevance:
            relevance = self.link_is_relevant(link)
            if (not relevance):
                return False
            need_reinit = relevance == -1
        else:
            self.links.append(link)
            self.link_names.append(link.name)
            self.link_changed()
            return True
        if need_reinit:
            meshlist0, _ = self.select_mesh_from_uid(link.uids[0])
            meshlist1, _ = self.select_mesh_from_uid(link.uids[1])
            re_links = SLM.distribute_link(meshlist0, meshlist1,
                link, working_gear=working_gear, exclusive=submesh_exclusive,
                inner_cache=self._shared_cache)
            if len(re_links) > 0:
                self.links.extend(re_links)
                self.link_names.extend([lnk.name for lnk in re_links])
                self.link_changed()
                return True
            else:
                return False
        else:
            self.links.append(link)
            self.link_names.append(link.name)
            self.link_changed()
            return True


    def add_link_from_coordinates(self, uid0, uid1, xy0, xy1,
                                  gear=(const.MESH_GEAR_INITIAL, const.MESH_GEAR_INITIAL),
                                  weight=None, submesh_exclusive=True,
                                  check_duplicates=True,
                                  **kwargs):
        """
        add link by providing the coordinates and the mesh uids.
        Args:
            uid0, uid1: the uid of meshes. The meshes must already loaded into
                the SLM.
            xy0, xy1: Nx2 ndarrays providing the coordinates of the matching
                points in the two meshes.
        Kwargs:
            gear(tuple): the gear to use when localizing the matching points to
                the meshes.
            weight: ndarray of shape (N,) giving the weights for the matching
                points.
            submesh_exclusive(bool): if one of the mesh involved in this link
                has been divided into submeshes so that the matching points
                should be re-distributed among the submeshes, whether each point
                should only appear in one of the submeshes.
            check_duplicates(bool): test if the link has already been loaded
                based on the name of the link.
        other kwargs refer to feabas.mesh.Mesh.tri_finder.
        """
        link_added = False
        if check_duplicates:
            if ('name' in kwargs) and (kwargs['name'] in self.link_names):
                return link_added
        mesh0_list, _ = self.select_mesh_from_uid(uid0)
        if len(mesh0_list) == 0:
            return link_added
        mesh1_list, _ = self.select_mesh_from_uid(uid1)
        if len(mesh1_list) == 0:
            return link_added
        for m0 in mesh0_list:
            for m1 in mesh1_list:
                link, mask = Link.from_coordinates(m0, m1, xy0, xy1, gear=gear, weight=weight, **kwargs)
                if link is None:
                    continue
                self.add_link(link, check_relevance=False, check_duplicates=False)
                link_added = True
                if submesh_exclusive:
                    xy0 = xy0[~mask]
                    xy1 = xy1[~mask]
                    if isinstance(weight, np.ndarray):
                        weight = weight[~mask]
        return link_added


    def prune_links(self, **kwargs):
        """
        prune links so that irrelevant links are removed and links associated
        with separated/combined meshes are updated.
        """
        modified = False
        if len(self.links) == 0:
            return modified
        relevance = np.array([self.link_is_relevant(lnk) for lnk in self.links])
        if np.all(relevance == 1):
            return modified
        else:
            modified = True
        new_links = []
        working_gear = kwargs.get('working_gear', const.MESH_GEAR_INITIAL)
        submesh_exclusive = kwargs.get('submesh_exclusive', True)
        for lnk, flag in zip(self.links, relevance):
            if flag == 1:
                new_links.append(lnk)
            elif flag == -1:
                m0_list, _ = self.select_mesh_from_uid(lnk.uids[0])
                m1_list, _ = self.select_mesh_from_uid(lnk.uids[1])
                dlinks = SLM.distribute_link(m0_list, m1_list, lnk,
                    working_gear=working_gear, exclusive=submesh_exclusive,
                    inner_cache=self._shared_cache)
                new_links.extend(dlinks)
        self.links = new_links
        self._link_names = []
        self.link_changed()
        return modified


    def clear_links(self):
        self.links = []
        self._link_names = []
        self.link_changed()


    def remove_disconnected_meshes(self):
        """
        remove meshes that are not connected to an unlocked mesh by links.
        """
        A = self.linkage_adjacency()
        moving = ~self.lock_flags
        to_keep = (A.dot(moving) > 0) | moving
        if not np.all(to_keep):
            self.meshes = [m for flag, m in zip(to_keep, self.meshes) if flag]
            self.mesh_changed()


    def divide_disconnected_submeshes(self, prune_links=True, **kwargs):
        modified = False
        new_meshes = []
        for m in self.meshes:
            if m.locked:
                new_meshes.append(m)
            else:
                dm = m.divide_disconnected_mesh()
                new_meshes.extend(dm)
                if len(dm) > 1:
                    modified = True
        if modified:
            self.meshes = new_meshes
            self.mesh_changed()
            if prune_links:
                self.prune_links(**kwargs)
        return modified


    def anneal(self, gear=(const.MESH_GEAR_MOVING, const.MESH_GEAR_FIXED), mode=const.ANNEAL_CONNECTED_RIGID):
        # need to manually reset the stiffness matrix if necessary
        for m in self.meshes:
            m.anneal(gear=gear, mode=mode)


    def adjust_link_weight_by_residue(self, gear=(const.MESH_GEAR_MOVING, const.MESH_GEAR_MOVING)):
        weight_modified = False
        connection_modified = False
        for lnk in self.links:
            w,c = lnk.adjust_weight_from_residue(gear=gear)
            weight_modified |= w
            connection_modified |= c
        if connection_modified:
            self.link_changed()
        elif weight_modified:
            self._linkage_adjacency = None
            self._crosslink_terms = None
        return weight_modified, connection_modified


    def set_link_residue_threshold(self, residue_len):
        for lnk in self.links:
            lnk.set_hard_residue_filter(residue_len)


    def set_link_residue_huber(self, residue_len):
        for lnk in self.links:
            lnk.set_huber_residue_filter(residue_len)


  ## ------------------------- equation components ------------------------- ##
    def stiffness_matrix(self,  gear=(const.MESH_GEAR_FIXED, const.MESH_GEAR_MOVING),
                         force_update=False, to_cache=True, **kwargs):
        """
        system stiffness matrix and current stress.
        Kwargs:
            gear(tuple): first item used for shape matrices, second gear for stress
                computation (and stiffness if nonlinear).
            inner_cache: the cache to store intermediate attributes like shape
                matrices. Use default if set to None.
            check_flip(bool): check if any triangles are flipped.
            continue_on_flip: whether to exit when a flipped triangle is detected.
        """
        if (self._stiffness_matrix is None) or force_update:
            STIFF_M = []
            STRESS_v = []
            for m in self.meshes:
                if m.locked:
                    continue
                stiff, stress = m.stiffness_matrix(gear=gear, **kwargs)
                if stiff is None:
                    return None, None
                STIFF_M.append(stiff * m.soft_factor)
                STRESS_v.append(stress * m.soft_factor)
            stiffness_matrix = sparse.block_diag(STIFF_M, format='csr')
            stress_vector = np.concatenate(STRESS_v, axis=None)
            if to_cache:
                self._stiffness_matrix = (stiffness_matrix, stress_vector)
            return (stiffness_matrix, stress_vector)
        else:
            return self._stiffness_matrix


    def crosslink_terms(self,  force_update=False, to_cache=True, **kwargs):
        """
        compute the terms associated with the links in the assembled equation.
        Kwargs:
            start_gear: gear that associated with the vertices before applying
                the displacement
            target_gear: gear that associated with the vertices at the final
                postions for locked meshes.
            batch_num_matches: the accumulated number of matches to scan before
                constructing the incremental sparse matrices. Larger number
                needs more RAM but faster
        """
        if (self._crosslink_terms is None) or force_update:
            start_gear = kwargs.get('start_gear', const.MESH_GEAR_MOVING)
            targt_gear = kwargs.get('target_gear', const.MESH_GEAR_MOVING)
            batch_num_matches = kwargs.get('batch_num_matches', None)
            if batch_num_matches is None:
                batch_num_matches = self.num_matches / 10
            dof = self.degree_of_freedom
            Cs_lft = sparse.csr_matrix((dof, dof), dtype=np.float32)
            Cs_rht = np.zeros(dof, dtype=np.float32)
            V_lft_a = []
            I_lft0_a = []
            I_lft1_a = []
            index_offsets_mapper = self.index_offsets
            num_match = 0
            for lnk in self.links:
                indx_offst = [index_offsets_mapper[uid] for uid in lnk.uids]
                v_lft, indices_lft, v_rht, indx_rht = lnk.equation_contrib(indx_offst,
                    start_gear=start_gear, target_gear=targt_gear)
                if v_lft is None:
                    continue
                V_lft_a.append(v_lft)
                I_lft0_a.append(indices_lft[0])
                I_lft1_a.append(indices_lft[1])
                num_match += lnk.num_matches
                if num_match > batch_num_matches:
                    Cs_lft0 = sparse.csr_matrix((np.concatenate(V_lft_a), (np.concatenate(I_lft0_a), np.concatenate(I_lft1_a))),
                        shape=(dof, dof), dtype=np.float32)
                    Cs_lft += Cs_lft0
                    V_lft_a = []
                    I_lft0_a = []
                    I_lft1_a = []
                    num_match = 0
                np.add.at(Cs_rht, indx_rht, v_rht)
            if num_match > 0:
                Cs_lft0 = sparse.csr_matrix((np.concatenate(V_lft_a), (np.concatenate(I_lft0_a), np.concatenate(I_lft1_a))),
                    shape=(dof, dof), dtype=np.float32)
                Cs_lft += Cs_lft0
            if to_cache:
                self._crosslink_terms = (Cs_lft, Cs_rht)
            return (Cs_lft, Cs_rht)
        else:
            return self._crosslink_terms


    @property
    def index_offsets(self):
        vnum = [m.num_vertices * 2 for m in self.meshes]
        activated_indx = ~self.lock_flags
        vnum = vnum * (activated_indx)
        vnum_accum = np.cumsum(vnum)
        index_offsets = np.concatenate(([0], vnum_accum[:-1]))
        index_offsets[~activated_indx] = -1
        index_offsets_mapper = defaultdict(lambda:-1)
        index_offsets_mapper.update({uid: offset for uid, offset in zip(self.mesh_uids, index_offsets)})
        return index_offsets_mapper


  ## ------------------------------- optimize ------------------------------ ##
    def optimize_translation_lsqr(self, **kwargs):
        """
        find the least-squares solutions that optimize the mesh translations.
        Kwargs:
            maxiter: maximum number of iterations in LSQR. None if no limit.
            tol: the stopping tolerance of the least-square iterations.
            start_gear: gear that associated with the vertices before applying
                the translation.
            target_gear: gear that associated with the vertices at the final
                postions for locked meshes. Also the results are saved to this
                gear as well.
        """
        maxiter = kwargs.get('maxiter', None)
        tol = kwargs.get('tol', 1e-07)
        start_gear = kwargs.get('start_gear', const.MESH_GEAR_FIXED)
        targt_gear = kwargs.get('target_gear', const.MESH_GEAR_FIXED)
        locked_flag = self.lock_flags
        active_index = np.nonzero(~locked_flag)[0]
        links = self.relevant_links
        num_links = len(links)
        if num_links == 0:
            return False
        mesh_uids = self.mesh_uids[~locked_flag]
        num_meshes = mesh_uids.size
        mesh_uids_mapper = {uid: k for k, uid in enumerate(mesh_uids)}
        if num_meshes == 0:
            return False
        conn_lbl, _ = self.connected_subsystems
        conn_lbl_active = conn_lbl[~locked_flag]
        conn_lbl_locked = conn_lbl[locked_flag]
        uncontraint_labels = set(conn_lbl_active).difference(set(conn_lbl_locked))
        num_constraints = len(uncontraint_labels)
        A = sparse.lil_matrix((num_links + num_constraints, num_meshes))
        bx = np.zeros(num_links + num_constraints)
        by = np.zeros(num_links + num_constraints)
        col_k = 0
        for lnk in links:
            wt = lnk.weight_sum ** 0.5
            if wt == 0:
                continue
            gears = []
            if lnk.uids[0] in mesh_uids_mapper:
                A[col_k, mesh_uids_mapper[lnk.uids[0]]] = wt
                gears.append(start_gear)
            else:
                gears.append(targt_gear)
            if lnk.uids[1] in mesh_uids_mapper:
                A[col_k, mesh_uids_mapper[lnk.uids[1]]] = -wt
                gears.append(start_gear)
            else:
                gears.append(targt_gear)
            dxy = np.nanmedian(lnk.dxy(gear=gears, use_mask=True), axis=0)
            bx[col_k] = dxy[0] * wt
            by[col_k] = dxy[1] * wt
            col_k += 1
        if col_k == 0:
            return False
        wt = (A.power(2).sum(axis=None) / A.getnnz(axis=None)) ** 0.5
        for lbl in uncontraint_labels:
            pos = np.nonzero(conn_lbl_active == lbl)[0][0]
            A[col_k, pos] = wt
            txy = self.meshes[active_index[pos]].estimate_translation(gear=(start_gear,targt_gear))
            bx[col_k] = txy[0] * wt
            by[col_k] = txy[1] * wt
            col_k += 1
        A = A.tocsr()
        Tx = sparse.linalg.lsqr(A, bx, atol=tol, btol=tol, iter_lim=maxiter)[0]
        Ty = sparse.linalg.lsqr(A, by, atol=tol, btol=tol, iter_lim=maxiter)[0]
        for idx, tx, ty in zip(active_index, Tx, Ty):
            self.meshes[idx].set_translation((tx, ty), gear=(start_gear,targt_gear))
        return np.any(Tx!=0, axis=None) or np.any(Ty!=0, axis=None)


    def optimize_affine_cascade(self, **kwargs):
        """
        sequentially estimiate the affine transforms starting from meshes
        immediately connected to locked ones, mostly for initialization.
        Kwargs:
            start_gear: gear that associated with the vertices before applying
                the translation.
            target_gear: gear that associated with the vertices at the final
                postions for locked meshes. Also the results are saved to this
                gear as well.
            svd_clip (tuple): the limit on the svds of the affine transforms.
                default to (1,1) as rigid.
        """
        targt_gear = kwargs.get('target_gear', const.MESH_GEAR_MOVING)
        start_gear = kwargs.get('start_gear', targt_gear)
        svd_clip = kwargs.get('svd_clip', (1, 1))
        Adj = self.linkage_adjacency()
        to_optimize = ~self.lock_flags
        linked_pairs = common.find_elements_in_array(self.mesh_uids, self.link_uids)
        idxt = np.any(linked_pairs<0, axis=-1, keepdims=False)
        linked_pairs[idxt] = -1
        modified = False
        while np.any(to_optimize):
            # first find the mesh that has the most robust links to optimized ones
            link_wt_sum = Adj.dot(~to_optimize) * to_optimize
            if not np.any(link_wt_sum > 0):
                link_wt_sum = Adj.dot(np.ones_like(to_optimize)) * to_optimize
                if not np.any(link_wt_sum > 0):
                    break
            idx0 = np.argmax(link_wt_sum)
            pair_locked_flag = ~to_optimize[linked_pairs]
            link_filter = np.nonzero(np.any(linked_pairs==idx0, axis=-1)
                & np.any(pair_locked_flag, axis=-1))[0]
            if link_filter.size == 0:
                to_optimize[idx0] = False
                continue
            xy0_list = []
            xy1_list = []
            weight_list = []
            for lidx in link_filter:
                lnk = self.links[lidx]
                if lnk.uids[0] == self.mesh_uids[idx0]:
                    xy0_list.append(lnk.xy0(gear=start_gear, use_mask=True, combine=True))
                    xy1_list.append(lnk.xy1(gear=targt_gear, use_mask=True, combine=True))
                elif lnk.uids[1] == self.mesh_uids[idx0]:
                    xy0_list.append(lnk.xy1(gear=start_gear, use_mask=True, combine=True))
                    xy1_list.append(lnk.xy0(gear=targt_gear, use_mask=True, combine=True))
                else:
                    raise RuntimeError('This should never happen...')
                weight_list.append(lnk.weight(use_mask=True))
            xy0 = np.concatenate(xy0_list, axis=0)
            if xy0.size == 0:
                to_optimize[idx0] = False
                continue
            xy1 = np.concatenate(xy1_list, axis=0)
            weight = np.concatenate(weight_list, axis=None)
            _, A = spatial.fit_affine(xy1, xy0, return_rigid=True, weight=weight, svd_clip=svd_clip, avoid_flip=True)
            if (not modified) and np.any(xy0!=xy1, axis=None):
                modified = True
            self.meshes[idx0].set_affine(A, gear=(start_gear, targt_gear))
            to_optimize[idx0] = False
        return modified


    def optimize_linear(self, **kwargs):
        """
        optimize the linear system or the tangent problem of non-linear system.
        kwargs:
            maxiter: maximum number of iterations in bicgstab. None if no limit.
            tol: the relative stopping tolerance of the bicgstab.
            atol: the absolute stopping tolerance of the bicgstab.
            shape_gear: gear to caculate shape matrix.
            start_gear: the gear that associated with the vertex positions before
                applying the field from optimization. Also used for computing
                current stress, cross link terms, and stiffness matrix for
                non-linear system.
            target_gear: gear to save the vertex positions after applying the
                field from optimization. Also used for defining the final
                positions of the locked meshes.
            stiffness_lambda: stiffness term multiplier.
            crosslink_lambda: crosslink term multiplier.
            inner_cache: the cache to store intermediate attributes.
            continue_on_flip(bool): whether to continue with flipped triangles
                detected.
            batch_num_matches: the accumulated number of matches to scan before
                constructing the incremental sparse matrices. Larger number
                needs more RAM but faster
            groupings(ndarray): a ndarray of shape (N_mesh,) with entries as
                the group identities of the corresponding meshes. meshes in the
                same group have identical deformations, therefore they need to
                have exact the same mesh as well.
            auto_clear(bool): automatically clear the stiffness term after
                optimization is done. In some occasions, like flip checking
                in Newton_Raphson method, this could be set to False.
        """
        maxiter = kwargs.get('maxiter', None)
        tol = kwargs.get('tol', 1e-7)
        atol = kwargs.get('atol', None)
        callback_settings = kwargs.get('callback_settings', {}).copy()
        shape_gear = kwargs.get('shape_gear', const.MESH_GEAR_FIXED)
        targt_gear = kwargs.get('target_gear', const.MESH_GEAR_MOVING)
        start_gear = kwargs.get('start_gear', targt_gear)
        stiffness_lambda = kwargs.get('stiffness_lambda', self._stiffness_lambda)
        crosslink_lambda = kwargs.get('crosslink_lambda', self._crosslink_lambda)
        inner_cache = kwargs.get('inner_cache', self._shared_cache)
        cont_on_flip = kwargs.get('continue_on_flip', False)
        batch_num_matches = kwargs.get('batch_num_matches', None)
        groupings = kwargs.get('groupings', None)
        auto_clear = kwargs.get('auto_clear', True)
        check_flip = not cont_on_flip
        stiff_m, stress_v = self.stiffness_matrix(gear=(shape_gear,start_gear),
            inner_cache=inner_cache, check_flip=check_flip,
            continue_on_flip=cont_on_flip)
        lock_flags = self.lock_flags
        if np.all(lock_flags):
            return 0, 0 # all locked, nothing to optimize
        if stiff_m is None:
            return None, None # flipped triangles
        Cs_lft, Cs_rht = self.crosslink_terms(start_gear=start_gear,
            target_gear=targt_gear, batch_num_matches=batch_num_matches)
        stiffness_lambda, crosslink_lambda = self.relative_lambda(stiffness_lambda, crosslink_lambda)
        A = stiffness_lambda * stiff_m + crosslink_lambda * Cs_lft
        A = 0.5*(A + A.transpose())
        b = crosslink_lambda * Cs_rht - stiffness_lambda * stress_v
        if groupings is not None:
            group_u, indx, group_nm, g_cnt = np.unique(groupings, return_index=True, return_inverse=True, return_counts=True)
            if group_u.size < groupings.size:
                grouped_lock_flags = np.zeros_like(indx, dtype=bool)
                np.logical_or.at(grouped_lock_flags, group_nm, lock_flags)
                lock_flags = grouped_lock_flags[group_nm]
                if np.all(lock_flags):
                    return 0, 0
                vnum = [self.meshes[s].num_vertices * 2 for s in indx]
                vnum = vnum * (~grouped_lock_flags)
                vnum_accum = np.cumsum(vnum)
                grouped_dof = int(vnum_accum[-1])
                grouped_index_offsets = np.concatenate(([0], vnum_accum[:-1]))
                grouped_index_offsets[grouped_lock_flags] = -1
                expanded_gio = grouped_index_offsets[group_nm]
                crnt_offet = 0
                indx0 = []
                indx1 = []
                for m, gio in zip(self.meshes, expanded_gio):
                    if m.locked:
                        continue
                    stf_sz = 2 * m.num_vertices
                    if gio >= 0:
                        indx0.append(np.arange(crnt_offet, crnt_offet+stf_sz))
                        indx1.append(np.arange(gio, gio+stf_sz))
                    crnt_offet += stf_sz
                indx0 = np.concatenate(indx0, axis=None)
                indx1 = np.concatenate(indx1, axis=None)
                T_m = sparse.csr_matrix((np.ones_like(indx0, dtype=np.float32),
                                        (indx1, indx0)), shape=(grouped_dof, A.shape[0]))
                A = T_m @ A @ T_m.transpose() / np.mean(g_cnt)
                b = T_m @ b / np.mean(g_cnt)
            else:
                groupings = None
        A_diag = A.diagonal()
        if A_diag.max() > 0:
            M = sparse.diags(1/(A_diag.clip(min(1.0, A_diag.max()/1000),None))) # Jacobi precondition
        else:
            M = None
        # dd, _ = sparse.linalg.bicgstab(A, b, tol=tol, maxiter=maxiter, atol=atol, M=M)
        dd = bicgstab(A, b, tol=tol, maxiter=maxiter, atol=atol, M=M, **callback_settings)
        cost = (np.linalg.norm(b), np.linalg.norm(A.dot(dd) - b))
        if cost[1] < cost[0]:
            index_offsets = self.index_offsets
            for k, m in enumerate(self.meshes):
                if m.locked or (m.uid not in index_offsets):
                    continue
                if groupings is None:
                    stt_idx = index_offsets[m.uid]
                else:
                    stt_idx = expanded_gio[k]
                if stt_idx < 0:
                    continue
                end_idx = stt_idx + m.num_vertices * 2
                dxy = dd[stt_idx:end_idx].reshape(-1,2)
                m.set_field(dxy, gear=(start_gear, targt_gear))
            if auto_clear:
                self.clear_equation_terms()
        return cost


    def optimize_Newton_Raphson(self, **kwargs):
        """
        optimize the non linear system using newton-raphson method.
        kwargs:
            maxepoch: maximum number of linear steps to use.
            maxiter: maximum number of iterations for each linear step. None if
                no limit.
            tol: the relative stopping tolerance for each linear step.
            atol: the absolute stopping tolerance for each linear step.
            residue_mode: the method to adjust crosslink weight accordint to
                the residues. Could be 'hard'(hard threshold), 'huber', or None.
            residue_len: characteristic length of residue used to dynamically
                adjust link weights.
            anneal_mode: mode used to anneal the meshes.
            stiffness_lambda: stiffness term multipliers for each linear step.
            crosslink_lambda: crosslink term multiplier for each linear step.
            inner_cache: the cache to store intermediate attributes.
            continue_on_flip(bool): whether to continue with flipped triangles
                detected.
            crosslink_shrink: in the presence of flipped triangles,
                the decay applied to the crosslink term so that it takes smaller
                step.
            shrink_trial: maximum number of trials to attempt when shrinking the
                crosslink_lambda to battle triangle flips.
            batch_num_matches: the accumulated number of matches to scan before
                constructing the incremental sparse matrices. Larger number
                needs more RAM but faster
        """
        maxepoch = kwargs.get('maxepoch', 5)
        tol = kwargs.get('tol', 1e-7)
        atol = kwargs.get('atol', None)
        maxiter = SLM.expand_to_list(kwargs.get('maxiter', None), maxepoch)
        step_tol = SLM.expand_to_list(kwargs.get('step_tol', tol), maxepoch)
        step_atol = SLM.expand_to_list(kwargs.get('step_atol', atol), maxepoch)
        stiffness_lambda = SLM.expand_to_list(kwargs.get('stiffness_lambda', self._stiffness_lambda), maxepoch)
        crosslink_lambda = SLM.expand_to_list(kwargs.get('crosslink_lambda', self._crosslink_lambda), maxepoch)
        residue_mode = SLM.expand_to_list(kwargs.get('residue_mode', None), maxepoch)
        residue_len = SLM.expand_to_list(kwargs.get('residue_len', 0), maxepoch)
        anneal_mode = SLM.expand_to_list(kwargs.get('anneal_mode', None), maxepoch)
        inner_cache = kwargs.get('inner_cache', self._shared_cache)
        cont_on_flip = kwargs.get('continue_on_flip', False)
        crosslink_shrink = kwargs.get('crosslink_shrink', 0.25)
        shrink_trial = kwargs.get('shrink_trial', 3)
        groupings = kwargs.get('groupings', None)
        batch_num_matches = kwargs.get('batch_num_matches', None)
        callback_settings = kwargs.get('callback_settings', {}).copy()
        shape_gear = const.MESH_GEAR_FIXED
        start_gear = const.MESH_GEAR_MOVING
        if cont_on_flip:
            target_gear = const.MESH_GEAR_MOVING
        else:
            target_gear = const.MESH_GEAR_STAGING
        # initialize cost and check flipped triangles
        check_flip = not cont_on_flip
        stiff_m, _ = self.stiffness_matrix(gear=(shape_gear,start_gear),
            force_update=True, to_cache=True,
            inner_cache=inner_cache, check_flip=check_flip,
            continue_on_flip=cont_on_flip)
        if stiff_m is None:
            return None, None
        _, _ = self.crosslink_terms(force_update=True, to_cache=True,
            start_gear=start_gear, target_gear=target_gear,
            batch_num_matches=batch_num_matches)
        cost0 = self.cost(stiffness_lambda[-1], crosslink_lambda[-1])
        cost = np.inf
        if tol is None:
            tol0 = atol
        elif atol is None:
            tol0 = cost0 * tol
        else:
            tol0 = min(cost0 * tol, atol)
        ke = 0 # epoch counter
        kshrk = 0 # crosslink_shrink counter
        cshrink = 1
        while ke < maxepoch:
            step_cost = self.optimize_linear(maxiter=maxiter[ke],
                tol=step_tol[ke], atol=step_atol[ke],
                shape_gear=shape_gear, start_gear=start_gear, target_gear=target_gear,
                stiffness_lambda=stiffness_lambda[ke],
                crosslink_lambda=crosslink_lambda[ke]*cshrink,
                inner_cache=inner_cache, continue_on_flip=cont_on_flip,
                batch_num_matches=batch_num_matches, groupings=groupings,
                auto_clear=False, callback_settings=callback_settings)
            if (step_cost[0] is None) or (step_cost[0] < step_cost[1]):
                break
            if anneal_mode[ke] is not None:
                self.anneal(gear=(target_gear, shape_gear), mode=anneal_mode[ke])
            stiff_m, _ = self.stiffness_matrix(gear=(shape_gear,target_gear),
                force_update=True, to_cache=True,
                inner_cache=inner_cache, check_flip=check_flip,
                continue_on_flip=cont_on_flip)
            if stiff_m is None:
                cshrink *= crosslink_shrink
                kshrk += 1
                if kshrk > shrink_trial:
                    break
                continue
            kshrk = 0
            if residue_mode[ke] is not None:
                if residue_len[ke] > 0:
                    if residue_mode[ke] == 'huber':
                        self.set_link_residue_huber(residue_len[ke])
                    else:
                        self.set_link_residue_threshold(residue_len[ke])
                self.adjust_link_weight_by_residue(gear=(target_gear, target_gear))
            _, _ = self.crosslink_terms(force_update=True, to_cache=True,
                start_gear=target_gear, target_gear=target_gear,
                batch_num_matches=batch_num_matches)
            if start_gear != target_gear:
                self.anneal(gear=(target_gear, start_gear), mode=const.ANNEAL_COPY_EXACT)
            cost = min(cost, self.cost(stiffness_lambda[-1], crosslink_lambda[-1]))
            if (tol0 is not None) and (cost < tol0):
                break
            ke += 1
            if ke >= len(stiffness_lambda):
                break
        return cost0, cost


    def optimize_elastic(self, **kwargs):
        if self.is_linear:
            return self.optimize_linear(**kwargs)
        else:
            return self.optimize_Newton_Raphson(**kwargs)


    def relative_lambda(self, stiffness_lambda, crosslink_lambda):
        if (stiffness_lambda < 0) or (crosslink_lambda < 0):
            if (self._stiffness_matrix is None) or (self._crosslink_terms is None):
                raise RuntimeError('System equation not initialized')
            ratio = abs(stiffness_lambda / crosslink_lambda)
            stiff_m, _ = self._stiffness_matrix
            Cs_lft, _ = self._crosslink_terms
            nm_stiff = sparse.linalg.norm(stiff_m)
            nm_cl = sparse.linalg.norm(Cs_lft)
            stiffness_lambda = abs(ratio * nm_cl / nm_stiff)
            crosslink_lambda = 1.0
        return stiffness_lambda, crosslink_lambda


    def cost(self, stiffness_lambda, crosslink_lambda):
        if (self._stiffness_matrix is None) or (self._crosslink_terms is None):
            raise RuntimeError('System equation not initialized')
        stiff_m, stress_v = self._stiffness_matrix
        if stiff_m is None:
            return None
        stiffness_lambda, crosslink_lambda = self.relative_lambda(stiffness_lambda, crosslink_lambda)
        Cs_rht, Cs_rht = self._crosslink_terms
        return np.linalg.norm(crosslink_lambda * Cs_rht - stiffness_lambda * stress_v)


    def flag_outcasts(self):
        """
        outcasts are defined as: 1) meshes not connected to any locked meshes
        (if there is any); or 2) meshes in the minority of the subsystems if
        all meshes are free-floating.
        """
        outcasts = np.zeros(self.num_meshes, dtype=bool)
        labels, n = self.connected_subsystems
        outcast0 = [m.is_outcast for m in self.meshes]
        if n == 1:
            return outcast0
        lock_flags = self.lock_flags
        if np.any(outcast0) or np.any(lock_flags):
            locked_label = labels[lock_flags]
            outcasts = ~np.isin(labels, locked_label)
        else:
            u, cnt = np.unique(labels, return_counts=True)
            indx = np.argmax(cnt)
            outcasts = labels != u[indx]
        for m, flg in zip(self.meshes, outcasts):
            m.is_outcast = flg
        return outcasts


    @property
    def is_linear(self):
        linearity = True
        for m in self.meshes:
            if m.locked:
                continue
            if not m.is_linear:
                linearity = False
                break
        return linearity


  ## ----------------------------- cached attr ----------------------------- ##
    @property
    def link_names(self):
        if not bool(self._link_names):
            self._link_names = [l.name for l in self.links]
        return self._link_names


    @property
    def link_uids(self):
        if self._link_uids is None:
            if self.num_links == 0:
                self._link_uids = np.empty((0,2))
            else:
                self._link_uids = np.array([lnk.uids for lnk in self.links])
        return self._link_uids


    @property
    def mesh_uids(self):
        if self._mesh_uids is None:
            self._mesh_uids = np.array([m.uid for m in self.meshes])
        return self._mesh_uids


    def linkage_adjacency(self, directional=False):
        """
        Adjacency matrix for the meshes in the system, where meshes with links
        is considered connected.
        """
        if self._linkage_adjacency is None:
            edges = common.find_elements_in_array(self.mesh_uids, self.link_uids)
            num_matches = np.array([lnk.weight_sum for lnk in self.links])
            indx = np.all(edges>=0, axis=-1, keepdims=False)
            if not np.all(indx):
                edges = edges[indx]
                num_matches = num_matches[indx]
            A = sparse.csr_matrix((num_matches, (edges[:,0], edges[:,1])), shape=(self.num_meshes, self.num_meshes))
            if not directional:
                A = A + A.transpose()
            A.eliminate_zeros()
            self._linkage_adjacency = A
        return self._linkage_adjacency


    @property
    def connected_subsystems(self):
        if self._connected_subsystems is None:
            n, labels = sparse.csgraph.connected_components(self.linkage_adjacency(), directed=False, return_labels=True)
            self._connected_subsystems = (labels, n)
        return self._connected_subsystems


  ## ------------------------------ properties ----------------------------- ##
    @property
    def lock_flags(self):
        return np.array([m.locked for m in self.meshes])


    @lock_flags.setter
    def lock_flags(self, flags):
        old_flags = self.lock_flags
        assert old_flags.size == flags.size
        changed = np.nonzero(old_flags != flags)[0]
        if changed.size> 0:
            for indx in changed:
                self.meshes[indx].locked = not self.meshes[indx].locked
            self.mesh_changed()


    @property
    def num_meshes(self):
        return len(self.meshes)


    @property
    def num_links(self):
        return len(self.links)


    @property
    def degree_of_freedom(self):
        vnum = [m.num_vertices * (1 - m.locked) for m in self.meshes]
        return 2 * np.sum(vnum)


    @property
    def num_matches(self):
        return np.sum([lnk.num_matches for lnk in self.links])


    @property
    def relevant_links(self):
        """
        links that are directly relevant to solving the system, i.e. connecting
        to at least one unlocked mesh.
        """
        return [lnk for lnk in self.links if (self.link_is_relevant(lnk) == 1)]


    def match_residues(self, gear=const.MESH_GEAR_MOVING, use_mask=False, quantile=1):
        dis = []
        for lnk in self.links:
            if use_mask and not lnk.relevant:
                dis.append(np.nan)
                continue
            dxy = np.sum(lnk.dxy(gear=gear, use_mask=use_mask)**2, axis=-1)**0.5
            if quantile == 1:
                dis.append(np.max(dxy))
            elif quantile == 0:
                dis.append(np.min(dxy))
            else:
                dis.append(np.quantile(dxy, quantile))
        return np.array(dis)


  ## ------------------------------ utilities ------------------------------ ##
    def link_is_relevant(self, link):
        """
        check if a link is useful for optimization.
        Return:
            0 - not useful; 1 - useful; -1 - useful but need to reinitialized.
        """
        if link is None:
            return 0
        if not link.relevant:
            return 0
        link_uids = link.uids
        for lid in link_uids:
            sel_mesh, exact = self.select_mesh_from_uid(lid)
            if len(sel_mesh) == 0:
                return 0
            elif not exact:
                return -1
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


    @staticmethod
    def distribute_link(mesh0_list, mesh1_list, link, exclusive=True,
                        working_gear=const.MESH_GEAR_INITIAL, **kwargs):
        """ distribute a single links to accommodate separated meshes. """
        if isinstance(link, Link):
            link_initialized = True
        elif isinstance(link, common.Match):
            link_initialized = False
        else:
            raise TypeError
        if link_initialized:
            xy0 = link.xy0(gear=working_gear, use_mask=False, combine=True)
            xy1 = link.xy1(gear=working_gear, use_mask=False, combine=True)
            weight = link.weight(use_mask=False)
            if link.name == link.default_name:
                name = None
            else:
                name = link.name
        else:
            xy0 = link.xy0
            xy1 = link.xy1
            weight = link.weight
            name = None
        out_links = []
        for m0 in mesh0_list:
            for m1 in mesh1_list:
                lnk, mask = Link.from_coordinates(m0, m1, xy0, xy1, gear=(working_gear, working_gear),
                    weight=weight, name=name, **kwargs)
                if lnk is None:
                    continue
                if link_initialized:
                    lnk.duplicate_weight_func(link)
                if lnk is not None:
                    out_links.append(lnk)
                if exclusive:
                    xy0 = xy0[~mask]
                    xy1 = xy1[~mask]
                    weight = weight[~mask]
                    if xy0.size == 0:
                        break
        return out_links


    @staticmethod
    def expand_to_list(elem, list_len):
        if (not hasattr(elem, '__len__')) or isinstance(elem, str):
            return [elem] * list_len
        else:
            return elem


class EarlyStopFlag(Exception):
    pass


class SLM_Callback:
    """
    call back function class fed to bicgstab optimizer.
    Args:
        A, b: equation terms to solve x: Ax = b.
    Kwargs:
        timeout: timeout time for the optimizer (in seconds);
        early_stop_thresh: the threshold for the differences between two
            consecutive solutions below which it is considered small update;
        eval_step: skip step to evaluate the callback function;
        chances: if the number of consecutive iterations with enlarged cost or
            small updates is larger than this number, envoke early stopping.
    """
    def __init__(self, A, b, timeout=None, early_stop_thresh=None, chances=5, eval_step=10):
        self._A = A
        self._b = b
        self._timout = timeout
        self._early_stop_thresh = early_stop_thresh
        self._eval_step = eval_step
        self._t0 = time.time()
        self._last_x = 0
        self._last_cost = np.inf
        self.min_cost = np.inf
        self.solution = None
        self._count = 0
        self._chances = chances
        self._exit_count = 0


    def callback(self, x):
        if self._count % self._eval_step == 0:
            cost = np.linalg.norm(self._A.dot(x) - self._b)
            if cost < self.min_cost:
                self.min_cost = cost
                self.solution = x
            if self._timout is not None:
                t = time.time() - self._t0
                if t > self._timout:
                    raise EarlyStopFlag
            if self._chances is not None:
                if cost > self._last_cost:
                    self._exit_count += 1
                elif self._early_stop_thresh is not None:
                    dis = np.max(np.abs(x - self._last_x))
                    if dis <= self._early_stop_thresh:
                        self._exit_count += 1
                    else:
                        self._exit_count = 0
                else:
                    self._exit_count = 0
                if self._exit_count > self._chances:
                    raise EarlyStopFlag
                self._last_x = x.copy()
                self._last_cost = cost
        self._count += 1
        return False


def bicgstab(A, b, tol=1e-07, atol=None, maxiter=None, M=None, **kwargs):
    timeout = kwargs.get('timeout', None)
    early_stop_thresh = kwargs.get('early_stop_thresh', None)
    chances = kwargs.get('chances', None)
    eval_step = kwargs.get('eval_step', 10)
    cb = SLM_Callback(A, b, timeout=timeout, early_stop_thresh=early_stop_thresh, chances=chances, eval_step=eval_step)
    callback = cb.callback
    try:
        x, _ = sparse.linalg.bicgstab(A, b, tol=tol, maxiter=maxiter, atol=atol, M=M, callback=callback)
        cost0 = np.linalg.norm(A.dot(x) - b)
        if cost0 > cb.min_cost:
            x = cb.solution
    except EarlyStopFlag:
        x = cb.solution
    return x



def transform_mesh(mesh_unlocked, mesh_locked, **kwargs):
    err_thresh = kwargs.pop('err_thresh', None)
    kwargs.setdefault('continue_on_flip', True)
    uid_mov = mesh_unlocked.uid
    locked_mov = mesh_unlocked.locked
    mesh_locked = mesh_locked.copy(override_dict={'locked': True, 'uid': 0})
    mesh_unlocked = mesh_unlocked.copy(override_dict={'locked': False, 'uid': 1})
    mesh_unlocked.change_resolution(mesh_locked.resolution)
    xy_fix = mesh_locked.vertices_w_offset(gear=const.MESH_GEAR_INITIAL)
    # xy_mov = mesh_unlocked.vertices_w_offset(gear=const.MESH_GEAR_INITIAL)
    # xy0 = np.concatenate((xy_fix, xy_mov), axis=0)
    xy0 = xy_fix
    opt = SLM([mesh_locked, mesh_unlocked], stiffness_lambda=0.01)
    opt.divide_disconnected_submeshes()
    opt.add_link_from_coordinates(mesh_locked.uid, mesh_unlocked.uid, xy0, xy0, check_duplicates=False)
    opt.optimize_affine_cascade()
    opt.anneal(gear=(const.MESH_GEAR_MOVING, const.MESH_GEAR_FIXED), mode=const.ANNEAL_CONNECTED_RIGID)
    opt.optimize_elastic(**kwargs)
    rel_meshes = [m for m in opt.meshes if np.floor(m.uid)==1]
    residue = [np.max(np.abs(lnk.dxy(gear=1))) for lnk in opt.links]
    print(f'{mesh_locked.name}: {residue}')
    mesh_unlocked = Mesh.combine_mesh(rel_meshes, uid=uid_mov, locked=locked_mov)
    mesh_unlocked.locked = locked_mov
    if err_thresh is not None:
        flag = np.any(np.array(residue) > err_thresh)
        return mesh_unlocked, flag
    else:
        return mesh_unlocked


def relax_mesh(M, free_vertices=None, free_triangles=None, **kwargs):
    gear = kwargs.get('gear', (const.MESH_GEAR_FIXED, const.MESH_GEAR_MOVING))
    maxiter = kwargs.get('maxiter', None)
    tol = kwargs.get('tol', 1e-7)
    atol = kwargs.get('atol', None)
    callback_settings = kwargs.get('callback_settings', {}).copy()
    modified = False
    if free_vertices is not None:
        vindx = free_vertices
    elif free_triangles is not None:
        T = M.triangles[~free_triangles]
        vindx = ~np.isin(np.arange(M.num_vertices), np.unique(T))
    else:
        return modified
    vmask = np.zeros(M.num_vertices, dtype=bool)
    vmask[vindx] = True
    if not np.any(vmask):
        return modified 
    vmask_pad = np.repeat(vmask, 2)
    M.anneal(gear=gear[::-1],  mode=const.ANNEAL_CONNECTED_RIGID)
    M._vertices_changed(gear=gear[0])
    stiff_M, stress_v = M.stiffness_matrix(gear=gear, continue_on_flip=True, cache=False)
    A = stiff_M[vmask_pad][:,vmask_pad]
    b = -stress_v[vmask_pad]
    A_diag = A.diagonal()
    if A_diag.max() > 0:
        cond = sparse.diags(1/(A_diag.clip(min(1.0, A_diag.max()/1000),None))) # Jacobi precondition
    else:
        cond = None
    dd = bicgstab(A, b, tol=tol, maxiter=maxiter, atol=atol, M=cond, **callback_settings)
    cost = (np.linalg.norm(b), np.linalg.norm(A.dot(dd) - b))
    if (cost[1] < cost[0]) and np.any(dd != 0):
        modified = True
        M.apply_field(dd.reshape(-1,2), gear[-1], vtx_mask=vmask)
    return modified
        