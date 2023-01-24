from collections import defaultdict
import copy
import cv2
import gc
import h5py
import inspect
import matplotlib.tri
import numpy as np
import os
from rtree import index
from scipy import sparse
import scipy.sparse.csgraph as csgraph
from scipy.spatial import KDTree
import shapely
import shapely.geometry as shpgeo
from shapely.ops import polygonize, unary_union
import triangle

from feabas import miscs, material, spatial
from feabas.constant import *


def gear_constant_to_str(gear_const):
    if isinstance(gear_const, (tuple, list)):
        gear_str = '_'.join([gear_constant_to_str(s) for s in gear_const])
    elif gear_const == MESH_GEAR_INITIAL:
        gear_str = 'INITIAL'
    elif gear_const == MESH_GEAR_FIXED:
        gear_str = 'FIXED'
    elif gear_const == MESH_GEAR_MOVING:
        gear_str = 'MOVING'
    elif gear_const == MESH_GEAR_STAGING:
        gear_str = 'STAGING'
    else:
        raise ValueError
    return gear_str


def gear_str_to_constant(gear_str):
    if '_' in gear_str:
        gear_const = tuple(gear_str_to_constant(s) for s in gear_str.split('_'))
    elif gear_str.upper() == 'INITIAL':
        gear_const = MESH_GEAR_INITIAL
    elif gear_str.upper() == 'FIXED':
        gear_const = MESH_GEAR_FIXED
    elif gear_str.upper() == 'MOVING':
        gear_const = MESH_GEAR_MOVING
    elif gear_str.upper() == 'STAGING':
        gear_const = MESH_GEAR_STAGING
    else:
        raise ValueError
    return gear_const


def config_cache(gear):
    """
    The decorator that determines the caching behaviour of the Mesh properties.
    gear: used to generate caching key. Possible values include:
        MESH_GEAR_INITIAL: the property is only related to the initial vertice
            positions and their connection;
        MESH_GEAR_FIXED: the property is also related to the position of fixed
            vertice positions;
        MESH_GEAR_MOVING: the property is also related to the position of moving
            vertice positions;
        'TBD': the vertices on which the property is caculated is determined on
            the fly. If 'gear' is provided in the keyward argument, use that;
            otherwise, use self._current_gear.
    cache: If False, no caching;
        If True, save to self as an attribute (default);
        If type of miscs.Cache, save to the cache object with key;
        If type defaultdict,  save to cache object with key under dict[prop_name].
    assign_value: if kwargs assign_value is given, instead of computing the property,
        directly return that value and force cache it if required.
    no_compute: if set to True, only probe if there exists cached value. If not,
        directly return None without compute. Otherwise return cached value.
        default to False.
    """
    def config_cache_wrap(func):
        prop_name0 = func.__name__
        def decorated(self, cache=None, force_update=False, no_compute=False, **kwargs):
            if 'assign_value' in kwargs:
                force_update = True
                assign_mode = True
                prop0 = kwargs['assign_value']
            else:
                assign_mode = False
            if gear == 'TBD':
                if 'gear' in kwargs:
                    cgear = kwargs['gear']
                else:
                    argspec = inspect.getargspec(func)
                    nd = len(argspec.args) - len(argspec.defaults)
                    kwnm = argspec.args[nd:]
                    if 'gear' in kwnm:
                        cindx = kwnm.index('gear')
                        cgear = argspec.defaults[cindx]
                    else:
                        cgear = self._current_gear
            else:
                cgear = gear
            if cgear is None:
                cgear = self._current_gear
            if cache is None:
                # if cache not provided, use default cache in self.
                cache = self._default_cache[cgear]
            masked_operation = False
            if ('tri_mask' in kwargs) and (kwargs['tri_mask'] is not None):
                tri_mask = np.array(kwargs['tri_mask'], copy=False)
                if tri_mask.dtype == bool:
                    if not np.all(tri_mask):
                        masked_operation = True
                else:
                    if tri_mask.size < self.num_triangles:
                        masked_operation = True
                    else:
                        tri_mask0 = np.zeros(self.num_triangles, dtype=bool)
                        tri_mask0[tri_mask] = True
                        if not np.all(tri_mask0):
                            masked_operation = True
            if (not masked_operation) and (kwargs.get('vtx_mask', None) is not None):
                vtx_mask = np.array(vtx_mask, copy=False)
                if vtx_mask.dtype == bool:
                    if not np.all(vtx_mask):
                        masked_operation = True
                else:
                    if vtx_mask.size < self.num_vertices:
                        masked_operation = True
                    else:
                        vtx_mask0 = np.zeros(self.num_vertices, dtype=bool)
                        vtx_mask0[vtx_mask] = True
                        if not np.all(vtx_mask0):
                            masked_operation = True
            if masked_operation:
                # if contain masked triangles or vertices, don't cache
                cache = False
                force_update = True
            if isinstance(cache, bool):
                sgear = gear_constant_to_str(cgear)
                prop_name = '_cached_' + prop_name0 + '_G_' + sgear
                if cache:  # save to self as an attribute
                    if not force_update and hasattr(self, prop_name):
                        # if already cached, use that
                        prop = getattr(self, prop_name)
                        if prop is not None:
                            return prop
                    if assign_mode:
                        prop = prop0
                    elif no_compute:
                        return None
                    else:
                        prop = func(self, **kwargs)
                    setattr(self, prop_name, prop)
                    return prop
                else: # no caching
                    if not force_update and hasattr(self, prop_name):
                        # if already cached, use that
                        prop = getattr(self, prop_name)
                        if prop is not None:
                            return prop
                    if assign_mode:
                        prop = prop0
                    elif no_compute:
                        return None
                    else:
                        prop = func(self, **kwargs)
                    return prop
            else:
                cache_key = self.caching_keys(gear=cgear)
                if isinstance(cache, miscs.CacheNull):
                    key = (*cache_key, prop_name0)
                    if not force_update and (key in cache):
                        prop = cache[key]
                        if prop is not None:
                            return prop
                    if assign_mode:
                        prop = prop0
                    elif no_compute:
                        return None
                    else:
                        prop = func(self, **kwargs)
                    cache.update_item(key, prop)
                    return prop
                elif isinstance(cache, dict):
                    cache_obj = cache[prop_name0]
                    key = cache_key
                    if not force_update and (key in cache_obj):
                        prop = cache_obj[key]
                        if prop is not None:
                            return prop
                    if assign_mode:
                        prop = prop0
                    elif no_compute:
                        return None
                    else:
                        prop = func(self, **kwargs)
                    cache_obj.update_item(key, prop)
                    return prop
                else:
                    raise TypeError('Cache type not recognized')
        return decorated
    return config_cache_wrap


class Mesh:
    """
    A class to represent a FEM Mesh.
    Args:
        vertices (NV x 2 ndarray): x-y cooridnates of the vertices.
        triangles (NT x 3 ndarray): each row is 3 vertex indices belong to a
            triangle.
    Kwargs:
        material_ids (NT ndarray of int16): material for each triangle. If None,
            set to the id of the default material defined in feabas.material.
        material_table (feabas.material.MaterialTable): table of
            material properties.
        resolution (float): resolution of the mesh, for automatical scaling when
            working with images with specified resolution. default to 4nm.
        name(str): name of the mesh, used for printing/saving.
        token(int): unique id number, used as the key for caching. If not set, use
            the hash of object attributes.
    """
    uid_counter = 0.0
  ## ------------------------- initialization & IO ------------------------- ##
    def __init__(self, vertices, triangles, **kwargs):
        vertices = vertices.reshape(-1, 2)
        triangles = triangles.reshape(-1, 3)
        self._vertices = {MESH_GEAR_INITIAL: vertices}
        self._vertices[MESH_GEAR_FIXED] = kwargs.get('fixed_vertices', vertices)
        self._vertices[MESH_GEAR_MOVING] = kwargs.get('moving_vertices', None)
        self._vertices[MESH_GEAR_STAGING] = kwargs.get('staging_vertices', None)
        self._offsets = {}
        self._offsets[MESH_GEAR_INITIAL] = kwargs.get('initial_offset', np.zeros((1,2), dtype=np.float64))
        self._offsets[MESH_GEAR_FIXED] = kwargs.get('fixed_offset', np.zeros((1,2), dtype=np.float64))
        self._offsets[MESH_GEAR_MOVING] = kwargs.get('moving_offset', np.zeros((1,2), dtype=np.float64))
        self._offsets[MESH_GEAR_STAGING] = kwargs.get('staging_offset', np.zeros((1,2), dtype=np.float64))
        self._current_gear = MESH_GEAR_FIXED
        tri_num = triangles.shape[0]
        mtb = kwargs.get('material_table', None)
        self.set_material_table(mtb)
        material_ids = kwargs.get('material_ids', None)
        if material_ids is None:
            # use default model
            default_mat = self._material_table['default']
            material_ids = np.full(tri_num, default_mat.uid, dtype=np.int8)
        indx = np.argsort(material_ids, axis=None)
        self._stiffness_multiplier = kwargs.get('stiffness_multiplier', None)
        if np.any(indx!=np.arange(indx.size)):
            triangles = triangles[indx]
            material_ids = material_ids[indx]
            if isinstance(self._stiffness_multiplier, np.ndarray):
                self._stiffness_multiplier = self._stiffness_multiplier[indx]
        self.triangles = triangles
        self._material_ids = material_ids
        self._resolution = kwargs.get('resolution', 4)
        self._epsilon = kwargs.get('epsilon', EPSILON0)
        self._name = kwargs.get('name', '')
        self.token = kwargs.get('token', None)
        self._default_cache = kwargs.get('cache', defaultdict(lambda: True))
        if self.token is None:
            self._hash_token()
        self._caching_keys_dict = {g: None for g in MESH_GEARS}
        self._caching_keys_dict[MESH_GEAR_INITIAL] = self.token
        # store the last caching keys for cleaning up
        self._latest_expired_caching_keys_dict = {g: None for g in MESH_GEARS}
        self._update_caching_keys(gear=MESH_GEAR_FIXED)
        # used for optimizer
        self.locked = kwargs.get('locked', False) # whether to allow modification
        uid = kwargs.get('uid', None)   # numbering of the mesh
        if uid is None:
            self.uid = float(Mesh.uid_counter)
            Mesh.uid_counter += 1
        else:
            self.uid = float(uid)
            Mesh.uid_counter = float(max(Mesh.uid_counter, uid) + 1)


    @classmethod
    def from_PSLG(cls, vertices, segments, markers=None, **kwargs):
        """
        initialize from PSLG (feabas.spatial.Geometry.PSLG).
        Args:
            vertices (Nx2 np.ndarray): vertices of PSLG.
            segments (Mx2 np.ndarray): list of endpoints' vertex id for each
                segment of PSLG.
            markers (dict of lists): marker points for each region names.
        Kwargs:
            mesh_size: the maximum edge length allowed in the mesh.
            min_mesh_angle: minimum angle allowed in mesh. May negatively affect
                meshing performance, default to 0.
        """
        material_table = kwargs.get('material_table', material.MaterialTable())
        resolution = kwargs.get('resolution', 4)
        mesh_size = kwargs.get('mesh_size', (400*4/resolution))
        min_angle = kwargs.get('min_mesh_angle', 0)
        mesh_area = mesh_size ** 2
        regions = []
        holes = []
        regions_no_steiner = []
        if segments is not None:
            PSLG = {'vertices': vertices, 'segments': segments}
            tri_opt = 'p'
        else:
            PSLG = {'vertices': vertices}
            tri_opt = 'c'
        if markers is not None:
            for mname, pts in markers.items():
                if not bool(pts): # no points in this marker
                    continue
                mat = material_table[mname]
                if not mat.enable_mesh:
                    holes.extend(pts)
                else:
                    area_constraint = float(mesh_area * mat.area_constraint)
                    region_id = mat.uid
                    if area_constraint == 0:
                        regions_no_steiner.append(region_id)
                    for rx, ry in pts:
                        regions.append([rx, ry, region_id, area_constraint])
            if bool(holes):
                PSLG['holes'] = holes
            if bool(regions):
                PSLG['regions'] = regions
            tri_opt += 'Aa'
        else:
            if mesh_area > 0:
                num_decimal = max(0, 2-round(np.log10(mesh_area)))
                area_opt = ('a{:.' + str(num_decimal) + 'f}').format(mesh_area)
                if '.' in area_opt:
                    area_opt = area_opt.rstrip('0')
                tri_opt += area_opt
        if min_angle > 0:
            T = triangle.triangulate(PSLG, opts=tri_opt+'q{}'.format(min_angle))
            angle_limited = True
        else:
            T = triangle.triangulate(PSLG, opts=tri_opt)
            angle_limited = False
        if 'triangle_attributes' in T:
            material_ids = T['triangle_attributes'].squeeze().astype(np.int16)
            if angle_limited and bool(regions_no_steiner):
                t_indx = ~np.isin(material_ids, regions_no_steiner)
                tri_keep = np.unique(np.concatenate((T['triangles'][t_indx], np.nonzero(T['vertex_markers'])[0]), axis=None))
                if T['vertices'].shape[0] != tri_keep.shape[0]:
                    v_keep = T['vertices'][tri_keep]
                    indx = np.zeros_like(T['segments'], shape=(T['vertices'].shape[0],))
                    indx[tri_keep] = np.arange(tri_keep.size)
                    seg_keep = indx[T['segments']]
                    PSLG.update({'vertices': v_keep, 'segments': seg_keep})
                    T = triangle.triangulate(PSLG, opts=tri_opt)
                    material_ids = T['triangle_attributes'].squeeze().astype(np.int16)
        else:
            material_ids = None
        vertices = T['vertices']
        triangles = T['triangles']
        connected = np.isin(np.arange(vertices.shape[0]), triangles)
        if not np.all(connected):
            vertices = vertices[connected]
            T_indx = np.full_like(triangles, -1, shape=connected.shape)
            T_indx[connected] = np.arange(np.sum(connected))
            triangles = T_indx[triangles]
        return cls(vertices, triangles, material_ids=material_ids, **kwargs)


    @classmethod
    def from_polygon_equilateral(cls, mask, **kwargs):
        """
        initialize an equilateral mesh that covers the (Multi)Polygon region
        defined by mask.
        """
        resolution = kwargs.get('resolution', 4)
        mesh_size = kwargs.get('mesh_size', (400*4/resolution))
        vertices = spatial.generate_equilat_grid_mask(mask, mesh_size)
        triangles = triangle.delaunay(vertices)
        edges = Mesh.triangle2edge(triangles, directional=True)
        edge_len = np.sum(np.diff(vertices[edges], axis=-2)**2, axis=-1)**0.5
        indx = ~np.any(edge_len.reshape(3, -1) > 1.25 * mesh_size, axis=0)
        triangles = triangles[indx]
        return cls(vertices, triangles, **kwargs)


    @classmethod
    def from_bbox(cls, bbox, **kwargs):
        # generate mesh with rectangular boundary defined by bbox
        # [xmin, ymin, xmax, ymax]
        return cls.from_boarder_bbox(bbox, bd_width=np.inf, roundup_bbox=False, mesh_growth=1.0, **kwargs)


    @classmethod
    def from_boarder_bbox(cls, bbox, bd_width=np.inf, roundup_bbox=True, mesh_growth=3.0, **kwargs):
        """
        rectangular ROI with different boader mesh settings (smaller edgesz +
        regular at boarders), mostly for stitching application.
        Args:
            bbox: [xmin, ymin, xmax, ymax]
        Kwargs:
            bd_width: border width. in pixel or ratio to the size of the bbox
            roundup_bbox(bool): if extend the bounding box size to make it
                multiples of the mesh size. Otherwise, adjust the mesh size
            mesh_growth: increase of the mesh size in the interior region.
        """
        resolution = kwargs.get('resolution', 4)
        mesh_size = kwargs.get('mesh_size', (400*4/resolution))
        tan_theta = np.tan(55*np.pi/180) / 2
        xmin, ymin, xmax, ymax = bbox
        ht = ymax - ymin
        wd = xmax - xmin
        if np.array(bd_width, copy=False).size > 1:
            bd_width_x = bd_width[0]
            bd_width_y = bd_width[1]
        else:
            bd_width_x = bd_width
            bd_width_y = bd_width
        if bd_width_x < 1:
            bd_width_x = bd_width_x * wd
        if bd_width_y < 1:
            bd_width_y = bd_width_y * ht
        if roundup_bbox:
            ht = np.ceil(ht/mesh_size) * mesh_size
            wd = np.ceil(wd/mesh_size) * mesh_size
        bd_width_x = max(bd_width_x, tan_theta*mesh_size*1.01)
        bd_width_y = max(bd_width_y, tan_theta*mesh_size*1.01)
        bd_width_x = min(bd_width_x, wd/2-mesh_size*tan_theta/2)
        bd_width_y = min(bd_width_y, ht/2-mesh_size*tan_theta/2)
        Vx = []
        Vy = []
        segs = []
        crnt_pt = 0
        # vertical quarters
        yp = np.arange(-ht/2, (-ht/2+bd_width_y), tan_theta*mesh_size)
        Np = int(np.ceil(wd/mesh_size) + 1) + 1
        for shrink, y0 in enumerate(yp):
            endpt = wd/2 - shrink*mesh_size*tan_theta
            if endpt > mesh_size*tan_theta*0.6:
                num_t = int(np.ceil(2*endpt/mesh_size) + 1)
                xx = np.linspace(-endpt, endpt, num=num_t, endpoint=True)
                yy = y0 * np.ones_like(xx)
                Nq = xx.size
                Vx.append(xx)
                Vy.append(yy)
                if num_t == Np-1:
                    ss = np.stack((np.arange(0, Nq-1),np.arange(1, Nq)), axis=-1)
                    segs.append(ss + crnt_pt)
                crnt_pt += Nq
                if num_t == Np-1:
                    segs.append(ss + crnt_pt)
                Vx.append(-xx)
                Vy.append(-yy)
                Np = Nq
                crnt_pt += Nq
        # horizontal quarters
        xp = np.arange(-wd/2, (-wd/2+bd_width_x), tan_theta*mesh_size)
        Np = int(np.ceil(ht/mesh_size) + 1) + 1
        for shrink, x0 in enumerate(xp):
            endpt = ht/2 - shrink*mesh_size*tan_theta
            if endpt > mesh_size*tan_theta*0.6:
                num_t = int(np.ceil(2*endpt/mesh_size) + 1)
                yy = np.linspace(-endpt, endpt, num=num_t, endpoint=True)
                xx = x0 * np.ones_like(yy)
                Nq = yy.size
                Vx.append(xx)
                Vy.append(yy)
                if num_t == Np-1:
                    ss = np.stack((np.arange(0, Nq-1),np.arange(1, Nq)), axis=-1)
                    segs.append(ss + crnt_pt)
                crnt_pt += Nq
                if num_t == Np-1:
                    segs.append(ss + crnt_pt)
                Np = Nq
                Vx.append(-xx)
                Vy.append(-yy)
                crnt_pt += Nq
        Vx = np.concatenate(Vx)
        Vy = np.concatenate(Vy)
        vertices = np.stack((Vx, Vy), axis=-1)
        vertices = np.append(vertices, np.array([[0,0]]), axis=0)
        vtx_round = np.round(vertices * 1000 / mesh_size)
        vertices = vertices + np.array((xmin+xmax, ymin+ymax))/2
        _, indx, rindx = np.unique(vtx_round, axis=0,  return_index=True, return_inverse=True)
        vertices = vertices[indx]
        if bool(segs):
            segments = np.concatenate(segs, axis=0)
            segments = rindx[segments]
        else:
            segments = None
        kwargs['mesh_size'] = mesh_size * mesh_growth
        return cls.from_PSLG(vertices, segments, **kwargs)


    def get_init_dict(self, save_material=True, vertex_flags=MESH_GEARS, **kwargs):
        """
        dictionary that can be used for initialization of a duplicate.
        """
        init_dict = {}
        init_dict['vertices'] = self._vertices[MESH_GEAR_INITIAL]
        if (MESH_GEAR_FIXED in vertex_flags) and (self._vertices[MESH_GEAR_FIXED] is not self._vertices[MESH_GEAR_INITIAL]):
            init_dict['fixed_vertices'] = self._vertices[MESH_GEAR_FIXED]
        init_dict['triangles'] = self.triangles
        if (MESH_GEAR_MOVING in vertex_flags) and (self._vertices[MESH_GEAR_MOVING] is not None):
            init_dict['moving_vertices'] = self._vertices[MESH_GEAR_MOVING]
        if (MESH_GEAR_STAGING in vertex_flags) and (self._vertices[MESH_GEAR_STAGING] is not None):
            init_dict['staging_vertices'] = self._vertices[MESH_GEAR_STAGING]
        if np.any(self._offsets[MESH_GEAR_INITIAL]):
            init_dict['initial_offset'] = self._offsets[MESH_GEAR_INITIAL]
        if (MESH_GEAR_FIXED in vertex_flags) and np.any(self._offsets[MESH_GEAR_FIXED]):
            init_dict['fixed_offset'] = self._offsets[MESH_GEAR_FIXED]
        if (MESH_GEAR_MOVING in vertex_flags) and np.any(self._offsets[MESH_GEAR_MOVING]):
            init_dict['moving_offset'] = self._offsets[MESH_GEAR_MOVING]
        if (MESH_GEAR_STAGING in vertex_flags) and np.any(self._offsets[MESH_GEAR_STAGING]):
            init_dict['staging_offset'] = self._offsets[MESH_GEAR_STAGING]
        if self._stiffness_multiplier is not None:
            init_dict['stiffness_multiplier'] = self._stiffness_multiplier
        if save_material:
            init_dict['material_ids'] = self._material_ids
            init_dict['material_table'] = self._material_table.save_to_json()
        init_dict['resolution'] = self._resolution
        init_dict['epsilon'] = self._epsilon
        if bool(self._name):
            init_dict['name'] = self._name
        init_dict['token'] = self.token
        init_dict['uid'] = self.uid
        init_dict.update(kwargs)
        return init_dict


    def _filter_triangles(self, tri_mask):
        """
        give a triangle mask, return a mask to filter the vertices and the
        updated triangle indices
        """
        if Mesh._masked_all(tri_mask):
            vindx = np.arange(self.num_vertices, dtype=self.triangles.dtype)
            T = self.triangles
        else:
            masked_tri = self.triangles[tri_mask]
            vindx = np.unique(masked_tri, axis=None)
            rindx = np.full_like(masked_tri, -1, shape=np.max(vindx)+1)
            rindx[vindx] = np.arange(vindx.size)
            T = rindx[masked_tri]
        return vindx, T


    def submesh(self, tri_mask, save_material=True, append_name=False, **kwargs):
        """
        return a subset of the mesh with tri_mask as the triangle mask.
        """
        if Mesh._masked_all(tri_mask):
            return self
        vindx, new_triangles = self._filter_triangles(tri_mask)
        init_dict = self.get_init_dict(save_material=save_material, **kwargs)
        init_dict['triangles'] = new_triangles
        vtx_keys = ['vertices','fixed_vertices','moving_vertices','staging_vertices']
        for vkey in vtx_keys:
            if init_dict.get(vkey, None) is not None:
                init_dict[vkey] = init_dict[vkey][vindx]
        tri_keys = ['material_ids', 'stiffness_multiplier']
        for tkey in tri_keys:
            if isinstance(init_dict.get(tkey, None), np.ndarray):
                init_dict[tkey] = init_dict[tkey][tri_mask]
        if ('token' not in kwargs) and ('token' in init_dict):
            # do not copy the token from the parent
            init_dict.pop('token')
        if append_name: # append the name with a hash to differentiate from parent
            if ('name' not in kwargs) and ('name' in init_dict):
                parent_name = init_dict['name']
                new_name = (parent_name, miscs.hash_numpy_array(tri_mask))
                init_dict['name'] = new_name
        if ('locked' not in kwargs):
            init_dict['locked'] = self.locked
        return self.__class__(**init_dict)


    def divide_disconnected_mesh(self, save_material=True, **kwargs):
        """
        break the mesh into several submeshes based on triangle connectivity.
        """
        N_conn, T_conn = self.connected_triangles()
        if N_conn == 1:
            return [self]
        else:
            lbls = np.unique(T_conn)
            meshes = []
            uid0 = self.uid
            uids = uid0 + 0.5 * (np.arange(lbls.size) + 1)/(10**(np.ceil(np.log10(lbls.size + 1))))
            for lbl, uid in zip(lbls, uids):
                mask = T_conn == lbl
                meshes.append(self.submesh(mask, save_material=save_material, uid=uid, **kwargs))
            return meshes


    @classmethod
    def combine_mesh(cls, meshes, save_material=True, **kwargs):
        if len(meshes) == 1:
            return meshes[0]
        init_dict = {}
        resolution0 = meshes[0].resolution
        offsets0 = {g: meshes[0].offset(gear=g) for g in MESH_GEARS}
        epsilon0 = meshes[0]._epsilon
        if save_material:
            material_table0 = meshes[0]._material_table.copy()
            material_ids = []
        vertices = {g: [] for g in MESH_GEARS}
        vertices_initialized = {g: False for g in MESH_GEARS}
        triangles = []
        stiffness = []
        num_vertices = 0
        for m in meshes:
            m.change_resolution(resolution0)
            for g in MESH_GEARS:
                v = m.vertices(gear=g) + (m.offset(gear=g) - offsets0[g])
                vertices[g].append(v)
                vertices_initialized[g] |= m.vertices_initialized(gear=g)
            triangles.append(m.triangles + num_vertices)
            num_vertices += m.num_vertices
            stiffness.append(m.stiffness_multiplier)
            epsilon0 = min(epsilon0, m._epsilon)
            if save_material:
                material_table0.combine_material_table(m._material_table)
                material_ids.append(m._material_ids)
        init_dict['vertices'] = np.concatenate(vertices[MESH_GEAR_INITIAL], axis=0)
        init_dict['triangles'] = np.concatenate(triangles, axis=0)
        if vertices_initialized[MESH_GEAR_FIXED]:
            init_dict['fixed_vertices'] = np.concatenate(vertices[MESH_GEAR_FIXED], axis=0)
        if vertices_initialized[MESH_GEAR_MOVING]:
            init_dict['moving_vertices'] = np.concatenate(vertices[MESH_GEAR_MOVING], axis=0)
        if vertices_initialized[MESH_GEAR_STAGING]:
            init_dict['staging_vertices'] = np.concatenate(vertices[MESH_GEAR_STAGING], axis=0)
        if np.any(offsets0[MESH_GEAR_INITIAL]):
            init_dict['initial_offset'] = offsets0[MESH_GEAR_INITIAL]
        if np.any(offsets0[MESH_GEAR_FIXED]):
            init_dict['fixed_offset'] = offsets0[MESH_GEAR_FIXED]
        if np.any(offsets0[MESH_GEAR_MOVING]):
            init_dict['moving_offset'] = offsets0[MESH_GEAR_MOVING]
        if np.any(offsets0[MESH_GEAR_STAGING]):
            init_dict['staging_offset'] = offsets0[MESH_GEAR_STAGING]
        stiffness_multiplier = np.concatenate(stiffness, axis=None)
        if np.ptp(stiffness_multiplier) > 0:
            init_dict['stiffness_multiplier'] = stiffness_multiplier
        else:
            init_dict['stiffness_multiplier'] = stiffness_multiplier[0]
        if save_material:
            init_dict['material_table'] = material_table0.save_to_json()
            init_dict['material_ids'] = np.concatenate(material_ids, axis=None)
        init_dict['resolution'] = resolution0
        init_dict['epsilon'] = epsilon0
        if isinstance(meshes[0]._name, tuple):
            init_dict['name'] = meshes[0]._name[0]
        else:
            init_dict['name'] = meshes[0]._name
        init_dict['uid'] = np.floor(meshes[0].uid)
        init_dict['locked'] = meshes[0].locked
        init_dict.update(kwargs)
        return cls(**init_dict)


    @classmethod
    def from_h5(cls, fname, prefix='', **kwargs):
        init_dict = {}
        if (len(prefix) > 0) and prefix[-1] != '/':
            prefix = prefix + '/'
        if isinstance(fname, h5py.File):
            if not prefix:
                for key in fname.keys():
                    init_dict[key] = fname[key][()]
            else:
                for key in fname[prefix[:-1]].keys():
                    init_dict[key] = fname[prefix+key][()]
        else:
            with h5py.File(fname, 'r') as f:
                if not prefix:
                    for key in f.keys():
                        init_dict[key] = f[key][()]
                else:
                    for key in f[prefix[:-1]].keys():
                        init_dict[key] = f[prefix+key][()]
        init_dict.update(kwargs)
        return cls(**init_dict)


    def save_to_h5(self, fname, vertex_flags=(MESH_GEAR_INITIAL, MESH_GEAR_MOVING),
                   override_dict={}, **kwargs):
        prefix = kwargs.get('prefix', '')
        save_material = kwargs.get('save_material', True)
        compression = kwargs.get('compression', True)
        out = self.get_init_dict(save_material=save_material, vertex_flags=vertex_flags, **override_dict)
        if ('token' in out) and (not 'token' in override_dict):
            out.pop('token') # hash not conistent between runs, no point to save
        if (len(prefix) > 0) and prefix[-1] != '/':
            prefix = prefix + '/'
        if isinstance(fname, h5py.File):
            for key, val in out.items():
                if val is None:
                    continue
                if isinstance(val, str):
                    val = miscs.str_to_numpy_ascii(val)
                if np.isscalar(val) or not compression:
                    _ = fname.create_dataset(prefix+key, data=val)
                else:
                    _ = fname.create_dataset(prefix+key, data=val, compression="gzip")
        else:
            if '.h5' not in fname:
                fname = os.path.join(fname, self.name + '.h5')
            with h5py.File(fname, 'w') as f:
                for key, val in out.items():
                    if val is None:
                        continue
                    if isinstance(val, str):
                        val = miscs.str_to_numpy_ascii(val)
                    if np.isscalar(val) or not compression:
                        _ = f.create_dataset(prefix+key, data=val)
                    else:
                        _ = f.create_dataset(prefix+key, data=val, compression="gzip")


    def copy(self, deep=True, save_material=True, override_dict={}):
        init_dict = self.get_init_dict(save_material=save_material, **override_dict)
        if deep:
            init_dict = copy.deepcopy(init_dict)
        return self.__class__(**init_dict)


  ## -------------------------- manipulate meshes -------------------------- ##
    def delete_vertices(self, vidx):
        """delete vertices indexed by vidx"""
        if isinstance(vidx, np.ndarray) and (vidx.dtype == bool):
            to_keep = ~vidx
        else:
            to_keep = np.ones(self.num_vertices, dtype=bool)
            to_keep[vidx] = False
        if not np.all(to_keep):
            for gear, v in self._vertices.items():
                if v is not None:
                    self._vertices[gear] = v[to_keep]
            indx = np.full(to_keep.size, -1, dtype=self.triangles.dtype)
            indx[to_keep] = np.arange(np.sum(to_keep))
            triangles = indx[self.triangles]
            tidx = np.any(triangles < 0, axis=-1, keepdims=False)
            self.triangles = triangles[tidx]
            if isinstance(self._stiffness_multiplier, np.ndarray):
                self._stiffness_multiplier = self._stiffness_multiplier[tidx]
            if isinstance(self._material_ids, np.ndarray):
                self._material_ids = self._material_ids[tidx]
            self.vertices_changed(gear=MESH_GEAR_INITIAL)


    def delete_orphaned_vertices(self):
        """remove vertices not included in any triangles"""
        connected = np.isin(np.arange(self.num_vertices), self.triangles)
        self.delete_vertices(~connected)


    def change_resolution(self, resolution):
        """modify the resolution of the mesh, scaling the vertices as well"""
        if resolution == self._resolution:
            return
        scale = self._resolution / resolution
        for gear in MESH_GEARS:
            if self._vertices[gear] is not None:
                self.set_vertices(spatial.scale_coordinates(self.vertices(gear=gear), scale), gear=gear)
                self._offsets[gear] = scale * self._offsets[gear]
        self._resolution = resolution
        self._epsilon = self._epsilon * scale


    def set_material_table(self, mtb):
        if isinstance(mtb, str):
            if mtb[-5:] == '.json':
                material_table = material.MaterialTable.from_json(mtb, stream=False)
            else:
                material_table = material.MaterialTable.from_json(mtb, stream=True)
        elif isinstance(mtb, material.MaterialTable):
            material_table = mtb
        elif isinstance(mtb, np.ndarray):
            ss = miscs.numpy_to_str_ascii(mtb)
            if ss[-5:] == '.json':
                material_table = material.MaterialTable.from_json(ss, stream=False)
            else:
                material_table = material.MaterialTable.from_json(ss, stream=True)
        else:
            material_table = material.MaterialTable()
        self._material_table = material_table


    def set_stiffness_multiplier(self, stiffness, tri_mask=None):
        if tri_mask is None:
            self._stiffness_multiplier = stiffness
        else:
            if self._stiffness_multiplier is None:
                self._stiffness_multiplier = np.ones(self.num_triangles)
            elif isinstance(self._stiffness_multiplier, float):
                self._stiffness_multiplier = np.full(self.num_triangles, self._stiffness_multiplier)
            else:
                self._stiffness_multiplier = self._stiffness_multiplier.copy()
            self._stiffness_multiplier[tri_mask] = stiffness
        self._update_caching_keys(gear=MESH_GEAR_INITIAL)


    def set_stiffness_multiplier_from_image(self, img, gear=MESH_GEAR_INITIAL, scale=1.0, tri_mask=None):
        if isinstance(img, str):
            img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        pts = self.triangle_centers(gear=gear, tri_mask=tri_mask) + self.offset(gear=gear)
        pts = np.round(spatial.scale_coordinates(pts, scale=scale))
        indx0 = (pts[:,1].clip(0, img.shape[0]-1)).astype(np.uint16)
        indx1 = (pts[:,0].clip(0, img.shape[1]-1)).astype(np.uint16)
        stiffness = img[indx0, indx1]
        if np.issubdtype(stiffness.dtype, np.integer):
            stiffness = stiffness.astype(float) / np.iinfo(stiffness.dtype).max
        self.set_stiffness_multiplier(stiffness, tri_mask=tri_mask)


    def lock(self):
        self.locked = True


    def unlock(self):
        self.locked = False


  ## ------------------------------ gear switch ---------------------------- ##
    def switch_gear(self, gear):
        if gear in self._vertices:
            self._current_gear = gear
        else:
            raise ValueError


    def switch_to_ini(self):
        self._current_gear = MESH_GEAR_INITIAL


    def switch_to_fix(self):
        self._current_gear = MESH_GEAR_FIXED


    def switch_to_mov(self):
        self._current_gear = MESH_GEAR_MOVING


    def switch_to_stg(self):
        self._current_gear = MESH_GEAR_STAGING


    def __getitem__(self, gear):
        if isinstance(gear, str):
            if gear.lower() in ('m', 'moving'):
                gear = MESH_GEAR_MOVING
            elif gear.lower() in ('f', 'fixed'):
                gear = MESH_GEAR_FIXED
            elif gear.lower() in ('i', 'initial'):
                gear = MESH_GEAR_INITIAL
            elif gear.lower() in ('s', 'staging'):
                gear = MESH_GEAR_STAGING
            else:
                raise KeyError
        self.switch_gear(gear)
        return self


    def vertices(self, gear=None):
        if gear is None:
            gear = self._current_gear
        if gear == MESH_GEAR_INITIAL:
            return self.initial_vertices
        elif gear == MESH_GEAR_FIXED:
            return self.fixed_vertices
        elif gear == MESH_GEAR_MOVING:
            return self.moving_vertices
        elif gear == MESH_GEAR_STAGING:
            return self.staging_vertices
        else:
            raise ValueError


    def vertices_initialized(self, gear=None):
        if gear is None:
            gear = self._current_gear
        return (self._vertices[gear]) is not None


    def offset(self, gear=None):
        if gear is None:
            gear = self._current_gear
        if self._vertices[gear] is None:
            if gear == MESH_GEAR_MOVING:
                return self._offsets[MESH_GEAR_FIXED]
            else:
                return self.offset(gear=MESH_GEAR_MOVING)
        else:
            return self._offsets[gear]


    def center_meshes_w_offsets(self, gear=None):
        if gear is None:
            for g in MESH_GEARS:
                self.center_meshes_w_offsets(gear=g)
        else:
            v = self._vertices[gear]
            if v is not None:
                m = v.mean(axis=0, keepdims=True)
                if np.max(np.abs(m), axis=None) > self._epsilon:
                    self._vertices[gear] = v - m
                    self._offsets[gear] = self._offsets[gear] + m
                    self.vertices_changed(gear=gear)


    def vertices_w_offset(self, gear=None):
        return self.vertices(gear=gear) + self.offset(gear=gear)


    @property
    def initial_vertices(self):
        return self._vertices[MESH_GEAR_INITIAL]


    @property
    def fixed_vertices(self):
        return self._vertices[MESH_GEAR_FIXED]


    @property
    def fixed_vertices_w_offset(self):
        return self.fixed_vertices + self.offset(gear=MESH_GEAR_FIXED)


    @property
    def moving_vertices(self):
        if self._vertices[MESH_GEAR_MOVING] is None:
            return self.fixed_vertices
        else:
            return self._vertices[MESH_GEAR_MOVING]


    @property
    def moving_vertices_w_offset(self):
        return self.moving_vertices + self.offset(gear=MESH_GEAR_MOVING)


    @property
    def staging_vertices(self):
        if self._vertices[MESH_GEAR_STAGING] is None:
            return self.moving_vertices
        else:
            return self._vertices[MESH_GEAR_STAGING]


    @property
    def staging_vertices_w_offset(self):
        return self.staging_vertices + self.offset(gear=MESH_GEAR_STAGING)


  ## -------------------------------- caching ------------------------------ ##
    def _hash_token(self):
        var0 = miscs.hash_numpy_array(self._vertices[MESH_GEAR_INITIAL])
        var1 = miscs.hash_numpy_array(self.triangles)
        var2 = miscs.hash_numpy_array(self._material_ids)
        var3 = miscs.hash_numpy_array(self._stiffness_multiplier)
        self.token = hash((var0, var1, var2, var3, self._resolution))


    def _update_caching_keys(self, gear=MESH_GEAR_INITIAL):
        """
        used to update caching keys when changes are made to the Mesh.
        also keep a copy of old (gear, hash) pairs in case old caches need to be
        freed.
        !!! Note that offsets are not considered here because elastic energies
        are not related to translation. Special care needs to be taken if
        the absolute position of the Mesh is relevant.
        """
        if gear == MESH_GEAR_INITIAL:
            self._hash_token()
            key = self.token
        else:
            v = self.vertices(gear=gear)
            key = miscs.hash_numpy_array(v)
        if key != self._caching_keys_dict[gear]:
            self._latest_expired_caching_keys_dict[gear] = self._caching_keys_dict[gear]
            self._caching_keys_dict[gear] = key


    def caching_keys(self, gear=MESH_GEAR_INITIAL, current_mesh=True):
        """
        hashing of the Mesh object served as the keys for caching. the key has
        the following format:
            (self_token, (gear, hash(self._vertices[gears])))
        gear: specify which vertices the key(s) is associated
        current_mesh: whether to get the latest expired mesh keys or the current
            ones.
        """
        if gear is None:
            gear = MESH_GEAR_INITIAL
        if isinstance(gear, str):
            gear = gear_str_to_constant(gear)
        if not isinstance(gear, tuple):
            gear = (gear,)
        mesh_version = []
        gear_name = []
        for g in gear:
            if g != MESH_GEAR_INITIAL:
                if current_mesh:
                    hashval = self._caching_keys_dict[g]
                else:
                    hashval = self._latest_expired_caching_keys_dict[g]
                mesh_version.append((g, hashval))
                gear_name.append(gear_constant_to_str(g))
        return (self.token, *gear_name, *mesh_version)


    def clear_cached_attr(self, gear=None, gc_now=False):
        prefix = '_cached_'
        if (gear is None) or (gear == MESH_GEAR_INITIAL):
            suffix = ''
        else:
            suffix = gear_constant_to_str(gear)
        attnames_to_delete = []
        for attname in self.__dict__:
            substrs = attname.split('_G_')
            if attname.startswith(prefix) and (len(substrs) > 1) and (suffix in substrs[1]):
                attnames_to_delete.append(attname)
        for attname in attnames_to_delete:
            delattr(self, attname)
        if gc_now:
            gc.collect()


    def clear_specified_caches(self, gear=None, cache=None, include_hash=True, keys_to_probe=None, gc_now=False):
        """
        clear cached properties in the specified cache.
        Note that if gear is not None, the current token is used regardless of
        current_mesh settings.
        Kwargs:
            gear: to clear all the cached properties associated with a specific
                gear. If set to None, only look at the token and clear every gear.
            cache: the cache from with to clear. If set to True, clear local
                attrubutes. If set to None, clear default caches.
            include_hash: if set to True, will also match the hash values
                associated with the gear. Otherwise clear all entries that match
                token and gear.
            keys_to_probe (tuple): specify a token followed by a set of tokens to
                probe. given same token, if one of the token provided is contained
                in one of the key in the cache, free the element associated to
                that key. If None, use the current caching keys as token.
            gc_now: if do garbage collection right away.
        """
        if isinstance(cache, bool):
            if cache:
                self.clear_cached_attr(gear=gear)
            return
        if cache is None:
            if gear is None:
                gear = MESH_GEARS
            elif isinstance(gear, int):
                gear = (gear, )
            elif isinstance(gear, str):
                gear = (gear_str_to_constant(gear),)
            else:
                gear = tuple(gear)
            for g in gear:
                c0 = self._default_cache[g]
                self.clear_specified_caches(gear=g, cache=c0, include_hash=include_hash, keys_to_probe=keys_to_probe)
        else:
            if keys_to_probe is None:
                current_keys = self.caching_keys(gear=gear)
                if include_hash:
                    keys_to_probe = (current_keys[0], *[s for s in current_keys[1:] if isinstance(s, tuple)])
                else:
                    keys_to_probe = (current_keys[0], *[s for s in current_keys[1:] if isinstance(s, str)])
            if isinstance(cache, miscs.CacheNull):
                if len(cache) == 0:
                    return
                keys_to_delete = []
                for key in cache:
                    if (len(keys_to_probe) == 1) and (keys_to_probe[0] == key[0]):
                        keys_to_delete.append(key)
                    else:
                        for token in keys_to_probe[1:]:
                            if token in key[1:]:
                                keys_to_delete.append(key)
                                break
                for key in keys_to_delete:
                    cache._evict_item_by_key(key)
            elif isinstance(cache, dict):
                for c0 in cache.values():
                    self.clear_specified_caches(gear=gear, cache=c0, include_hash=include_hash, keys_to_probe=keys_to_probe)
        if gc_now:
            gc.collect()


    def vertices_changed(self, gear):
        self._update_caching_keys(gear=gear)
        self.clear_cached_attr(gear=gear)
        for g in MESH_GEARS:
            if (g > gear) and (self._vertices[g] is None):
                self.vertices_changed(gear=g)


    def set_default_cache(self, cache=True, gear=None):
        assert (cache is not None)
        if gear is None:
            self._default_cache = defaultdict(lambda: cache)
        else:
            self._default_cache[gear] = cache


  ## ------------------------------ properties ----------------------------- ##
    @property
    def num_vertices(self):
        return self.initial_vertices.shape[0]


    @property
    def num_triangles(self):
        return self.triangles.shape[0]


    @config_cache(MESH_GEAR_INITIAL)
    def edges(self, tri_mask=None):
        """edge indices of the triangulation mesh."""
        if Mesh._masked_all(tri_mask):
            T = self.triangles
        else:
            T = self.triangles[tri_mask]
        edges = Mesh.triangle2edge(T, directional=False)
        return edges


    @config_cache(MESH_GEAR_INITIAL)
    def _edge_to_tid_lut(self):
        """edge to triangle id look-up table."""
        edges = Mesh.triangle2edge(self.triangles, directional=True)
        tids0 = np.arange(edges.shape[0]) % self.num_triangles
        Npt = self.num_vertices
        lut = sparse.csr_matrix((tids0+1, (edges[:,0], edges[:,1])), shape=(Npt, Npt))
        return lut


    def edge_to_tid(self, edges, directed=False, **kwargs):
        lut = self._edge_to_tid_lut(**kwargs)
        if lut is None:
            return None
        tid = np.array(lut[edges[:,0], edges[:,1]]).ravel() - 1
        if not directed:
            tid1 = np.array(lut[edges[:,1], edges[:,0]]).ravel() - 1
            tid = np.stack((tid, tid1), axis=-1)
        return tid


    def segments(self, tri_mask=None, **kwargs):
        """edge indices for edges on the borders."""
        return self.segments_w_triangle_ids(tri_mask=tri_mask, **kwargs)[0]


    @property
    def stiffness_multiplier(self):
        if self._stiffness_multiplier is None:
            return np.ones(self.num_triangles)
        elif isinstance(self._stiffness_multiplier, float):
            return np.full(self.num_triangles, self._stiffness_multiplier)
        else:
            return self._stiffness_multiplier


    @property
    def resolution(self):
        return self._resolution


    @config_cache(MESH_GEAR_INITIAL)
    def segments_w_triangle_ids(self, tri_mask=None):
        """edge indices for edges on the borders, also return the triangle ids"""
        if Mesh._masked_all(tri_mask):
            T = self.triangles
        else:
            T = self.triangles[tri_mask]
        edges = Mesh.triangle2edge(T, directional=True)
        _, indx, cnt = np.unique(np.sort(edges, axis=-1), axis=0, return_index=True, return_counts=True)
        indx = indx[cnt == 1]
        tid = indx % T.shape[0]
        return edges[indx], tid


    @config_cache(MESH_GEAR_INITIAL)
    def vertex_adjacencies(self, vtx_mask=None, tri_mask=None):
        """sparse adjacency matrix of vertices."""
        if Mesh._masked_all(vtx_mask):
            edges = self.edges(tri_mask=tri_mask)
            idx0 = edges[:,0]
            idx1 = edges[:,1]
            V = np.ones_like(idx0, dtype=bool)
            Npt = self.num_vertices
            A = sparse.csr_matrix((V, (idx0, idx1)), shape=(Npt, Npt))
            return A
        else:
            A = self.vertex_adjacencies(vtx_mask=None, tri_mask=tri_mask)
            return A[vtx_mask][:, vtx_mask]


    @config_cache('TBD')
    def vertex_distances(self, gear=MESH_GEAR_INITIAL, vtx_mask=None, tri_mask=None):
        """sparse matrix storing lengths of the edges."""
        if Mesh._masked_all(vtx_mask):
            vertices = self.vertices(gear=gear)
            A = self.vertex_adjacencies(tri_mask=tri_mask)
            idx0, idx1 = A.nonzero()
            edges_len = np.sum((vertices[idx0] - vertices[idx1])**2, axis=-1)**0.5
            Npt = self.num_vertices
            D = sparse.csr_matrix((edges_len, (idx0, idx1)), shape=(Npt, Npt))
            return D
        else:
            D = self.vertex_distances(gear=gear, vtx_mask=None, tri_mask=tri_mask)
            return D[vtx_mask][:, vtx_mask]


    @config_cache(MESH_GEAR_INITIAL)
    def triangle_adjacencies(self, tri_mask=None):
        """
        sparse adjacency matrix of triangles.
        triangles that share an edge are considered adjacent
        """
        if Mesh._masked_all(tri_mask):
            T = self.triangles
        else:
            A0 = self.triangle_adjacencies(tri_mask=None, no_compute=True)
            if A0 is not None:
                return A0[tri_mask][:, tri_mask]
            T = self.triangles[tri_mask]
        edges = np.sort(Mesh.triangle2edge(T, directional=True), axis=-1)
        tids0 = np.arange(edges.shape[0]) % T.shape[0]
        edges_complex = edges[:,0] + edges[:,1] *1j
        idxt = np.argsort(edges_complex)
        tids = tids0[idxt]
        edges_complex = edges_complex[idxt]
        indx = np.nonzero(np.diff(edges_complex)==0)[0]
        idx0 = tids[indx]
        idx1 = tids[indx+1]
        indx0 = np.minimum(idx0, idx1)
        indx1 = np.maximum(idx0, idx1)
        Ntr = T.shape[0]
        V = np.ones_like(idx0, dtype=bool)
        A = sparse.csr_matrix((V, (indx0, indx1)), shape=(Ntr, Ntr))
        return A


    @config_cache('TBD')
    def triangle_centers(self, gear=MESH_GEAR_INITIAL, tri_mask=None):
        """corodinates of the centers of the triangles (Ntri x 2)"""
        if Mesh._masked_all(tri_mask):
            T = self.triangles
        else:
            m0 = self.triangle_centers(gear=gear, tri_mask=None, no_compute=True)
            if m0 is not None:
                return m0[tri_mask]
            T = self.triangles[tri_mask]
        vertices = self.vertices(gear=gear)
        vtri = vertices[T]
        return vtri.mean(axis=1)


    @config_cache('TBD')
    def triangle_bboxes(self, gear=MESH_GEAR_MOVING, tri_mask=None):
        """bounding boxes of triangles as in [xmin, ymin, xmax, ymax]."""
        if tri_mask is None:
            T = self.triangles
        else:
            bboxes0 = self.triangle_bboxes(gear=gear, tri_mask=None, no_compute=True)
            if bboxes0 is not None:
                return bboxes0[tri_mask]
            T = self.triangles[tri_mask]
        vertices = self.vertices(gear=gear)
        V = vertices[T]
        xy_min = V.min(axis=-2)
        xy_max = V.max(axis=-2)
        bboxes = np.concatenate((xy_min, xy_max), axis=-1)
        return bboxes


    @config_cache('TBD')
    def triangle_distances(self, gear=MESH_GEAR_INITIAL, tri_mask=None):
        """sparse matrix storing distances of neighboring triangles."""
        if tri_mask is not None:
            D0 = self.triangle_distances(gear=gear, tri_mask=None, no_compute=True)
            if D0 is not None:
                return D0[tri_mask][:, tri_mask]
        tri_centers = self.triangle_centers(gear=gear, tri_mask=tri_mask)
        A = self.triangle_adjacencies(tri_mask=tri_mask)
        idx0, idx1 = A.nonzero()
        dis = np.sum((tri_centers[idx0] - tri_centers[idx1])**2, axis=-1)**0.5
        D = sparse.csr_matrix((dis, (idx0, idx1)), shape=A.shape)
        return D


    @config_cache(MESH_GEAR_INITIAL)
    def connected_vertices(self, tri_mask=None, local_index=True):
        """
        connected components vertices.
        return as (number_of_components, vertex_labels).
        """
        if tri_mask is not None:
            vtx_mask = np.zeros(self.num_vertices, dtype=bool)
            vtx_idx = np.unique(tri_mask, axis=None)
            vtx_mask[vtx_idx] = True
        else:
            vtx_mask = None
        A = self.vertex_adjacencies(vtx_mask=vtx_mask, tri_mask=tri_mask)
        N_conn, V_conn0 = csgraph.connected_components(A, directed=False, return_labels=True)
        if (tri_mask is not None) and (not local_index):
            V_conn = np.full_like(V_conn0, -1, shape=(self.num_vertices,))
            V_conn[vtx_mask] = V_conn0
        else:
            V_conn = V_conn0
        return N_conn, V_conn


    @config_cache(MESH_GEAR_INITIAL)
    def connected_triangles(self, tri_mask=None):
        """
        connected components of triangles.
        triangles sharing an edge are considered adjacent.
        """
        A = self.triangle_adjacencies(tri_mask=tri_mask)
        N_conn, T_conn = csgraph.connected_components(A, directed=False, return_labels=True)
        return N_conn, T_conn


    @config_cache(MESH_GEAR_INITIAL)
    def grouped_segment_chains(self, tri_mask=None):
        """
        group segments into chains.
        return a list of list of chains. Segment chains belong to the same
        connected regions are first grouped together. Then within each group,
        outer boundary are put at the head, followed by holes.
        """
        sgmnts, tids = self.segments_w_triangle_ids(tri_mask=tri_mask)
        N_conn, T_conn = self.connected_triangles(tri_mask=tri_mask)
        chains = miscs.chain_segment_rings(sgmnts, directed=True, conn_lable=T_conn[tids])
        vertices = self.initial_vertices
        grouped_chains = [[] for _ in range(N_conn)]
        if Mesh._masked_all(tri_mask):
            T = self.triangles
        else:
            T = self.triangles[tri_mask]
        for chain in chains:
            tidx = np.sum((T == chain[0]) + (T == chain[1]), axis=-1) > 1
            cidx = np.max(T_conn[tidx])
            lr = shpgeo.LinearRing(vertices[chain])
            if lr.is_ccw:
                grouped_chains[cidx].insert(0, chain)
            else:
                grouped_chains[cidx].append(chain)
        return grouped_chains


    @config_cache(MESH_GEAR_INITIAL)
    def triangle_mask_for_render(self):
        mid_norender = []
        for _, m in self._material_table:
            if not m.render:
                mid_norender.append(m.uid)
        if len(mid_norender) > 0:
            mask = ~np.isin(self._material_ids, mid_norender)
        else:
            mask = np.ones(self.num_triangles, dtype=bool)
        return mask


    def shapely_regions(self, gear=None, tri_mask=None):
        """
        return the shapely (Multi)Polygon that cover the region of the triangles.
        """
        if gear is None:
            gear = self._current_gear
        grouped_chains = self.grouped_segment_chains(tri_mask=tri_mask)
        vertices = self.vertices_w_offset(gear=gear)
        polygons = []
        for chains in grouped_chains:
            P0 = shpgeo.Polygon(vertices[chains[0]]).buffer(0)
            for hole in chains[1:]:
                P0 = P0.difference(shpgeo.Polygon(vertices[hole]).buffer(0))
            polygons.append(P0)
        return unary_union(polygons)


    @config_cache('TBD')
    def triangle_affine_tform(self, gear=(MESH_GEAR_INITIAL, MESH_GEAR_MOVING), tri_mask=None):
        """
        affine transform matrices for each triangles in mesh.
        Return a tuple (m0, A, m1), so that:
            (vetices[gear0] - m0) = (vertices[gear1] - m1) @ A
        """
        if tri_mask is not None:
            s0 = self.triangle_affine_tform(gear=gear, tri_mask=None, no_compute=True)
            if s0 is not None:
                m0, A, m1 = s0
                return m0[tri_mask], A[tri_mask], m1[tri_mask]
        v0 = self.vertices(gear=gear[0])
        v1 = self.vertices(gear=gear[1])
        if Mesh._masked_all(tri_mask):
            T = self.triangles
        else:
            T = self.triangles[tri_mask]
        T0 = v0[T]
        T1 = v1[T]
        m0 = T0.mean(axis=-2, keepdims=True)
        m1 = T1.mean(axis=-2, keepdims=True)
        T0 = T0 - m0
        T1 = T1 - m1
        m0 = m0.squeeze() + self.offset(gear=gear[0])
        m1 = m1.squeeze() + self.offset(gear=gear[1])
        T0_pad = np.insert(T0, 2, 1, axis=-1)
        T1_pad = np.insert(T1, 2, 1, axis=-1)
        A = np.linalg.solve(T1_pad, T0_pad) # ax = b
        return m0, A, m1


    @config_cache('TBD')
    def triangle_tform_svd(self, gear=(MESH_GEAR_INITIAL, MESH_GEAR_MOVING), tri_mask=None):
        """
        singular values of the affine transforms for each triangle.
        """
        if tri_mask is not None:
            s0 = self.triangle_tform_svd(gear=gear, tri_mask=None, no_compute=True)
            if s0 is not None:
                return s0[tri_mask]
        _, A, _ = self.triangle_affine_tform(gear=gear, tri_mask=tri_mask)
        s = np.linalg.svd(A[:,:2,:2],compute_uv=False)
        det = np.linalg.det(A[:,:2,:2])
        return s * det.reshape(-1,1)


    @config_cache('TBD')
    def triangle_tform_deform(self, gear=(MESH_GEAR_INITIAL, MESH_GEAR_MOVING), tri_mask=None):
        """
        deformation of the affine transforms for each triangle.
        """
        svds0 = self.triangle_tform_svd(gear=gear, tri_mask=tri_mask)
        return Mesh.svds_to_deform(svds0)


    @config_cache('TBD')
    def tri_info(self, gear=None, tri_mask=None, include_flipped=False, contigeous=True):
        # return geometry STRtree, matplotlib tri list, global index list and border segment STRtree
        if gear is None:
            gear = self._current_gear
        groupings = self.nonoverlap_triangle_groups(gear=gear, contigeous=contigeous, include_flipped=include_flipped, tri_mask=tri_mask)
        geometry_list = []
        mattri_list = []
        tindex_list = []
        vindex_list = []
        segs = []
        seg_tids = []
        group_ids = np.unique(groupings[groupings >= 0])
        for g in group_ids:
            g_mask = groupings == g
            if not np.any(g_mask):
                continue
            geometry_list.append(self.shapely_regions(gear=gear, tri_mask=g_mask))
            mpl_tri, v_indx, _ = self.mpl_tri(gear=gear, tri_mask=g_mask)
            mattri_list.append(mpl_tri)
            tindex_list.append(np.nonzero(g_mask)[0])
            vindex_list.append(v_indx)
            seg0, seg_tid0 = self.segments_w_triangle_ids(tri_mask=g_mask)
            segs.append(seg0)
            seg_tids.append(Mesh.masked_index_to_global_index(g_mask, seg_tid0))
        region_tree = shapely.STRtree(geometry_list)
        segs = np.concatenate(segs, axis=0)
        seg_tids = np.concatenate(seg_tids, axis=None)
        vertices = self.vertices(gear=gear)
        lines = [shpgeo.LineString(vertices[s]) for s in segs]
        seg_tree = shapely.STRtree(lines)
        tri_info = {'region_tree': region_tree, 'matplotlib_tri': mattri_list,
            'triangle_index': tindex_list, 'vertex_index': vindex_list,
            'segment_tree': seg_tree, 'segment_tid': seg_tids}
        return tri_info


    def mpl_tri(self, gear=None, tri_mask=None):
        v_indx, new_T = self._filter_triangles(tri_mask)
        vertices = self.vertices(gear=gear)[v_indx]
        mpl_tri = matplotlib.tri.Triangulation(vertices[:,0], vertices[:,1], triangles=new_T)
        return mpl_tri, v_indx, new_T


    @config_cache('TBD')
    def triangle_collisions(self, gear=None, tri_mask=None):
        return self.find_triangle_overlaps(gear=gear, tri_mask=tri_mask)


    @property
    def name(self):
        if isinstance(self._name, str):
            return self._name
        else:
            return '_sub_'.join(str(s) for s in self._name)


    def __bool__(self):
        return self.num_triangles > 0


  ## -------------------------------- query -------------------------------- ##
    def tri_finder(self, pts, gear=None, tri_mask=None, **kwargs):
        """
        given a set of points, find which triangles they are in.
        Args:
            pts (N x 2 ndarray): x-y coordinates of querry points
        """
        include_flipped = kwargs.get('inner_cache', False)
        mode = kwargs.get('mode', MESH_TRIFINDER_LEAST_DEFORM)
        contigeous = kwargs.get('contigeous', True)
        extrapolate = kwargs.get('extrapolate', False)
        inner_cache = kwargs.get('inner_cache', None)
        if gear is None:
            gear = self._current_gear
        tri_info = self.tri_info(gear=gear, tri_mask=tri_mask, include_flipped=include_flipped, contigeous=contigeous, cache=inner_cache)
        tree = tri_info['region_tree']
        mattri_list = tri_info['matplotlib_tri']
        index_list = tri_info['triangle_index']
        seg_tree = tri_info['segment_tree']
        seg_tids = tri_info['segment_tid']
        pts = (pts - self.offset(gear=gear)).reshape(-1,2)
        if len(mattri_list) > 1:
            mpts = shpgeo.MultiPoint(pts)
            pts_list = list(mpts.geoms)
            hits = tree.query(pts_list, predicate='intersects')
        else:
            hits = np.tile(np.arange(pts.shape[0]), (2,1))
            hits[1] *= 0
        tid_out = np.full(pts.shape[0], -1, dtype=self.triangles.dtype)
        if hits.size == 0:
            return tid_out
        hits_gidx_u = np.unique(hits[1])
        tri_finders = {k: m.get_trifinder() for k, m in enumerate(mattri_list) if k in hits_gidx_u}
        if len(mattri_list) > 1:
            uhits, uidx, cnts = np.unique(hits[0], return_index=True, return_counts=True)
            conflict = np.any(cnts > 1)
            if conflict:
                if mode == MESH_TRIFINDER_WHATEVER:
                    hits = hits[:, uidx]
                    conflict = False
                elif mode == MESH_TRIFINDER_INNERMOST:
                    conflict_pts_indices = uhits[cnts > 1]
                    for pt_idx in conflict_pts_indices:
                        pmask = hits[0] == pt_idx
                        geo_indices = hits[1, pmask]
                        mxdis = -1
                        g_sel = geo_indices[0]
                        for g in geo_indices:
                            dis0 = tree.geometries[g].boundary.distance(pts_list[pt_idx])
                            if dis0 >= mxdis:
                                g_sel = g
                                mxdis = dis0
                        hits[1, pmask] = g_sel
                    hits = hits[:, uidx]
                    conflict = False
        else:
            conflict = False
        hits_pidx = hits[0]
        hits_gidx = hits[1]
        pts_indices = []
        tri_indices = []
        for gindx in np.unique(hits_gidx):
            pidx = hits_pidx[hits_gidx == gindx]
            pxy = pts[pidx]
            tid0 = tri_finders[gindx](pxy[:,0], pxy[:,1])
            pts_indices.append(pidx[tid0 >= 0])
            tri_indices.append(index_list[gindx][tid0[tid0 >= 0]])
        pts_indices = np.concatenate(pts_indices, axis=0)
        tri_indices = np.concatenate(tri_indices, axis=0)
        if conflict:
            if mode == MESH_TRIFINDER_LEAST_DEFORM:
                deforms0 = self.triangle_tform_deform(gear=(MESH_GEAR_INITIAL, gear), tri_mask=None)
                deforms = deforms0[tri_indices]
                idxt = np.argsort(deforms)
                pts_indices = pts_indices[idxt]
                tri_indices = tri_indices[idxt]
                pts_indices, uidx = np.unique(pts_indices, return_index=True)
                tri_indices = tri_indices[uidx]
            else:
                raise ValueError("Mesh tri_finder conflict resolution mode not implemented")
        tid_out[pts_indices] = tri_indices
        if extrapolate and np.any(tid_out == -1):
            mepts = shpgeo.MultiPoint(pts[tid_out == -1])
            epts_list = list(mepts.geoms)
            nearest_segs = seg_tree.nearest(epts_list)
            etids = seg_tids[nearest_segs]
            tid_out[tid_out == -1] = etids
        return tid_out


    def cart2bary(self, xy, gear, tid=None, **kwargs):
        """Cartesian to Barycentric coordinates"""
        if tid is None:
            tid = self.tri_finder(xy, gear=gear, **kwargs)
        vertices = self.vertices(gear=gear)
        tri_pt = vertices[np.atleast_2d(self.triangles[tid,:])]
        tri_pt_m = tri_pt.mean(axis=1, keepdims=True)
        tri_pt = tri_pt - tri_pt_m
        xy = xy - self.offset(gear=gear) - tri_pt_m
        ss = max(tri_pt.std(), 0.01)
        tri_pt_pad = np.insert(tri_pt / ss, 2, 1, axis=-1)
        xy_pad = np.insert(xy / ss, 2, 1, axis=-1)
        B = np.linalg.solve(np.swapaxes(tri_pt_pad, -2, -1), xy_pad)
        B[tid==-1] = np.nan
        return tid, B


    def bary2cart(self, tid, B, gear, offsetting=True):
        """Barycentric to Cartesian coordinates"""
        indx = np.atleast_2d(self.triangles[tid,:])
        if offsetting:
            tri_pt = self.vertices_w_offset(gear=gear)[indx]
        else:
            tri_pt = self.vertices(gear=gear)[indx]
        xy = np.sum(tri_pt * B.reshape(-1,3,1), axis=-2, keepdims=False)
        return xy


  ## -------------------------- transformations ---------------------------- ##
    def set_vertices(self, v, gear, vtx_mask=None):
        if self.locked:
            return
        if Mesh._masked_all(vtx_mask):
            self._vertices[gear] = v
        else:
            self._vertices[gear] = self.vertices(gear=gear).copy()
            self._vertices[gear][vtx_mask] = v
        self.vertices_changed(gear=gear)


    def set_offset(self, offset, gear):
        if self.locked:
            return
        self._offsets[gear] = offset


    @fixed_vertices.setter
    def fixed_vertices(self, v):
        self._vertices[MESH_GEAR_FIXED] = v
        self.vertices_changed(gear=MESH_GEAR_FIXED)


    @moving_vertices.setter
    def moving_vertices(self, v):
        if self.locked:
            return
        self._vertices[MESH_GEAR_MOVING] = v
        self.vertices_changed(gear=MESH_GEAR_MOVING)


    @staging_vertices.setter
    def staging_vertices(self, v):
        if self.locked:
            return
        self._vertices[MESH_GEAR_STAGING] = v
        self.vertices_changed(gear=MESH_GEAR_STAGING)


    def apply_translation(self, dxy, gear, vtx_mask=None):
        dxy = np.array(dxy, copy=False).reshape(1,2)
        if self.locked:
            return
        if not np.any(dxy, axis=None):
            return
        v = self.vertices(gear=gear)
        self._vertices[gear] = v
        offset = self.offset(gear=gear)
        if Mesh._masked_all(vtx_mask):
            self.set_offset(offset + dxy, gear=gear)
        else:
            self.set_vertices(v[vtx_mask] + dxy, gear=gear, vtx_mask=vtx_mask)
            self.set_offset(offset, gear=gear)


    def set_translation(self, dxy, gear=(MESH_GEAR_FIXED, MESH_GEAR_MOVING), vtx_mask=None):
        if self.locked:
            return
        if gear[0] == gear[-1]:
            self.apply_translation(dxy, gear[0], vtx_mask=vtx_mask)
            return
        dxy = np.array(dxy, copy=False).reshape(1,2)
        v0 = self.vertices(gear=gear[0])
        offset0 = self.offset(gear=gear[0])
        if Mesh._masked_all(vtx_mask):
            self.set_vertices(v0, gear=gear[-1], vtx_mask=None)
            self.set_offset(offset0 + dxy, gear=gear[-1])
        else:
            offset1 = self.offset(gear=gear[-1])
            dxy = dxy + offset0 - offset1
            v1 = v0[vtx_mask] + dxy
            self.set_vertices(v1, gear=gear[-1], vtx_mask=vtx_mask)
            self.set_offset(offset1, gear=gear[-1])


    def estimate_translation(self, gear=(MESH_GEAR_FIXED, MESH_GEAR_MOVING), vtx_mask=None):
        if gear[0] == gear[-1]:
            return np.zeros(2)
        offset0 = self.offset(gear=gear[0])
        offset1 = self.offset(gear=gear[-1])
        v0 = self.vertices(gear=gear[0])
        v1 = self.vertices(gear=gear[-1])
        if vtx_mask is not None:
            v0 = v0[vtx_mask]
            v1 = v1[vtx_mask]
        dxy = v1.mean(axis=0) - v0.mean(axis=0)
        dof = offset1 - offset0
        return dxy.ravel() - dof.ravel()


    def apply_affine(self, A, gear, vtx_mask=None):
        if self.locked:
            return
        if np.all(A == np.eye(3)):
            return
        v0 = self.vertices(gear=gear)
        offset0 = self.offset(gear=gear)
        if Mesh._masked_all(vtx_mask):
            v1 = v0 @ A[:-1,:-1]
            offset1 = offset0 @ A[:-1,:-1] + A[-1,:-1]
            self.set_vertices(v1, gear=gear, vtx_mask=None)
            self.set_offset(offset1, gear=gear)
        else:
            v1 = v0[vtx_mask] @ A[:-1,:-1] + offset0 @ A[:-1,:-1] + A[-1,:-1] - offset0
            self.set_vertices(v1, gear=gear, vtx_mask=vtx_mask)
            self.set_offset(offset0, gear=gear)


    def set_affine(self, A, gear=(MESH_GEAR_FIXED, MESH_GEAR_MOVING), vtx_mask=None):
        if self.locked:
            return
        if gear[0] == gear[-1]:
            self.apply_affine(A, gear[0], vtx_mask=vtx_mask)
            return
        v0 = self.vertices(gear=gear[0])
        offset0 = self.offset(gear=gear[0])
        if Mesh._masked_all(vtx_mask):
            v1 = v0 @ A[:-1,:-1]
            offset1 = offset0 @ A[:-1,:-1] + A[-1,:-1]
            self.set_vertices(v1, gear=gear[-1], vtx_mask=None)
            self.set_offset(offset1, gear=gear[-1])
        else:
            offset1 = self.offset(gear=gear[-1])
            v1 = v0[vtx_mask] @ A[:-1,:-1] + offset0 @ A[:-1,:-1] + A[-1,:-1] - offset1
            self.set_vertices(v1, gear=gear[-1], vtx_mask=vtx_mask)
            self.set_offset(offset1, gear=gear[-1])


    def apply_field(self, dxy, gear, vtx_mask=None):
        if self.locked:
            return
        if not np.any(dxy, axis=None):
            return
        v0 = self.vertices(gear=gear)
        offset0 = self.offset(gear=gear)
        if Mesh._masked_all(vtx_mask):
            m = np.mean(dxy.reshape(-1,2), axis=0, keepdims=True)
            v1 = v0 + (dxy - m)
            offset1 = offset0 + m
            self.set_vertices(v1, gear=gear, vtx_mask=None)
            self.set_offset(offset1, gear=gear)
        else:
            v1 = v0[vtx_mask] + dxy
            self.set_vertices(v1, gear=gear, vtx_mask=vtx_mask)
            self.set_offset(offset0, gear=gear)


    def set_field(self, dxy, gear=(MESH_GEAR_FIXED, MESH_GEAR_MOVING), vtx_mask=None):
        if self.locked:
            return
        if gear[0] == gear[-1]:
            self.apply_field(dxy, gear[0], vtx_mask=vtx_mask)
            return
        v0 = self.vertices(gear=gear[0])
        offset0 = self.offset(gear=gear[0])
        if Mesh._masked_all(vtx_mask):
            m = np.mean(dxy.reshape(-1,2), axis=0, keepdims=True)
            v1 = v0 + (dxy - m)
            offset1 = offset0 + m
            self.set_vertices(v1, gear=gear[-1], vtx_mask=None)
            self.set_offset(offset1, gear=gear[-1])
        else:
            offset1 = self.offset(gear=gear[-1])
            v1 = v0[vtx_mask] + dxy + offset0 - offset1
            self.set_vertices(v1, gear=gear[-1], vtx_mask=vtx_mask)
            self.set_offset(offset1, gear=gear[-1])


    def anneal(self, gear=(MESH_GEAR_MOVING, MESH_GEAR_FIXED), mode=ANNEAL_CONNECTED_RIGID):
        """
        adjust the fixed vertices closer to the moving vertices as the new resting
        state for relaxation.
        """
        if self.locked:
            return
        if mode in (ANNEAL_GLOBAL_RIGID, ANNEAL_GLOBAL_AFFINE):
            v0 = self.vertices_w_offset(gear=gear[0])
            v1 = self.vertices_w_offset(gear=gear[1])
            if mode == ANNEAL_CONNECTED_RIGID:
                _, R = spatial.fit_affine(v0, v1, return_rigid=True)
                self.apply_affine(R, gear[1])
            else:
                A = spatial.fit_affine(v0, v1, return_rigid=False)
                self.apply_affine(A, gear[1])
        elif mode in (ANNEAL_CONNECTED_RIGID, ANNEAL_CONNECTED_AFFINE):
            N_conn, V_conn = self.connected_vertices()
            self.anneal(gear=gear, mode=ANNEAL_GLOBAL_RIGID) # center the mesh
            if (N_conn == 1) and (mode == ANNEAL_GLOBAL_RIGID):
                return
            v0 = self.vertices_w_offset(gear=gear[0])
            v1 = self.vertices_w_offset(gear=gear[1])
            for cid in range(N_conn):
                idx = V_conn == cid
                if mode == ANNEAL_CONNECTED_RIGID:
                    _, R = spatial.fit_affine(v0[idx], v1[idx], return_rigid=True)
                    self.apply_affine(R, gear[1], vtx_mask=idx)
                else:
                    A = spatial.fit_affine(v0[idx], v1[idx], return_rigid=False)
                    self.apply_affine(A, gear[1], vtx_mask=idx)
        elif mode == ANNEAL_COPY_EXACT:
            offset0 = self.offset(gear=gear[0])
            v0 = self.vertices(gear=gear[0])
            self.set_vertices(v0, gear[1])
            self.set_offset(offset0, gear[1])
        else:
            raise ValueError


  ## ------------------------ collision management ------------------------- ##
    def is_valid(self, gear=None, tri_mask=None):
        if gear is None:
            gear = self._current_gear
        vertices = self.vertices(gear=gear)
        if Mesh._masked_all(tri_mask):
            T = self.triangles
        else:
            vindx, T = self._filter_triangles(tri_mask)
            vertices = vertices[vindx]
        matplt_tri = matplotlib.tri.Triangulation(vertices[:,0], vertices[:,1], triangles = T)
        try:
            matplt_tri.get_trifinder()
            return True
        except RuntimeError:
            return False


    def _triangles_rtree_generator(self, gear=None, tri_mask=None):
        if gear is None:
            gear = self._current_gear
        if Mesh._masked_all(tri_mask):
            tids = np.arange(self.num_triangles)
        elif tri_mask.dtype == bool:
            tids = np.nonzero(tri_mask)[0]
        else:
            tids = np.sort(tri_mask)
        for k, bbox in enumerate(self.triangle_bboxes(gear=gear, tri_mask=tri_mask)):
            yield (k, bbox, tids[k])


    def triangles_rtree(self, gear=None, tri_mask=None):
        if gear is None:
            gear = self._current_gear
        return index.Index(self._triangles_rtree_generator(gear=gear, tri_mask=tri_mask))


    def check_segment_collision(self, gear=None, tri_mask=None):
        """check if segments have collisions among themselves."""
        if gear is None:
            gear = self._current_gear
        vertices = self.vertices(gear=gear)
        SRs = self.grouped_segment_chains(tri_mask=tri_mask)
        covered = None
        valid = True
        for sr in SRs:
            outL = vertices[sr[0]]
            if len(sr) > 1:
                holes = [vertices[s] for s in sr[1:]]
            else:
                holes = None
            p = shpgeo.Polygon(outL, holes=holes)
            if not p.is_valid:
                valid = False
                break
            if covered is None:
                covered = p
            elif p.intersects(covered):
                valid = False
                break
            else:
                covered = covered.union(p)
        return valid


    def locate_segment_collision(self, gear=None, tri_mask=None, check_flipped=True):
        """
        find the segments that collide. Return a list of collided segments and
        their (local) triangle ids.
        """
        if gear is None:
            gear = self._current_gear
        SRs = self.grouped_segment_chains(tri_mask=tri_mask)
        vertices = self.vertices(gear=gear)
        boundaries = []
        polygons_rings = []
        thickened = []
        for sr in SRs:
            for line_indx in sr:
                line = shpgeo.LinearRing(vertices[line_indx])
                boundaries.append(line)
                polygons_rings.append(shpgeo.Polygon(line))
                if shapely.__version__ >= '2.0.0':
                    line = shpgeo.LinearRing(vertices[np.append(line_indx, line_indx[0])])
                    line_t = line.buffer(self._epsilon, join_style=2, single_sided=True)
                    thickened.append(line_t)
                else:
                    line_t = line.parallel_offset(self._epsilon, 'left', join_style=2)
                    thickened.append(line_t.buffer(-self._epsilon, single_sided=True, join_style=2))
        polygons_partition = list(polygonize(unary_union(boundaries)))
        polygons_tokeep = []
        for pp in polygons_partition:
            pts = pp.representative_point()
            winding_number = 0
            for pr in polygons_rings:
                if pr.contains(pts):
                    winding_number += 1
            if winding_number % 2:
                polygons_tokeep.append(pp)
        polygon_odd_wind = unary_union(polygons_tokeep + thickened)
        polygon_odd_wind = polygon_odd_wind.buffer(self._epsilon/10, join_style=2)
        polygon_odd_wind = polygon_odd_wind.buffer(-self._epsilon/2, join_style=2, mitre_limit=10)
        segments, tids0 = self.segments_w_triangle_ids(tri_mask=tri_mask)
        collided_segments = []
        rest_of_segments = []
        for seg, tid in zip(segments, tids0):
            lineseg = shpgeo.LineString(vertices[seg])
            if polygon_odd_wind.intersects(lineseg):
                collided_segments.append((seg, tid))
            else:
                rest_of_segments.append((seg, tid))
        if check_flipped:
            # check if exist flipped triangles where all three points on segments
            Tseg_flag = np.sum(np.isin(self.triangles, segments), axis=-1) == 3
            if Mesh._masked_all(tri_mask):
                _, T = self.locate_flipped_triangles(gear=gear, tri_mask=Tseg_flag, return_triangles=True)
            else:
                _, T = self.locate_flipped_triangles(gear=gear, tri_mask=(Tseg_flag & tri_mask), return_triangles=True)
            if T.size > 0:
                S_flp = Mesh.triangle2edge(T, directional=True)
                for segwid in rest_of_segments:
                    if np.any(np.all(S_flp==segwid[0], axis=-1)):
                        collided_segments.append(segwid)
        segs = np.array([s[0] for s in collided_segments], dtype=segments.dtype)
        tids = np.array([s[1] for s in collided_segments], dtype=tids0.dtype)
        return segs, tids


    def locate_flipped_triangles(self, gear=None, tri_mask=None, return_triangles=False):
        if gear is None:
            gear = self._current_gear
        vertices0 = self.initial_vertices
        if Mesh._masked_all(tri_mask):
            T = self.triangles
        else:
            T = self.triangles[tri_mask]
        vertices = self.vertices(gear=gear)
        A0 = miscs.signed_area(vertices0, T)
        A1 = miscs.signed_area(vertices, T)
        flipped_sel = (A0 * A1) <= 0
        if return_triangles:
            return np.nonzero(flipped_sel)[0], T[flipped_sel]
        else:
            return np.nonzero(flipped_sel)[0]


    def find_triangle_overlaps(self, gear=None, tri_mask=None):
        if gear is None:
            gear = self._current_gear
        if self.is_valid(gear=gear, tri_mask=tri_mask):
            return np.empty((0,2), dtype=self.triangles.dtype)
        collided_segs, _ = self.locate_segment_collision(gear=gear, tri_mask=tri_mask, check_flipped=False)
        if collided_segs.size > 0:
            seg_lines = self.vertices(gear=gear)[collided_segs]
            P_segs = list(polygonize(unary_union(shpgeo.MultiLineString([s for s in seg_lines]))))
            seg_bboxes = np.array([p.bounds for p in P_segs])
        else:
            seg_bboxes = np.empty((0,4))
        flip_tids = self.locate_flipped_triangles(gear=gear, tri_mask=tri_mask)
        if flip_tids.size > 0:
            flip_tids_g = Mesh.masked_index_to_global_index(tri_mask, flip_tids)
            flip_bboxes = self.triangle_bboxes(gear=gear, tri_mask=flip_tids_g)
        else:
            flip_bboxes = np.empty((0,4))
        init_bboxes = np.concatenate((seg_bboxes, flip_bboxes), axis=0)
        if init_bboxes.size == 0:
            return np.empty((0,2), dtype=self.triangles.dtype)
        rtree0 = self.triangles_rtree(gear=gear, tri_mask=tri_mask)
        candidate_tids = []
        for bbox in init_bboxes:
            bbox_t = (bbox[0]-self._epsilon, bbox[1]-self._epsilon, bbox[2]+self._epsilon, bbox[3]+self._epsilon)
            candidate_tids.extend(list(rtree0.intersection(bbox_t, objects=False)))
        candidate_tids = np.unique(candidate_tids)
        tri_mask_c = Mesh.masked_index_to_global_index(tri_mask, candidate_tids)
        rtree_c = self.triangles_rtree(gear=gear, tri_mask=tri_mask_c)
        vertices = self.vertices(gear=gear)[self.triangles[tri_mask_c]]
        Ts = [shpgeo.Polygon(v) for v in vertices]
        collisions = []
        for tid0, t0 in enumerate(Ts):
            hits = rtree_c.intersection(t0.bounds)
            for tid1 in hits:
                if tid0 == tid1:
                    continue
                if t0.intersects(Ts[tid1]) and t0.intersection(Ts[tid1]).area > 0:
                    collisions.append((tid0, tid1))
            rtree_c.delete(tid0, t0.bounds)
        return candidate_tids[np.array(collisions)]


    def _graph_coloring_overlapped_triangles(self, collisions=None, gear=None, tri_mask=None, include_flipped=False):
        if gear is None:
            gear = self._current_gear
        if collisions is None:
            collisions = self.triangle_collisions(gear=gear, tri_mask=tri_mask)
        if Mesh._masked_all(tri_mask):
            groupings = np.zeros(self.num_triangles, dtype=self.triangles.dtype)
        elif isinstance(tri_mask, np.ndarray) and (tri_mask.dtype == bool):
            groupings = np.zeros(np.sum(tri_mask), dtype=self.triangles.dtype)
        else:
            tri_mask = miscs.indices_to_bool_mask(tri_mask, size=self.num_triangles)
            groupings = np.zeros(np.sum(tri_mask), dtype=self.triangles.dtype)
        if collisions.size == 0:
            return groupings
        indx_loc, collisions_loc = np.unique(collisions, axis=None, return_inverse=True)
        collisions_loc = collisions_loc.reshape(collisions.shape)
        indx_glb = Mesh.masked_index_to_global_index(tri_mask, indx_loc)
        svds = self.triangle_tform_svd(gear=(MESH_GEAR_INITIAL, gear), tri_mask=indx_glb)
        deforms = Mesh.svds_to_deform(svds)
        # graph coloring for grouping
        order = np.argsort(deforms) # start from good ones
        deforms_ordered = deforms[order]
        order_nonflip = order[deforms_ordered<1]
        order_flip = order[deforms_ordered>=1]
        N_ov = order.size
        colors = np.full(N_ov, -1, dtype=self.triangles.dtype)
        G = sparse.csr_matrix((np.ones(collisions.shape[0], dtype=bool), (collisions_loc[:,0],collisions_loc[:,1])), shape=(N_ov, N_ov))
        G = G + G.transpose()
        available_color0 = np.ones(N_ov+1, dtype=bool)
        for t0 in order_nonflip:
            available_color = available_color0.copy()
            _, neighbors = G[t0].nonzero()
            connected_color = colors[neighbors]
            available_color[connected_color] = False
            group_id = np.min(np.nonzero(available_color)[0])
            colors[t0] = group_id
        if include_flipped:
            available_color0[:(colors.max()+1)] = False
            for t0 in order_flip:
                available_color = available_color0.copy()
                _, neighbors = G[t0].nonzero()
                connected_color = colors[neighbors]
                available_color[connected_color] = False
                group_id = np.min(np.nonzero(available_color)[0])
                colors[t0] = group_id
        else:
            colors[order_flip] = -1
        groupings[indx_loc] = colors
        return groupings


    @config_cache('TBD')
    def nonoverlap_triangle_groups(self, gear=MESH_GEAR_MOVING, contigeous=True, include_flipped=False, tri_mask=None):
        """
        devide triangles to subgroups so that within each group there are no
        overlapping triangles. This prevents error when using matplotlib.tri to
        generate trapezoidal map.
        Kwargs:
            gear (int): which gear the groupings should ensure no overlaps in.
            contigeous (bool): whether to break unconnected triangles into
                different groups, even if no overlaps occur.
            include_flipped (bool): whether to also group flipped triangles. If
                False, the group id for all the flipped triangles will be -1.
        """
        if contigeous:
            N_conn0, T_conn0 = self.connected_triangles(tri_mask=tri_mask)
            groupings0 = np.full(self.num_triangles, -1, dtype=self.triangles.dtype)
            num_groups0 = 0
            for lbl in range(N_conn0):
                l_mask = T_conn0 == lbl
                if tri_mask is not None:
                    l_mask = np.nonzero(l_mask)[0]
                    l_mask = Mesh.masked_index_to_global_index(tri_mask, l_mask)
                if self.is_valid(gear=gear, tri_mask=l_mask):
                    grp = np.full(self._num_masked_tri(l_mask), num_groups0, dtype=groupings0.dtype)
                else:
                    grp = self._graph_coloring_overlapped_triangles(gear=gear, tri_mask=l_mask, include_flipped=include_flipped)
                    grp[grp >= 0] += num_groups0
                groupings0[l_mask] = grp
                num_groups0 = grp.max() + 1
            groupings = groupings0.copy()
            lbls = np.unique(groupings0[groupings0 >= 0])
            num_groups = 0
            for lbl in lbls:
                l_mask = groupings0 == lbl
                N_conn, T_conn = self.connected_triangles(tri_mask=l_mask)
                groupings[l_mask] = T_conn + num_groups
                num_groups += N_conn
        else:
            if self.is_valid(gear=gear, tri_mask=tri_mask):
                grp = np.zeros(self._num_masked_tri(tri_mask), dtype=self.triangles.dtype)
            else:
                grp = self._graph_coloring_overlapped_triangles(gear=gear, tri_mask=tri_mask, include_flipped=include_flipped)
            if tri_mask is not None:
                groupings = np.full(self.num_triangles, -1, dtype=self.triangles.dtype)
                groupings[tri_mask] = grp
            else:
                groupings = grp
        return groupings


    def _smooth_mesh_by_regular_resample(self, gear=MESH_GEAR_MOVING, tri_mask=None, decimate=16):
        covered_region = self.shapely_regions(gear=MESH_GEAR_INITIAL, tri_mask=tri_mask)
        num_tri0 = self._num_masked_tri(tri_mask=tri_mask)
        avg_tri_area0 = covered_region.area / num_tri0
        mesh_size = (avg_tri_area0 * 2.31) ** 0.5 * decimate
        coarse_mesh = Mesh.from_polygon_equilateral(covered_region, mesh_size=mesh_size, resolution=self._resolution)
        NotImplemented


    def fix_segment_collision(self):
        NotImplemented


  ## ------------------------- stiffness matrices -------------------------- ##
    @config_cache('TBD')
    def stiffness_shape_matrices(self, gear=MESH_GEAR_FIXED):
        material_table = self._material_table.id_table
        material_ids = self._material_ids
        v = self.vertices(gear=gear)
        shape_matrices = {}
        for mid in np.unique(material_ids):
            mat = material_table[mid]
            indx = np.nonzero(material_ids == mid)[0]
            tripts = v[self.triangles[indx]]
            B, area = mat.shape_matrix_from_vertices(tripts)
            shape_matrices[mid] = (indx, (B, area))
        return shape_matrices


    @config_cache('TBD')
    def element_stiffness_matrices(self, gear=(MESH_GEAR_FIXED, MESH_GEAR_MOVING), inner_cache=None, **kwargs):
        """
        compute the stiffness matrices for each element. Each element follow the order
            [u1, v1, u2, v2, u3, v3]
        Kwargs:
            gear(tuple): first item used for shape matrices, second gear for stress
                computation (and stiffness if nonlinear).
            inner_cache: the cache to store intermediate attributes like shape
                matrices. Use default if set to None.
            check_flip(bool): check if any triangles are flipped.
            continue_on_flip(bool): whether to continue with flipped triangles
                detected.
        Return:
            F1 (list): element_stiffness(Nx6x6), computed at the current displacement.
            indices (list): the triangle indices (N,) for each element.
            F0 (list): stress(Nx6x1) used in Newton-Raphson iterations.
            flipped (bool): whether flipped triangles exist at current displacement.
        """
        continue_on_flip = kwargs.get('continue_on_flip', False)
        v0 = self.vertices(gear=gear[0])
        v1 = self.vertices(gear=gear[-1])
        dxy = v1 - v0
        shape_matrices = self.stiffness_shape_matrices(gear=gear[0], cache=inner_cache)
        material_table = self._material_table.id_table
        flipped = False
        multiplier = self.stiffness_multiplier
        indices = []
        F0 = []
        F1 = []
        for mid, vals in shape_matrices.items():
            mat = material_table[mid]
            indx, Ms = vals
            uv = dxy[self.triangles[indx]].reshape(-1, 6)
            K, P, flp = mat.element_stiffness_matrices_from_shape_matrices(Ms, uv=uv, **kwargs)
            if flp:
                flipped = True
                if not continue_on_flip:
                    return None, None, None, flipped
            if K is None:
                continue
            mm = multiplier[indx].reshape(-1,1,1)
            indices.append(indx)
            F0.append(P * mm)
            F1.append(K * mm)
        return F1, indices, F0, flipped


    @config_cache('TBD')
    def stiffness_matrix(self, gear=(MESH_GEAR_FIXED, MESH_GEAR_MOVING), inner_cache=None, **kwargs):
        """
        compute the stiffness matrix and the current stress.
        Kwargs:
            gear(tuple): first item used for shape matrices, second gear for stress
                computation (and stiffness if nonlinear).
            inner_cache: the cache to store intermediate attributes like shape
                matrices. Use default if set to None.
            check_flip(bool): check if any triangles are flipped.
            continue_on_flip(bool): whether to continue with flipped triangles
                detected.
        """
        continue_on_flip = kwargs.get('continue_on_flip', False)
        F1, indices, F0, flipped = self.element_stiffness_matrices(gear=gear, inner_cache=inner_cache, cache=inner_cache, **kwargs)
        if flipped and (not continue_on_flip):
            return None, None
        stf_sz = 2 * self.num_vertices
        T = np.repeat(self.triangles * 2, 2, axis=-1)
        T[:,1::2] += 1
        STIFF_M = sparse.csr_matrix((stf_sz, stf_sz), dtype=np.float32)
        STRESS_v = np.zeros(stf_sz, dtype=np.float32)
        for f1, tidx, f0 in zip(F1, indices, F0):
            idx_1d = T[tidx]
            idx_1 = np.tile(idx_1d.reshape(-1,1,6), (1,6,1))
            idx_2 = np.swapaxes(idx_1, 1, 2)
            idx_1 = idx_1.ravel()
            idx_2 = idx_2.ravel()
            V = f1.ravel()
            M = sparse.csr_matrix((V, (idx_1, idx_2)),shape=(stf_sz, stf_sz))
            STIFF_M = STIFF_M + M
            np.add.at(STRESS_v, idx_1d.ravel(), f0.ravel())
        return STIFF_M, STRESS_v


  ## ------------------------- utility functions --------------------------- ##
    def nearest_vertices(self, xy, gear=MESH_GEAR_MOVING, offset=True):
        """find nearest vertices for debug."""
        if offset:
            v = self.vertices_w_offset(gear=gear)
        else:
            v = self.vertices(gear=gear)
        T = KDTree(v)
        dd, ii = T.query(np.reshape(xy, (-1,2)))
        return ii, dd


    def _num_masked_tri(self, tri_mask):
        if tri_mask is None:
            return self.num_triangles
        elif isinstance(tri_mask, np.ndarray) and (tri_mask.dtype == bool):
            return np.sum(tri_mask)
        else:
            indx = np.unique(tri_mask)
            return indx.size


    @staticmethod
    def _masked_all(mask):
        if (mask is None) or (isinstance(mask, np.ndarray) and (mask.dtype == bool) and (np.all(mask))):
            return True
        else:
            return False


    @staticmethod
    def triangle2edge(triangles, directional=False):
        """Convert triangle indices to edge indices."""
        edges = np.concatenate((triangles[:,[0,1]], triangles[:,[1,2]], triangles[:,[2,0]]), axis=0)
        if not directional:
            edges = np.unique(np.sort(edges, axis=-1), axis=0)
        return edges


    @staticmethod
    def masked_index_to_global_index(mask_sel, local_indx):
        """
        convert local index of a subarray to the global index.
        Args:
        mask_sel (np.ndarray[bool]): mask to select subarray from a larger array
            (True to select). If None, the subarray is the global array.
        local_indx: the local index of the subarray
        Return:
        global_indx(np.ndarray[int]): the index in the parent array.
        """
        if mask_sel is None:
            return local_indx
        if mask_sel.dtype == bool:
            sel_indx = np.nonzero(mask_sel)[0]
        else:
            sel_indx = np.unique(mask_sel)
        return sel_indx[local_indx]


    @staticmethod
    def global_index_to_masked_index(mask_sel, global_indx):
        """
        convert global index to the local index a subarray.
        Args:
        mask_sel (np.ndarray[bool]): mask to select subarray from a larger array
            (True to select). If None, the subarray is the global array.
        global_indx: the global index
        Return:
        local_indx(np.ndarray[int]): the index in the subarray. If an element is
            not masked, return -1
        """
        if mask_sel is None:
            return global_indx
        g_indx = np.full_like(global_indx, -1, shape=mask_sel.shape)
        if mask_sel.dtype == bool:
            g_indx[mask_sel] = np.arange(np.sum(mask_sel))
        else:
            mask_sel = np.unique(mask_sel, axis=None)
            g_indx[mask_sel] = np.arange(mask_sel.size)
        return g_indx[global_indx]


    @staticmethod
    def svds_to_deform(s):
        """
        given (N,2) singular values, return a (N,) deformation value that varies
        between 0~1 if not flipped, and 1~inf if flipped
        """
        d = np.piecewise(s, [s<1, s>=1], [lambda x: 1-x, lambda x: 1-1/x])
        return np.max(d, axis=-1)
