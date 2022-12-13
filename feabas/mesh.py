from collections import defaultdict
import copy
import gc
import h5py
import numpy as np
from rtree import index
from scipy import sparse
import scipy.sparse.csgraph as csgraph
import shapely
import shapely.geometry as shpgeo
from shapely.ops import polygonize, unary_union
import triangle

from feabas import miscs, material
from feabas.constant import *


def gear_constant_to_str(gear_const):
    if isinstance(gear_const, (tuple, list)):
        gearstr = '_'.join([gear_constant_to_str(s) for s in gear_const])
    elif gear_const == MESH_GEAR_INITIAL:
        gearstr = 'INITIAL'
    elif gear_const == MESH_GEAR_FIXED:
        gearstr = 'FIXED'
    elif gear_const == MESH_GEAR_MOVING:
        gearstr = 'MOVING'
    elif gear_const == MESH_GEAR_STAGING:
        gearstr = 'STAGING'
    else:
        raise ValueError
    return gearstr


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
                    cgear = self._current_gear
            else:
                cgear = gear
            if cache is None:
                # if cache not provided, use default cache in self.
                cache = self._default_cache[cgear]
            masked_operation = False
            if ('tri_mask' in kwargs) and (kwargs['tri_mask'] is not None):
                tri_mask = np.array(tri_mask, copy=False)
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
                prop_name = '_cached_' + prop_name0 + '_' + sgear
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
        vertices (NVx2 ndarray): x-y cooridnates of the vertices.
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
        uid(int): unique id number, used as the key for caching.
    """
  ## ------------------------- initialization & IO ------------------------- ##
    def __init__(self, vertices, triangles, **kwargs):
        self._vertices = {MESH_GEAR_INITIAL: vertices}
        self._vertices[MESH_GEAR_FIXED] = kwargs.get('fixed_vertices', vertices)
        self._vertices[MESH_GEAR_MOVING] = kwargs.get('moving_vertices', None)
        self._vertices[MESH_GEAR_STAGING] = kwargs.get('staging_vertices', None)
        self._fixed_offset = kwargs.get('fixed_offset', np.zeros((1,2), dtype=np.float64))
        self._moving_offset = kwargs.get('moving_offset', np.zeros((1,2), dtype=np.float64))
        self._current_gear = MESH_GEAR_FIXED
        tri_num = triangles.shape[0]
        mtb = kwargs.get('material_table', None)
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

        material_ids = kwargs.get('material_ids', None)
        if material_ids is None:
            # use default model
            default_mat = self._material_table['default']
            material_ids = np.full(tri_num, default_mat.uid, dtype=np.int8)
        indx = np.argsort(material_ids, axis=None)
        if np.any(indx!=np.arange(indx.size)):
            triangles = triangles[indx]
            material_ids = material_ids[indx]
        self.triangles = triangles
        self._material_ids = material_ids
        self._resolution = kwargs.get('resolution', 4)
        self._epsilon = kwargs.get('epsilon', EPSILON0)
        self._name = kwargs.get('name', '')
        self.uid = kwargs.get('uid', None)
        self._default_cache = kwargs.get('cache', defaultdict(lambda: True))
        self._caching_keys = defaultdict(lambda: None)
        self._caching_keys[MESH_GEAR_INITIAL] = self.uid # if set to None, intialize later


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
        return cls(vertices, triangles, material_ids=material_ids, **kwargs)


    @classmethod
    def from_bbox(cls, bbox, **kwargs):
        # generate mesh with rectangular boundary defined by bbox
        # [xmin, ymin, xmax, ymax]
        return cls.from_boarder_bbox(bbox, bd_width=np.inf, roundup_bbox=False, mesh_growth=1.0, **kwargs)


    @classmethod
    def from_boarder_bbox(cls, bbox, bd_width=np.inf, roundup_bbox=True, mesh_growth=3.0, **kwargs):
        # rectangular ROI with different boader mesh settings
        # (smaller edgesz + regular at boarders)
        # mostly for stitching application.
        # bbox: [xmin, ymin, xmax, ymax]
        # bd_width: border width. in pixel or ratio to the size of the bbox
        # roundup_bbox: if extend the bounding box size to make it multiples of
        #   the mesh size. Otherwise, adjust the mesh size
        # mesh_growth: increase of the mesh size in the interior region.
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


    def get_init_dict(self, save_material=True,  **kwargs):
        """
        dictionary that can be used for initialization of a duplicate.
        """
        init_dict = {}
        init_dict['vertices'] = self._vertices[MESH_GEAR_INITIAL]
        if self._vertices[MESH_GEAR_FIXED] is not self._vertices[MESH_GEAR_INITIAL]:
            init_dict['fixed_vertices'] = self._vertices[MESH_GEAR_FIXED]
        init_dict['triangles'] = self.triangles
        if (self._vertices[MESH_GEAR_MOVING]) is not None:
            init_dict['moving_vertices'] = self._vertices[MESH_GEAR_MOVING]
        if (self._vertices[MESH_GEAR_STAGING]) is not None:
            init_dict['staging_vertices'] = self._vertices[MESH_GEAR_STAGING]
        if np.any(self._fixed_offset):
            init_dict['fixed_offset'] = self._fixed_offset
        if np.any(self._moving_offset):
            init_dict['moving_offset'] = self._moving_offset
        if save_material:
            init_dict['material_ids'] = self._material_ids
            init_dict['material_table'] = self._material_table.save_to_json()
        init_dict['resolution'] = self._resolution
        if bool(self._name):
            init_dict['name'] = self._name
        if self.uid is not None:
            init_dict['uid'] = self.uid
        init_dict.update(kwargs)
        return init_dict


    def submesh(self, tri_mask, **kwargs):
        """
        return a subset of the mesh with tri_mask as the triangle mask.
        """
        if np.all(tri_mask):
            return self
        masked_tri = self.triangles[tri_mask]
        vindx = np.unique(masked_tri, axis=None)
        rindx = np.full_like(masked_tri, -1, shape=np.max(vindx)+1)
        rindx[vindx] = np.arange(vindx.size)
        init_dict = self.get_init_dict(**kwargs)
        init_dict['triangles'] = rindx[masked_tri]
        vtx_keys = ['vertices','fixed_vertices','moving_vertices','staging_vertices']
        for vkey in vtx_keys:
            if init_dict.get(vkey, None) is not None:
                init_dict[vkey] = init_dict[vkey][vindx]
        tri_keys = ['material_ids']
        for tkey in tri_keys:
            if init_dict.get(tkey, None) is not None:
                init_dict[tkey] = init_dict[tkey][tri_mask]
        if ('uid' not in kwargs) and ('uid' in init_dict):
            # do not copy the uid from the parent
            init_dict.pop('uid')
        return self.__class__(**init_dict)


    @classmethod
    def from_h5(cls, fname, prefix=''):
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
        return cls(**init_dict)


    def save_to_h5(self, fname, vertex_flag=MESH_GEAR_INITIAL, override_dict={}, **kwargs):
        prefix = kwargs.get('prefix', '')
        save_material = kwargs.get('save_material', True)
        compression = kwargs.get('compression', True)
        out = self.get_init_dict(save_material=save_material, vertex_flag=vertex_flag, **override_dict)
        if (len(prefix) > 0) and prefix[-1] != '/':
            prefix = prefix + '/'
        if isinstance(fname, h5py.File):
            for key, val in out.items():
                if val is None:
                    continue
                if isinstance(val, str):
                    val = miscs.str_to_numpy_ascii()
                if np.isscalar(val) or not compression:
                    _ = fname.create_dataset(prefix+key, data=val)
                else:
                    _ = fname.create_dataset(prefix+key, data=val, compression="gzip")
        else:
            with h5py.File(fname, 'w') as f:
                for key, val in out.items():
                    if val is None:
                        continue
                    if isinstance(val, str):
                        val = miscs.str_to_numpy_ascii()
                    if np.isscalar(val) or not compression:
                        _ = f.create_dataset(prefix+key, data=val)
                    else:
                        _ = f.create_dataset(prefix+key, data=val, compression="gzip")


    def copy(self, deep=True, save_material=True, override_dict={}):
        init_dict = self.get_init_dict(save_material=save_material, **override_dict)
        if deep:
            init_dict = copy.deepcopy(init_dict)
        return self.__class__(**init_dict)


  ## --------------------------- manipulate meshe -------------------------- ##
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
            indx = np.cumsum(to_keep) - 1
            self.triangles = indx[self.triangles]
            self.clear_cached_attr()


    def delete_orphaned_vertices(self):
        """remove vertices not included in any triangles"""
        connected = np.isin(np.arange(self.num_vertices), self.triangles)
        self.delete_vertices(~connected)


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

    @property
    def vertices(self):
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


    def offset(self, gear=None):
        if gear is None:
            gear = self._current_gear
        if gear == MESH_GEAR_INITIAL:
            return np.zeros((1,2), dtype=np.float64)
        elif gear == MESH_GEAR_FIXED:
            return self._fixed_offset
        elif gear == MESH_GEAR_MOVING:
            return self._moving_offset
        elif gear == MESH_GEAR_STAGING:
            return self._moving_offset
        else:
            raise ValueError


    @property
    def vertices_w_offset(self):
        if self._current_gear == MESH_GEAR_INITIAL:
            return self.initial_vertices
        elif self._current_gear == MESH_GEAR_FIXED:
            return self.fixed_vertices_w_offset
        elif self._current_gear == MESH_GEAR_MOVING:
            return self.moving_vertices_w_offset
        elif self._current_gear == MESH_GEAR_STAGING:
            return self.staging_vertices_w_offset
        else:
            raise ValueError

    @property
    def initial_vertices(self):
        return self._vertices[MESH_GEAR_INITIAL]

    @property
    def fixed_vertices(self):
        return self._vertices[MESH_GEAR_FIXED]

    @fixed_vertices.setter
    def fixed_vertices(self, v):
        self._vertices[MESH_GEAR_FIXED] = v

    @property
    def fixed_vertices_w_offset(self):
        return self.fixed_vertices + self.offset(gear=MESH_GEAR_FIXED)

    @property
    def moving_vertices(self):
        if self._vertices[MESH_GEAR_MOVING] is None:
            return self.initial_vertices
        else:
            return self._vertices[MESH_GEAR_MOVING]

    @moving_vertices.setter
    def moving_vertices(self, v):
        self._vertices[MESH_GEAR_MOVING] = v

    @property
    def moving_vertices_w_offset(self):
        return self.moving_vertices + self.offset(gear=MESH_GEAR_MOVING)

    @property
    def staging_vertices(self):
        if self._vertices[MESH_GEAR_STAGING] is None:
            return self.initial_vertices
        else:
            return self._vertices[MESH_GEAR_STAGING]

    @staging_vertices.setter
    def staging_vertices(self, v):
        self._vertices[MESH_GEAR_STAGING] = v

    @property
    def staging_vertices_w_offset(self):
        return self.staging_vertices + self.offset(gear=MESH_GEAR_STAGING)


  ## -------------------------------- caching ------------------------------ ##
    def caching_keys(self, use_hash=False, force_update=False, gear=MESH_GEAR_INITIAL):
        """
        hashing of the Mesh object served as the keys for caching.
            key0: Meshes with same initial vertices & triangles shares key0. If
                uid is not None, will use uid in the place of key0
            key1: Meshes with same fixed vertices (except for a global offset)
                shares key1.
            key2: Meshes with same moving vertices (except for a global offset)
                shares key2.
        If use_hash is True, hash the vertices to generate key (slow), otherwise
        use their id.
        If force_update is True, will re-caculate all keys.
        gear: specify which vertices the key is associated to (fixed, moving...)
        !!! Note that offsets are not considered here because elastic energies
            are not related to translation. Special care needs to be taken if
            the absolute position of the Mesh is relevant.
        """
        if use_hash:
            key_func = lambda s: hash(tuple(np.array(s, copy=False).ravel()))
        else:
            key_func = id
        if force_update or self._caching_keys[MESH_GEAR_INITIAL] is None:
            default_mat = self._material_table['default']
            if np.all(self._material_ids == default_mat.uid):
                mat_key = 0 # trivial material pattern
            else:
                mat_key = key_func(self._material_ids)
            self._caching_keys[MESH_GEAR_INITIAL] = (key_func(self._vertices[MESH_GEAR_INITIAL]),
                                                key_func(self.triangles), mat_key)
        if key_func == id:
            # very cheap anyway, can update every call
            force_update = True
        if isinstance(gear, str):
            if gear.upper() == 'INITIAL':
                gear = MESH_GEAR_INITIAL
            elif gear.upper() == 'FIXED':
                gear = MESH_GEAR_FIXED
            elif gear.upper() == 'MOVING':
                gear = MESH_GEAR_MOVING
            elif gear.upper() == 'STAGING':
                gear = MESH_GEAR_STAGING
            else:
                raise ValueError
        if gear == MESH_GEAR_INITIAL:
            return (self._caching_keys[MESH_GEAR_INITIAL], )
        else:
            if force_update or self._caching_keys[gear] is None:
                vertices = self._vertices[gear]
                if (vertices is None) or (vertices is self.initial_vertices) or \
                   np.all(vertices==self.initial_vertices):
                    self._caching_keys[gear] = 0
                else:
                    self._caching_keys[gear] = key_func(vertices)
            return (self._caching_keys[MESH_GEAR_INITIAL], self._caching_keys[gear])


    def clear_cached_attr(self, gear=None, gc_now=False):
        prefix = '_cached_'
        if gear == MESH_GEAR_INITIAL:
            suffix = '_INITIAL'
        elif gear == MESH_GEAR_FIXED:
            suffix = '_FIXED'
        elif gear == MESH_GEAR_MOVING:
            suffix = '_MOVING'
        elif gear == MESH_GEAR_STAGING:
            suffix = '_STAGING'
        else:
            suffix = ''
        for attname in self.__dict__:
            if attname.startswith(prefix) and attname.endswith(suffix):
                setattr(self, attname, None)
        if gc_now:
            gc.collect()


    def set_default_cache(self, gear=None, cache=True):
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
        if tri_mask is None:
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
        swid = self.segments_w_triangle_ids(tri_mask=tri_mask, **kwargs)
        if swid is not None:
            return swid[0]
        else:
            return None


    @config_cache(MESH_GEAR_INITIAL)
    def segments_w_triangle_ids(self, tri_mask=None):
        """edge indices for edges on the borders, also return the triangle ids"""
        if tri_mask is None:
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
        if vtx_mask is None:
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
        if vtx_mask is None:
            gear0 = self._current_gear
            if gear0 != gear:
                self.switch_gear(gear)
                vertices = self.vertices
            self.switch_gear(gear0)
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
        if tri_mask is None:
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
        Ntr = T.shape[0]
        V = np.ones_like(idx0, dtype=bool)
        A = sparse.csr_matrix((V, (idx0, idx1)), shape=(Ntr, Ntr))
        return A


    @config_cache('TBD')
    def triangle_centers(self, gear=MESH_GEAR_INITIAL, tri_mask=None):
        """corodinates of the centers of the triangles (Ntri x 2)"""
        if tri_mask is None:
            T = self.triangles
        else:
            m0 = self.triangle_centers(gear=gear, tri_mask=None, no_compute=True)
            if m0 is not None:
                return m0[tri_mask]
            T = self.triangles[tri_mask]
        gear0 = self._current_gear
        if gear0 != gear:
            self.switch_gear(gear)
            vertices = self.vertices
        self.switch_gear(gear0)
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
        gear0 = self._current_gear
        self.switch_gear(gear=gear)
        vertices = self.vertices
        V = vertices[T]
        self.switch_gear(gear=gear0)
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
        if tri_mask is None:
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


    @config_cache('TBD')
    def triangle_tform_svd(self, gear_fix=MESH_GEAR_INITIAL, gear=MESH_GEAR_MOVING, tri_mask=None):
        """
        singular values of the affine transforms for each triangle.
        """
        if tri_mask is not None:
            s0 = self.triangle_tform_svd(gear=gear, tri_mask=None, no_compute=True)
            if s0 is not None:
                return s0[tri_mask]
        gear0 = self._current_gear
        self.switch_gear(gear=gear_fix)
        v0 = self.vertices
        self.switch_gear(gear=gear)
        v1 = self.vertices
        self.switch_gear(gear=gear0)
        if tri_mask is None:
            T = self.triangles
        else:
            T = self.triangles[tri_mask]
        T0 = v0[T]
        T1 = v1[T]
        m0 = T0.mean(axis=-2, keepdims=True)
        m1 = T1.mean(axis=-2, keepdims=True)
        T0 = T0 - m0
        T1 = T1 - m1
        T0_pad = np.insert(T0, 2, 1, axis=-1)
        T1_pad = np.insert(T1, 2, 1, axis=-1)
        A = np.linalg.solve(T0_pad, T1_pad)
        s = np.linalg.svd(A[:,:2,:2],compute_uv=False)
        return s


  ## ------------------------ collision management ------------------------- ##
    def _triangles_rtree_generator(self, gear=MESH_GEAR_MOVING, tri_mask=None):
        if tri_mask is None:
            tids = np.arange(self.num_triangles)
        elif tri_mask.dtype == bool:
            tids = np.nonzero(tri_mask)[0]
        else:
            tids = np.sort(tri_mask)
        for k, bbox in enumerate(self.triangle_bboxes(gear=gear, tri_mask=tri_mask)):
            yield (k, bbox, tids[k])


    def triangles_rtree(self, gear=MESH_GEAR_MOVING, tri_mask=None):
        return index.Index(self._triangles_rtree_generator(gear=gear, tri_mask=tri_mask))


    def check_segment_collision(self, gear=MESH_GEAR_MOVING, tri_mask=None):
        """check if segments have collisions among themselves."""
        gear0 = self._current_gear
        self.switch_gear(gear=gear)
        vertices = self.vertices
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
        self.switch_gear(gear=gear0)
        return valid


    def locate_segment_collision(self, gear=MESH_GEAR_MOVING, tri_mask=None, check_flipped=True):
        """
        find the segments that collide. Return a list of collided segments and
        their (local) triangle ids.
        """
        gear0 = self._current_gear
        self.switch_gear(gear=gear)
        SRs = self.grouped_segment_chains(tri_mask=tri_mask)
        vertices = self.vertices
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
            if tri_mask is None:
                _, T = self.locate_flipped_triangles(gear=gear, tri_mask=Tseg_flag, return_triangles=True)
            else:
                _, T = self.locate_flipped_triangles(gear=gear, tri_mask=(Tseg_flag & tri_mask), return_triangles=True)
            if T.size > 0:
                S_flp = Mesh.triangle2edge(T, directional=True)
                for segwid in rest_of_segments:
                    if np.any(np.all(S_flp==segwid[0], axis=-1)):
                        collided_segments.append(segwid)
        self.switch_gear(gear=gear0)
        segs = np.array([s[0] for s in collided_segments])
        tids = np.array([s[1] for s in collided_segments])
        return segs, tids


    def locate_flipped_triangles(self, gear=MESH_GEAR_MOVING, tri_mask=None, return_triangles=False):
        gear0 = self._current_gear
        vertices0 = self.initial_vertices
        self.switch_gear(gear=gear)
        if tri_mask is None:
            T = self.triangles
        else:
            T = self.triangles[tri_mask]
        vertices = self.vertices
        A0 = miscs.signed_area(vertices0, T)
        A1 = miscs.signed_area(vertices, T)
        flipped_sel = (A0 * A1) <= 0
        self.switch_gear(gear=gear0)
        if return_triangles:
            return np.nonzero(flipped_sel)[0], T[flipped_sel] 
        else:
            return np.nonzero(flipped_sel)[0]


    def find_triangle_overlaps(self, gear=MESH_GEAR_MOVING, tri_mask=None):
        _, seg_tids = self.locate_segment_collision(gear=gear, tri_mask=tri_mask, check_flipped=False)
        flip_tids = self.locate_flipped_triangles(gear=gear, tri_mask=tri_mask)
        rtree0 = self.triangles_rtree(gear=gear, tri_mask=tri_mask)
        init_tids = np.concatenate((seg_tids, flip_tids), axis=0)
        init_tids = Mesh.masked_index_to_global_index(tri_mask, init_tids)
        init_bboxes = self.triangle_bboxes(gear=gear, tri_mask=init_tids)
        candidate_tids = []
        for bbox in init_bboxes:
            candidate_tids.extend(list(rtree0.intersection(bbox, objects=False)))
        candidate_tids = np.sort(np.unique(candidate_tids))
        tri_mask_c = Mesh.masked_index_to_global_index(tri_mask, candidate_tids)
        rtree_c = self.triangles_rtree(gear=gear, tri_mask=tri_mask_c)
        gear0 = self._current_gear
        self.switch_gear(gear=gear)
        vertices = self.vertices[self.triangles[tri_mask_c]]
        self.switch_gear(gear=gear0)
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


    def group_overlapped_triangles(self, collisions=None, gear=MESH_GEAR_MOVING, tri_mask=None):
        if collisions is None:
            collisions = self.find_triangle_overlaps(gear=gear, tri_mask=tri_mask)
        collisions_g = Mesh.masked_index_to_global_index(tri_mask, np.unique(collisions, axis=None))



    def fix_segment_collision(self):
        pass


  ## ------------------------- utility functions --------------------------- ##
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
            sel_indx = np.sort(sel_indx)
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
            g_indx[np.sort(mask_sel,axis=None)] = np.arange(mask_sel.size)
        return g_indx[global_indx]
