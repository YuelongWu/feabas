from collections import defaultdict
import copy
import gc
import h5py
import numpy as np
from scipy import sparse
import scipy.sparse.csgraph as csgraph
import triangle

from feabas import miscs, spatial, material

def dynamic_cache(gear):
    """
    The decorator that determines the caching behaviour of the Mesh properties.
    gear: used to generate caching key. Possible values include:
        'INITIAL': the property is only related to the initial vertice positions
            and their connection;
        'FIXED': the property is also related to the position of fixed vertice
            positions;
        'MOVING': the property is also related to the position of moving vertice
            positions;
        'TBD': the vertices on which the property is caculated is determined on
            the fly. If 'gear' is provided in the keyward argument, use that;
            otherwise, use self._current_gear.
    cache: If False, no caching;
        If True, save to self as an attribute (default);
        If type of miscs.Cache, save to the cache object with key;
        If type defaultdict,  save to cache object with key under dict[prop_name].
    assign_value: if kwargs assign_value is given, instead of computing the property,
        directly return that value and force cache it if required.
    """
    def dynamic_cache_wrap(func):
        prop_name0 = func.__name__
        def decorated(self, cache=None, force_update=False, **kwargs):
            if cache is None:
                # if cache not provided, use default cache in self.
                cache = self._default_cache
            if (kwargs.get('tri_mask', None) is not None) or \
               (kwargs.get('vtx_mask', None) is not None):
                # if contain masked triangles or vertices, don't cache
                cache = False
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
            if isinstance(cache, bool):
                if cgear == Mesh.INITIAL:
                    sgear = 'INITIAL'
                elif cgear == Mesh.FIXED:
                    sgear = 'FIXED'
                elif cgear == Mesh.MOVING:
                    sgear = 'MOVING'
                elif cgear == Mesh.STAGING:
                    sgear = 'STAGING'
                else:
                    sgear = cgear
                prop_name = '_cached_' + prop_name0 + '_' + sgear
                if cache:  # save to self as an attribute
                    if not force_update and hasattr(self, prop_name):
                        # if already cached, use that
                        prop = getattr(self, prop_name)
                        if prop is not None:
                            return prop
                    if assign_mode:
                        prop = prop0
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
                    else:
                        prop = func(self, **kwargs)
                    cache_obj.update_item(key, prop)
                    return prop
                else:
                    raise TypeError('Cache type not recognized')
        return decorated
    return dynamic_cache_wrap


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
    INITIAL = -1    # initial fixed vertices
    FIXED = 0       # fixed vertices
    MOVING = 1      # moving vertices
    STAGING = 2     # moving vertices before validity checking and committing
  ## ------------------------- initialization & IO ------------------------- ##
    def __init__(self, vertices, triangles, **kwargs):
        self._vertices = {self.INITIAL: vertices}
        self._vertices[self.FIXED] = kwargs.get('fixed_vertices', vertices)
        self._vertices[self.MOVING] = kwargs.get('moving_vertices', None)
        self._vertices[self.STAGING] = kwargs.get('staging_vertices', None)
        self._fixed_offset = kwargs.get('fixed_offset', np.zeros((1,2), dtype=type(vertices)))
        self._moving_offset = kwargs.get('moving_offset', np.zeros((1,2), dtype=type(vertices)))
        self._current_gear = self.FIXED
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
        self._name = kwargs.get('name', '')
        self.uid = kwargs.get('uid', None)
        self._internal_cache = kwargs.get('internal_cache', None)
        self._default_cache = kwargs.get('cache', True)
        self._caching_keys = defaultdict(lambda: None)
        self._caching_keys[self.INITIAL] = self.uid


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
        init_dict['vertices'] = self._vertices[self.INITIAL]
        if self._vertices[self.FIXED] is not self._vertices[self.INITIAL]:
            init_dict['fixed_vertices'] = self._vertices[self.FIXED]
        init_dict['triangles'] = self.triangles
        if (self._vertices[self.MOVING]) is not None:
            init_dict['moving_vertices'] = self._vertices[self.MOVING]
        if (self._vertices[self.STAGING]) is not None:
            init_dict['staging_vertices'] = self._vertices[self.STAGING]
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


    def save_to_h5(self, fname, vertex_flag=INITIAL, override_dict={}, **kwargs):
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
        self._current_gear = self.INITIAL

    def switch_to_fix(self):
        self._current_gear = self.FIXED

    def switch_to_mov(self):
        self._current_gear = self.MOVING

    def switch_to_stg(self):
        self._current_gear = self.STAGING

    def __getitem__(self, gear):
        if isinstance(gear, str):
            if gear.lower() in ('m', 'moving'):
                gear = self.MOVING
            elif gear.lower() in ('f', 'fixed'):
                gear = self.FIXED
            elif gear.lower() in ('i', 'initial'):
                gear = self.INITIAL
            elif gear.lower() in ('s', 'staging'):
                gear = self.STAGING
            else:
                raise KeyError
        self.switch_gear(gear)
        return self

    @property
    def vertices(self):
        gear = self._current_gear
        if gear == self.INITIAL:
            return self.initial_vertices
        elif gear == self.FIXED:
            return self.fixed_vertices
        elif gear == self.MOVING:
            return self.moving_vertices
        elif gear == self.STAGING:
            return self.staging_vertices
        else:
            raise ValueError

    @property
    def vertices_w_offset(self):
        if self._current_gear == self.INITIAL:
            return self.initial_vertices
        elif self._current_gear == self.FIXED:
            return self.fixed_vertices_w_offset
        elif self._current_gear == self.MOVING:
            return self.moving_vertices_w_offset
        elif self._current_gear == self.STAGING:
            return self.staging_vertices_w_offset
        else:
            raise ValueError

    @property
    def initial_vertices(self):
        return self._vertices[self.INITIAL]

    @property
    def fixed_vertices(self):
        return self._vertices[self.FIXED]

    @fixed_vertices.setter
    def fixed_vertices(self, v):
        self._vertices[self.FIXED] = v

    @property
    def fixed_vertices_w_offset(self):
        return self.fixed_vertices + self._fixed_offset

    @property
    def moving_vertices(self):
        if self._vertices[self.MOVING] is None:
            return self.initial_vertices
        else:
            return self._vertices[self.MOVING]

    @moving_vertices.setter
    def moving_vertices(self, v):
        self._vertices[self.MOVING] = v

    @property
    def moving_vertices_w_offset(self):
        return self.moving_vertices + self._moving_offset

    @property
    def staging_vertices(self):
        if self._vertices[self.STAGING] is None:
            return self.initial_vertices
        else:
            return self._vertices[self.STAGING]

    @staging_vertices.setter
    def staging_vertices(self, v):
        self._vertices[self.STAGING] = v

    @property
    def staging_vertices_w_offset(self):
        return self.staging_vertices + self._moving_offset


  ## -------------------------------- caching ------------------------------ ##
    def caching_keys(self, use_hash=False, force_update=False, gear=INITIAL):
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
        if force_update or self._caching_keys[self.INITIAL] is None:
            default_mat = self._material_table['default']
            if np.all(self._material_ids == default_mat.uid):
                mat_key = 0 # trivial material pattern
            else:
                mat_key = key_func(self._material_ids)
            self._caching_keys[self.INITIAL] = (key_func(self._vertices[self.INITIAL]),
                                                key_func(self.triangles), mat_key)
        if key_func == id:
            # very cheap anyway, can update every call
            force_update = True
        if isinstance(gear, str):
            if gear.upper() == 'INITIAL':
                gear = self.INITIAL
            elif gear.upper() == 'FIXED':
                gear = self.FIXED
            elif gear.upper() == 'MOVING':
                gear = self.MOVING
            elif gear.upper() == 'STAGING':
                gear = self.STAGING
            else:
                raise ValueError
        if gear == self.INITIAL:
            return (self._caching_keys[self.INITIAL], )
        else:
            if force_update or self._caching_keys[gear] is None:
                vertices = self._vertices[gear]
                if (vertices is None) or (vertices is self.initial_vertices) or \
                   np.all(vertices==self.initial_vertices):
                    self._caching_keys[gear] = 0
                else:
                    self._caching_keys[gear] = key_func(vertices)
            return (self._caching_keys[self.INITIAL], self._caching_keys[gear])


    def clear_cached_attr(self, gear=None, gc_now=False):
        prefix = '_cached_'
        if gear == self.INITIAL:
            suffix = '_INITIAL'
        elif gear == self.FIXED:
            suffix = '_FIXED'
        elif gear == self.MOVING:
            suffix = '_MOVING'
        elif gear == self.STAGING:
            suffix = '_STAGING'
        else:
            suffix = ''
        for attname in self.__dict__:
            if attname.startswith(prefix) and attname.endswith(suffix):
                setattr(self, attname, None)
        if gc_now:
            gc.collect()


  ## ------------------------------ properties ----------------------------- ##
    @property
    def num_vertices(self):
        return self.initial_vertices.shape[0]


    @property
    def num_triangles(self):
        return self.triangles.shape[0]


    @dynamic_cache('INITIAL')
    def edges(self, tri_mask=None):
        """edge indices of the triangulation mesh."""
        if tri_mask is None:
            T = self.triangles
        else:
            T = self.triangles[tri_mask]
        edges = Mesh.triangle2edge(T, directional=False)
        return edges


    @dynamic_cache('INITIAL')
    def segments(self, tri_mask=None):
        """edge indices for edges on the borders."""
        if tri_mask is None:
            T = self.triangles
        else:
            T = self.triangles[tri_mask]
        edges = Mesh.triangle2edge(T, directional=True)
        _, indx, cnt = np.unique(np.sort(edges, axis=-1), axis=0, return_index=True, return_counts=True)
        indx = indx[cnt == 1]
        return edges[indx]


    @dynamic_cache('INITIAL')
    def vertex_adjacencies(self, vtx_mask=None):
        """sparse adjacency matrix of vertices."""
        if vtx_mask is None:
            edges = self.edges()
            idx0 = edges[:,0]
            idx1 = edges[:,1]
            V = np.ones_like(idx0, dtype=bool)
            Npt = self.num_vertices
            A = sparse.csr_matrix((V, (idx0, idx1)), shape=(Npt, Npt))
            return A
        else:
            A = self.vertex_adjacencies(vtx_mask=None)
            return A[vtx_mask][:, vtx_mask]


    @dynamic_cache('TBD')
    def vertex_distances(self, gear=INITIAL, vtx_mask=None):
        """sparse matrix storing lengths of the edges."""
        if vtx_mask is None:
            gear0 = self._current_gear
            if gear0 != gear:
                self.switch_gear(gear)
                vertices = self.vertices
            self.switch_gear(gear0)
            A = self.vertex_adjacencies()
            idx0, idx1 = A.nonzero()
            edges_len = np.sum((vertices[idx0] - vertices[idx1])**2, axis=-1)**0.5
            Npt = self.num_vertices
            D = sparse.csr_matrix((edges_len, (idx0, idx1)), shape=(Npt, Npt))
            return D
        else:
            D = self.vertex_adjacencies(gear=gear, vtx_mask=None)
            return D[vtx_mask][:, vtx_mask]


    @dynamic_cache('INITIAL')
    def triangle_adjacencies(self, tri_mask=None):
        """ sparse adjacency matrix of triangles."""
        if tri_mask is None:
            edges = np.sort(Mesh.triangle2edge(self.triangles, directional=True), axis=-1)
            tids0 = np.arange(edges.shape[0]) % self.num_triangles
            edges_complex = edges[:,0] + edges[:,1] *1j
            idxt = np.argsort(edges_complex)
            tids = tids0[idxt]
            edges_complex = edges_complex[idxt]
            indx = np.nonzero(np.diff(edges_complex)==0)[0]
            idx0 = tids[indx]
            idx1 = tids[indx+1]
            Ntr = self.num_triangles
            V = np.ones_like(idx0, dtype=bool)
            A = sparse.csr_matrix((V, (idx0, idx1)), shape=(Ntr, Ntr))
            return A
        else:
            A = self.triangle_adjacencies(tri_mask=None)
            return A[tri_mask][:, tri_mask]


    @dynamic_cache('TBD')
    def triangle_centers(self, gear=INITIAL):
        """corodinates of the centers of the triangles (Ntri x 2)"""
        gear0 = self._current_gear
        if gear0 != gear:
            self.switch_gear(gear)
            vertices = self.vertices
        self.switch_gear(gear0)
        vtri = vertices[self.triangles]
        return vtri.mean(axis=1)


    @dynamic_cache('TBD')
    def triangle_distances(self, gear=INITIAL, vtx_mask=None):
        """sparse matrix storing distances of neighboring triangles."""
        if vtx_mask is None:
            tri_centers = self.triangle_centers(gear=gear)
            A = self.triangle_adjacencies()
            idx0, idx1 = A.nonzero()
            dis = np.sum((tri_centers[idx0] - tri_centers[idx1])**2, axis=-1)**0.5
            Ntri = self.num_triangles
            D = sparse.csr_matrix((dis, (idx0, idx1)), shape=(Ntri, Ntri))
            return D
        else:
            D = self.triangle_distances(gear=gear, vtx_mask=None)
            return D[vtx_mask][:, vtx_mask]


    @dynamic_cache('INITIAL')
    def connected_components(self):
        """
        connected components of the vertices & triangles.
        return as (number_of_components, vertex_labels, triangle_labels).
        """
        A = self.vertex_adjacencies()
        N_conn, V_conn = csgraph.connected_components(A, directed=False, return_labels=True)
        T_conn = V_conn[self.triangles[:,0]]
        return N_conn, V_conn, T_conn


  ## ------------------------- utility functions --------------------------- ##
    @staticmethod
    def triangle2edge(triangles, directional=False):
        """Convert triangle indices to edge indices."""
        edges = np.concatenate((triangles[:,[0,1]], triangles[:,[1,2]], triangles[:,[2,0]]), axis=0)
        if not directional:
            edges = np.unique(np.sort(edges, axis=-1), axis=0)
        return edges
