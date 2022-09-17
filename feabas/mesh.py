import copy
import h5py
import json
import numpy as np
from scipy import sparse
import triangle

from feabas import miscs, spatial, material


def dynamic_cache(func):
    """
    The decorator that determines the caching behaviour of the Mesh properties.
    cache: If None, save to self as an attribute;
           If type of miscs.Cache, save to the cache object;
           By default, set to CacheNull (No caching).
    """
    prop_name = '_' + func.__name__
    def decorated(self, cache=miscs.CacheNull(), force_update=False, **kwargs):
        if cache is None: # save to self as an attribute
            if not force_update and hasattr(self, prop_name):
                # if already cached, use that
                prop = getattr(self, prop_name)
                if prop is not None:
                    return prop
            prop = func(self, **kwargs)
            setattr(self, prop_name, prop)
            return prop
        elif type(cache) == miscs.CacheNull:
            # no cache to use by default
            if not force_update and hasattr(self, prop_name):
                # if already cached, use that
                prop = getattr(self, prop_name)
                if prop is not None:
                    return prop
            prop = func(self, **kwargs)
            return prop
        elif isinstance(cache, miscs.CacheNull):
            key = (self.uid, prop_name)
            if not force_update and (key in cache):
                prop = cache[key]
                if prop is not None:
                    return prop
            prop = func(self, **kwargs)
            cache.update_item(key, prop)
            return prop
        else:
            raise TypeError('Cache type not recognized')
    return decorated



class Mesh:
    """
    A class to represent a FEM Mesh
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
        uid(int): unique id number, used as the key for caching. If set to None,
            no caching will be performed.
    """
    INITIAL = -1    # initial fixed vertices
    FIXED = 0       # fixed vertices
    MOVING = 1      # moving vertices
    STAGING = 2     # moving vertices before validity checking and committing
  ## ------------------------- initialization & IO ------------------------- ##
    def __init__(self, vertices, triangles, **kwargs):
        self._vertices = {self.INITIAL: vertices, self.FIXED: vertices}
        self._vertices[self.MOVING] = kwargs.get('moving_vertices', None)
        self._vertices[self.STAGING] = None

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
        self.triangles = triangles[indx]
        self._material_ids = material_ids[indx]

        self._resolution = kwargs.get('resolution', 4)
        self._name = kwargs.get('name', '')
        self._uid = kwargs.get('uid', None)


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
                    area_constraint = mesh_area * mat.area_constraint
                    region_id = mat.uid
                    if area_constraint == 0:
                        regions_no_steiner.append(region_id)
                    for rx, ry in pts:
                        regions.append([rx, ry, region_id, area_constraint])
            if bool(holes):
                PSLG['holes'] = holes
            if bool(regions):
                PSLG['regions'] = regions
            tri_opt += 'A'
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


    def get_init_dict(self, save_material=True, vertex_flag=INITIAL, **kwargs):
        """
        dictionary that can be used for initialization of a duplicate.
        """
        init_dict = {}
        init_dict['vertices'] = self._vertices[vertex_flag]
        init_dict['triangles'] = self.triangles
        if (self._vertices[self.MOVING]) is not None and (vertex_flag != self.MOVING):
            init_dict['moving_vertices'] = self._vertices[self.MOVING]
        if save_material:
            init_dict['material_ids'] = self._material_ids
            init_dict['material_table'] = self.material_table.save_to_json()
        init_dict['resolution'] = self._resolution
        if bool(self._name):
            init_dict['name'] = self._name
        if self._uid is not None:
            init_dict['uid'] = self._uid
        init_dict.update(kwargs)
        return init_dict


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


    @staticmethod
    def triangle2edge(triangles, to_sort=True):
        """Convert triangle indices to edge indices."""
        edges = np.concatenate((triangles[:,[0,1]], triangles[:,[1,2]], triangles[:,[2,0]]), axis=0)
        if to_sort:
            edges = np.unique(np.sort(edges, axis=-1), axis=0)
        return edges
