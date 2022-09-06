import json
import numpy as np
from scipy import sparse
import triangle

from feabas import miscs, spatial, material


class Mesh:
    """
    A class to represent a FEM Mesh
    Args:
        vertices (NVx2 ndarray): x-y cooridnates of the vertices.
        triangles (NT x 3 ndarray): each row is 3 vertex indices belong to a
            triangle.
    Kwargs:
        material_ids (NT ndarray of int8): material for each triangle. If None, 
            set to the id of the default material defined in feabas.material.
        material_table (feabas.material.MaterialTable): table of
            material properties.
        edges (Ne x 2 ndarray): each row is 2 vertex indices belong to an edge.
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
        material_ids = kwargs.get('material_ids', None)
        if material_ids is None:
            # use default model
            default_mat = material.MATERIAL_DEFAULT
            material_ids = np.full(tri_num, default_mat.uid, dtype=np.int8)
        indx = np.argsort(material_ids, axis=None)
        self.triangles = triangles[indx]
        self._material_ids = material_ids[indx]

        self._material_table = kwargs.get('material_table', material.MaterialTable())

        self._edges = kwargs.get('edges', None)
        self._resolution = kwargs.get('resolution', 4)
        self._name = kwargs.get('name', '')
        self._uid = kwargs.get('uid', None)


    @classmethod
    def from_PSLG(cls, vertices, segments, markers=None, **kwargs):
        """
        initialize from PSLG (feabas.spatial.Geometry.PSLG).
        Args:
            vertices (Nx2 np.ndarray): vertices of PSLG
            segments (Mx2 np.ndarray): list of endpoints' vertex id for each
                segment of PSLG
            markers (dict of lists): marker points for each region names.
        Kwargs:
            mesh_size: the maximum area allowed in the mesh
            min_mesh_angle: minimum angle allowed in mesh. May negatively affect
                meshing performance, default to 0.
        """
        material_table = kwargs.get('material_table', material.MaterialTable())
        resolution = kwargs.get('resolution', 4)
        mesh_size = kwargs.get('mesh_size', (400*4/resolution)**2)
        min_angle = kwargs.get('min_mesh_angle', 0)
        regions = []
        holes = []
        PSLG = {'vertices': vertices, 'segments': segments}
        tri_opt = 'pe'
        if markers is not None:
            for mname, pts in markers.items():
                if not bool(pts): # no points in this marker
                    continue
                mat = material_table[mname]
                if not mat.enable_mesh:
                    holes.extend(pts)
                else:
                    area_constraint = mesh_size * mat.area_constraint
                    region_id = mat.uid
                    for rx, ry in pts:
                        regions.append([rx, ry, region_id, area_constraint])
            if bool(holes):
                PSLG['holes'] = holes
            if bool(regions):
                PSLG['regions'] = regions
            tri_opt += 'A'
        else:
            if mesh_size > 0:
                num_decimal = max(0, 2-round(np.log10(mesh_size)))
                area_opt = ('{:.' + str(num_decimal) + 'f}').format(num_decimal)
                if '.' in area_opt:
                    area_opt = area_opt.rstrip('0')
                tri_opt += area_opt
        if min_angle > 0:
            try:
                T = triangle.triangulate(PSLG, opts=tri_opt+'q{}'.format(min_angle))
            except:
                T = triangle.triangulate(PSLG, opts=tri_opt)
        else:
            T = triangle.triangulate(PSLG, opts=tri_opt)
        vertices = T['vertices']
        triangles = T['triangles']
        if 'triangle_attributes' in T:
            material_ids = T['triangle_attributes'].squeeze().astype(np.int8)
        else:
            material_ids = None
        edges = T['edges']
        return cls(vertices, triangles, material_ids=material_ids, edges=edges,
            **kwargs)

