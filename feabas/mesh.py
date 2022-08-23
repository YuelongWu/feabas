import json
import numpy as np
from scipy import sparse
import triangle

from fem_aligner import miscs, spatial, material


class Mesh:
    """
    A class to represent a FEM Mesh
    Args:
        vertices (NVx2 ndarray): x-y cooridnates of the vertices.
        triangles (NT x 3 ndarray): each row is 3 vertex indices belong to a
            triangle.
        egments (Ns x 2 ndarray): each row is 2 vertex indices belong to an edge
            on the material boundaries.
        edges (Ne x 2 ndarray): each row is 2 vertex indices belong to an edge.
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
        material_ids = kwargs.get('material_ids', np.ones(tri_num, np.int8))
        indx = np.argsort(material_ids, axis=None)
        self.triangles = triangles[indx]
        self._material_ids = material_ids[indx]

        self._material_table = kwargs.get('material_table', material.MaterialTable())
        
        self._segments = kwargs.get('segments', None)
        self._edges = kwargs.get('edges', None)
        self._triangle_mask = kwargs.get('triangle_mask', np.zeros(tri_num, dtype=bool))
        self._resolution = kwargs.get('resolution', 4)
        self._name = kwargs.get('name', '')
        self._uid = kwargs.get('id', 0)


    @classmethod
    def from_PSLG(cls, vertices, segments, markers=None, **kwargs):
        """
        initialize from PSLG (fem_aligner.spatial.Geometry.PSLG).
        Args:
            vertices (Nx2 np.ndarray): vertices of PSLG
            segments (Mx2 np.ndarray): list of endpoints' vertex id for each
                segment of PSLG
            markers (dict of lists): marker points for each region names.
        Kwargs:
            material_table (fem_aligner.material.MaterialTable): table of
                material properties
            
        """
        material_table = kwargs.get('material_table', material.MaterialTable())
        angle_quality = kwargs.get('angle_quality', None)

        logger = kwargs.get('logger', None)
