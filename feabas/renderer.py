import matplotlib.tri
import numpy as np

from feabas.constant import *


class MeshRenderer:
    """
    A class to apply transforms according to a Mesh and render images.
    """
    def __init__(self, mattri_list, dxy_list, **kwargs):
        pass


    @classmethod
    def from_mesh(cls, srcmesh, gear=(MESH_GEAR_MOVING, MESH_GEAR_INITIAL), **kwargs):
        include_flipped = kwargs.get('include_flipped', False)
        weight_params = kwargs.get('weight_params', MESH_TRIFINDER_LEAST_INNERMOST)
        render_mask = srcmesh.triangle_mask_for_render()
        tri_info = srcmesh.tri_info(gear=gear[0], tri_mask=render_mask, include_flipped=include_flipped)
        offset0 = srcmesh.offset(gear=gear[0])
        offset1 = srcmesh.offset(gear=gear[-1])
        region_tree = tri_info['region_tree']
        mattri_list = tri_info['matplotlib_tri']
        tidx_list = tri_info['triangle_index']
        vidx_list = tri_info['vertex_index']
        collisions = srcmesh.triangle_collisions(gear=gear[0], tri_mask=render_mask)
        if (collisions.size == 0) or (len(mattri_list) <= 1):
            weight_params = MESH_TRIFINDER_WHATEVER
        else:
            render_mask_indx = np.nonzero(render_mask)[0]
            collision_tidx = render_mask_indx[np.unique(collisions)]
        vertices_img = srcmesh.vertices(gear=gear[-1])
        interpolators = []
        for mattri, tidx, vidx in zip(mattri_list, tidx_list, vidx_list):
            v0 = vertices_img[vidx]
            xinterp = matplotlib.tri.LinearTriInterpolator(mattri, v0[:,0])
            yinterp = matplotlib.tri.LinearTriInterpolator(mattri, v0[:,1])
            interpolators.append((xinterp, yinterp))
