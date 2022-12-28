import matplotlib.tri
import numpy as np
import shapely
import shapely.geometry as shpgeo

from feabas.constant import *


class MeshRenderer:
    """
    A class to apply transforms according to a Mesh and render images.
    """
    def __init__(self, interpolators, **kwargs):
        self._interpolators = interpolators
        n_region = len(self._interpolators)
        self._offset = kwargs.get('offset', np.zeros((1,2)))
        self._region_tree = kwargs.get('region_tree', None)
        self._weight_params = kwargs.get('weight_params', MESH_TRIFINDER_WHATEVER)
        self.weight_generator = kwargs.get('weight_generator', [None]*n_region)
        self._collision_region = kwargs.get('collision_region', [None]*n_region)


    @classmethod
    def from_mesh(cls, srcmesh, gear=(MESH_GEAR_MOVING, MESH_GEAR_INITIAL), **kwargs):
        include_flipped = kwargs.get('include_flipped', False)
        weight_params = kwargs.get('weight_params', MESH_TRIFINDER_LEAST_INNERMOST)
        render_mask = srcmesh.triangle_mask_for_render()
        tri_info = srcmesh.tri_info(gear=gear[0], tri_mask=render_mask, include_flipped=include_flipped)
        offset0 = srcmesh.offset(gear=gear[0])
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
        vertices_img = srcmesh.vertices_w_offset(gear=gear[-1])
        interpolators = []
        weight_generator = []
        collision_region = []
        for mattri, tidx, vidx, region in zip(mattri_list, tidx_list, vidx_list, region_tree.geometries):
            v0 = vertices_img[vidx]
            xinterp = matplotlib.tri.LinearTriInterpolator(mattri, v0[:,0])
            yinterp = matplotlib.tri.LinearTriInterpolator(mattri, v0[:,1])
            interpolators.append((xinterp, yinterp))
            if weight_params == MESH_TRIFINDER_WHATEVER:
                weight_generator.append(None)
                collision_region.append(None)
            else:
                hitidx = np.intersect1d(tidx, collision_tidx)
                if hitidx.size == 0:
                    weight_generator.append(None)
                    collision_region.append(None)
                    continue
                mpl_tri, _, _ = srcmesh.mpl_tri(gear=gear[0], tri_mask=hitidx)
                collision_region.append(srcmesh.shapely_regions(gear=gear[0], tri_mask=hitidx))
                if weight_params == MESH_TRIFINDER_LEAST_INNERMOST:
                    cx = mpl_tri.x
                    cy = mpl_tri.y
                    mpts = list(shpgeo.MultiPoint(np.stack((cx, cy), axis=-1)).geoms)
                    dis0 = region.boundary.distance(mpts)
                    inside = region.intersects(mpts)
                    dis0[~inside] *= -1
                    weight_generator.append(matplotlib.tri.LinearTriInterpolator(mpl_tri, dis0))
                elif weight_params == MESH_TRIFINDER_LEAST_DEFORM:
                    deform = srcmesh.triangle_tform_deform(gear=gear[::-1], tri_mask=hitidx)
                    wt = np.exp(-2 * deform**2)
                    weight_generator.append((mpl_tri.get_trifinder(), wt))
                else:
                    raise ValueError
        return cls(interpolators, offset=offset0, region_tree=region_tree, weight_params=weight_params,
            weight_generator=weight_generator, collision_region=collision_region)
