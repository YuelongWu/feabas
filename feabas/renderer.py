import matplotlib.tri
import numpy as np
import shapely
import shapely.geometry as shpgeo
from shapely.ops import unary_union

from feabas.constant import *


class MeshRenderer:
    """
    A class to apply transforms according to a Mesh and render images.
    """
    def __init__(self, interpolators, **kwargs):
        self._interpolators = interpolators
        n_region = len(self._interpolators)
        self._offset = np.array(kwargs.get('offset', np.zeros((1,2))), copy=False).reshape(1,2)
        self._region_tree = kwargs.get('region_tree', None)
        self._weight_params = kwargs.get('weight_params', MESH_TRIFINDER_WHATEVER)
        self.weight_generator = kwargs.get('weight_generator', [None]*n_region)
        self._collision_region = kwargs.get('collision_region', [None]*n_region)
        self._image_loader = kwargs.get('image_loader', None)


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
            else:
                hitidx = np.intersect1d(tidx, collision_tidx)
                if hitidx.size == 0:
                    weight_generator.append(None)
                    continue
                mpl_tri, _, _ = srcmesh.mpl_tri(gear=gear[0], tri_mask=hitidx)
                collision_region.append(srcmesh.shapely_regions(gear=gear[0], tri_mask=hitidx))
                if weight_params == MESH_TRIFINDER_LEAST_INNERMOST:
                    cx = mpl_tri.x
                    cy = mpl_tri.y
                    mpts = list(shpgeo.MultiPoint(np.stack((cx, cy), axis=-1)).geoms)
                    dis0 = region.boundary.distance(mpts)
                    inside = region.intersects(mpts)
                    dis0[inside] *= -1
                    weight_generator.append(matplotlib.tri.LinearTriInterpolator(mpl_tri, dis0))
                elif weight_params == MESH_TRIFINDER_LEAST_DEFORM:
                    deform = srcmesh.triangle_tform_deform(gear=gear[::-1], tri_mask=hitidx)
                    wt = np.exp(-2 * deform**2)
                    weight_generator.append((mpl_tri.get_trifinder(), wt))
                else:
                    raise ValueError
        if len(collision_region) > 0:
            collision_region = unary_union(collision_region)
        else:
            collision_region = None
        return cls(interpolators, offset=offset0, region_tree=region_tree, weight_params=weight_params,
            weight_generator=weight_generator, collision_region=collision_region)


    def link_image_loader(self, imgloader):
        self._image_loader = imgloader


    def remap_subregions(self, field, **kwargs):
        pass


    def region_finder_for_points(self, xy):
        """
        find the regions a collection of points belong to.
        Args:
            xy (N x 2 ndarray): the querry points
        Return:
            rid (N ndarray): the region ids
        """
        xy0 = np.array(xy, copy=False).reshape(-1,2) - self._offset
        if len(self._interpolators) == 1:
            return np.zeros(xy0.shape[0], dtype=np.int32)
        else:
            rids_out = np.full(xy0.shape[0], -1, dtype=np.int32)
        mpts = shpgeo.MultiPoint(xy0)
        pts_list = list(mpts.geoms)
        hits = self._region_tree.query(pts_list, predicate='intersects')
        if hits.size == 0:
            return rids_out
        uhits, uidx, cnts = np.unique(hits[0], return_index=True, return_counts=True)
        conflict = np.any(cnts > 1)
        if conflict:
            if self._weight_params in (MESH_TRIFINDER_LEAST_INNERMOST, MESH_TRIFINDER_LEAST_DEFORM):
                conflict_pts_indices = uhits[cnts > 1]
                for pt_idx in conflict_pts_indices:
                    pmask = hits[0] == pt_idx
                    r_indices = hits[1, pmask]
                    mndis = np.inf
                    r_sel = r_indices[0]
                    pxy = xy0[pt_idx].ravel()
                    for r_id in r_indices:
                        wg = self.weight_generator[r_id]
                        if wg is None:
                            continue
                        if self._weight_params == MESH_TRIFINDER_LEAST_INNERMOST:
                            dis_m = wg(pxy[0], pxy[1])
                            if dis_m.mask:
                                continue
                            dis = dis_m.data
                        else:
                            trfd, wt0 = wg
                            tid = trfd(pxy[0], pxy[1])
                            if tid < 0:
                                continue
                            dis = wt0[tid]
                        if dis < mndis:
                            r_sel = r_id
                            mndis = dis
                    hits[1, pmask] = r_sel
            hits = hits[:, uidx]
        rids_out[hits[0]] = hits[1]
        return rids_out


    def field_w_weight(self, bbox, region_id=None):
        bbox0 = np.array(bbox).reshape(4) - np.tile(self._offset.ravel(), 2)


    def crop(self, bbox, return_empty=False, **kwargs):
        image_loader = kwargs.get('image_loader', self._image_loader)
        if image_loader is None:
            raise RuntimeError('Image loader not defined.')


    @property
    def bounds(self):
        if not hasattr(self, '_bounds'):
            xmin, ymin, xmax, ymax = np.inf, np.inf, -np.inf, -np.inf
            for m in self._weight_params:
                xmin = min(xmin, np.min(m.x) + self._offset.ravel()[0])
                ymin = min(ymin, np.min(m.y) + self._offset.ravel()[1])
                xmax = max(xmax, np.max(m.x) + self._offset.ravel()[0])
                ymax = max(ymax, np.max(m.y) + self._offset.ravel()[1])
            self._bounds = (xmin, ymin, xmax, ymax)
        return self._bounds
