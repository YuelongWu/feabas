from collections import defaultdict
from functools import partial
import matplotlib.tri
import numpy as np
import json
from scipy.sparse import csgraph
import shapely
import shapely.geometry as shpgeo
from shapely.ops import unary_union
import tensorstore as ts
import time

from feabas.caching import CacheFIFO
from feabas.concurrent import submit_to_workers
import feabas.constant as const
from feabas.mesh import Mesh
from feabas import common, spatial, dal, logging, storage
from feabas.config import DEFAULT_RESOLUTION, SECTION_THICKNESS, montage_resolution, TS_TIMEOUT, CHECKPOINT_TIME_INTERVAL

H5File = storage.h5file_class()

class MeshRenderer:
    """
    A class to apply transforms according to a Mesh and render images.
    """
    def __init__(self, interpolators, **kwargs):
        self._interpolators = interpolators
        n_region = len(self._interpolators)
        self._offset = np.array(kwargs.get('offset', np.zeros((1,2))), copy=False).reshape(1,2)
        self._region_tree = kwargs.get('region_tree', None)
        self._weight_params = kwargs.get('weight_params', const.MESH_TRIFINDER_WHATEVER)
        self.weight_generator = kwargs.get('weight_generator', [None for _ in range(n_region)])
        self.weight_multiplier = kwargs.get('weight_multiplier', [None for _ in range(n_region)])
        self._collision_region = kwargs.get('collision_region', None)
        self._image_loader = kwargs.get('image_loader', None)
        self.resolution = kwargs.get('resolution', DEFAULT_RESOLUTION)
        self._default_fillval = kwargs.get('fillval', None)
        self._dtype = kwargs.get('dtype', None)
        self._geodesic_mask = kwargs.get('geodesic_mask', False)
        self._geodesic_info = kwargs.get('geodesic_info', None)


    @classmethod
    def from_mesh(cls, srcmesh, gear=(const.MESH_GEAR_MOVING, const.MESH_GEAR_INITIAL), **kwargs):
        include_flipped = kwargs.get('include_flipped', False)
        weight_params = kwargs.pop('weight_params', const.MESH_TRIFINDER_INNERMOST)
        local_cache = kwargs.get('cache', False)
        render_weight_threshold = kwargs.get('render_weight_threshold', 0)
        geodesic_mask = kwargs.get('geodesic_mask', False)
        msh_wt = srcmesh.weight_multiplier_for_render()
        if np.any(msh_wt != 1):
            weighted_material = True
        else:
            weighted_material = False
        render_mask = srcmesh.triangle_mask_for_render(render_weight_threshold=render_weight_threshold)
        if not np.any(render_mask):
            return None
        collisions = srcmesh.triangle_collisions(gear=gear[0], tri_mask=render_mask)
        if (collisions.size == 0):
            weight_params = const.MESH_TRIFINDER_WHATEVER
        else:
            render_mask_indx = np.nonzero(render_mask)[0]
            collision_tidx = render_mask_indx[np.unique(collisions)]
        if weight_params == const.MESH_TRIFINDER_INNERMOST:
            asymmetry = False
        else:
            asymmetry = True
        tri_info = srcmesh.tri_info(gear=gear[0], tri_mask=render_mask,
            include_flipped=include_flipped, cache=local_cache, asymmetry=asymmetry)
        offset0 = srcmesh.offset(gear=gear[0])
        region_tree = tri_info['region_tree']
        mattri_list = tri_info['matplotlib_tri']
        tidx_list = tri_info['triangle_index']
        vidx_list = tri_info['vertex_index']
        vertices_img = srcmesh.vertices_w_offset(gear=gear[-1])
        if geodesic_mask:
            geodesic_info = {}
            geodesic_info['vertex_adjacency'] = srcmesh.vertex_distances(gear=gear[0], tri_mask=render_mask, cache=False)
            geodesic_info['region_tri'] = mattri_list
            geodesic_info['region_vindx'] = vidx_list
            vtx0 = srcmesh.vertices(gear=gear[0])
            segs = srcmesh.segments(tri_mask=render_mask)
            geodesic_info['seg_line'] = unary_union([shpgeo.LineString(vtx0[s]) for s in segs])
        else:
            geodesic_info = None
        interpolators = []
        weight_generator = []
        collision_region = []
        weight_multiplier = []
        for mattri, tidx, vidx, region in zip(mattri_list, tidx_list, vidx_list, region_tree.geometries):
            v0 = vertices_img[vidx]
            xinterp = matplotlib.tri.LinearTriInterpolator(mattri, v0[:,0])
            yinterp = matplotlib.tri.LinearTriInterpolator(mattri, v0[:,1])
            interpolators.append((xinterp, yinterp))
            if weight_params == const.MESH_TRIFINDER_WHATEVER:
                weight_generator.append(None)
                weight_multiplier.append(None)
            else:
                hitidx = np.intersect1d(tidx, collision_tidx)
                if hitidx.size == 0:
                    weight_generator.append(None)
                    weight_multiplier.append(None)
                    continue
                mpl_tri, _, _ = srcmesh.mpl_tri(gear=gear[0], tri_mask=hitidx)
                collision_region.append(srcmesh.shapely_regions(gear=gear[0], tri_mask=hitidx))
                try:
                    mpl_tri.get_trifinder()
                except RuntimeError:
                    mpl_tri = mattri
                    hitidx = tidx
                if weight_params == const.MESH_TRIFINDER_INNERMOST:
                    cx = mpl_tri.x
                    cy = mpl_tri.y
                    mpts = list(shpgeo.MultiPoint(np.stack((cx, cy), axis=-1)).geoms)
                    dis0 = region.boundary.distance(mpts) + 1
                    inside = region.intersects(mpts)
                    dis0[~inside] *= -1
                    weight_generator.append(matplotlib.tri.LinearTriInterpolator(mpl_tri, dis0))
                elif weight_params == const.MESH_TRIFINDER_LEAST_DEFORM:
                    deform = srcmesh.triangle_tform_deform(gear=gear[::-1], tri_mask=hitidx)
                    wt = np.exp(-2 * deform**2)
                    weight_generator.append((mpl_tri.get_trifinder(), wt))
                else:
                    raise ValueError
                if not weighted_material:
                    weight_multiplier.append(None)
                else:
                    wt = msh_wt[hitidx]
                    weight_multiplier.append((mpl_tri.get_trifinder(), wt))
        if len(collision_region) > 0:
            collision_region = unary_union(collision_region)
        else:
            collision_region = None
        resolution = srcmesh.resolution
        return cls(interpolators, offset=offset0, region_tree=region_tree,
            weight_params=weight_params, weight_generator=weight_generator,
            weight_multiplier=weight_multiplier,
            collision_region=collision_region, resolution=resolution,
            geodesic_info=geodesic_info,
            **kwargs)


    def link_image_loader(self, imgloader):
        self._image_loader = imgloader


    def region_finder_for_points(self, xy, offsetting=True):
        """
        find the regions a collection of points belong to.
        Args:
            xy (N x 2 ndarray): the querry points
        Return:
            rid (N ndarray): the region ids
        """
        xy0 = np.array(xy, copy=False).reshape(-1,2)
        if offsetting:
            xy0 = xy0 - self._offset
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
            if self._weight_params in (const.MESH_TRIFINDER_INNERMOST, const.MESH_TRIFINDER_LEAST_DEFORM):
                conflict_pts_indices = uhits[cnts > 1]
                for pt_idx in conflict_pts_indices:
                    pmask = hits[0] == pt_idx
                    r_indices = hits[1, pmask]
                    mndis = -np.inf
                    r_sel = r_indices[0]
                    pxy = xy0[pt_idx].ravel()
                    for r_id in r_indices:
                        wg = self.weight_generator[r_id]
                        if wg is None:
                            continue
                        if self._weight_params == const.MESH_TRIFINDER_INNERMOST:
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
                        if dis > mndis:
                            r_sel = r_id
                            mndis = dis
                    hits[1, pmask] = r_sel
            hits = hits[:, uidx]
        rids_out[hits[0]] = hits[1]
        return rids_out


    def region_finder_for_bbox(self, bbox, offsetting=True):
        """
        find the regions that intersect a bounding box.
        Args:
            bbox [xmin, ymin, xmax, ymax]: the bonding box
        Return:
            rid (N ndarray): the region ids
        """
        bbox0 = np.array(bbox).reshape(4)
        if len(self._interpolators) == 1:
            hits = [0]
        else:
            if offsetting:
                bbox0 = bbox0 - np.tile(self._offset.ravel(), 2)
            rect = shpgeo.box(*bbox0)
            hits = self._region_tree.query(rect, predicate='intersects')
        return np.atleast_1d(hits)


    def bbox_hit_collision(self, bbox, offsetting=True):
        """
        test if a bounding box hit the collision regions of the mesh.
        """
        if (self._collision_region is None) or (hasattr(self._collision_region, 'area') and (self._collision_region.area == 0)):
            return False
        bbox0 = np.array(bbox).reshape(4)
        if offsetting:
            bbox0 = bbox0 - np.tile(self._offset.ravel(), 2)
        rect = shpgeo.box(*bbox0)
        return rect.intersects(self._collision_region)


    def field_w_weight(self, bbox, **kwargs):
        """
        compute the deformation field and their weight given a bounding box.
        Args:
            bbox: bounding box (xmin, ymin, xmax, ymax) in the output space,
                left/top included & right/bottom excluded
        Kwargs:
            region_id: which region to use when selecting the interpolator. If
                set to None, find the region that encloses the center of the
                bounding box
            compute_wt: whether to compute weight. If False, return the mask of
                the interpolated results as weight.
            out_resolution: output resolution. If set to None, assume the output
                resolution is the same as the intrinsic renderer resoluton.
        Return:
            x-field (ndarray): deformation field in x direction. None if
                bounding box not intersecting the interpolator.
            y-field (ndarray): deformation field in y direction. None if
                bounding box not intersecting the interpolator.
            weight (ndarray): the weight of the output field. None if bounding
                box not intersecting the interpolator.
        """
        region_id = kwargs.get('region_id', None)
        compute_wt =  kwargs.get('compute_wt', True)
        out_resolution = kwargs.get('out_resolution', None)
        offsetting = kwargs.get('offsetting', True)
        invalid_output = (None, None, None)
        bbox0 = np.array(bbox, copy=False).reshape(4)
        if offsetting:
            bbox0 = bbox0 - np.tile(self._offset.ravel(), 2)
        bcntr = ((bbox0[0] + bbox0[2] - 1)/2, (bbox0[1] + bbox0[3] - 1)/2)
        if region_id is None:
            region_id = self.region_finder_for_points(bcntr, offsetting=False).item()
        if region_id == -1:
            return invalid_output
        interpX, interpY = self._interpolators[region_id]
        xs = np.linspace(bbox0[0], bbox0[2], num=round(bbox0[2]-bbox0[0]), endpoint=False, dtype=float)
        ys = np.linspace(bbox0[1], bbox0[3], num=round(bbox0[3]-bbox0[1]), endpoint=False, dtype=float)
        if out_resolution is not None:
            scale = out_resolution / self.resolution
            xs = spatial.scale_coordinates(xs, scale)
            ys = spatial.scale_coordinates(ys, scale)
        xx, yy = np.meshgrid(xs, ys)
        map_x = interpX(xx, yy)
        map_y = interpY(xx, yy)
        mask = map_x.mask | map_y.mask
        if np.all(mask, axis=None):
            return invalid_output
        x_field = np.nan_to_num(map_x.data, copy=False)
        y_field = np.nan_to_num(map_y.data, copy=False)
        weight = 1 - mask.astype(np.float32)
        if self._geodesic_mask and (self._geodesic_info is not None):
            if self._geodesic_info['seg_line'].intersects(shpgeo.box(*bbox0)):
                if not hasattr(self, '_cached_geodesic_distance'):
                    self._cached_geodesic_distance = CacheFIFO(maxlen=5)
                if bcntr in self._cached_geodesic_distance:
                    dis_g0 = self._cached_geodesic_distance[bcntr]
                else:
                    hit_id = self.region_finder_for_points(bcntr, offsetting=False).item()
                    if hit_id == -1:
                        return invalid_output
                    hit_mtri = self._geodesic_info['region_tri'][hit_id]
                    t_finder = hit_mtri.get_trifinder()
                    hit_tidx = t_finder(bcntr[0], bcntr[1]).item()
                    if hit_tidx == -1:
                        return invalid_output
                    hit_vidx_loc = hit_mtri.triangles[hit_tidx]
                    hit_dis = ((hit_mtri.x[hit_vidx_loc] - bcntr[0]) ** 2 + (hit_mtri.y[hit_vidx_loc] - bcntr[1]) ** 2) ** 0.5
                    hit_vidx_glob = self._geodesic_info['region_vindx'][hit_id][hit_vidx_loc]
                    dis_t = csgraph.shortest_path(self._geodesic_info['vertex_adjacency'],
                                                  directed=False, return_predecessors=False,
                                                  unweighted=False, overwrite=False,
                                                  indices=hit_vidx_glob)
                    dis_t = dis_t + hit_dis.reshape(3, 1)
                    dis_g0 = np.min(dis_t, axis=0)
                    self._cached_geodesic_distance[bcntr] = dis_g0
                mtri = self._geodesic_info['region_tri'][region_id]
                vidx = self._geodesic_info['region_vindx'][region_id]
                dis_g = dis_g0[vidx]
                dis_e = ((mtri.x - bcntr[0])**2 + (mtri.y - bcntr[1])**2)**0.5
                with np.errstate(divide='ignore', invalid='ignore'):
                    dis_ratio = np.nan_to_num(dis_e/dis_g, nan=1)
                ginterp = matplotlib.tri.LinearTriInterpolator(mtri, dis_ratio)
                wt = np.nan_to_num(ginterp(xx, yy).data, nan=0)
                weight = weight * wt.clip(0,1)
        else:
            weight_generator = self.weight_generator[region_id]
            weight_multiplier = self.weight_multiplier[region_id]
            if compute_wt and (weight_generator is not None):
                if self._weight_params == const.MESH_TRIFINDER_INNERMOST:
                    wt = weight_generator(xx, yy)
                    if not np.all(wt.mask, axis=None):
                        wtmx = wt.max()
                        weight = weight * np.nan_to_num(wt.data, copy=False, nan=wtmx)
                elif self._weight_params == const.MESH_TRIFINDER_LEAST_DEFORM:
                    trfd, wt0 = weight_generator
                    tid = trfd(xx, yy)
                    omask = tid < 0
                    if not np.all(tid < 0, axis=None):
                        wt = wt0[tid]
                        wt[omask] = 1
                        weight = weight * wt
            if compute_wt and (weight_multiplier is not None):
                trfd, wt0 = weight_multiplier
                tid = trfd(xx, yy)
                omask = tid < 0
                if not np.all(tid < 0, axis=None):
                    wt = wt0[tid]
                    wt[omask] = 1
                    weight = weight * wt
        return x_field, y_field, weight


    def local_affine_tform(self, pt, offsetting=True, svd_clip=None):
        pt0 = np.array(pt, copy=False).ravel()
        if offsetting:
            pt0 = pt0 - self._offset.ravel()
        region_id = self.region_finder_for_points(pt0, offsetting=False).item()
        interpX, interpY = self._interpolators[region_id]
        axx, axy = interpX.gradient(pt0[0], pt0[1])
        if axx.mask:
            return None, None
        ayx, ayy = interpY.gradient(pt0[0], pt0[1])
        A = np.array([[axx.data, ayx.data], [axy.data, ayy.data]])
        if svd_clip is not None:
            u, s, vh = np.linalg.svd(A, compute_uv=True)
            if hasattr(svd_clip, '__len__'):
                s = s.clip(svd_clip[0], svd_clip[-1])
            else:
                s = s.clip(1/(1+svd_clip), 1+svd_clip)
            A = u @ np.diag(s) @ vh
        pt1_x = interpX(pt0[0], pt0[1])
        pt1_y = interpY(pt0[0], pt0[1])
        pt1 = np.array([pt1_x.data, pt1_y.data])
        t = pt1 - pt0 @ A
        return A, t


    def crop_field(self, bbox, **kwargs):
        """
        compute the deformation field within a bounding box.
        Args:
            bbox: bounding box (xmin, ymin, xmax, ymax) in the output space,
                left/top included & right/bottom excluded
        Kwargs:
            out_resolution: output resolution. If set to None, assume the output
                resolution is the same as the intrinsic renderer resoluton.
        Return:
            x-field (ndarray): deformation field in x direction. None if
                bounding box not intersecting the interpolator.
            y-field (ndarray): deformation field in y direction. None if
                bounding box not intersecting the interpolator.
            mask (ndarray): region of valid pixels.
        """
        mode = kwargs.get('mode', const.RENDER_FULL)
        offsetting = kwargs.get('offsetting', True)
        out_resolution = kwargs.get('out_resolution', None)
        bbox0 = np.array(bbox).reshape(4)
        outwd = round(bbox0[2]-bbox0[0])
        outht = round(bbox0[3]-bbox0[1])
        empty_output = (np.zeros((outht, outwd), dtype=np.float32),
                        np.zeros((outht, outwd), dtype=np.float32),
                        np.zeros((outht, outwd), dtype=bool))
        if offsetting:
            bbox0 = bbox0 - np.tile(self._offset.ravel(), 2)
        if (mode == const.RENDER_CONTIGEOUS) and (not self.bbox_hit_collision(bbox0, offsetting=False)):
            mode = const.RENDER_FULL
        if mode in (const.RENDER_LOCAL_RIGID, const.RENDER_LOCAL_AFFINE):
            bcntr = ((bbox0[0] + bbox0[2] - 1)/2, (bbox0[1] + bbox0[3] - 1)/2)
            if mode == const.RENDER_LOCAL_RIGID:
                svd_clip = 0
            else:
                svd_clip = kwargs.get('svd_clip', None)
            A, t = self.local_affine_tform(bcntr, offsetting=False, svd_clip=svd_clip)
            if A is None:
                return empty_output
            xs = np.linspace(bbox0[0], bbox0[2], num=outwd, endpoint=False, dtype=float)
            ys = np.linspace(bbox0[1], bbox0[3], num=outht, endpoint=False, dtype=float)
            if out_resolution is not None:
                scale = out_resolution / self.resolution
                xs = spatial.scale_coordinates(xs, scale)
                ys = spatial.scale_coordinates(ys, scale)
            xx, yy = np.meshgrid(xs, ys)
            x_field = xx * A[0,0] + yy * A[1,0] + t[0]
            y_field = xx * A[0,1] + yy * A[1,1] + t[1]
            mask = np.ones_like(x_field, dtype=bool)
        elif mode == const.RENDER_CONTIGEOUS:
            x_field, y_field, weight = self.field_w_weight(bbox0, region_id=None,
                out_resolution=out_resolution, offsetting=False, compute_wt=True)
            if weight is None:
                mask = None
            else:
                mask = weight > 0
        elif mode == const.RENDER_FULL:
            regions = self.region_finder_for_bbox(bbox0, offsetting=False)
            if regions.size == 0:
                return empty_output
            elif regions.size == 1:
                blend = const.BLEND_NONE
            else:
                blend = kwargs.get('blend', const.BLEND_MAX)
            initialized = False
            for rid in regions:
                xf, yf, wt = self.field_w_weight(bbox0, region_id=rid,
                    out_resolution=out_resolution, offsetting=False, compute_wt=True)
                if xf is None:
                    continue
                if not initialized:
                    x_field = np.zeros_like(xf)
                    y_field = np.zeros_like(yf)
                    weight = np.zeros_like(wt)
                    initialized = True
                if blend == const.BLEND_LINEAR:
                    x_field = x_field + xf * wt
                    y_field = y_field + yf * wt
                    weight = weight + wt
                elif blend == const.BLEND_MAX:
                    tmask = wt >= weight
                    x_field[tmask] = xf[tmask]
                    y_field[tmask] = yf[tmask]
                    weight[tmask] = wt[tmask]
                else:
                    tmask = wt > 0
                    x_field[tmask] = xf[tmask]
                    y_field[tmask] = yf[tmask]
                    weight[tmask] = wt[tmask]
            if not initialized:
                return empty_output
            if blend == const.BLEND_LINEAR:
                with np.errstate(invalid='ignore', divide='ignore'):
                    x_field = np.nan_to_num(x_field / weight, nan=0,posinf=0, neginf=0)
                    y_field = np.nan_to_num(y_field / weight, nan=0,posinf=0, neginf=0)
            mask = weight > 0
        else:
            raise ValueError
        if self._geodesic_mask:
            mask = weight
        return x_field, y_field, mask


    def crop(self, bbox, **kwargs):
        image_loader = kwargs.get('image_loader', self._image_loader)
        log_sigma = kwargs.get('log_sigma', 0) # apply laplacian of gaussian filter if > 0
        if image_loader is None:
            raise RuntimeError('Image loader not defined.')
        x_field, y_field, mask = self.crop_field(bbox, **kwargs)
        if self._geodesic_mask:
            weight = mask
            mask = weight > 0
        if image_loader.resolution != self.resolution:
            scale = self.resolution / image_loader.resolution
            x_field = spatial.scale_coordinates(x_field, scale)
            y_field = spatial.scale_coordinates(y_field, scale)
        imgt = common.render_by_subregions(x_field, y_field, mask, image_loader, **kwargs)
        if (log_sigma > 0) and (imgt is not None):
            if len(imgt.shape) > 2:
                imgt = np.moveaxis(imgt, -1, 0)
            imgt = common.masked_dog_filter(imgt, log_sigma, mask=mask)
            if len(imgt.shape) > 2:
                imgt = np.moveaxis(imgt, 0, -1)
        if (self._geodesic_mask) and (imgt is not None):
            dtp = imgt.dtype
            kk = 2
            weight = (np.arctan((weight - 0.5)*2*kk*np.pi) + np.arctan(kk*np.pi)) / (2*np.arctan(kk*np.pi))
            imgt = (imgt * weight).astype(dtp)
        return imgt


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


    @property
    def dtype(self):
        if self._dtype is None:
            if hasattr(self._image_loader, 'dtype') and self._image_loader.dtype is not None:
                self._dtype = self._image_loader.dtype
            else:
                self._dtype = np.uint8
        return self._dtype


    @property
    def default_fillval(self):
        if self._default_fillval is None:
            if hasattr(self._image_loader, 'default_fillval') and self._image_loader.default_fillval is not None:
                self._default_fillval = self._image_loader.default_fillval
            else:
                self._default_fillval = 0
        return self._default_fillval



def render_whole_mesh(mesh, image_loader, prefix, **kwargs):
    driver = kwargs.get('driver', 'image')
    num_workers = kwargs.pop('num_workers', 1)
    max_tile_per_job = kwargs.pop('max_tile_per_job', None)
    canvas_bbox = kwargs.pop('canvas_bbox', None)
    tile_size = kwargs.pop('tile_size', (4096, 4096))
    pattern = kwargs.pop('pattern', 'tr{ROW_IND}_tc{COL_IND}.png')
    scale = kwargs.pop('scale', 1)
    one_based = kwargs.pop('one_based', False)
    if 'weight_params' in kwargs and isinstance(kwargs['weight_params'], str):
        kwargs['weight_params'] = const.TRIFINDER_MODE_LIST.index(kwargs['weight_params'])
    keywords = ['{ROW_IND}', '{COL_IND}', '{X_MIN}', '{Y_MIN}', '{X_MAX}', '{Y_MAX}']
    resolution = image_loader.resolution
    mesh.change_resolution(resolution / scale)
    kwargs.setdefault('target_resolution', resolution / scale)
    tileht, tilewd = tile_size
    if canvas_bbox is None:
        gx_min, gy_min, gx_max, gy_max = mesh.bbox(gear=const.MESH_GEAR_MOVING)
        x0, y0 = 0, 0
    else:
        gx_min, gy_min, gx_max, gy_max = canvas_bbox
        x0, y0 = gx_min, gy_min
        gx_max = gx_max - gx_min
        gy_max = gy_max - gy_min
        gx_min, gy_min = 0, 0
    if driver != 'image':
        gx_max, gy_max = int(np.ceil(gx_max)), int(np.ceil(gy_max))
        gx_min, gy_min = int(np.floor(gx_min)), int(np.floor(gy_min))
        while tile_ht > gy_max or tile_wd > gx_max:
            tile_ht = tile_ht // 2
            tile_wd = tile_wd // 2
    tx_max = int(np.ceil(gx_max  / tilewd))
    ty_max = int(np.ceil(gy_max  / tileht))
    tx_min = int(np.floor(gx_min  / tilewd))
    ty_min = int(np.floor(gy_min  / tileht))
    cols, rows = np.meshgrid(np.arange(tx_min, tx_max), np.arange(ty_min, ty_max))
    cols, rows = cols.ravel(), rows.ravel()
    idxz = common.z_order(np.stack((rows, cols), axis=-1))
    cols, rows = cols[idxz], rows[idxz]
    bboxes = []
    region = mesh.shapely_regions(gear=const.MESH_GEAR_MOVING, offsetting=True)
    if driver == 'image':
        filenames = []
        for r, c in zip(rows, cols):
            bbox = (c*tilewd + x0, r*tileht + y0, (c+1)*tilewd + x0, (r+1)*tileht + y0)
            if not region.intersects(shpgeo.box(*bbox)):
                continue
            bboxes.append(bbox)
            xmin, ymin, xmax, ymax = bbox
            keyword_replaces = [str(r+one_based), str(c+one_based), str(xmin), str(ymin), str(xmax), str(ymax)]
            fname = pattern
            for kw, kwr in zip(keywords, keyword_replaces):
                fname = fname.replace(kw, kwr)
            filenames.append(prefix + fname)
        rendered = {}
    else:
        bboxes_out = []
        if not prefix.endswith('/'):
            prefix = prefix + '/'
        kv_headers = ('gs://', 'http://', 'https://', 'file://', 'memory://', 's3://')
        for kvh in kv_headers:
            if prefix.startswith(kvh):
                break
        else:
            prefix = 'file://' + prefix
        number_of_channels = image_loader.number_of_channels
        dtype = kwargs.get('dtype_out', image_loader.dtype)
        fillval = kwargs.get('fillval', image_loader.default_fillval)
        schema = {
            "chunk_layout":{
                "grid_origin": [0, 0, 0, 0],
                "inner_order": [3, 2, 1, 0],
                "read_chunk": {"shape": [tile_wd, tile_ht, 1, number_of_channels]},
                "write_chunk": {"shape": [tile_wd, tile_ht, 1, number_of_channels]},
            },
            "domain":{
                "exclusive_max": [gx_max, gy_max, 1, number_of_channels],
                "inclusive_min": [0, 0, 0, 0],
                "labels": ["x", "y", "z", "channel"]
            },
            "dimension_units": [[mesh.resolution, "nm"], [mesh.resolution, "nm"], [SECTION_THICKNESS, "nm"], None],
            "dtype": np.dtype(dtype).name,
            "rank" : 4
        }
        mip_level_str = str(int(np.log(mesh.resolution/montage_resolution())/np.log(2)))
        if driver == 'zarr':
            ts_specs = {
                "driver": "zarr",
                "kvstore": prefix + mip_level_str + '/',
                "key_encoding": ".",
                "metadata": {
                    "zarr_format": 2,
                    "fill_value": fillval,
                    "compressor": {"id": "gzip", "level": 6}
                },
                "schema": schema,
                "open": True,
                "create": True,
                "delete_existing": False
            }
        elif driver == 'n5':
            ts_specs = {
                "driver": "n5",
                "kvstore": prefix + 's' + mip_level_str + '/',
                "metadata": {
                    "compression": {"type": "gzip"}
                },
                "schema": schema,
                "open": True,
                "create": True,
                "delete_existing": False
            }
        elif driver == 'neuroglancer_precomputed':
            if tile_ht % 256 == 0:
                read_ht = 256
            else:
                read_ht = tile_ht
            if tile_wd % 256 == 0:
                read_wd = 256
            else:
                read_wd = tile_wd
            schema["codec"]= {
                "driver": "neuroglancer_precomputed",
                "encoding": "raw",
                "shard_data_encoding": "gzip"
            }
            schema['chunk_layout']["read_chunk"]["shape"] = [read_wd, read_ht, 1, number_of_channels]
            ts_specs = {
                "driver": "neuroglancer_precomputed",
                "kvstore": prefix,
                "schema": schema,
                "open": True,
                "create": True,
                "delete_existing": False
            }
        else:
            raise ValueError(f'{driver} not supported')
        for r, c in zip(rows, cols):
            bbox = (c*tilewd + x0, r*tileht + y0, (c+1)*tilewd + x0, (r+1)*tileht + y0)
            bbox_out = (c*tilewd, r*tileht, (c+1)*tilewd, (r+1)*tileht)
            if not region.intersects(shpgeo.box(*bbox)):
                continue
            bboxes.append(bbox)
            bboxes_out.append(bbox_out)
        rendered  = []
    num_tiles = len(bboxes)
    if num_tiles == 0:
        return rendered
    if isinstance(image_loader, dal.AbstractImageLoader):
        image_loader = image_loader.init_dict()
    if driver == 'image':
        target_func = partial(subprocess_render_mesh_tiles, image_loader, **kwargs)
    else:
        target_func = partial(subprocess_render_mesh_tiles, image_loader, outnames=ts_specs, **kwargs)
    if (num_workers > 1) and (num_tiles > 1):
        num_tile_per_job = max(1, int((num_tiles // num_workers)**0.5))
        if max_tile_per_job is not None:
            num_tile_per_job = min(num_tile_per_job, max_tile_per_job)
            max_tasks_per_child = max(1, round(max_tile_per_job/num_tile_per_job))
        else:
            max_tasks_per_child = None
        N_jobs = max(1, round(num_tiles / num_tile_per_job))
        indices = np.round(np.linspace(0, num_tiles, num=N_jobs+1, endpoint=True))
        indices = np.unique(indices).astype(np.uint32)
        bboxes_list = []
        bboxes_out_list = []
        filenames_list = []
        bbox_unions = []
        for idx0, idx1 in zip(indices[:-1], indices[1:]):
            idx0, idx1 = int(idx0), int(idx1)
            bbox_t = bboxes[idx0:idx1]
            bboxes_list.append(bbox_t)
            bbox_unions.append(common.bbox_enlarge(common.bbox_union(bbox_t), tile_size[0]//2))
            if driver == 'image':
                filenames_list.append(filenames[idx0:idx1])
            else:
                bboxes_out_list.append(bboxes_out[idx0:idx1])
        submeshes = mesh.submeshes_from_bboxes(bbox_unions, save_material=True)
        args_list = []
        for k in range(len(submeshes)):
            msh = submeshes[k]
            if msh is None:
                continue
            msh_dict = msh.get_init_dict(save_material=True, vertex_flags=(const.MESH_GEAR_INITIAL, const.MESH_GEAR_MOVING))
            bbox = bboxes_list[k]
            if driver == 'image':
                fnames = filenames_list[k]
                args_list.append((msh_dict, bbox, fnames))
            else:
                bbox_out = bboxes_out_list[k]
                args_list.append((msh_dict, bbox, bbox_out))
        for res in submit_to_workers(target_func, args=args_list, num_workers=num_workers, max_tasks_per_child=max_tasks_per_child):
            if isinstance(rendered, dict):
                rendered.update(res)
            else:
                rendered.extend(res)
    else:
        if driver == 'image':
            rendered = target_func(mesh, bboxes, filenames)
        else:
            rendered = target_func(mesh, bboxes, bboxes_out=bboxes_out)
    return rendered


def subprocess_render_mesh_tiles(imgloader, mesh, bboxes, outnames, **kwargs):
    target_resolution = kwargs.pop('target_resolution')
    bboxes_out = kwargs.pop('bboxes_out', bboxes)
    if isinstance(imgloader, (str, dict)):
        imgloader = dal.get_loader_from_json(imgloader)
    if isinstance(mesh, str):
        M = Mesh.from_h5(mesh)
    elif isinstance(mesh, dict):
        M = Mesh(**mesh)
    else:
        M = mesh
    M.change_resolution(target_resolution)
    if isinstance(outnames, (dict, ts.TensorStore)):
        use_tensorstore = True
    else:
        use_tensorstore = False
    renderer = MeshRenderer.from_mesh(M, **kwargs)
    if renderer is None:
        if use_tensorstore:
            return []
        else:
            return {}
    renderer.link_image_loader(imgloader)
    fillval = kwargs.get('fillval', renderer.default_fillval)
    if use_tensorstore:
        rendered = []
        if not isinstance(outnames, ts.TensorStore):
            dataset = ts.open(outnames).result()
        else:
            dataset = outnames
        driver = dataset.spec().to_json()['driver']
        data_domain = dataset.domain
        inclusive_min = data_domain.inclusive_min
        exclusive_max = data_domain.exclusive_max
        ts_xmin, ts_ymin = inclusive_min[0], inclusive_min[1]
        ts_xmax, ts_ymax = exclusive_max[0], exclusive_max[1]
        ts_bbox = (ts_xmin, ts_ymin, ts_xmax, ts_ymax)
        if driver in ('neuroglancer_precomputed', 'n5'):
            kwargs['fillval'] = 0
        for bbox, bbox_out in zip(bboxes, bboxes_out):
            imgt = renderer.crop(bbox, **kwargs)
            if (imgt is not None) and np.any(imgt != fillval, axis=None):
                img_crp, indx = common.crop_image_from_bbox(imgt, bbox_out, ts_bbox,
                                                    return_index=True, flip_indx=True)
                if img_crp is None:
                    continue
                data_view = dataset[indx[1], indx[0]]
                try:
                    data_view.write(img_crp.T.reshape(data_view.shape)).result(timeout=TS_TIMEOUT)
                except TimeoutError:
                    dataset = ts.open(dataset.spec(minimal_spec=True)).result(timeout=TS_TIMEOUT)
                    data_view = dataset[indx[1], indx[0]]
                    data_view.write(img_crp.T.reshape(data_view.shape)).result(timeout=TS_TIMEOUT)
                rendered.append(tuple(bbox))
    else:
        rendered = {}
        for fname, bbox in zip(outnames, bboxes):
            if storage.file_exists(fname):
                rendered[fname] = bbox
                continue
            imgt = renderer.crop(bbox, **kwargs)
            if (imgt is not None) and np.any(imgt != fillval, axis=None):
                common.imwrite(fname, imgt)
                rendered[fname] = bbox
    return rendered



class VolumeRenderer:
    """
    A class to render Tensorstore Volume from mesh transformation.
    Args:
        meshes(list): mesh files of transformations
        loaders(list): list of loaders from rendering step in stitching
        kvstore(str): kvstore of the output tensorstore
        z_indx: z indices of each section in the output tensorstore
    Kwargs:
        driver: tensorstore driver type
        chunk_shape: shape of (write) chunks
        read_chunk_shape: shape of read chunks
        canvas_bbox: bounding box of the canvas. By default it will be 
            offset to (0,0) in the output space
        resolution: resolution of the renderer
        out_offset: xy offset of the output space
    """
    def __init__(self, meshes, loaders, kvstore, z_indx=None, **kwargs):
        if z_indx is None:
            z_indx = np.arange(len(meshes))
        assert len(meshes) == len(loaders)
        assert len(meshes) == len(z_indx)
        self._meshes = meshes
        self._zindx = z_indx
        self._zmin = kwargs.get('z_min', 0)
        self._zmax = kwargs.get('z_max', None)
        self._jpeg_compression = kwargs.get('jpeg_compression', False)
        self._pad_to_tile_size = kwargs.get('pad_to_tile_size', True)
        self._loaders = loaders
        driver = kwargs.get('driver', 'neuroglancer_precomputed')
        self.flag_dir = kwargs.get('flag_dir', None)
        self.checkpoint_dir = kwargs.get('checkpoint_dir', storage.join_paths(self.flag_dir, 'checkpoint'))
        self._canvas_bbox = kwargs.get('canvas_bbox', None)
        if self._canvas_bbox is not None:
            default_offset = -np.array(self._canvas_bbox)[:2].reshape(1,2)
        else:
            default_offset =  np.zeros((1,2), dtype=np.int64)
        self._offset = kwargs.get('out_offset', default_offset.astype(np.int64))
        self._ts_verified = False
        self.resolution = kwargs.get('resolution', montage_resolution())
        self.mip = int(np.log2(self.resolution / montage_resolution()))
        self._number_of_channels = kwargs.get('number_of_channels', None)
        self._dtype = kwargs.get('dtype', None)
        self._chunk_shape = kwargs.get('chunk_shape', (1024, 1024, 16))
        self._read_chunk_shape = kwargs.get('read_chunk_shape', self._chunk_shape)
        schema = {'dimension_units': [[self.resolution, "nm"], [self.resolution, "nm"],
                                      [SECTION_THICKNESS, "nm"], None]}
        self._ts_spec = {'driver': driver, 'kvstore': kvstore, 'schema': schema}


    def plan_one_slab(self, z_ind=0, **kwargs):
        num_workers = kwargs.get('num_workers', 1)
        max_tile_per_job = kwargs.get('max_tile_per_job', None)
        cache_capacity = kwargs.pop('cache_capacity', None)
        _, _, Z0, _, _, Z1 = self.writer.write_grids
        z0, z1 = Z0[z_ind], Z1[z_ind]
        flag_name = kwargs.get('flag_name', f'z{z0}_{z1}')
        render_seriers = []
        zz = np.arange(z0, z1)
        flag_file = storage.join_paths(self.flag_dir, flag_name + '.json')
        checkpoint_file = storage.join_paths(self.checkpoint_dir, flag_name + '.h5')
        if (flag_file is not None) and storage.file_exists(flag_file):
            with storage.File(flag_file, 'r') as f:
                z_rendered = json.load(f)
        else:
            z_rendered = []
        check_points = {z: False for z in z_rendered}
        z_to_render = [z for z in zz if z not in z_rendered]
        if len(z_to_render) == 0:
            return render_seriers, check_points
        if (checkpoint_file is not None) and storage.file_exists(checkpoint_file):
            with H5File(checkpoint_file, 'r') as f:
                for z in z_to_render:
                    if str(z) in f:
                        check_points[z] = f[str(z)][()]
        id_x, id_y = self.writer.morton_xy_grid()
        bboxes = self.writer.grid_indices_to_bboxes(id_x, id_y)
        bboxes_tree = shapely.STRtree([shpgeo.box(*bbox) for bbox in bboxes])
        num_xy_grids = id_x.size
        hit_counts = np.zeros(num_xy_grids, dtype=np.uint16)
        full_meshes = {}
        loaders = {}
        for z, rm in zip(z_to_render, self.region_generator(indx=z_to_render)):
            rr = rm[0]
            if rr is None:
                continue
            if z not in check_points:
                idxt = bboxes_tree.query(rr, predicate='intersects')
                bb = np.zeros(num_xy_grids, dtype=bool)
                bb[idxt] = True
                check_points[z] = bb
            hit_counts += check_points[z]
            full_meshes[z] = rm[1]
            loaders[z] = self.loader_lut.get(z, None)
        if len(full_meshes) == 0:
            return render_seriers, check_points
        midx_hits = np.nonzero(hit_counts > 0)[0]
        bboxes = bboxes[midx_hits]
        hit_counts = hit_counts[midx_hits]
        hit_counts_acc = np.insert(np.cumsum(hit_counts), 0, 0)
        num_tiles = hit_counts_acc[-1]
        num_tile_per_job = max(1, num_tiles // num_workers)
        if max_tile_per_job is not None:
            num_tile_per_job = min(num_tile_per_job, max_tile_per_job)
        if cache_capacity is not None:
            chunk_mb = np.prod(self.writer.write_chunk_shape[:2]) * self.writer.number_of_channels / (1024 ** 2)
            max_chunk_per_proc = max(1, cache_capacity / (chunk_mb * num_workers))
            num_tile_per_job = min(num_tile_per_job, max_chunk_per_proc)
        N_jobs = max(1, round(num_tiles / num_tile_per_job))
        indices_tile = np.round(np.linspace(0, num_tiles, num=N_jobs+1, endpoint=True))
        indices_chunk = np.searchsorted(hit_counts_acc, indices_tile, side='left')
        indices_chunk = np.unique(indices_chunk).astype(np.uint64)
        out_ts = self.ts_spec
        bboxes_unions = []
        b_dilate = np.max(self.writer.write_chunk_shape[:2]) // 2
        task_id = 0
        for idx0, idx1 in zip(indices_chunk[:-1], indices_chunk[1:]):
            idx0, idx1 = int(idx0), int(idx1)
            bbox_b = bboxes[idx0:idx1]
            if bbox_b.size == 0:
                continue
            mindx = midx_hits[idx0:idx1]
            b_flag = {z: check_points[z][mindx] for z in z_to_render}
            bkw = { 'task_id': task_id,
                    'loaders': loaders,
                    'meshes': {},
                    'morton_indx': mindx,
                    'out_ts': out_ts,
                    'target_resolution': self.resolution,
                    'mip': self.mip,
                    'offset': self._offset,
                    'flags': b_flag}
            task_id = task_id + 1
            render_seriers.append(bkw)
            bboxes_unions.append(common.bbox_enlarge(common.bbox_union(bbox_b), b_dilate))
        for z, mesh in full_meshes.items():
            submeshes = mesh.submeshes_from_bboxes(bboxes_unions, save_material=True)
            for msh, bkw in zip(submeshes, render_seriers):
                if msh is None:
                    bkw['meshes'][z] = None
                else:
                    msh_dict = msh.get_init_dict(save_material=True, vertex_flags=(const.MESH_GEAR_INITIAL, const.MESH_GEAR_MOVING))
                    bkw['meshes'][z] = msh_dict
        return render_seriers, check_points


    def render_volume(self, skip_indx=None , **kwargs):
        num_workers = kwargs.pop('num_workers', 1)
        max_tile_per_job = kwargs.pop('max_tile_per_job', None)
        cache_capacity = kwargs.pop('cache_capacity', None)
        logger_info = kwargs.pop('logger', None)
        logger = logging.get_logger(logger_info)
        _, _, Z0, _, _, Z1 = self.writer.write_grids
        Z_indices = np.arange(Z0.size)
        if skip_indx is not None:
            Z_indices = Z_indices[skip_indx]
        for z_ind in Z_indices:
            t0 = time.time()
            err_raised = False
            num_chunks = 0
            flag_name = f'z{Z0[z_ind]}_{Z1[z_ind]}'
            flag_file = storage.join_paths(self.flag_dir, flag_name + '.json')
            checkpoint_file = storage.join_paths(self.checkpoint_dir, flag_name + '.h5')
            render_seriers, checkpoints = self.plan_one_slab(z_ind=z_ind, num_workers=num_workers, flag_name=flag_name, max_tile_per_job=max_tile_per_job, cache_capacity=cache_capacity)
            if len(render_seriers) == 0:
                continue
            morton_LUT = {s['task_id']: s['morton_indx'] for s in render_seriers}
            if cache_capacity is None:
                max_tasks_per_child = None
            else:
                max_tasks_per_child = 1
            logger.info(f'start rendering block z={Z0[z_ind]}->{Z1[z_ind]}')
            actual_num_workers = min(num_workers, len(render_seriers))
            for bkw in render_seriers:
                bkw.update(kwargs)
            t_check = time.time()
            res_cnt = 0
            for res in submit_to_workers(subprocess_render_partial_ts_slab, kwargs=render_seriers, num_workers=actual_num_workers, max_tasks_per_child=max_tasks_per_child):
                task_id, flag_b, errmsg = res
                res_cnt += 1
                if len(errmsg) > 0:
                    err_raised = True
                    logger.error(errmsg)
                morton_idx = morton_LUT[task_id]
                added_chunk = np.zeros(morton_idx.size, dtype=bool)
                for zz, flg in flag_b.items():
                    old_flag =  checkpoints[zz][morton_idx]
                    new_flag = old_flag & flg
                    added_chunk = added_chunk | (new_flag < old_flag)
                    checkpoints[zz][morton_idx] = new_flag
                num_chunks += np.sum(added_chunk)
                if (checkpoint_file is not None) and ((time.time() - t_check) > CHECKPOINT_TIME_INTERVAL) and (res_cnt >= num_workers):
                    res_cnt = 0
                    t_check = time.time()
                    storage.makedirs(self.checkpoint_dir)
                    with H5File(checkpoint_file, 'w') as f:
                        for zz, flg in checkpoints.items():
                            if isinstance(flg, np.ndarray):
                                f.create_dataset(str(zz), data=flg, compression="gzip")
            if self.flag_dir is not None:
                if num_chunks > 0:
                    if err_raised:
                        z_rendered = []
                        for zz, flg in checkpoints.items():
                            if not np.any(flg):
                                z_rendered.append(zz)
                    else:
                        z_rendered = list(checkpoints.keys())
                    if len(z_rendered) > 0:
                        z_rendered = [int(zz) for zz in z_rendered]
                        with storage.File(flag_file, 'w') as f:
                            json.dump(z_rendered, f)
                    if not err_raised and (checkpoint_file is not None):
                        storage.remove_file(checkpoint_file)
            logger.info(f'blocks z={Z0[z_ind]}->{Z1[z_ind]}: added {num_chunks} chunks | {(time.time()-t0)/60} min')
        return self.writer.spec


    @property
    def canvas_bbox(self):
        if self._canvas_bbox is None:
            bbox_union = None
            for rm in self.region_generator():
                rr = rm[0]
                if rr is None:
                    continue
                bbox = np.array(rr.bounds)
                bbox[:2] = np.floor(bbox[:2])
                bbox[-2:] = np.ceil(bbox[-2:]) + 1
                bbox = bbox.astype(np.int64)
                if bbox_union is None:
                    bbox_union = bbox
                else:
                    bbox_union = common.bbox_union((bbox_union, bbox))
            self._canvas_bbox = bbox_union
            self._offset = -bbox_union[:2].reshape(1,2)
        return self._canvas_bbox


    def region_generator(self, indx=None):
        if indx is None:
            indx = self._zindx
        mesh_lut = self.mesh_lut
        for z in indx:
            M = VolumeRenderer._get_mesh(mesh_lut.get(z, None))
            if M is None:
                yield None, None
            else:
                M.change_resolution(self.resolution)
                msk = M.triangle_mask_for_render(cache=False)
                R = M.shapely_regions(gear=const.MESH_GEAR_MOVING, tri_mask=msk, offsetting=True)
                yield R, M


    @property
    def mesh_lut(self):
        if not hasattr(self, '_mesh_lut'):
            self._mesh_lut = {z: msh for z, msh in zip(self._zindx, self._meshes)}
        return self._mesh_lut


    @property
    def loader_lut(self):
        if not hasattr(self, '_loader_lut'):
            self._loader_lut = {z: ldr for z, ldr in zip(self._zindx, self._loaders)}
        return self._loader_lut


    @property
    def number_of_channels(self):
        if self._number_of_channels is None:
            for ldr in self._loaders:
                loader = VolumeRenderer._get_loader(ldr, mip=self.mip)
                if loader is not None:
                    break
            else:
                raise RuntimeError('no valid loader found')
            self._number_of_channels = loader.number_of_channels
            if self._dtype is None:
                self._dtype = loader.dtype
        return self._number_of_channels


    @property
    def dtype(self):
        if self._dtype is None:
            for ldr in self._loaders:
                loader = VolumeRenderer._get_loader(ldr, mip=self.mip)
                if loader is not None:
                    break
            else:
                raise RuntimeError('no valid loader found')
            self._dtype = loader.dtype
            if self._number_of_channels is None:
                self._number_of_channels = loader.number_of_channels
        return self._dtype


    def _verify_ts(self):
        if not self._ts_verified:
            spec_copy = self._ts_spec.copy()
            spec_copy.update({'open': True, 'create': False, 'delete_existing': False})
            try:
                writer = dal.TensorStoreWriter.from_json_spec(spec_copy)
                dataset = writer.dataset
            except ValueError:
                spec_copy.update({'create': True})
                xmin, ymin, xmax, ymax = self.canvas_bbox
                _offset = self._offset.ravel()
                xmin, ymin, xmax, ymax = xmin + _offset[0], ymin + _offset[1], xmax + _offset[0], ymax + _offset[1]
                zmin, zmax = np.min(self._zindx), np.max(self._zindx) + 1
                if (self._zmin is not None) and zmin > self._zmin:
                    zmin = self._zmin
                if (self._zmax is not None) and zmax < self._zmax:
                    zmax = self._zmax
                write_chunk = list(self._chunk_shape)
                read_chunk = list(self._read_chunk_shape)
                if self._pad_to_tile_size:
                    shp_x, shp_y = read_chunk[:2]
                    xmin = int(np.floor(xmin / shp_x)) * shp_x
                    ymin = int(np.floor(ymin / shp_y)) * shp_y
                    xmax = int(np.ceil(xmax / shp_x)) * shp_x
                    ymax = int(np.ceil(ymax / shp_y)) * shp_y
                canvas_min = np.array((xmin, ymin, zmin))
                canvas_max = np.array((xmax, ymax, zmax))
                num_channels = self.number_of_channels
                if len(write_chunk) < 3:
                    write_chunk = write_chunk + [1]
                if len(write_chunk) < 4:
                    write_chunk = write_chunk + [num_channels]
                if len(read_chunk) < 3:
                    read_chunk = read_chunk + [1]
                if len(read_chunk) < 4:
                    read_chunk = read_chunk + [num_channels]
                exclusive_max = list(canvas_max) + [num_channels]
                inclusive_min = list(canvas_min) + [0]
                schema_extra = {
                    "chunk_layout":{
                        "grid_origin": [0, 0, 0, 0],
                        "inner_order": [3, 2, 1, 0],
                        "read_chunk": {"shape_soft_constraint": read_chunk},
                        "write_chunk": {"shape_soft_constraint": write_chunk},
                    },
                    "domain":{
                        "exclusive_max": exclusive_max,
                        "inclusive_min": inclusive_min,
                        "labels": ["x", "y", "z", "channel"]
                    },
                    "codec":{
                        "driver": "neuroglancer_precomputed"
                    },
                    "dimension_units": [[self.resolution, "nm"], [self.resolution, "nm"], [SECTION_THICKNESS, "nm"], None],
                    "dtype": np.dtype(self.dtype).name,
                    "rank" : 4
                }
                sharding = np.any(np.array(read_chunk) != np.array(write_chunk))
                if self._jpeg_compression:
                    schema_extra["codec"].update({"encoding": "jpeg", "jpeg_quality": 95})
                    if sharding:
                        schema_extra["codec"].update({"shard_data_encoding": "raw"})
                else:
                    schema_extra["codec"].update({"encoding": "raw"})
                    if sharding:
                        schema_extra["codec"].update({"shard_data_encoding": "gzip"})
                spec_copy['schema'].update(schema_extra)
                writer = dal.TensorStoreWriter.from_json_spec(spec_copy)
                dataset = writer.dataset
                self._ts_spec = dataset.spec(minimal_spec=True).to_json()
            self._writer = writer
            self._read_chunk_shape = dataset.schema.chunk_layout.read_chunk.shape[:3]
            self._chunk_shape = dataset.schema.chunk_layout.write_chunk.shape[:3]
            self._ts_verified = True


    @property
    def ts_spec(self):
        self._verify_ts()
        return self._ts_spec


    @property
    def writer(self):
        self._verify_ts()
        return self._writer

    @property
    def chunk_shape(self):
        self._verify_ts()
        return self._chunk_shape


    @property
    def read_chunk_shape(self):
        self._verify_ts()
        return self._read_chunk_shape


    @staticmethod
    def _get_mesh(msh):
        if isinstance(msh, Mesh):
            M = msh
        elif isinstance(msh, dict):
            M = Mesh(**msh)
        elif isinstance(msh, str) and storage.file_exists(msh):
            M = Mesh.from_h5(msh)
        else:
            M = None
        return M


    @staticmethod
    def _get_loader(ldr, **kwargs):
        if isinstance(ldr, dal.AbstractImageLoader):
            loader = ldr
        elif ldr is None:
            loader = None
        else:
            try:
                loader = dal.get_loader_from_json(ldr, **kwargs)
            except Exception:
                loader = None
        return loader


def subprocess_render_partial_ts_slab(loaders, meshes, morton_indx, out_ts, **kwargs):
    task_id = kwargs.pop('task_id', None)
    target_resolution = kwargs.pop('target_resolution')
    mip = kwargs.pop('mip', None)
    loader_config = kwargs.pop('loader_config', {})
    offset = kwargs.pop('offset', np.zeros((1,2), dtype=np.int64))
    flags0 = kwargs.pop('flags', {})
    flags = {}
    loaders = {int(z): VolumeRenderer._get_loader(ldr, mip=mip) for z, ldr in loaders.items()}
    meshes = {int(z): VolumeRenderer._get_mesh(msh) for z, msh in meshes.items()}
    zindx = []
    renderers = {}
    to_skip = True
    err_str = ''
    for z in set(loaders.keys()).intersection(set(meshes.keys())):
        ldr = loaders[z]
        msh = meshes[z]
        if (ldr is None) or (msh is None):
            rndr = None
        elif (z in flags0) and (not np.any(flags0[z])):
            rndr = None
        else:
            msh.change_resolution(target_resolution)
            rndr = MeshRenderer.from_mesh(msh, **loader_config)
            if rndr is not None:
                rndr.link_image_loader(ldr)
                zindx.append(z)
                flags[z] = np.array(flags0.get(z, np.ones(len(morton_indx), dtype=bool)))
                if to_skip:
                    fillval = kwargs.get('fillval', rndr.default_fillval)
                    to_skip = False
        renderers[z] = rndr
    if to_skip:
        return task_id, flags, err_str
    if isinstance(out_ts, dal.TensorStoreWriter):
        writer = out_ts
    else:
        writer = dal.TensorStoreWriter.from_json_spec(out_ts)
    id_x, id_y = writer.morton_xy_grid(morton_indx)
    bboxes_out = writer.grid_indices_to_bboxes(id_x, id_y)
    bboxes = bboxes_out - np.tile(offset, 2).reshape(1, 4)
    for kb in range(len(bboxes)):
        bbox, bbox_out = bboxes[kb], bboxes_out[kb]
        updated = False
        imgs = []
        bbox_3d = []
        flags_b = []
        for z in zindx:
            try:    # http error may crash the program
                if (z in flags) and (not flags[z][kb]):
                    continue
                rndr = renderers[z]
                if rndr is None:
                    continue
                imgt = rndr.crop(bbox, **kwargs)
                if (imgt is not None) and np.any(imgt != fillval, axis=None):
                    imgt = np.swapaxes(imgt, 0, 1).copy()
                    imgs.append(imgt)
                    bbox_3d.append([*bbox_out[:2], z, *bbox_out[2:], z+1])
                    updated = True
                flags_b.append(z)
            except Exception as err:
                if isinstance(err, TimeoutError):
                    err_str = err_str + f'\tz={z} Tensorstore timed out.\n'
                else:
                    err_str = err_str + f'\tz={z} {err}\n'
        if updated:
            try:
                writer.write_chunks_w_transaction(bboxes=bbox_3d, imgs=imgs)
            except Exception as err:
                if isinstance(err, TimeoutError):
                    err_str = err_str + f'\tcommit error: Tensorstore timed out.\n'
                else:
                    err_str = err_str + f'\tcommit error: {err}\n'
                break
            else:
                for z in flags_b:
                    flags[z][kb] = False
    for z in zindx:
        val = flags[z]
        if not np.any(val):
            flags[z] = False
        elif np.all(val):
            flags[z] = True
    return task_id, flags, err_str
