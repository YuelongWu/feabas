from collections import defaultdict, OrderedDict

import cv2
import numpy as np
import shapely.geometry as shpgeo
from shapely.ops import unary_union, linemerge, polygonize
from shapely import wkb, get_coordinates, minimum_rotated_rectangle

from feabas import dal, common, material
import feabas.constant as const
from feabas.config import data_resolution
from feabas.storage import h5file_class

H5File = h5file_class()

JOIN_STYLE = shpgeo.JOIN_STYLE.mitre


def fit_affine(pts0, pts1, return_rigid=False, weight=None, svd_clip=(1,1), avoid_flip=True):
    # pts0 = pts1 @ A
    pts0 = pts0.reshape(-1,2)
    pts1 = pts1.reshape(-1,2)
    assert pts0.shape[0] == pts1.shape[0]
    mm0 = pts0.mean(axis=0)
    mm1 = pts1.mean(axis=0)
    pts0 = pts0 - mm0
    pts1 = pts1 - mm1
    std0 = np.sum(np.std(pts0, axis=0)**2, axis=None) ** 0.5
    std1 = np.sum(np.std(pts1, axis=0)**2, axis=None) ** 0.5
    std_scl = max(std0, std1)
    if std_scl < 1e-6:
        std_scl = 1
    pts0_pad = np.insert(pts0/std_scl, 2, 1, axis=-1)
    pts1_pad = np.insert(pts1/std_scl, 2, 1, axis=-1)
    if weight is not None:
        weight = weight ** 0.5
        pts0_pad = pts0_pad * weight.reshape(-1, 1)
        pts1_pad = pts1_pad * weight.reshape(-1, 1)
    res = np.linalg.lstsq(pts1_pad, pts0_pad, rcond=None)
    r1 = np.linalg.matrix_rank(pts0_pad)
    A = res[0]
    r = min(res[2], r1)
    if avoid_flip and np.linalg.det(A) < 0:
        r = 2
    if r == 1:
        A = np.eye(3)
    elif r == 2:
        pts0_rot90 = pts0[:,::-1] * np.array([1,-1])
        pts1_rot90 = pts1[:,::-1] * np.array([1,-1])
        pts0 = np.concatenate((pts0, pts0_rot90), axis=0)
        pts1 = np.concatenate((pts1, pts1_rot90), axis=0)
        pts0_pad = np.insert(pts0/std_scl, 2, 1, axis=-1)
        pts1_pad = np.insert(pts1/std_scl, 2, 1, axis=-1)
        res = np.linalg.lstsq(pts1_pad, pts0_pad, rcond=None)
        A = res[0]
    if return_rigid:
        u, s, vh = np.linalg.svd(A[:2,:2], compute_uv=True)
        if svd_clip is not None:
            s = s.clip(svd_clip[0], svd_clip[-1])
        R = A.copy()
        R[:2,:2] = u @ np.diag(s) @ vh
        R[-1,:2] = R[-1,:2] + mm0 - mm1 @ R[:2,:2]
        R[:,-1] = np.array([0,0,1])
    A[-1,:2] = A[-1,:2] + mm0 - mm1 @ A[:2,:2]
    A[:,-1] = np.array([0,0,1])
    if return_rigid:
        return A, R
    else:
        return A



def scale_coordinates(coordinates, scale):
    """
    scale coordinates, at the same time align the center of the top-left corner
    pixel to (0, 0).
    """
    coordinates = np.array(coordinates, copy=False)
    if np.all(scale == 1):
        return coordinates
    else:
        return (coordinates + 0.5) * scale - 0.5



def find_contours(mask):
    """
    wrapper of findContour function in opencv to accommodate version change.
    """
    approx_mode = cv2.CHAIN_APPROX_SIMPLE  # cv2.CHAIN_APPROX_NONE
    if not np.any(mask):
        return (), None
    mask = mask.astype(np.uint8, copy=False)
    if cv2.__version__ < '4':
        _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, approx_mode)
    else:
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, approx_mode)
    contours = _pad_concave_corner(contours)
    return contours, hierarchy


def _pad_concave_corner(countours):
    padded = []
    for ct in countours:
        xy0 = ct.squeeze()
        xy1 = np.roll(xy0, 1, axis=0)
        stp = xy1 - xy0
        diag_indx = np.nonzero(np.all(stp != 0, axis=-1))[0]
        if diag_indx.size > 0:
            dxy = stp[diag_indx]
            dxy = (dxy + dxy[:,::-1] * np.array([-1, 1])) / 2
            txy = xy0[diag_indx] + dxy
            xy0 = np.insert(xy0, diag_indx, txy, axis=0)
            stp0 = (np.roll(xy0, 1, axis=0) - xy0 != 0) | (np.roll(xy0, -1, axis=0) - xy0 != 0)
            keep_indx = np.all(stp0, axis=-1)
            xy0 = xy0[keep_indx]
        padded.append(xy0)
    return padded



def countours_to_polygon(contours, hierarchy, offset, scale, upsample):
    """
    convert opencv contours to shapely (Multi)Polygons.
    Args:
        contours, hierarchy: output from find_countours.
    Kwargs:
        offsets: x, y translation offset to add to the output polygon.
        scale: scaling factor of the geometries.
        upsample: upsample factor for better accuracy.
    """
    polygons_staging = {}
    holes = {}
    buffer_r = 0.51 * scale / upsample # expand by half pixel
    for indx, ct in enumerate(contours):
        number_of_points = ct.shape[0]
        if number_of_points < 3:
            continue
        xy = scale_coordinates(ct.reshape(number_of_points, -1), scale=1/upsample)
        xy = scale_coordinates(xy + np.asarray(offset), scale=scale)
        lr = shpgeo.polygon.LinearRing(xy)
        # lr = smooth_zigzag(lr, scale=scale)
        pp = shpgeo.Polygon(lr).buffer(0)
        if lr.is_ccw:
            holes[indx] = pp
        else:
            polygons_staging[indx] = pp
    if len(polygons_staging) == 0:
        return None     # no region found
    hierarchy = hierarchy.reshape(-1, 4)
    for indx, hole in holes.items():
        parent_indx = hierarchy[indx][3]
        if parent_indx not in polygons_staging:
            raise RuntimeError('found an orphan hole...') # should never happen
        polygons_staging[parent_indx] = polygons_staging[parent_indx].difference(hole)
    return unary_union([s.buffer(0) for s in polygons_staging.values()]).buffer(buffer_r, join_style=JOIN_STYLE)



def images_to_polygons(imgs, labels, offset=(0, 0), scale=1.0, upsample=2):
    """
    Convert images to shapely Polygons.
    Args:
        imgs(MosaicLoader/ndarray/str): label images.
        labels (OrderedDict): names to labels mapper.
    Kwargs:
        offsets: global x, y translation offset to add to the output polygons.
            the scaling of the offsets is the same as the images.
        scale: scaling factor of the geometries.
        upsample: upsample factor to prevent one-line of pixels
    """
    if not isinstance(labels, dict):
        labels = OrderedDict((str(s), s) for s in labels)
    polygons = {}
    if isinstance(imgs, dal.MosaicLoader):
        xmin, ymin, xmax, ymax = imgs.bounds
        # align bounds to corners of pixels by substracting 0.5
        xmin += offset[0] - 0.5
        xmax += offset[0] - 0.5
        ymin += offset[1] - 0.5
        ymax += offset[1] - 0.5
        xmin, ymin, xmax, ymax = scale_coordinates((xmin, ymin, xmax, ymax), scale)
        extent = shpgeo.box(xmin, ymin, xmax, ymax)
        regions_staging = defaultdict(list)
        for bbox in imgs.file_bboxes(margin=3):
            tile = imgs.crop(bbox, return_empty=False)
            if tile is None:
                continue
            xy0 = np.array(bbox[:2], copy=False) + np.array(offset)
            for name, lbl in labels.items():
                if lbl is None:
                    continue
                if len(tile.shape) > 2: # RGB
                    mask = np.all(tile == np.array(lbl), axis=-1)
                else:
                    mask = (tile == lbl)
                if upsample != 1:
                    mask = cv2.resize(mask.astype(np.uint8), None, fx=upsample, fy=upsample, interpolation=cv2.INTER_NEAREST)
                ct, h = find_contours(mask.astype(np.uint8))
                pp = countours_to_polygon(ct, h, offset=xy0, scale=scale, upsample=upsample)
                if pp is not None:
                    regions_staging[name].append(pp)
        for name, pp in regions_staging.items():
            p_lbl = unary_union(pp)
            if p_lbl.area > 0:
                polygons[name] = p_lbl
    else:
        if isinstance(imgs, str): # input is a file path
            tile = common.imread(imgs, flag=cv2.IMREAD_UNCHANGED)
        elif isinstance(imgs, np.ndarray):
            tile = imgs
        else:
            raise TypeError
        xmin = offset[0] - 0.5
        xmax = offset[0] + tile.shape[1] - 0.5
        ymin = offset[1] - 0.5
        ymax = offset[1] + tile.shape[0] - 0.5
        xmin, ymin, xmax, ymax = scale_coordinates((xmin, ymin, xmax, ymax), scale)
        extent = shpgeo.box(xmin, ymin, xmax, ymax)
        for name, lbl in labels.items():
            if lbl is None:
                continue
            if len(tile.shape) > 2: # RGB
                mask = np.all(tile == np.array(lbl), axis=-1)
            else:
                mask = (tile == lbl)
            if not np.any(mask, axis=None):
                continue
            if upsample != 1:
                mask = (cv2.resize(255*mask.astype(np.uint8), None, fx=upsample, fy=upsample, interpolation=cv2.INTER_LINEAR)) > 127
            ct, h = find_contours(mask)
            p_lbl = countours_to_polygon(ct, h, offset=np.array(offset), scale=scale, upsample=upsample)
            if p_lbl is not None:
                polygons[name] = p_lbl
    return polygons, extent



def get_polygon_representative_point(poly):
    """
    Get representative point(s) for shapely (Multi)Polygon.
    """
    if hasattr(poly, 'geoms'):
        points = []
        for elem in poly.geoms:
            pts = get_polygon_representative_point(elem)
            points.extend(pts)
    elif isinstance(poly, shpgeo.Polygon):
        points = list(poly.representative_point().coords)
    else:
        points = []
    return points



def polygon_area_filter(poly, area_thresh=0):
    if area_thresh == 0:
        return poly
    if isinstance(poly, shpgeo.Polygon):
        if poly.is_empty or shpgeo.Polygon(poly.exterior).area < area_thresh:
            return shpgeo.Polygon()
        Bs = poly.boundary
        if hasattr(Bs,'geoms'):
            # may need to fill small holes
            holes_to_fill = []
            for linestr in Bs.geoms:
                pp = shpgeo.Polygon(linestr)
                if pp.area < area_thresh:
                    holes_to_fill.append(pp)
            if len(holes_to_fill) > 0:
                return poly.union(unary_union(holes_to_fill))
            else:
                return poly
        else:
            return poly
    elif hasattr(poly, 'geoms'):
        new_poly_list = []
        for pp in poly.geoms:
            pp_updated = polygon_area_filter(pp, area_thresh=area_thresh)
            if not pp_updated.is_empty:
                new_poly_list.append(pp_updated)
        if len(new_poly_list) > 0:
            return unary_union(new_poly_list)
        else:
            return shpgeo.Polygon()
    elif isinstance(poly, dict):
        new_dict = {}
        for key, pp in poly.items():
            pp_updated = polygon_area_filter(pp, area_thresh=area_thresh)
            if not pp_updated.is_empty:
                new_dict[key] = pp_updated
        return new_dict
    elif isinstance(poly, (tuple, list)):
        new_list = []
        for pp in poly:
            pp_updated = polygon_area_filter(pp, area_thresh=area_thresh)
            if not pp_updated.is_empty:
                new_list.append(pp_updated)
        if isinstance(poly, tuple):
            new_list = tuple(new_list)
        return new_list
    elif hasattr(poly, 'area') and poly.area == 0:
        return shpgeo.Polygon()
    else:
        raise TypeError


def smooth_zigzag(boundary, roi=None, scale=1.0, tol=0.5):
    """
    smooth the zigzag border of a polygon defined by bit-map images.
    boundary (shapely.geometry.LineString): boundary of the polygon
    """
    tol = tol * scale
    boundary = boundary.simplify(1.0e-3 * tol, preserve_topology=False)
    if hasattr(boundary, 'geoms'):
        boundaries = list(l for l in boundary.geoms)
    else:
        boundaries = [boundary]
    smoothened = []
    for lr in boundaries:
        if lr.length == 0:
            continue
        vertices = get_coordinates(lr)
        if vertices.shape[0] <= 2:
            smoothened.append(lr)
            continue
        if np.all(vertices[0] == vertices[-1]):
            is_closed = True
        else:
            is_closed = False
        vpts = [pt for pt in shpgeo.MultiPoint(vertices).geoms]
        mid_points = (vertices[:-1] + vertices[1:]) / 2
        if is_closed:
            ml = shpgeo.LinearRing(mid_points)
        else:
            ml = shpgeo.LineString(mid_points)
        dis = ml.distance(vpts)
        to_keep = dis >= tol
        if roi is not None:
            dis0 = roi.distance(vpts)
            to_keep = to_keep | (dis0 < tol)
        if not is_closed:
            to_keep[0] = True
            to_keep[-1] = True
        vindx = 2 * (np.nonzero(to_keep)[0])
        mindx = 2 * np.arange(mid_points.shape[0]) + 1
        vertices_new = np.concatenate((vertices[to_keep], mid_points), axis=0)
        sindx = np.argsort(np.concatenate((vindx, mindx)))
        vertices_new = vertices_new[sindx]
        if is_closed:
            lr_new = shpgeo.LinearRing(vertices_new)
        else:
            lr_new = shpgeo.LineString(vertices_new)
        lr_new = lr_new.simplify(1.0e-3 * tol, preserve_topology=False)
        smoothened.append(lr_new)
    if len(smoothened) == 1:
        smoothened = smoothened[0]
    else:
        smoothened = unary_union(smoothened)
    return smoothened


def clean_up_small_regions(regions, roi=None, area_thresh=4, buffer=1e-3):
    if area_thresh == 0:
        return regions, shpgeo.Polygon()
    if not isinstance(regions, dict):
        regions_dict = {k: rg for k, rg in enumerate(regions)}
    else:
        regions_dict = regions
    small_pieces = []
    if roi is None:
        roi = unary_union(list(regions_dict.values()))
    default_region = roi
    for lbl, pp in regions_dict.items():
        if pp.is_empty:
            continue
        pp_updated = polygon_area_filter(pp, area_thresh=area_thresh)
        pp_residue = pp.difference(pp_updated)
        default_region = default_region.difference(pp_updated)
        if not pp_residue.is_empty:
            regions_dict[lbl] = pp_updated
            if hasattr(pp_residue, 'geoms'):
                small_pieces.extend(list(pp_residue.geoms))
            else:
                small_pieces.append(pp_residue)
    if 'default' in regions_dict:
        regions_dict['default'] = regions_dict['default'].union(default_region)
        orgin_contain_default = True
    else:
        regions_dict['default'] = default_region
        orgin_contain_default = False
    if not regions_dict['default'].is_empty:
        pp_updated = polygon_area_filter(regions_dict['default'], area_thresh=area_thresh)
        pp_residue = regions_dict['default'].difference(pp_updated)
        if not pp_residue.is_empty:
            regions_dict['default'] = pp_updated
            if hasattr(pp_residue, 'geoms'):
                small_pieces.extend(list(pp_residue.geoms))
            else:
                small_pieces.append(pp_residue)
    for p0 in small_pieces:
        p0b = p0.buffer(buffer)
        assigned_label = None
        max_intersection = 0
        for lbl, pp in regions_dict.items():
            if pp.is_empty:
                continue
            if not p0b.intersects(pp):
                continue
            intersect_area = p0b.intersection(pp).area
            if intersect_area > max_intersection:
                assigned_label = lbl
                max_intersection = intersect_area
        if assigned_label is not None:
            regions_dict[assigned_label] = regions_dict[assigned_label].union(p0)
    if not orgin_contain_default:
        regions_dict.pop('default')
    if not isinstance(regions, dict):
        return_type = type(regions)
        out = return_type([s for s in regions_dict.values() if not s.is_empty])
    else:
        out = regions_dict
    if bool(small_pieces):
        modified_area = unary_union(small_pieces)
    else:
        modified_area = shpgeo.Polygon()
    return out, modified_area



def generate_equilat_grid_bbox(bbox, side_len, anchor_point=None, buffer=None):
    """
    generate equilateral triangle grid points that covers the bounding box
    """
    Xmin, Ymin, Xmax, Ymax = bbox
    center_point = [(Xmin+Xmax)/2, (Ymin+Ymax)/2]
    if buffer is None:
        dx = side_len
        dy = side_len * np.sin(np.deg2rad(60))
    else:
        dx = buffer
        dy = buffer
    half_Nx = int(np.ceil((Xmax - Xmin) / (2 * dx)) + 2)
    half_Ny = int(np.ceil((Ymax - Ymin) / (2 * dy)) + 2)
    Xmin_os = center_point[0] - half_Nx * dx
    Xmax_os = center_point[0] + half_Nx * dx
    Ymin_os = center_point[1] - half_Ny * dy
    Ymax_os = center_point[1] + half_Ny * dy
    x0 = np.linspace(Xmin_os, Xmax_os, 2*half_Nx+1, endpoint=True)
    y0 = np.linspace(Ymin_os, Ymax_os, 2*half_Ny+1, endpoint=True)
    if anchor_point is not None:
        x_nearest = np.argmin(np.abs(x0 - anchor_point[0]))
        x0 = x0 - x0[x_nearest] + anchor_point[0]
        y_nearest = np.argmin(np.abs(y0 - anchor_point[1]))
        y0 = y0 - y0[y_nearest] + anchor_point[1]
    else:
        y_nearest = half_Ny
    xv, yv = np.meshgrid(x0, y0)
    xv[((y_nearest+1)%2)::2, :] += dx/2
    indx = (xv >= (Xmin-dx)) & (xv <= (Xmax+dx)) & (yv >= (Ymin-dy)) & (yv <= (Ymax+dy))
    vertices = np.stack((xv[indx], yv[indx]), axis=-1)
    return vertices


def generate_equilat_grid_mask(mask, side_len, anchor_point=None, buffer=None):
    """
    generate equilateral triangle grid points that covers the shapely geometries
    """
    if buffer is None:
        buffer = side_len
    if isinstance(mask, (tuple, list, np.ndarray)):
        mask_np = np.array(mask)
        if mask_np.size == 4: #bbox
            xmin, ymin, xmax, ymax = mask_np.ravel()
            mask = shpgeo.box(xmin, ymin, xmax, ymax)
        else:
            mask = shpgeo.Polygon(mask_np.reshape(-1, 2))
    if hasattr(mask, 'geoms'):
        # if multiple geometries, filter out ones with 0 areas
        to_keep = []
        for g in mask.geoms:
            if g.area > 0:
                to_keep.append(g)
        mask = unary_union(to_keep)
    if anchor_point is None:
        rpts = mask.representative_point()
        anchor_point = (rpts.x, rpts.y)
    maskd = mask.buffer(1.001 * buffer) 
    v = generate_equilat_grid_bbox(maskd.bounds, side_len, anchor_point=anchor_point, buffer=buffer)
    pts = shpgeo.MultiPoint(v).intersection(maskd)
    if hasattr(pts, 'geoms'):
        return np.array([(p.x, p.y) for p in pts.geoms])
    else:
        return np.array((pts.x, pts.y))
    

def find_rotation_for_minimum_rectangle(poly, rounding=0):
    bbox = minimum_rotated_rectangle(poly)
    corner_xy = np.array(bbox.boundary.coords)
    corner_dxy = np.diff(corner_xy, axis=0)
    sides = np.sum(corner_dxy**2, axis=-1) ** 0.5
    if sides[0] > sides[1]:
        side_vec = corner_dxy[0]
    else:
        side_vec = corner_dxy[1]
    theta = np.arctan2(side_vec[1], side_vec[0])
    if rounding > 0:
        delta = rounding * np.pi / 180
        theta = np.round(theta / delta) * delta
    if np.abs(theta) > np.pi/2:
        theta = np.pi + theta
    return theta


class Geometry:
    """
    Class to represent a collection of 2d geometries that defines the shapes of
    subregions within a section. On each subregion we can assign its own
    mechanical properties

    Kwargs:
        roi(shapely Polygon or MultiPolygon): polygon the defines the shape of
            the outline of the geometry.
        regions(dict): dictionary contains polygons defining the shape of
            subregions with mechanical properties different from default.
        zorder(list): list of keys in the regions dict. If overlaps exist
            between regions, regions later in the list will trump early ones
    """
    def __init__(self, roi=None, regions=None, **kwargs):
        if regions is None:
            regions = {}
        self._roi = roi
        self._default_region = None
        self._regions = regions
        self._resolution = kwargs.get('resolution', data_resolution())
        self._zorder = kwargs.get('zorder', list(self._regions.keys()))
        self._committed = False
        self._epsilon = kwargs.get('epsilon', const.EPSILON0) # small value used for buffer


    @classmethod
    def from_image_mosaic(cls, image_loader, material_table=None, region_names=None, **kwargs):
        """
        Args:
            image_loader(feabas.dal.MosaicLoader): image loader to load mask
                images in mosaic form. Can also be a single-tile image
        kwargs:
            resolution(float): resolution of the geometries. Different scalings
                align at the top-left corner pixel as (0, 0).
            oor_label(int): label assigned to out-of-roi region.
            material_table(feabas.material.MaterialTable): table of materials
                that defines the region names and labels.
            region_names(OrderedDict): OrderedDict mapping region names to their
                corresponding labels, in addition to material_table.
            dilate(float): dilation radius to grow regions.
            scale(float): if image_loader is not a MosaicLoader, use this to
                define scaling factor.
        """
        resolution = kwargs.get('resolution', data_resolution())
        oor_label = kwargs.get('oor_label', None)
        roi_erosion = kwargs.get('roi_erosion', 0.5)
        dilate = kwargs.get('dilate', 0.1)
        scale = kwargs.get('scale', 1.0) # may change if imageloader has different resolution
        name2label = OrderedDict() # region name to label mapping
        if material_table is not None:
            name2label = material_table.name_to_label_mapping
        if region_names is not None:
            name2label.update(region_names)
        if isinstance(image_loader, dal.AbstractImageLoader):
            if 'scale' in kwargs and 'resolution' not in kwargs:
                resolution = image_loader.resolution / scale
            else:
                scale = image_loader.resolution / resolution
        roi_erosion = roi_erosion * scale
        dilate = dilate * scale
        if (oor_label is not None) and (oor_label not in name2label.values()):
            name2label.update({'out_of_roi_label': oor_label})
        if 'default' in name2label:
            name2label.pop('default')
        regions, roi = images_to_polygons(image_loader, name2label, scale=scale)
        if roi_erosion > 0:
            roi = roi.buffer(-roi_erosion, join_style=JOIN_STYLE)
        if oor_label is not None and 'out_of_roi_label' in regions:
            pp = regions.pop('out_of_roi_label')
            if roi_erosion > 0:
                pp = pp.buffer(roi_erosion, join_style=JOIN_STYLE)
            roi = roi.difference(pp)
        if dilate > 0:
            for lbl, pp in regions.items():
                regions[lbl] = pp.buffer(dilate, join_style=JOIN_STYLE)
        return cls(roi=roi, regions=regions, resolution=resolution,
            zorder=list(name2label.keys()), epsilon=const.EPSILON0*scale)


    @classmethod
    def from_h5(cls, h5name):
        kwargs = {}
        regions = {}
        with H5File(h5name, 'r') as f:
            if 'resolution' in f:
                kwargs['resolution'] = f['resolution'][()]
            if 'zorder' in f:
                kwargs['zorder'] = common.numpy_to_str_ascii(f['zorder'][()]).split('\n')
            if 'epsilon' in f:
                kwargs['epsilon'] = f['epsilon'][()]
            if 'roi' in f:
                roi = wkb.loads(bytes.fromhex(f['roi'][()].decode()))
            if 'regions' in f:
                for rname in f['regions']:
                    regions[rname] = wkb.loads(bytes.fromhex(f['regions/'+rname][()].decode()))
        return cls(roi=roi, regions=regions, **kwargs)


    def save_to_h5(self, h5name):
        with H5File(h5name, 'w') as f:
            _ = f.create_dataset('resolution', data=self._resolution)
            _ = f.create_dataset('epsilon', data=self._epsilon)
            if bool(self._zorder):
                zorder_encoded = common.str_to_numpy_ascii('\n'.join(self._zorder))
                _ = f.create_dataset('zorder', data=zorder_encoded)
            if hasattr(self._roi, 'wkb_hex'):
                _ = f.create_dataset('roi', data=self._roi.wkb_hex)
            for name, pp in self._regions.items():
                if hasattr(pp, 'wkb_hex'):
                    keyname = 'regions/{}'.format(name)
                    _ = f.create_dataset(keyname, data=pp.wkb_hex)


    def add_regions(self, regions, mode='u', pos=None):
        """
        add regions to geometry.
        Args:
            regions(dict): contains regions in shapely.Polygon format
        Kwargs:
            mode: 'u'-union; 'r'-replace.
            pos: position to insert in _zorder. None for append.
        """
        for lbl, pp in regions.items():
            if (mode=='r') or (lbl not in self._regions):
                self._regions[lbl] = pp
            else:
                self._regions[lbl] = self._regions[lbl].union(pp)
            if lbl not in self._zorder:
                if pos is None:
                    self._zorder.append(lbl)
                else:
                    self._zorder.insert(pos, lbl)
        self._committed = False


    def add_regions_from_image(self, image, material_table=None, region_names=None, **kwargs):
        resolution = kwargs.get('resolution', data_resolution())
        dilate = kwargs.get('dilate', 0.1)
        scale = kwargs.get('scale', 1.0)
        mode = kwargs.get('mode', 'u')
        pos = kwargs.get('pos', None)
        if isinstance(image, dal.MosaicLoader):
            scale = image.resolution / resolution
        dilate = dilate * scale
        name2label = OrderedDict() # region name to label mapping
        if material_table is not None:
            name2label = material_table.name_to_label_mapping
        if region_names is not None:
            name2label.update(region_names)
        regions, _ = images_to_polygons(image, name2label, scale=scale)
        if dilate > 0:
            for lbl, pp in regions.items():
                regions[lbl] = pp.buffer(dilate, join_style=JOIN_STYLE)
        epsilon1 = const.EPSILON0 * scale
        if epsilon1 < self._epsilon:
            self._epsilon = epsilon1
        self.add_regions(regions, mode=mode, pos=pos)


    def modify_roi(self, roi, mode='r'):
        """
        modify the roi geometry.
        Args:
            roi(shapely Polygon)
        Kwargs:
            mode: 'u'-union; 'r'-replace; 'i'-intersect
        """
        if (mode == 'r') or self._roi is None:
            self._roi = roi
        elif mode == 'i':
            self._roi = self._roi.intersection(roi)
        else:
            self._roi = self._roi.union(roi)
        self._committed = False


    def modify_roi_from_image(self, image, roi_label=0, **kwargs):
        resolution = kwargs.get('resolution', data_resolution())
        roi_erosion = kwargs.get('roi_erosion', 0)
        scale = kwargs.get('scale', 1.0)
        mode = kwargs.get('mode', 'r')
        if isinstance(image, dal.MosaicLoader):
            scale = image.resolution / resolution
        roi_erosion = roi_erosion * scale
        poly, extent = images_to_polygons(image, {'roi': roi_label}, scale=scale)
        if 'roi' in poly and poly['roi'].area > 0:
            roi = poly['roi']
        else:
            roi = extent
        if roi_erosion > 0:
            roi = roi.buffer(-roi_erosion, join_style=JOIN_STYLE)
        epsilon1 = const.EPSILON0 * scale
        if epsilon1 < self._epsilon:
            self._epsilon = epsilon1
        self.modify_roi(roi, mode=mode)


    def commit(self, **kwargs):
        """
        rectify the regions to make them mutually exclusive and within ROI.
        Kwargs:
            area_threshold(float): area (in the image resolution) threshold below
                which the regions should be discarded.
        """
        area_thresh = kwargs.get('area_thresh', 0)
        if self._roi is None:
            raise RuntimeError('ROI not defined')
        mask = self._roi
        covered_list = [mask]
        for lbl in reversed(self._zorder):
            if lbl not in self._regions:
                continue
            poly = (self._regions[lbl]).intersection(mask)
            poly_updated = polygon_area_filter(poly, area_thresh=area_thresh)
            if poly_updated.is_empty:
                self._regions.pop(lbl)
            else:
                self._regions[lbl] = poly_updated.buffer(0)
                covered_list.append(poly_updated)
                if lbl != 'default':
                    mask = mask.difference(poly_updated)
        filtered_roi = polygon_area_filter(mask, area_thresh=area_thresh)
        if not filtered_roi.is_empty:
            self._default_region = filtered_roi.buffer(0)
        covered = unary_union(covered_list)
        covered_boundary = covered.boundary
        if hasattr(covered_boundary, 'geoms'):
            # if boundary has multiple line strings, check for holes
            filled_cover = []
            for linestr in covered_boundary.geoms:
                area_sign = shpgeo.LinearRing(linestr).is_ccw
                if area_sign:
                    filled_cover.append(shpgeo.Polygon(linestr))
            holes = unary_union(filled_cover).difference(covered)
            if holes.area > 0:
                if 'exclude' in self._regions:
                    self._regions['exclude'] = (self._regions['exclude'].union(holes)).buffer(0)
                else:
                    self._regions['exclude'] = holes.buffer(0)
        self._committed = True


    def collect_boundaries(self, **kwargs):
        if not self._committed:
            self.commit(**kwargs)
        boundaries = []
        if self._default_region is not None:
            boundaries.append(self._default_region.boundary)
        for pp in self._regions.values():
            boundaries.append(pp.boundary)
        boundaries = unary_union(boundaries)
        if hasattr(boundaries, 'geoms'):
            boundaries = linemerge(boundaries)
        return boundaries


    def collect_region_markers(self, **kwargs):
        """get a arbitrary maker point for each connected region"""
        if not self._committed:
            self.commit(**kwargs)
        points = {}
        points['default'] = get_polygon_representative_point(self._default_region)
        for lbl, pp in self._regions.items():
            points[lbl] = get_polygon_representative_point(pp)
        return points


    def compare(self, other):
        region_lbls = set(self._regions) | set(other._regions)
        roi0 = self._roi
        roi1 = other._roi
        IOUs = {}
        for lbl in region_lbls:
            if lbl == 'default':
                continue
            if (lbl not in self._regions) or (lbl not in other._regions):
                IOUs[lbl] = 0
                continue
            pp0 = self._regions[lbl]
            pp1 = other._regions[lbl]
            if (pp0 is None) or (hasattr(pp0, 'is_empty') and pp0.is_empty):
                IOUs[lbl] = 0
                continue
            else:
                roi0 = roi0.difference(pp0)
            if (pp1 is None) or (hasattr(pp1, 'is_empty') and pp1.is_empty):
                IOUs[lbl] = 0
                continue
            else:
                roi1 = roi1.difference(pp1)
            iou = pp0.intersection(pp1).area / pp0.union(pp1).area
            IOUs[lbl] = iou
        IOUs['default'] = roi0.intersection(roi1).area / roi0.union(roi1).area
        return IOUs



    def simplify(self, region_tol=1.5, roi_tol=1.5, inplace=True, scale=1.0, method=const.SPATIAL_SIMPLIFY_GROUP, area_thresh=0):
        """
        simplify regions and roi so they have fewer line segments.
        Kwargs:
            region_tol(dict or scalar): maximum tolerated distance bwteen the
                points on the simplified regions and the unsimplified version.
                Could be a scalar that decines the universal behavior, or a dict
                to specify the tolerance for each region key.
            roi_tol(scalar): maximum tolerated distance for outer roi boundary.
            method(int): if simplify by regions or simplify by segments or
                grouped segments.
        """
        if self._committed:
            import warnings
            warnings.warn('Geometry alread commited. Simplification aborted', RuntimeWarning)
            return self
        if not isinstance(region_tol, dict):
            region_tols = defaultdict(lambda: region_tol)
        else:
            region_tols = region_tol.copy()
        region_tols.setdefault('default', np.inf)
        if method == const.SPATIAL_SIMPLIFY_SEGMENT:
            G = self.simplify_by_segments(region_tols, roi_tol=roi_tol,
                inplace=inplace, scale=scale, area_thresh=area_thresh)
        elif method == const.SPATIAL_SIMPLIFY_REGION:
            G = self.simplify_by_regions(region_tols, roi_tol=roi_tol,
                inplace=inplace, scale=scale, area_thresh=area_thresh)
        else:
            G = self.simplify_by_segment_groups(region_tols, roi_tol=roi_tol,
                inplace=inplace, scale=scale, area_thresh=area_thresh)
        return G


    def simplify_by_segments(self, region_tols, roi_tol=1.5, inplace=True, **kwargs):
        """
        simplify regions and roi by simplify segments first then polygonize them
        into regions.
        """
        scale = kwargs.get('scale', 1.0)
        area_thresh = kwargs.get('area_thresh', 0.0)
        if roi_tol > 0:
            roi = self._roi.simplify(roi_tol*scale, preserve_topology=True)
            if inplace:
                self._roi = roi
        else:
            roi = self._roi
        epsilon1 = const.EPSILON0 * scale
        if epsilon1 < self._epsilon:
            self._epsilon = epsilon1
        covered = None
        bu0 = [roi.boundary]
        polygons_cleaned = {}
        for lbl in reversed(self._zorder):
            if lbl not in self._regions:
                continue
            pp = self._regions[lbl].intersection(roi)
            if covered is None:
                bb = pp.boundary
                bu0.append(bb)
                polygons_cleaned[lbl] = pp
                covered = pp
            else:
                bb = pp.boundary.difference(covered.buffer(-self._epsilon))
                bu0.append(bb)
                polygons_cleaned[lbl] = pp.difference(covered)
                covered = covered.union(pp)
        bu0 = unary_union(bu0)
        polygons_formalized = list(polygonize(bu0))
        polygons_formalized, _ = clean_up_small_regions(polygons_formalized, area_thresh=area_thresh, buffer=self._epsilon)
        formalized_polygon_areas = [p.area for p in polygons_formalized]
        min_poly_area = np.min(formalized_polygon_areas)
        boundaries = OrderedDict()
        poly_assigned = np.zeros(len(polygons_formalized), dtype=bool)
        for lbl in reversed(self._zorder):
            if (lbl == 'default') or (lbl not in polygons_cleaned):
                continue
            poly = polygons_cleaned[lbl]
            if poly.area == 0:
                continue
            bndr = []
            area_left = poly.area
            for kf, pf in enumerate(polygons_formalized):
                if area_left < min_poly_area * 0.5:
                    break
                if poly_assigned[kf]:
                    continue
                if not pf.intersects(poly):
                    continue
                area_ints = pf.intersection(poly).area
                if  area_ints / formalized_polygon_areas[kf] > 0.99:
                    bndr.append(pf.boundary)
                    poly_assigned[kf] = True
                    area_left -= area_ints
            boundaries[lbl] = unary_union(bndr)
        if not np.all(poly_assigned):
            left_overs = unary_union([pf for pf, flg in zip(polygons_formalized, poly_assigned) if not flg])
            boundaries['default'] = left_overs.boundary
        b_merged = unary_union(boundaries.values())
        if hasattr(b_merged, 'geoms'):
            b_merged = linemerge(b_merged)
        if not hasattr(b_merged, 'geoms'):
            bag_of_segs = [b_merged]
        else:
            bag_of_segs = list(b_merged.geoms)
        region_names = sorted(list(self._regions.keys()), key=lambda s:region_tols[s]) + ['default']
        segs_in_regions = defaultdict(list)
        for seg_idx, seg in enumerate(bag_of_segs):
            for lbl, bndr in boundaries.items():
                if bndr.intersects(seg) and bndr.intersection(seg).length > 0:
                    segs_in_regions[lbl].append(seg_idx)
        simplify_order = []
        simplify_tol = []
        for lbl in region_names:
            if lbl not in segs_in_regions:
                continue
            tol = region_tols[lbl]
            slist = [s for s in segs_in_regions[lbl] if s not in simplify_order]
            simplify_order.extend(slist)
            simplify_tol.extend([tol]*len(slist))
        bag_of_segs = [smooth_zigzag(s) for s in bag_of_segs]
        for sidx, tol in zip(simplify_order, simplify_tol):
            if np.isinf(tol):
                continue
            seg_target = unary_union([s for k, s in enumerate(bag_of_segs) if (k==sidx)])
            segs_except_target = unary_union([s for k, s in enumerate(bag_of_segs) if (k!=sidx)])
            segs_all = unary_union(bag_of_segs)
            # fix segments other than the one to be simplified by duplicating them
            segs_combined = unary_union([segs_except_target, segs_all])
            segs_simplified = segs_combined.simplify(tol*scale, preserve_topology=True)
            seg_new = segs_simplified.difference(segs_except_target)
            tol_g = tol*scale
            while not seg_target.boundary.difference(seg_new.boundary).is_empty:
                tol_g /= 1.414
                segs_simplified = segs_combined.simplify(tol_g, preserve_topology=True)
                seg_new = segs_simplified.difference(segs_except_target)
            if hasattr(seg_new, 'geoms'):
                seg_new = linemerge(seg_new)
            if seg_new.length > 0:
                bag_of_segs[sidx] = seg_new
        regions_new = {} # reassemble boundaries
        for lbl in self._regions:
            sidx = segs_in_regions[lbl]
            boundaries_new = unary_union([bag_of_segs[s] for s in sidx])
            polys = list(polygonize(boundaries_new))
            cnt = np.zeros(len(polys), dtype=np.uint16)
            for p0 in polys:
                p0c = shpgeo.Polygon(p0.exterior)
                for k1, p1 in enumerate(polys):
                    if p0 is p1:
                        continue
                    if p0c.contains(p1):
                        cnt[k1] += 1
            pp_updated = unary_union([p for k, p in enumerate(polys) if (cnt[k]%2 == 0)])
            if inplace:
                self._regions[lbl] = pp_updated
            else:
                regions_new[lbl] = pp_updated
        roi = unary_union(list(polygonize(bag_of_segs)))
        if inplace:
            self._roi = roi
        if inplace:
            return self
        else:
            return Geometry(roi=roi, regions=regions_new,
                resolution=self._resolution, zorder=self._zorder)


    def simplify_by_segment_groups(self, region_tols, roi_tol=1.5, inplace=True, **kwargs):
        """
        simplify regions and roi by simplify segment groups first then polygonize
        them into regions. should be faster than simplify_by_segments, but may
        fail when very small features exist
        """
        scale = kwargs.get('scale', 1.0)
        area_thresh = kwargs.get('area_thresh', 0)
        if roi_tol > 0:
            roi = self._roi.simplify(roi_tol*scale, preserve_topology=True)
            if inplace:
                self._roi = roi
        else:
            roi = self._roi
        epsilon1 = const.EPSILON0 * scale
        if epsilon1 < self._epsilon:
            self._epsilon = epsilon1
        covered = None
        bu0 = [roi.boundary]
        polygons_cleaned = {}
        for lbl in reversed(self._zorder):
            if lbl not in self._regions:
                continue
            pp = self._regions[lbl].intersection(roi)
            if covered is None:
                bb = pp.boundary
                bu0.append(bb)
                polygons_cleaned[lbl] = pp
                covered = pp
            else:
                bb = pp.boundary.difference(covered.buffer(-self._epsilon))
                bu0.append(bb)
                polygons_cleaned[lbl] = pp.difference(covered)
                covered = covered.union(pp)
        bu0 = unary_union(bu0)
        polygons_formalized = list(polygonize(bu0))
        polygons_formalized, modified_area = clean_up_small_regions(polygons_formalized, area_thresh=area_thresh, buffer=self._epsilon)
        formalized_polygon_areas = [p.difference(modified_area).area for p in polygons_formalized]
        min_poly_area = np.min(formalized_polygon_areas)
        boundaries = OrderedDict()
        poly_assigned = np.zeros(len(polygons_formalized), dtype=bool)
        for lbl in reversed(self._zorder):
            if (lbl == 'default') or (lbl not in polygons_cleaned):
                continue
            poly = polygons_cleaned[lbl]
            if poly.area == 0:
                continue
            bndr = []
            area_left = poly.area
            for kf, pf in enumerate(polygons_formalized):
                if area_left < min_poly_area * 0.25:
                    break
                if poly_assigned[kf]:
                    continue
                if not pf.intersects(poly):
                    continue
                area_ints = pf.intersection(poly).area
                if  area_ints / formalized_polygon_areas[kf] > 0.99:
                    bndr.append(pf.boundary)
                    poly_assigned[kf] = True
                    area_left -= area_ints
            boundaries[lbl] = unary_union(bndr)
        if not np.all(poly_assigned):
            left_overs = unary_union([pf for pf, flg in zip(polygons_formalized, poly_assigned) if not flg])
            boundaries['default'] = left_overs.boundary
        b_merged = unary_union(list(boundaries.values()))
        if hasattr(b_merged, 'geoms'):
            b_merged = linemerge(b_merged)
        if not hasattr(b_merged, 'geoms'):
            bag_of_segs = [b_merged]
        else:
            bag_of_segs = list(b_merged.geoms)
        region_names = sorted(list(self._regions.keys()), key=lambda s:region_tols[s]) + ['default']
        region_names_lut = {rn:krn for krn, rn in enumerate(region_names)}
        labels_of_segs = defaultdict(list)
        for seg_idx, seg in enumerate(bag_of_segs):
            for lbl, bndr in boundaries.items():
                if bndr.intersects(seg) and bndr.intersection(seg).length > 0:
                    labels_of_segs[seg_idx].append(region_names_lut[lbl])
        seg_groups = defaultdict(list)
        for seg_idx, lbl_ids in labels_of_segs.items():
            seg_groups[tuple(sorted(lbl_ids))].append(bag_of_segs[seg_idx])
        seg_groups = {lbl_id: smooth_zigzag(linemerge(lines)) for lbl_id, lines in seg_groups.items()}
        group_indices = sorted(seg_groups.keys())
        group_tols = [region_tols[region_names[lbl[0]]] for lbl in group_indices]
        for gidx, tol in zip(group_indices, group_tols):
            if np.isinf(tol):
                continue
            seg_target = unary_union([s for k, s in seg_groups.items() if (k==gidx)])
            segs_except_target = unary_union([s for k, s in seg_groups.items() if (k!=gidx)])
            segs_all = unary_union(list(seg_groups.values()))
            # fix segments other than the one to be simplified by duplicating them
            segs_combined = unary_union([segs_except_target, segs_all])
            segs_simplified = segs_combined.simplify(tol*scale, preserve_topology=True)
            seg_new = segs_simplified.difference(segs_except_target)
            tol_g = tol*scale
            while not seg_target.boundary.difference(seg_new.boundary).is_empty:
                tol_g /= 1.414
                segs_simplified = segs_combined.simplify(tol_g, preserve_topology=True)
                seg_new = segs_simplified.difference(segs_except_target)
            if hasattr(seg_new, 'geoms'):
                seg_new = linemerge(seg_new)
            seg_groups[gidx] = seg_new
        regions_new = {} # reassemble boundaries
        for lbl in self._regions:
            lbl_id = region_names_lut[lbl]
            boundaries_new = unary_union([s for gidx, s in seg_groups.items() if lbl_id in gidx])
            polys = list(polygonize(boundaries_new))
            cnt = np.zeros(len(polys), dtype=np.uint16)
            for p0 in polys:
                p0c = shpgeo.Polygon(p0.exterior)
                for k1, p1 in enumerate(polys):
                    if p0 is p1:
                        continue
                    if p0c.contains(p1):
                        cnt[k1] += 1
            pp_updated = unary_union([p for k, p in enumerate(polys) if (cnt[k]%2 == 0)])
            if inplace:
                self._regions[lbl] = pp_updated
            else:
                regions_new[lbl] = pp_updated
        roi = unary_union(list(polygonize(list(seg_groups.values()))))
        if inplace:
            self._roi = roi
        if inplace:
            return self
        else:
            return Geometry(roi=roi, regions=regions_new,
                resolution=self._resolution, zorder=self._zorder)


    def simplify_by_regions(self, region_tols, roi_tol=1.5, inplace=True, **kwargs):
        """
        simplify regions and roi by directly simplify individual regions. might
        produce small fragments between two non-default regions but faster.
        """
        scale = kwargs.get('scale', 1.0)
        area_thresh = kwargs.get('area_thresh', 0)
        if roi_tol > 0:
            roi = self._roi.simplify(roi_tol*scale, preserve_topology=True)
            if inplace:
                self._roi = roi
        epsilon1 = const.EPSILON0 * scale
        if epsilon1 < self._epsilon:
            self._epsilon = epsilon1
        regions_new = {}
        for key, pp in self._regions.items():
            if (region_tols[key] > 0) and (pp is not None):
                pp_updated = pp.simplify(region_tols[key]*scale, preserve_topology=True)
                if inplace:
                    self._regions[key] = pp_updated
                else:
                    regions_new[key] = pp_updated
        regions_new, _ = clean_up_small_regions(regions_new, roi=roi, area_thresh=area_thresh, buffer=self._epsilon)
        if inplace:
            return self
        else:
            return Geometry(roi=roi, regions=regions_new,
                resolution=self._resolution, zorder=self._zorder)


    def PSLG(self, **kwargs):
        """
        generate a Planar Straight Line Graph representation of the geometry to
        feed to the triangulator library.
        Kwargs:
            region_tol, roi_tol: distance tolerances passed to
                self.simplify.
            method(int): whether to use simplify_by_segments or
                simplify_by_regions or simplify_by_segment_groups.
            area_thresh: area minimum threshold passed to self.commit.
            snap_decimal: decimal number to round the coordinates so that close
                points would snap together.
        Return:
            vertices(Nx2 np.ndarray): vertices of PSLG
            segments(Mx2 np.ndarray): list of endpoints' vertex id for each
                segment of PSLG
            markers(dict of lists): marker points for each region names.
        """
        region_tol = kwargs.get('region_tol', 0)
        roi_tol = kwargs.get('roi_tol', 0)
        method = kwargs.get('method', const.SPATIAL_SIMPLIFY_GROUP)
        area_thresh = kwargs.get('area_thresh', 0)
        snap_decimal = kwargs.get('snap_decimal', None)
        scale = kwargs.get('scale', 1.0)
        area_thresh = area_thresh * (scale**2)
        if isinstance(region_tol, dict) or (region_tol > 0) or (roi_tol > 0):
            self.simplify(region_tol=region_tol, roi_tol=roi_tol,
                inplace=True, scale=scale, method=method)
        self.commit(area_thresh=area_thresh)
        boundaries = self.collect_boundaries()
        markers = self.collect_region_markers()
        if hasattr(boundaries, 'geoms'):
            vertices_staging = []
            segments_staging = []
            crnt_len = 0
            for linestr in boundaries.geoms:
                xy = np.asarray(linestr.coords)
                Npt = xy.shape[0]
                vertices_staging.append(xy)
                seg = np.stack((np.arange(Npt-1), np.arange(1, Npt)), axis=-1) + crnt_len
                segments_staging.append(seg)
                crnt_len += Npt
            vertices = np.concatenate(vertices_staging, axis=0)
            segments = np.concatenate(segments_staging, axis=0)
        else:
            vertices = np.asarray(boundaries.coords)
            Npt = vertices.shape[0]
            segments = np.stack((np.arange(Npt-1), np.arange(1, Npt)), axis=-1)
        if snap_decimal is not None:
            vertices = np.round(vertices, decimals=snap_decimal)
        vertices, indx = np.unique(vertices, return_inverse=True, axis=0)
        segments = indx[segments]
        PSLG = {'vertices': vertices,
                'segments': segments,
                'markers': markers,
                'resolution': self._resolution,
                'epsilon': self._epsilon}
        return PSLG


    @property
    def region_default(self):
        mask = self._roi
        for lb, pp in self._regions.items():
            if lb != 'default':
                mask = mask.difference(pp)
        return mask


    @staticmethod
    def region_names_from_material_dict(material_dict):
        if not isinstance(material_dict, dict):
            mt = material.MaterialTable.from_pickleable(material_dict)
            material_dict = mt.label_table
        region_names = OrderedDict()
        for label, mat in material_dict.items():
            if isinstance(mat, material.Material):
                matval = mat._mask_label
                if matval is None:
                    raise RuntimeError('material label not defined in material table.')
            else:
                matval = mat
            if isinstance(matval, int):
                region_names[label] = matval
            elif isinstance(matval, (list, tuple, np.ndarray)):
                region_names[label] = np.asarray(matval)
            else:
                raise TypeError('invalid material label value type.')
        return region_names
