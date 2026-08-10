"""
Visualization tools for debugging.
"""
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib.tri
import numpy as np
import shapely.geometry as shpgeo
import feabas.constant as const
from feabas.common import bbox_intersections
from feabas.dal import StreamLoader
from feabas.mesh import Mesh
from feabas.renderer import MeshRenderer
from feabas.spatial import fit_affine

def rgb2hex(r,g,b):
    r = min(max(r, 0), 255)
    g = min(max(g, 0), 255)
    b = min(max(b, 0), 255)
    return "#{:02x}{:02x}{:02x}".format(r,g,b)

def hex2rgb(hexcode):
    return tuple(int(hexcode[i:i+2], 16) for i in (1, 3, 5))


def random_color():
    R, G = np.random.randint(256, size=2)
    B = 255 * 2 - R - G
    return rgb2hex(R, G, B)


def dynamic_typing_decorator(func):
    def wrapped(geo_obj, **kwargs):
        if isinstance(geo_obj, (tuple, list)):
            for g in geo_obj:
                wrapped(g, **kwargs)
        elif isinstance(geo_obj, dict):
            for g in geo_obj.values():
                wrapped(g, **kwargs)
        elif hasattr(geo_obj, 'geoms'):
            uniform_color = kwargs.get('uniform_color', True)
            kwargs_new = kwargs.copy()
            if uniform_color and (kwargs_new.get('color', None) is None):
                kwargs_new.update({'color': random_color()})
            for g in geo_obj.geoms:
                wrapped(g, **kwargs_new)
        elif geo_obj is None:
            pass
        else:
            func(geo_obj, **kwargs)
    return wrapped


def plot_mesh(M, show_mat=False, show_conn=False, show_group=False, gear=const.MESH_GEAR_MOVING, show=False, colors=('b', 'k')):
    """
    Visualize the triangulation of feabas.mesh.Mesh object.
    Args:
        M (feabas.mesh.Mesh): mesh object.
    Kwargs:
        show_mat (bool): display triangles of different material types with
            different color.
        show_conn (bool): display triangles belong to different connected
            components with different color.
        show_group (bool): show the partitions of the mesh with no
            self-intersection. Mostly for debugging.
        gear (int): set of vertices to show.
            0 for fixed vertices, 1 for moving vertices.
        show (bool):  whether to call plt.show() after leaving the function.
        colors (tuple): interior color followed by boundary color.
    """
    if isinstance(M, (list, tuple)):
        for m in M:
            plot_mesh(m, show_mat=show_mat, show_conn=show_conn, show_group=show_group, gear=gear, show=False)
    else:
        if show_mat or show_conn or show_group:
            if show_mat:
                mat_ids = M._material_ids
            elif show_conn:
                _, mat_ids = M.connected_triangles()
            elif show_group:
                mat_ids = M.nonoverlap_triangle_groups(gear=gear)
            for mid in np.unique(mat_ids):
                indx = mat_ids == mid
                R, G = np.random.randint(256, size=2)
                B = 255 * 2 - R - G
                color = rgb2hex(R, G, B)
                T = matplotlib.tri.Triangulation(M.vertices_w_offset(gear=gear)[:,0],
                    M.vertices_w_offset(gear=gear)[:,1], M.triangles[indx])
                plt.triplot(T, color=color, alpha=0.5, linewidth=0.5)
        else:
            T = matplotlib.tri.Triangulation(M.vertices_w_offset(gear=gear)[:,0],
                    M.vertices_w_offset(gear=gear)[:,1], M.triangles)
            plt.triplot(T, color=colors[0], alpha=0.5, linewidth=0.5)
        segs = M.vertices_w_offset(gear=gear)[M.segments()]
        xx = segs[:,:,0]
        yy = segs[:,:,1]
        plt.plot(xx.T, yy.T, colors[-1], alpha=1, linewidth=1)
    if show:
        plt.show()


def plot_montage(M, bbox_only=False):
    """
    Visualize the triangulation of feabas.stitcher.Stitcher object after
    optimization step.
    Args:
        M (feabas.stitcher.Stitcher): montage object. Can be initialized from
        Stitcher.from_h5(H5_FILES_AFTER_OPTIMIZATION)
    Kwargs:
        bbox_only (bool): Set to True to plot each tile as a rectangle (faster).
            Otherwise, plot the deformed outline of each tile.
    """
    if M._connected_subsystem is not None:
        lbls = M._connected_subsystem
    else:
        lbls = np.zeros(M.num_tiles, dtype=np.int8)
    for lbl in np.unique(lbls):
        mindx = np.nonzero(lbls == lbl)[0]
        color = random_color()
        outlines = []
        for idx in mindx:
            m = M.meshes[idx]
            if bbox_only:
                tile = shpgeo.box(*m.bbox(gear=const.MESH_GEAR_MOVING, offsetting=True))
            else:
                tile = m.shapely_regions(gear=const.MESH_GEAR_MOVING)
            outlines.append(tile)
        plot_geometries(outlines, color=color)


def plot_link(link, gear=const.MESH_GEAR_MOVING, minimum_residue=0, num_matches=None, show=False):
    """
    visualize feabas.optimizer.Link object
    """
    m0, m1 = link.meshes
    xy0 = link.xy0(gear=gear, use_mask=True)
    xy1 = link.xy1(gear=gear, use_mask=True)
    d = np.sum((xy0 - xy1)**2, axis=-1)**0.5
    idx_s = np.argsort(d)[::-1]
    xy0, xy1, d = xy0[idx_s], xy1[idx_s], d[idx_s]
    num_mtch0 = d.size
    if minimum_residue > 0:
        idx = d > minimum_residue
        xy0, xy1 = xy0[idx], xy1[idx]
    if num_matches is not None:
        if num_matches < 1:
            num_matches = max(1, int(num_mtch0 * num_matches))
        xy0, xy1 = xy0[:num_matches], xy1[:num_matches]
    plot_mesh(m0, gear=gear, colors='r')
    plot_mesh(m1, gear=gear, colors='b')
    plt.plot([xy0[:,0],xy1[:,0]], [xy0[:,1],xy1[:,1]], 'k:')
    plt.plot(xy0[:,0], xy0[:,1], 'm*')
    plt.plot(xy1[:,0], xy1[:,1], 'c*')
    if show:
        plt.show()


@dynamic_typing_decorator
def plot_points(pts, **kwargs):
    color = kwargs.get('color', '#ff0000')
    alpha = kwargs.get('alpha', 1)
    xy = np.asarray(pts.coords)
    plt.plot(xy[..., 0], xy[..., 1], '*', color=color, alpha=alpha)


@dynamic_typing_decorator
def plot_lines(lines, **kwargs):
    color = kwargs.get('color', None)
    alpha = kwargs.get('alpha', 1)
    if color is None:
        R, G = np.random.randint(256, size=2)
        B = 255 * 2 - R - G
        color = rgb2hex(R, G, B)
    coords = np.asarray(lines.coords)
    plt.plot(coords[...,0], coords[...,1], '-', color=color, alpha=alpha)


@dynamic_typing_decorator
def plot_polygons(polygons, **kwargs):
    color = kwargs.get('color', None)
    alpha = kwargs.get('alpha', 0.5)
    if color is None:
        R, G = np.random.randint(256, size=2)
        B = 255 * 2 - R - G
        facecolor = rgb2hex(R, G, B)
        edgecolor = rgb2hex(R//2, G//2, B//2)
    else:
        R, G, B = hex2rgb(color)
        facecolor = rgb2hex(R, G, B)
        edgecolor = rgb2hex(R//2, G//2, B//2)
    ax = plt.gca()
    patch = PolygonPatch(polygons, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, zorder=1)
    ax.add_patch(patch)
    plot_lines(polygons.boundary, alpha=0)


@dynamic_typing_decorator
def plot_geometries(geo_obj, **kwargs):
    """
    visualize shapely.Geometry objects.
    """
    if hasattr(geo_obj, 'is_empty') and geo_obj.is_empty:
        pass
    elif isinstance(geo_obj, shpgeo.Polygon):
        plot_polygons(geo_obj, **kwargs)
    elif isinstance(geo_obj, (shpgeo.LinearRing, shpgeo.LineString)):
        plot_lines(geo_obj, **kwargs)
    elif isinstance(geo_obj, shpgeo.Point):
        plot_points(geo_obj, **kwargs)
    elif geo_obj is None:
        pass
    else:
        raise TypeError


def show_image_pairs(img0=None, img1=None, mesh0=None, mesh1=None, xy0=None, xy1=None, **kwargs):
    bbox = kwargs.get('bbox', None)
    gear = kwargs.get('gear', const.MESH_GEAR_MOVING)
    affine_approx_tol = kwargs.get('affine_approx_tol', 0.1)
    if img0 is None:
        show_image = False
    else:
        show_image = kwargs.get('show_image', True)
    if xy0 is None:
        show_match = False
    else:
        show_match = kwargs.get('show_match', True)
    if mesh0 is None:
        show_mesh = False
        if bbox is None:
            pass
        if show_image:
            if isinstance(img0, np.ndarray):
                img0 = StreamLoader(img0)
            if isinstance(img1, np.ndarray):
                img1 = StreamLoader(img1)
            if xy0 is not None:
                if bbox is not None:
                    bbox_t, _ = bbox_intersections(img0.bounds, bbox)
                else:
                    bbox_t = img0.bounds
                M1 = Mesh.from_bbox(img1.bounds, cartesian=True, mesh_size=np.max(img1.bounds), resolution=img1.resolution)
                A, _ = fit_affine(xy0, xy1, return_rigid=True)
                M1.apply_affine(A, gear=const.MESH_GEAR_MOVING)
                R1 = MeshRenderer.from_mesh(M1, image_loader=img1, affine_approx_tol=affine_approx_tol)
                img0t = img0.crop(bbox_t)
                img1t = R1.crop(bbox_t)
                xy0t = xy0
                xy1t = xy1 @ A[:2,:2] + A[-1,:2]
            else:
                bbox_t, _ = bbox_intersections(img0.bounds, img1.bounds)
                if bbox is not None:
                    bbox_t, _ = bbox_intersections(bbox, bbox_t)
                img0t = img0.crop(bbox_t)
                img1t = img1.crop(bbox_t)
    else:
        show_mesh = kwargs.get('show_mesh', True)
        if bbox is not None:
            mesh0 = mesh0.submeshes_from_bboxes([bbox], gear=gear)[0]
            mesh1 = mesh1.submeshes_from_bboxes([bbox], gear=gear)[0]
        else:
            bbox, _ =  bbox_intersections(mesh0.bbox(gear=gear), mesh1.bbox(gear=gear))
        if show_image:
            if isinstance(img0, np.ndarray):
                img0 = StreamLoader(img0)
            if isinstance(img1, np.ndarray):
                img1 = StreamLoader(img1)
            R0 = MeshRenderer.from_mesh(mesh0, image_loader=img0, affine_approx_tol=affine_approx_tol)
            R1 = MeshRenderer.from_mesh(mesh1, image_loader=img1, affine_approx_tol=affine_approx_tol)
            img0t = R0.crop(bbox)
            img1t = R1.crop(bbox)
            mesh0.apply_translation((-bbox[0], -bbox[1]), gear=gear)
            mesh1.apply_translation((-bbox[0], -bbox[1]), gear=gear)
        if show_match:
            tid0, B0 = mesh0.cart2bary(xy0, const.MESH_GEAR_INITIAL, tid=None, extrapolate=False)
            tid1, B1 = mesh1.cart2bary(xy1, const.MESH_GEAR_INITIAL, tid=None, extrapolate=False)
            idxt = (tid0 >= 0) & (tid1 >= 0)
            if not np.any(idxt):
                show_match = False
            else:
                xy0t = mesh0.bary2cart(tid0[idxt], B0[idxt], gear, offsetting=True)
                xy1t = mesh1.bary2cart(tid1[idxt], B1[idxt], gear, offsetting=True)
    if show_image:
        img_ov = np.stack((img0t, img1t, img0t), axis=-1)
        plt.imshow(img_ov)
    if show_mesh:
        plot_mesh(mesh0, gear=const.MESH_GEAR_MOVING, colors='r')
        plot_mesh(mesh1, gear=const.MESH_GEAR_MOVING, colors='b')
    if show_match:
        plt.plot((xy0t[:,0], xy1t[:,0]), (xy0t[:,1], xy1t[:,1]), 'k:')
        plt.plot(xy0t[:,0], xy0t[:,1], 'm.')
        plt.plot(xy1t[:,0], xy1t[:,1], 'g.')


# from descartes:
class Polygon(object):
    # Adapt Shapely or GeoJSON/geo_interface polygons to a common interface
    def __init__(self, context):
        if hasattr(context, 'interiors'):
            self.context = context
        else:
            self.context = getattr(context, '__geo_interface__', context)
    @property
    def geom_type(self):
        return (getattr(self.context, 'geom_type', None)
                or self.context['type'])
    @property
    def exterior(self):
        return (getattr(self.context, 'exterior', None)
                or self.context['coordinates'][0])
    @property
    def interiors(self):
        value = getattr(self.context, 'interiors', None)
        if value is None:
            value = self.context['coordinates'][1:]
        return value


def PolygonPath(polygon):
    """Constructs a compound matplotlib path from a Shapely or GeoJSON-like
    geometric object"""
    this = Polygon(polygon)
    assert this.geom_type == 'Polygon'
    def coding(ob):
        # The codes will be all "LINETO" commands, except for "MOVETO"s at the
        # beginning of each subpath
        n = len(getattr(ob, 'coords', None) or ob)
        vals = np.ones(n, dtype=Path.code_type) * Path.LINETO
        vals[0] = Path.MOVETO
        return vals
    vertices = np.concatenate(
                    [np.asarray(this.exterior.coords)[:, :2]]
                    + [np.asarray(r.coords)[:, :2] for r in this.interiors])
    codes = np.concatenate(
                [coding(this.exterior)]
                + [coding(r) for r in this.interiors])
    return Path(vertices, codes)


def PolygonPatch(polygon, **kwargs):
    """Constructs a matplotlib patch from a geometric object

    The `polygon` may be a Shapely or GeoJSON-like object with or without holes.
    The `kwargs` are those supported by the matplotlib.patches.Polygon class
    constructor. Returns an instance of matplotlib.patches.PathPatch.
    Example (using Shapely Point and a matplotlib axes):
      >>> b = Point(0, 0).buffer(1.0)
      >>> patch = PolygonPatch(b, fc='blue', ec='blue', alpha=0.5)
      >>> axis.add_patch(patch)
    """
    return PathPatch(PolygonPath(polygon), **kwargs)