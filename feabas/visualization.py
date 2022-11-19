"""
Visualization tools for debugging.
"""
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib.tri
import numpy as np
import shapely.geometry as shpgeo


def rgb2hex(r,g,b):
    r = min(max(r, 0), 255)
    g = min(max(g, 0), 255)
    b = min(max(b, 0), 255)
    return "#{:02x}{:02x}{:02x}".format(r,g,b)

def hex2rgb(hexcode):
    return tuple(int(hexcode[i:i+2], 16) for i in (1, 3, 5))


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
                R, G = np.random.randint(256, size=2)
                B = 255 * 2 - R - G
                color = rgb2hex(R, G, B)
                kwargs_new.update({'color': color})
            for g in geo_obj.geoms:
                wrapped(g, **kwargs_new)
        elif geo_obj is None:
            pass
        else:
            func(geo_obj, **kwargs)
    return wrapped


def visualize_mesh(M, show_mat=False, show_conn=False, gear='m', show=False):
    M = M[gear]
    if show_mat:
        mat_ids = M._material_ids
        for mid in np.unique(mat_ids):
            indx = mat_ids == mid
            R, G = np.random.randint(256, size=2)
            B = 255 * 2 - R - G
            color = rgb2hex(R, G, B)
            T = matplotlib.tri.Triangulation(M.vertices_w_offset[:,0],
                M.vertices_w_offset[:,1], M.triangles[indx])
            plt.triplot(T, color=color, alpha=0.5, linewidth=0.5)
    elif show_conn:
        _, t_conn = M.connected_triangles()
        for lbl in np.unique(t_conn):
            indx = t_conn == lbl
            R, G = np.random.randint(256, size=2)
            B = 255 * 2 - R - G
            color = rgb2hex(R, G, B)
            T = matplotlib.tri.Triangulation(M.vertices_w_offset[:,0],
                M.vertices_w_offset[:,1], M.triangles[indx])
            plt.triplot(T, color=color, alpha=0.5, linewidth=0.5)
    else:
        T = matplotlib.tri.Triangulation(M.vertices_w_offset[:,0],
                M.vertices_w_offset[:,1], M.triangles)
        plt.triplot(T, color='b', alpha=0.5, linewidth=0.5)
    segs = M.vertices[M.segments()]
    xx = segs[:,:,0]
    yy = segs[:,:,1]
    plt.plot(xx.T, yy.T, 'k', alpha=1, linewidth=1)
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