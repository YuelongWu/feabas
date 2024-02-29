import numpy as np
import os
import glob
import shapely

from feabas.mesh import Mesh
from feabas import config, constant

"""
apply rigid transforms to the meshes in {WORK_DIR}/align/tform/ folder so that
the minimum rotated rectangle of the union of the mesh regions rests its long
side along the x axis and one corner sits at (0,0)
"""


TILT_ANGLE = 0
OFFSET = (0, 0)

if __name__ == '__main__':
    root_dir =  config.get_work_dir()
    tlist = sorted(glob.glob(os.path.join(root_dir, 'align', 'tform', '*.h5')))

    regions = None
    print('finding transformations')
    for tname in tlist:
        M = Mesh.from_h5(tname)
        M.change_resolution(config.montage_resolution())
        R = M.shapely_regions(gear=constant.MESH_GEAR_MOVING, offsetting=True)
        R = shapely.convex_hull(R)
        if regions is None:
            regions = R
        else:
            regions = regions.union(R)
    bbox = shapely.minimum_rotated_rectangle(regions)
    corner_xy = np.array(bbox.boundary.coords)
    corner_dxy = np.diff(corner_xy, axis=0)
    sides = np.sum(corner_dxy**2, axis=-1) ** 0.5
    if sides[0] > sides[1]:
        side_vec = corner_dxy[0]
    else:
        side_vec = corner_dxy[1]
    theta = np.arctan2(side_vec[1], side_vec[0])
    if np.abs(np.pi - theta) < np.abs(theta):
        theta = np.pi - theta
    theta = theta - TILT_ANGLE * np.pi / 180
    Rt = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    R = np.eye(3)
    R[:2,:2] = Rt
    corner_txy = corner_xy @ Rt
    txy = -np.min(corner_txy, axis=0) + np.array(OFFSET)
    print('applying transforms')
    for tname in tlist:
        M = Mesh.from_h5(tname)
        M.change_resolution(config.montage_resolution())
        M.apply_affine(R, gear=constant.MESH_GEAR_FIXED)
        M.apply_translation(txy, gear=constant.MESH_GEAR_FIXED)
        M.apply_affine(R, gear=constant.MESH_GEAR_MOVING)
        M.apply_translation(txy, gear=constant.MESH_GEAR_MOVING)
        M.save_to_h5(tname, vertex_flags=constant.MESH_GEARS, save_material=True)
    print('finished')