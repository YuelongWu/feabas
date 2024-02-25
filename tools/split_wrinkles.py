import numpy as np
import os
import glob
from feabas.mesh import Mesh
from feabas.optimizer import relax_mesh
from feabas import config, constant
"""
split the wrinkle artifact after mesh relaxation
"""

DEFORM_THRESHOLD = 1.6 # wrinkle element with larger than this expansion ratio will be split.


if __name__ == '__main__':
    root_dir =  config.get_work_dir()
    tlist = sorted(glob.glob(os.path.join(root_dir, 'align', 'tform', '*.h5')))
    for tname in tlist:
        M = Mesh.from_h5(tname)
        mtb = M._material_table.named_table
        if 'wrinkle' not in mtb:
            continue
        mid = mtb['wrinkle'].uid
        tid = M.material_ids == mid
        if not np.any(tid):
            continue
        if 'soft' in mtb:
            mid_new = mtb['soft'].uid
        else:
            mid_new = mtb['default'].uid
        ss = np.max(1 / M.triangle_tform_svd(), axis=-1)
        cid = tid & (ss > DEFORM_THRESHOLD)
        if not np.any(cid):
            continue
        M.incise_region(cid)
        tid = M._material_ids==mid
        M._material_ids[tid] = mid_new
        relax_mesh(M, free_triangles=tid)
        M.save_to_h5(tname, vertex_flags=constant.MESH_GEARS, save_material=True)
    print('finished')
