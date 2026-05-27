import argparse
import numpy as np
from feabas.mesh import Mesh
from feabas.optimizer import relax_mesh
from feabas import config, constant, storage
from feabas.concurrent import submit_to_workers
"""
split the wrinkle artifact after mesh relaxation
"""

DEFORM_THRESHOLD = 1.2 # wrinkle element with larger than this expansion ratio will be split.


def split_wrinkle_for_one_section(tname):
    M = Mesh.from_h5(tname)
    mtb = M.named_material_table
    if 'wrinkle' not in mtb:
        return
    mid = mtb['wrinkle'].uid
    tid = M.material_ids == mid
    if not np.any(tid):
        return
    if 'soft' in mtb:
        mid_new = mtb['soft'].uid
    else:
        mid_new = mtb['default'].uid
    ss = np.max(1 / M.triangle_tform_svd(), axis=-1)
    cid = tid & (ss > DEFORM_THRESHOLD)
    if not np.any(cid):
        return
    M.incise_region(cid)
    tid = M._material_ids==mid
    M._material_ids[tid] = mid_new
    M.anneal(gear=(constant.MESH_GEAR_INITIAL, constant.MESH_GEAR_STAGING), mode=constant.ANNEAL_COPY_EXACT)
    M.anneal(gear=(constant.MESH_GEAR_FIXED, constant.MESH_GEAR_STAGING), mode=constant.ANNEAL_CONNECTED_AFFINE)
    relax_mesh(M, free_triangles=tid, gear=(constant.MESH_GEAR_STAGING, constant.MESH_GEAR_FIXED), tolerated_perturbation=0.1)
    M._vertices[constant.MESH_GEAR_STAGING] = None
    M._offsets[constant.MESH_GEAR_STAGING] = np.zeros((1,2))
    M.save_to_h5(tname, vertex_flags=constant.MESH_GEARS, save_material=True)

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="cut wrinkles open after mesh relaxation (for aesthetics in rendered image).")
    parser.add_argument("--worker",  metavar="worker", type=int, default=1)
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args()
    root_dir =  config.get_work_dir()
    tlist = sorted(storage.list_folder_content(storage.join_paths(root_dir, 'align', 'tform', '*.h5')))
    for _ in submit_to_workers(split_wrinkle_for_one_section, [[s] for s in tlist], num_workers=args.worker):
        pass
    print('finished')
