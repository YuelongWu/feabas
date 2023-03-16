from collections import defaultdict
import glob
import os
from concurrent.futures.process import ProcessPoolExecutor
from functools import partial
from multiprocessing import get_context

from feabas import mesh, spatial, material, dal
import feabas.constant as const


def generate_mesh_from_mask(maskname, outname, **kwargs):
    resolution_src = kwargs.get('resolution_src', const.DEFAULT_RESOLUTION)
    resolution_tgt = kwargs.get('resolution_tgt', resolution_src)
    material_table =kwargs.get('material_table', material.MaterialTable())
    mesh_size = kwargs.get('mesh_size', 800 * const.DEFAULT_RESOLUTION / resolution_src)
    region_tols = kwargs.get('region_tol', defaultdict(lambda: 2))
    roi_tol = kwargs.get('roi_tol', 2)
    oor_label = kwargs.get('oor_label', None) # out-of-region label
    area_thresh = kwargs.get('area_thresh', 25)
    if not isinstance(material_table, material.MaterialTable):
        if isinstance(material_table, dict):
            material_table = material.MaterialTable(table=material_table)
        elif isinstance(material_table, str):
            material_table = material.MaterialTable.from_json(material_table, stream=not material_table.endswith('.json'))
        else:
            raise TypeError
    if isinstance(maskname, dict) or isinstance(maskname, str):
        try:
            maskname = dal.get_loader_from_json(maskname)
        except ValueError:
            pass
    G = spatial.Geometry.from_image_mosaic(maskname, material_table=material_table, oor_label=oor_label, resolution=resolution_src)
    PSLG = G.PSLG(region_tol=region_tols,  roi_tol=roi_tol, area_thresh=area_thresh)
    M = mesh.Mesh.from_PSLG(**PSLG, material_table=material_table, mesh_size=mesh_size, min_mesh_angle=20)
    M.change_resolution(resolution_tgt)
    mshname = os.path.splitext(os.path.basename(maskname))[0]
    M.save_to_h5(outname, save_material=True, override_dict={'name': mshname})


def generate_mesh_from_mask_main(maskdir, outdir, **kwargs):
    num_workers = kwargs.pop('num_workers', 1)
    input_ext = kwargs.pop('input_ext', '.png')
    masklist = sorted(glob.glob(os.path.join(maskdir, '*'+input_ext)))
    if len(masklist) == 0:
        return
    
    if num_workers > 1:
        jobs = []
        target_func = partial(generate_mesh_from_mask, **kwargs)
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=get_context('spawn')) as executor:
            for maskname in masklist:
                outname = os.path.join(outdir, os.path.basename(maskname).replace(input_ext, '.h5'))
                if os.path.isfile(outname):
                    continue
                jobs.append(executor.submit(target_func, maskname, outname))
            for job in jobs:
                job.result()
    else:
        for maskname in masklist:
            outname = os.path.join(outdir, os.path.basename(maskname).replace(input_ext, '.h5'))
            if os.path.isfile(outname):
                continue
            generate_mesh_from_mask(maskname, outname, **kwargs)
    print('finished.')


if __name__ == '__main__':
    maskdir = 'F:/Fish2/test_alignment/64nm/masks_match'
    outdir = 'F:/Fish2/test_alignment/64nm/meshes_for_matching'
    material_table = 'configs/example_material_table.json'
    region_tols = {'hole': 5, 'tape': 2.5}
    resolution_src = 64
    resolution_tgt = 16
    generate_mesh_from_mask_main(maskdir, outdir, material_table=material_table,
        resolution_src=resolution_src, resolution_tgt=resolution_tgt, region_tol=region_tols)