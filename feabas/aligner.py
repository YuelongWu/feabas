from collections import namedtuple, defaultdict, OrderedDict
import h5py
import numpy as np
import glob
import os
import yaml

from feabas.mesh import Mesh
from feabas import dal
from feabas.spatial import scale_coordinates
from feabas.matcher import section_matcher
import feabas.constant as const
from feabas.common import str_to_numpy_ascii


Match = namedtuple('Match', ('xy0', 'xy1', 'weight'))


def read_matches_from_h5(match_name, target_resolution=None):
    with h5py.File(match_name, 'r') as f:
        xy0 = f['xy0'][()]
        xy1 = f['xy1'][()]
        weight = f['weight'][()].ravel()
        resolution = f['resolution'][()]
        if isinstance(resolution, np.ndarray):
            resolution = resolution.item()
    if target_resolution is not None:
        scale = resolution / target_resolution
        xy0 = scale_coordinates(xy0, scale)
        xy1 = scale_coordinates(xy1, scale)
    return Match(xy0, xy1, weight)


def match_section_from_initial_matches(match_name, meshes, loaders, out_dir, conf=None):
    """
    given the coarse matches saved in H5 format, caculate the fine matches.
    Args:
        match_name(str): the full path to the H5 initial match file. In the file
            there should be datasets: 'xy0', 'xy1', 'weight' and 'resolution'.
        meshes: the meshes of the two section. could be the parent directory
            that contains the H5 Mesh files, or a list of two H5 Mesh file paths,
            or a list of Mesh objects.
        loaders: the image loaders of the two section. could be the parent
            directory that contains the JSON/TXT loader files, or a list of two
            loader file paths, or a list of Loader objects.
        out_dir(str): output directory to save the results.
        conf: the alignment configurations. Could be the path to a YAML config
            file or a dictionary containing the settings.
    """
    outname = os.path.join(out_dir, os.path.basename(match_name))
    if os.path.isfile(outname):
        return None
    if isinstance(conf, str) and conf.endswith('.yaml'):
        with open(conf, 'r') as f:
            conf = yaml.safe_load(f)
    if 'matching' in conf:
        conf = conf['matching']
    elif conf is None:
        conf = {}
    elif not isinstance(conf, dict):
        raise TypeError('configuration type not supported.')
    match_name_delimiter = conf.get('match_name_delimiter', '__to__')
    resolution = conf.get('working_resolution', const.DEFAULT_RESOLUTION)
    loader_config = conf.get('loader_config', {})
    matcher_config = conf.get('matcher_config', {})
    secnames = os.path.splitext(os.path.basename(match_name))[0].split(match_name_delimiter)
    if 'cache_size' in loader_config and loader_config['cache_size'] is not None:
        loader_config = loader_config.copy()
        loader_config['cache_size'] = loader_config['cache_size'] // (2 * matcher_config.get('num_workers',1))
    if isinstance(meshes, str):
        meshes = (os.path.join(meshes, secnames[0]+'.h5'), os.path.join(meshes, secnames[1]+'.h5'))
    if isinstance(meshes, (tuple, list)):
        mesh0, mesh1 = meshes
        if isinstance(mesh0, str):
            mesh0 = Mesh.from_h5(mesh0)
        elif isinstance(mesh0, dict):
            mesh0 = Mesh(**mesh0)
        if isinstance(mesh1, str):
            mesh1 = Mesh.from_h5(mesh1)
        elif isinstance(mesh1, dict):
            mesh1 = Mesh(**mesh1)
        mesh0.uid = 0.0
        mesh1.uid = 1.0
    else:
        raise TypeError('mesh input type not supported.')
    if isinstance(loaders, str):
        if os.path.isfile(os.path.join(loaders, secnames[0]+'.json')):
            loader0 = os.path.join(loaders, secnames[0]+'.json')
        elif os.path.isfile(os.path.join(loaders, secnames[0]+'.txt')):
            loader0 = os.path.join(loaders, secnames[0]+'.txt')
        else:
            raise RuntimeError(f'cannot find loaders for {secnames[0]}')
        if os.path.isfile(os.path.join(loaders, secnames[1]+'.json')):
            loader1 = os.path.join(loaders, secnames[1]+'.json')
        elif os.path.isfile(os.path.join(loaders, secnames[1]+'.txt')):
            loader1 = os.path.join(loaders, secnames[1]+'.txt')
        else:
            raise RuntimeError(f'cannot find loaders for {secnames[1]}')
        loaders = (loader0, loader1)
    if isinstance(loaders, (tuple, list)):
        loader0, loader1 = loaders
        if not isinstance(loader0, dal.AbstractImageLoader):
            loader0 = dal.get_loader_from_json(loader0, loader_type='MosaicLoader', **loader_config)
        if not isinstance(loader1, dal.AbstractImageLoader):
            loader1 = dal.get_loader_from_json(loader1, loader_type='MosaicLoader', **loader_config)
    else:
        raise TypeError('loader input type not supported.')
    mesh0.change_resolution(resolution)
    mesh1.change_resolution(resolution)
    initial_matches = read_matches_from_h5(match_name, target_resolution=resolution)
    xy0, xy1, weight = section_matcher(mesh0, mesh1, loader0, loader1,
        initial_matches=initial_matches, **matcher_config)
    with h5py.File(outname, 'w') as f:
        f.create_dataset('xy0', data=xy0, compression="gzip")
        f.create_dataset('xy1', data=xy1, compression="gzip")
        f.create_dataset('weight', data=weight, compression="gzip")
        f.create_dataset('resolution', data=resolution)
        f.create_dataset('name0', data=str_to_numpy_ascii(secnames[0]))
        f.create_dataset('name1', data=str_to_numpy_ascii(secnames[1]))
    return len(xy0)



class Stack:
    """
    A stack of sections used for optimization.
    Args:
        section_list(tuple): a list of section names in the order as in the stack.
        match_list(tuple): a list of all the match names.
        section_list & match_list should mostly be immutable once created.
    Kwargs:
        mesh_dir(str): path to the folder where the mesh can be cached/retrieved.
        match_dir(str): path to the folder where the matches can be cached/retrieved.
        mesh_cache(dict): maps mesh name to a list of connected mesh.Mesh object.
        link_cache(dict): maps link name to their optimizer.Link object.
        mesh_cache_size, link_cache_size (int): maximum size of the caches.
        match_name_delimiter: delimiter to split match name into two section names.
    """
    def __init__(self, section_list, match_list=None, **kwargs):
        assert len(section_list) == len(set(section_list))
        self.section_list = tuple(section_list)
        self._mesh_dir = kwargs.get('mesh_dir', None)
        self._match_dir = kwargs.get('match_dir', None)
        self._mesh_cache_size = kwargs.get('mesh_cache_size', None)
        self._link_cache_size = kwargs.get('link_cache_size', None)
        self._match_name_delimiter = kwargs.get('match_name_delimiter', '__to__')
        self.lock_flags = kwargs.get('lock_flags', defaultdict(lambda: False))
        mesh_cache = kwargs.get('mesh_cache', {})
        link_cache = kwargs.get('link_cache', {})
        self._mesh_cache = OrderedDict()
        self._mesh_cache.update(mesh_cache)
        self._link_cache = OrderedDict()
        self._link_cache.update(link_cache)
        if bool(self._mesh_cache):
            for meshname, m in self._mesh_cache.items():
                self.lock_flags[meshname] = m[0].locked
        if match_list is None:
            if self._match_dir is not None:
                mlist = glob.glob(os.path.join(self._match_dir, '*.h5'))
                if bool(mlist):
                    match_list = [os.path.basename(m).replace('.h5', '') for m in mlist]
            if match_list is None:
                if bool(self._link_cache):
                    match_list = list(self._link_cache.keys())
                else:
                    raise RuntimeError('no match list found.')
        self.match_list = self.filtered_match_list(match_list=match_list)


    def get_mesh(self, secname):
        if not isinstance(secname, str):
            secname = self.section_list[int(secname)]   # indexing by id
        if secname in self._mesh_cache:
            return self._mesh_cache[secname]
        elif self._mesh_dir is None:
            raise RuntimeError('mesh_dir not defined.')
        else:
            meshpath = os.path.join(self._mesh_dir, secname+'.h5')
            if not os.path.isfile(meshpath):
                raise RuntimeError(f'{meshpath} not found.')
            uid = self.secname_to_id(secname)
            locked = self.lock_flags[secname]
            M = Mesh.from_h5(meshpath, uid=uid, locked=locked, name=secname)
            Ms = M.divide_disconnected_mesh(save_material=True)
            if (self._mesh_cache_size is not None) and self._mesh_cache_size > 0:
                self._mesh_cache[secname] = Ms
            if (self._mesh_cache_size is not None):
                while len(self._mesh_cache) > self._mesh_cache_size:
                    cached_name, cached_Ms = self._mesh_cache.popitem(last=False)
                    if self._mesh_dir is not None:
                        cached_M = Mesh.combine_mesh(cached_Ms, save_material=True)
                        outname = os.path.join(self._mesh_dir, cached_name+'.h5')
                        cached_M.save_to_h5(outname, vertex_flags=const.MESH_GEARS, save_material=True)
            return Ms


    def flush(self):
        while bool(self._mesh_cache):
            cached_name, cached_Ms = self._mesh_cache.popitem(last=False)
            if self._mesh_dir is not None:
                cached_M = Mesh.combine_mesh(cached_Ms, save_material=True)
                outname = os.path.join(self._mesh_dir, cached_name+'.h5')
                cached_M.save_to_h5(outname, vertex_flags=const.MESH_GEARS, save_material=True)


    def filtered_match_list(self, match_list=None, secnames=None):
        if match_list is None:
            match_list = self.match_list
        if secnames is None:
            secnames = self.section_list
        filtered_match_list = []
        for matchname in self.match_list:
            names = matchname.split(self._match_name_delimiter)
            if names[0] in secnames and names[1] in secnames:
                filtered_match_list.append(matchname)
        return tuple(filtered_match_list)


    def secname_to_id(self, secname):
        if not hasattr(self, '_name_id_lut'):
            self._name_id_lut = {name: k for k, name in enumerate(self.section_list)}
        return self._name_id_lut.get(secname, -1)


    @property
    def num_sections(self):
        return len(self.section_list)