from collections import defaultdict, OrderedDict
from functools import partial
import hashlib
import json
import numpy as np
import os
from scipy.ndimage import distance_transform_cdt
from scipy import sparse
import scipy.sparse.csgraph as csgraph
import shapely
import yaml
import time

from feabas import config, dal, logging, storage
from feabas.mesh import Mesh, transform_mesh
from feabas.concurrent import submit_to_workers, REMOTE_FRAMEWORKS, is_daemon_process
from feabas.spatial import scale_coordinates
from feabas.matcher import section_matcher
from feabas.optimizer import SLM
import feabas.constant as const
from feabas.common import str_to_numpy_ascii, Match, rearrange_section_order, parse_json_file

H5File = storage.h5file_class()

def read_matches_from_h5(match_name, target_resolution=None):
    with H5File(match_name, 'r') as f:
        xy0 = f['xy0'][()]
        xy1 = f['xy1'][()]
        weight = f['weight'][()].ravel()
        resolution = f['resolution'][()]
        if isinstance(resolution, np.ndarray):
            resolution = resolution.item()
        if 'strain' in f.keys():
            strain = f['strain'][()]
            if isinstance(strain, np.ndarray):
                strain = strain.item()
        else:
            strain = config.DEFAULT_DEFORM_BUDGET
    if target_resolution is not None:
        scale = resolution / target_resolution
        xy0 = scale_coordinates(xy0, scale)
        xy1 = scale_coordinates(xy1, scale)
    return Match(xy0, xy1, weight, strain)


def match_section_from_initial_matches(match_name, meshes, loaders, out_dir, conf=None, ignore_initial_match=False):
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
    outname = storage.join_paths(out_dir, os.path.basename(match_name))
    if storage.file_exists(outname):
        return None
    if isinstance(conf, str) and conf.lower().endswith('.yaml'):
        with storage.File(conf, 'r') as f:
            conf = yaml.safe_load(f)
    if 'matching' in conf:
        conf = conf['matching']
    elif conf is None:
        conf = {}
    elif not isinstance(conf, dict):
        raise TypeError('configuration type not supported.')
    match_name_delimiter = conf.get('match_name_delimiter', '__to__')
    working_mip_level = conf.get('working_mip_level', 0)
    resolution = config.montage_resolution() * (2 ** working_mip_level)
    loader_config = conf.get('loader_config', {}).copy()
    matcher_config = conf.get('matcher_config', {}).copy()
    secnames = os.path.splitext(os.path.basename(match_name))[0].split(match_name_delimiter)
    if 'cache_size' in loader_config and loader_config['cache_size'] is not None:
        loader_config['cache_size'] = loader_config['cache_size'] // (2 * matcher_config.get('num_workers',1))
    if 'cache_capacity' in loader_config and loader_config['cache_capacity'] is not None:
        loader_config['cache_capacity'] = loader_config['cache_capacity'] // (2 * matcher_config.get('num_workers',1))
    if isinstance(meshes, str):
        meshes = (storage.join_paths(meshes, secnames[0]+'.h5'), storage.join_paths(meshes, secnames[1]+'.h5'))
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
        if storage.file_exists(storage.join_paths(loaders, secnames[0]+'.json')):
            loader0 = storage.join_paths(loaders, secnames[0]+'.json')
        elif storage.file_exists(storage.join_paths(loaders, secnames[0]+'.txt')):
            loader0 = storage.join_paths(loaders, secnames[0]+'.txt')
        else:
            raise RuntimeError(f'cannot find loaders for {secnames[0]}')
        if storage.file_exists(storage.join_paths(loaders, secnames[1]+'.json')):
            loader1 = storage.join_paths(loaders, secnames[1]+'.json')
        elif storage.file_exists(storage.join_paths(loaders, secnames[1]+'.txt')):
            loader1 = storage.join_paths(loaders, secnames[1]+'.txt')
        else:
            raise RuntimeError(f'cannot find loaders for {secnames[1]}')
        loaders = (loader0, loader1)
    if isinstance(loaders, (tuple, list)):
        loader0, loader1 = loaders
        if not isinstance(loader0, dal.AbstractImageLoader):
            loader0 = dal.get_loader_from_json(loader0, mip=working_mip_level, **loader_config)
        if not isinstance(loader1, dal.AbstractImageLoader):
            loader1 = dal.get_loader_from_json(loader1, mip=working_mip_level, **loader_config)
    else:
        raise TypeError('loader input type not supported.')
    mesh0.change_resolution(resolution)
    mesh1.change_resolution(resolution)
    if ignore_initial_match:
        initial_matches = None
    else:
        initial_matches = read_matches_from_h5(match_name, target_resolution=resolution)
    xy0, xy1, weight, strain = section_matcher(mesh0, mesh1, loader0, loader1,
        initial_matches=initial_matches, **matcher_config)
    if xy0 is None:
        return 0
    else:
        with H5File(outname, 'w') as f:
            f.create_dataset('xy0', data=xy0, compression="gzip")
            f.create_dataset('xy1', data=xy1, compression="gzip")
            f.create_dataset('weight', data=weight, compression="gzip")
            f.create_dataset('resolution', data=resolution)
            f.create_dataset('strain', data=strain)
            f.create_dataset('name0', data=str_to_numpy_ascii(secnames[0]))
            f.create_dataset('name1', data=str_to_numpy_ascii(secnames[1]))
        return len(xy0)


def get_convex_hull(tname, wkb=False, resolution=None):
    M = Mesh.from_h5(tname)
    if resolution is not None:
        M.change_resolution(resolution)
    R = M.shapely_regions(gear=const.MESH_GEAR_MOVING, offsetting=True)
    R = shapely.convex_hull(R)
    if wkb:
        return shapely.to_wkb(R)
    else:
        return R


def apply_transform_normalization(tname, out_dir=None, R=np.eye(3), txy=np.zeros(2),resolution=None):
    M = Mesh.from_h5(tname)
    locked = M.locked
    M.locked = False
    if resolution is not None:
        M.change_resolution(resolution)
    M.apply_affine(R, gear=const.MESH_GEAR_FIXED)
    M.apply_translation(txy, gear=const.MESH_GEAR_FIXED)
    if M.vertices_initialized(gear=const.MESH_GEAR_MOVING):
        M.apply_affine(R, gear=const.MESH_GEAR_MOVING)
        M.apply_translation(txy, gear=const.MESH_GEAR_MOVING)
    if out_dir is not None:
        outname = storage.join_paths(out_dir, os.path.basename(tname))
    else:
        outname = tname
    M.locked = locked
    M.save_to_h5(outname, vertex_flags=const.MESH_GEARS, save_material=True)


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
  ## --------------------------- initialization ---------------------------- ##
    def __init__(self, section_list=None, match_list=None, **kwargs):
        self._mesh_dir = kwargs.get('mesh_dir', None)
        self._mesh_out_dir = kwargs.get('mesh_out_dir', None)
        self._tform_dir = kwargs.get('tform_dir', self._mesh_out_dir)
        if self._tform_dir == self._mesh_out_dir:
            self._mesh_dir_list = (self._mesh_dir, self._tform_dir)
        else:
            self._mesh_dir_list = (self._mesh_dir, self._mesh_out_dir, self._tform_dir)
        self._match_dir = kwargs.get('match_dir', None)
        if section_list is None:
            if self._mesh_dir is None:
                raise RuntimeError('mesh_dir not defined.')
            slist = storage.list_folder_content(storage.join_paths(self._mesh_dir, '*.h5'))
            if bool(slist):
                section_list = sorted([os.path.basename(s).replace('.h5', '') for s in slist])
            else:
                raise RuntimeError('no section found.')
            section_order_file = kwargs.get('section_order_file', None)
            if section_order_file is not None:
                section_list = rearrange_section_order(section_list, section_order_file)[0]
        assert len(section_list) == len(set(section_list))
        self._specified_out_dirs = kwargs.get('specified_out_dirs', {}) # can define the output directory for specified section
        self.section_list = tuple(section_list)
        self._mesh_cache_size = kwargs.get('mesh_cache_size', 0)
        self._link_cache_size = kwargs.get('link_cache_size', None)
        self._match_name_delimiter = kwargs.get('match_name_delimiter', '__to__')
        lock_flags = kwargs.get('lock_flags', None)
        mesh_cache = kwargs.get('mesh_cache', {})
        link_cache = kwargs.get('link_cache', {})
        mip_level = kwargs.get('mip_level', 0)
        self._resolution = config.montage_resolution() * (2 ** mip_level)
        self._mesh_cache = OrderedDict()
        self._mesh_cache.update(mesh_cache)
        self._link_cache = OrderedDict()
        self._link_cache.update(link_cache)
        self.lock_flags = defaultdict(lambda: False)
        if lock_flags is None:
            lock_flags = self.aligned_and_committed()
        if isinstance(lock_flags, dict):
            self.lock_flags.update(lock_flags)
        elif isinstance(lock_flags, (tuple, list, np.ndarray)):
            self.lock_flags.update({nm: flg for nm, flg in zip(self.section_list, lock_flags)})
        if bool(self._mesh_cache):
            self.normalize_mesh_lock_status()
            self.normalize_mesh_resoltion()
        if match_list is None:
            if self._match_dir is not None:
                mlist = storage.list_folder_content(storage.join_paths(self._match_dir, '*.h5'))
                if bool(mlist):
                    match_list = [os.path.basename(m).replace('.h5', '') for m in mlist]
            if match_list is None:
                if bool(self._link_cache):
                    match_list = list(self._link_cache.keys())
                else:
                    raise RuntimeError('no match list found.')
        self.match_list = self.filtered_match_list(match_list=match_list)
        self.save_overflow = kwargs.get('save_overflow', True)
        self._logger = kwargs.get('logger', None)


    def normalize_mesh_resoltion(self):
        for mshes in self._mesh_cache.values():
            for msh in mshes:
                msh.change_resolution(self._resolution)


    def normalize_mesh_lock_status(self, secnames=None):
        if secnames is None:
            secnames = list(self._mesh_cache.keys())
        for mshname in secnames:
            if mshname not in self._mesh_cache:
                continue
            mshes = self._mesh_cache[mshname]
            lock_flag = self.lock_flags[mshname]
            outcast_flg = [m.is_outcast for m in mshes]
            if np.all(outcast_flg):
                self.lock_flags[mshname] = False
            for msh in mshes:
                if not msh.is_outcast:
                    msh.locked = lock_flag
                else:
                    msh.locked = False


    def init_dict(self, secnames=None, include_cache=False, check_lock=False, **kwargs):
        init_dict = {}
        init_dict['mesh_dir'] = self._mesh_dir
        init_dict['mesh_out_dir'] = self._mesh_out_dir
        init_dict['tform_dir'] = self._tform_dir
        init_dict['match_dir'] = self._match_dir
        match_list = self.filtered_match_list(secnames=secnames, check_lock=check_lock)
        section_list = self.filter_section_list_from_matches(match_list)
        section_list_set = set(section_list)
        init_dict['section_list'] = section_list
        init_dict['match_list'] = match_list
        init_dict['lock_flags'] = {s: self.lock_flags[s] for s in section_list_set}
        if isinstance(self._specified_out_dirs, dict) and len(self._specified_out_dirs) > 0:
            init_dict['specified_out_dirs'] = {s:p for s,p in self._specified_out_dirs.items() if s in section_list_set}
        elif isinstance(self._specified_out_dirs, str):
            init_dict['specified_out_dirs'] = self._specified_out_dirs
        init_dict['resolution'] = self._resolution
        init_dict['mesh_cache_size'] = self._mesh_cache_size
        init_dict['link_cache_size'] = self._link_cache_size
        if include_cache:
            init_dict['mesh_cache'] = {s: self._mesh_cache[s] for s in section_list if s in self._mesh_cache}
            init_dict['link_cache'] = {s: self._link_cache[s] for s in match_list if s in self._link_cache}
        init_dict['logger'] = self._logger
        init_dict.update(kwargs)
        return init_dict


    def assign_section_to_chunks(self, chunk_map):
        chunk_id_lut = {}
        for k, secnames in enumerate(chunk_map.values()):
            chunk_locks = self.aligned_and_committed(secnames=secnames)
            self.update_lock_flags({s:l for s, l in zip(secnames, chunk_locks)})
            for sn in secnames:
                chunk_id_lut[sn] = k
        secnames = [s for s in self.section_list if s in chunk_id_lut]
        match_list = []
        muted_match_list = []
        for matchname in self.match_list:
            name0, name1 = self.matchname_to_secnames(matchname)
            if (name0 in chunk_id_lut) and (name1 in chunk_id_lut) and (chunk_id_lut[name0] == chunk_id_lut[name1]):
                match_list.append(matchname)
            else:
                muted_match_list.append(matchname)
        self.update_section_list(secnames)
        self.update_match_list(match_list)
        return muted_match_list


  ## --------------------------- meshes & matches -------------------------- ##
    def update_section_list(self, section_list):
        if self.section_list != section_list:
            self.section_list = section_list
            self._name_id_lut = None
            self._section_connection_matrix = None
            self._matchname_to_secids_mapper = None


    def update_match_list(self, match_list):
        if self.match_list != match_list:
            self.match_list = match_list
            self._secname_to_matchname_mapper = None
            self._section_connection_matrix = None
            self._matchname_to_secids_mapper = None


    def get_mesh(self, secname, divide=True):
        if not isinstance(secname, str):
            secname = self.section_list[int(secname)]   # indexing by id
        if secname in self._mesh_cache:
            self._mesh_cache.move_to_end(secname, last=True)
            return self._mesh_cache[secname]
        else:
            for fdir in self._mesh_dir_list[::-1]:
                meshpath = storage.join_paths(fdir, secname+'.h5')
                if (meshpath is not None) and storage.file_exists(meshpath):
                    uid = self.secname_to_id(secname)
                    locked = self.lock_flags[secname]
                    M = Mesh.from_h5(meshpath, uid=uid, locked=locked, name=secname)
                    M.change_resolution(self._resolution)
                    if divide:
                        Ms = M.divide_disconnected_mesh(save_material=True)
                    else:
                        Ms = [M]
                    if (self._mesh_cache_size is None) or (self._mesh_cache_size > 0):
                        self._mesh_cache[secname] = Ms
                        self.trim_mesh_cache()
                    break
            else:
                Ms = []
                # raise RuntimeError(f'mesh for {secname} not found.')
            return Ms


    def flush_meshes(self):
        while bool(self._mesh_cache):
            self.dump_first_mesh()


    def trim_mesh_cache(self):
        if self._mesh_cache_size is not None:
            while len(self._mesh_cache) > self._mesh_cache_size:
                self.dump_first_mesh()


    def dump_first_mesh(self):
        """self._mesh_cache is a FIFO cache"""
        cached_name, cached_Ms = self._mesh_cache.popitem(last=False)
        rel_match_names = self.secname_to_matchname_mapper[cached_name]
        if self.save_overflow:
            self.save_mesh_for_one_section(cached_name, cached_Ms)
        for matchname in rel_match_names:
            self.dump_link(matchname)


    def save_mesh_for_one_section(self, secname, Ms=None):
        saved = False
        flag = None
        if isinstance(self._specified_out_dirs, str):
            out_dir = self._specified_out_dirs
        elif isinstance(self._specified_out_dirs, dict) and (secname in self._specified_out_dirs):
            out_dir = self._specified_out_dirs[secname]
        else:
            flag = np.max(list(self.mesh_versions.values())+[1])
            out_dir = self._mesh_dir_list[flag]
        if out_dir is None:
            return saved
        if (flag is None) and (out_dir in self._mesh_dir_list):
            flag = self._mesh_dir_list.index(out_dir)
        if Ms is None:
            if secname in self._mesh_cache:
                Ms = self._mesh_cache[secname]
            else:
                return saved
        anchored_meshes = [m for m in Ms if not m.is_outcast]
        if len(anchored_meshes) != len(Ms):
            logger = logging.get_logger(self._logger)
            if len(anchored_meshes) == 0:
                self.lock_flags[secname] = False
                logger.warning(f'{secname}: not anchored.')
            else:
                logger.warning(f'{secname}: partially not anchored.')
        if len(anchored_meshes) > 0:
            M = Mesh.combine_mesh(anchored_meshes, save_material=True)
            outname = storage.join_paths(out_dir, secname + '.h5')
            if M.modified_in_current_session or not storage.file_exists(outname):
                M.save_to_h5(outname, vertex_flags=const.MESH_GEARS, save_material=True)
                saved = True
                if (flag is not None) and hasattr(self, '_mesh_versions') and (self._mesh_versions is not None):
                    self._mesh_versions[secname] = flag
                for m in anchored_meshes:
                    m.modified_in_current_session = False
        return saved


    def get_link(self, matchname):
        if matchname in self._link_cache:
            self._link_cache.move_to_end(matchname, last=True)
            return self._link_cache[matchname]
        names = self.matchname_to_secnames(matchname)
        if (not names[0] in self._mesh_cache) or (not names[1] in self._mesh_cache):
            return None
        else:
            mesh_list0 = self._mesh_cache[names[0]]
            mesh_list1 = self._mesh_cache[names[1]]
        if self._match_dir is None:
            raise RuntimeError('match_dir not defined.')
        else:
            matchpath = storage.join_paths(self._match_dir, matchname+'.h5')
            if not storage.file_exists(matchpath):
                raise RuntimeError(f'{matchpath} not found.')
            mtch = read_matches_from_h5(matchpath, target_resolution=self._resolution)
            links = SLM.distribute_link(mesh_list0, mesh_list1, mtch)
            if (self._link_cache_size is None) or (self._link_cache_size > 0):
                self._link_cache[matchname] = links
            if self._link_cache_size is not None:
                while len(self._link_cache) > self._link_cache_size:
                    self.dump_link()
            return links


    def dump_link(self, matchname=None):
        if matchname is None:
            self._link_cache.popitem(last=False)
        elif matchname in self._link_cache:
            self._link_cache.pop(matchname)


    def filtered_match_list(self, match_list=None, secnames=None, check_lock=True):
        if match_list is None:
            match_list = self.match_list
        if secnames is None:
            secnames = self.section_list
        secnames = set(secnames)
        filtered_match_list = []
        for matchname in match_list:
            names = self.matchname_to_secnames(matchname)
            if (names[0] not in secnames) or (names[1] not in secnames):
                continue
            if check_lock:
                if self.lock_flags[names[0]] and self.lock_flags[names[1]]:
                    continue
            filtered_match_list.append(matchname)
        return tuple(filtered_match_list)


    def filter_section_list_from_matches(self, match_list=None, secnames=None):
        if match_list is None:
            match_list = self.match_list
        if secnames is None:
            secnames = self.section_list
        new_section_list = set()
        for matchname in match_list:
            names = self.matchname_to_secnames(matchname)
            new_section_list.update(names)
        id_sec = sorted([(self.secname_to_id(s), s) for s in new_section_list])
        filtered_sections = [s[1] for s in id_sec if s[0]>=0]
        return tuple(filtered_sections)


  ## ----------------------------- optimization ---------------------------- ##
    def initialize_SLM(self, secnames=None, **kwargs):
        # temporarily increase the mesh cache size to make sure all the meshes
        # exist in the cache during the time of SLM creation. they are saved in
        # SLM anyway so no point to free them up now.
        mesh_cache_size0 = self._mesh_cache_size
        link_cache_size = self._link_cache_size
        if secnames is None:
            secnames = self.section_list
        match_list = self.filtered_match_list(secnames=secnames, check_lock=True)
        if link_cache_size is None:
            self._link_cache_size = len(match_list)
        else:
            self._link_cache_size = max(len(match_list), link_cache_size)
        secnames = self.filter_section_list_from_matches(match_list=match_list)
        if mesh_cache_size0 is not None:
            self._mesh_cache_size = max(len(secnames), mesh_cache_size0)
        meshes = []
        for secname in secnames:
            meshes.extend(self.get_mesh(secname))
        links = []
        for matchname in match_list:
            links.extend(self.get_link(matchname))
        optm = SLM(meshes, links, **kwargs)
        self._mesh_cache_size = mesh_cache_size0
        self._link_cache_size = link_cache_size
        return optm


    def optimize_slide_window(self, **kwargs):
        num_workers = kwargs.pop('num_workers', 1)
        start_loc = kwargs.get('start_loc', 'L') # L or R or M, in case no locked sections for references
        window_size = kwargs.get('window_size', None)
        worker_settings = kwargs.get('worker_settings', {}).copy()
        ensure_continuous = kwargs.pop('ensure_continuous', False) # ensure all sections are connected or linked to a reference section, otherwise align the largest bunch
        no_slide = kwargs.get('no_slide', False)    # align only small segments of the stack that can be done in one shot and don't require window to slide
        if kwargs.get('logger', None) is not None:
            self._logger = kwargs['logger']
        else:
            kwargs.setdefault('logger', self._logger)
        if (window_size is None) or window_size > self.num_sections:
            window_size = self.num_sections
            buffer_size = 0
        else:
            buffer_size = kwargs.get('buffer_size', window_size//4)
        if buffer_size < 1:
            buffer_size = round(buffer_size * window_size)
        parallel_framework = worker_settings.get('parallel_framework', config.parallel_framework())
        sent_to_remote = (parallel_framework in REMOTE_FRAMEWORKS) and (not is_daemon_process())
        updated_sections = []
        residues = {}
        if (self.num_sections == 0) or (len(self.match_list) == 0):
            to_optimize = np.zeros(1, dtype=bool)
        else:
            to_optimize = ~self.locked_array
        while np.any(to_optimize):
            connected_free_sections = self.connected_sections(section_filter=to_optimize)
            if len(connected_free_sections) > 1:
                kwarg_list = [kwargs]
                anchored = np.zeros(len(connected_free_sections), dtype=bool)
                secname_lists = []
                secnums = np.zeros(len(connected_free_sections), dtype=np.uint32)
                for ks, slist in enumerate(connected_free_sections):
                    secnames = self.pad_section_list_w_refs(section_list=slist)
                    if len(secnames) > len(slist):
                        anchored[ks] = True
                    secnums[ks] = len(slist)
                    secname_lists.append(secnames)
                if ensure_continuous:
                    if np.any(self.locked_array):
                        secname_lists = [s for flg, s in zip(anchored, secname_lists) if flg]
                    else:
                        indx = np.argmax(secnums)
                        secname_lists = [secname_lists[indx]]
                if no_slide:
                    secname_lists = [s for s in secname_lists if len(secname_lists) <= (window_size + buffer_size)]
                args_list = [(self.init_dict(secnames=sn, check_lock=True),) for sn in secname_lists]
                if len(args_list) == 0:
                    break
                for reslt in submit_to_workers(Stack.subprocess_optimize_stack, args=args_list, kwargs=kwarg_list, num_workers=num_workers, **worker_settings):
                    snms, res = reslt
                    updated_sections.extend(snms)
                    residues.update(res)
                    for sn in snms:
                        self._mesh_cache.pop(sn, None)
                self.update_lock_flags({s: True for s in updated_sections})
                self._mesh_versions = None
                break
            else:
                seclist0 = connected_free_sections[0]
                seclist_w_ref = self.pad_section_list_w_refs(section_list=seclist0)
                if ensure_continuous and (len(seclist0) == len(seclist_w_ref)) and np.any(self.locked_array):
                    break
                if np.sum(to_optimize) <= (window_size + buffer_size):
                    if sent_to_remote:
                        args_list = [(self.init_dict(secnames=seclist_w_ref, check_lock=True),)]
                        for reslt in submit_to_workers(Stack.subprocess_optimize_stack, args=args_list, kwargs=[kwargs], num_workers=1, **worker_settings):
                            snms, res = reslt
                            updated_sections.extend(snms)
                            residues.update(res)
                            for sn in snms:
                                self._mesh_cache.pop(sn, None)
                        self._mesh_versions = None
                    else:
                        res = self.optimize_section_list(seclist_w_ref, **kwargs)
                        residues.update(res)
                        for secname in seclist0:
                            updated = self.save_mesh_for_one_section(secname)
                            if updated:
                                updated_sections.append(secname)
                    self.update_lock_flags({s: True for s in seclist0})
                else:
                    if no_slide:
                        break
                    seeding = to_optimize
                    if np.all(seeding):
                        seeding = seeding.copy()
                        if isinstance(start_loc, int):
                            seeding[start_loc] = False
                        elif isinstance(start_loc, str):
                            if start_loc.lower().startswith('l'):
                                seeding[0] = False
                            elif start_loc.lower().startswith('r'):
                                seeding[-1] = False
                            else:
                                seeding[seeding.size//2] = False
                        else:
                            raise TypeError
                    dis_seed = distance_transform_cdt(seeding)
                    dis_seed[~to_optimize] = np.max(dis_seed) + 1
                    indx_d = np.argsort(dis_seed)
                    indx_opt = indx_d[:(window_size+buffer_size)]
                    seclist_opt = [self.section_list[s] for s in indx_opt]
                    if buffer_size == 0:
                        seclist_cmt = seclist_opt
                    else:
                        flag_opt = np.zeros_like(to_optimize)
                        flag_opt[indx_opt] = True
                        flag_fut = to_optimize & (~flag_opt)
                        dis_e = distance_transform_cdt(~flag_fut).clip(0, None)
                        dis_t = -np.ones_like(dis_e, dtype=np.float32)
                        lbl_opt = np.cumsum(np.diff(flag_opt, prepend=0).clip(0, None)) * flag_opt
                        for lbl in range(1, np.max(lbl_opt)+1):
                            idxt = lbl_opt == lbl
                            dis_t[idxt] = min(np.max(dis_e[idxt]) / 2, buffer_size)
                        indx_cmt = np.nonzero((dis_e > dis_t) & flag_opt)[0]
                        seclist_cmt = [self.section_list[s] for s in indx_cmt]
                    seclist = self.pad_section_list_w_refs(section_list=seclist_opt)
                    if sent_to_remote:
                        args_list = [(self.init_dict(secnames=seclist, check_lock=True),)]
                        for reslt in submit_to_workers(Stack.subprocess_optimize_stack, args=args_list, kwargs=[kwargs], num_workers=1, **worker_settings):
                            snms, res = reslt
                            updated_sections.extend(snms)
                            residues.update(res)
                            for sn in snms:
                                self._mesh_cache.pop(sn, None)
                        self._mesh_versions = None
                    else:
                        res = self.optimize_section_list(seclist, **kwargs)
                        residues.update(res)
                        for secname in seclist_cmt:
                            updated = self.save_mesh_for_one_section(secname)
                            if updated:
                                updated_sections.append(secname)
                    self.update_lock_flags({s: True for s in seclist_cmt})
                to_optimize = ~self.locked_array
        return updated_sections, residues


    def optimize_section_list(self, section_list, **kwargs):
        target_gear = kwargs.get('target_gear', const.MESH_GEAR_MOVING)
        optimize_rigid = kwargs.get('optimize_rigid', True)
        rigid_params = kwargs.get('rigid_params', {}).copy()
        optimize_elastic = kwargs.get('optimize_elastic', True)
        elastic_params = kwargs.get('elastic_params', {}).copy()
        residue_len = kwargs.get('residue_len', 0)
        residue_mode = kwargs.get('residue_mode', None)
        logger_info = kwargs.get('logger', None)
        need_anchor = kwargs.get('need_anchor', False)
        logger = logging.get_logger(logger_info)
        if residue_len < 0:
            aspect_ratio = config.section_thickness() / self._resolution
            residue_len = max(1, abs(residue_len) * aspect_ratio)
        residue = {}
        if len(section_list) == 0:
            logger.info('no section to optimize.')
            return residue
        optm = self.initialize_SLM(section_list)
        if len(optm.meshes) == 0:
            logger.info(f'{section_list[0]} -> {section_list[-1]}: all sections settled.')
            return residue
        outcasts = optm.flag_outcasts()
        if np.all(outcasts):
            logger.error(f'{optm.meshes[0].name} -> {optm.meshes[-1].name}: disconnected due to lack of matches. abort.')
            return residue
        if need_anchor and (not np.any(optm.lock_flags)):
            logger.error(f'{optm.meshes[0].name} -> {optm.meshes[-1].name}: disconnected due to lack of matches. abort.')
        cost = None
        t0 = time.time()
        if optimize_rigid:
            optm.optimize_affine_cascade(target_gear=target_gear, **rigid_params)
            if target_gear != const.MESH_GEAR_FIXED:
                optm.anneal(gear=(target_gear, const.MESH_GEAR_FIXED), mode=const.ANNEAL_CONNECTED_RIGID)
        if optimize_elastic:
            stiffness_lambda = elastic_params.get('stiffness_lambda', None)
            mesh_soft_power = elastic_params.get('mesh_soft_power', 1.5)
            if (stiffness_lambda is None) or (mesh_soft_power > 0): # determine overall mesh stiffness based on matching strains
                mesh_strains = defaultdict(list)
                for lnk in optm.links:
                    mesh_strains[lnk.uids[0]].append(lnk.strain)
                    mesh_strains[lnk.uids[1]].append(lnk.strain)
                for m_uid, stns in mesh_strains.items():
                    mesh_strains[m_uid] = max(np.median(stns), 1e-3)
                avg_deform = np.mean([s for s in mesh_strains.values()])
                if mesh_soft_power > 0:
                    for m in optm.meshes:
                        m.soft_factor = min(2, (avg_deform / mesh_strains.get(m.uid, avg_deform)) ** mesh_soft_power) # make mesh with high matching distortion softer
                if stiffness_lambda is None:
                    elastic_params['stiffness_lambda'] = 0.5 * (config.DEFAULT_DEFORM_BUDGET / avg_deform) ** 2
            if 'callback_settings' in elastic_params:
                elastic_params['callback_settings'].setdefault('early_stop_thresh', config.montage_resolution() / self._resolution)
            cost = optm.optimize_elastic(target_gear=target_gear, **elastic_params)
            if (residue_mode is not None) and (residue_len > 0):
                if residue_mode == 'huber':
                    optm.set_link_residue_huber(residue_len)
                else:
                    optm.set_link_residue_threshold(residue_len)
                weight_modified, _ = optm.adjust_link_weight_by_residue(gear=(target_gear, target_gear), relax_first=True)
                if weight_modified:
                    cost1 = optm.optimize_elastic(target_gear=target_gear, **elastic_params)
                    cost = (cost[0], cost1[-1])
        for matchname, lnks in self._link_cache.items():
            dxy = np.concatenate([lnk.dxy(gear=1) for lnk in lnks], axis=0)
            dis = np.sum(dxy ** 2, axis=1)**0.5
            residue[matchname] = (dis.max(), dis.mean())
        logger.info(f'{optm.meshes[0].name} -> {optm.meshes[-1].name}: cost {cost} | {time.time()-t0} sec')
        return residue


    def update_lock_flags(self, flags):
        self.lock_flags.update(flags)
        self.normalize_mesh_lock_status(secnames=list(flags.keys()))


    def unlock_all(self):
        flags = {secname: False for secname in self.section_list}
        self.update_lock_flags(flags)


    def lock_all(self):
        flags = {secname: True for secname in self.section_list}
        self.update_lock_flags(flags)


    @property
    def locked_array(self):
        return np.array([self.lock_flags[s] for s in self.section_list], copy=False)


  ## -------------------------------- queries ------------------------------ ##
    def secname_to_id(self, secname):
        if (not hasattr(self, '_name_id_lut')) or (self._name_id_lut is None):
            self._name_id_lut = {name: k for k, name in enumerate(self.section_list)}
        return self._name_id_lut.get(secname, -1)


    def matchname_to_secnames(self, matchname):
        return matchname.split(self._match_name_delimiter)


    @property
    def section_connection_matrix(self):
        if not hasattr(self, '_section_connection_matrix') or (self._section_connection_matrix is None):
            edges = np.array([s for s in self.matchname_to_secids_mapper.values()])
            Nsec = self.num_sections
            if edges.size == 0:
                self._section_connection_matrix = sparse.csc_matrix((Nsec, Nsec), dtype=bool)
            else:
                idx0 = edges[:,0]
                idx1 = edges[:,1]
                V = np.ones_like(idx0, dtype=bool)
                A = sparse.csr_matrix((V, (idx0, idx1)), shape=(Nsec, Nsec))
                A = (A + A.T) > 0
                A.eliminate_zeros()
                self._section_connection_matrix = A
                A = (A + A.T) > 0
                A.eliminate_zeros()
                self._section_connection_matrix = A
        return self._section_connection_matrix


    def connected_sections(self, section_filter=None):
        if section_filter is None:
            section_ids = np.arange(self.num_sections)
        elif isinstance(section_filter, np.ndarray):
            if section_filter.dtype == bool:
                section_ids = np.nonzero(section_filter)[0]
            else:
                section_ids = np.array(section_filter, dtype=np.int32)
        elif isinstance(section_filter[0], str):
            section_filter = set(section_filter)
            section_ids = np.array([k for k, s in enumerate(self.section_list) if s in section_filter])
        else:
            raise TypeError
        A = self.section_connection_matrix[section_ids][:, section_ids]
        ngrps, lbls = csgraph.connected_components(A, directed=False, return_labels=True)
        conn_sections = []
        for lbl in range(ngrps):
            sel = section_ids[lbls == lbl]
            conn_sections.append([self.section_list[k] for k in sel])
        return conn_sections


    def pad_section_list_w_refs(self, section_list):
        sel_flag = np.array([(s in section_list) for s in self.section_list])
        A = self.section_connection_matrix
        ref_flag = A.dot(sel_flag) & (~sel_flag) & self.locked_array
        combined_flag = sel_flag | ref_flag
        return [s for flg, s in zip(combined_flag, self.section_list) if flg]


    @property
    def mesh_versions(self):
        if (not hasattr(self, '_mesh_versions')) or (self._mesh_versions is None):
            version_list = []
            for k, fdir in enumerate(self._mesh_dir_list[1:]):
                tlist = storage.list_folder_content(storage.join_paths(fdir, '*.h5'))
                tnames0 = [os.path.basename(s).replace('.h5', '') for s in tlist]
                tnames = [s for s in tnames0 if s in self.section_list]
                version_list.append({s: (k+1) for s in tnames})
            self._mesh_versions = defaultdict(int)
            for tnames in version_list:
                self._mesh_versions.update(tnames)
        return self._mesh_versions


    def aligned_and_committed(self, secnames=None):
        if secnames is None:
            secnames = self.section_list
        flags = np.array([self.mesh_versions[s] for s in secnames])
        committed = (flags > 0) & (flags == np.max(flags))
        return committed


    @property
    def secname_to_matchname_mapper(self):
        if not hasattr(self, '_secname_to_matchname_mapper') or (self._secname_to_matchname_mapper is None):
            self._secname_to_matchname_mapper = defaultdict(set)
            for matchname in self.match_list:
                names = self.matchname_to_secnames(matchname)
                self._secname_to_matchname_mapper[names[0]].add(matchname)
                self._secname_to_matchname_mapper[names[1]].add(matchname)
        return self._secname_to_matchname_mapper


    @property
    def matchname_to_secids_mapper(self):
        if not hasattr(self, '_matchname_to_secids_mapper') or (self._matchname_to_secids_mapper is None):
            self._matchname_to_secids_mapper  = OrderedDict()
            for matchname in self.match_list:
                names = self.matchname_to_secnames(matchname)
                id0, id1 = self.secname_to_id(names[0]), self.secname_to_id(names[1])
                if (id0 >= 0) and (id1 >= 0):
                    self._matchname_to_secids_mapper[matchname] = [id0, id1]
        return self._matchname_to_secids_mapper


    @property
    def num_sections(self):
        return len(self.section_list)


    @staticmethod
    def subprocess_optimize_stack(init_dict, process_name='optimize_slide_window', **kwargs):
        stack = Stack(**init_dict)
        func = getattr(stack, process_name)
        return func(**kwargs)



class Aligner():
    UNALIGNED = 0
    CHUNK_ALIGNED = 1
    PREDEFORMED = 2
    ALIGNED = 3
    def __init__(self, mesh_dir, tform_dir, match_dir, **kwargs):
        self._mesh_dir = mesh_dir
        self._tform_dir = tform_dir
        self._match_dir = match_dir
        self._section_order_file = kwargs.get('section_order_file', None)
        self._chunk_dir = kwargs.get('chunk_dir', storage.join_paths(os.path.dirname(self._mesh_dir), 'chunked_tform'))
        self._predeform_dir = kwargs.get('predeform_dir', storage.join_paths(self._chunk_dir, 'predeformed'))
        self._mesh_dirs_map = (
            (Aligner.UNALIGNED, self._mesh_dir),
            (Aligner.CHUNK_ALIGNED, self._chunk_dir),
            (Aligner.PREDEFORMED, self._predeform_dir),
            (Aligner.ALIGNED, self._tform_dir),
        )
        self._match_name_delimiter = kwargs.get('match_name_delimiter', '__to__')
        chunk_map = kwargs.get('chunk_map', None)
        self._mip_level = kwargs.get('mip_level', 0)
        if isinstance(chunk_map, str):
            chunk_map, _ = parse_json_file(chunk_map)
        self._chunk_map = chunk_map
        self._auto_chunk = chunk_map is None
        self._chunk_map_file = kwargs.get('chunk_map_file', storage.join_paths(self._chunk_dir, 'chunk_map.json'))
        if (self._chunk_map_file is None) or (not storage.file_exists(self._chunk_map_file)):
            previous_chunk_map = None
        else:
            previous_chunk_map, _ = parse_json_file(self._chunk_map_file)
        self._previous_chunk_map = previous_chunk_map
        self._junction_width = kwargs.get('junction_width', 0.2)
        self._default_chunk_size = kwargs.get('default_chunk_size', 16)
        self._meta_dir = kwargs.get('meta_dir', storage.join_paths(self._chunk_dir, 'meta_sections'))
        self._meta_mesh_dir = storage.join_paths(self._meta_dir, 'mesh')
        self._meta_tform_dir = storage.join_paths(self._meta_dir, 'tform')
        self._meta_match_dir = storage.join_paths(self._meta_dir, 'matches')
        self._logger = kwargs.get('logger', None)
        self._user_section_list = kwargs.get('section_list', None)


    def get_section_list(self):
        sec_dict = {}
        for mvid_mdir in self._mesh_dirs_map:
            mvid, mdir = mvid_mdir
            meshlist = storage.list_folder_content(storage.join_paths(mdir, '*.h5'))
            sec_dict.update({os.path.basename(s).replace('.h5', ''): mvid for s in meshlist})
        secnames = sorted(sec_dict)
        if self._user_section_list is not None:
            secnames = [s for s in self._user_section_list if s in set(secnames)]
        if self._chunk_map is not None:
            secname_filt = set().union(*self._chunk_map.values())
            secnames = [s for s in secnames if s in secname_filt]
        if self._section_order_file is not None:
            secnames = rearrange_section_order(secnames, self._section_order_file)[0]
        self._section_list = secnames
        self._mesh_versions = {s: sec_dict[s] for s in self._section_list}


    def update_chunk_map(self):
        step_c = self._default_chunk_size
        self._junctional_sections = None
        if self._chunk_map is None:
            if self._previous_chunk_map is None:
                self._chunk_map = {}
                self._section_chunk_id = np.zeros(self.num_sections, dtype=np.int32)
                self._chunknames = []
                for kc, k0 in enumerate(range(0, self.num_sections, step_c)):
                    secnames = self.section_list[k0:(k0+step_c)]
                    prefix = secnames[0].split('_META')[0]
                    postfix = Aligner._hash_name('_'.join(secnames))
                    chnknm = '_META'.join((prefix, postfix))
                    self._chunknames.append(chnknm)
                    self._chunk_map[chnknm] = secnames
                    self._section_chunk_id[k0:(k0+step_c)] = kc
            else:
                section_chunk_id, chunknames = Aligner.get_section_chunk_id(self._previous_chunk_map, self.section_list)
                if not np.any(section_chunk_id == -1):
                    self._section_chunk_id = section_chunk_id
                    self._chunknames = chunknames
                    self._chunk_map = self._previous_chunk_map
                else:
                    self._chunk_map = self._previous_chunk_map.copy()
                    self._chunknames = chunknames
                    edt, inds = distance_transform_cdt(section_chunk_id==-1, return_indices=True)
                    inds = inds.ravel()
                    new_section_chunk_id = section_chunk_id[inds]
                    if section_chunk_id[0] == -1:
                        end_cid = new_section_chunk_id[0]
                        head_cnt1 = np.sum(new_section_chunk_id == end_cid)
                        chnknm0 = self._chunknames[0]
                        self._chunk_map.pop(chnknm0)
                        if head_cnt1 > step_c:
                            head_cnt0 = head_cnt1 - edt[0]
                            endpt = edt[0] - np.arange(max(0, step_c - head_cnt0), edt[0], step_c)
                            for idxt in endpt:
                                new_section_chunk_id[:idxt] -= 1
                        new_chunknames = []
                        for cid in range(np.min(new_section_chunk_id), end_cid+1):
                            idxt = np.flatnonzero(new_section_chunk_id == cid)
                            secnames = [self.section_list[kt] for kt in idxt]
                            prefix = secnames[0].split('_META')[0]
                            postfix = Aligner._hash_name('_'.join(secnames))
                            chnknm = '_META'.join((prefix, postfix))
                            new_chunknames.append(chnknm)
                            self._chunk_map[chnknm] = secnames
                        self._chunknames = new_chunknames + self._chunknames[1:]
                        new_section_chunk_id = new_section_chunk_id - min(new_section_chunk_id)
                    if section_chunk_id[-1] == -1:
                        end_cid = new_section_chunk_id[-1]
                        tail_cnt1 = np.sum(new_section_chunk_id == end_cid)
                        chnknm0 = self._chunknames[-1]
                        self._chunk_map.pop(chnknm0)
                        if tail_cnt1 > step_c:
                            tail_cnt0 = tail_cnt1 - edt[-1]
                            endpt = self.num_sections - edt[-1] + np.arange(max(0, step_c - tail_cnt0), edt[-1], step_c)
                            for idxt in endpt:
                                new_section_chunk_id[idxt:] += 1
                        new_chunknames = []
                        for cid in range(end_cid, np.max(new_section_chunk_id)+1):
                            idxt = np.flatnonzero(new_section_chunk_id == cid)
                            secnames = [self.section_list[kt] for kt in idxt]
                            prefix = secnames[0].split('_META')[0]
                            postfix = Aligner._hash_name('_'.join(secnames))
                            chnknm = '_META'.join((prefix, postfix))
                            new_chunknames.append(chnknm)
                            self._chunk_map[chnknm] = secnames
                        self._chunknames =  self._chunknames[:-1] + new_chunknames
                    self._section_chunk_id = new_section_chunk_id
        else:
            section_chunk_id, chunknames = Aligner.get_section_chunk_id(self._chunk_map, self.section_list)
            if np.any(section_chunk_id < 0):
                self._section_list = [s for cid, s in zip(section_chunk_id, self.section_list) if (cid >= 0)]
                section_chunk_id = section_chunk_id[section_chunk_id >= 0]
            self._section_chunk_id = section_chunk_id
            self._chunknames = chunknames


    @property
    def section_list(self):
        if (not hasattr(self, '_section_list')) or (self._section_list is None):
            self.get_section_list()
        return self._section_list


    @property
    def mesh_versions(self):
        if (not hasattr(self, '_mesh_versions')) or (self._mesh_versions is None):
            self.get_section_list()
        return self._mesh_versions


    def mesh_locations(self, secnames=None):
        if secnames is None:
            secnames = self.section_list
        mesh_dir_dict = dict(self._mesh_dirs_map)
        locs = {}
        for snm in secnames:
            mvid = self.mesh_versions.get(snm, None)
            if mvid is None:
                locs[snm] = None
            else:
                pdir = mesh_dir_dict[mvid]
                locs[snm] = storage.join_paths(pdir, snm+'.h5')
        return locs


    @property
    def mesh_versions_array(self):
        return np.array([self.mesh_versions.get(s, Aligner.UNALIGNED) for s in self.section_list])


    @property
    def chunk_map(self):
        if self._chunk_map is None:
            self.update_chunk_map()
        return self._chunk_map


    @property
    def section_chunk_id(self):
        if (not hasattr(self, '_section_chunk_id')) or (self._section_chunk_id is None):
            self.update_chunk_map()
        return self._section_chunk_id

    @property
    def chunk_names(self):
        if (not hasattr(self, '_chunknames')) or (self._chunknames is None):
            self.update_chunk_map()
        return self._chunknames


    @property
    def num_sections(self):
        return len(self.section_list)


    @property
    def junctional_sections(self):
        if (not hasattr(self, '_junctional_sections')) or (self._junctional_sections is None):
            section_chunk_id = self.section_chunk_id
            edt0 = distance_transform_cdt(np.diff(section_chunk_id) == 0)
            edt = np.insert(edt0, 0, 0)
            edt[:-1] = np.maximum(edt[:-1], edt[1:])
            if self._junction_width < 1:
                _, invindx, cnts = np.unique(section_chunk_id, return_inverse=True, return_counts=True)
                dis_thresh = cnts[invindx] * self._junction_width
            else:
                dis_thresh = self._junction_width
            self._in_junction = edt < dis_thresh
            juction_indx = np.flatnonzero( self._in_junction)
            self._junctional_sections = set(self.section_list[s] for s in juction_indx)
        return self._junctional_sections


    @property
    def junctional_sections_array(self):
        self.junctional_sections
        return self._in_junction


    def run(self, **kwargs0):
        num_workers = kwargs0.get('num_workers', 1)
        stiffness = kwargs0.pop('stiffness', None)
        chunked_to_depth = kwargs0.pop('chunked_to_depth', 0)
        residue_file = kwargs0.pop('residue_file', None)
        deform_target = kwargs0.pop('deform_target', None)
        kwargs = {
            'slide_window': {
                'num_workers': num_workers,
                'elastic_params':{
                    'stiffness_lambda': stiffness,
                    'deform_target': deform_target,
                    'tolerated_perturbation': 0.25,
                }
            }
        }
        config.merge_config(kwargs, kwargs0)
        worker_settings = kwargs.get('worker_settings', {})
        second_smooth = kwargs.get('second_smooth', True)
        if residue_file is None:
            residue_file = storage.join_paths(self._tform_dir, 'residue.csv')
        residues = {}
        if chunked_to_depth == 0:
            kwargs.setdefault('ensure_continuous', True)
            residues.update(self.window_align(**kwargs))
        else:
            kwargs.pop('no_slide', None)
            kwargs.pop('ensure_continuous', None)
            kwargs.pop('intermediate_dir', None)
            res0 = self.window_align(no_slide=True, ensure_continuous=True, intermediate_dir=self._predeform_dir, save_to_tform=True, **kwargs)
            residues.update(res0)
            Aligner.write_residue_file(res0, residue_file)
            if np.any(self.mesh_versions_array != Aligner.ALIGNED):
                res0, def0 = self.align_within_chunks(**kwargs)
                residues.update(res0)
                if (len(def0) > 0) and (stiffness is not None):
                    deform_target = np.array(list(def0.values()))
                    kwargs0['deform_target'] = np.median(deform_target) / 2
                Aligner.write_residue_file(res0, residue_file)
                meta_aligner_settings = {
                    "mesh_dir": self._meta_mesh_dir,
                    "tform_dir": self._meta_tform_dir,
                    "match_dir": self._meta_match_dir,
                    "match_name_delimiter": self._match_name_delimiter,
                    "mip_level": self._mip_level,
                    "junction_width": self._junction_width,
                    "default_chunk_size": self._default_chunk_size,
                    "logger": self._logger,
                    "section_list": self.chunk_names
                }
                meta_aligner = Aligner(**meta_aligner_settings)
                if chunked_to_depth is not None:
                    chunked_to_depth -= 1
                meta_aligner.run(stiffness=stiffness, chunked_to_depth=chunked_to_depth, **kwargs0)
                self.predeform_sections_by_chunk(num_workers=num_workers, worker_settings=worker_settings, commit_directly=(not second_smooth))
                if second_smooth:
                    locked_array = self.mesh_versions_array == Aligner.ALIGNED
                    locked_array = locked_array | (~self.junctional_sections_array)
                    specified_out_dirs = {s: None for s, lck in zip(self._section_list, locked_array) if lck}
                    kwargs_smooth = {
                        "stack_config":{
                            "lock_flags": locked_array,
                            "specified_out_dirs": specified_out_dirs
                        }
                    }
                    config.merge_config(kwargs_smooth, kwargs)
                    residues.update(self.window_align(no_slide=False, ensure_continuous=True, intermediate_dir=self._predeform_dir, save_to_tform=True, **kwargs_smooth))
                residues.update(self.window_align(no_slide=False, ensure_continuous=True, intermediate_dir=self._predeform_dir, save_to_tform=True, **kwargs))
        Aligner.write_residue_file(residues, residue_file)
        return residues


    def initialize_stack(self, intermediate_dir=None, **kwargs):
        kwargs.setdefault('mesh_dir', self._mesh_dir)
        kwargs.setdefault('match_dir', self._match_dir)
        kwargs.setdefault('section_list', self.section_list)
        kwargs.setdefault('logger', self._logger)
        kwargs.setdefault('mip_level', self._mip_level)
        lock_flags = kwargs.get('lock_flags', None)
        if lock_flags is None:
            kwargs.setdefault('lock_flags', self.mesh_versions_array==Aligner.ALIGNED)
        if intermediate_dir is not None:
            kwargs.setdefault('mesh_out_dir', intermediate_dir)
            kwargs.setdefault('tform_dir', self._tform_dir)
        else:
            kwargs.setdefault('mesh_out_dir', self._tform_dir)
        return Stack(**kwargs)


    def window_align(self, **kwargs):
        intermediate_dir = kwargs.pop('intermediate_dir', None)
        stack_config = kwargs.get('stack_config', {}).copy()
        slide_window = kwargs.get('slide_window', {}).copy()
        save_to_tform = kwargs.pop('save_to_tform', False)
        if save_to_tform:
            if isinstance(stack_config.get('specified_out_dirs', None), dict):
                specified_out_dirs = {s: self._tform_dir for s in self.section_list}
                specified_out_dirs.update(stack_config['specified_out_dirs'])
            else:
                specified_out_dirs = self._tform_dir
            stack_config['specified_out_dirs'] = self._tform_dir
        slide_window.setdefault('no_slide', kwargs.pop('no_slide', False))
        slide_window.setdefault('worker_settings', kwargs.pop('worker_settings', {}).copy())
        slide_window.setdefault('ensure_continuous', kwargs.pop('ensure_continuous', False))
        stack = self.initialize_stack(intermediate_dir=intermediate_dir, **stack_config)
        updated_sections, residues = stack.optimize_slide_window(**slide_window)
        if len(updated_sections) > 0:
            self._mesh_versions = None
        return residues


    def align_within_chunks(self, **kwargs):
        stack_config = kwargs.get('stack_config', {}).copy()
        slide_window = kwargs.get('slide_window', {}).copy()
        num_workers = slide_window.get('num_workers', 1)
        worker_settings = kwargs.pop('worker_settings', {})
        slide_window.setdefault('no_slide', False)
        worker_settings = slide_window.setdefault('worker_settings', worker_settings)
        slide_window.setdefault('ensure_continuous', False)
        changed_chunks, _ = self.resolve_chunk_version_differences(remove_file=True)
        if (self._chunk_map_file is not None) and ((len(changed_chunks) > 0) or (self._previous_chunk_map is None)):
            storage.makedirs(os.path.dirname(self._chunk_map_file))
            chunk_map_out = {ky:self.chunk_map[ky] for ky in self.chunk_names}
            with storage.File(self._chunk_map_file, 'w') as f:
                json.dump(chunk_map_out, f, indent=2)
        mesh_versions_array = self.mesh_versions_array
        lock_flags = mesh_versions_array == Aligner.ALIGNED
        stack_config['lock_flags'] = lock_flags # set the lock array so that muted matches are not filtered out during initialization
        stack = self.initialize_stack(include_chunk_dir=True, **stack_config)
        stack._specified_out_dirs = self._chunk_dir
        muted_match_list = stack.assign_section_to_chunks(self.chunk_map)
        storage.makedirs(self._chunk_dir)
        section_locked_for_chunk = mesh_versions_array > Aligner.UNALIGNED
        stack.update_lock_flags({s:l for s, l in zip(self.section_list, section_locked_for_chunk)})
        updated_sections, residues = stack.optimize_slide_window(**slide_window)
        section_name2id_lut = {s:k for k, s in enumerate(self.section_list)}
        section_chunk_id = self.section_chunk_id
        if len(updated_sections) > 0:
            self._mesh_versions = None
            sids = np.array([section_name2id_lut.get(s, None) for s in updated_sections if s in section_name2id_lut])
            cids = section_chunk_id[sids]
            updated_chunks = set(cids)
        else:
            updated_chunks = set()
        storage.makedirs(self._meta_mesh_dir)
        storage.makedirs(self._meta_tform_dir)
        storage.makedirs(self._meta_match_dir)
        n_chunk = int(np.max(section_chunk_id)+1)
        chunk_versions = np.full_like(mesh_versions_array, np.min(mesh_versions_array)-1, shape=n_chunk)
        np.maximum.at(chunk_versions, section_chunk_id, mesh_versions_array)
        chunk_lock_flags = chunk_versions == Aligner.ALIGNED
        chunk_matches = defaultdict(list)
        resolution = stack._resolution
        for mtch in muted_match_list:
            secnames = mtch.split(self._match_name_delimiter)
            sid0 = section_name2id_lut.get(secnames[0], None)
            sid1 = section_name2id_lut.get(secnames[-1], None)
            if (sid0 is None) or (sid1 is None):
                continue
            cid0 = section_chunk_id[sid0]
            cid1 = section_chunk_id[sid1]
            assert cid0 != cid1
            locked0 = chunk_lock_flags[cid0]
            locked1 = chunk_lock_flags[cid1]
            if locked0 and locked1:
                continue
            chunk_matches[(cid0, cid1)].append(mtch)
        args_list = []
        for cids, mtchnames in chunk_matches.items():
            cid0, cid1 = cids
            flip_flag = [False] * len(mtchnames)
            if (cid1, cid0) in chunk_matches:
                if cid1 < cid0:
                    continue
                else:
                    mtchnames_flip = chunk_matches[(cid1, cid0)]
                    flip_flag = flip_flag + [True] * len(mtchnames_flip)
                    mtchnames = mtchnames + mtchnames_flip
            outname = storage.join_paths(self._meta_match_dir, self.chunk_names[cid0] + self._match_name_delimiter + self.chunk_names[cid1] + '.h5')
            if (cid0 not in updated_chunks) and (cid1 not in updated_chunks) and storage.file_exists(outname):
                continue
            secnames = set(snm for mnm in mtchnames for snm in mnm.split(self._match_name_delimiter))
            mesh_list = self.mesh_locations(secnames=secnames)
            args_list.append([mtchnames, flip_flag, outname, mesh_list])
        if len(args_list) > 0:
            tfunc = partial(Aligner._merge_chunked_matches, match_dir=self._match_dir, resolution=resolution, match_name_delimiter=self._match_name_delimiter)
            for _ in submit_to_workers(tfunc, args=args_list, num_workers=num_workers, **worker_settings):
                pass
        relvant_chunk_ids = set().union(*chunk_matches)
        args_list = []
        for cid in relvant_chunk_ids:
            chnkname = self.chunk_names[cid]
            if chunk_lock_flags[cid]:
                outname = storage.join_paths(self._meta_tform_dir, chnkname+'.h5')
            else:
                outname = storage.join_paths(self._meta_mesh_dir, chnkname + '.h5')
            if storage.file_exists(outname) and (cid not in updated_chunks):
                continue
            secnames = self.chunk_map[chnkname]
            mesh_list = self.mesh_locations(secnames=secnames)
            m_init_dict = {'resolution': resolution, 'name': chnkname, 'uid':cid, 'locked': chunk_lock_flags[cid]}
            args_list.append([mesh_list, outname, m_init_dict])
        deformations = {}
        for df in submit_to_workers(Aligner._merge_chunked_meshes, args=args_list, num_workers=num_workers, **worker_settings):
            deformations.update(df)
        return residues, deformations


    def predeform_sections_by_chunk(self, **kwargs):
        num_workers = kwargs.pop('num_workers', 1)
        worker_settings = kwargs.get('worker_settings', {})
        commit_directly = kwargs.get('commit_directly', True)
        junctional_sections = self.junctional_sections
        mesh_versions = self.mesh_versions
        section_chunk_id = self.section_chunk_id
        chunk_aligned = self.mesh_versions_array == Aligner.CHUNK_ALIGNED
        n_chunk = int(np.max(section_chunk_id)+1)
        needed_deform = np.zeros(n_chunk, dtype=bool)
        np.maximum.at(needed_deform, section_chunk_id, chunk_aligned)
        chunks_to_process = np.flatnonzero(needed_deform)
        logger = logging.get_logger(self._logger)
        args_list = []
        args_list_j = []
        for cid in chunks_to_process:
            chnkname = self.chunk_names[cid]
            secnames0 = self.chunk_map[chnkname]
            meta_tform = storage.join_paths(self._meta_tform_dir, chnkname+'.h5')
            if not storage.file_exists(meta_tform):
                logger.warning(f'tranformation for meta section {meta_tform} not found')
                continue
            secnames = []
            secnames_j = []
            for snm in secnames0:
                if (mesh_versions.get(snm, None) == Aligner.CHUNK_ALIGNED):
                    if snm in junctional_sections:
                        secnames_j.append(snm)
                    else:
                        secnames.append(snm)
            if len(secnames) > 0:
                args_list.append([meta_tform, secnames])
            if len(secnames_j) > 0:
                args_list_j.append([meta_tform, secnames_j])
        storage.makedirs(self._predeform_dir)
        if len(args_list_j) > 0:
            tfunc = partial(Aligner._predeform_meshes, outdir=self._predeform_dir, srcdir=self._chunk_dir, flag=Aligner.PREDEFORMED)
            for res in submit_to_workers(tfunc, args=args_list_j, num_workers=num_workers, **worker_settings):
                self._mesh_versions.update(res)
        if len(args_list) > 0:
            if commit_directly:
                output_dir = self._tform_dir
                flag=Aligner.ALIGNED
            else:
                output_dir = self._predeform_dir
                flag=Aligner.PREDEFORMED
            tfunc = partial(Aligner._predeform_meshes, outdir=output_dir, srcdir=self._chunk_dir, flag=flag)
            for res in submit_to_workers(tfunc, args=args_list, num_workers=num_workers, **worker_settings):
                self._mesh_versions.update(res)


    def resolve_chunk_version_differences(self, remove_file=True):
        changed_chunks, changed_sections = Aligner.compare_chunk_maps(self._previous_chunk_map, self.chunk_map)
        if self._previous_chunk_map is not None:
            mesh_dir_list = (self._chunk_dir, self._predeform_dir)
            meta_dir_list = (self._meta_mesh_dir, self._meta_tform_dir)
            if remove_file:
                for mesh_dir in mesh_dir_list:
                    msh_list = storage.list_folder_content(storage.join_paths(mesh_dir, '*.h5'))
                    for secnm in msh_list:
                        if os.path.basename(secnm).replace('.h5', '') in changed_sections:
                            if storage.remove_file(secnm):
                                self._mesh_versions = None
                for mesh_dir in meta_dir_list:
                    msh_list = storage.list_folder_content(storage.join_paths(mesh_dir, '*.h5'))
                    for secnm in msh_list:
                        if os.path.basename(secnm).replace('.h5', '') in changed_chunks:
                            storage.remove_file(secnm)
                mtch_list = storage.list_folder_content(storage.join_paths(self._meta_match_dir, '*.h5'))
                for mtchnm in mtch_list:
                    snms = os.path.basename(mtchnm).replace('.h5', '').split(self._match_name_delimiter)
                    if (snms[0] in changed_chunks) or snms[-1] in changed_chunks:
                        storage.remove_file(mtchnm)
        return changed_chunks, changed_sections


    @staticmethod
    def get_section_chunk_id(chunk_map, section_list):
        indx_map = defaultdict(list)
        chunk_map_set = {k:set(v) for k, v in chunk_map.items()}
        for k, sname in enumerate(section_list):
            for cname, slist in chunk_map_set.items():
                if sname in slist:
                    indx_map[cname].append(k)
                    break
        cnames = sorted(indx_map, key=lambda s: np.min(indx_map[s]))
        section_chunk_id = np.full(len(section_list), -1, dtype=np.int32)
        for k, cnm in enumerate(cnames):
            section_chunk_id[indx_map[cnm]] = k
        return section_chunk_id, cnames


    @staticmethod
    def compare_chunk_maps(old_map, new_map):
        changed_chunks = []
        changed_sections = []
        if old_map is not None:
            old_secnames = set().union(*old_map.values())
            new_secnames = set().union(*new_map.values())
            common_secnames = old_secnames.intersection(new_secnames)
            new_map_breakdowns = {}
            for chknm, secnames in new_map.items():
                secname_shared = [s for s in secnames if s in common_secnames]
                if len(secname_shared) == 0:
                    continue
                new_map_breakdowns[tuple(secname_shared)] = (chknm, len(secname_shared)==len(secnames))
            for chknm, secnames0 in old_map.items():
                secname_shared = [s for s in secnames0 if s in common_secnames]
                if len(secname_shared) == 0:
                    continue
                paired = new_map_breakdowns.get(tuple(secname_shared), None)
                if paired is None:
                    changed_sections.extend(secname_shared)
                    changed_chunks.append(chknm)
                elif (len(secname_shared) != len(secnames0)) or (paired != (chknm, True)):
                    changed_chunks.append(chknm)
        changed_chunks, changed_sections = set(changed_chunks), set(changed_sections)
        return changed_chunks, changed_sections


    @staticmethod
    def write_residue_file(residues, filename):
        if (filename is None) or (len(residues) == 0):
            return
        res0 = {}
        if storage.file_exists(filename):
            with storage.File(filename, 'r') as f:
                lines = f.readlines()
            for line in lines:
                strlist = line.split(', ')
                if len(strlist) > 1:
                    res0[strlist[0]] = [float(dis) for dis in strlist[1:]]
        res0.update(residues)
        if len(res0) > 0:
            storage.makedirs(os.path.dirname(filename))
            mnames = sorted(list(res0.keys()))
            with storage.File(filename, 'w') as f:
                for ky in mnames:
                    strlist = [ky] + [str(s) for s in res0[ky]]
                    line = ', '.join(strlist)
                    f.write(line + '\n')


    @staticmethod
    def _hash_name(name, num_digits=9, collision_pool=None):
        hashed_str = hashlib.sha256(name.encode()).hexdigest()
        out_candidate = hashed_str[:num_digits]
        if collision_pool is not None:
            stt = 1
            while out_candidate in collision_pool:
                out_candidate = hashed_str[stt:(stt+num_digits)]
                if len(out_candidate) < num_digits:
                    raise RuntimeError('How could this still collide...')
                stt += 1
        return out_candidate


    @staticmethod
    def _predeform_meshes(tform_file, secnames, outdir, srcdir, flag=None):
        Mc = Mesh.from_h5(tform_file)
        res = {}
        if isinstance(outdir, str):
            outdir = [outdir] * len(secnames)
        for snm, odir in zip(secnames, outdir):
            outname = storage.join_paths(odir, snm +'.h5')
            if not storage.file_exists(outname, use_cache=True):
                srcname = storage.join_paths(srcdir, snm+'.h5')
                m0 = Mesh.from_h5(srcname)
                m0 = transform_mesh(m0, Mc, gears=(const.MESH_GEAR_MOVING, const.MESH_GEAR_MOVING))
                m0.anneal(gear=(const.MESH_GEAR_MOVING, const.MESH_GEAR_FIXED), mode=const.ANNEAL_COPY_EXACT)
                m0.save_to_h5(outname, save_material=True)
            if flag is not None:
                res[snm] = flag
        return res


    @staticmethod
    def _merge_chunked_meshes(mesh_list, outname, m_init_dict):
        resolution = m_init_dict.get('resolution')
        regions = []
        region_areas = 0
        num_tri = 0
        deformations = {}
        for meshname in mesh_list.values():
            if meshname is None:
                continue
            m0 = Mesh.from_h5(meshname)
            m0.change_resolution(resolution)
            v0 = m0.vertices(gear=const.MESH_GEAR_FIXED)
            v1 = m0.vertices(gear=const.MESH_GEAR_MOVING)
            dv = v1 - v0
            dv = dv - np.mean(dv, axis=0, keepdims=True)
            if np.any(dv != 0, axis=None):
                v0 = v0 - np.mean(v0, axis=0, keepdims=True)
                stiff_m, _ = m0.stiffness_matrix(gear=(const.MESH_GEAR_FIXED, const.MESH_GEAR_MOVING), continue_on_flip=True)
                df = (stiff_m.dot(dv.ravel()).dot(dv.ravel()) / (stiff_m.dot(v0.ravel()).dot(v0.ravel()))) ** 0.5
                deformations[os.path.basename(meshname)] = df
            reg = m0.shapely_regions(gear=const.MESH_GEAR_MOVING, offsetting=True)
            region_areas += reg.area
            num_tri += m0.num_triangles
            regions.append(shapely.convex_hull(reg))
        if len(regions) > 0:
            union_region = shapely.unary_union(regions)
            mesh_size = 2 * (region_areas / num_tri) ** 0.5
            M = Mesh.from_polygon_equilateral(union_region, mesh_size=mesh_size, **m_init_dict)
            M.save_to_h5(outname, save_material=True)
        return deformations


    @staticmethod
    def _merge_chunked_matches(match_names, flipped, outname, mesh_list, **kwargs):
        match_dir = kwargs.get('match_dir')
        resolution = kwargs.get('resolution')
        match_name_delimiter = kwargs.get('match_name_delimiter', '__to__')
        updated = 0
        XY0 = []
        XY1 = []
        WTS = []
        STRNS = []
        for mnm, flp in zip(match_names, flipped):
            mtchpath = storage.join_paths(match_dir, mnm+'.h5')
            mtch = read_matches_from_h5(mtchpath, target_resolution=resolution)
            xy0_i, xy1_i, wt, strain = mtch
            secnames = mnm.split(match_name_delimiter)
            m0 = Mesh.from_h5(mesh_list[secnames[0]])
            m0.change_resolution(resolution)
            m1 = Mesh.from_h5(mesh_list[secnames[1]])
            m1.change_resolution(resolution)
            tid0, B0 = m0.cart2bary(xy0_i, gear=const.MESH_GEAR_INITIAL, tid=None, extrapolate=True)
            tid1, B1 = m1.cart2bary(xy1_i, gear=const.MESH_GEAR_INITIAL, tid=None, extrapolate=True)
            xy0 = m0.bary2cart(tid0, B0, gear=const.MESH_GEAR_MOVING, offsetting=True)
            xy1 = m1.bary2cart(tid1, B1, gear=const.MESH_GEAR_MOVING, offsetting=True)
            if flp:
                XY0.append(xy1)
                XY1.append(xy0)
            else:
                XY0.append(xy0)
                XY1.append(xy1)
            WTS.append(wt)
            STRNS.append((strain, np.sum(wt)))
        if len(XY0) > 0:
            XY0 = np.concatenate(XY0, axis=0)
            XY1 = np.concatenate(XY1, axis=0)
            WTS = np.concatenate(WTS)
            STRNS = np.array(STRNS)
            strain = np.sum(STRNS[:,0] * STRNS[:,1]) / np.sum(STRNS[:,1])
            out_bname = os.path.basename(outname).replace('.h5', '')
            chnknames = out_bname.split(match_name_delimiter)
            with H5File(outname, 'w') as f:
                f.create_dataset('xy0', data=XY0, compression="gzip")
                f.create_dataset('xy1', data=XY1, compression="gzip")
                f.create_dataset('weight', data=WTS, compression="gzip")
                f.create_dataset('strain', data=strain)
                f.create_dataset('resolution', data=resolution)
                f.create_dataset('name0', data=str_to_numpy_ascii(chnknames[0]))
                f.create_dataset('name1', data=str_to_numpy_ascii(chnknames[1]))
            updated = np.sum(WTS)
        return {outname: updated}