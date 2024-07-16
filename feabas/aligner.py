from collections import defaultdict, OrderedDict
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures import as_completed
from multiprocessing import get_context
import numpy as np
import glob
import os
from scipy import sparse
import scipy.sparse.csgraph as csgraph
import yaml
import time

from feabas.mesh import Mesh
from feabas import dal
from feabas import logging
from feabas.spatial import scale_coordinates
from feabas.matcher import section_matcher
from feabas.optimizer import SLM
import feabas.constant as const
from feabas.common import str_to_numpy_ascii, Match, rearrange_section_order
from feabas.config import montage_resolution
from feabas.cloud import H5File


def read_matches_from_h5(match_name, target_resolution=None):
    with H5File(match_name, 'r') as f:
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
    if isinstance(conf, str) and conf.lower().endswith('.yaml'):
        with open(conf, 'r') as f:
            conf = yaml.safe_load(f)
    if 'matching' in conf:
        conf = conf['matching']
    elif conf is None:
        conf = {}
    elif not isinstance(conf, dict):
        raise TypeError('configuration type not supported.')
    match_name_delimiter = conf.get('match_name_delimiter', '__to__')
    working_mip_level = conf.get('working_mip_level', 0)
    resolution = montage_resolution() * (2 ** working_mip_level)
    loader_config = conf.get('loader_config', {}).copy()
    matcher_config = conf.get('matcher_config', {}).copy()
    secnames = os.path.splitext(os.path.basename(match_name))[0].split(match_name_delimiter)
    if 'cache_size' in loader_config and loader_config['cache_size'] is not None:
        loader_config['cache_size'] = loader_config['cache_size'] // (2 * matcher_config.get('num_workers',1))
    if 'cache_capacity' in loader_config and loader_config['cache_capacity'] is not None:
        loader_config['cache_capacity'] = loader_config['cache_capacity'] // (2 * matcher_config.get('num_workers',1))
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
            loader0 = dal.get_loader_from_json(loader0, mip=working_mip_level, **loader_config)
        if not isinstance(loader1, dal.AbstractImageLoader):
            loader1 = dal.get_loader_from_json(loader1, mip=working_mip_level, **loader_config)
    else:
        raise TypeError('loader input type not supported.')
    mesh0.change_resolution(resolution)
    mesh1.change_resolution(resolution)
    initial_matches = read_matches_from_h5(match_name, target_resolution=resolution)
    xy0, xy1, weight = section_matcher(mesh0, mesh1, loader0, loader1,
        initial_matches=initial_matches, **matcher_config)
    if xy0 is None:
        return 0
    else:
        with H5File(outname, 'w') as f:
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
  ## --------------------------- initialization ---------------------------- ##
    def __init__(self, section_list=None, match_list=None, **kwargs):
        self._mesh_dir = kwargs.get('mesh_dir', None)
        self._mesh_out_dir = kwargs.get('mesh_out_dir', self._mesh_dir)
        self._match_dir = kwargs.get('match_dir', None)
        if section_list is None:
            if self._mesh_dir is None:
                raise RuntimeError('mesh_dir not defined.')
            slist = glob.glob(os.path.join(self._mesh_dir, '*.h5'))
            if bool(slist):
                section_list = sorted([os.path.basename(s).replace('.h5', '') for s in slist])
            else:
                raise RuntimeError('no section found.')
            section_order_file = kwargs.get('section_order_file', os.path.join(self._mesh_dir, 'section_order.txt'))
            section_list = rearrange_section_order(section_list, section_order_file)[0]
        assert len(section_list) == len(set(section_list))
        self.section_list = tuple(section_list)
        self._mesh_cache_size = kwargs.get('mesh_cache_size', self.num_sections)
        self._link_cache_size = kwargs.get('link_cache_size', None)
        self._match_name_delimiter = kwargs.get('match_name_delimiter', '__to__')
        lock_flags = kwargs.get('lock_flags', None)
        mesh_cache = kwargs.get('mesh_cache', {})
        link_cache = kwargs.get('link_cache', {})
        mip_level = kwargs.get('mip_level', 0)
        self._resolution = montage_resolution() * (2 ** mip_level)
        self._mesh_cache = OrderedDict()
        self._mesh_cache.update(mesh_cache)
        self._link_cache = OrderedDict()
        self._link_cache.update(link_cache)
        self.lock_flags = defaultdict(lambda: False)
        if isinstance(lock_flags, dict):
            self.lock_flags.update(lock_flags)
        elif isinstance(lock_flags, (tuple, list, np.ndarray)):
            self.lock_flags.update({nm: flg for nm, flg in zip(self.section_list, lock_flags)})
        if bool(self._mesh_cache):
            self.normalize_mesh_lock_status()
            self.normalize_mesh_resoltion()
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
        self.save_overflow = kwargs.get('save_overflow', True)
        self._logger = None


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
        init_dict['match_dir'] = self._match_dir
        match_list = self.filtered_match_list(secnames=secnames, check_lock=check_lock)
        section_list = self.filter_section_list_from_matches(match_list)
        init_dict['section_list'] = section_list
        init_dict['match_list'] = match_list
        init_dict['lock_flags'] = {s: self.lock_flags[s] for s in section_list}
        init_dict['resolution'] = self._resolution
        init_dict['mesh_cache_size'] = self._mesh_cache_size
        init_dict['link_cache_size'] = self._link_cache_size
        if include_cache:
            init_dict['mesh_cache'] = {s: self._mesh_cache[s] for s in section_list if s in self._mesh_cache}
            init_dict['link_cache'] = {s: self._link_cache[s] for s in match_list if s in self._link_cache}
        init_dict.update(kwargs)
        return init_dict


  ## --------------------------- meshes & matches -------------------------- ##
    def get_mesh(self, secname):
        if not isinstance(secname, str):
            secname = self.section_list[int(secname)]   # indexing by id
        if secname in self._mesh_cache:
            self._mesh_cache.move_to_end(secname, last=True)
            return self._mesh_cache[secname]
        elif self._mesh_dir is None:
            raise RuntimeError('mesh_dir not defined.')
        else:
            meshpath = os.path.join(self._mesh_out_dir, secname+'.h5')
            if not os.path.isfile(meshpath):
                meshpath = os.path.join(self._mesh_dir, secname+'.h5')
            if not os.path.isfile(meshpath):
                raise RuntimeError(f'{meshpath} not found.')
            uid = self.secname_to_id(secname)
            locked = self.lock_flags[secname]
            M = Mesh.from_h5(meshpath, uid=uid, locked=locked, name=secname)
            M.change_resolution(self._resolution)
            Ms = M.divide_disconnected_mesh(save_material=True)
            if self._mesh_cache_size > 0:
                self._mesh_cache[secname] = Ms
            while len(self._mesh_cache) > self._mesh_cache_size:
                self.dump_first_mesh()
            return Ms


    def flush_meshes(self):
        while bool(self._mesh_cache):
            self.dump_first_mesh()


    def dump_first_mesh(self):
        """self._mesh_cache is a FIFO cache"""
        cached_name, cached_Ms = self._mesh_cache.popitem(last=False)
        rel_match_names = self.secname_to_matchname_mapper[cached_name]
        if (self._mesh_out_dir is not None) and self.save_overflow:
            anchored_meshes = [m for m in cached_Ms if not m.is_outcast]
            if len(anchored_meshes) != len(cached_Ms):
                logger = logging.get_logger(self._logger)
                if len(anchored_meshes) == 0:
                    self.lock_flags[cached_name] = False
                    logger.warning(f'{cached_name}: not anchored.')
                else:
                    logger.warning(f'{cached_name}: partially not anchored.')
            if len(anchored_meshes) > 0:
                cached_M = Mesh.combine_mesh(anchored_meshes, save_material=True)
                outname = os.path.join(self._mesh_out_dir, cached_name+'.h5')
                cached_M.save_to_h5(outname, vertex_flags=const.MESH_GEARS, save_material=True)
        for matchname in rel_match_names:
            self.dump_link(matchname)


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
            matchpath = os.path.join(self._match_dir, matchname+'.h5')
            if not os.path.isfile(matchpath):
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
        if secnames is None:
            secnames = self.section_list
        match_list = self.filtered_match_list(secnames=secnames, check_lock=True)
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
        return optm


    def reduced_stack(self, mesh_reduction_factor=0, gear=const.MESH_GEAR_INITIAL, **kwargs):
        """
        Generate a stack with a coarse version of each mesh from the original
        aligner. The whole stack will be saved in RAM at the same time, so make
        sure the mesh is coarse enough. Args:
            mesh_reduction_factor(float): an be 0~1, with smaller number
                indicates coarser meshing. set 0 to use affine.
            gear: which gear to use when generating the coarse stack.
        """
        pass


    def optimize_slide_window(self, **kwargs):
        num_workers = kwargs.pop('num_workers', 1)
        locked_section = kwargs.pop('locked_section', None)
        start_loc = kwargs.pop('start_loc', 'L') # L or R or M
        window_size = kwargs.get('window_size', None)
        func_name = 'optimize_slide_window'
        if kwargs.get('logger', None) is not None:
            self._logger = kwargs['logger']
        if (window_size is None) or window_size > self.num_sections:
            window_size = self.num_sections
            start_loc = 'L'
            buffer_size = 0
        else:
            buffer_size = kwargs.get('buffer_size', window_size//4)
        if locked_section is not None:
            self.unlock_all()
            if not isinstance(locked_section, str):
                locked_section = self.section_list[int(locked_section)]   # indexing by id
            self.update_lock_flags({locked_section: True})
            locked_idx = self.secname_to_id(locked_section)
            init_idx0 = locked_idx - window_size//2
            if (init_idx0 + window_size) >= self.num_sections:
                init_idx0 = self.num_sections - window_size
                start_loc = 'R'
            elif init_idx0 <= 0:
                init_idx0 = 0
                start_loc = 'L'
            else:
                start_loc = 'M'
        elif start_loc == 'L':
            init_idx0 = 0
        elif start_loc == 'R':
            init_idx0 = self.num_sections - window_size
        elif start_loc == 'M':
            init_idx0 = (self.num_sections - window_size)//2
        costs = {}
        if start_loc == 'L':
            indices0 = np.arange(init_idx0, self.num_sections, window_size-buffer_size)
            for idx0 in indices0:
                seclst = self.section_list[init_idx0:(idx0+window_size)]
                if np.all([self.lock_flags[s] for s in seclst]):
                    continue
                cst = self.optimize_section_list(seclst, **kwargs)
                if cst is not None:
                    costs.update(cst)
                if buffer_size > 0:
                    self.update_lock_flags({s:True for s in seclst[:-buffer_size]})
                else:
                    self.update_lock_flags({s:True for s in seclst})
            else:
                self.flush_meshes()
        elif start_loc == 'R':
            indices1 = np.arange(self.num_sections, 0, buffer_size-window_size)
            for idx1 in indices1:
                idx0 = max(0, idx1 - window_size)
                seclst = self.section_list[idx0:]
                if np.all([self.lock_flags[s] for s in seclst]):
                    continue
                cst = self.optimize_section_list(seclst, **kwargs)
                if cst is not None:
                    costs.update(cst)
                if buffer_size > 0:
                    self.update_lock_flags({s:True for s in seclst[buffer_size:]})
                else:
                    self.update_lock_flags({s:True for s in seclst})
            else:
                self.flush_meshes()
        else:
            assert window_size > 2 * buffer_size + 1
            seclst = self.section_list[init_idx0:(init_idx0+window_size)]
            cst = self.optimize_section_list(seclst, **kwargs)
            if cst is not None:
                costs.update(cst)
            if buffer_size > 0:
                sections_to_lock = seclst[buffer_size:-buffer_size]
            else:
                sections_to_lock = seclst
            self.update_lock_flags({s:True for s in sections_to_lock})
            self.flush_meshes()
            kwargs.setdefault('need_anchor', True)
            lock_id0, lock_id1 = self.secname_to_id(sections_to_lock[0]), self.secname_to_id(sections_to_lock[-1])
            matchlist_l = self.filtered_match_list(secnames=self.section_list[lock_id0:], check_lock=True)
            seclist_l = self.filter_section_list_from_matches(match_list=matchlist_l)
            kwargs_l = kwargs.copy()
            init_dict_l = self.init_dict(secnames=seclist_l, check_lock=True)
            kwargs_l['start_loc'] = 'L'
            matchlist_r = self.filtered_match_list(secnames=self.section_list[:lock_id1], check_lock=True)
            seclist_r = self.filter_section_list_from_matches(match_list=matchlist_r)
            init_dict_r = self.init_dict(secnames=seclist_r, check_lock=True)
            kwargs_r = kwargs.copy()
            kwargs_r['start_loc'] = 'R'
            if num_workers > 1:
                jobs = []
                with ProcessPoolExecutor(max_workers=2, mp_context=get_context('spawn')) as executor:
                    jobs.append(executor.submit(Stack.subprocess_optimize_stack, init_dict_l, func_name, **kwargs_l))
                    jobs.append(executor.submit(Stack.subprocess_optimize_stack, init_dict_r, func_name, **kwargs_r))
                    for job in as_completed(jobs):
                        costs.update(job.result())
            else:
                cst = Stack.subprocess_optimize_stack(init_dict_l, func_name, **kwargs_l)
                costs.update(cst)
                cst = Stack.subprocess_optimize_stack(init_dict_r, func_name, **kwargs_r)
                costs.update(cst)
        return costs


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
            return residue
        cost = None
        t0 = time.time()
        if optimize_rigid:
            optm.optimize_affine_cascade(target_gear=target_gear, **rigid_params)
            if target_gear != const.MESH_GEAR_FIXED:
                optm.anneal(gear=(target_gear, const.MESH_GEAR_FIXED), mode=const.ANNEAL_CONNECTED_RIGID)
        if optimize_elastic:
            if 'callback_settings' in elastic_params:
                elastic_params['callback_settings'].setdefault('early_stop_thresh', montage_resolution() / self._resolution)
            cost = optm.optimize_elastic(target_gear=target_gear, **elastic_params)
            if (residue_mode is not None) and (residue_len > 0):
                if residue_mode == 'huber':
                    optm.set_link_residue_huber(residue_len)
                else:
                    optm.set_link_residue_threshold(residue_len)
                weight_modified, _ = optm.adjust_link_weight_by_residue(gear=(target_gear, target_gear))
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
        if not hasattr(self, '_name_id_lut'):
            self._name_id_lut = {name: k for k, name in enumerate(self.section_list)}
        return self._name_id_lut.get(secname, -1)


    def matchname_to_secnames(self, matchname):
        return matchname.split(self._match_name_delimiter)


    @property
    def section_connection_matrix(self):
        if not hasattr(self, '_section_connection_matrix'):
            edges = np.array([s for s in self.matchname_to_secids_mapper.values()])
            idx0 = edges[:,0]
            idx1 = edges[:,1]
            V = np.ones_like(idx0, dtype=bool)
            Nsec = self.num_sections
            A = sparse.csr_matrix((V, (idx0, idx1)), shape=(Nsec, Nsec))
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
    def secname_to_matchname_mapper(self):
        if not hasattr(self, '_secname_to_matchname_mapper'):
            self._secname_to_matchname_mapper = defaultdict(set)
            for matchname in self.match_list:
                names = self.matchname_to_secnames(matchname)
                self._secname_to_matchname_mapper[names[0]].add(matchname)
                self._secname_to_matchname_mapper[names[1]].add(matchname)
        return self._secname_to_matchname_mapper


    @property
    def matchname_to_secids_mapper(self):
        if not hasattr(self, '_matchname_to_secids_mapper'):
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
    def subprocess_optimize_stack(init_dict, process_name, **kwargs):
        stack = Stack(**init_dict)
        func = getattr(stack, process_name)
        return func(**kwargs)
