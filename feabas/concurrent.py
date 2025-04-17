from collections import defaultdict
from feabas import config, storage
import sys

PY_VERSION = tuple(sys.version_info)
DEFAUL_FRAMEWORK = config.parallel_framework()

def parse_inputs(args, kwargs):
    if args is None:
        args = defaultdict(list)
    if kwargs is None:
        kwargs = defaultdict(dict)
    N_args, N_kwargs = len(args), len(kwargs)
    N = max(N_args, N_kwargs)
    if (N_args == 1) and (N_kwargs > 1):
        val_a = args[0]
        args = defaultdict(lambda: val_a)
    if (N_kwargs == 1) and (N_args > 1):
        val_k = kwargs[0]
        kwargs = defaultdict(lambda: val_k)
    return N, args, kwargs


def submit_to_workers(func, args=None, kwargs=None, **settings):
    parallel_framework = settings.pop('parallel_framework', DEFAUL_FRAMEWORK)
    num_workers = settings.get('num_workers', 1)
    force_remote = settings.pop('force_remote', parallel_framework=='slurm')
    N, args_n, kwargs_n = parse_inputs(args, kwargs)
    if N == 0:
        return []
    from multiprocessing import current_process
    if current_process().daemon:
        num_workers = 1
        force_remote = False
    if ((num_workers == 1) or (N == 1)) and (not force_remote):
        for k in range(N):
            args_b = args_n[k]
            kwargs_b = kwargs_n[k]
            res = func(*args_b, **kwargs_b)
            yield res
    else:
        if parallel_framework == 'process':
            yield from submit_to_process_pool(func, args, kwargs, **settings)
        elif parallel_framework == 'thread':
            yield from submit_to_thread_pool(func, args, kwargs, **settings)
        elif parallel_framework == 'dask':
            yield from submit_to_dask_localcluster(func, args, kwargs, **settings)
        elif parallel_framework == 'slurm':
            yield from submit_to_dask_slurmcluster(func, args, kwargs, **settings)
        else:
            raise ValueError(f'unsupported worker type {type}')


def submit_to_process_pool(func, args=None, kwargs=None, **settings):
    """
    Python built-in concurrent multiprocessing backend
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import get_context
    num_workers = settings.get('num_workers', 1)
    max_tasks_per_child = settings.get('max_tasks_per_child', None)
    N, args, kwargs = parse_inputs(args, kwargs)
    if ((max_tasks_per_child is None) or (max_tasks_per_child == 1)) and (PY_VERSION[1]>10):
        futures = []
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=get_context('spawn'), max_tasks_per_child=max_tasks_per_child) as executor:
            for k in range(N):
                args_b = args[k]
                kwargs_b = kwargs[k]
                job = executor.submit(func, *args_b, **kwargs_b)
                futures.append(job)
            for job in as_completed(futures):
                res = job.result()
                yield res
    else:
        if max_tasks_per_child is None:
            batch_size = N
        else:
            batch_size = num_workers * max_tasks_per_child
        index0 = list(range(N))
        indices = [index0[k:(k+batch_size)] for k in range(0, N, batch_size)]
        for idx in indices:
            futures = []
            with ProcessPoolExecutor(max_workers=num_workers, mp_context=get_context('spawn')) as executor:
                for k in idx:
                    args_b = args[k]
                    kwargs_b = kwargs[k]
                    job = executor.submit(func, *args_b, **kwargs_b)
                    futures.append(job)
                for job in as_completed(futures):
                    res = job.result()
                    yield res


def submit_to_thread_pool(func, args=None, kwargs=None, **settings):
    """
    Python built-in concurrent multithreading backend
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    num_workers = settings.get('num_workers', 1)
    N, args, kwargs = parse_inputs(args, kwargs)
    futures = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for k in range(N):
            args_b = args[k]
            kwargs_b = kwargs[k]
            job = executor.submit(func, *args_b, **kwargs_b)
            futures.append(job)
        for job in as_completed(futures):
            res = job.result()
            yield res


def submit_to_dask_localcluster(func, args=None, kwargs=None, **settings):
    """
    Dask Local Cluster scheduler: dask.distributed.LocalCluster
    """
    from dask.distributed import LocalCluster, Client, as_completed
    num_workers = settings.get('num_workers', 1)
    max_tasks_per_child = settings.get('max_tasks_per_child', None)
    memory_limit = settings.get('memory_limit', 'auto')
    threads_per_worker = settings.get('threads_per_worker', 1)
    N, args, kwargs = parse_inputs(args, kwargs)
    index0 = list(range(N))
    if max_tasks_per_child is None:
        indices = [index0]
    else:
        batch_size = num_workers * max_tasks_per_child
        indices = [index0[k:(k+batch_size)] for k in range(0, N, batch_size)]
    for idx in indices:
        with LocalCluster(n_workers=num_workers, processes=True, threads_per_worker=threads_per_worker, memory_limit=memory_limit) as cluster:
            with Client(cluster) as client:
                futures = []
                for k in idx:
                    args_b = args[k]
                    kwargs_b = kwargs[k]
                    fut = client.submit(func, *args_b, **kwargs_b)
                    futures.append(fut)
                for fut in as_completed(futures):
                    yield fut.result()


def submit_to_dask_slurmcluster(func, args=None, kwargs=None, **settings):
    """
    Dask SLURMCluster scheduler: dask_jobqueue.SLURMCluster
    """
    from dask_jobqueue import SLURMCluster
    from dask.distributed import Client, as_completed
    num_workers = settings.pop('num_workers', 1)
    config_name = settings.pop('config_name', None)
    if (config_name is not None) and storage.file_exists(config_name):
        cluster_settings = {'config_name': config_name}
    else:
        cluster_settings = settings
    N, args, kwargs = parse_inputs(args, kwargs)
    with SLURMCluster(**cluster_settings) as cluster:
        cluster.scale(jobs=num_workers)
        futures = []
        with Client(cluster) as client:
            client.wait_for_workers(num_workers)
            for k in range(N):
                args_b = args[k]
                kwargs_b = kwargs[k]
                job = client.submit(func, *args_b, **kwargs_b)
                futures.append(job)
            for job in as_completed(futures):
                res = job.result()
                yield res
