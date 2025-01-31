from collections import defaultdict
from feabas import config

DEFAUL_FRAMEWORK = config.parallel_framework()

def parse_inputs(args, kwargs):
    if args is None:
        args = defaultdict(list)
    if kwargs is None:
        kwargs = defaultdict(dict)
    N = max(len(args), len(kwargs))
    if len(args) == 1:
        val = args[0]
        args = defaultdict(lambda: val)
    if len(kwargs) == 1:
        val = kwargs[0]
        kwargs = defaultdict(lambda: val)
    return N, args, kwargs


def submit_to_workers(func, args=None, kwargs=None, **settings):
    parallel_framework = settings.pop('parallel_framework', DEFAUL_FRAMEWORK)
    num_workers = settings.get('num_workers', 1)
    force_remote = settings.pop('force_remote', False)
    N, args, kwargs = parse_inputs(args, kwargs)
    if N == 0:
        return []
    if (num_workers == 1) and (not force_remote):
        for k in range(N):
            args_b = args[k]
            kwargs_b = kwargs[k]
            res = func(*args_b, **kwargs_b)
            yield res
    else:
        if parallel_framework == 'builtin':
            yield from submit_to_builtin_pool(func, args, kwargs, **settings)
        elif parallel_framework == 'dask':
            yield from submit_to_dask_localcluster(func, args, kwargs, **settings)
        else:
            raise ValueError(f'unsupported worker type {type}')


def submit_to_builtin_pool(func, args=None, kwargs=None, **settings):
    """
    Python built-in concurrent backend
    """
    from concurrent.futures.process import ProcessPoolExecutor
    from concurrent.futures import as_completed
    from multiprocessing import get_context
    num_workers = settings.get('num_workers', 1)
    max_tasks_per_child = settings.get('max_tasks_per_child', None)
    N, args, kwargs = parse_inputs(args, kwargs)
    if (max_tasks_per_child is None) or (max_tasks_per_child == 1):
        jobs = []
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=get_context('spawn'), max_tasks_per_child=max_tasks_per_child) as executor:
            for k in range(N):
                args_b = args[k]
                kwargs_b = kwargs[k]
                job = executor.submit(func, *args_b, **kwargs_b)
                jobs.append(job)
            for job in as_completed(jobs):
                res = job.result()
                yield res
    else:
        batch_size = num_workers * max_tasks_per_child
        index0 = list(range(N))
        indices = [index0[k:(k+batch_size)] for k in range(0, N, batch_size)]
        for idx in indices:
            jobs = []
            with ProcessPoolExecutor(max_workers=num_workers, mp_context=get_context('spawn')) as executor:
                for k in idx:
                    args_b = args[k]
                    kwargs_b = kwargs[k]
                    job = executor.submit(func, *args_b, **kwargs_b)
                    jobs.append(job)
                for job in as_completed(jobs):
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
