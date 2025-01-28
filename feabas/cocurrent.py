from collections import defaultdict

def parse_inputs(args, kwargs):
    if args is None:
        args = defaultdict(list)
    if kwargs is None:
        kwargs = defaultdict(dict)
    N = max(len(args), len(kwargs))
    if len(args) == 1:
        args = defaultdict(lambda: args[0])
    if len(kwargs) == 1:
        kwargs = defaultdict(lambda: kwargs[0])
    return N, args, kwargs


def submit_to_workers(func, args=None, kwargs=None, **settings):
    mp_type = settings.pop('mp_type', 'builtin')
    num_workers = settings.get('num_workers', 1)
    force_remote = settings.pop('force_remote', False)
    N, args, kwargs = parse_inputs(args, kwargs)
    result = []
    if N == 0:
        return result
    if (num_workers == 1) and (not force_remote):
        for k in range(N):
            args_b = args[k]
            kwargs_b = kwargs[k]
            result.append(func(*args_b, **kwargs_b))
    else:
        if mp_type == 'builtin':
            result = submit_to_builtin_pool(func, args, kwargs, **settings)
        elif mp_type == 'dask':
            result = submit_to_dask_localcluster(func, args, kwargs, **settings)
        else:
            raise ValueError(f'unsupported worker type {type}')
    return result


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
    result = []
    if (max_tasks_per_child is None) or (max_tasks_per_child == 1):
        jobs = []
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=get_context('spawn'), max_tasks_per_child=max_tasks_per_child) as executor:
            for k in range(N):
                args_b = args[k]
                kwargs_b = kwargs[k]
                job = executor.submit(func, *args_b, **kwargs_b)
                jobs.append(job)
            for job in as_completed(jobs):
                result.append(job.result())
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
                    result.append(job.result())
    return result


def submit_to_dask_localcluster(func, args=None, kwargs=None, **settings):
    """
    Dask Local Cluster scheduler: dask.distributed.LocalCluster
    """
    from dask.distributed import LocalCluster, Client, WorkerPlugin
    num_workers = settings.get('num_workers', 1)
    max_tasks_per_child = settings.get('max_tasks_per_child', None)
    memory_limit = settings.get('memory_limit', 'auto')
    N, args, kwargs = parse_inputs(args, kwargs)
    result = []
    with LocalCluster(n_workers=num_workers, processes=True, memory_limit=memory_limit) as cluster:
        with Client(cluster, set_as_default=False) as client:
            if max_tasks_per_child is not None:
                class TaskLimit(WorkerPlugin):
                    def __init__(self, task_limit):
                        self._task_limit = task_limit
                        self._counter = 0
                    def setup(self, worker):
                        self._worker = worker
                    def transition(self, key, start, finish, **kwargs):
                        if finish == 'memory':
                            self._counter += 1
                            if self._counter >= self._task_limit:
                                self._worker.close()    # hopefully nanny will spin up a replacement
                                self._counter = 0
                task_limit_plugin = TaskLimit(task_limit=max_tasks_per_child)
                client.register_worker_plugin(task_limit_plugin)
            futures = []
            for k in range(N):
                args_b = args[k]
                kwargs_b = kwargs[k]
                fut = client.submit(func, *args_b, **kwargs_b)
                futures.append(fut)
            result = client.gather(futures)
    return result
