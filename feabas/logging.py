import logging
import logging.handlers
import os
import time
from multiprocessing import Process, Manager, managers

from feabas import config, storage


_time_stamp = time.strftime("%Y%m%d%H%M")

LEVELS = {'DEBUG': logging.DEBUG, 'INFO': logging.INFO,
          'WARNING': logging.WARNING, 'ERROR': logging.ERROR,
          'NONE': None}

TMP_DIR = storage.LOCAL_TEMP_FOLDER

def _get_log_configs():
    conf = config.general_settings()
    log_dir = config.get_log_dir()
    logfile_level = LEVELS.get(conf.get('logfile_level', 'WARNING').upper(), logging.WARNING)
    console_level = LEVELS.get(conf.get('console_level', 'INFO').upper(), logging.INFO)
    archive_level = LEVELS.get(conf.get('archive_level', 'INFO').upper(), logging.INFO)
    log_conf = {
        'log_dir': log_dir,
        'logfile_level': logfile_level,
        'console_level': console_level,
        'archive_level': archive_level
    }
    return log_conf


log_conf = _get_log_configs()


class FileHandler(logging.FileHandler):
    def __init__(self, filename, **kwargs):
        driver, filename = storage.parse_file_driver(filename)
        self._file_driver = driver
        self._tgtname = filename
        if driver == 'gs':
            self._localname = os.path.join(TMP_DIR, filename.replace('gs://', ''))
        else:
            self._localname = filename
        os.makedirs(os.path.dirname(self._localname), exist_ok=True)
        super().__init__(self._localname, **kwargs)

    def close(self):
        super().close()
        if (self._file_driver == 'gs') and (storage.file_exists(self._localname)):
            blob = storage.GCP_get_blob(self._tgtname)
            blob.upload_from_filename(self._localname)
            os.remove(self._localname)


class QueueHandler(logging.handlers.QueueHandler):
    def __init__(self, queue):
        super().__init__(queue)
        self._is_local_queue = isinstance(queue, managers.BaseProxy)

    def enqueue(self, record):
        if self._is_local_queue:
            super().enqueue(record)
        else:   # in case it's dask distributed Queue
            self.queue.put(record)


def get_main_logger(logger_name):
    main_logger = logging.getLogger(logger_name)
    main_logger.setLevel(logging.WARNING)
    log_dir = log_conf['log_dir']
    logger_prefix = logger_name.replace('.', '_')
    formatter = logging.Formatter(fmt='%(asctime)s-%(levelname)s: %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S')
    if log_conf['console_level'] is not None:
        main_logger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_conf['console_level'])
        main_logger.addHandler(console_handler)
    if log_conf['archive_level'] is not None:
        main_logger.setLevel(logging.DEBUG)
        archive_dir = storage.join_paths(log_dir, 'archive')
        import socket
        hostname = socket.gethostname()
        hostname = hostname.split('.')[0]
        archivefile = storage.join_paths(archive_dir, _time_stamp + '_' + hostname + '_' + logger_prefix + '.log')
        archive_handler = FileHandler(archivefile, mode='a', delay=True)
        archive_handler.setFormatter(formatter)
        archive_handler.setLevel(log_conf['archive_level'])
        main_logger.addHandler(archive_handler)
    if log_conf['logfile_level'] is not None:
        main_logger.setLevel(logging.DEBUG)
        import socket
        hostname = socket.gethostname()
        hostname = hostname.split('.')[0]
        warnfile = storage.join_paths(log_dir, _time_stamp + '_' + hostname + '_' + logger_prefix + '.log')
        warn_handler = FileHandler(warnfile, mode='a', delay=True)
        warn_handler.setFormatter(formatter)
        warn_handler.setLevel(logging.WARNING)
        main_logger.addHandler(warn_handler)
    return main_logger


def listener_process(queue, logger_name):
    main_logger = get_main_logger(logger_name=logger_name)
    while True:
        try:
            message = queue.get()
        except EOFError:
            message = None
        if message is None:
            logging.shutdown()
            break
        main_logger.handle(message)


def initialize_main_logger(logger_name='log', mp=False):
    if mp:
        queue = Manager().Queue(-1)
        listener = Process(target=listener_process, args=(queue, logger_name))
        listener.start()
        return queue, listener
    else:
        logger = get_main_logger(logger_name=logger_name)
        return logger, None


def get_logger(logger_info):
    if isinstance(logger_info, logging.Logger):
        logger = logger_info
    elif logger_info is None:
        logger = logging.Logger('worker')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s-%(levelname)s: %(message)s'))
        logger.addHandler(handler)
    else:
        logger = logging.Logger('worker')
        logger.setLevel(logging.DEBUG)
        handler = QueueHandler(logger_info)
        logger.addHandler(handler)
    return logger


def terminate_logger(queue, listener):
    if listener is not None:
        queue.put(None)
        listener.join()
    else:
        logging.shutdown()
