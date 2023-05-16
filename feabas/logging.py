import logging
import logging.handlers
import os
import yaml
import time
from multiprocessing import Process, Manager

from feabas import path


_time_stamp = time.strftime("%Y%m%d%H%M")

LEVELS = {'DEBUG': logging.DEBUG, 'INFO': logging.INFO,
          'WARNING': logging.WARNING, 'ERROR': logging.ERROR,
          'NONE': None}

def _get_log_configs():
    config_file = os.path.join('configs', 'general_configs.yaml')
    if os.path.isfile(config_file):
        with open(config_file, 'r') as f:
            conf = yaml.safe_load(f)
    else:
        conf = {}
    log_dir = path.get_log_dir()
    logfile_level = LEVELS.get(conf.get('logfile_level', 'WARNING').upper(), logging.WARNING)
    console_level = LEVELS.get(conf.get('console_level', 'INFO').upper(), logging.INFO)
    archive_level = LEVELS.get(conf.get('archive_level', 'INFO').upper(), logging.INFO)
    os.makedirs(log_dir, exist_ok=True)
    log_conf = {
        'log_dir': log_dir,
        'logfile_level': logfile_level,
        'console_level': console_level,
        'archive_level': archive_level
    }
    return log_conf


log_conf = _get_log_configs()


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
        archive_dir = os.path.join(log_dir, 'archive')
        os.makedirs(archive_dir, exist_ok=True)
        archivefile = os.path.join(archive_dir, _time_stamp + '_' + logger_prefix + '.log')
        archive_handler = logging.FileHandler(archivefile, mode='a', delay=True)
        archive_handler.setFormatter(formatter)
        archive_handler.setLevel(log_conf['archive_level'])
        main_logger.addHandler(archive_handler)
    if log_conf['logfile_level'] is not None:
        main_logger.setLevel(logging.DEBUG)
        warnfile = os.path.join(log_dir, _time_stamp + '_' + logger_prefix + '.log')
        warn_handler = logging.FileHandler(warnfile, mode='a', delay=True)
        warn_handler.setFormatter(formatter)
        warn_handler.setLevel(logging.WARNING)
        main_logger.addHandler(warn_handler)
    return main_logger


def listener_process(queue, logger_name):
    main_logger = get_main_logger(logger_name=logger_name)
    while True:
        message = queue.get()
        if message is None:
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
        handler = logging.handlers.QueueHandler(logger_info)
        logger.addHandler(handler)
    return logger


def terminate_logger(queue, listener):
    if listener is not None:
        queue.put_nowait(None)
        listener.join()
