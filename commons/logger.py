import logging
import time
import os
from colorlog import ColoredFormatter
from typing import Union

# base directory for this project. may be parent directory of this-file
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOGGER_PATH = ''
RESOURCE_PATH = os.path.join(BASE_PATH, "resource")

__version__ = "1.1.0"
INIT_LOGGER_NAME = "AutoPlanning"
runtime_logger = []


def initialize_logger(logger_name, default_level=logging.INFO):

    log_format = (
        '%(asctime)s - '
        '%(name)s - '
        '%(funcName)s - '
        '%(log_color)s%(levelname)s - '
        '%(message)s'
    )
    # "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s"
    formatter = ColoredFormatter(
        log_format,
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'yellow',
            'FATAL': 'red',
        }
    )

    logger = logging.getLogger(logger_name)

    logger.setLevel(default_level)

    logger_path = get_logger_path()
    if not os.path.exists(logger_path):
        os.makedirs(logger_path)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(logger_path, "{}.log".format(logger_name)))
    fh.setLevel(default_level)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(default_level)
    # ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def set_logger_path(path):
    global LOGGER_PATH
    LOGGER_PATH = path


def get_logger_path():
    logger_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../logs')
    logger_path = os.path.abspath(logger_path)
    return LOGGER_PATH or logger_path


def get_runtime_logger() -> object:

    # packagename = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    packagename = INIT_LOGGER_NAME
    if len(runtime_logger) == 0:
        runtime_logger.append(packagename)
        initialize_logger(packagename)
    return logging.getLogger(packagename)


def get_runtime_handler():
    logger = get_runtime_logger()
    if logger.handlers:
        return logger.handlers


def change_runtime_logger_stream_level(level:Union[str, int]):
    if isinstance(level, str):
        level_str = level.lower()
        table = {
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR,
            'fatal': logging.FATAL,
            'critical': logging.CRITICAL,
        }
        level = table.get(level_str, logging.DEBUG)

    assert level in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]
    logger = get_runtime_logger()
    for hndler in logger.handlers:
        if isinstance(hndler, logging.StreamHandler):
            hndler.setLevel(level)
    logger.setLevel(level)


def test():
    packagename = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    print(packagename)

    logger = get_runtime_logger()
    logger.info("test")
    logger.fatal("test")
    logger.fatal("test")
    logger.critical("test")
    logger.info("test")


def get_pid_and_time_str():
    t1 = time.time()
    d = t1 - int(t1)
    us = int(d*1e6)
    pid = os.getpid()
    res = time.strftime('%Y%m%d%H%M%S')
    return 'pid[' + str(pid) + ']:' + res + '_' + str(us)

def add_handler(handlers):
    logger = get_runtime_logger()
    for handle in handlers:
        logger.addHandler(handle)

