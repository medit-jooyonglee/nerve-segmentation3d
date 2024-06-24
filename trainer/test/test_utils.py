import os
from trainer.utils import get_logger
from colorlog import ColoredFormatter
import logging
get_logger('df').info('dsf')
get_logger('df').error('dsf')


def get_logger_path():
    return ''


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
    if not os.path.exists(logger_path) and logger_path:
        os.makedirs(logger_path, exist_ok=True)
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

logger = initialize_logger('test')

logger.info('info')
logger.error('error')
logger.warning('warning')
