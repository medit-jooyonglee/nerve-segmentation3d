from typing import Tuple
import time
import os
import json

from commons import get_runtime_logger
# from commons.version import version
# from version import __version__
import version

class AppConfigure(object):
    def __init__(self):
        self.time_out = 60 * 60 * 6
        self.interval = 60 * 5
        self.last_activation_time = time.strftime('%c', time.localtime())

        self.debug = False
        self.debug_path = 'd:/temp/rest_api'

        # use only fo debugging...
        self.memory_trace = False
        self.version = version.__version__

    def __repr__(self):
        return '{}'.format(self.__dict__)


def get_localappdata_path():
    """
    지정된 localappdata 경로를 가져온다.
    Returns
    -------

    """
    return os.path.join(os.getenv('localappdata'), 'ApREST')


def get_configure_filename():
    """
    지정된 localappdata configure 파일을 가져온다.
    Returns
    -------

    """
    os.makedirs(get_localappdata_path(), exist_ok=True)
    return os.path.join(get_localappdata_path(), 'config.json')


def init_create_config(filename):
    # filename = get_configure_filename()
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            config = AppConfigure()
            json.dump(config.__dict__, f, indent='\t')


def write_activation_time():
    """
    지정된 localappdata 경로에 활동 시간(activation)을 기록한다.
    Returns
    -------

    """

    filename = get_configure_filename()
    init_create_config(filename)

    config = read_app_configure()
    config.last_activation_time = time.strftime('%c', time.localtime())
    with open(filename, 'w') as f:
        json.dump(config.__dict__, f, indent='\t')


def read_app_configure(forced_different_version=False) -> AppConfigure:
    """
    %localappdata% 설정파일의 version과 비교하여 읽어오거나, 또는 class 원형을 읽어온다.
    version이 같을 경우는 로컬저장소의 파일을 읽어온다.(업데이트한다)
    Parameters
    ----------
    forced_different_version : bool,
        loca app-configure 의 버전과 비교해서 읽을지에 대한 옵션
    Returns
    -------

    """
    filename = get_configure_filename()

    logger = get_runtime_logger()
    config = AppConfigure()

    try:
        with open(filename, 'r') as f:
            res = json.load(f)

            if forced_different_version:
                # version 없을 경우 ''로 초기화
                res['version'] = res.get('version') or ''
                config.__dict__.update(res)
            else:
                # 버전 같을 때만 업데이트
                if config.version == res.get('version', ''):
                    config.__dict__.update(res)
    except Exception as e:
        logger.error(e.args[0])

    return config


def read_last_activation_time() -> str:
    """
    마지막 활동시간을 가져온다.
    Returns
    -------

    """
    filename = get_configure_filename()

    logger = get_runtime_logger()
    config = AppConfigure()

    try:
        with open(filename, 'r') as f:
            res = json.load(f)
            config.__dict__.update(res)
    except Exception as e:
        logger.error(e.args[0])

    return config.last_activation_time


def last_time_since_last_activity() -> Tuple[int, str]:
    """
    마지막 활동시간에서 경과한 시간을 sec 단위로 가져온다.
    Returns
        int 마지막 활동에서 경과한 시간
        str 마지막 활돛 시간
    -------

    """
    time_str = read_last_activation_time()
    t1 = time.strptime(time_str)
    delta = time.time() - time.mktime(t1)
    return delta, time_str
