import importlib
import logging
import sys
import os
import shutil
import torch
import ml_collections
import GPUtil
from colorlog import ColoredFormatter

import psutil

loggers = {}

LOGGER_PATH = ''

def get_logger_path():
    logger_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../logs')
    logger_path = os.path.abspath(logger_path)
    return LOGGER_PATH or logger_path


def get_logger(name, level=logging.INFO):
    global loggers
    if loggers.get(name) is not None:
        return loggers[name]
    else:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        # Logging to console
        log_format = '%(asctime)s %(log_color)s%(levelname)s [%(threadName)s]  %(name)s - %(message)s'
        # log_format = (
        #     '%(asctime)s - '
        #     '%(name)s - '
        #     '%(funcName)s - '
        #     '%(log_color)s%(levelname)s - '
        #     '%(message)s'
        # )
        formatter = ColoredFormatter(
            log_format,
            datefmt=None,
            reset=True,
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'yellow',
                'FATAL': 'red',
            }
        )

        # stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)

        logger_path = get_logger_path()
        if not os.path.exists(logger_path):
            os.makedirs(logger_path)

        fh = logging.FileHandler(os.path.join(logger_path, "{}.log".format(name)))
        fh.setLevel(level)
        fh.setFormatter(formatter)
        # formatter = logging.Formatter(
        #     )
        #
        # fh = logging.FileHandler(os.path.join('', "{}.log".format(name)))
        # fh.setLevel(level)
        logger.addHandler(fh)
        logger.addHandler(stream_handler)

        loggers[name] = logger

        return logger


def get_class(class_name, modules):
    for module in modules:
        try:
            m = importlib.import_module(module)
        except Exception as e:
            logger = get_logger('error')
            logger.error(e.args)
            continue
            # return None
        clazz = getattr(m, class_name, None)
        if clazz is not None:
            return clazz
    raise RuntimeError(f'Unsupported class: {class_name}')


def get_function(function_name, modules):
    for module in modules:
        m = importlib.import_module(module)
        func = getattr(m, function_name, None)
        if func is not None:
            return func
    raise RuntimeError(f'unsupported function name:{function_name}')



def create_class(clazz, params):
    logger = get_logger(__name__)
    try:
        return clazz(**params)
    except Exception as e:
        logger.warning(e.args)

    try:
        arg = ml_collections.ConfigDict(params)
        return clazz(arg)
    except Exception as e:
        logger.warning(e.args)

    try:
        return clazz()
    except Exception as e:
        raise NotImplementedError(e.args)


def save_checkpoint(state, is_best, checkpoint_dir):
    """Saves model and training parameters at '{checkpoint_dir}/last_checkpoint.pytorch'.
    If is_best==True saves '{checkpoint_dir}/best_checkpoint.pytorch' as well.

    Args:
        state (dict): contains model's state_dict, optimizer's state_dict, epoch
            and best evaluation metric value so far
        is_best (bool): if True state contains the best model seen so far
        checkpoint_dir (string): directory where the checkpoint are to be saved
    """

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    last_file_path = os.path.join(checkpoint_dir, 'last_checkpoint.pytorch')
    torch.save(state, last_file_path)
    if is_best:
        best_file_path = os.path.join(checkpoint_dir, 'best_checkpoint.pytorch')
        shutil.copyfile(last_file_path, best_file_path)


def load_checkpoint(checkpoint_path, model, optimizer=None,
                    model_key='model_state_dict', optimizer_key='optimizer_state_dict', strict=True):
    """Loads model and training parameters from a given checkpoint_path
    If optimizer is provided, loads optimizer's state_dict of as well.

    Args:
        checkpoint_path (string): path to the checkpoint to be loaded
        model (torch.nn.Module): model into which the parameters are to be copied
        optimizer (torch.optim.Optimizer) optional: optimizer instance into
            which the parameters are to be copied

    Returns:
        state
    """
    if not os.path.exists(checkpoint_path):
        raise IOError(f"Checkpoint '{checkpoint_path}' does not exist")

    state = torch.load(checkpoint_path, map_location='cpu')
    res = model.load_state_dict(state[model_key], strict=strict)
    if res.missing_keys and res.unexpected_keys:
        print(res)
    if optimizer is not None:
        optimizer.load_state_dict(state[optimizer_key])

    return state


class RunningAverage:
    """Computes and stores the average
    """

    def __init__(self, name=''):
        self.name = name
        self.count = 0
        self.sum = 0
        self._avg = 0

        self.sum_dict = dict()
        self._avg_dict = dict()
        self.count_dict  = dict()

        self._scalar = None # type: bool

    @property
    def avg_dict(self):
        if self._scalar:
            return {
                self.name: self.avg
            }
        else:
            return self._avg_dict

    @property
    def mean_avg_str(self):
        # assert self._scalar is False
        if self._scalar:
            return self._avg
        else:
            out_str = ''
            for i, (k, v) in enumerate(self.avg_dict.items()):
                out_str += '{}:{:.4f}'.format(k, v)
                if i < (len(self.avg_dict) - 1):
                    out_str += '    '
            return out_str

    @property
    def mean_avg(self):

        val = 0.
        for v in self.avg_dict.values():
            val += v
        if len(self.avg_dict) > 0:
            val /= len(self.avg_dict)
        return val

    @property
    def avg(self):
        if self._scalar:
            val = self._avg
        else:
            val = self.mean_avg
        return val
        # return '{:.5f}'.format(val)

    def _safe_init(self, some_dict, key):
        if not (key in some_dict):
            some_dict[key] = 0

    def update(self, value, n=1):
        if isinstance(value, dict):
            self._scalar = False
            # self.count += n
            for k, v in value.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                for som_dict in [self.sum_dict, self.avg_dict, self.count_dict]:
                    self._safe_init(som_dict, k)
                self.count_dict[k] += n
                self.sum_dict[k] += v * n
                self._avg_dict[k] = self.sum_dict[k] / self.count_dict[k]

        else:
            if isinstance(value, torch.Tensor):
                value = value.item()
            self._scalar = True
            self.count += n
            self.sum += value * n
            self._avg = self.sum / self.count

    def __repr__(self):
        if self._scalar:
            return self.avg
        else:
            out_str = ''
            for i, (k, v) in enumerate(self.avg_dict.items()):
                out_str += '{}:{:.4f}'.format(k, v)
                if i < (len(self.avg_dict) -1):
                    out_str += '    '
            return out_str


def add_unit(mem: float) -> str:
    if mem > 1024:
        mem = round(mem / 1024, 2)
        mem = f"{mem}GiB"
    else:
        mem = round(mem, 2)
        mem = f"{mem}MiB"
    return mem


def print_gpu_usage(name=''):
    """
    https://hanryang1125.tistory.com/49
    :param name:
    :return:
    """
    # if name:
    #     print(f'--------------------------------{name}--------------------------------')
    for gpu in GPUtil.getGPUs():
        gpu_util = f"{gpu.load}%"
        mem_total = add_unit(gpu.memoryTotal)
        mem_used = add_unit(gpu.memoryUsed)
        mem_used_percent = f"{round(gpu.memoryUtil * 100, 2)}%"
        print(f"[{name}] - ID: {gpu.id}, Util: {gpu_util}, Memory: {mem_used} / {mem_total} ({mem_used_percent})")


def print_cpu_usage():
    """
    https://www.geeksforgeeks.org/how-to-get-current-cpu-and-ram-usage-in-python/
    :return:
    """
    # Getting loadover15 minutes
    load1, load5, load15 = psutil.getloadavg()

    cpu_usage = (load15 / os.cpu_count()) * 100

    print("The CPU usage is : ", cpu_usage)


def print_ram_usage():
    # Getting all memory using os.popen()
    total_memory, used_memory, free_memory = map(
        int, os.popen('free -t -m').readlines()[-1].split()[1:])

    # Memory usage
    print("RAM memory % used:", round((used_memory / total_memory) * 100, 2))


def print_system_memory():
    # Importing the library
    import psutil

    # Getting % usage of virtual_memory ( 3rd field)
    print('RAM memory % used:', psutil.virtual_memory()[2])
    # Getting usage of virtual_memory in GB ( 4th field)
    print('RAM Used (GB):', psutil.virtual_memory()[3] / 1000000000)


def print_system_usage():
    print_gpu_usage()
    print_cpu_usage()
    print_ram_usage()

if __name__ == '__main__':
    print_system_usage()
