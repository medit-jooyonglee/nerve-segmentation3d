import os
from .logger import *
from .common_utils import timefn, timefn2, get_localappdata_path
from .teethConfigure import get_teeth_color_table


# base work-directory
app_base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RELEASE = 'RELEASE'
DEBUG = 'DEBUG'
MODE = DEBUG

def get_torch_memory_allocated():
    import torch

    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
    import tarfile
    with tarfile.open() as f:
        f.add()