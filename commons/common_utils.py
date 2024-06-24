import os
import re
import sys
import time
import numpy as np
import logging
from .logger import get_runtime_logger
from . import get_runtime_logger
# from teethnet import common

LOGGER_NAME = 'teethnet'
log_level = logging.DEBUG

def timefn(fn):
    # @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        # logger = logging.getLogger(LOGGER_NAME)
        if log_level in [logging.DEBUG]:
            logger = get_runtime_logger()
            time_str = "@timefn: {} took {} secons".format(fn.__qualname__, t2 - t1)
            if logger is not None:
                logger.info(time_str)
            else:
                print(time_str)
        return result
    return measure_time


def timefn2(fn):
    # @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        # logger = logging.getLogger(LOGGER_NAME)
        if log_level in [logging.DEBUG]:
            logger = get_runtime_logger()
            # time_str = "@timefn: {} took {} secons".format(fn.__qualname__, t2 - t1)
            time_str = "@timefn: {} took {} secons".format(fn.__name__, t2 - t1)
            if logger is not None:
                logger.info(time_str)
            else:
                print(time_str)
        return result
    return measure_time



def read_label_itk_snap(filename):
    """
    read label from format-file of the ITK-SNAP label-editor

    :return:
        label_rgb_table : label(volume label value) - rgb-value
        label_name_table: key:label, value: fdi number(volume label value) - tooh FDI number
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    # filename = "D:/DataSet/teeth_segmentation/teethLabel.txt"
    tag_list = ['label', 'r', 'g', 'b', 'a', 'b', 'c', 'name']
    label_rgb_table = dict()
    label_name_table = dict()

    for line in open(filename):
        li = line.strip()
        if not li.startswith("#"):
            name_list = line.strip().split('"')
            name_tag = line.strip().split()

            assert len(name_list) == 3 and len(name_tag) >= 8, "{}, {}".format(len(name_list), len(name_tag))

            names = name_list[1]

            label = int(name_tag[0])
            rgb = [int(c) for c in name_tag[1:4]]
            label_rgb_table[label] = rgb

            if names.find("Tooth") < 0:
                if names.find("Unknown") < 0:
                    instance_label = 0
                else:
                    instance_label = 1
            else:
                instance_label = int(names.split("Tooth")[1])

            if names == "Mandible":
                instance_label = 49

            if names == "Maxilla":
                instance_label = 50


            label_name_table[label] = instance_label

    return label_rgb_table, label_name_table


# def trim_boxes(boxes, shape):
#     shape = np.asarray(shape)
#     assert boxes.shape[1] == shape.size * 2



def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask > 0,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image



############################################################
#  Data Formatting
############################################################

def compose_image_meta(image_id, original_image_shape, image_shape,
                       window, scale, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array.

    image_id: An int ID of the image. Useful for debugging.
    original_image_shape: [H, W, C] before resizing or padding.
    image_shape: [H, W, C] after resizing and padding
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
    scale: The scaling factor applied to the original image (float32)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array(
        [image_id] +                  # size=1
        list(original_image_shape) +  # size=3
        list(image_shape) +           # size=3
        list(window) +                # size=6 (z1, y1, x1, z2, y2, x2) in image cooredinates
        [scale] +                     # size=1
        list(active_class_ids)        # size=num_classes
    )
    return meta



def parse_image_meta(meta):
    """Parses an array that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_CLASSES

    Returns a dict of the parsed values.
    """
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:13]  # (z1, y1, x1, z2, y2, x2) window of image in in pixels
    scale = meta[:, 13]
    active_class_ids = meta[:, 14:]
    return {
        "image_id": image_id.astype(np.int32),
        "original_image_shape": original_image_shape.astype(np.int32),
        "image_shape": image_shape.astype(np.int32),
        "window": window.astype(np.int32),
        "scale": scale.astype(np.float32),
        "active_class_ids": active_class_ids.astype(np.int32),
    }


def parse_image_meta_graph(meta):
    """Parses a tensor that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_CLASSES

    Returns a dict of the parsed tensors.
    """
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:13]  # (z1, y1, x1, z2, y2, x2) window of image in in pixels
    scale = meta[:, 13]
    active_class_ids = meta[:, 14:]
    return {
        "image_id": image_id,
        "original_image_shape": original_image_shape,
        "image_shape": image_shape,
        "window": window,
        "scale": scale,
        "active_class_ids": active_class_ids,
    }


def _split_savename_timestr(dcmpath):
    workname = os.path.basename(os.path.dirname(dcmpath))

    return "{}_{}.vtk".format(workname, time.strftime("%Y%m%d%H%M%S"))


def gen_savename(dcmpath):

    base_name = os.path.dirname(dcmpath)
    save_name = _split_savename_timestr(dcmpath)
    save_fullname = os.path.join(base_name, save_name)
    return save_fullname


def get_localappdata_path():
    """
    지정된 localappdata 경로를 가져온다.
    Returns
    -------

    """
    path = os.path.join(os.getenv('localappdata'), 'ApREST')
    os.makedirs(path, exist_ok=True)
    return path


def get_stack_path(size,  start=0):
    # start = 2
    # subname = '_'.join([sys._getframe(i).f_code.co_name for i in reversed(range(start, start + size))])
    try:
        return [sys._getframe(i).f_code.co_name for i in reversed(range(start, start + size))]
    except Exception as e:
        return ['']

def clean_fname(fname:str, replace='_'):
    """
    https://stackoverflow.com/questions/1033424/how-to-remove-bad-path-characters-in-python
    Parameters
    ----------
    fname :
    replace :

    Returns
    -------

    """
    # subpath = '_'.join(get_stack_path(3))
    return re.sub(r'[<>:"/\|?*]', replace, fname)
