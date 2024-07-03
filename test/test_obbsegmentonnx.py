import shutil

import pytest
import os
import yaml
import numpy as np
import torch
import psutil
from commons import get_runtime_logger
from models.obbsegmentonnx import ObbSegmentOnnx

from trainer import diskmanager
from tools import dicom_read_wrapper
from tools import vtk_utils


def load_config(filename='3DUnet_multiclass/train_config.yaml'):
    """
    절대경로로 입력할 경후 해당 파일을 로드.
    상대경로로 파일을 입력할 경구 dataset/resources 경로에서 파일 로드
    Parameters
    ----------
    filename :

    Returns
    -------

    """
    path = os.path.dirname(os.path.realpath(__file__))
    pathname = os.path.join(path, '../../dataset/resources/' + filename)
    pathname = os.path.realpath(pathname)
    pathname = filename if os.path.isabs(filename) else pathname

    assert os.path.exists(pathname), 'emtpy config yaml file:{}'.format(pathname)
    # trim path
    logger = get_runtime_logger()

    logger.info('loading config file:{}'.format(pathname))
    # parser = argparse.ArgumentParser(description='UNet3D')
    # parser.add_argument('--config', type=str, help='Path to the YAML config file', required=True)
    # args = parser.parse_args()
    config = yaml.safe_load(open(pathname, 'r', encoding='utf-8'))

    return config


@pytest.fixture(scope='session')
def obb_nerve_model():
    # global model
    # global config
    # import shutil
    shutil.cop()
    os.path.samefile()
    os.path.relpath()
    # shutil.copytree(dirs_exist_ok=)
    # shutil.copy()
    app_base_path = '../'
    # FIXME: set the finale config in asset-path

    detect_config_filename = 'models/asset/test_nerve_roi_detection.yaml'
    detect_config_abs_path = os.path.realpath(os.path.join(app_base_path, detect_config_filename))
    detect_config = load_config(detect_config_abs_path)

    segment_config_filename = 'models/asset/train_transunet_nerve_roi_segmentation_liteweight.yaml'
    segment_config_abs_path = os.path.realpath(os.path.join(app_base_path, segment_config_filename))
    segment_config = load_config(segment_config_abs_path)


    detect_model_path = "asset/nerve_detection_model.onnx"
    segment_model_path = "asset/nerve_segmentation_model.onnx"
    #
    base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models')
    detect_model_path = os.path.realpath(os.path.join(base_path, detect_model_path))
    segment_model_path = os.path.realpath(os.path.join(base_path, segment_model_path))

    # if model is None:
    model = ObbSegmentOnnx(detect_config, segment_config, detect_model_path, segment_model_path)

    return model


@pytest.fixture(scope='session')
def cbct_smaple():
    from tools import dicom_read_wrapper
    path = 'D:/dataset/ai_hub_labels/CTDATA/52/CT3D'
    assert os.path.exists(path), 'empty path'
    volume, spacing = dicom_read_wrapper.read_dicom_wrapper(path)
    return volume, spacing
# @pytest.fixture()
def test_inference(obb_nerve_model, cbct_smaple):
    # volume = np.r


    basepath = 'D:/dataset/ai_hub_labels/CTDATA'
    found = diskmanager.deep_search_directory(basepath, exts=['.dcm'], filter_func=lambda x: len(x) > 0)
    assert len(found) > 0
    print(f'found #{len(found)=}')
    for i, pathname in enumerate(found):
        volume, spacing = dicom_read_wrapper.read_dicom_wrapper(pathname)

        # volume, spacing = cbct_smaple
        full_segments, splines = obb_nerve_model.full_segment(volume, spacing, compute_full_segment=True, order='zyx')
        # init_nerve_segment_model()
        vtk_utils.split_show([volume], [volume, full_segments])

        # obb_nerve_model

def test_inference_empty_model(obb_nerve_model):
    inputs = np.random.randn(300, 200, 400)
    spacing = np.array([0.3]* 3)

    full_segments, splines = obb_nerve_model.full_segment(inputs, spacing, compute_full_segment=True, order='zyx')
    assert full_segments.shape == inputs.shape
    # init_nerve_segment_model()
    # vtk_utils.split_show([volume], [volume, full_segments])


if __name__ == '__main__':
    pytest.main([
        '-s',
        '--color=yes',
        '-rGA',
        # 'test_obbsegmentonnx.py::test_inference',
        'test_obbsegmentonnx.py::test_inference_empty_model',
    ])