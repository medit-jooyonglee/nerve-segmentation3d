import os.path
from ml_collections import ConfigDict
import pytest
import torch
from models import vit_seg_interpolator_modeling
import trainer
from trainer import get_model


@pytest.fixture()
def model_detection():
    basename = os.path.dirname(os.path.dirname(__file__))
    config_file = os.path.join(basename, 'configure/train_nerve_roi_detection.yaml')
    assert os.path.exists(config_file), f'empty configure file: {config_file}'
    config = trainer.load_config(config_file)
    model_config = ConfigDict(config['model'])
    return get_model(model_config).cuda()


@pytest.fixture()
def model_segmentation():
    basename = os.path.dirname(os.path.dirname(__file__))
    config_file = os.path.join(basename, 'configure/train_transunet_nerve_roi_segmentation_liteweight.yaml')
    assert os.path.exists(config_file), f'empty configure file: {config_file}'
    config = trainer.load_config(config_file)
    model_config = ConfigDict(config['model'])
    return get_model(model_config).cuda()


def test_run_detection_model(model_detection):
    shape = (128, ) * 3
    inputs = torch.randn(1, 1, *shape).cuda()
    outputs = model_detection(inputs)
    assert outputs.shape[1] == 3
    assert outputs.shape[2:] == inputs.shape[2:]


def test_run_segmentation_model(model_segmentation):
    shape = (128, ) * 3
    inputs = torch.randn(1, 1, *shape).cuda()
    outputs = model_segmentation(inputs)
    assert outputs.shape[1] == 2
    assert outputs.shape[2:] == inputs.shape[2:]


if __name__ == '__main__':
    pytest.main([
        '-s',
        '--color=yes',
        '-rGA',
        # 'test_model.py',
        # 'test_model.py::test_run_detection_model',
        'test_model.py::test_run_segmentation_model',
    ])