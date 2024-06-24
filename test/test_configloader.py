import os.path

import ml_collections
import pytest
from trainer import find_config, load_config


def test_find_config():
    test_config_fname = 'test_configure'
    'test'
    found = find_config(test_config_fname, '../configure')
    assert os.path.exists(found)


def test_find_config2():
    test_config_fname = 'configure'
    'test'
    found = find_config(test_config_fname, '../')
    assert os.path.exists(found)


def test_empty_config_file():
    test_config_fname = 'some_test_empty_path'
    found = find_config(test_config_fname, '../')

    assert not os.path.exists(found)


def test_load_config():
    config_filename = 'configure/train_nerve_roi_detection.yaml'
    config = load_config(config_filename, '../')

    assert isinstance(config, ml_collections.ConfigDict)

# def test_empty_find_config():
if __name__ == '__main__':
    pytest.main(['-s',
                 '--color=yes',
                 '-rGA',
                 __file__
                 # 'test_con',
                 ])

