import os.path

import pytest
from trainer import get_model, create_trainer, load_config

root = os.path.dirname(__file__)
@pytest.fixture()
def nerve_detection_config():

    config_file = 'train_nerve_roi_detection.yaml'
    config = load_config(config_file, os.path.join(root, '../configure'))
    return config


@pytest.fixture()
def nerve_roi_segmentation_config():

    config_file = 'train_transunet_nerve_roi_segmentation_liteweight.yaml'
    config = load_config(config_file, os.path.join(root, '../configure'))
    return config

@pytest.fixture()
def nerved_detection_emptyset_config():

    config_file = 'train_nerve_roi_detection.yaml'
    config = load_config(config_file, os.path.join(root, '../configure'))
    return config


@pytest.fixture()
def nerve_roi_segmentation_emptyseg_config():

    config_file = 'train_transunet_nerve_roi_segmentation_liteweight.yaml'
    config = load_config(config_file, os.path.join(root, '../configure'))
    return config


def test_trainer_nerve_segmentation(nerve_detection_config):
    # config = get_model(nerve_detection_config)
    trainer = create_trainer(nerve_detection_config)
    trainer.num_iterations = 2
    trainer.log_after_iters = 1
    trainer.fit()


def test_trainer_nerve_detection(nerve_roi_segmentation_config):
    # config = get_model(nerve_detection_config)
    trainer = create_trainer(nerve_roi_segmentation_config)
    trainer.num_iterations = 2
    trainer.log_after_iters = 1
    trainer.fit()


def test_nerved_detection_emptyset(nerved_detection_emptyset_config):
    # config = get_model(nerve_detection_config)
    trainer = create_trainer(nerved_detection_emptyset_config)
    trainer.num_iterations = 2
    trainer.log_after_iters = 1
    trainer.fit()


def test_nerve_roi_segmentation_emptyseg(nerve_roi_segmentation_emptyseg_config):
    # config = get_model(nerve_detection_config)
    trainer = create_trainer(nerve_roi_segmentation_emptyseg_config)
    trainer.num_iterations = 2
    trainer.log_after_iters = 1
    trainer.fit()


def main():
    pytest.main([
        '-s',
        '--color=yes',
        '-rGA',
        # __file__,
        # __file__ + '::test_trainer_nerve_segmentation',
        # __file__ + '::test_trainer_nerve_detection',
        # __file__ + '::test_nerved_detection_emptyset',
        __file__ + '::test_nerve_roi_segmentation_emptyseg',
    ])


if __name__ == '__main__':
    main()

