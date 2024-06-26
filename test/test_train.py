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

    config_file = 'train_nerve_roi_detection_empty.yaml'
    config = load_config(config_file, os.path.join(root, '../configure'))
    return config


@pytest.fixture()
def nerve_roi_segmentation_emptyseg_config():

    config_file = 'train_transunet_nerve_roi_segmentation_liteweight_empty.yaml'
    config = load_config(config_file, os.path.join(root, '../configure'))
    return config

@pytest.fixture()
def teeth_center_proposal_config():

    config_file = 'mesh_voxelizer_center_proposal_max_dilate_mask_onlyseg_avg_resnetblock_loss_weight_50_double_decoder.yaml'
    config = load_config(config_file, os.path.join(root, '../configure/teeth_configure'))
    return config


@pytest.fixture()
def teeth_roi_segment_config():

    config_file = 'mesh_voxelizer_teeth_roi_segment.yaml'
    config = load_config(config_file, os.path.join(root, '../configure/teeth_configure'))
    return config


def test_trainer_nerve_segmentation(nerve_detection_config):
    # config = get_model(nerve_detection_config)
    try:
        trainer = create_trainer(nerve_detection_config)
    except Exception as e:
        print(e.args)
        assert False
    trainer.max_num_iterations = 2
    trainer.log_after_iters = 1
    trainer.fit()


def test_trainer_nerve_detection(nerve_roi_segmentation_config):
    # config = get_model(nerve_detection_config)
    try:
        trainer = create_trainer(nerve_roi_segmentation_config)
    except Exception as e:
        print(e.args)
        assert False
    trainer.max_num_iterations = 2
    trainer.log_after_iters = 1
    trainer.fit()


def test_nerved_detection_emptyset(nerved_detection_emptyset_config):
    # config = get_model(nerve_detection_config)
    try:
        trainer = create_trainer(nerved_detection_emptyset_config)
    except Exception as e:
        print(e.args)
        assert False

    trainer.max_num_iterations = 2
    trainer.num_epochs = 2
    trainer.log_after_iters = 1
    trainer.fit()


def test_nerve_roi_segmentation_emptyseg(nerve_roi_segmentation_emptyseg_config):
    # config = get_model(nerve_detection_config)
    try:
        trainer = create_trainer(nerve_roi_segmentation_emptyseg_config)
    except Exception as e:
        print(e.args)
        assert False
    trainer.max_num_iterations = 2
    trainer.num_epochs = 2
    trainer.log_after_iters = 1
    trainer.fit()


def test_teeth_center_proposal_emptyseg(teeth_center_proposal_config):
    # config = get_model(nerve_detection_config)
    try:
        trainer = create_trainer(teeth_center_proposal_config)
    except Exception as e:
        print(e.args)
        assert False
    trainer.max_num_iterations = 2
    trainer.num_epochs = 2
    trainer.log_after_iters = 1
    trainer.fit()


def test_teeth_roi_segment_emptyseg(teeth_roi_segment_config):
    # config = get_model(nerve_detection_config)
    try:
        trainer = create_trainer(teeth_roi_segment_config)
    except Exception as e:
        print(e.args)
        assert False
    trainer.max_num_iterations = 2
    trainer.num_epochs = 2
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
        # __file__ + '::test_nerve_roi_segmentation_emptyseg',
        # __file__ + '::test_nerve_roi_segmentation_emptyseg',
        __file__ + '::test_teeth_roi_segment_emptyseg',
        # __file__
    ])


if __name__ == '__main__':
    main()

