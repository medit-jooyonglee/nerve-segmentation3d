import SimpleITK
import pytest

from trainer import get_model, create_trainer, load_config

@pytest.fixture()
def nerve_detection_config():

    config_file = 'train_nerve_roi_detection.yaml'
    config = load_config(config_file, '../configure')
    return config


def test_trainer_nerve_segmentation(nerve_detection_config):
    # config = get_model(nerve_detection_config)
    trainer = create_trainer(nerve_detection_config)
    trainer.num_iterations = 2
    trainer.fit()


if __name__ == '__main__':
    pytest.main([
        '-s',
        '--color=yes',
        '-rGA',
        __file__
    ])


