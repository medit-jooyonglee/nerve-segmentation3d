import os.path

import pytest
import shutil
import numpy as np
from trainer import image_utils

_shape = (4, 5, 10)

@pytest.fixture()
def source_image():
    return np.random.uniform(0, 255, _shape)


@pytest.fixture()
def mask_image():
    return np.random.uniform(0, 1, _shape) > .5

@pytest.fixture()
def mask_image_others():
    masks = []
    for _ in range(int(np.random.uniform(1, 5, ))):
        masks.append(np.random.uniform(0, 1, _shape) > .5)
    return masks


def test_compare_image_singlemask(source_image, mask_image):
    save_path = './temp'
    image_utils.compare_image(source_image,
                              mask_image,
                              save_path=save_path,
                              show=False,
                              create_sub_dir=False,
                              image_save=True)

    assert os.path.exists(save_path)
    # shutil.rmtree('/sdfdsfsdf', ignore_errors=True)
    shutil.rmtree(save_path, ignore_errors=True)


def test_compare_image_multiple_mask(source_image, mask_image, mask_image_others):

    save_path = './temp'
    image_utils.compare_image(source_image,
                              mask_image,
                              other_mask_image=mask_image_others,
                              save_path=save_path,
                              show=False,
                              create_sub_dir=False,
                              image_save=True)

    assert os.path.exists(save_path)
    shutil.rmtree(save_path, ignore_errors=True)


if __name__ == '__main__':
    pytest.main([
        '-s',
        '--color=yes',
        # __file__+'::test_compare_image_singlemask',
        __file__ + '::test_compare_image_multiple_mask',
    ])

