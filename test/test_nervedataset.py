# import vtk_utils
import numpy as np
from tools import vtk_utils
from trainer import torch_utils
from dataset.nervedataset import NerveDetectionSet, NerveBoxDataset
from dataset.miccaitoothfairy import NerveMICCAISet

def nerve_roi_segmentation_set():
    args = {
        'paths': [
            [
                'D:/temp/make/source',
                'D:/temp/make/gt',
            ],
        ]
    }
    dataset = NerveBoxDataset(**args)
    assert len(dataset) > 0

    for _ in range(4):
        for i in range(len(dataset)):
            items = dataset[i]
            print(torch_utils.get_shape(items))
            src, lab = items
            print(lab.dtype)
            vtk_utils.show_actors([src, lab.astype(np.int32), vtk_utils.get_axes(50)])


def nerve_detection_set():
    args = {
        'paths': [
            [
                'D:/temp/make/source',
                'D:/temp/make/gt',
            ],
        ]
    }
    dataset = NerveDetectionSet(**args)
    assert len(dataset) > 0
    # from tools impo
    from tools import image_utils
    stride = 1
    save_path = 'd:/temp/nerve/miccai'
    dataset.en_augmented = False
    for _ in range(4):
        for i in range(len(dataset)):
            items = dataset[i]

            print(torch_utils.get_shape(items))
            src, lab = items
            src = src[0]
            src, lab = [np.transpose(v, [1, 2, 0]) for v in [src, lab]]
            vtk_utils.show_actors([src, lab, vtk_utils.get_axes(50)])
            # -
            image_utils.show_2mask_image(src * 255, lab, show=False,
                                         save_path=save_path, image_save=True, create_sub_dir=True, transpose=True,
                                         stride=stride, in_rainbow_size=10)


def miccai_nerve_detection_set():
    args = {
        'paths': [
            [
                '../dataset/data_split_miccai.json',
                'train',
            ],
        ]
    }
    dataset = NerveMICCAISet(**args)
    assert len(dataset) > 0
    # from tools impo
    from tools import image_utils
    stride = 1
    save_path = 'd:/temp/nerve/miccai2'
    dataset.en_augmented = False
    dataset.train(False)
    for _ in range(4):
        for i in range(len(dataset)):
            items = dataset[i]

            print(torch_utils.get_shape(items))
            src, lab = items
            src = src[0]
            src, lab = [np.transpose(v, [1, 2, 0]) for v in [src, lab]]
            vtk_utils.show_actors([src, lab, vtk_utils.get_axes(50)])
            # -
            # image_utils.show_2mask_image(src * 255, lab, show=False,
            #                              save_path=save_path, image_save=True, create_sub_dir=True, transpose=True,
            #                              stride=stride, in_rainbow_size=10)


if __name__ == '__main__':
    # nerve_roi_segmentation_set()
    # nerve_detection_set()
    # main()
    miccai_nerve_detection_set()