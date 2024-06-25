# import vtk_utils
import numpy as np
from tools import vtk_utils
from trainer import torch_utils
from dataset.nervedataset import NerveDetectionSet, NerveBoxDataset


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

    for _ in range(4):
        for i in range(len(dataset)):
            items = dataset[i]
            print(torch_utils.get_shape(items))
            src, lab = items
            vtk_utils.show_actors([src, lab, vtk_utils.get_axes(50)])

if __name__ == '__main__':
    nerve_roi_segmentation_set()
    # nerve_detection_set()
    # main()