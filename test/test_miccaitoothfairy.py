import matplotlib.pyplot as plt
import numpy as np
from dataset.miccaitoothfairy import NerveMICCAISet, NerveMICCAIBoxSet
from trainer import torch_utils
from tools import vtk_utils, image_utils
from commons import timefn2
import matplotlib.pyplot as plt
import matplotlib
# plt.get_backend

# matplotlib.use('Qt5Agg')
matplotlib.use('TkAgg')
# GTK3Agg, GTK3Cairo, GTK4Agg, GTK4Cairo, MacOSX, nbAgg, notebook, QtAgg,
# QtCairo, TkAgg, TkCairo, WebAgg, WX, WXAgg, WXCairo, Qt5Agg, Qt5Cairo

# TKAgg','GTKAgg','Qt4Agg','WXAgg'
def nerve_miccai_dataset():
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

    for _ in range(4):
        for i in range(len(dataset)):
            @timefn2
            def load_data():
                items = dataset[i]
                return items

            items = load_data()
            print(torch_utils.get_shape(items))
            src, lab = items
            # vtk_utils.show_actors([src, lab, vtk_utils.get_axes(50)
            #                        ])

            # plt.imshow()
            image_utils.compare_image(np.squeeze(src) * 255, lab)

def nerve_miccai_roi_segmentation_set():
    args = {
        'paths': [
            [
                '../dataset/data_split_miccai.json',
                'train',
            ],
        ]
    }
    dataset = NerveMICCAIBoxSet(**args)
    assert len(dataset) > 0

    for _ in range(4):
        for i in range(len(dataset)):
            items = dataset[i]
            print(torch_utils.get_shape(items))
            src, lab = items
            print(lab.dtype)
            vtk_utils.show_actors([src, lab.astype(np.int32), vtk_utils.get_axes(50)])



if __name__ == '__main__':
    nerve_miccai_dataset()
    # nerve_miccai_roi_segmentation_set()
    # nerve_detection_set()
    # main()