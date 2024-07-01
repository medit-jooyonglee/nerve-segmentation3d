import numpy as np
import pytest
import os
from trainer import load_config, diskmanager
from tools import vtk_utils
from dataset.mha import read_image_volume
from models.fullsegment import ObbSegmentTorch


def init_inference_model(detect_config, segment_config):
    root = os.path.dirname(os.path.dirname(__file__))
    detection_config = load_config(detect_config, root=root)  # two class segmentaiton + roi segmentation
    segmentation_config = load_config(segment_config, root=root)
    load_model_data = ObbSegmentTorch(detection_config, segmentation_config)
    return load_model_data


@pytest.fixture(scope='session')
def inference_model():
    detect_config = 'configure/roi_detection_miccai.yaml'
    segment_config = 'configure/roi_segmentation_miccai.yaml'
    # config_file = '3DUnet_multiclass/train_transunet_nerve_roi_segmentation_liteweight.yaml'
    # test_mode = 'teethnet'
    return init_inference_model(detect_config, segment_config)


def test_full_model_input_volume(inference_model):

    vol = np.random.randn(300, 300, 300)
    spacing = np.asarray([0.3,] * 3)
    res = inference_model.full_segment(vol, spacing)
    assert res.shape == vol.shape


def test_visualize_full_model_read_miccai_dataset(inference_model):
    pathname = 'D:/dataset/miccai/Dataset112_ToothFairy2/imagesTr'
    found = diskmanager.deep_serach_files(pathname, ['.mha'])
    assert len(found) > 0, f'empty foound: {pathname}'
    found = found[::-1]
    for filename in found:
        vol, info = read_image_volume(filename)
        spacing = info.GetSpacing()
        pred_mask = inference_model.ful_segment(vol, spacing)

        res_vtk = volume_mask_coloring(pred_mask)
        #                                            color=colors[(i + 1) * 3]) for i, v in enumerate(nonzers)]
        # res_vtk = vtk_utils.numpyvolume2vtkvolume(res, color=(1, 1, 0))
        vol_vtk = vtk_utils.volume_coloring(vol)
        vtk_utils.split_show([vol_vtk], [vol_vtk, res_vtk])


def volume_mask_coloring(mask):
    nonzers = np.unique(mask)
    nonzers = nonzers[nonzers > 0]
    colors = vtk_utils.get_teeth_color_table()
    res_vtk = [vtk_utils.numpyvolume2vtkvolume((mask == v).astype(np.int32),
                                               color=colors[(i + 1) * 3]) for i, v in enumerate(nonzers)]
    return res_vtk

def test_visualize_full_model_read_miccai_dataset_with_gt(inference_model):
    pathname = 'D:/dataset/miccai/Dataset112_ToothFairy2/imagesTr'
    gt_pathname = 'D:/dataset/miccai/Dataset112_ToothFairy2/labelsTr'
    found = diskmanager.deep_serach_files(pathname, ['.mha'])
    gt_found = diskmanager.deep_serach_files(gt_pathname, ['.mha'])
    assert len(found) > 0, f'empty foound: {pathname}'
    assert len(found) > 0, f'empty foound: {pathname}'
    found = found[::-1]
    gt_found = gt_found[::-1]
    def split_id(filename, i): return int(filename.split('_')[-i])

    mapper = np.zeros([100], dtype=np.uint8)
    mapper[np.array([3, 4])] = np.array([1, 2])


    for filename, gt_filename in zip(found, gt_found):
        # os.path.splitext
        # src_ids = split_id(filename, 2)
        # os.path.

        vol, info = read_image_volume(filename)
        gt_mask, _ = read_image_volume(gt_filename, normalize=False)
        gt_mask = mapper[gt_mask]
        spacing = info.GetSpacing()
        pred_mask = inference_model.full_segment(vol, spacing)
        mask_vtk = volume_mask_coloring(pred_mask)
        gt_vtk = volume_mask_coloring(gt_mask)
        # res_vtk = vtk_utils.numpyvolume2vtkvolume(res, color=(1, 1, 0))
        vol_vtk = vtk_utils.volume_coloring(vol)
        vtk_utils.split_show([vol_vtk], [vol_vtk, mask_vtk], [vol_vtk, gt_vtk])

if __name__ == '__main__':
    pytest.main(['-s',
                 '--color=yes',
                 '-rGA',
                 # __file__ + '::test_full_model_input_volume',
                 # __file__ + '::test_visualize_full_model_read_miccai_dataset',
                 __file__ + '::test_visualize_full_model_read_miccai_dataset_with_gt',

                 ])
    # vtk_utils.show([pred])