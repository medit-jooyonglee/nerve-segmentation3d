import logging

import numpy as np
import pytest
import os
import logging
from trainer import load_config, diskmanager
from tools import vtk_utils
from trainer import utils
from dataset.mha import read_image_volume
from models.fullsegment import ObbSegmentTorch
from trainer import get_logger, image_utils


logger = get_logger(os.path.basename(__file__), logging.DEBUG)

def init_inference_model(detect_config, segment_config):
    root = os.path.dirname(os.path.dirname(__file__))
    detection_config = load_config(detect_config, root=root)  # two class segmentaiton + roi segmentation
    segmentation_config = load_config(segment_config, root=root)
    load_model_data = ObbSegmentTorch(detection_config, segmentation_config)
    return load_model_data


@pytest.fixture(scope='session')
def inference_model():
    # old version LPS option
    # detect_config = 'configure/roi_detection_miccai.yaml'
    # segment_config = 'configure/roi_segmentation_miccai.yaml'

    # old version. LAI option
    detect_config = 'configure/nerve_roi_detection_oldver.yaml'
    segment_config = 'configure/roi_segmentation_oldver.yaml'


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
    # vtk  lai, itk - lps
    for filename in found:
        vol, info = read_image_volume(filename)

        # input_volume_direction = {
        #
        # }
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


def test_visualize_fullsegment_predict(inference_model):

    test_ct_basename = 'D:/dataset/ai_hub_labels/CTDATA'
    from tools.simpleitk_utils import read_sitk_volume
    found = diskmanager.deep_search_directory(test_ct_basename, exts=['.dcm'], filter_func=lambda x: len(x) > 0)
    print(f'found {len(found)}')
    assert len(found) > 0, f'empty foound: {test_ct_basename}'
    # used_volume_direction = 'lps'
    used_volume_direction = 'rai'

    for file in found:

        vol, spacing, window = read_sitk_volume(file)

        if used_volume_direction == 'rai':
            # z, y, inversion
            vol = vol[::-1, ::-1]
        elif used_volume_direction == 'rpi':
            # miccai dataset
            vol = vol[::-1, :,::-1]
            pass

        pred_mask = inference_model.full_segment(vol, spacing)
        mask_vtk = volume_mask_coloring(pred_mask)

        vol_vtk = vtk_utils.volume_coloring(vol)

        vtk_utils.split_show([
            vol_vtk,
        ], [vol_vtk, mask_vtk, vtk_utils.get_axes(100)])
        # vtk_utils.read_vtk()

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

    save_path = 'd:/temp/nerve'

    for filename, gt_filename in zip(found, gt_found):
        subpath = os.path.basename(filename)
        save_pathname = os.path.join(save_path, subpath)
        inference_model.debug = False
        inference_model.debug_path = save_pathname
        vol, info = read_image_volume(filename)
        gt_mask, _ = read_image_volume(gt_filename, normalize=False)
        gt_mask = mapper[gt_mask]
        spacing = info.GetSpacing()
        pred_mask = inference_model.full_segment(vol, spacing)
        mask_vtk = volume_mask_coloring(pred_mask)
        gt_vtk = volume_mask_coloring(gt_mask)
        # res_vtk = vtk_utils.numpyvolume2vtkvolume(res, color=(1, 1, 0))
        vol_vtk = vtk_utils.volume_coloring(vol)
        _2d_image_save = True
        dsc_val, dsc_lists = dsc(pred_mask, gt_mask, 3)

        # from tools import image_utils

        # if visualize:
        #     vtk_utils.show_actors([
        #         full_volume * 0.75,  # 가시화를 위해 hu 값 줄이기
        #         *vtk_utils.auto_refinement_mask(full_seg)
        #     ])
        if np.any(dsc_lists < .5):
            inference_model.debug = True
            inference_model.full_segment(vol, spacing)
            # full segment
            image_utils.compare_image(*[
                np.transpose(v, [1, 0, 2]) for v in [vol * 255, pred_mask]
            ], other_mask_image=[np.transpose(gt_mask, [1, 0, 2])], concat_original=True, pause_sec=0.001,
                                      full_region=False, image_save=True,
                                      save_path=save_pathname,
                                      show=False)

        vtk_utils.split_show([vol_vtk], [vol_vtk, mask_vtk], [vol_vtk, gt_vtk])



def dsc(pred, tar, num_ch):
    smooth = 1e-5
    values = []
    for i in range(1, num_ch):
        pred0, tar0 = pred == i, tar == i
        pred_flat = pred0.ravel()
        true_flat = tar0.ravel()
        intersection = (pred_flat * true_flat).sum()
        val = ( 2. * intersection + smooth ) / (pred_flat.sum() + true_flat.sum()) + smooth
        values.append(val)
    logger.debug(values)
    mdsc = np.mean(values)
    return mdsc, np.asarray(values)

if __name__ == '__main__':
    pytest.main(['-s',
                 '--color=yes',
                 '-rGA',
                 # __file__ + '::test_full_model_input_volume',
                 # __file__ + '::test_visualize_full_model_read_miccai_dataset',
                 # __file__ + '::test_visualize_full_model_read_miccai_dataset_with_gt',
                 __file__ + '::test_visualize_fullsegment_predict',
                 ])
    # vtk_utils.show([pred])