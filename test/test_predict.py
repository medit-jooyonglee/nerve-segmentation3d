import os
import pytest
import glob
import numpy as np
import torch

import vtk_utils
from tools import image_utils
from trainer import get_model, load_config, find_config, create_evaluator, get_logger
from trainer import utils, diskmanager, torch_utils
import trainer
from dataset.mha import read_image_volume


@pytest.fixture(scope='session')
def detect_model():
    # config_path = 'trainer/CHECKPOINT_DIR/roi_detection_miccai_cgr_doubleconv'
    config_path = 'trainer/CHECKPOINT_DIR/roi_detection_miccai_gcr_resnetblock_sse'

    config_file = find_config(config_path, '../')

    config = load_config(config_file)
    # model = get_model(config)
    # eval = create_evaluator(config)

    model = get_model(config['model'])

    checkpoint_dir = os.path.join(os.path.dirname(trainer.__file__), config.trainer.checkpoint_dir)
    # model = eval.mo\del
    assert os.path.exists(checkpoint_dir)
            # checkpoint_dir = os.path.abspath(config_path)
    # logger = get_lo
    best_checkpoints = glob.glob(os.path.join(checkpoint_dir, 'best*'))
    logger = get_logger()
    if len(best_checkpoints) >= 1:
        resume = best_checkpoints[0]
        state = utils.load_checkpoint(resume, model, None, strict=False)
        logger.info(
            f"Checkpoint loaded from '{resume}'. Epoch: {state['num_epochs']}.  Iteration: {state['num_iterations']}. "
            f"Best val score: {state['best_eval_score']}."
        )
    assert os.path.exists(resume), 'No checkpoint found at {}'

    return model.cuda()


def volume_flip(volume, flips_index):
    num = flips_index
    outs = []
    eval_str = 'volume['
    for i in reversed(range(volume.ndim)):
        flip = flips_index // ( 2** i)
        # print(f' {i} bin / {flip} // {num} ')
        # if flip > 0:
        # if flip > 0:
        rev = -1 if flip > 0 else 1
        eval_str += f'::{rev}'
        outs.append(flip)
        flips_index = flips_index % (2 **i)

        eval_str += ','\
            # if i < len(volume.ndim) else ''
    eval_str = eval_str[:-1] + ']'
    # print(eval_str)
    volume = eval(eval_str).copy()
    # volume = eval(f'volume[')

    # print(outs)
    return volume
    # f0 = flips_index //4
    # flips_index = flips_index 4
    # flips_index
    # if flips_index // 4:


def test_roi_detect_cbct(detect_model):
    test_ct_basename = 'D:/dataset/ai_hub_labels/CTDATA'
    from tools.simpleitk_utils import read_sitk_volume
    found = diskmanager.deep_search_directory(test_ct_basename, exts=['.dcm'], filter_func=lambda x: len(x) > 0)

    found = found[::-1]


    mapper = np.zeros([100], dtype=np.uint8)
    mapper[np.array([3, 4])] = np.array([1, 2])

    # save_path = 'd:/temp/nerve'

    for filename in found:
        subpath = os.path.basename(filename)
        target_shape = [128, ] * 3
        # src_vol, info = read_image_volume(filename)
        src_vol, spacing, window = read_sitk_volume(filename)

        rsz_vol = image_utils.auto_cropping_keep_ratio(src_vol, target_shape)

        for i in range(8):
            rsz_vol_flip = volume_flip(rsz_vol, i)

            rsz_vol_tensor = torch_utils.data_convert(rsz_vol_flip[None])

            with torch.no_grad():
                res = detect_model(rsz_vol_tensor)
            input_vol, pred_seg = torch_utils.to_numpy([rsz_vol_flip, torch.argmax(res, dim=1)])

            cube = vtk_utils.get_cube_from_shape(input_vol.shape)
            vtk_utils.show([
                input_vol, pred_seg, vtk_utils.get_axes(100), cube
            ])
# test_ct_basename = 'D:/dataset/ai_hub_labels/CTDATA'
# from tools.simpleitk_utils import read_sitk_volume
# found = diskmanager.deep_search_directory(test_ct_basename, exts=['.dcm'], filter_func=lambda x: len(x) > 0)
# print(f'found {len(found)}')
# assert len(found) > 0, f'empty foound: {test_ct_basename}'
# used_volume_direction = 'lai'
# # used_volume_direction = 'rai'
# inference_model.debug = True
# save_path = 'd:/temp/nerve/old_detect_new_seg'
# for file in found:
#     name = '_'.join(diskmanager.split_dir_paths(file)[-4:])
#     inference_model.debug_path = os.path.join(save_path, name) #'d:/temp/nerve/ai_hub_result'
#
#     src_vol, spacing, window = read_sitk_volume(file)
#
def test_roi_detect_mha(detect_model):
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

    # save_path = 'd:/temp/nerve'

    for filename, gt_filename in zip(found, gt_found):
        subpath = os.path.basename(filename)
        target_shape = [128,] * 3
        src_vol, info = read_image_volume(filename)

        rsz_vol = image_utils.auto_cropping_keep_ratio(src_vol, target_shape)


        for i in range(8):
            rsz_vol_flip = volume_flip(rsz_vol, i)

            rsz_vol_tensor = torch_utils.data_convert(rsz_vol_flip[None])

            with torch.no_grad():
                res = detect_model(rsz_vol_tensor)
            input_vol, pred_seg = torch_utils.to_numpy([rsz_vol_flip, torch.argmax(res, dim=1)])

            cube = vtk_utils.get_cube_from_shape(input_vol.shape)
            vtk_utils.show([
                input_vol, pred_seg, vtk_utils.get_axes(100), cube
            ])
        # vtk_utils.sho
        # res = inference_model.detection_roi(src_vol[:, ::-1, ::-1].copy(), spacing)
        # res = inference_model.detection_roi(src_vol, spacing)
# main()

# def

if __name__ == '__main__':
    pytest.main([
        '-s',
        '-rGA',
        '--color=yes',
        __file__ + '::test_roi_detect_cbct',
        # __file__ + '::test_roi_detect_mha',
    ])