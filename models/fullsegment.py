import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import glob
from scipy.ndimage import gaussian_filter
from skimage import morphology

from commons import get_runtime_logger, timefn2, app_base_path
from tools import image_utils, vtk_utils
from tools import InterpolateWrapper
from trainer import get_model, utils, torch_utils
from tools.obb import Obb
import trainer
#     def forward(self, x):
#         return super(InferencePadUnet3D, self).forward(x)
def remove_small_objects(image, threshold, method):
    assert method in ['skimage', 'pymeshlibs']
    return morphology.remove_small_objects(image.astype(np.bool_), threshold)



class ObbSegmentBase(object):
    def __init__(self, config):
        self.visualize = False
        self.patch_size = [256, 80, 80]

        # pool 8 ---> 16 으로 변경
        self.pool_seed = [16, 16, 16]
        # shape = [360, 100, 60]
        shape = [320, 100, 60]
        self.target_shape = [(a//b) * b for a, b in zip(shape, self.pool_seed)] # [360, (100 // 8) * 8, (60 // 8) * 8]
        self.detect_in_ch = config['model'].get('in_channels')
        self.detect_out_ch = config['model'].get('out_channels')
        pass

    def detection_roi(self, volume) -> np.ndarray:
        raise NotImplementedError

    def full_segment(self, full_volume:np.ndarray, spacing, obb_scaling=1.0) -> np.ndarray:
        raise NotImplementedError


class ObbSegmentTorch(ObbSegmentBase):
    def __init__(self, det_config, seg_model_path_or_config, test_mode:str='3dunet'):
        super(ObbSegmentTorch, self).__init__(det_config)
        assert test_mode in ['teethnet', '3dunet']
        self.test_mode = test_mode
        det_model = self._init_det_model(det_config)
        det_model.eval()
        self.det_model = det_model

        seg_model, in_ch, out_ch = self._init_seg_model(seg_model_path_or_config)
        seg_model.eval()
        self.seg_in_ch = in_ch
        self.seg_out_ch = out_ch
        self.seg_model = seg_model

        seg_model_config = seg_model_path_or_config.get('model') if isinstance(seg_model_path_or_config, dict) else {}

        self.seg_detection_concat = seg_model_config.get('detection_concat', False)
        self.seg_statistics_model_concat = seg_model_config.get('statistics_model_concat', False)

        self.stat_models = np.array([])
        if self.seg_statistics_model_concat:
            self.stat_models = self._load_statistical_model()

        self.debug = False
        self.debug_items = []
        self.debug_path = ''

    def _load_statistical_model(self):
        logger = get_runtime_logger()
        stat_model_path = os.path.join(app_base_path, 'resources/stat_nerve.pkl')
        assert os.path.exists(\
            stat_model_path), 'cannot find stat models. refer "nerve_detection/mnodels/statisticsmodel.py"'
        with open(stat_model_path, 'rb') as f:
            data = pickle.load(f)
            logger.debug(f'stat models : {data.keys()}')
            # (num, d, h, w)
            stat_volumes = np.stack(list(data.values()), axis=0)
        return stat_volumes.astype(np.float32)

    def _init_det_model(self, config, phase='val', en_cuda=True):
        logger = get_runtime_logger()

        # Create the model
        model = get_model(config['model'])

        # check point directory로 변경
        model_path = os.path.join(config['trainer']['checkpoint_dir'])
        # path = os.path.dirname(os.path.realpath(trainer.__file__))
        path = os.path.dirname(trainer.__file__)
        pathname = os.path.join(path, model_path)
        pathname = os.path.realpath(pathname)
        best_checkpoints = glob.glob(os.path.join(pathname, 'best*'))
        assert len(best_checkpoints) == 1, 'emtpy or multiple best checkpoints'

        logger.info(f'Loading model from {model_path}...')
        state = utils.load_checkpoint(best_checkpoints[0], model)
        logger.info(
            f"Checkpoint loaded from '{best_checkpoints}'. Epoch: {state['num_epochs']}.  Iteration: {state['num_iterations']}. "
            f"Best val score: {state['best_eval_score']}."
        )
        # use DataParallel if more than 1 GPU available
        if en_cuda:
            if torch.cuda.device_count() > 1 and not config['device'] == 'cpu':
                model = nn.DataParallel(model)
                logger.info(f'Using {torch.cuda.device_count()} GPUs for prediction')
                model = model.cuda()
            if torch.cuda.is_available() and not config.get('device') == 'cpu':
                model = model.cuda()

        return model

    def _init_seg_model(self, model_dir_or_config):
        #
        in_ch = 1
        out_ch = 1
        # if self.test_mode == '3dunet':
        #     state = torch.load(model_dir_or_config)
        #     model = PadUNet3D(1, 1)
        #     model = nn.DataParallel(model)
        #     model.load_state_dict(state['state_dict'])
        # else:
        model = self._init_det_model(model_dir_or_config)
        model.eval()

        in_ch = model_dir_or_config['model'].get('in_channels')
        out_ch = model_dir_or_config['model'].get('n_classes')
        return model, in_ch, out_ch


    def obb_vol_segment(self, full_volume, rsz_volume, pred_obb_volume, t_src, target_shape, obb_scaling, target_resize= False):
        logger = get_runtime_logger()

        full_segment = np.zeros_like(full_volume, dtype=np.int32)
        origin_volume_shape = np.asarray(full_volume.shape)
        temp_pred_msk = np.argmax(pred_obb_volume, axis=0)

        small_obj_threshold = 10
        removed_small_mask = remove_small_objects(temp_pred_msk, small_obj_threshold, method='skimage')
        pred_msk = np.where(removed_small_mask, temp_pred_msk, np.zeros_like(temp_pred_msk))

        if self.debug:
            color = vtk_utils.get_rainbow_color_table(10)
            cubes = []
            for i in range(1, self.detect_out_ch):
                coords = np.argwhere(pred_msk == i)
                obb = Obb().create(coords)
                cube = obb.as_vtkcube(color=color[i])
                cubes.append(cube)

            vtk_utils.split_show([rsz_volume, pred_msk, *cubes], [pred_msk, *cubes], show=False, image_save=True,
                                 savename=self.debug_path + '_detection.png',
                                 cam_direction=(-3, -3, -2))

        labels_text = {
            1: 'left',
            2: 'right'
        }
        volume_list = [
            full_volume
        ]



        for i in range(1, self.detect_out_ch):
            coords = np.argwhere(pred_msk == i)

            logger.info(f'detecting obb segmentaion label :{labels_text.get(i)}')
            # pool_seed = []
            pool_seed = np.array([64, 32, 32])
            # pool_seed = np.array([16, 16, 16])
            coords = coords if coords.size > 3 else np.stack(np.unravel_index([1, 2, 3], pred_msk.shape), axis=-1)
            coords_in_src = image_utils.apply_trasnform_np(coords, t_src)

            obb_voles, _, warp_points, t_obb2src = image_utils.volume_sampling_from_coords(
                volume_list, coords_in_src, pool_seed=pool_seed, scale=obb_scaling,
                augment_param={}, return_warp_points=True, return_transform=True)

            if not target_resize:
                reshape_obb_vol = obb_voles[0]
                warp_points_in_src = warp_points
                t_src2obbrsz = np.linalg.inv(t_obb2src)
            else:
                obb_grids = np.stack(np.meshgrid(*[np.arange(i) for i in obb_voles[0].shape], indexing='ij'), axis=-1)
                warp_points_computed = image_utils.apply_trasnform_np(obb_grids.reshape([-1, 3]), t_obb2src)
                assert np.allclose(warp_points, warp_points_computed)

                reshape_obb_vol, t_obbrsz2obb = image_utils.auto_cropping_keep_ratio(obb_voles[0], target_shape,
                                                                                     return_transform=True)

                obbrsz_grids = np.stack(np.meshgrid(*[np.arange(i) for i in reshape_obb_vol.shape], indexing='ij'), axis=-1)

                t_src2obb = np.linalg.inv(t_obb2src)
                t_obb2obbrsz = np.linalg.inv(t_obbrsz2obb)
                t_src2obbrsz = image_utils.concat_transform([t_obb2obbrsz, t_src2obb])
                assert np.allclose(t_src2obbrsz, t_obb2obbrsz.dot(t_src2obb))
                t_obbrsz2src = np.linalg.inv(t_src2obbrsz)

                warp_points_in_src = image_utils.apply_trasnform_np(obbrsz_grids.reshape([-1, 3]), t_obbrsz2src)

                assert reshape_obb_vol.shape == tuple(target_shape)

            if self.seg_in_ch == 2 or self.seg_detection_concat:
                # warp_points_in_src
                t_src2rsz = np.linalg.inv(t_src)
                warp_points_in_rsz = image_utils.apply_trasnform_np(warp_points_in_src, t_src2rsz)
                pred_sel_prob = pred_obb_volume[i]
                pred_obb = InterpolateWrapper(pred_sel_prob)(warp_points_in_rsz)
                pred_obb = pred_obb.reshape(reshape_obb_vol.shape)


                def post_proc_concat_feat(arr):
                    return gaussian_filter(arr, 1.0)

                pred_obb = post_proc_concat_feat(pred_obb)
                inputs = np.stack([reshape_obb_vol, pred_obb], axis=0)
                if self.seg_statistics_model_concat:
                    resize_stat_models = [image_utils.auto_cropping_keep_ratio(vol, target_shape=reshape_obb_vol.shape) for vol in
                                          self.stat_models]
                    resize_stat_models = np.stack(resize_stat_models, axis=0)
                    inputs = np.concatenate([inputs, resize_stat_models])
                # = np.stack([pre])
                obb_items = torch_utils.data_convert([inputs])
            else:
                # concat 해서 처리할 때 segmentation channel = 2
                obb_items = torch_utils.data_convert([reshape_obb_vol[None]])

            with torch.no_grad():
                # if self.test_mode == '3dunet':
                #     pred_seg = self.seg_model(*obb_items, None)
                # elif self.test_mode == 'teethnet':
                pred_seg_prob = self.seg_model(*obb_items)
                pred_seg = torch.argmax(pred_seg_prob, dim=1)
                # else:
                #     raise ValueError(self.test_mode)

            # if self.test_mode == '3dunet':
            pred_obb_mask = torch_utils.to_numpy(pred_seg).astype(np.int32)
            # elif self.test_mode == 'teethnet':
            #     pred_obb_mask = torch_utils.to_numpy(torch.argmax(pred_seg, dim=1))
            # else:
            #     raise ValueError(self.test_mode)


            if self.debug:
                self.debug_items.append((pred_obb_mask, *obb_items))
                pose_str = ['back', 'left', 'right'][i]
                image_utils.show_2mask_image(reshape_obb_vol * 255, pred_obb_mask, show=False,
                                             save_path=self.debug_path + '/' + pose_str, image_save=True, create_sub_dir=False,
                                             stride=4, in_rainbow_size=2)
                # logger.info('save 2d image:{}'.format(save_image_2d_path))

            valid_restred, valid_grids = image_utils.restored_obb_segment_to_original_grid(pred_obb_mask,
                                                                                           warp_points_in_src,
                                                                                           t_src2obbrsz,
                                                                                           origin_volume_shape,
                                                                                           method='nearest')
            i0, j0, k0 = valid_grids[:, 0], valid_grids[:, 1], valid_grids[:, 2]

            label = i
            full_segment[i0, j0, k0] = valid_restred * (label)

        return full_segment

    def detection_roi(self, volume:np.ndarray, return_numpy=False) -> np.ndarray:
        if volume.ndim == 3:
            volume = volume[None]
        device = next(self.det_model.parameters()).device

        with torch.no_grad():
            volume_tensors = torch_utils.data_convert(volume, dtype='float32', device=device)
            roi_pred = self.det_model(volume_tensors)

        roi_pred = torch_utils.to_numpy(roi_pred) if return_numpy else roi_pred
        return roi_pred

    @timefn2
    def full_segment(self, full_volume:np.ndarray, spacing, obb_scaling=1.0, target_resize=True) -> np.ndarray:
        if self.debug:
            self.debug_items.clear()

        visualize = self.visualize
        target_shape = self.target_shape

        rsz_vol, t_src = image_utils.auto_cropping_keep_ratio(full_volume, [128, ] * 3, return_transform=True)
        obb_volume = self.detection_roi(rsz_vol, return_numpy=True)

        full_seg = self.obb_vol_segment(full_volume, rsz_vol, obb_volume, t_src, target_shape, obb_scaling, target_resize=target_resize)

        if visualize:
            vtk_utils.show_actors([
                full_volume * 0.75,  # 가시화를 위해 hu 값 줄이기
                *vtk_utils.auto_refinement_mask(full_seg)
            ])

            image_utils.compare_image(*[
                np.transpose(v, [1, 0, 2]) for v in [full_volume * 255, full_seg]
            ], concat_original=True, pause_sec=0.001,
                                      full_region=False, image_save=False,
                                      save_path='d:/temp/nerve',
                                      show=True)

        return full_seg
