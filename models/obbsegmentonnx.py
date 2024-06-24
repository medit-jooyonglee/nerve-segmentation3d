import os
import time
import numpy as np
import onnxruntime

from scipy.ndimage import gaussian_filter

from typing import Tuple, List

from .predict_onnx import NerveOnnx, data_convert_onnx
from .postprocess import segmentmask_to_spline
from .postprocess import remove_small_objects
from tools import vtk_utils, image_utils, InterpolateWrapper
from commons import get_runtime_logger, timefn
from commons import common_utils
# from commons.common_utils import get_stack_path

g_param = {
    'count': 0,
    'id_type': 'count'
}


def increase_runtime_id():
    g_param['count'] += 1


def post_proc_concat_feat(arr):
    return gaussian_filter(arr, 1.0)


def get_runtime_id():
    if g_param['id_type'] == 'count':
        return '{:04d}'.format(g_param['count'])
    else:
        return time.strftime("%Y%m%d%H%M%S")


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


class ObbSegmentOnnx(ObbSegmentBase):
    def __init__(self, detection_config, segment_config, det_filename, seg_filename):
        super(ObbSegmentOnnx, self).__init__(detection_config)

        self.seg_in_ch = segment_config['model'].get('in_channels')
        self.seg_out_ch = segment_config['model'].get('n_classes')

        self.seg_detection_concat = segment_config['model'].get('detection_concat', False)
        self.seg_statistics_model_concat = segment_config['model'].get('statistics_model_concat', False)

        self.detect_in_ch = detection_config['model'].get('in_channels')
        self.detect_out_ch = detection_config['model'].get('out_channels')

        det_onnx_predictor = NerveOnnx(det_filename)
        seg_onnx_predictor = NerveOnnx(seg_filename)

        self.det_onnx_predictor = det_onnx_predictor
        self.seg_onnx_predictor = seg_onnx_predictor

        self.debug = False
        self.debug_path = 'd:/temp/nerve'

        # pool 8 ---> 16 으로 변경
        # BUG: export한 모델 shape 크기가 아예 고정되어 있음. pool_seed 8 -> 16으로 변경하다보니 문제 발생. 임시로 고정크기값 다시 변경함.
        # self.target_shape = [360, 96, 56]
        # self.target_shape =

    def release(self):
        del self.det_onnx_predictor
        del self.seg_onnx_predictor

    def __del__(self):
        self.release()

    def detection_roi_binding(self, x, y):
        x = data_convert_onnx(x)
        return self.det_onnx_predictor.predict_binding(x, y)

    def obb_vol_segment_binding(self, x, y):
        x = data_convert_onnx(x)
        return self.seg_onnx_predictor.predict_binding(x, y)

    @timefn
    def detection_roi(self, input_data) -> np.ndarray:
        x = data_convert_onnx(input_data)
        return self.det_onnx_predictor.predict(x)

    @timefn
    def obb_vol_segment(self, input_data):
        x = data_convert_onnx(input_data)
        return self.seg_onnx_predictor.predict(x)

    @timefn
    def full_segment(self, full_volume: np.ndarray, spacing, obb_scaling=1.0, order='xyz',
                     binding_mode: bool = False, threshold=0.5, compute_full_segment=False, target_resize = False) -> Tuple[
        np.ndarray, List[np.ndarray]]:
        """
        full-segment & spline 좌표점 회귀 처리
        spline - segmentation  구성하는 주요 좌표점들이다.
        Parameters
        ----------
        full_volume : np.ndarray (d, h, w) volume data
        spacing : np.ndarray (3,) pixel spacing
        obb_scaling : float
        order : str
            스플라인 좌표점에 대한 순서. 볼륨기준으로 할 때는 zyx, 최종 인터페이스를 고려하면 xyz를 사용
        binding_mode :
        compute_full_segment : bool, full-segmentation computing option
        Returns
        -------
        np.ndarray :
            full segment result. same shape with full_volume
        list[np.ndarray]
            the main coordinate points that make up the curve
            #2 , (left, right)


        """
        # logger = get_runtime_logger()
        assert order in ['xyz', 'zyx']
        compute_full_segment = compute_full_segment or self.debug
        labels_text = {
            1: 'left',
            2: 'right'
        }

        pool_seed = self.pool_seed
        target_shape = self.target_shape
        logger = get_runtime_logger()


        # det_out_ch = config['model'].get('out_channels')
        det_out_ch = self.detect_out_ch

        rsz_vol, t_src = image_utils.auto_cropping_keep_ratio(full_volume, [128, ] * 3, return_transform=True)
        rsz_vol = np.expand_dims(rsz_vol, 0)
        if binding_mode:
            pred_obb_volume = np.squeeze(self.detection_roi_binding(rsz_vol, [1, 3, 128, 128, 128]), 0)
        else:
            pred_obb_volume = np.squeeze(self.detection_roi(rsz_vol), 0)
        assert pred_obb_volume.ndim == 4

        full_segment = np.zeros_like(full_volume, dtype=np.int32)
        origin_volume_shape = np.asarray(full_volume.shape)
        temp_pred_msk = np.argmax(pred_obb_volume, axis=0)
        # experimental small obejcts.
        small_obj_threshold = 8
        removed_small_mask = remove_small_objects(temp_pred_msk, small_obj_threshold, method='skimage')
        pred_msk = np.where(removed_small_mask, temp_pred_msk, np.zeros_like(temp_pred_msk))
        volume_list = [full_volume]

        if self.debug:
            subpath = '_'.join(common_utils.get_stack_path(3))
            subpath = common_utils.clean_fname(subpath)
            # try:
            savepath = os.path.join(self.debug_path, subpath)
            os.makedirs(savepath, exist_ok=True)
            # savename1 = os.path.join(savepath, f'detection_before_small_objects_{get_runtime_id()}.png')
            savename_right = os.path.join(savepath, f'detection_after_small_objects_{get_runtime_id()}_right.png')
            savename_left = os.path.join(savepath, f'detection_after_small_objects_{get_runtime_id()}_left.png')
            # vtk_utils.split_show([rsz_vol, temp_pred_msk], [temp_pred_msk], show=False, image_save=True,
            #                      cam_direction=(0.4, -.9, -0.3),
            #                      savename=savename1, view_up=(0, 0, -1))


            from tools.obb import Obb
            obb = Obb().create(np.argwhere(pred_msk == 1))

            vtk_utils.split_show([rsz_vol, pred_msk], [pred_msk], show=False,
                                 cam_direction=(0.5, -.75, 0.3), image_save=True, savename=savename_right,
                                 view_up=(0, 0, -1))

            vtk_utils.split_show([rsz_vol, pred_msk], [pred_msk], show=False,
                                 cam_direction=(-0.5, -.75, 0.3), image_save=True, savename=savename_left,
                                 view_up=(0, 0, -1))

            cubes = [Obb().create(np.argwhere(pred_msk == i)) for i in range(1, 3)]
            cube_actors = [cube.as_vtkcube() for cube in cubes]
            vtk_utils.change_actor_color(cube_actors, (0, 1, 0))
            vtk_utils.split_show([
                rsz_vol,
                pred_msk.astype(np.int32)
            ], [
                rsz_vol,
                pred_msk.astype(np.int32),
                *cube_actors
            ])
            # vtk_utils.show([pred_msk, vtk_utils.change_actor_color(obb.as_vtkcube(), (0, 1, 0))])
            # vtk_utils.show([pred_msk, vtk_utils.change_actor_color(obb.as_vtkcube(), (0, 1, 0))])
        #

        coordinates = []
        for i in range(1, det_out_ch):
            coords = np.argwhere(pred_msk == i)
            if coords.size == 0:
                coordinates.append(np.zeros([0, 3]))
                logger.error('no coordinates for detection mask:{}'.format(labels_text.get(i)))
                continue

            coords_in_src = image_utils.apply_trasnform_np(coords, t_src)

            fixed_shape = self.target_shape # if target_resize else None

            obb_target_shape = target_shape if not target_resize else None

            obb_voles, _, warp_points, t_obb2src = image_utils.volume_sampling_from_coords(
                volume_list, coords_in_src, pool_seed=pool_seed, scale=1.0,
                augment_param={}, return_warp_points=True, return_transform=True, fixed_shape=obb_target_shape)

            # reshape_obb_vol = obb_voles[0]
            # warp_points_in_src = warp_points
            # t_src2obbrsz = np.linalg.inv(t_obb2src)
            # t_obbrsz2src = t_obb2src

            if not target_resize:
                assert obb_voles[0].shape == tuple(target_shape)
                reshape_obb_vol = obb_voles[0]
                warp_points_in_src = warp_points
                t_src2obbrsz = np.linalg.inv(t_obb2src)
                t_obbrsz2src = t_obb2src
            else:
                # logger.warning('===============to be deprecated===============')

                obb_grids = np.stack(np.meshgrid(*[np.arange(i) for i in obb_voles[0].shape], indexing='ij'), axis=-1)
                warp_points_computed = image_utils.apply_trasnform_np(obb_grids.reshape([-1, 3]), t_obb2src)
                assert np.allclose(warp_points, warp_points_computed)

                reshape_obb_vol, t_obbrsz2obb = image_utils.auto_cropping_keep_ratio(obb_voles[0], target_shape,
                                                                                     return_transform=True)
                t_obb2obbrsz = np.linalg.inv(t_obbrsz2obb)
                assert reshape_obb_vol.shape == tuple(target_shape)

                ##
                obbrsz_grids = np.stack(np.meshgrid(*[np.arange(i) for i in reshape_obb_vol.shape], indexing='ij'),
                                        axis=-1)

                t_src2obb = np.linalg.inv(t_obb2src)
                t_src2obbrsz = image_utils.concat_transform([t_obb2obbrsz, t_src2obb])
                assert np.allclose(t_src2obbrsz, t_obb2obbrsz.dot(t_src2obb))
                t_obbrsz2src = np.linalg.inv(t_src2obbrsz)

                warp_points_in_src = image_utils.apply_trasnform_np(obbrsz_grids.reshape([-1, 3]), t_obbrsz2src)


            if self.seg_in_ch == 2 or self.seg_detection_concat:
                t_src2rsz = np.linalg.inv(t_src)
                warp_points_in_rsz = image_utils.apply_trasnform_np(warp_points_in_src, t_src2rsz)
                pred_sel_prob = pred_obb_volume[i]
                pred_obb = InterpolateWrapper(pred_sel_prob)(warp_points_in_rsz)
                pred_obb = pred_obb.reshape(reshape_obb_vol.shape)
                pred_obb = post_proc_concat_feat(pred_obb)
                obb_items = np.stack([reshape_obb_vol, pred_obb], axis=0)

                if self.seg_statistics_model_concat:
                    raise NotImplementedError
            else:
                # obb_items = utils.data_convert(reshape_obb_vol)
                obb_items = np.expand_dims(reshape_obb_vol, 0)
            # @timefn2
            def obb_vol_seg():
                if binding_mode:
                    return self.obb_vol_segment_binding(obb_items, [1, 2, 360, 96, 56])
                else:
                    return self.obb_vol_segment(obb_items)


            pred_seg = obb_vol_seg()
            pred_seg = np.squeeze(pred_seg)[1, :, :, :]
            assert pred_seg.ndim == 3

            pred_obb_mask = (pred_seg > threshold).astype(np.int32)



            obb_coords = segmentmask_to_spline(pred_obb_mask, order='zyx', remove_outlier=True)
            obb_coords_in_src = image_utils.apply_trasnform_np(obb_coords, t_obbrsz2src)
            coordinates.append(obb_coords_in_src)

            if compute_full_segment:
                valid_restred, valid_grids = image_utils.restored_obb_segment_to_original_grid(pred_obb_mask,
                                                                                               warp_points_in_src,
                                                                                               t_src2obbrsz,
                                                                                               origin_volume_shape,
                                                                                               method='nearest')
                i0, j0, k0 = valid_grids[:, 0], valid_grids[:, 1], valid_grids[:, 2]

                label = 3 * i
                full_segment[i0, j0, k0] = valid_restred * label

            if self.debug:
                subpath = '_'.join(common_utils.get_stack_path(3))
                subpath = common_utils.clean_fname(subpath)
                # try:
                savepath = os.path.join(self.debug_path, subpath)
                os.makedirs(savepath, exist_ok=True)

                image_utils.compare_image(reshape_obb_vol * 255, pred_obb_mask, show=False, image_save=True,
                                          save_path=os.path.join(savepath, labels_text.get(i)))

        if order == 'xyz':
            coordinates = [v[:, ::-1] for v in coordinates]

        if self.debug:

            savename = os.path.join(savepath, f'nerve_segment_result_{get_runtime_id()}' + '{}.png')
            spheres = []
            # for p in coordinates]
            for p in coordinates:
                spheres.extend(vtk_utils.create_sphere(p, 6.5))

            vtk_utils.split_show([
                full_volume, full_segment
            ], [
                *[vtk_utils.create_curve_actor(p, line_width=6) for p in coordinates],
                *spheres
            ], item3=[
                full_segment,
            ], show=False,
                image_save=True,
                view_up=(0, 0, -1),
                cam_direction=(0.5, -.75, 0.3),
                savename=savename.format('right'))

            vtk_utils.split_show([
                full_volume, full_segment
            ], [
                *[vtk_utils.create_curve_actor(p, line_width=6) for p in coordinates],
                *spheres
            ], item3=[
                full_segment,
            ], show=False,
                image_save=True,
                view_up=(0, 0, -1),
                cam_direction=(-0.5, -.75, 0.3),
                savename=savename.format('left'))

        increase_runtime_id()

        return full_segment, coordinates
