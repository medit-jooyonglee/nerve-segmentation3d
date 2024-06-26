import importlib
import os.path
import time
from scipy.ndimage.morphology import grey_dilation
from sklearn.decomposition import PCA
import numpy as np
import torch
import mlflow

from tools import vtk_utils, image_utils
from trainer import torch_utils
from tools.obb import Obb
from tools import geometry_numpy


def get_function(function_name, modules):
    for module in modules:
        m = importlib.import_module(module)
        func = getattr(m, function_name, None)
        if func is not None:
            return func
    return RuntimeError(f'unsupported function name:{function_name}')


def get_image_saver(config):
    modules_list = [
        'teethnet.models.pytorchyolo3d.utils.imagesaver',
        'teethnet.models.unet3d.imagesaver'

    ]

    try:
        funcs = get_function(config['name'], modules_list)
    except Exception as e:
        funcs = None

    return funcs


def _image_save_seg_offset(inputs, targets, outputs, savepath, prefix=''):
    pred_seg, pred_offset = outputs
    target_seg, t_offset = targets
    t_seg, t_mask = target_seg.split(1, dim=1)

    t_seg = t_seg.squeeze(1)
    # n_classes = pred_seg.size(1)

    pred_seg_mask = torch.argmax(pred_seg, dim=1)
    # t_seg_onehot = torch.nn.functional.one_hot(t_seg, n_classes).permute(
    #     [0, t_seg.ndim, *tuple(range(1, t_seg.ndim))]).contiguous().to(pred_seg.dtype)

    # offset = (seg > self.seg_threshold ).to(offset.dtype) * offset
    # seg_dice = self.dice(pred_seg, t_seg_onehot)
    repeats = torch.div(torch.tensor(t_offset.size()), torch.tensor(t_mask.size()), rounding_mode='trunc')
    t_mask_repeat = t_mask.repeat(*repeats)

    # (
    ijk = torch.meshgrid(*[torch.arange(i, device=t_offset.device) for i in t_offset.shape[2:]], indexing='ij')
    # and batch expansion
    ijk = torch.stack(ijk, dim=0)[None]

    # coords_scale = torch.tensor(pred_seg.shape[2:])

    coords_scale = torch.tensor(pred_seg.shape[2:], device=pred_seg.device)
    coords_scale = coords_scale.reshape([-1, *[1 for _ in range(3)]])

    denorm_offset = pred_offset * coords_scale
    pred_center = ijk + denorm_offset
    gt_center = ijk + t_offset * coords_scale
    pred_center = pred_center[..., pred_seg_mask[0] > 0]
    gt_center = gt_center[..., t_seg[0] > 0]
    pred_seg_prob = pred_seg[..., pred_seg_mask[0] > 0]

    fnamebase = time.strftime('%Y%m%d%H%M%S') + '_' + prefix
    # fnamebase = str
    # '_'.join(time_str, prefix])

    pred_seg_mask, pred_seg_prob, t_seg, pred_center, gt_center, offset = torch_utils.to_numpy(
        [pred_seg_mask, pred_seg_prob, t_seg, pred_center, gt_center, denorm_offset])
    # unique_gt_center = np.unique(gt_center, axis=-1).T

    cenroids, maks_per_label, coords_per_label = image_utils.fast_cluster(pred_seg_mask, offset)
    obbs = [Obb().create(v) for k, v in coords_per_label.items() if v.shape[0] > 3]

    savename = os.path.join(savepath, fnamebase + 'seg.png')
    vtk_utils.split_show([
        pred_seg_mask,
    ], [
        t_seg
    ], show=False, image_save=True, savename=savename)

    mlflow.log_artifact(savename)

    # vtk_utils.show_pc()
    savename = os.path.join(savepath, fnamebase + 'offset.png')
    vtk_utils.split_show([
        # vtk_utils.show_pc(pred_center.T[:, ::-1], pred_seg_prob[1], is_show=False) if pred_center.size > 0 else []
        vtk_utils.create_points_actor(pred_center.T[:, ::-1], point_size=4,
                                      color_value=(1, 0, 0)) if pred_center.size > 0 else gt_center.T[:, ::-1]
        # vtk_utils.create_points_actor(gt_center.T[:, ::-1], point_size=5, color_value=(0, 1, 0))

    ], [
        vtk_utils.create_points_actor(gt_center.T[:, ::-1], point_size=5, color_value=(0, 1, 0))
    ], item3=[
        vtk_utils.auto_refinement_mask(maks_per_label, max_trunc=100),
        [obb.as_vtkcube() for obb in obbs],
    ],
        show=False, image_save=True, savename=savename,
        rgba=False)

    mlflow.log_artifact(savename)


def image_save_center_skeleton_segmentation(inputs, targets, outputs, save_path='d:/temp/'):
    assert len(outputs) == len(targets)
    for i, (pred, target) in enumerate(zip(outputs, targets)):
        _image_save_seg_offset(inputs, target, pred, save_path, f'{i:03d}')




def _image_save_mesh_seg_offset(inputs, targets, outputs, savepath, prefix=''):
    pred_seg, pred_offset = outputs
    target_seg, t_offset = targets
    t_seg, t_mask = target_seg.split(1, dim=1)

    t_seg = t_seg.squeeze(1)
    # n_classes = pred_seg.size(1)

    pred_seg_mask = torch.argmax(pred_seg, dim=1)
    # t_seg_onehot = torch.nn.functional.one_hot(t_seg, n_classes).permute(
    #     [0, t_seg.ndim, *tuple(range(1, t_seg.ndim))]).contiguous().to(pred_seg.dtype)

    # offset = (seg > self.seg_threshold ).to(offset.dtype) * offset
    # seg_dice = self.dice(pred_seg, t_seg_onehot)
    repeats = torch.div(torch.tensor(t_offset.size()), torch.tensor(t_mask.size()), rounding_mode='trunc')
    t_mask_repeat = t_mask.repeat(*repeats)

    # (
    ijk = torch.meshgrid(*[torch.arange(i, device=t_offset.device) for i in t_offset.shape[2:]], indexing='ij')
    # and batch expansion
    ijk = torch.stack(ijk, dim=0)[None]

    # coords_scale = torch.tensor(pred_seg.shape[2:])

    coords_scale = torch.tensor(pred_seg.shape[2:], device=pred_seg.device)
    coords_scale = coords_scale.reshape([-1, *[1 for _ in range(3)]])

    # pred_seg_mask = torch.where(t_seg > 0, pred_seg_mask, torch.zeros_like(pred_seg_mask))
    pred_seg_mask = torch.where(t_mask.squeeze() > 0, pred_seg_mask, torch.zeros_like(pred_seg_mask))


    denorm_offset = pred_offset * coords_scale
    pred_center = ijk + denorm_offset
    gt_center = ijk + t_offset * coords_scale
    pred_center = pred_center[..., pred_seg_mask[0] == 1]
    gt_center = gt_center[..., t_seg[0] == 1]
    pred_seg_prob = pred_seg[..., pred_seg_mask[0] == 1]

    fnamebase = time.strftime('%Y%m%d%H%M%S') + '_' + prefix
    # fnamebase = str
    # '_'.join(time_str, prefix])

    pred_seg_mask, pred_seg_prob, t_seg, pred_center, gt_center, offset = torch_utils.to_numpy(
        [pred_seg_mask, pred_seg_prob, t_seg, pred_center, gt_center, denorm_offset])

    pose = np.argwhere(t_seg > 0)
    pca = PCA().fit(pose)
    # z, y, x ---> x, y, z
    cam_direct = pca.components_[-1][::-1]

    # pred_seg_mask = np.where(t_seg > 0, pred_seg_mask, np.zeros_like(pred_seg_mask))
    # unique_gt_center = np.unique(gt_center, axis=-1).T

    cenroids, maks_per_label, coords_per_label = image_utils.fast_cluster(pred_seg_mask, offset)
    obbs = [Obb().create(v) for k, v in coords_per_label.items() if v.shape[0] > 3]

    maks_per_label = grey_dilation(maks_per_label, 2)

    savename = os.path.join(savepath, fnamebase + 'seg.png')
    vtk_utils.split_show([
        pred_seg_mask,
    ], [
        t_seg
    ], show=False, image_save=True, savename=savename, cam_direction=cam_direct)

    mlflow.log_artifact(savename)
    # vtk_utils.show_pc(pred_center.T, pred_seg_prob[1], point_size=2)
    pred_pts, pred_prob = (pred_center.T[:, ::-1], pred_seg_prob[1]) if pred_center.size > 10 else (gt_center.T[:, ::-1], np.ones([gt_center.shape[-1]]))
    # vtk_utils.show_pc()
    savename = os.path.join(savepath, fnamebase + 'offset.png')
    vtk_utils.split_show([
        vtk_utils.show_pc(pred_pts, pred_prob , is_show=False, point_size=2)
        # vtk_utils.create_points_actor(pred_center.T[:, ::-1], point_size=4,
        #                               color_value=(1, 0, 0)) if pred_center.size > 0 else gt_center.T[:, ::-1]
        # vtk_utils.create_points_actor(gt_center.T[:, ::-1], point_size=5, color_value=(0, 1, 0))

    ], [
        vtk_utils.create_points_actor(gt_center.T[:, ::-1], point_size=5, color_value=(0, 1, 0))
    ], item3=[
        vtk_utils.auto_refinement_mask(maks_per_label, max_trunc=100),
        [obb.as_vtkcube() for obb in obbs],
    ],
        show=False, image_save=True, savename=savename,
        rgba=False, cam_direction=cam_direct)

    mlflow.log_artifact(savename)


def image_save_center_mesh_segmentation(inputs, targets, outputs, save_path='d:/temp/'):
    assert len(outputs) == len(targets)
    for i, (pred, target) in enumerate(zip(outputs, targets)):
        _image_save_mesh_seg_offset(inputs, target, pred, save_path, f'{i:03d}')



def image_save_segmentation(inputs, targets, outputs, save_path='d:/temp'):
    x_input, y_true, y_pred = torch_utils.to_numpy([inputs, targets, torch.argmax(outputs, dim=1)])

    fnamebase = time.strftime('%Y%m%d%H%M%S')

    # fnamebase = str
    # '_'.join(time_str, prefix])
    inputs_as_list = x_input if isinstance(x_input, (list, tuple)) else [x_input]
    y_pred_actors = vtk_utils.auto_refinement_mask(y_pred)
    y_true_actors = vtk_utils.auto_refinement_mask(y_true)

    vtk_utils.split_show([
        *inputs_as_list,
    ], [
        *y_pred_actors,
    ], item3=[
        *y_true_actors,
    ], show=False, image_save=True, savename=os.path.join(save_path, fnamebase + '_right' + '.png'),
        cam_direction=(5, -1, 2))

    vtk_utils.split_show([
        *inputs_as_list,
    ], [
        *y_pred_actors,
    ], item3=[
        *y_true_actors,
    ], show=False, image_save=True, savename=os.path.join(save_path, fnamebase + '_left' + '.png'),
        cam_direction=(-5, -1, 2))

    vtk_utils.split_show([
        *inputs_as_list,
    ], [
        *y_pred_actors,
    ], item3=[
        *y_true_actors,
    ], show=False, image_save=True, savename=os.path.join(save_path, fnamebase + '_front' + '.png'),
        cam_direction=(.1, -5, .1))


def mesh_voxelize_segment_classification(inputs, targets, outputs, save_path='d:/temp'):
    # inputs, targets, outputs, save_path = 'd:/temp'
    inputs0, targets0, outputs0 = torch_utils.squeeze(
        torch_utils.to_numpy([inputs, targets, outputs])
    )

    (roi_volumes, input_obb_meta), (_, stat_obb_meta) = inputs0

    obbs = [Obb().from_meta(v) for v in input_obb_meta]
    pool_shape = roi_volumes.shape[2:]

    # dataset label to fdis
    fdis = np.concatenate([np.arange(11, 19), np.arange(21, 29), np.arange(31, 39), np.arange(41, 49)], axis=0)
    label2fdi = np.zeros([50], dtype=np.int64)
    label2fdi[np.arange(1, 1 + fdis.size)] = fdis
    target_mask, target_class = targets0

    pred_mask, pred_class = outputs0
    pred_label = np.argmax(pred_class, axis=-1)
    pred_mask = np.argmax(pred_mask, axis=1)

    num_target = target_mask.shape[0]
    target_fdi_label = label2fdi[target_class[:num_target]]
    pred_fdi_label = label2fdi[pred_label[:num_target]]
    pred_res, gt_res = [], []

    def mask_to_actors(masks, inmat):
        actors = vtk_utils.auto_refinement_mask(masks)
        for a in actors:
            a.SetUserTransform(vtk_utils.myTransform(geometry_numpy.reverse_transform(inmat)))
        return actors

    for obb, pmask, tmask in zip(obbs, pred_mask, target_mask):
        # TODO: the scaling the of dataset
        pts, mat = obb.pooling_points(pool_shape, 1.1, return_transform=True)
        gt_res.extend(mask_to_actors(tmask, mat))
        # if a:
        pred_res.extend(mask_to_actors(pmask ,mat))



    def coloring_posit_actors(actors, labels):

        labels = labels[:num_target]
        posit_actors = [a for i, a in enumerate(actors) if labels[i] > 0]
        posit_labels = labels[labels > 0]
        vtk_utils.change_teeth_color(posit_actors, posit_labels)
        return posit_actors

    pred_actors = coloring_posit_actors(pred_res, pred_fdi_label)
    gt_actors = coloring_posit_actors(gt_res, target_fdi_label)
    # vtk_utils.show_actors([res])
    # vtk_utils.numpyvolume2vtkvolume()
    savefilename = os.path.join(save_path, f'{time.strftime("%Y%m%d%H%M%S")}_seg.png')
    vtk_utils.split_show([
        pred_actors
    ], [
        gt_actors
    ], show=False, image_save=True,
        savename=savefilename, cam_direction=(0, 0, 1),
    )

    mlflow.log_artifact(savefilename)








