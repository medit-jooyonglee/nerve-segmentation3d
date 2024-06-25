import os.path
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation
from sklearn.neighbors import NearestNeighbors
from typing import List, Union

from commons import timefn2, get_runtime_logger, timefn

import vtk_utils
from tools import InterpolateWrapper, vtk_utils, utils_numpy
from tools import geometry_numpy
from tools.geometry_numpy import decompose_general_boxes, aabb2obb, obb2aabb, align_transform

global_param_dict = {}

def init_params():
    augment_param = {
        "translate": np.array([0.04, 0.005, 0.005]),
        "rotation": np.array([np.pi / 25, np.pi / 80, np.pi / 80]),
        "scaling": np.array([1.1, 1.1, 1.1]),
        "cropping":True,
        "name": ""
    }
    return augment_param

def get_augment_param(name="training"):
    if not name in global_param_dict:
        param = init_params()
        if name == "inference":
            param.clear()
        global_param_dict[name] =  param
    return global_param_dict[name]

# def set_augment_param(mode):
#     assert mode in ["training", "inference"]
#     if mode == "inference":
#         augment_param.clear()


def translation_mat(translation):
    dim = translation.size
    mat = np.eye(dim + 1)
    mat[:dim, dim] = translation
    return mat

def scaling_mat(scale):
    scale = np.asarray(scale)
    dim = scale.size
    mat = np.eye(dim + 1)
    mat[:dim, :dim] = np.diag(scale)
    return mat


def rotation_mat(rot):
    dim = rot.shape[0]
    mat = np.eye(dim + 1)
    mat[:dim, :dim] = rot
    return mat

def create_transform(rot, trans):
    basisT = rot
    transform_mat = np.eye(4)
    transform_mat[:3, :3] = basisT
    transform_mat[:3, 3] = -np.dot(basisT, trans)
    return transform_mat


def concat_transform(transform_list, in_order=False):

    """
    transform list [t1, t2, ...tn]
    if in_order is True
        res = t1 * t2 * ....l * tn
    else
        res = tn * .... * t2 * t1


    symbol * is matrix-multiplication

    :param transform_list: list of array [4,4]
    :return:
    """
    mat = np.eye(4)
    if in_order:
        for t in transform_list:
            mat = np.dot(mat, t)
    else:
        for t in reversed(transform_list):
            mat = np.dot(t, mat)

    return mat


def norm_obb(obbmeta, shape):
    centers, size, theta = decompose_genral_boxes(obbmeta)
    pose_scale = np.asarray(shape) - 1.0

    norm_centers = centers / pose_scale
    norm_size = size / pose_scale
    meta = np.concatenate([norm_centers, norm_size, theta], axis=-1)
    return meta


def denorm_obb(obbmeta, shape):
    centers, size, theta = decompose_genral_boxes(obbmeta)
    pose_scale = np.asarray(shape) - 1.0

    norm_centers = centers * pose_scale
    denorm_size = size * pose_scale
    meta = np.concatenate([norm_centers, denorm_size, theta], axis=-1)
    return meta

def compose_general_boxes(center, size, theta):
    return  np.concatenate([
            center, size, theta
        ], axis=-1)


def decompose_genral_boxes(general_boxes):
    assert general_boxes.shape[-1] == 9
    if general_boxes.ndim == 1:
        return general_boxes[:3], general_boxes[3:6], general_boxes[6:]
    elif general_boxes.ndim == 2:
        return general_boxes[:, :3], general_boxes[:, 3:6], general_boxes[:, 6:]
    else:
        raise ValueError(general_boxes.shape)




def create_matrix_from_euler(theta):
    """
    :param theta: [N,3] or [3] array [N,(wz,wy,wx)]
    :return: [N,3,3] or [3,3] transform matrix
    """
    wz, wy, wx = np.split(theta, 3, axis=-1)
    # [N, 1]
    cosz = np.cos(wz)
    cosy = np.cos(wy)
    cosx = np.cos(wx)

    sinz = np.sin(wz)
    siny = np.sin(wy)
    sinx = np.sin(wx)

    zero = np.zeros_like(wz)
    one = np.ones_like(wz)

    # [3, 3, N, 1]
    rotz = np.array([
        [cosz, -sinz, zero],
        [sinz, cosz, zero],
        [zero, zero, one],
    ]
    )

    roty = np.array([
        [cosy, zero, siny],
        [zero, one, zero],
        [-siny, zero, cosy],
    ]
    )

    rotx = np.array([
        [one, zero, zero],
        [zero, cosx, -sinx],
        [zero, sinx, cosx]
    ]
    )

    # [3, 3, N, 1]-----> [3, 3, N]
    rotz, roty, rotx = [np.squeeze(p, axis=-1) for p in [rotz, roty, rotx]]
    if theta.ndim == 2:
        # [ N, 3, 3]
        rotz, roty, rotx = [np.transpose(p, [2, 0, 1]) for p in [rotz, roty, rotx]]

    rot = np.matmul(np.matmul(rotz, roty), rotx)
    if theta.ndim == 2:
        # affmat = np.repeat(np.expand_dims(rot, axis=0), rot.shape[0], 0)
        affmat = np.repeat(np.expand_dims(np.eye(4), axis=0), rot.shape[0], 0)
        affmat[:, :3, :3] = rot
    else:
        affmat = np.eye(4)
        affmat[:3, :3] = rot

    return affmat

#
# def crop_and_resize(volume, norm_boundary, crop_shape, return_interp_func=False,
#              method="linear",
#              param=dict()):
#     """
#     norm_boundary 기준으로 param 으로 augmentation 처리되어 cropping & resize한다
#     pixel-spacing-그러니깐 스케일링은 crop_spacing / volume_spacing으로 결전된다.
#
#     Parameters
#     ----------
#     volume : np.ndarray
#         [d, h, w] volume
#     norm_boundary : np.ndarray
#         axis-alienged bounding box,normalized [6] [z1,y1,x1,z2,y2,x2]
#     crop_shape : np.ndarray or tuple(int)
#         (3,) target shape (d, h, w)
#     return_interp_func :
#     method : str
#         'linear' or 'nearest'
#     param : dict
#         augment paramegers
#         'rotation':
#         'scale':
#         'translation':
#     Returns
#     -------
#          resampled voxel : all the same shape voxel,
#         affine tranform matrix : [4,4] from resample-domaiin to source-domain
#     """
#     """
#
#     :param volume:
#     :param norm_boundary:
#     :param crop_shape: int type 3 tuple, crop shape
#     :return:
#
#     """
#     logger = get_runtime_logger()
#
#     # convert aabb to obb-meta
#     p1, p2 = norm_boundary[:3], norm_boundary[3:]
#     center = (p1 + p2)/2.
#     fsize = (p2 - p1)
#
#     volume_shape = np.array(volume.shape)
#     crop_shape = np.asarray(crop_shape)
#
#
#     box_ratio = np.max(volume_shape) / volume_shape
#     auto_cropping_keep_ratio_with_box()
#     #
#     # np.argmax(fsize)
#     # np.max(crop_shape)
#     # crop_box_shape = np.full(fsize.shape, np.max(fsize), fsize.dtype) * box_ratio #volume_shape / vo / scaling * scale_weights
#     # scale_w = param.get("scale", np.ones([3]))
#     # crop_box_shape = crop_box_shape * scale_w
#     # translate_param = param.get("translate", np.zeros([3]))
#     # theta = param.get("rotation", np.zeros([3]))
#     shift = fsize * translate_param
#     # shift = np.zeros([3])
#
#     crop_obb = np.concatenate([center + shift, fsize, theta])
#     crop_obb[3:6] = crop_box_shape
#
#     denormed_obb = denorm_obb(crop_obb[np.newaxis], volume.shape)[0]
#     #target_shape = [np.max(volume.shape)] * 3
#
#     # if return_interp_func:
#     interp_func = InterpolateWrapper(volume, method=method, bounds_error=True, fill_value=0)
#
#     # get transform from resample-domain to source-domain
#     crop_mask, transform_to_source = voi_align_from_obb(volume, crop_obb[np.newaxis],
#                                                         volume_shape,
#                                                         crop_shape,
#                                                         interpolate_func=interp_func,
#                                                         return_affine_transform=True)
#     crop_mask = np.squeeze(crop_mask)
#     crop_mask = crop_mask if method == "linear" else crop_mask.astype('int64')
#     transform_to_source = np.squeeze(transform_to_source)
#
#     out = [crop_mask, transform_to_source, denormed_obb]
#     if return_interp_func:
#         out.append(interp_func)
#
#     return out



def cropping(volume, norm_boundary, volume_spacing, crop_shape, crop_spacing, return_interp_func=False,
             method="linear",
             param=dict()):
    """
    norm_boundary 기준으로 param 으로 augmentation 처리되어 cropping & resize한다
    pixel-spacing-그러니깐 스케일링은 crop_spacing / volume_spacing으로 결전된다.
    :param volume: [d, h, w]
    :param norm_boundary: axis-alienged bounding box,normalized [6] [z1,y1,x1,z2,y2,x2]
    :param volume_spacing: mm/pixels,
    :param crop_shape: int type 3 tuple, crop shape
    :param crop_spacing:
    :param roi_crop: bool if true, cropping from norm_boundary, if false, cropping entire dimension[0.., 1...]
    :return:
        resampled voxel : all the same shape voxel,
        affine tranform matrix : [4,4] from resample-domaiin to source-domain
    """
    logger = get_runtime_logger()
    logger.warning('legacy,.. use crop_and_resize(...)')
    if np.max(norm_boundary) > 1.:
        logger.warning('normalized boundary greater than 1:{}'.format(norm_boundary))
        # norm_boundary = np.clip(norm_boundary, 0, 1.)
    # en_roi_crop = param.get('roi_crop', True)
    # if en_roi_crop:
    #     if np.max(norm_boundary) > 1.:
    #         logger.error('normalized boundary greater than 1:{}'.format(norm_boundary))
    #         norm_boundary = np.clip(norm_boundary, 0, 1.)
    #     else:
    #         pass
    # else:
    #     norm_boundary = np.concatenate([np.zeros([3]), np.ones([3])])
    # print(norm_boundary)
    # assert np.max(norm_boundary) < 1., "make sure boundary box to be normalized"
    # convert aabb to obb-meta
    p1, p2 = norm_boundary[:3], norm_boundary[3:]
    center = (p1 + p2)/2.
    fsize = (p2 - p1)

    volume_shape = np.array(volume.shape)
    crop_shape = np.asarray(crop_shape)
    # compute crop_box-shape by resampling-spacing
    # scaling = crop_spacing / np.max(volume_spacing)
    # crop_box_shape = (crop_shape * scaling) / np.max(volume.shape)
    # crop_box_shape = np.ones_like(crop_box_shape) * np.max(crop_box_shape)
    # weights for all the same length bbox
    box_ratio = np.max(volume_shape) / volume_shape
    #
    # np.argmax(fsize)
    # np.max(crop_shape)
    crop_box_shape = np.full(fsize.shape, np.max(fsize), fsize.dtype) * box_ratio #volume_shape / vo / scaling * scale_weights
    scale_w = param.get("scale", np.ones([3]))
    crop_box_shape = crop_box_shape * scale_w
    translate_param = param.get("translate", np.zeros([3]))
    theta = param.get("rotation", np.zeros([3]))
    shift = fsize * translate_param
    # shift = np.zeros([3])

    crop_obb = np.concatenate([center + shift, fsize, theta])
    crop_obb[3:6] = crop_box_shape

    denormed_obb = denorm_obb(crop_obb[np.newaxis], volume.shape)[0]
    #target_shape = [np.max(volume.shape)] * 3

    # if return_interp_func:
    interp_func = InterpolateWrapper(volume, method=method, bounds_error=True, fill_value=0)


    # get transform from resample-domain to source-domain
    crop_mask, transform_to_source = voi_align_from_obb(volume, crop_obb[np.newaxis],
                                                         volume_shape,
                                                         crop_shape,
                                                         interpolate_func=interp_func,
                                                        return_affine_transform=True)
    crop_mask = np.squeeze(crop_mask)
    crop_mask = crop_mask if method == "linear" else crop_mask.astype('int64')
    transform_to_source = np.squeeze(transform_to_source)

    out = [crop_mask, transform_to_source, denormed_obb]
    if return_interp_func:
        out.append(interp_func)

    return out


def auto_cropping_scale(volume, scale:float, return_transform=False):
    src_shape = np.asarray(volume.shape)
    target_shape = np.ceil(scale * src_shape).astype('int32')


def param2matrix(augent_param, denorm_trans=1.0):
    """

    Parameters
    ----------
    augent_param :

    Returns
    -------
        (rotation 4x4, transation 4x4, scale (

    """
    aug_euler = augent_param.get('rotate', np.zeros([3]))
    aug_scale = augent_param.get('scale', np.ones([3]))
    aug_trans = augent_param.get('translate', np.zeros([3]))

    aug_rot = Rotation.from_euler('zyx', aug_euler, degrees=False).as_matrix()
    # aug_trans = translation_mat(aug_trans)
    return rotation_mat(aug_rot), translation_mat(aug_trans * denorm_trans), scaling_mat(aug_scale)
    # concat_transform([aug_rot])


def auto_cropping_keep_ratio_with_box_augment(volume:np.ndarray, crop_box, target_shape, return_transform=False, augent_param=dict(), method='linear'):
    """
    auto_cropping_keep_ratio_with_box(...) + augmentation
    Parameters
    ----------
    volume : np.ndarray
        (d, h, w) volume array
    crop_box : np.ndarray
        (6,) aabb. in zyx order
    target_shape : (3,)
        target shape
    return_transform : bool

    augent_param : dict
        rotate: (3,)
        translate: (3,) normalize translation with reference to the size of the crop_box
        scale: (3,)

    Returns
    -------
        np.ndarray
            crop & resized volume
        np.ndarray optional
            transform 4x4 matrix

    """
    target_shape = np.asarray(target_shape)
    src_shape = np.array(volume.shape)

    crop_size = crop_box[3:] - crop_box[:3]
    crop_ctr = (crop_box[3:] + crop_box[:3])/2.
    max_i = np.argmax(crop_size)
    # ratio = target_shape[max_i] / src_shape[max_i]
    aug_r, aug_t, aug_s = param2matrix(augent_param, denorm_trans=crop_size)

    _p0, p0 = translation_mat(-crop_ctr), translation_mat(crop_ctr)
    # aug_r_pivot = concat_transform([p0, aug_r, _p0])
    # (d, h, w, 3)
    pose = np.stack(np.meshgrid(*[np.arange(i) for i in target_shape], indexing='ij'), axis=-1)

    scale = (crop_size[max_i]) / (target_shape[max_i] -1)
    translate = crop_size/2 - (target_shape-1) * scale / 2 + crop_box[:3]
    # translate = ?-(crop_size[max_i] - crop_size)/2 + crop_box[:3]
    pose = pose.reshape([-1, 3])
    # warp_pose = pose * scale + translate
    s0 = scaling_mat([scale,]*3)
    t0 = translation_mat(translate)
    # t1 = concat_transform([t0, s0, _p0, aug_r, aug_s, p0, aug_t], True)
    t1 = concat_transform([aug_t, p0, aug_s, aug_r, _p0, t0, s0])
    # t1 = concat_transform([t0, s0], True)

    warp_pose = apply_trasnform_np(pose, t1)

    wrapper = InterpolateWrapper(volume, method=method)
    res = wrapper(warp_pose.reshape([-1, 3]))
    res_vol = res.reshape(target_shape)
    if return_transform:
        return res_vol, t1
    else:
        return res_vol



def auto_cropping_keep_ratio_with_box(volume:np.ndarray, crop_box, target_shape, return_transform=False, method='linear'):
    assert method in ['linear', 'nearest']
    target_shape = np.asarray(target_shape)
    src_shape = np.array(volume.shape)

    crop_size = crop_box[3:] - crop_box[:3]
    max_i = np.argmax(crop_size)
    # ratio = target_shape[max_i] / src_shape[max_i]

    # (d, h, w, 3)
    pose = np.stack(np.meshgrid(*[np.arange(i) for i in target_shape], indexing='ij'), axis=-1)

    scale = (crop_size[max_i]) / (target_shape[max_i] -1)
    translate = crop_size/2 - (target_shape-1) * scale / 2 + crop_box[:3]
    # translate = ?-(crop_size[max_i] - crop_size)/2 + crop_box[:3]
    pose = pose.reshape([-1, 3])
    warp_pose = pose * scale + translate
    s0 = scaling_mat([scale,]*3)
    t0 = translation_mat(translate)
    t1 = concat_transform([t0, s0], True)

    wrapper = InterpolateWrapper(volume, method=method)
    res = wrapper(warp_pose.reshape([-1, 3]))
    res_vol = res.reshape(target_shape)
    if return_transform:
        return res_vol, t1
    else:
        return res_vol


def compute_target_shape(shape, scale0):
    """
    keep_ratio resize를 위한 scale값 계산과 target_shape 을 계산
    target_shape은 shape 전 영역을 커버되야한다.
    shape 에 대한
    Parameters
    ----------
    shape : 입력 shape
    scale0 : flaot 초기 스케일값

    Returns
    -------
        target_shape : scale된 target_shape
        scale : corretion scale
    """
    shape = np.asarray(shape)
    i = np.argmax(shape)

    m = np.ceil((shape[i] - 1) * scale0 + 1).astype(np.int32)
    scale = (m-1)/(shape[i]-1)
    target_shape = np.ceil((shape - 1) * scale + 1).astype(np.int32)
    return target_shape, scale


def auto_cropping_keep_ratio(volume:np.ndarray, target_shape, return_transform=False, method='linear'):
    target_shape = np.asarray(target_shape)
    src_shape = np.array(volume.shape)
    max_i = np.argmax(src_shape)
    # ratio = target_shape[max_i] / src_shape[max_i]

    # (d, h, w, 3)
    pose = np.stack(np.meshgrid(*[np.arange(i) for i in target_shape], indexing='ij'), axis=-1)
    # norm_pose = pose /
    # src_center = (src_shape - 1)/2

    scale = ( src_shape[max_i] - 1) / ( target_shape[max_i] -1)
    # set origin center
    translate = (src_shape - 1)/2 - (target_shape - 1) * scale / 2
    pose = pose.reshape([-1, 3])
    warp_pose = pose * scale + translate
    s0 = scaling_mat([scale,]*3)
    t0 = translation_mat(translate)
    t1 = concat_transform([t0, s0], True)
    # p2 = apply_trasnform_np(pose, T0)
    # assert np.isclose(p2, warp_pose).all()

    wrapper = InterpolateWrapper(volume, method=method)
    res = wrapper(warp_pose.reshape([-1, 3]))
    res_vol = res.reshape(target_shape)
    if return_transform:
        return res_vol, t1
    else:
        return res_vol

    # InterpolateWrapper



def toedgemap(mask, method="nearest"):
    """
    :param masks: [d, h, w] 3dim array, binary array, background - zero, foreground - nonezero
    :param blur_sig:
    :return:
    """

    # b1 = gaussian_filter(mask, 1.0)
    # b2 = gaussian_filter(mask, 2.0)
    # edge_mask_blur = np.abs(b1 - b2) > 0.1
    #
    # return edge_mask_blur
    # neighbor index in 3d
    shifts = []
    for i in range(8):
        shifts.append([int(t) * 2 for t in format(i, '#05b')[2:]])

    d, h, w = mask.shape

    # padding array, [d,h,w]---->[d+2, h+2, w+2]
    mask_ex = np.pad(mask, [[1, 1], [1, 1], [1, 1]], mode="reflect")

    # get volume neighbor index
    neighbor_mask = []
    for shift in shifts:
        i, j, k = shift
        neighbor_mask.append(mask_ex[i:i+d, j:j+h, k:k+w])

    stack_neighbor = np.stack(neighbor_mask, axis=-1)
    edge_mask = np.logical_xor(np.expand_dims(mask, axis=-1), stack_neighbor).any(axis=-1)
    # except background
    edge_mask = np.where(mask > 0, edge_mask, np.zeros_like(edge_mask))

    return edge_mask

def extract_edgemap(mask, blur_sig=1.5):
    """
    :param masks: [d, h, w] 3dim array, binary array, background - zero, foreground - nonezero
    :param blur_sig:
    :return:
    """

    # b1 = gaussian_filter(mask, 1.0)
    # b2 = gaussian_filter(mask, 2.0)
    # edge_mask_blur = np.abs(b1 - b2) > 0.1
    #
    # return edge_mask_blur
    # neighbor index in 3d
    shifts = []
    for i in range(8):
        shifts.append([int(t) * 2 for t in format(i, '#05b')[2:]])

    d, h, w = mask.shape

    # padding array, [d,h,w]---->[d+2, h+2, w+2]
    mask_ex = np.pad(mask, [[1, 1], [1, 1], [1, 1]], mode="reflect")

    # get volume neighbor index
    neighbor_mask = []
    for shift in shifts:
        i, j, k = shift
        neighbor_mask.append(mask_ex[i:i+d, j:j+h, k:k+w])

    edge_mask = np.zeros_like(mask)

    # compare different(edge) value in center with neighbor 8 index value
    for sm in neighbor_mask:
        edge_mask = np.where(np.logical_xor(mask, sm), np.ones_like(mask), edge_mask)
    edge_mask = np.where(mask>0, edge_mask, np.zeros_like(edge_mask))
    edge_mask_blur = gaussian_filter(edge_mask.astype(np.float), blur_sig)

    upper = edge_mask_blur[edge_mask_blur>.5]
    if upper.size > 0:
        clip_thres = upper.mean() + upper.std()
        edge_mask_blur = np.clip(edge_mask_blur / clip_thres, 0, 1)
    # vtk_utils.visulaize(edge_mask_blur*255, 125)
    return edge_mask_blur

@timefn2
def resampling(volume, target_shape, method="linear", keep_ratio=True, return_box=False):
    if keep_ratio:
        d, h, w = volume.shape
        max_length = np.max(volume.shape)
        rescale = np.max(target_shape) / max_length
        keep_ratio_shape = np.round(np.array(volume.shape) * rescale).astype(np.int)
        # max_length = np.max(volume.shape)

        rsz_volume = get_interpolation(volume, keep_ratio_shape, method=method)
        remainder = np.asarray(target_shape) - keep_ratio_shape
        left_pad = remainder//2
        right_pad = remainder - left_pad
        z1, y1, x1 = left_pad
        z2, y2, x2 = right_pad
        img = np.pad(rsz_volume, [[z1, z2], [y1, y2], [x1, x2]], mode='constant')
        outs = img
        if return_box:

            p2 = np.array(rsz_volume.shape)-1
            p1 = np.zeros_like(p2)
            shift = np.concatenate([left_pad, left_pad], axis=0)
            bbox = np.concatenate([p1, p2]) + shift
            outs = [img, bbox]

        return outs
    else:
        return get_interpolation(volume, target_shape, method=method)

def crop_and_resize(volume, bounds, shape, method="linear"):
    d, h, w = volume.shape
    mz = np.linspace(0, d - 1, d)
    my = np.linspace(0, h - 1, h)
    mx = np.linspace(0, w - 1, w)

    source_interpolation = RegularGridInterpolator((mz, my, mx), volume, method=method, bounds_error=False,
                                                   fill_value=0)

    ctr = (bounds[3:] + bounds[:3]) / 2
    max_length = np.max(bounds[3:] - bounds[:3])
    p1 = ctr - max_length/2
    p2 = ctr + max_length/2
    z1, y1, x1 = p1
    z2, y2, x2 = p2

    z = np.linspace(z1, z2, shape[0])
    y = np.linspace(y1, y2, shape[1])
    x = np.linspace(x1, x2, shape[2])

    crop_cube = np.concatenate([p1, p2])
    crop_icube = np.round(crop_cube).astype(np.int)

    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')

    coords = np.stack([Z, Y, X], axis=-1)
    shape = coords.shape[:3]
    # interp_func = self.get_interpolation_func(item_num)
    coords_reshpae = coords.reshape([-1, 3])
    array_ravel = source_interpolation(coords_reshpae)
    return array_ravel.reshape(shape), crop_icube


def get_interpolation(volume, shape, method="nearest"):
    """
    :param volume: [D, H, W]
    :param shape: output shape 3 tuple(out_d, out_h, out_w), integer
    :param method: sicpy interpolation method 'nearest' or 'linear' in general
    :return:
    """
    d, h, w = volume.shape
    mz = np.linspace(0, d - 1, d)
    my = np.linspace(0, h - 1, h)
    mx = np.linspace(0, w - 1, w)

    source_interpolation = RegularGridInterpolator((mz, my, mx), volume, method=method, bounds_error=False, fill_value=0)


    # boxes = utils.denorm_boxes(boxes, func.values.shape)
    # z1, y1, x1, z2, y2, x2 = [k[0] for k in np.split(boxes, 6, axis=0)]
    z = np.linspace(0, d-1, shape[0])
    y = np.linspace(0, h-1, shape[1])
    x = np.linspace(0, w-1, shape[2])
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')

    coords = np.stack([Z, Y, X], axis=-1)
    shape = coords.shape[:3]
    # interp_func = self.get_interpolation_func(item_num)
    coords_reshpae = coords.reshape([-1, 3])
    array_ravel = source_interpolation(coords_reshpae)
    return array_ravel.reshape(shape)


def get_interpolation_from_source(source_volume, sample_shape, bbox, method="nearest"):
    """
    :param source_volume:
    :param shape:
    :param bbox:  6 array z1, y1, x1, z2, y2, x2
    :param method:
    :return:
    """
    d, h, w = source_volume.shape
    mz = np.linspace(0, d - 1, d)
    my = np.linspace(0, h - 1, h)
    mx = np.linspace(0, w - 1, w)

    source_interpolation = RegularGridInterpolator((mz, my, mx), source_volume, method=method, fill_value=0, bounds_error=False)

    # interpolation sample pose
    sd, sh, sw = sample_shape
    z1, y1, x1, z2, y2, x2 = bbox
    z = np.linspace(z1, z2, sd)
    y = np.linspace(y1, y2, sh)
    x = np.linspace(x1, x2, sw)

    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')

    coords = np.stack([Z, Y, X], axis=-1)
    shape = coords.shape[:3]
    # interp_func = self.get_interpolation_func(item_num)
    coords_reshpae = coords.reshape([-1, 3])
    array_ravel = source_interpolation(coords_reshpae)
    return array_ravel.reshape(shape)







def norm_boxes(boxes, shape):
    """Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [N, (z1, y1, x1, z2, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (y1, x1, y2, x2)] in normalized coordinates
    """
    box_ndim = boxes.ndim
    if box_ndim == 1:
        boxes = np.expand_dims(boxes, axis=0)
    d, h, w = shape
    scale = np.array([d - 1, h - 1, w - 1, d - 1,  h - 1, w - 1])
    shift = np.array([0, 0, 0, 1, 1, 1])
    res = np.divide((boxes - shift), scale).astype(np.float32)
    if box_ndim == 1:
        res = np.squeeze(res)
    return res


def norm_boxes_bounds(boxes, shape):
    scale = np.asarray(shape) - 1
    scale = np.pad(scale, [0, 3], mode='wrap')
    # d, h, w = shape
    # scale = np.array([d - 1, h - 1, w - 1, d - 1,  h - 1, w - 1])
    # shift = np.array([0, 0, 0, 1, 1, 1])inner_crop
    return np.divide(boxes, scale).astype(np.float32)


def denorm_boxes_bounds(boxes, shape):
    scale = np.asarray(shape)-1
    scale = np.pad(scale, [0, 3], mode='wrap')
    return np.multiply(boxes, scale)

def denorm_boxes(boxes, shape, return_integer=True):
    """Converts boxes from normalized coordinates to pixel coordinates.
    opencv 와 같이 bounding box 정보가 width크기를 포함하는 경우에 대한 boxes정보를 denoralize
    (widthxheight) 이미지의 경우 opencv는 bounding box 정보를 (0, 0, width, height)로(rect)
    이미지 index 범위로 보면 0~width-1, 0~height-1 이기 때문에,
    normalize하기 위해서 shift값을 처리해주고, denoramlize의 경우 다시 보상해준다.

    https://docs.opencv.org/3.4/d2/d44/classcv_1_1Rect__.html

    reF) denorm_boxes_bounds 는 이와 반대로 x1~x2 , x2가 -1 or +1 보상된 값이라고 가정하고 처리해는 함수

    boxes: [N, (z1, y1, x1, z2, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (z1, y1, x1, z2, y2, x2)] in pixel coordinates
    """
    box_ndim = boxes.ndim
    if box_ndim == 1:
        boxes = np.expand_dims(boxes, axis=0)
    d, h, w = shape
    scale = np.array([d - 1, h - 1, w - 1, d - 1, h - 1, w - 1])
    shift = np.array([0, 0, 0, 1, 1, 1])
    res = np.multiply(boxes, scale) + shift
    if return_integer:
        res = np.around(res).astype(np.int32)
    if box_ndim == 1:
        res = np.squeeze(res)
    return res

def apply_trasnform_np(pts, transform):
    ndim = pts.shape[-1]
    # if pts.ndim == 2:

    #     return
    # elif pts.ndim == 1:
    return np.dot(pts, transform[:ndim, :ndim].T) + transform[:ndim, ndim]


def _apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask > 0,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image

def gray2color(img):
    """
    :param img: 2dim array
    :return:
    """
    if img.ndim == 2:
        return np.repeat(np.expand_dims(img, axis=-1), 3, 2)
    elif img.ndim == 2 and img.shape[2] == 1:
        return np.repeat(img, 3, 2)
    else:
        raise ValueError

def display_masking(source_volume, volume_mask, alpha=.5, threshold=0):
    for img, mask in zip(np.squeeze(source_volume), np.squeeze(volume_mask)):
        drawing = (gray2color(img)*255).astype(np.uint8)
        _apply_mask(drawing, mask > threshold, (0, 1, 0), alpha)
        plt.imshow(drawing)
        plt.pause(0.08)





def get_rainbow_color_table(number):
    rainbow_table = [
        [255, 0, 0],
        [255, 125, 0],
        [255, 255, 0],
        [125, 255, 0],
        [0, 255, 0],
        [0, 255, 125],
        [0, 255, 255],
        [0, 0, 255],
        [0, 5, 70],
        [100, 0, 255]
    ]

    length = len(rainbow_table)
    np_rainbow_table = np.array(rainbow_table)
    if number < np_rainbow_table.shape[0]:
        return np_rainbow_table / 255.
    landmark_color_table = []
    for i in range(number):
        value = (i / (number - 1) * (length - 1))
        upper = np.ceil(value)
        lower = np.floor(value)
        x = abs(value - upper)
        index = int(lower)
        # print(i, value, x, 1-x , index)

        if index == length - 1:
            color = np_rainbow_table[index]
        else:

            if x < 1e-10:
                x = 1

            color = x * np_rainbow_table[index] + (1 - x) * np_rainbow_table[index + 1]

        landmark_color_table.append(tuple(color / 255.0))
    return landmark_color_table


def compare_image(vol_image, mask_image, thres=.5, pause_sec=0.01, stride=1,
                  concat_original=False, full_region=True, image_save=False, save_path='temp', show=True,
                  transpose=False,
                  create_sub_dir=True,
                  in_rainbow_size=-1
                  ):
    """
    :param vol_image: [D,H,W] 0~255
    :param mask_image:
    :return:
    """


    if show:
        fig = plt.figure()
    if full_region:
        start, end = 0, vol_image.shape[0]
    else:
        zyx = np.argwhere(mask_image > 0)

        if zyx.size == 0:
            import logging
            logging.getLogger().error('emtpy foreground')
            return
        start, end = zyx[:, 0].min(), zyx[:, 0].max()

    in_rainbow_mask = in_rainbow_size > 0
    if in_rainbow_mask:
        color_table = get_rainbow_color_table(mask_image.max())
        color_mapping = np.concatenate([np.zeros([1, 3], dtype=np.uint8), (255 * np.asarray(color_table)).astype(np.uint8)])

        # color_table =

    vol_image = np.clip(vol_image, 0, 255)
    if image_save:
        if create_sub_dir:
            save_path = os.path.join(save_path, time.strftime("%Y%m%d%H%M%S"))
        os.makedirs(save_path, exist_ok=True)

    for i, (src, ma) in enumerate(zip(vol_image, mask_image)):
        if i % stride != 0:
            continue
        if start <= i < end:
            pass
        else:
            continue

        if transpose:
            src = src.T
            ma = ma.T

        drawing = np.repeat(np.expand_dims(src, axis=-1), 3, 2).astype(np.uint8)
        drawing_src = drawing.copy()
        if in_rainbow_mask:
            mask = color_mapping[ma]
            utils_numpy.apply_blending_mask(drawing, mask)
        else:
            utils_numpy.apply_mask(drawing, ma, (0, 1, 0), thres)
        if concat_original:
            bound_w = int(drawing.shape[1] * 0.2)
            boundary = np.full([drawing.shape[0], bound_w, drawing.shape[2]], 255, dtype=drawing.dtype)
            drawing = np.concatenate([drawing_src, boundary, drawing], axis=1)

        if show:
            plt.cla()
            plt.imshow(drawing)
            plt.pause(pause_sec)

        if image_save:
            plt.imsave(os.path.join(save_path, '{:03d}.png'.format(i)), drawing)

    plt.close('all')




def show_2mask_image(vol_image, mask_image, thres=.5, pause_sec=0.01, stride=1,
                  concat_original=False, full_region=True, image_save=False, save_path='temp', show=True,
                  transpose=False,
                  create_sub_dir=True,
                  in_rainbow_size=-1):
    """
    :param vol_image: [D,H,W] 0~255
    :param mask_image:
    :return:
    """

    if show:
        fig = plt.figure()
    if full_region:
        start, end = 0, vol_image.shape[0]
    else:
        zyx = np.argwhere(mask_image > 0)
        start, end = zyx[:, 0].min(), zyx[:, 0].max()

    if np.issubdtype(mask_image.dtype, np.integer):
        mask_image = mask_image.astype(np.int32)

    in_rainbow_mask = in_rainbow_size > 0
    if in_rainbow_mask:
        color_table = get_rainbow_color_table(mask_image.max())
        color_mapping = np.concatenate([np.zeros([1, 3], dtype=np.uint8), (255 * np.asarray(color_table)).astype(np.uint8)])

        # color_table =

    vol_image = np.clip(vol_image, 0, 255)
    if image_save:
        if create_sub_dir:
            save_path = os.path.join(save_path, time.strftime("%Y%m%d%H%M%S"))
        os.makedirs(save_path, exist_ok=True)

    for i, (src, ma) in enumerate(zip(vol_image, mask_image)):
        if i % stride != 0:
            continue
        if start <= i < end:
            pass
        else:
            continue

        if transpose:
            src = src.T
            ma = ma.T

        drawing = np.repeat(np.expand_dims(src, axis=-1), 3, 2).astype(np.uint8)
        drawing_src = drawing.copy()
        if in_rainbow_mask:
            mask = color_mapping[ma]
            utils_numpy.apply_blending_mask(drawing, mask)
        # else:
        #     utils_numpy.apply_mask(drawing, ma, (0, 1, 0), thres)
        # if concat_original:
        mask = color_mapping[ma]
        bound_w = int(drawing.shape[1] * 0.2)
        boundary = np.full([drawing.shape[0], bound_w, drawing.shape[2]], 255, dtype=drawing.dtype)
        drawing = np.concatenate([drawing_src, boundary, drawing, boundary, mask], axis=1)

        if show:
            plt.cla()
            plt.imshow(drawing)
            plt.pause(pause_sec)

        if image_save:
            plt.imsave(os.path.join(save_path, '{:03d}.png'.format(i)), drawing)

    plt.close('all')



def show_2d_image(vol_image, pause_sec=0.01, stride=1, image_save=False, save_path='temp', show=True, dtype=np.uint8):
    """

    Parameters
    ----------
    vol_image : (d,h,w) volume
    pause_sec :
    stride :
    image_save : bool
        option image saving
    save_path : str
        the pathname to save the image
    show : bool
        option show matplotlib

    Returns
    -------

    """
    if show:
        fig = plt.figure()
    for i, src in enumerate(vol_image):
        if i % stride != 0:
            continue
        if dtype == np.uint8:
            drawing = np.repeat(np.expand_dims(src, axis=-1), 3, 2).astype(np.uint8)
        else:
            drawing = src


        # utils_numpy.apply_mask(drawing, ma, (0, 1, 0), thres)
        if show:
            plt.cla()
            plt.imshow(drawing)
            plt.pause(pause_sec)

        if image_save:
            os.makedirs(save_path, exist_ok=True)
            plt.imsave(os.path.join(save_path, '{:03d}.png'.format(i)), drawing)
    plt.close('all')

def voi_align_from_aabb(volume, aabb, meta_shape, pool_shape, method='linear',  interpolate_func:InterpolateWrapper=None,
                        return_affine_transform=False, return_warp_points=False):
    """
    voi_align_from_obb(...) 이용해서 resampling
    converting form aabb to obb. then apply function 'voi_align_from_obb'

    Parameters
    ----------
    volume : np.ndarray [D,H,W]
    aabb : np.ndarray [num_rois, 6] or (6,) format (z1,y1,x1,z2,y2,x2)
    meta_shape : np.ndarray input shape of volume (= volume.shape)
    pool_shape : np.ndarray target shape (3,) array
    method : str 'linear' or 'nearest'
    interpolate_func :
    return_affine_transform : np.ndaaray optional (4,4) transform matrix.
    return_warp_points : bool

    Returns
    -------

    """
    obb = aabb2obb(aabb)
    return voi_align_from_obb(volume, obb, meta_shape, pool_shape, method, interpolate_func, return_affine_transform=return_affine_transform, return_warp_points=return_warp_points)


def voi_align_from_obb(volume, normalized_obb, meta_shape, pool_shape, method="linear",
                       interpolate_func:RegularGridInterpolator=None,
                       return_affine_transform=False,
                       return_warp_points=False,
                       obb_normalized=True):
    """
    :param volume: [D, H, W]
    :param obb: [num_rois, 9] or (9,) rois
    :param meta_shape: used to denomalize obb-meta
    :param pool_shape: mask shape interger [d, h, w]
    :return: [num_rois, d, h, w] or [d, h, w] num_rois is dependent on obb's shape
    """
    input_roi_ndim = normalized_obb.ndim
    assert input_roi_ndim <= 2, '1dim or 2dim for obb'

    obb = np.expand_dims(normalized_obb, axis=0) if input_roi_ndim == 1 else normalized_obb

    obb = denorm_obb(obb, meta_shape) if obb_normalized else obb
    shape = np.asarray(pool_shape)
    center, fsize, theta = decompose_general_boxes(obb)
    # matrix = create_mat4x4_from_euler_vtk(theta)
    matrix = create_matrix_from_euler(theta)

    # orthonormal basis : row-vector ----> colume-vector
    basis = np.transpose(matrix[:, :3, :3], [0, 2, 1])

    mz, my, mx = shape[0], shape[1], shape[2]
    # [N,1]
    z = np.linspace(-1., 1., mz)
    y = np.linspace(-1., 1., my)
    x = np.linspace(-1., 1., mx)
    # y = np.linspace(0, dh, 2)
    # x = np.linspace(0, dw, 2)

    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
    # [d, h, w, 3]
    zyx_grid = np.stack([Z, Y, X], axis=-1)

    # [N, d*h*w, 3]
    # scalding size
    size_scale = (fsize)/ 2.
    zyx_pose = np.reshape(zyx_grid, [1, -1, 3]) * np.expand_dims(size_scale, axis=1)

    basisT = np.transpose(basis,  [0, 2, 1])
    # shift from center to origin1
    # [N, d*h*w, 3]
    warp_points = np.expand_dims(center, axis=1) + np.matmul(zyx_pose, basisT)
    # [N*d*h*w, 3]
    reshape_warp_points = np.reshape(warp_points, [-1, 3])

    # max_ix = np.array(volume.shape[:3]) - 1.

    # scaling = (fsize - 1.) / (np.array(pool_shape) - 1)
    # 좌표기준으로 scaling값을 계산하는 것이므로 shape에 1을 빼줘야한다.
    scaling = (fsize) / (np.array(pool_shape)-1)

    affine_transform = []
    for i, (b, s, ctr) in enumerate(zip(basis, scaling, center)):
        Ro = rotation_mat(b)
        t1 = ctr - vtk_utils.apply_trasnform_np(fsize[i], Ro) / 2.
        Sc = scaling_mat(s)
        Tr = translation_mat(t1)
        mat = np.matmul(Tr, np.dot(Ro, Sc))
        affine_transform.append(mat)
    affine_transform = np.array(affine_transform)

    DEBUG = False
    if DEBUG:
        # compare warping coords and its grid coordinate of pools-shape.
        # coords from pooling volume is transformed
        az = np.linspace(0., mz-1., mz)
        ay = np.linspace(0., my-1., my)
        ax = np.linspace(0., mx-1., mx)
        ZYX = np.stack(np.meshgrid(az, ay, ax, indexing='ij'), axis=-1).reshape([-1, 3])
        np.diff(reshape_warp_points.max(axis=0) - reshape_warp_points.min(axis=0), 0.).all()
        mat = affine_transform[0]
        tzyx = vtk_utils.apply_trasnform_np(ZYX, mat)
        assert obb.shape[0] == 1
        print(reshape_warp_points.min(axis=0), reshape_warp_points.max(axis=0))
        print(tzyx.min(axis=0), tzyx.max(axis=0))
        # vtk_utils.show_plots(tzyx[::100])
        inv_mat = np.linalg.inv(mat)
        print("check points", np.isclose(reshape_warp_points - tzyx, 0.).all())
        print("transform restortation check ", np.isclose(np.dot(mat, inv_mat) - np.eye(4), 0.).all())
        print("transform & inverse check",
              np.isclose(vtk_utils.apply_trasnform_np(vtk_utils.apply_trasnform_np(ZYX, mat), inv_mat) - ZYX, 0.).all())


    num_rois = np.shape(obb)[0]

    if interpolate_func is None:
        interpolate_func = InterpolateWrapper(volume, method=method)

    resample_array = interpolate_func(reshape_warp_points)
    resample_array = resample_array.reshape([num_rois, *zyx_grid.shape[:3]])

    if input_roi_ndim == 1:
        resample_array = np.squeeze(resample_array)
        affine_transform = np.squeeze(affine_transform)
        warp_points = np.squeeze(warp_points)

    if return_affine_transform or return_warp_points:
        outs = [resample_array]
        if return_affine_transform:
            outs.append(affine_transform)
        if return_warp_points:
            outs.append(warp_points)
        return tuple(outs)
        # out = [resample_array, affine_transform]
    else:
        out = resample_array

    return out


def volume_sampling_from_coords(
        volumes:List[np.ndarray],
        coords:np.ndarray,
        pool_seed:np.ndarray,
        scale:Union[float, np.ndarray],
        augment_param:dict,
        extend_size:float=1.05,
        return_warp_points=False,
        return_transform=False,
        fixed_shape=None,
        max_size=130**3
):
    """
    대상 좌표값을 주성분분석(pca)를 통해 주축을 계산
    대상을 포함하는 obb(oriented bounding box) 를 계산(likewise convexhull)
    obb를 볼륨으로 샘플링한다.

    샘플링 볼륨 사이즈(obb)는 다음과 같이 계산
    1. 샘플링 볼륨 사이즈는 pool_seed의 정수 배수값이 되도록 설정
    2.
    pool_shape = ceil( ( scale * obb_size * extend_size) / pool_seed) * pool_seed
    그리고 최종 obb_size는
    obb_size = pool_shape / scale 로 재계산된다.(scaling xyz 동일한 값으로 처리하기 위해(keep-ratio))
    sampling method는 linear, nearest 를 자동으로 설정한다.
    volume data-type이 floating type 일 경우 linear로
    volume data-type이 interge type 일 경우 nearest로 처리한다.
    See Also
    ---------
    restored_obb_segment_to_original_grid : from obb-volume to original-volume

    Parameters
    ----------
    volumes : list[np.ndarray]
        volume element ->  (D, H, W)
    coords : np.ndarray
        (N, 3) zyx coordinates - 볼륨 공간사엥 샘플링할 좌표
    pool_seed : np.ndarray[int]
        특점 배수의 볼륨 shape이 되도록 설정
    scale : float
        이미지 해상도. scale값이 클 수록 정밀도가 높다. 결과적으로 볼륨크기가 증가
    augment_param : dict
        theta : euler theta, default 0
        sclae : scale . default 1.0
        translate: translate default 0.0
    extend_size : float
        bounding box 경계 부분 여유를 주기 위해, bounding-box를 크기를 조절한다. defautl 1.05
    return_warp_points : bool
        default false
    return_transform : bool
        from obb-coordinate to source coordinate
    fixed_shape: unioni[tuple[foat], np.ndarray]
    max_size : int
        memory 제한 때문에 입력 사이즈 제한
    Returns
    -------

    """

    align_t = align_transform(coords)
    # print('coords:', left_coords.shape)
    align_pose = apply_trasnform_np(coords, align_t)

    obb_meta = geometry_numpy.obbmeta_from_pose(align_pose, align_t)

    aug_theta = augment_param.get('theta', np.zeros([3]))
    aug_scale = augment_param.get('scale', np.ones([3]))
    aug_trans = augment_param.get('translate', np.zeros([3]))

    # 헬이다 헬....xyz ...
    theta = obb_meta[6:9]
    # size off
    obb_meta[3:6] = obb_meta[3:6] * extend_size
    obb_meta[6:9] = theta[::-1] + aug_theta
    obb_meta[0:3] = obb_meta[0:3] + aug_trans
    obb_meta[3:6] = obb_meta[3:6] * aug_scale

    # if fixed_shape is not None:
    #     assert len(fixed_shape) == 3
    #     obb_meta[3:6] = np.asarray(fixed_shape)

    box_size = obb_meta[3:6]
    total_size = np.prod(box_size)

    obb_meta[3:6] = box_size if total_size < max_size else box_size * (max_size / total_size)

    sample_volumes = []

    return_obb = obb_meta
    for vol in volumes:
        obb = obb_meta.copy()
        if fixed_shape is None:
            obb_size = obb_meta[3:6]
            pool_shape = np.ceil((obb_size * scale) / pool_seed) * pool_seed
            pool_obb_size = pool_shape / scale
            pool_shape = pool_shape.astype(np.int64)
            obb[3:6] = pool_obb_size
        else:
            temp_pool_size = np.asarray(fixed_shape)
            obb[3:6] = np.max(obb[3:6] / (temp_pool_size - 1) ) * temp_pool_size
            pool_shape = fixed_shape

        # replica new oobb size

        normed_obb = norm_obb(obb, vol.shape)
        return_obb = obb

        method = 'linear' if np.issubdtype(vol.dtype, np.floating) else 'nearest'
        sample_vol, transform, warp_points = voi_align_from_obb(
            vol,
            normed_obb,
            vol.shape,
            pool_shape,
            method=method,
            return_warp_points=True,
            return_affine_transform=True
        )
        sample_volumes.append(sample_vol)

    items = [sample_volumes, return_obb]
    if return_warp_points:
        items.append(warp_points)
    if return_transform:
        items.append(transform)

    return items


@timefn
def restored_obb_segment_to_original_grid(pred_obb_volume, obb_warp_points_in_src, t_src2obb, origin_volume_shape,
                                          method='nearest'):
    """
    obb-volume(pred_obb_volume) 과 obb-volume을 구성하는 coordinates(obb_warp_points_in_src) 및 변환행렬값을 이용해서 원본 볼륨 grid로 segmnetation 복원
    1. 유효 영역은 obb를 포함하면서 원본 볼륨의 유효한 aabb를 산출하고 mesh-grid 를 생성
    2. mesh-grid 를 다시 obb좌표로 벼환하여 유효한 grid인 mask 정보를 산출
    3. 유효한 mask에 대해서만 interpolation 적용
    4. 유효한 mask의 원본 grid 정보 산출

    See Also :
    ----------
    volume_sampling_from_coords : from original-volume to obb-volume

    Parameters
    ----------
    pred_obb_volume : np.ndarray
        (d, h, w) obb segmentation volume
    obb_warp_points_in_src : np.ndarray
        (N, 3), warping points in source-volume, N = d * h * w. pred_obb_volume을 구성하는 좌표값
    t_src2obb : np.ndarray
        (4, 4) transform matrix from source coordinate to obb-volume coordinate
    origin_volume_shape : np.ndarray
        (3,) source volume shape

    method :

    Returns
    -------
        np.ndarray :
            sampling volume value. (M,)  M 값은 original volume & obb grid 유효한 영역에 대한 grid수
        np.ndarray :
            sampling volume grid coordinates (M, 3) as integer. M 값은 original volume & obb grid 유효한 영역에 대한 grid수

    """
    obb_vol_shape = np.asarray(pred_obb_volume.shape)
    wmin, wmax = obb_warp_points_in_src.min(axis=0), obb_warp_points_in_src.max(axis=0)
    wmin, wmax = np.floor(wmin).astype(np.int32), np.floor(wmax).astype(np.int32)
    # original volume-shape 유효한 영역기준으로 mesh-grid 생성
    wmin = np.clip(wmin, np.zeros_like(wmin), origin_volume_shape - 1)
    wmax = np.clip(wmax, np.zeros_like(wmin), origin_volume_shape - 1)
    # generate original grid from obb info
    grids = [np.linspace(wmin[k], wmax[k], wmax[k] - wmin[k] + 1, dtype=np.int32) for k in range(wmin.size)]
    mesh_grids = np.stack(np.meshgrid(*grids, indexing='ij'), axis=-1)

    # t_src2obb = np.linalg.inv(t_obb2src)
    mesh_grids_in_obb = apply_trasnform_np(mesh_grids.reshape([-1, 3]), t_src2obb)

    # 복셀에 유효한 영역만 mask 추출
    valid_mask = np.logical_and(mesh_grids_in_obb >= 0, mesh_grids_in_obb <= (np.asarray(obb_vol_shape) - 1))
    valid_mask_all = valid_mask.all(axis=-1)

    valid_mesh_grids_in_obb = mesh_grids_in_obb[valid_mask_all]

    restored = InterpolateWrapper(pred_obb_volume, method=method)(valid_mesh_grids_in_obb)

    valid_mesh_grid = mesh_grids.reshape([-1, 3])[valid_mask_all]
    return restored, valid_mesh_grid


        # vtk_utils.show_actors([
        #     sample_vol
        # ])

def voxel_edge_detection(volume, threshold=0.7):
    from scipy.ndimage import convolve

    # Apply Sobel operator along each exis
    sobel_x = convolve(volume, np.array([[[1,0,-1],[2,0,-2],[1,0,-1]]]))
    sobel_y = convolve(volume, np.array([[[1], [2], [1]], [[0], [0], [0]], [[-1], [-2], [-1]]]))
    sobel_z = convolve(volume, np.array([[[1, 2, 1]], [[0, 0, 0]], [[-1, -2, -1]]]))

    # Conpute the magnitude of the gradient
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2 + sobel_z**2)

    # Apply a threshold to obtain binary edges
    edge_mask = gradient_magnitude > threshold

    return edge_mask

def voxel_edge_detection_torch(volume, threshold=0.7):
    import torch
    import torch.nn.functional as F

    # Apply 3D Sobel operator for edge detection along each dimension
    sobel_x = F.conv3d(volume, weight=torch.Tensor([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]]).unsqueeze(0).unsqueeze(0).float(), padding=1)
    sobel_y = F.conv3d(volume, weight=torch.Tensor([[[[1], [2], [1]]], [[[0], [0], [0]]], [[[-1], [-2], [-1]]]]).unsqueeze(0).unsqueeze(0).float(), padding=1)
    sobel_z = F.conv3d(volume, weight=torch.Tensor([[[[1, 2, 1]]], [[[0, 0, 0]]], [[[-1, -2, -1]]]]).unsqueeze(0).unsqueeze(0).float(), padding=1)

    # Compute the magnitude of the gradient
    gradient_magnitude = torch.sqrt(sobel_x**2 + sobel_y**2 + sobel_z**2)

    # Apply a threshold to obtain binary edges
    edge_mask = (gradient_magnitude > threshold).float()

    return edge_mask

def fit_points_3d(x,y,z):
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.optimize import curve_fit

    def quadratic_model(x, a,b,c):
        return a * x**2 + b*x + c

    # Initial guess for the parameters
    initial_guess = [1, 1, 1]

    # Fit the model to the data
    params, covariance = curve_fit(quadratic_model, x, z, p0=initial_guess)

    # Print the fitted parameters
    print("Fitted Parameters:", params)

    # Generate points for the fitted curve
    x_fit = np.linspace(min(x), max(x), 300)
    z_fit = quadratic_model(x_fit, *params)

    # Return the fitted 3D points array

    # Create an array with shape (n, 3) representing the fitted 3D points
    fitted_points = np.column_stack((x_fit, np.ones_like(x_fit) * y.mean(), z_fit))

    # Return the array of fitted 3D points
    return fitted_points

def find_nearest_neighbors(source_points, target_points):
    from scipy.spatial import cKDTree

    tree = cKDTree(target_points)

    _, indices = tree.query(source_points)

    neareast = target_points[indices]

    return neareast


def peak_density_index(points, counts_thresh) -> List[np.ndarray]:
    """

    Args:
        points ():
        counts_thresh ():

    Returns:

    """
    if not np.issubdtype(points.dtype, np.integer):
        points = points.asytpe(np.int32)
    # counts_thresh = 3 * (quantized ** 3)

    coord, index, inverse, counts = np.unique(points, return_index=True, return_inverse=True, return_counts=True,
                                              axis=0)
    sort_inverse = np.sort(inverse)
    index_inverse = index[sort_inverse]
    # split_index = np.split(sort_inverse, counts)

    cumsum_counts = np.cumsum(counts)
    split_groups_index = np.split(index_inverse, cumsum_counts[:-1])

    count_index = np.where(counts > counts_thresh)[0]
    split_groups_index_peak_density = [split_groups_index[i] for i in count_index]
    return split_groups_index_peak_density


# vtk_utils.show()
def fast_cluster(seg, off, voting_thresh=3, weight_dis_thresh=5, weight_score_thresh=10, seg_thresh=.5):
    """
    https://github.com/ErdanC/Tooth-and-alveolar-bone-segmentation-from-CBCT
    implementation of the paper 'Clustering by fast search and find of density peaks'
    Args:
    bn_seg: predicted binary segmentation results -> (batch_size, 2, 120, 120, 120)
    off: predicted offset of x. y, z -> (batch_size, 3, 120, 120, 120)
    Returns:
    The centroids obtained from the cluster algorithm
    """
    seg_mask = seg.copy()
    posit = seg > seg_thresh
    negat = np.logical_not(posit)
    seg_mask[posit] = 1
    seg_mask[negat] = 0

    grid = np.array(np.nonzero(posit))

    off_posit = off[:, posit]
    init_coord = grid + off_posit
    coord = np.round(init_coord).astype(np.int32)

    peak_dense_indices = peak_density_index(coord.T, voting_thresh)

    num_pts = len(peak_dense_indices)
    if num_pts < 2 or num_pts > 5000:
        centroids = grid.T.mean(axis=0) if grid.size > 0 else np.zeros([3])
        # if num_pts < 2:
        coords_per_label = {
            1: grid.T
        }
        return centroids, seg, coords_per_label

    weight_scores = np.array([i.size for i in peak_dense_indices])
    samples = np.stack([init_coord.T[i].mean(axis=0) for i in peak_dense_indices])

    samples_dis = np.linalg.norm(samples[:, None] - samples[None], axis=-1)
    samples_score = weight_scores[:, None] - weight_scores[None]

    #  diagonal &
    samples_dis[samples_score > -0.5] = 1e10
    weight_dis = np.amin(samples_dis, axis=1)

    mask = (weight_dis > weight_dis_thresh) * (weight_scores > weight_score_thresh)
    if mask.sum() < 2:
        centroids = grid.T.mean(axis=0) if grid.size > 0 else np.zeros([3])
        # if num_pts < 2:
        coords_per_label = {
            1: grid.T
        }

        return centroids, seg, coords_per_label

    centroids = samples[mask]



    neighbor = NearestNeighbors().fit(centroids)
    args = neighbor.kneighbors(init_coord.T, n_neighbors=1, return_distance=False)
    label = np.arange(centroids.shape[0]) + 1
    label_assign = label[args[:, 0]]
    label_mask = np.zeros_like(seg)
    label_mask[posit] = label_assign

    indice_per_label = uni(label_assign)
    coords_per_label = {k: grid.T[v] for k, v in indice_per_label.items()}

    return centroids, label_mask, coords_per_label


# def unique_indices():

# @timefn2
def unique_indices(records_array):
    """
    How to get a list of all indices of repeated elements in a numpy array
    https://stackoverflow.com/questions/30003068/how-to-get-a-list-of-all-indices-of-repeated-elements-in-a-numpy-array
    Args:
        records_array ():

    Returns:

    """
    # np_unique test
    idx_sort = np.argsort(records_array)
    sorted_records_array = records_array[idx_sort]
    vals, idx_start, count = np.unique(sorted_records_array, return_counts=True, return_index=True)
    res = np.split(idx_sort, idx_start[1:])
    return dict(zip(vals, res))

uni = unique_indices