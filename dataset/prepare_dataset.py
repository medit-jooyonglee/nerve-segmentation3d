import os
import numpy as np
import pytest
import vtk_utils
import json

from trainer import diskmanager, get_logger
from trainer import get_logger
from trainer import image_utils
from trainer import utils
from tools import utils_numpy
from tools.simpleitk_utils import read_sitk_volume
from tools import dicom_read_wrapper, vtk_utils

get_runtime_logger = get_logger

LEFT = 1
RIGHT = 2

def read_aihub_json(filename):
    # assert len(json_file) > 0
    with open(filename, 'r', encoding='UTF8') as f:
        data = json.load(f)
    # data['data-group']
    # meta_keys = ['de']
    # meta = data['meta']
    # 'data-group', 'case', 'facility', 'patient', 'study', 'meta', 'annotation'

    right_coords = data['annotation']['tooth']['Right']['coordinate']
    left_coords = data['annotation']['tooth']['Left']['coordinate']
    return np.asarray(left_coords).reshape([-1, 3]), np.asarray(right_coords).reshape([-1, 3]), data


def read_aihub_json_nerve(filename, vtk_order=True):
    """

    Parameters
    ----------
    filename
    vtk_order

    Returns
    -------
    (left, right) coords in xyz order (N, 3) array,
    shape(meta) (3,)
    json data
    """
    # read_aihub_json()
    left, right, data = read_aihub_json(filename)
    left_size = len(left)
    shape_keys = ['depth', 'height', 'width']
    meta = data['meta']
    shape = [meta[k] for k in shape_keys]
    coords = np.concatenate([left, right], axis=0)
    # coords = np.asarray(coords)
    coords_reshape = coords.reshape([-1, 3])
    coords2 = coords_reshape
    def reverse_coords(pose, shape, axes):
        out = pose.copy()
        for k in axes:
            out[:, k] = shape[k] - pose[:, k]

            # out[:, k] = shape - pose[::-1, k]
        return out
    if vtk_order:
        coords3 = reverse_coords(coords2, shape[::-1], [1])
    else:
        coords3 = reverse_coords(coords2, shape[::-1], [2])

    return (coords3[:left_size], coords3[left_size:]), shape, data

def recon_volume(coords, shape, target_shape=None, coords_order='xyz'):
    target_shape = shape if target_shape is None else np.asarray(target_shape)
    volume = np.zeros([*shape], dtype=np.float32)
    for i, coord in enumerate(coords):
        i0, j0, k0 = [np.squeeze(v).astype(np.int32) for v in np.split(coord, 3, axis=-1)]
        if coords_order == 'xyz':
            volume[k0, j0, i0] = i + 1
        else:
            volume[i0, j0, k0] = i + 1
    res = image_utils.auto_cropping_keep_ratio(volume, target_shape=target_shape, method='nearest')
    return res


def recon_volume2(coords, shape, coords_order='xyz', dtype='float32'):
    lefts, rights = coords
    # target_shape = shape if target_shape is None else np.asarray(target_shape)
    volume = np.zeros([*shape], dtype=dtype)
    for i, (coord, label) in enumerate(zip([lefts, rights ], [LEFT, RIGHT])):
        i0, j0, k0 = [np.squeeze(v).astype(np.int32) for v in np.split(coord, 3, axis=-1)]
        if coords_order == 'xyz':
            volume[k0, j0, i0] = label
        else:
            volume[i0, j0, k0] = label

    return volume



def test_read_nerve():
    filename = 'D:/dataset/ai_hub_labels/CTDATA/32/CT3D'
    pass
    assert os.path.exists(filename)
    from tools import dicom_read_wrapper
    import glob
    vtk_order = False
    volume, spacing = dicom_read_wrapper.read_dicom_wrapper(filename, vtkorder=vtk_order)

    print(volume.shape, spacing)
    json_file = glob.glob(os.path.join(filename, '*.json'))

    coords3 = read_aihub_json_nerve(json_file[0])

    vtk_utils.split_show([volume, coords3, vtk_utils.get_axes(100)], [coords3, vtk_utils.get_axes(100)])


def convert_main(souce_path, label_path, save_path, label_ext='.json', vtk_order=False,
                 target_max_size=350):
    # right to left, from posterio to anterior form superior to inferior
    # LAI
    # LPS / vtk, pydicom
    orientation_list = ['lai', 'lps']
    orientation = orientation_list[0] if vtk_order else orientation_list[-1]

    logger = get_logger(__name__)
    save_path_source, save_path_gt = [os.path.join(save_path, sub) for sub in ['source', 'gt']]
    os.makedirs(save_path_source, exist_ok=True)
    os.makedirs(save_path_gt, exist_ok=True)
    sample_files = diskmanager.deep_serach_files(label_path, [label_ext])

    logger.info(f'founded {len(sample_files)=} samples')
    # librarymanager.init_library_manager('teeth_template.pkl')
    # vtk_order = True

    for i, sample_path in enumerate(sample_files):
        sample_dirname = os.path.dirname(sample_path)
        sample_name = '_'.join(diskmanager.split_dir_paths(sample_dirname)[-3:])
        # logger.info(f'[{i}] converting....{sample_path}....\n{i/len(sample_files)}..........')
        #     continue
        os.path.dirname(sample_path)
        label_rel_path = os.path.relpath(os.path.dirname(sample_path), label_path)
        founded_ct_paths = diskmanager.deep_search_directory(os.path.join(souce_path,
                                                                          label_rel_path), ['.dcm', '.DCM'],
                                                             filter_func=lambda x: len(x) > 100)
        if len(founded_ct_paths) != 1:
            logger.error(f'zero founded or 2 more ct founded #{len(founded_ct_paths)}')
            continue

        dicom_path = founded_ct_paths.pop()


        # print(npy_gt_name, npy_src_name)

        try:
            # left, right = read_aihub_json_nerve(sample_path, [500, 500, 500])
            coords, source_shape, json_data = read_aihub_json_nerve(sample_path, vtk_order)
        except Exception as e:
            logger.error(f'load filaed  {sample_path=}')
            logger.error(e.args)
            continue

        # target_shape = np.asarray(target_shape)
        max_i = np.argmax(source_shape)
        scale = target_max_size / source_shape[max_i]
        target_shape = (np.asarray(source_shape) * scale).astype(np.int32)
        # scale =

        # src_label_vol = recon_volume2(coords, source_shape)
        label_src_volume = recon_volume2(coords, source_shape)
        label_resize_volume = image_utils.auto_cropping_keep_ratio(label_src_volume, target_shape=target_shape, method='nearest')

        norm_src_volume, spacing, _ = read_sitk_volume(dicom_path)
        # norm_src_volume, spacing = dicom_read_wrapper.read_dicom_wrapper(dicom_path, vtkorder=vtk_order, normalize=True)
        resize_spacing = np.asarray(spacing) / scale
        rsz_volume = image_utils.auto_cropping_keep_ratio(norm_src_volume, target_shape=target_shape)

        if norm_src_volume.shape != tuple(source_shape):
            logger.error('ct volume and json data meta info are different')
            continue
        # assert norm_src_volume.shape == tuple(source_shape), 'diffeent size'

        npy_src_name = os.path.join(save_path_source, sample_name + '.npy')
        npy_gt_name = os.path.join(save_path_gt, sample_name + '.npy')

        meta_data = {
            'source_shape': np.asarray(norm_src_volume.shape),
            'source_spacing': resize_spacing,
            'spacing': np.asarray(spacing),
            'shape': target_shape,
            'vtkorder': vtk_order,
        }

        dtype = np.uint16
        dtypeinfo = np.iinfo(dtype)
        # 데이터 용량을 위해 변환.
        rsz_volume_dtype = (rsz_volume * dtypeinfo.max).astype(dtype)
        np.save(npy_src_name, {
            'data': rsz_volume_dtype,
            'normalize': False,
            **meta_data
        })

        label_src_volume_dtype = label_resize_volume.astype(np.uint8)
        poses = {p: np.argwhere(label_src_volume_dtype == p) for p in [LEFT, RIGHT]}
            # np.argwhere()
        tables = {
            LEFT: 'left', RIGHT: 'right'
        }
        np.save(npy_gt_name, {
            'data': poses,
            'order': 'zyx',
            'table': tables,
            **meta_data
        })
        logger.info(f'save complete\n {npy_src_name=}\n {npy_gt_name=}')


def display_2d_image(source_path='D:/temp/make/source',
                     gt_path='D:/temp/make/gt', save_path='d:/temp/make/result',
                     stride_ratio=0.05):

    src_files = diskmanager.deep_serach_files(source_path, ['.npy'])
    gt_files = diskmanager.deep_serach_files(gt_path, ['.npy'])

    def normalize_volume(volume):
        dtypeinfo = np.iinfo(volume.dtype)
        return (volume - dtypeinfo.min) / (dtypeinfo.max - dtypeinfo.min)
    for src, gt in zip(src_files, gt_files):
        src_item = np.load(src, allow_pickle=True).item()
        gt_item = np.load(gt, allow_pickle=True).item()

        norm_volume = normalize_volume(src_item.get('data'))

        coord_order = gt_item['order']
        # gt_item['data']
        # gt_item
        coords = gt_item['data']

        table = gt_item['table']
        left_right_key = [k for p in ['left', 'right'] for k, v in table.items() if p == v]
        left_right_coords = [coords[key] for key in left_right_key]
        # [k for k, v in gt_item['table'].items() if v == 'left']
        label_volume = recon_volume2(left_right_coords, norm_volume.shape, coords_order=coord_order, dtype='uint8')

        stride = int(norm_volume.shape[0] * stride_ratio)
        # axial
        image_utils.show_2mask_image(norm_volume * 255, label_volume, show=False,
                                     save_path=save_path, image_save=True, create_sub_dir=True,
                                     stride=stride, in_rainbow_size=10)
        # coronal
        vol1, vol2 = [np.transpose(v, [1, 2, 0]) for v in [norm_volume, label_volume]]
        stride = int(norm_volume.shape[0] * stride_ratio)

        image_utils.show_2mask_image(vol1 * 255, vol2, show=False,
                                     save_path=save_path, image_save=True, create_sub_dir=True, transpose=True,
                                     stride=stride, in_rainbow_size=10)


def main(args=None):
    import argparse
    import sys
    if args is not None:
        for a in args:
            sys.argv.extend(a.split('='))
    parser = argparse.ArgumentParser(description='UNet3D')
    parser.add_argument('--source', default='', type=str, help='source ct directory')
    parser.add_argument( '--gt',  default='', type=str, help='source gt directory')
    parser.add_argument( '--savedir',  default='', type=str, help='save npy directoreis')

    args = parser.parse_args()
    # from trainer import
    convert_main(args.source, args.gt, args.savedir)


if __name__ == '__main__':
    # convert_main()
    # pytest.main()
    # import pdb; pdb.set_trace()
    # pytest.main()
    main([
        '--source=D:/dataset/ai_hub_labels/CTDATA',
        '--gt=D:/dataset/ai_hub_labels/CTDATA',
        '--savedir=d:/temp/make_not_vtk',
    ])

    # display_2d_image()