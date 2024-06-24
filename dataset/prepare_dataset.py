import os
import numpy as np
import pytest
import vtk_utils
import json


from trainer import get_logger


get_runtime_logger = get_logger

def read_aihub_json(filename):
    # assert len(json_file) > 0
    with open(filename, 'r', encoding='UTF8') as f:
        data = json.load(f)

    right_coords = data['annotation']['tooth']['Right']['coordinate']
    left_coords = data['annotation']['tooth']['Left']['coordinate']
    return np.asarray(left_coords).reshape([-1, 3]), np.asarray(right_coords).reshape([-1, 3])


def test_read_nerve():
    filename = 'D:/dataset/ai_hub_labels/CTDATA/32/CT3D'
    pass
    assert os.path.exists(filename)
    from tools import dicom_read_wrapper
    vtk_order = False
    volume, spacing = dicom_read_wrapper.read_dicom_wrapper(filename, vtkorder=vtk_order)

    print(volume.shape, spacing)
    json_file = glob.glob(os.path.join(filename, '*.json'))
    # assert len(json_file) > 0
    # with open(json_file[0], 'r', encoding='UTF8') as f:
    #     data = json.load(f)
    # coords1 = data['annotation']['tooth']['Right']['coordinate']
    # coords2 = data['annotation']['tooth']['Left']['coordinate']
    left, right = read_aihub_json(json_file[0])
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
        coords3 = reverse_coords(coords2, volume.shape[::-1], [1])
    else:

        coords3 = reverse_coords(coords2, volume.shape[::-1], [2])

    vtk_utils.split_show([volume, coords3, vtk_utils.get_axes(100)], [coords3, vtk_utils.get_axes(100)])


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

    from trainer import diskmanager
    args = parser.parse_args()

    # print(args)


if __name__ == '__main__':
    # convert_main()
    pytest.main()
    # import pdb; pdb.set_trace()
    pytest.main()
    main([
        '--source=d:/temp',
        '--gt=d:/temp',
        '--savedir=d:/temp/test',
    ])
