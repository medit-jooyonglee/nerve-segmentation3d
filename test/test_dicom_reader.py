import os
import sys

import pydicom.filereader


def handle_base_path():
    sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

handle_base_path()
    # pydicom.dcmread()
from tools import dicom_read_wrapper


def test_read_single_image_dicom():
    print(os.getcwd())
    filename = 'test/20231102155648764_0619.dcm'
    image_array = dicom_read_wrapper.read_single_image_array(filename)
    # image_array = dicom_read_wrapper.(filename)
    print('loading complete', image_array.shape)
# image = dicom_read_wrapper.read_dicom_wrapper()

def test_read_volume():
    import pydicom
    pydicom.dcmread('D:/dataset/ai_hub_labels/CTDATA/7/CT3D')

def main():
    test_read_single_image_dicom()


def test_visualize_dicom_read():
    filename = 'D:/dataset/ai_hub_labels/CTDATA/7/CT3D'

    volume, spacing, windowing = dicom_read_wrapper.read_volume_pyciom(filename)


if __name__ == '__main__':
    # main()
    # test_read_volume()
    test_visualize_dicom_read()
