import os
import sys

def handle_base_path():
    sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

handle_base_path()

from tools import dicom_read_wrapper


def test_read_single_image_dicom():
    print(os.getcwd())
    filename = 'test/20231102155648764_0619.dcm'
    image_array = dicom_read_wrapper.read_single_image_array(filename)
    # image_array = dicom_read_wrapper.(filename)
    print('loading complete', image_array.shape)
# image = dicom_read_wrapper.read_dicom_wrapper()


def main():
    test_read_single_image_dicom()


if __name__ == '__main__':
    main()
