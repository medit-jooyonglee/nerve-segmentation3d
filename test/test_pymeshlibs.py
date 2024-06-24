import os.path
import os
import cv2
import numpy as np
from tools.dicom_read_wrapper import read_dicom_wrapper
from tools import vtk_utils
# from trainer import vtk_

# pathname = 'D:/dataset/TeethSegmentation/marker_set(only2class)/sample1/17/CT_김명래 ct'
# pathname = 'D:/dataset/TeethSegmentation/itk_snap_dataset/train/set009/57912_170908165128(4)/CTData'
pathname = 'D:/dataset/ai_hub_labels/CTDATA/3/CT3D'
# pathname = 'D:/dataset/TeethSegmentation/itk_snap_dataset/train/set009/36843_170914174616(4)1210/CTData'

def test_readdicom_reader_wrapper():
    vol, sp = read_dicom_wrapper(pathname)
    # vol, sp = vtk_utils.read_dicom(pathname)
    print(vol.shape, sp)
    assert vol.size > 0
    return vol

vol = test_readdicom_reader_wrapper()
# print(vol.shape)
for i in range(225, 235):
    # a, b = img.min(), img.max()
    cv2.imshow('', (vol[i]*255).astype(np.uint8))
    cv2.waitKey()


# vtk_utils.show_actors([vol])

def test_pydicom():
    import pydicom
    import cv2
    import numpy as np

    for i in range(225, 232):
        file = os.path.join(pathname, f'0000 ({i}).dcm')
        ds = pydicom.read_file(file)

        img = ds.pixel_array
        dc, dw = float(ds.WindowCenter), float(ds.WindowWidth)
        a, b = dc - dw/2, dc + dw/2
        res = ((img - a) * (255/(b-a))).astype(np.uint8)
        # a, b = img.min(), img.max()
        cv2.imshow('', res)
        cv2.waitKey()

