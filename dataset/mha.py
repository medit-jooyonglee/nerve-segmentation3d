import pathlib
import SimpleITK as sitk
from tools import vtk_utils
import numpy as np
from commons import timefn2

from panimg.image_builders.metaio_utils import (
    ADDITIONAL_HEADERS,
    FLOAT_OR_FLOAT_ARRAY_MATCH_REGEX,
    load_sitk_image,
    parse_mh_header,
)
# from tests import RESOURCE_PATH
#
# MHD_WINDOW_DIR = RESOURCE_PATH / "mhd_window"

# @timefn2
def read_image_volume(filename, normalize=True):
    path = pathlib.Path(filename)
    header = parse_mh_header(path)
    # print(header)
    img_ref = load_sitk_image(path)
    img_volume = sitk.GetArrayFromImage(img_ref)
    if normalize:
        imin, imax = img_volume.min(), img_volume.max()
        img_volume = ( img_volume - imin ) / (imax - imin)
    else:
        pass
    return img_volume, img_ref

def main():

    # for _ in range()

    filename1 = 'D:/dataset/miccai/Dataset112_ToothFairy2/imagesTr/ToothFairy2F_001_0000.mha'
    filename2 = 'D:/dataset/miccai/Dataset112_ToothFairy2/labelsTr/ToothFairy2F_001.mha'

    src_volume, ref1 = read_image_volume(filename1)
    label_volume, ref2 = read_image_volume(filename2, False)
    # vtk_utils.show([img_volume])
    print(src_volume.shape, label_volume.shape)
    label_volume2 = np.zeros_like(src_volume, dtype=label_volume.dtype)

    label_volume2[-label_volume.shape[0]:] = label_volume

    # extract_vol = np.where(np.logical_or(
    # label_volume2 == 19, label_volume2 == 20
    # ), label_volume2, np.zeros_like(label_volume2))
    actors = vtk_utils.auto_refinement_mask(label_volume2, random_coloring=True)

    vtk_utils.split_show([
        src_volume
    ],  [
        label_volume2
    ],
    [src_volume, *actors]
    )

    cannals__sinus = np.where(np.logical_and(
    label_volume2 >= 3, label_volume2 <= 6
    ), label_volume2, np.zeros_like(label_volume2))

    vtk_utils.split_show([
        src_volume
    ],  [
        cannals__sinus
    ],
    [src_volume, *vtk_utils.auto_refinement_mask(cannals__sinus, random_coloring=True)]
    )


    vtk_utils.split_show([
        src_volume
    ],  [
        label_volume2
    ],
    [src_volume, *actors]
    )

    cannals__sinus = np.where(np.logical_and(
    label_volume2 >= 3, label_volume2 <= 6
    ), label_volume2, np.zeros_like(label_volume2))


    # label_volume2 == 19, label_volume2 == 20
    # ), label_volume2, np.zeros_like(label_volume2))