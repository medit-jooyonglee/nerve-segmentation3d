import SimpleITK as sitk
import numpy as np
import vtk_utils
from commons import timefn2

@timefn2
def read_sitk_volume(filename):

    print("Reading Dicom directory:", filename)
    reader = sitk.ImageSeriesReader()

    dicom_names = reader.GetGDCMSeriesFileNames(filename)
    reader.SetFileNames(dicom_names)
    # reader.LoadPrivateTagsOn()
    image = reader.Execute()

    size = image.GetSize()
    print("Image size:", size[0], size[1], size[2])


    # https://simpleitk.readthedocs.io/en/master/link_DicomSeriesReader_docs.html

    def get_major_meta():
        singlereader = sitk.ImageFileReader()
        singlereader.LoadPrivateTagsOn()
        singlereader.SetFileName(dicom_names[0])
        singlereader.ReadImageInformation()

        # window center (0028, 1050)
        # window width (0028, 1051)
        for k in singlereader.GetMetaDataKeys():
            v = singlereader.GetMetaData(k)
            # print(f'({k}) = = "{v}"')
        keys = ['0028|1050', '0028|1051']
        res = [singlereader.GetMetaData(k) for k in keys]
        wc, wl = [float(v) for v in res]
        vmin, vmax = wc - wl /2, wc + wl/2
        return (vmin, vmax)
        # singlereader.GetMetaData()
    drange = get_major_meta()

    spacing = np.array(image.GetSpacing()[::-1])
    print(f"Image Size: {image.GetSize()}")
    print(f"Image PixelType: {sitk.GetPixelIDValueAsString(image.GetPixelID())}")
    # singlereader = sitk.ImageFileReader()
    img_array = sitk.GetArrayFromImage(image)
    norm_array = np.clip((img_array - drange[0]) / (drange[1] - drange[0]), 0, 1)
    return norm_array, spacing, drange

from tools import dicom_read_wrapper
@timefn2
def read_pydicom_volume(filename):
    return dicom_read_wrapper.read_dicom_wrapper(filename)
def test_read_sitk_volume():
    # if len(sys.argv) < 3:
    #     print("Usage: DicomSeriesReader <input_directory> <output_file>")
    #     sys.exit(1)

    filename = 'D:/dataset/ai_hub_labels/CTDATA/4/CT3D'
    res = read_pydicom_volume(filename)
    vol, spacing, drange = read_sitk_volume(filename)
    vtk_utils.show([vol, vtk_utils.get_axes(100)])
    # print("Reading Dicom directory:", filename)
    # reader = sitk.ImageSeriesReader()
    #
    # dicom_names = reader.GetGDCMSeriesFileNames(filename)
    # reader.SetFileNames(dicom_names)
    # # reader.LoadPrivateTagsOn()
    # image = reader.Execute()
    #
    # size = image.GetSize()
    # print("Image size:", size[0], size[1], size[2])
    #
    #
    # # https://simpleitk.readthedocs.io/en/master/link_DicomSeriesReader_docs.html
    #
    # def get_major_meta():
    #     singlereader = sitk.ImageFileReader()
    #     singlereader.LoadPrivateTagsOn()
    #     singlereader.SetFileName(dicom_names[0])
    #     singlereader.ReadImageInformation()
    #
    #     # window center (0028, 1050)
    #     # window width (0028, 1051)
    #     for k in singlereader.GetMetaDataKeys():
    #         v = singlereader.GetMetaData(k)
    #         # print(f'({k}) = = "{v}"')
    #     keys = ['0028|1050', '0028|1051']
    #     res = [singlereader.GetMetaData(k) for k in keys]
    #     wc, wl = [float(v) for v in res]
    #     vmin, vmax = wc - wl /2, wc + wl/2
    #     return (vmin, vmax)
    #     # singlereader.GetMetaData()
    # drange = get_major_meta()
    #
    # print(f"Image Size: {reader.GetSize()}")
    # print(f"Image PixelType: {sitk.GetPixelIDValueAsString(reader.GetPixelID())}")
    # # singlereader = sitk.ImageFileReader()
    # img_array = sitk.GetArrayFromImage(image)
    # norm_array = (img_array - drange[0]) / (drange[1] - drange[0])
    # norm_array = norm_array.clip(0, 1)
    # d

    # print("Writing image:", sys.argv[2])
    #
    # sitk.WriteImage(image, sys.argv[2])
    #
    # if "SITK_NOSHOW" not in os.environ:
    #     sitk.Show(image, "Dicom Series")

test_read_sitk_volume()