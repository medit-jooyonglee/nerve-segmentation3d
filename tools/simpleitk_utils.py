import numpy as np
import SimpleITK as sitk


def read_sitk_volume(pathname):

    print("Reading Dicom directory:", pathname)
    reader = sitk.ImageSeriesReader()

    dicom_names = reader.GetGDCMSeriesFileNames(pathname)
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