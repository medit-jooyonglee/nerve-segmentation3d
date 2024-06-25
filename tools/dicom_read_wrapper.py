# import vtk_utils
import re
import numpy as np
import os, glob
# from tools.meshDicom.meshlibs import CVolumeDicomReader
try:
    from tools.pymeshlibs import CVolumeDicomReader
except Exception as e:
    print(e.args)
    CVolumeDicomReader = None

import pydicom
from tools import vtk_utils


def read_single_image_array(file):
    ds = pydicom.read_file(file)

    img = ds.pixel_array
    dc, dw = float(ds.WindowCenter), float(ds.WindowWidth)
    a, b = dc - dw / 2, dc + dw / 2
    res = ((img - a) * (255 / (b - a))).astype(np.uint8)
    return res


def read_volume_pydicom(path, **kwargs):
    # def read_dicom_series(folder_path):
    vtkorder = kwargs.get('vtkorder', False)
    return_windowing = kwargs.get('return_windowing', False)
    normalize = kwargs.get('normalize', True)
    dicom_files = glob.glob(os.path.join(path, '*.dcm'))
    slices = [pydicom.dcmread(file) for file in dicom_files]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    ds = slices[0]

    instance_numbers = [int(slice_instance.InstanceNumber) for slice_instance in slices]
    volume = np.stack([slice_instance.pixel_array for slice_instance in slices], axis=0)

    image_position_tag = (0x0020, 0x0032)
    image_orientation_tag = (0x0020, 0x0037)
    slice_location_tag = (0x0020, 0x1041)
    position = ds[image_position_tag]
    orientation = ds[image_orientation_tag]
    spacing_tag = (0x0028,0x0030)
    spacing = ds[spacing_tag].value
    spacing = [*spacing, spacing[0]] if len(spacing) == 2 else spacing
    # ds
    ww, wc = ds.WindowWidth, ds.WindowCenter
    dmin = int(wc) - int(ww) / 2
    dmax = int(wc) + int(ww) / 2

    volume = volume[::-1, ::-1] if vtkorder else volume
    if normalize:
        norm_volume = np.clip((volume - dmin) / (dmax - dmin), 0, 1)
    else:
        norm_volume = volume
    windowing = (dmin, dmax)

    outs = [norm_volume, spacing]
    if return_windowing:
        outs.append(windowing)
    return outs


def read_dicom_wrapper(*args, **kwargs):
    """

    Parameters
    ----------
    path : str
        pathname
    kwargs :
        normalize : bool optioon voxel-array normalized
        return_windowing : bool option return windowing (min, max) for visualize
        progress_callback : function callback dicom callback as list
        method : str 'mybinding' or 'vtk'
    Returns
    -------
    norm_voxel : np.ndarray
        the normalize volume or original volume
    spacing : np.ndarray
        the pixel spacing. actual size(ex. mm) per pixel
    windowing : tuple[float], optional
        the range value. min~max
    """
    if CVolumeDicomReader is None:
        return read_volume_pydicom(args[0], **kwargs)
    # np.ndarray,
    #     volume (d, h, w)
    # np.ndarray,
    #     spacing (3,)
    # tuple[float]
    #     range min, max

    method = kwargs.pop('method', 'mybinding')
    assert method in ['mybinding', 'vtk']
    if method == 'vtk' or CVolumeDicomReader is None:
        vtkorder = kwargs.get('vtkorder', True)
        res = vtk_utils.read_dicom(*args, **kwargs)
        if vtkorder:
            return res
        else:
            res = list(tuple)
            # LAI --> LPS
            res[0] = res[0][::-1, ::-1]
            return tuple(res)
    else:
        pathname = args[0]
        hanCount = len(re.findall(u'[\u3130-\u318F\uAC00-\uD7A3]+', pathname))
        # encode_path = path.encode('euc-kr') if hanCount > 0 else path
        encode_path = pathname.encode('euc-kr') if hanCount > 0 else pathname

        reader = CVolumeDicomReader()
        vtkorder = kwargs.get('vtkorder', True)
        reader.vtkOrder = vtkorder
        res = reader.readVolume(encode_path)
        if res:
            volume = reader.getData()


            vol_array = volume.reshape(reader.shape)
            ww, wl = reader.windowing

            dmin = wl - ww / 2
            dmax = wl + ww / 2

            if kwargs.get('normalize', True):
                vol_array = vol_array.astype(np.float32)
                norm_voxel = np.clip((vol_array - dmin) / (dmax - dmin), 0., 1.)
            else:
                norm_voxel = vol_array
            spacing = np.array(reader.spacing)
            # np.isclose(spacing[0])
            # tag 값 spacing 정보 없는 경우가 있다.
            if np.isclose(spacing, 0).any():
                zeros = np.isclose(spacing, 0)
                nonzero = spacing[np.logical_not(zeros)][0]
                spacing = np.where(zeros, nonzero, spacing)
            if kwargs.get('return_windowing', False):
                return (norm_voxel, spacing, (dmin, dmax))
            else:
                return (norm_voxel, spacing)

        else:
            # dcmtk 으로 읽기 실패 시 vtk로 읽기 시도
            vtkorder = kwargs.get('vtkorder', True)
            res = vtk_utils.read_dicom(*args, **kwargs)
            if vtkorder:
                return res
            else:
                res = list(tuple)
                # LAI --> LPS
                res[0] = res[0][::-1, ::-1]
                return tuple(res)
            # raise ValueError('failed loading')



