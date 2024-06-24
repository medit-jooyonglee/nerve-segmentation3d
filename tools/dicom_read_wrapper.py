# import vtk_utils
import re
import numpy as np

# from tools.meshDicom.meshlibs import CVolumeDicomReader
try:
    from tools.pymeshlibs import CVolumeDicomReader
except Exception as e:
    print(e.args)
    CVolumeDicomReader = None
from tools import vtk_utils

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



