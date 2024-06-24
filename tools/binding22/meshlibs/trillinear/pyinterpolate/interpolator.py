from pyInterpolator import CTrillinearf, CTrillineard
# CTrillinearf
import time
import numpy as np
# import vtk
# vtk.vtkPolyData
# CTrillinearf
def is_sliced_numpy(arr:np.ndarray):
    """
    checked if array is sliced
    :param arr:
    :return:
    """
    if arr.ndim > 1:
        elem = np.array(arr.strides) // arr.itemsize
        return (elem[0] * arr.shape[0]) != arr.size
    else:
        raise ValueError('cannot check')


class TrilinearInterpolator(object):
    def __init__(self, voxels, method="linear", bounds_error=True, fill_value=0):
        assert voxels.ndim == 3
        # CTrillinear.LINEAR
        assert method in ["linear", "nearest"]
        method_dictf = {
            "linear" : CTrillinearf.Method.LINEAR,
            "nearest": CTrillinearf.Method.NEAREST
        }

        method_dictd = {
            "linear" : CTrillinearf.Method.LINEAR,
            "nearest": CTrillinearf.Method.NEAREST
        }
        # method_dictf =
        if is_sliced_numpy(voxels):
            voxels = voxels.copy()
            print('sliced array')

        # met = method_dict[method]
        self.dtype = voxels.dtype
        # binding float32 으로 구현되어 있음
        if np.issubdtype(voxels.dtype, np.floating):
            if voxels.dtype == np.float32:
                self.interpolator = CTrillinearf(voxels, bounds_error, fill_value)
                met = method_dictf[method]
            elif voxels.dtype == np.float64:
                self.interpolator = CTrillineard(voxels, bounds_error, fill_value)
                met = method_dictd[method]
            else:
                voxels = voxels.astype(np.float32)
                self.interpolator = CTrillinearf(voxels, bounds_error, fill_value)
                met = method_dictf[method]


        else:
            voxels = voxels.astype(np.float32)
            self.interpolator = CTrillinearf(voxels, bounds_error, fill_value)
            met = method_dictf[method]
        # if voxels.dtype != 'float32':
        #     voxels = voxels.astype('float32')
        # 원본 타입으로 복구하는 옵션
        self._restore_source_float_type = False
        self._restore_source_int_type = True
        self.voxels = voxels
        self.image_shape = np.array(voxels.shape)
        # bound_error = fill_value is not None
        # set default bound value as zero
        # bound_error_value = 0.0 if fill_value is None else fill_value
        # super(TrilinearInterpolator, self).__init__(voxels.ravel(), self.image_shape, met, bound_error, bound_error_value)
        # super(TrilinearInterpolator, self).__init__(voxels.ravel(), self.image_shape, met, True, bound_error_value))
        # super(_TrilinearInterpolatorf, self).__init__(voxels, bounds_error, fill_value)
        self.interpolator._method = met

    def __call__(self, warp_points, bounds_error=True):
        assert warp_points.shape[1] == 3
        # if not bounds_error:
        #     max_indices = self.image_shape - 1
        #     warp_points = np.clip(warp_points, np.zeros([3]), max_indices)
        res = np.squeeze(self.interpolator.interpolate(warp_points), axis=-1)
        # source type으로 복구
        if np.issubdtype(res.dtype, np.floating) and self._restore_source_float_type and res.dtype != self.dtype:
            res = res.astype(self.dtype)
        elif np.issubdtype(res.dtype, np.integer) and self._restore_source_int_type and res.dtype != self.dtype:
            res = res.astype(self.dtype)
        else:
            pass

        return res

    @property
    def method(self):
        return self._method


def test_main():
    from scipy.interpolate import RegularGridInterpolator

    image_shape = np.array([125, 200, 300])
    image_data=  np.random.randn(*image_shape) * 100
    image_data = image_data.astype(np.float32)
    warps = np.random.uniform(np.zeros_like(image_shape), image_shape-1, [1000000, 3])
    print("warp:", warps.min(axis=0), warps.max(axis=0))
    # print(image_data.shape, image_shape)
    tril = TrilinearInterpolator(image_data)

    t1 = time.time()
    results1 = tril(warps, bounds_error=True)
    t2 = time.time()
    dt_pybind = t2 - t1
    print("tact pybind", dt_pybind)

    d, h, w = image_data.shape
    mz = np.linspace(0, d - 1, d)
    my = np.linspace(0, h - 1, h)
    mx = np.linspace(0, w - 1, w)

    interp = RegularGridInterpolator((mz, my, mx), image_data, bounds_error=False, fill_value=0, method="linear")
    t1 = time.time()
    results2 = interp(warps)
    t2 = time.time()
    dt_scipy = t2 - t1
    print("tact scipy interpolator", dt_scipy)
    diff = np.abs(results2 - results1)

    print("faster than x{}".format(dt_scipy/dt_pybind))
    print("max difference:{:.2f}".format(np.max(diff)))


def test_enum():

    image_shape = np.array([125, 200, 300])
    image_data=  np.random.randn(*image_shape) * 100
    image_data = image_data.astype(np.float32)
    x = TrilinearInterpolator(image_data)
    y = TrilinearInterpolator(image_data, "nearest")
    print(x.method)
    test_main()


if __name__=="__main__":
    # test_enum()
    test_main()