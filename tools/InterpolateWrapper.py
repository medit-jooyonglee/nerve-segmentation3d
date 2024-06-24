import numpy as np
from scipy.interpolate import RegularGridInterpolator

from pyinterpolate.interpolator import TrilinearInterpolator


class InterpolateWrapper(object):
    """
    this class trillinear interpolation class.
    1. scipy - RegularGridInterpolator
    2. user-implemented funtion(LJY) by pybind11
    ex)
    volume # (D,H,W) voxel
    interolator = InterpolateWrapper(volume)
    to_sample_pose # (N, 3)
    warp_image = interolator(to_sample_pose)

    """
    def __init__(self, values, method="linear", api="my", bounds_error=True, fill_value=0.):

        assert method in ["linear", "nearest"]
        assert api in ["my", "scipy"], "set 'my' or 'scipy', 'my' method is implemented c++ & openmp"
        assert len(values.shape) == 3, "supported 3-dimension array"

        if api == "my":
            self.interpolator = TrilinearInterpolator(values,
                                                      method=method, bounds_error=bounds_error,
                                                      fill_value=fill_value)
        elif api == "scipy":

            d, h, w = values.shape
            mz = np.linspace(0, d - 1, d)
            my = np.linspace(0, h - 1, h)
            mx = np.linspace(0, w - 1, w)
            self.interpolator = RegularGridInterpolator(
                (mz, my, mx),
                values, bounds_error=bounds_error,
                fill_value=fill_value,
                method=method
            )
        else:
            raise NotImplementedError("invalid value:{}".format(api))

    def __call__(self, *args, **kwargs):
        return self.interpolator(*args, **kwargs)

