# from interpolator import TrilinearInterpolator

# from tools.trilinear.interpolator import TrilinearInterpolator
# import
from pyinterpolate import InterpolateWrapper, TrilinearInterpolator
# from interpolator import TrilinearInterpolator
# from pyInterpolator import CTrillinear
# from tools.trilinear.pyInterpolator import CTrillinear
# from interpolator import is_slicced_numpy
from pyinterpolate.interpolator import is_sliced_numpy
# from InterpolateWrapper import InterpolateWrapper
import time
import numpy as np
# from commons import timefn2


def timefn2(fn):
    # @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        time_str = "@timefn: {} took {} secons".format(fn.__name__, t2 - t1)
        print(time_str)
        return result
    return measure_time


def test_main_innterp_float32_64():

    image_shape = np.array([4, 5, 7])
    image_data=  np.random.randn(*image_shape) * 100
    image_data = image_data.astype(np.float64)
    x = np.arange(24).reshape([2, 3, 4]).astype('float64')
    # t = CTrillinear(x, True, 0)
    image_data_32 = image_data.astype(np.float32)

    wrapper1 = InterpolateWrapper(image_data)

    wrapper2 = InterpolateWrapper(image_data)
    #
    # @timefn2
    # def run_my():
    #     res = wrapper1(warps)

    warps = np.random.uniform(np.zeros_like(image_shape), image_shape - 1, [1000000, 3])
    res1 = wrapper1(warps)
    res2 = wrapper2(warps)

    assert np.allclose(res1, res2)


def test_main2():

    image_shape = np.array([200, 320, 150])
    image_data=  np.random.randn(*image_shape) * 100
    image_data = image_data.astype(np.float64)
    x = np.arange(24).reshape([2, 3, 4]).astype('float64')
    # t = CTrillinear(x, True, 0)
    # t.debug_print()

    warps = np.random.uniform(np.zeros_like(image_shape), image_shape - 1, [1000000, 3])

    @timefn2
    def init():
        InterpolateWrapper(image_data)
    for i in range(10):
        init()

    wrapper1 = InterpolateWrapper(image_data)

    wrapper2 = InterpolateWrapper(image_data, api="scipy")

    @timefn2
    def run_my():
        res = wrapper1(warps)
        return res


    @timefn2
    def run_scipy():
        res = wrapper2(warps)
        return res

    r1 = run_my()
    run_my()
    r2 = run_scipy()
    np.testing.assert_allclose(r1, r2)
    run_scipy()





def test_main():
    from scipy.interpolate import RegularGridInterpolator

    image_shape = np.array([600, 600, 600])
    image_data=  np.random.randn(*image_shape) * 100
    image_data = image_data.astype(np.float64)
    warps = np.random.uniform(np.zeros_like(image_shape), image_shape-1, [1000000, 3])
    print("warp:", warps.min(axis=0), warps.max(axis=0))
    # print(image_data.shape, image_shape)
    num_build = 10
    t0 = time.time()
    for _ in range(num_build):
        tril = TrilinearInterpolator(image_data)
    time_build_my = time.time() - t0

    t1 = time.time()
    results1 = tril(warps, bounds_error=True)
    t2 = time.time()
    dt_pybind = t2 - t1
    print("tact pybind", dt_pybind)

    d, h, w = image_data.shape
    mz = np.linspace(0, d - 1, d)
    my = np.linspace(0, h - 1, h)
    mx = np.linspace(0, w - 1, w)

    t_regular = time.time()
    for _ in range(num_build):
        interp = RegularGridInterpolator((mz, my, mx), image_data, bounds_error=False, fill_value=0, method="linear")
    t1 = time.time()
    time_build_scipy = t_regular - t1
    results2 = interp(warps)
    t2 = time.time()
    dt_scipy = t2 - t1
    print("tact scipy interpolator", dt_scipy)
    diff = np.abs(results2 - results1)

    print("faster than x{}".format(dt_scipy/dt_pybind))
    print(f'build time {time_build_scipy=} / {time_build_my=}')
    print("faster than x{} in total(included build time)".format((dt_scipy + time_build_scipy)/(dt_pybind + time_build_my)))
    print("max difference:{:.2f}".format(np.max(diff)))

def test_enum():

    image_shape = np.array([125, 200, 300])
    image_data=  np.random.randn(*image_shape) * 100
    image_data = image_data.astype(np.float32)
    x = TrilinearInterpolator(image_data)
    y = TrilinearInterpolator(image_data, "nearest")
    print(x.method)
    test_main()


def test_interp():
    image_shape = np.array([4, 5, 7])
    image_data =  np.abs(np.random.randn(*image_shape)) * 100 + 10
    grid = np.stack(np.meshgrid(*[np.arange(i) for i in image_shape], indexing='ij'), axis=-1)
    grid = grid.reshape([-1, 3])
    i, j, k = grid[:, 0], grid[:, 1], grid[:, 2]
    assert np.all(image_data[i, j, k] == image_data.ravel())
    interp = InterpolateWrapper(image_data)
    # warp = interp(grid[:, ::-1].astype('float32'))
    warp = interp(grid.astype('float32'))
    res = warp.reshape(image_shape)
    assert np.all(np.isclose(res, image_data)), 'not same original data'

def test_out_of_bound():
    image_shape = np.array([4, 5, 7])
    image_data =  np.abs(np.random.randn(*image_shape)) * 100 + 10
    grid = np.stack(np.meshgrid(*[np.arange(i*10) for i in image_shape], indexing='ij'), axis=-1)
    grid = grid.reshape([-1, 3])
    # i, j, k = grid[:, 0], grid[:, 1], grid[:, 2]
    # assert np.all(image_data[i, j, k] == image_data.ravel())
    interp = InterpolateWrapper(image_data)
    # warp = interp(grid[:, ::-1].astype('float32'))
    warp = interp(grid.astype('float32'))
    # res = warp.reshape(image_shape)

    # assert np.isclose(warp, 0).any()
    assert np.any(warp == 0.)
    interp2 = InterpolateWrapper(image_data, bounds_error=False)
    res2 = interp2(grid.astype('float32'))
    assert np.all(res2>0.)
    # assert np.all(np.isclose(res, image_data)), 'not same original data'

def is_slicced_numpy(arr:np.ndarray):
    if arr.ndim > 1:
        elem = np.array(arr.strides) // arr.itemsize
        return (elem[0] * arr.shape[0]) != arr.size
    else:
        raise ValueError('cannot check')


def test_numpy_sliced():
    x = np.random.randn(5, 3, 2)
    y = x[1:2, 1:2, 0:2]
    assert is_slicced_numpy(x) == False
    assert is_slicced_numpy(y) == True
    z = np.arange(50)

    # assert is_slicced_numpy(z) == False
    z0 = z[5:]
    # assert is_slicced_numpy(z0) == True

if __name__=="__main__":
    # test_enum()
    # test_main()
    # test_main2()
    # test_interp()
    # test_out_of_bound()
    # test_main_innterp_float32_64()
    test_main()
    # test_numpy_sliced()