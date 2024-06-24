import pytest
import numpy as np
from tools import InterpolateWrapper


# def sample_coords()
def test_interpolate():
    x = np.random.randn(50, 30, 30)

    interp = InterpolateWrapper(x, method='linear')
    sample_corods = np.random.uniform(np.zeros_like(x.shape), x.shape, [5000, 3])
    # assert sam
    sampling = interp(sample_corods)
    assert sampling.shape == (sample_corods.shape[0],)


def test_equal_sampling():
    x = np.random.randn(50, 30, 30)

    interp = InterpolateWrapper(x, method='linear')
    sample_corods = np.stack(np.meshgrid(*[np.arange(i) for i in x.shape], indexing='ij'), axis=-1)
    sample_corods_reshape = sample_corods.reshape([-1, 3])
    sampling = interp(sample_corods_reshape).reshape(x.shape)
    assert sampling.shape == x.shape
    assert np.allclose(sampling, x)

    # assert sam
    # assert np.all()
if __name__ == '__main__':
    pytest.main([
        '-s',
        '--color=yes',
        '-rGA',
        'test_trilinaer.py',

    ])

    # assert


