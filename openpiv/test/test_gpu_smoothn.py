"""Test module for gpu_smoothn.py.

Still need to test for non-isotropic spacing and n-dim vector fields.

"""

import numpy as np
import pytest
from math import floor

import scipy.fft as fft
import pycuda.gpuarray as gpuarray

import openpiv.gpu_smoothn as gpu_smoothn
from openpiv.test.test_gpu_misc import generate_np_array, generate_array_pair

DTYPE_i = np.int32
DTYPE_f = np.float32
DTYPE_c = np.complex64

fixture_dir = './fixtures/gpu_smoothn/'


# UTILS
def generate_cosine_field(shape, wavelength=10, amplitude=1):
    """Returns multi-modal cosine field."""
    y = amplitude * np.ones(shape)
    mesh_coords = np.meshgrid(*[np.arange(dim) for dim in shape])
    for coords in mesh_coords:
        y *= np.cos(coords * 2 * np.pi / wavelength)

    return y.astype(DTYPE_f)


def generate_noise_field(shape, scale: float = 1):
    """Returns field of pseaudo-random gaussian noise."""
    np.random.seed(0)
    return np.random.normal(scale=scale, size=shape).astype(DTYPE_f)


# UNIT TESTS
@pytest.mark.parametrize('shape', [(13, 13), (14, 14), (15, 15), (16, 16)])
@pytest.mark.parametrize('norm', ['forward', 'backward', 'ortho'])
def test_gpu_fft(shape, norm):
    y, y_d = generate_array_pair(shape, magnitude=2, offset=-1)

    f_fft = fft.fft(y, norm=norm)
    f_fft_gpu = gpu_smoothn.gpu_fft(y_d, norm=norm, full_frequency=True).get()

    assert np.allclose(f_fft_gpu, f_fft)


@pytest.mark.parametrize('shape', [(13, 13), (14, 14), (15, 15), (16, 16)])
@pytest.mark.parametrize('norm', ['forward', 'backward', 'ortho'])
def test_gpu_ifft(shape, norm):
    m, n = shape

    y = generate_np_array(shape, magnitude=2, offset=-1)
    y_fft = fft.fft(y, norm=norm)
    y_fft_d = gpuarray.to_gpu(y_fft)

    y_ifft = fft.ifft(y_fft, norm=norm)
    y_ifft_gpu = gpu_smoothn.gpu_ifft(y_fft_d, norm=norm, inverse_width=n, full_frequency=True).get()

    assert np.allclose(y_ifft_gpu, y_ifft, rtol=1.e-4)


@pytest.mark.parametrize('shape', [(13, 13), (14, 14), (15, 15), (16, 16)])
@pytest.mark.parametrize('norm', ['forward', 'backward', 'ortho'])
def test_gpu_dct(shape, norm):
    y, y_d = generate_array_pair(shape, magnitude=2, offset=-1)

    y_dct = fft.dct(y, norm=norm)
    y_dct_gpu = gpu_smoothn.gpu_dct(y_d, norm=norm).get()

    assert np.allclose(y_dct_gpu, y_dct, rtol=1.e-4)


@pytest.mark.parametrize('shape', [(13, 13), (14, 14), (15, 15), (16, 16)])
@pytest.mark.parametrize('norm', ['forward', 'backward', 'ortho'])
def test_gpu_idct(shape, norm):
    y = generate_np_array(shape, magnitude=2, offset=-1)
    y_dct = fft.dct(y, norm=norm)
    y_dct_d = gpuarray.to_gpu(y_dct)

    y_idct = fft.idct(y_dct, norm=norm)
    y_idct_gpu = gpu_smoothn.gpu_idct(y_dct_d, norm=norm).get()

    assert np.allclose(y_idct_gpu, y_idct, rtol=1.e-4)


@pytest.mark.parametrize('shape', [(16,), (16, 16), (16, 16, 16)])
def test_replace_non_finite(shape, ndarrays_regression):
    # Test doesn't check whether nans are replaced by nearest neighbour.
    y = generate_np_array(shape, magnitude=2, offset=-1)
    finite = generate_np_array(shape, magnitude=2, d_type=DTYPE_i, seed=1).astype(bool)
    f0 = y.copy()
    y[finite == 0] = np.nan

    gpu_smoothn.replace_non_finite(y, finite)

    assert np.array_equal(y[finite], f0[finite])


@pytest.mark.parametrize('shape', [(16,), (16, 16), (16, 16, 16)])
def test_initial_guess(shape: tuple, data_regression):
    y = generate_cosine_field(shape, wavelength=100)
    noise = generate_noise_field(shape, scale=0.1)

    z0 = gpu_smoothn._initial_guess([y + noise])[0]
    z0_l2_norm = np.linalg.norm(z0 - y)
    noise_l2_norm = np.linalg.norm(noise)
    noise_ratio = float(z0_l2_norm / noise_l2_norm)

    data_regression.check({'noise_ratio': noise_ratio})


@pytest.mark.parametrize('shape', [(16,), (16, 16), (16, 16, 16)])
@pytest.mark.parametrize('smooth_order', [0, 1, 2])
def test_p_bounds(shape, smooth_order, data_regression):
    p_min, p_max = gpu_smoothn._p_bounds(shape, smooth_order=smooth_order)

    data_regression.check({'p_min': p_min, 'p_max': p_max})


@pytest.mark.parametrize('shape', [(16,), (16, 16), (16, 16, 16)])
def test_lambda(shape, ndarrays_regression):
    y = generate_np_array(shape, magnitude=2, offset=-1)

    spacing = np.ones(y.ndim, dtype=DTYPE_f)
    lambda_ = gpu_smoothn._lambda(y, spacing)

    ndarrays_regression.check({'lambda': lambda_})


@pytest.mark.parametrize('shape', [(16,), (16, 16), (16, 16, 16)])
@pytest.mark.parametrize('transform', [fft.dct, fft.idct])
def test_dct_nd(shape, transform):
    """Test that output arrays are C-contiguous, which is required for proper CUDA indexing."""
    data = generate_np_array(shape, magnitude=2, offset=-1)

    data_dct_nd = gpu_smoothn._dct_nd(data, f=transform)

    assert data_dct_nd.flags.c_contiguous


@pytest.mark.parametrize('shape', [(16,), (16, 16), (16, 16, 16)])
@pytest.mark.parametrize('w_mean', [None, 1])
def test_gcv(shape, w_mean, data_regression):
    p = np.log10(0.5)

    y = generate_np_array(shape, magnitude=2, offset=-1)
    y_dct = fft.dct(y)
    w = generate_np_array(shape, seed=1)
    if w_mean is None:
        w_mean = np.mean(w)
    is_finite = generate_np_array(shape, magnitude=2, d_type=DTYPE_i, seed=2).astype(bool)
    nof = np.sum(is_finite)
    spacing = np.ones(y.ndim, dtype=DTYPE_f)
    lambda_ = gpu_smoothn._lambda(y, spacing=spacing)

    gcv_score_non_weighted = gpu_smoothn._gcv(p, [y], [y_dct], w, lambda_, is_finite, 1, nof)
    gcv_score_weighted = gpu_smoothn._gcv(p, [y], [y_dct], w, lambda_, is_finite, w_mean, nof)

    data_regression.check({'gcv_score_non_weighted': gcv_score_non_weighted, 'gcv_score_weighted': gcv_score_weighted})


@pytest.mark.parametrize('shape', [(16,), (16, 16), (16, 16, 16)])
@pytest.mark.parametrize('smooth_order', [0, 1, 2])
def test_leverage(shape: tuple, smooth_order, data_regression):
    spacing = np.ones(len(shape), dtype=DTYPE_f)
    h = gpu_smoothn._leverage(0.5, spacing, smooth_order=smooth_order)

    data_regression.check({'h': h})


@pytest.mark.parametrize('shape', [(16,), (16, 16), (16, 16, 16)])
@pytest.mark.parametrize('weight_method', ['cauchy', 'talworth', 'bisquare'])
def test_robust_weights(shape, weight_method, ndarrays_regression):
    y = generate_np_array(shape, magnitude=2, offset=-1)
    z = generate_np_array(shape, magnitude=2, offset=-1, seed=1)
    is_finite = generate_np_array(shape, magnitude=2, d_type=DTYPE_i, seed=2).astype(bool)
    spacing = np.ones(y.ndim, dtype=DTYPE_i)
    h = gpu_smoothn._leverage(0.5, spacing)

    w = gpu_smoothn._robust_weights([y], [z], is_finite, h, weight_str=weight_method)

    ndarrays_regression.check({'w': w})


@pytest.mark.parametrize('shape', [(13, 13), (14, 14), (15, 15), (16, 16)])
@pytest.mark.parametrize('direction', ['forward', 'backward'])
def test_dct_order_forward(shape, direction):
    m, n = shape
    y, y_d = generate_array_pair(shape, magnitude=2, offset=-1)

    z = gpu_smoothn._dct_order(y_d, direction=direction).get()
    if direction == 'backward':
        y, z = z, y
    # Sequence given by Eq. 20 of Makhoul, 1980.
    y_forward_sequence_a = y[:, [2 * i for i in range(0, floor((n - 1) / 2) + 1)]]
    y_forward_sequence_b = y[:, [2 * n - 2 * i - 1 for i in range(floor((n + 1) / 2), n)]]
    y_forward = np.hstack([y_forward_sequence_a, y_forward_sequence_b])

    assert np.array_equal(z, y_forward)


@pytest.mark.parametrize('shape', [(13, 13), (14, 14), (15, 15), (16, 16)])
@pytest.mark.parametrize('offset', [0, 1])
@pytest.mark.parametrize('left_pad', [0, 1])
def test_flip_frequency_real(shape, offset, left_pad):
    m, n = shape
    y, y_d = generate_array_pair(shape, magnitude=2, offset=-1)

    flip_width = n - 1
    y_flipped = np.flip(y, axis=1)[:, offset:offset + flip_width - left_pad]
    z = gpu_smoothn._flip_frequency_real(y_d, flip_width=flip_width, offset=offset, left_pad=left_pad).get()

    if left_pad > 0:
        assert np.all(z[:, :left_pad] == 0)
    assert np.array_equal(z[:, left_pad:], y_flipped)


@pytest.mark.parametrize('shape', [(13, 13), (14, 14), (15, 15), (16, 16)])
@pytest.mark.parametrize('offset', [0, 1])
@pytest.mark.parametrize('left_pad', [0, 1])
def test_flip_frequency_comp(shape, offset, left_pad):
    m, n = shape
    y = generate_np_array(shape, magnitude=2, offset=-1) + 1j * generate_np_array(shape, magnitude=2, offset=-1, seed=1)
    y_d = gpuarray.to_gpu(y)

    flip_width = n - 1
    y_flipped = np.flip(y, axis=1)[:, offset:offset + flip_width - left_pad].conj()
    z = gpu_smoothn._flip_frequency_comp(y_d, flip_width=flip_width, offset=offset, left_pad=left_pad).get()

    if left_pad > 0:
        assert np.all(z[:, :left_pad] == 0)
    assert np.array_equal(z[:, left_pad:], y_flipped)


@pytest.mark.parametrize('shape', [(13, 13), (14, 14), (15, 15), (16, 16)])
def test_reflect_frequency_comp(shape):
    m, n = shape
    freq_width = n // 2 + 1
    y = generate_np_array((m, freq_width), magnitude=2, offset=-1) + 1j * generate_np_array((m, freq_width),
                                                                                            magnitude=2, offset=-1,
                                                                                            seed=1)
    y_d = gpuarray.to_gpu(y)

    y_full = np.hstack([y, np.flip(y[:, 1:n - freq_width + 1], axis=1).conj()])
    z = gpu_smoothn._reflect_frequency_comp(y_d, full_width=n).get()

    assert np.array_equal(y_full, z)


# INTEGRATION TESTS
# Need tests for: mask, max_iter, smooth_order, w, z0.
# Need to test for unexpected inputs.
@pytest.mark.integtest
@pytest.mark.parametrize('shape', [(16,), (16, 16), (16, 16, 16)])
@pytest.mark.parametrize('s', [None, 50])
@pytest.mark.parametrize('robust', [True, False])
def test_smoothn(shape, s, robust, data_regression):
    y = generate_cosine_field(shape, wavelength=50)
    noise = generate_noise_field(shape, scale=0.5)

    z = gpu_smoothn.smoothn(y + noise, s=s, robust=robust)[0]
    z_l2_norm = np.linalg.norm(z - y)
    noise_l2_norm = np.linalg.norm(noise)
    noise_ratio = float(z_l2_norm / noise_l2_norm)
    z_gpu = gpu_smoothn.smoothn(y + noise, s=s, robust=robust)[0]

    data_regression.check({'noise_ratio': noise_ratio})
    assert np.array_equal(z_gpu, z)
