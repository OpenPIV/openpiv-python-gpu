"""Test module for gpu_smoothn.py.

Still need to test for non-isotropic spacing and n-dim vector fields.

"""

import numpy as np
import pytest
from math import floor
from numbers import Number

import scipy.fft as fft
import pycuda.gpuarray as gpuarray

import openpiv.gpu_smoothn as gpu_smoothn

DTYPE_i = np.int32
DTYPE_f = np.float32
DTYPE_c = np.complex64


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
@pytest.mark.parametrize("shape", [(13, 13), (14, 14), (15, 15), (16, 16)])
@pytest.mark.parametrize("norm", ["forward", "backward", "ortho"])
def test_gpu_fft(shape, norm, array_pair):
    y, y_d = array_pair(shape, center=0.0, half_width=1.0)

    f_fft = fft.fft(y, norm=norm)
    f_fft_gpu = gpu_smoothn.gpu_fft(y_d, norm=norm, full_frequency=True).get()

    assert np.allclose(f_fft_gpu, f_fft)


@pytest.mark.parametrize("shape", [(13, 13), (14, 14), (15, 15), (16, 16)])
@pytest.mark.parametrize("norm", ["forward", "backward", "ortho"])
def test_gpu_ifft(shape, norm, np_array):
    m, n = shape

    y = np_array(shape, center=0.0, half_width=1.0)
    y_fft = fft.fft(y, norm=norm)
    y_fft_d = gpuarray.to_gpu(y_fft)

    y_ifft = fft.ifft(y_fft, norm=norm)
    y_ifft_gpu = gpu_smoothn.gpu_ifft(
        y_fft_d, norm=norm, inverse_width=n, full_frequency=True
    ).get()

    assert np.allclose(y_ifft_gpu, y_ifft, rtol=1.0e-4)


@pytest.mark.parametrize("shape", [(13, 13), (14, 14), (15, 15), (16, 16)])
@pytest.mark.parametrize("norm", ["forward", "backward", "ortho"])
def test_gpu_dct(shape, norm, array_pair):
    y, y_d = array_pair(shape, center=0.0, half_width=1.0)

    y_dct = fft.dct(y, norm=norm)
    y_dct_gpu = gpu_smoothn.gpu_dct(y_d, norm=norm).get()

    assert np.allclose(y_dct_gpu, y_dct, rtol=1.0e-4)


@pytest.mark.parametrize("shape", [(13, 13), (14, 14), (15, 15), (16, 16)])
@pytest.mark.parametrize("norm", ["forward", "backward", "ortho"])
def test_gpu_idct(shape, norm, np_array):
    y = np_array(shape, center=0.0, half_width=1.0)
    y_dct = fft.dct(y, norm=norm)
    y_dct_d = gpuarray.to_gpu(y_dct)

    y_idct = fft.idct(y_dct, norm=norm)
    y_idct_gpu = gpu_smoothn.gpu_idct(y_dct_d, norm=norm).get()

    assert np.allclose(y_idct_gpu, y_idct, rtol=1.0e-4)


@pytest.mark.parametrize("shape", [(16,), (16, 16), (16, 16, 16)])
def test_replace_non_finite(shape, np_array, boolean_np_array):
    # Test doesn't check whether nans are replaced by nearest neighbour.
    y = np_array(shape, center=0.0, half_width=1.0)
    finite = boolean_np_array(shape, seed=1).astype(bool)
    y0 = y.copy()
    y[finite == 0] = np.nan

    gpu_smoothn.replace_non_finite(y, finite)

    assert np.array_equal(y[finite], y0[finite])
    assert np.all(np.isfinite(y))


@pytest.mark.parametrize(
    "shape, allowed_noise_ratio",
    [((16,), 0.75), ((16, 16), 0.58), ((16, 16, 16), 0.69)],
)
def test_initial_guess(shape: tuple, allowed_noise_ratio):
    y = generate_cosine_field(shape, wavelength=100)
    noise = generate_noise_field(shape, scale=0.1)

    z0 = gpu_smoothn._initial_guess([y + noise])[0]
    z0_l2_norm = np.linalg.norm(z0 - y)
    noise_l2_norm = np.linalg.norm(noise)
    noise_ratio = float(z0_l2_norm / noise_l2_norm)

    assert noise_ratio < allowed_noise_ratio


@pytest.mark.parametrize(
    "shape, p",
    [
        ((16,), (-3.777, 10.193)),
        ((16, 16), (-4.079, 4.196)),
        ((16, 16, 16), (-4.255, 2.21)),
    ],
)
def test_p_bounds(shape, p):
    # Need to test different smooth orders.
    p_min, p_max = gpu_smoothn._p_bounds(shape)
    assert abs(p_min - p[0]) < 0.001
    assert abs(p_max - p[1]) < 0.001


@pytest.mark.parametrize(
    "shape, expected_lambda",
    [((16,), 3.961), ((16, 16), 7.923), ((16, 16, 16), 11.885)],
)
def test_lambda(shape, expected_lambda, np_array):
    y = np_array(shape, center=0.0, half_width=1.0)
    spacing = np.ones(y.ndim, dtype=DTYPE_f)
    lambda_ = gpu_smoothn._lambda(y, spacing)

    assert lambda_.flatten()[0] == 0
    assert abs(lambda_.flatten()[-1] - expected_lambda) < 0.001


@pytest.mark.parametrize("shape", [(16,), (16, 16), (16, 16, 16)])
@pytest.mark.parametrize("transform", [fft.dct, fft.idct])
def test_dct_nd(shape, transform, np_array):
    """Test that output arrays are C-contiguous, which is required for proper CUDA
    ndexing."""
    data = np_array(shape, center=0.0, half_width=1.0)

    data_dct_nd = gpu_smoothn._dct_nd(data, f=transform)

    assert data_dct_nd.flags.c_contiguous


@pytest.mark.parametrize(
    "shape, expected_gcv", [((16,), 13.283), ((16, 16), 4.30), ((16, 16, 16), 1.702)]
)
def test_gcv(shape, expected_gcv, np_array, boolean_np_array):
    # Need to test weighted vs non-weighted.
    p = np.log10(0.5)

    y = np_array(shape, center=0.0, half_width=1)
    y_dct = fft.dct(y)
    w = np_array(shape, seed=1)
    w_mean = np.mean(w)
    is_finite = boolean_np_array(shape, seed=2).astype(bool)
    nof = np.sum(is_finite)
    spacing = np.ones(y.ndim, dtype=DTYPE_f)
    lambda_ = gpu_smoothn._lambda(y, spacing=spacing)

    gcv_score = gpu_smoothn._gcv(p, [y], [y_dct], w, lambda_, is_finite, w_mean, nof)

    assert abs(gcv_score - expected_gcv) < 0.001


@pytest.mark.parametrize(
    "shape, expected_h", [((16,), 0.471), ((16, 16), 0.222), ((16, 16, 16), 0.105)]
)
def test_leverage(shape: tuple, expected_h):
    # Need to test different smooth orders.
    spacing = np.ones(len(shape), dtype=DTYPE_f)
    h = gpu_smoothn._leverage(0.5, spacing)

    assert abs(h - expected_h) < 0.001


@pytest.mark.parametrize(
    "shape, expected_w",
    [
        ((16,), (0.989, 0.795)),
        ((16, 16), (0.991, 0.734)),
        ((16, 16, 16), (0.991, 0.956)),
    ],
)
def test_robust_weights(shape, expected_w, np_array, boolean_np_array):
    # Need to test different weight methods.
    y = np_array(shape, center=0.0, half_width=1.0)
    z = np_array(shape, center=0.0, half_width=1.0, seed=1)
    is_finite = boolean_np_array(shape, seed=2).astype(bool)
    spacing = np.ones(y.ndim, dtype=DTYPE_i)
    h = gpu_smoothn._leverage(0.5, spacing)

    w = gpu_smoothn._robust_weights([y], [z], is_finite, h)

    assert abs(w.flatten()[0] - expected_w[0]) < 0.001
    assert abs(w.flatten()[-1] - expected_w[1]) < 0.001


@pytest.mark.parametrize("shape", [(13, 13), (14, 14), (15, 15), (16, 16)])
@pytest.mark.parametrize("direction", ["forward", "backward"])
def test_dct_order_forward(shape, direction, array_pair):
    m, n = shape
    y, y_d = array_pair(shape, center=0.0, half_width=1.0)

    z = gpu_smoothn._dct_order(y_d, direction=direction).get()
    if direction == "backward":
        y, z = z, y
    # Sequence given by Eq. 20 of Makhoul, 1980.
    y_forward_sequence_a = y[:, [2 * i for i in range(0, floor((n - 1) / 2) + 1)]]
    y_forward_sequence_b = y[
        :, [2 * n - 2 * i - 1 for i in range(floor((n + 1) / 2), n)]
    ]
    y_forward = np.hstack([y_forward_sequence_a, y_forward_sequence_b])

    assert np.array_equal(z, y_forward)


@pytest.mark.parametrize("shape", [(13, 13), (14, 14), (15, 15), (16, 16)])
@pytest.mark.parametrize("offset", [0, 1])
@pytest.mark.parametrize("left_pad", [0, 1])
def test_flip_frequency_real(shape, offset, left_pad, array_pair):
    m, n = shape
    y, y_d = array_pair(shape, center=0.0, half_width=1.0)

    flip_width = n - 1
    y_flipped = np.flip(y, axis=1)[:, offset : offset + flip_width - left_pad]
    z = gpu_smoothn._flip_frequency_real(
        y_d, flip_width=flip_width, offset=offset, left_pad=left_pad
    ).get()

    if left_pad > 0:
        assert np.all(z[:, :left_pad] == 0)
    assert np.array_equal(z[:, left_pad:], y_flipped)


@pytest.mark.parametrize("shape", [(13, 13), (14, 14), (15, 15), (16, 16)])
@pytest.mark.parametrize("offset", [0, 1])
@pytest.mark.parametrize("left_pad", [0, 1])
def test_flip_frequency_comp(shape, offset, left_pad, np_array):
    m, n = shape
    y = np_array(shape, center=0.0, half_width=1.0) + 1j * np_array(
        shape, center=0.0, half_width=1.0, seed=1
    )
    y_d = gpuarray.to_gpu(y)

    flip_width = n - 1
    y_flipped = np.flip(y, axis=1)[:, offset : offset + flip_width - left_pad].conj()
    z = gpu_smoothn._flip_frequency_comp(
        y_d, flip_width=flip_width, offset=offset, left_pad=left_pad
    ).get()

    if left_pad > 0:
        assert np.all(z[:, :left_pad] == 0)
    assert np.array_equal(z[:, left_pad:], y_flipped)


@pytest.mark.parametrize("shape", [(13, 13), (14, 14), (15, 15), (16, 16)])
def test_reflect_frequency_comp(shape, np_array):
    m, n = shape
    freq_width = n // 2 + 1
    y = np_array((m, freq_width), center=0.0, half_width=1.0) + 1j * np_array(
        (m, freq_width), center=0.0, half_width=1.0, seed=1
    )
    y_d = gpuarray.to_gpu(y)

    y_full = np.hstack([y, np.flip(y[:, 1 : n - freq_width + 1], axis=1).conj()])
    z = gpu_smoothn._reflect_frequency_comp(y_d, full_width=n).get()

    assert np.array_equal(y_full, z)


# # INTEGRATION TESTS
@pytest.mark.integtest
@pytest.mark.parametrize(
    "shape, expected_noise_ratio",
    [((64,), 0.425), ((64, 64), 0.136), ((64, 64, 64), 0.0641)],
)
def test_smoothn_noise_ratio(shape, expected_noise_ratio):
    """Check against expected noise ratio."""
    # Need to check all parameters.
    y = generate_cosine_field(shape, wavelength=50)
    noise = generate_noise_field(shape, scale=0.5)

    z = gpu_smoothn.smoothn(
        y + noise,
        mask=None,
        w=None,
        s=None,
        robust=False,
        z0=None,
        max_iter=100,
        tol_z=1e-3,
        weight_method="bisquare",
        smooth_order=2,
        spacing=None,
    )[0]
    z_l2_norm = np.linalg.norm(z - y)
    noise_l2_norm = np.linalg.norm(noise)
    noise_ratio = float(z_l2_norm / noise_l2_norm)

    assert noise_ratio < expected_noise_ratio


import pytest


@pytest.fixture
def fixt(request):
    return request.param * 3


@pytest.mark.parametrize("fixt", ["a", "b"], indirect=True)
def test_indirect(fixt):
    assert len(fixt) == 3


@pytest.fixture
def smoothn(request):
    shape = request.param
    y = generate_cosine_field(shape, wavelength=50)
    noise = generate_noise_field(shape, scale=0.5)

    def smoothn(**params):
        z, s = gpu_smoothn.smoothn(y + noise, **params)

        assert isinstance(z, np.ndarray)
        assert np.all(np.isfinite(z))
        assert isinstance(s, Number)

    return smoothn


@pytest.mark.integtest
class TestSmoothnParams:
    shape = (16, 16)

    @pytest.mark.parametrize(
        "smoothn",
        [
            shape,
        ],
        indirect=True,
    )
    @pytest.mark.parametrize("mask", [shape, False])
    def test_mask(self, mask, boolean_np_array, smoothn):
        mask = boolean_np_array(mask) if mask else None
        smoothn(mask=mask)

    @pytest.mark.parametrize(
        "smoothn",
        [
            shape,
        ],
        indirect=True,
    )
    @pytest.mark.parametrize("w", [shape, False])
    def test_w(self, w, np_array, smoothn):
        w = np_array(w, center=0.5, half_width=0.5) if w else None
        smoothn(w=w)

    @pytest.mark.parametrize(
        "smoothn",
        [
            shape,
        ],
        indirect=True,
    )
    @pytest.mark.parametrize("s", [None, 10])
    def test_s(self, s, smoothn):
        smoothn(s=s)

    @pytest.mark.parametrize(
        "smoothn",
        [
            shape,
        ],
        indirect=True,
    )
    @pytest.mark.parametrize("robust", [True, False])
    def test_robust(self, robust, smoothn):
        smoothn(robust=robust)

    @pytest.mark.parametrize(
        "smoothn",
        [
            shape,
        ],
        indirect=True,
    )
    @pytest.mark.parametrize("z0", [shape, False])
    def test_z0(self, z0, np_array, smoothn):
        z0 = np_array(z0, center=0, half_width=1.0) if z0 is None else None
        smoothn(z0=z0)

    @pytest.mark.parametrize(
        "smoothn",
        [
            shape,
        ],
        indirect=True,
    )
    @pytest.mark.parametrize("max_iter", [1, 10, 100])
    def test_max_iter(self, max_iter, smoothn):
        smoothn(max_iter=max_iter)

    @pytest.mark.parametrize(
        "smoothn",
        [
            shape,
        ],
        indirect=True,
    )
    @pytest.mark.parametrize("tol_z", [1e-3, 1e-2, 1])
    def test_tol_z(self, tol_z, smoothn):
        smoothn(tol_z=tol_z)

    @pytest.mark.parametrize(
        "smoothn",
        [
            shape,
        ],
        indirect=True,
    )
    @pytest.mark.parametrize("weight_method", ["bisquare", "talworth", "cauchy"])
    def test_weight_method(self, weight_method, smoothn):
        smoothn(weight_method=weight_method)

    @pytest.mark.parametrize(
        "smoothn",
        [
            shape,
        ],
        indirect=True,
    )
    @pytest.mark.parametrize("smooth_order", [0, 1, 2])
    def test_smooth_order(self, smooth_order, smoothn):
        smoothn(smooth_order=smooth_order)

    @pytest.mark.parametrize(
        "smoothn",
        [
            shape,
        ],
        indirect=True,
    )
    @pytest.mark.parametrize("spacing", [None, (0.5, 0.5), (1, 1)])
    def test_spacing(self, spacing, smoothn):
        smoothn(spacing=spacing)


# REGRESSION TESTS
@pytest.mark.regression
@pytest.mark.parametrize("shape", [(16,), (16, 16), (16, 16, 16)])
@pytest.mark.parametrize("s", [None, 50])
@pytest.mark.parametrize("robust", [True, False])
def test_smoothn_regression(shape, s, robust, ndarrays_regression):
    """Check that output is unchanged.

    This test will fail the first time it is run, and when the saved regression data
    is deleted.

    """
    # Need to check all parameters.
    y = generate_cosine_field(shape, wavelength=50)
    noise = generate_noise_field(shape, scale=0.5)

    z = gpu_smoothn.smoothn(y + noise, s=s, robust=robust)[0]

    ndarrays_regression.check({"z": z})
