import numpy as np
import pytest
from math import sqrt, log2

import scipy.fft as fft
from scipy.ndimage import distance_transform_edt, shift, map_coordinates
from scipy.signal import correlate2d
from skimage import img_as_ubyte
from skimage.util import random_noise
from imageio.v2 import imread
import pycuda.gpuarray as gpuarray

# noinspection PyUnresolvedReferences
import pycuda.autoinit

import openpiv.gpu_process as gpu_process

# GLOBAL VARIABLES
DTYPE_i = np.int32
DTYPE_f = np.float32

# dirs
data_dir = "../data/"

_image_size_rectangle = (1024, 1024)
_image_size_square = (1024, 512)
_u_shift = 8
_v_shift = -4
_accuracy_tolerance = 0.1
_identity_tolerance = 1e-6
_trim_slice = slice(2, -2, 1)


# UTILS
def create_pair_shift(image_size, u_shift, v_shift):
    """Creates a pair of images with a roll/shift"""
    frame_a = np.zeros(image_size, dtype=np.int32)
    frame_a = random_noise(frame_a)
    frame_a = img_as_ubyte(frame_a)
    frame_b = shift(frame_a, (v_shift, u_shift), mode="wrap")

    return frame_a.astype(np.int32), frame_b.astype(np.int32)


def create_pair_roll(image_size, roll_shift):
    """Creates a pair of images with a roll/shift"""
    frame_a = np.zeros(image_size, dtype=np.int32)
    frame_a = random_noise(frame_a)
    frame_a = img_as_ubyte(frame_a)
    frame_b = np.roll(frame_a, roll_shift)

    return frame_a.astype(np.int32), frame_b.astype(np.int32)


def generate_cpu_gpu_pair(size, magnitude=1, dtype=DTYPE_f):
    """Returns a pair of cpu and gpu arrays with random values."""
    np.random.seed(0)
    cpu_array = (np.random.random(size) * magnitude).astype(dtype)
    _gpu_array = gpuarray.to_gpu(cpu_array)

    return cpu_array, _gpu_array


def nearest_neighbour_interp(f, mask, spacing=1):
    nearest_neighbour = distance_transform_edt(
        mask, sampling=spacing, return_distances=False, return_indices=True
    )
    if f.ndim == 1:
        neighbour_index = nearest_neighbour.squeeze()
    else:
        neighbour_index = tuple(
            nearest_neighbour[i] for i in range(nearest_neighbour.shape[0])
        )
    f[mask] = f[neighbour_index][mask]


def interp_mask_np(x0, y0, x1, y1, f0, mask):
    # Get the interpolating coordinate
    ht, wd = mask.shape
    m = y1.size
    buffer_x = x0[0]
    buffer_y = y0[0]
    spacing_x = x0[1] - buffer_x
    spacing_y = y0[1] - buffer_y
    mask = mask.astype(bool)

    x = ((x1 - buffer_x) / spacing_x).astype(DTYPE_f)
    y = ((y1 - buffer_y) / spacing_y).astype(DTYPE_f)

    # Coerce interpolation point to within limits of domain.
    x = x * ((x >= 0.0) & (x <= wd - 1)) + (wd - 1) * (x > wd - 1)
    y = y * ((y >= 0.0) & (y <= ht - 1)) + (ht - 1) * (y > ht - 1)

    # Get the upper and lower bounds
    x1 = (np.floor(x) - (x == wd - 1)).astype(np.int32)
    x2 = x1 + 1
    y1 = (np.floor(y) - (y == ht - 1)).astype(np.int32)
    y2 = y1 + 1

    # Get masked values.
    m11 = mask[np.ix_(y1, x1)]
    m21 = mask[np.ix_(y1, x2)]
    m12 = mask[np.ix_(y2, x1)]
    m22 = mask[np.ix_(y2, x2)]
    m_y1 = m11 & m21
    m_y2 = m12 & m22

    # Do interpolation along x
    f11 = ((x2 - x) * (~m11 & ~m21) + (~m11 & m21)) * f0[np.ix_(y1, x1)]
    f21 = ((x - x1) * (~m11 & ~m21) + (m11 & ~m21)) * f0[np.ix_(y1, x2)]
    f12 = ((x2 - x) * (~m12 & ~m22) + (~m12 & m22)) * f0[np.ix_(y2, x1)]
    f22 = ((x - x1) * (~m12 & ~m22) + (m12 & ~m22)) * f0[np.ix_(y2, x2)]

    # Do interpolation along y
    f_y1 = ((y2 - y).reshape((m, 1)) * (~m_y1 & ~m_y2) + (~m_y1 & m_y2)) * (f11 + f21)
    f_y2 = ((y - y1).reshape((m, 1)) * (~m_y1 & ~m_y2) + (m_y1 & ~m_y2)) * (f12 + f22)
    f1 = f_y1 + f_y2

    return f1


def grid_coords(shape, window_size, spacing):
    x, y = gpu_process.field_coords(shape, window_size, spacing)
    x = x[0, :].astype(DTYPE_f)
    y = y[:, 0].astype(DTYPE_f)

    return x, y


def gpu_arrays(*f_l):
    return [gpuarray.to_gpu(f) for f in f_l]


def np_arrays(*f_l):
    return [f.get() for f in f_l]


def cross_correlate_np(a, b):
    fft_product = fft.fft2(a).conj() * fft.fft2(b)
    return fft.ifft(fft_product).real


def window_slice_np(frame, field_shape, window_size, spacing, buffer):
    m, n = field_shape
    ws = window_size

    win_np = np.zeros((m * n, ws, ws), dtype=DTYPE_f)
    for i in range(m):
        row0 = buffer + spacing * i
        row1 = buffer + spacing * i + ws
        # Get the beginning and end indexes that are inside the windows.
        row_inside0 = max(0, row0)
        row_inside1 = min(m, row1)

        frame_row_idx = slice(row0, row1)
        win_row_idx = slice(row_inside0 - row0, ws - (row_inside1 - row1))

        for j in range(n):
            col0 = buffer + spacing * j
            col1 = buffer + spacing * j + ws
            col_inside0 = max(0, col0)
            col_inside1 = min(n, col1)

            frame_col_idx = slice(col0, col1)
            win_col_idx = slice(col_inside0 - col0, ws - (col_inside1 - col1))

            win_np[i * n + j, win_row_idx, win_col_idx] = frame[
                frame_row_idx, frame_col_idx
            ]

    return win_np


def gaussian_peak(shape, row_peak, col_peak):
    n_windows, ht, wd = shape
    x, y = np.meshgrid(np.arange(wd), np.arange(ht))
    x0 = col_peak
    y0 = row_peak
    a, b = (1, 1)

    correlation = np.empty(shape, dtype=DTYPE_f)
    for i in range(n_windows):
        correlation[i] = np.exp(
            -((x - x0[i]) ** 2 / (2 * a**2) + (y - y0[i]) ** 2 / (2 * b**2))
        )

    return gpuarray.to_gpu(correlation)


def parabolic_peak(shape, row_peak, col_peak):
    n_windows, ht, wd = shape
    x, y = np.meshgrid(np.arange(wd), np.arange(ht))
    x0 = col_peak
    y0 = row_peak
    a, b = (1, 1)

    correlation = np.empty(shape, dtype=DTYPE_f)
    for i in range(n_windows):
        correlation[i] = (x - x0[i]) ** 2 / (2 * a**2) + (y - y0[i]) ** 2 / (
            2 * b**2
        )

    return gpuarray.to_gpu(correlation)


def mask_peak_np(correlation, row_peak, col_peak, width):
    n_windows, ht, wd = correlation.shape

    correlation_masked = correlation.copy()

    for i in range(-width, width + 1):
        for j in range(-width, width + 1):
            row = row_peak + i
            row[row < 0] = 0
            row[row > ht - 1] = ht - 1
            col = col_peak + j
            col[col < 0] = 0
            col[col > wd - 1] = wd - 1
            correlation_masked[
                np.arange(
                    n_windows,
                ),
                row,
                col,
            ] = 0

    return correlation_masked


def l2_norm_2d(a, b):
    return np.sqrt(np.sum(a**2 + b**2)) / sqrt(a.size)


# UNIT TESTS
def test_correlation_gpu_free_frame_data(correlation_gpu):
    correlation_gpu.free_frame_data()

    assert correlation_gpu.frame_a is None
    assert correlation_gpu.frame_b is None


def test_correlation_gpu_signal_to_noise(correlation_gpu, piv_field_gpu):
    assert isinstance(correlation_gpu.sig2noise, gpuarray.GPUArray)


def test_correlation_gpu_init_fft_shape(correlation_gpu, piv_field_gpu):
    fft_shape = correlation_gpu.fft_shape

    assert round(log2(fft_shape[0])) == log2(fft_shape[0])
    assert round(log2(fft_shape[1])) == log2(fft_shape[1])


def test_correlation_gpu_stack_iw():
    # TODO test that buffers are computed properly
    # TODO fold stack_iw with correlate_windows into another method
    pass


def test_correlation_gpu_correlate_windows(correlation_gpu, piv_field_gpu):
    shape = (32, 32)
    fft_shape = correlation_gpu.fft_shape[0]
    a = 1
    c = 4

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    r = (x - (shape[1] - 1) / 2) ** 2 + (y - (shape[0] - 1) / 2) ** 2
    win = a * np.exp(-r / (2 * c**2))
    win_d = gpuarray.to_gpu(win.reshape(1, *shape).astype(DTYPE_f))
    corr = correlation_gpu._correlate_windows(win_d, win_d).get()
    peak_idx = np.argmax(corr)

    assert peak_idx == (fft_shape / 2) * (fft_shape + 1)


def test_correlation_gpu_check_zero_correlation(correlation_gpu, piv_field_gpu):
    correlation_gpu(piv_field_gpu)

    corr_peak = correlation_gpu.corr_peak1.get()
    corr_peak[:] = 0
    correlation_gpu.corr_peak1 = gpuarray.to_gpu(corr_peak)
    correlation_gpu._check_zero_correlation()

    assert np.all(correlation_gpu.row_peak.get() == 32)
    assert np.all(correlation_gpu.col_peak.get() == 32)


def test_correlation_gpu_get_displacement(correlation_gpu):
    fft_shape = correlation_gpu.fft_shape[0]
    field_shape = correlation_gpu.piv_field.shape
    field_size = correlation_gpu.piv_field.size

    row_sp = np.arange(field_size).reshape(field_shape)
    col_sp = np.arange(field_size).reshape(field_shape)
    i_peak, j_peak = correlation_gpu._get_displacement(row_sp, col_sp)

    assert np.array_equal(i_peak, col_sp - fft_shape // 2)
    assert np.array_equal(j_peak, row_sp - fft_shape // 2)


@pytest.mark.parametrize("s2n_method", ["peak2mean", "peak2energy", "peak2peak"])
def test_correlation_gpu_get_s2n(correlation_gpu, s2n_method):
    correlation_gpu.s2n_method = s2n_method
    s2n = correlation_gpu._get_s2n()

    assert isinstance(s2n, gpuarray.GPUArray)


def test_correlation_gpu_get_second_peak(correlation_gpu):
    shape = (32, 32)
    fft_shape = correlation_gpu.fft_shape[0]
    a = 1
    c = 4

    correlation_gpu.row_peak = gpuarray.to_gpu(
        np.array(fft_shape // 2)
        .astype(DTYPE_i)
        .reshape(
            1,
        )
    )
    correlation_gpu.col_peak = gpuarray.to_gpu(
        np.array(fft_shape // 2)
        .astype(DTYPE_i)
        .reshape(
            1,
        )
    )

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    r = (x - (shape[1] - 1) / 2) ** 2 + (y - (shape[0] - 1) / 2) ** 2
    win = a * np.exp(-r / (2 * c**2))
    win_d = gpuarray.to_gpu(np.expand_dims(win, 0).astype(DTYPE_f))
    corr_d = correlation_gpu._correlate_windows(win_d, win_d)
    corr_max2 = correlation_gpu._get_second_peak(corr_d, 1).get()

    assert corr_max2 < np.amax(corr_d.get())


def test_piv_field_gpu_get_mask(piv_field_gpu):
    mask = piv_field_gpu.get_mask(return_array=True)

    assert isinstance(mask, gpuarray.GPUArray)


def test_piv_field_gpu_free_data(piv_field_gpu):
    piv_field_gpu.free_data()

    assert piv_field_gpu._mask is None


def test_piv_field_gpu_coords(piv_field_gpu):
    x, y = piv_field_gpu.coords

    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)


def test_piv_field_gpu_grid_coords(piv_field_gpu):
    x_grid, y_grid = piv_field_gpu.coords

    assert isinstance(x_grid, np.ndarray)
    assert isinstance(y_grid, np.ndarray)


def test_piv_field_gpu_center_buffer(piv_field_gpu):
    buffer_x, buffer_y = piv_field_gpu.center_buffer

    assert isinstance(buffer_x, int)
    assert isinstance(buffer_y, int)


def test_piv_gpu_coords(piv_gpu):
    x, y = piv_gpu.coords

    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)


def test_piv_gpu_field_mask(piv_gpu):
    mask = piv_gpu.field_mask

    assert isinstance(mask, np.ndarray)


def test_piv_gpu_s2n(piv_gpu):
    sig2noise = piv_gpu.sig2noise

    assert isinstance(sig2noise, gpuarray.GPUArray)


def test_piv_gpu_free_data(piv_gpu):
    piv_gpu.free_data()

    assert piv_gpu._corr is None
    assert piv_gpu._piv_fields is None
    assert piv_gpu._frame_mask is None


def test_piv_gpu_get_piv_fields(piv_gpu):
    sig2noise = piv_gpu._get_piv_fields()

    assert isinstance(sig2noise, list)


def test_piv_gpu_mask_frame(piv_gpu):
    # Need to test the switch cases.
    frame_a = imread(data_dir + "test1/exp1_001_a.bmp").astype(np.float32)
    frame_b = imread(data_dir + "test1/exp1_001_b.bmp").astype(np.float32)
    frame_a_masked, frame_b_masked = piv_gpu._mask_frame(frame_a, frame_b)

    assert isinstance(frame_a_masked, gpuarray.GPUArray)
    assert isinstance(frame_a_masked, gpuarray.GPUArray)


def test_piv_gpu_get_frame_mask(piv_gpu, np_array):
    shape = (16, 16)

    piv_gpu.mask = np_array(shape, center=0.5, half_width=0.5, d_type=DTYPE_i)
    piv_gpu._frame_mask = None
    frame_mask = piv_gpu._get_frame_mask()

    assert isinstance(frame_mask, gpuarray.GPUArray)


def test_piv_gpu_get_search_size(piv_gpu):
    piv_gpu._k = 0
    piv_gpu._piv_field_k.window_size = 8
    piv_gpu.extend_ratio = 2
    search_size = piv_gpu._get_search_size()

    assert search_size == piv_gpu._piv_field_k.window_size * piv_gpu.extend_ratio


def test_piv_gpu_get_window_deformation(piv_gpu, gpu_array):
    shape = (16, 16)

    dp_u = gpu_array(shape, center=0.0, half_width=1.0)
    mask = gpu_array(shape, d_type=DTYPE_i)
    piv_gpu._piv_field_k._mask = mask
    shift_, strain = piv_gpu._get_window_deformation(dp_u, dp_u)

    assert isinstance(shift_, gpuarray.GPUArray)
    assert isinstance(strain, gpuarray.GPUArray)


def test_piv_gpu_update_values(piv_gpu, gpu_array):
    # Need to test the switch cases.
    shape = (16, 16)

    i_peak = gpu_array(shape, center=0.0, half_width=1.0)
    mask = gpu_array(shape, d_type=DTYPE_i)
    piv_gpu._piv_field_k._mask = mask
    u, v = piv_gpu._update_values(i_peak, i_peak, i_peak, i_peak)

    assert isinstance(u, gpuarray.GPUArray)
    assert isinstance(v, gpuarray.GPUArray)


def test_piv_gpu_validate_fields():
    # TODO Ensure that the validation is executed each loop
    pass


@pytest.mark.parametrize("k", [0, 1])
def test_piv_gpu_gpu_replace_vectors(k, piv_gpu, gpu_array):
    # Need to test different dimensions.
    shape = (16, 16)

    u = gpu_array(shape, center=0.0, half_width=1.0)
    mask = gpu_array(shape, d_type=DTYPE_i)
    val_locations = gpu_array(shape, d_type=DTYPE_i, seed=1)
    piv_gpu._piv_field_k._mask = mask
    u, v = piv_gpu._gpu_replace_vectors(u, u, u, u, u, u, val_locations)

    assert isinstance(u, gpuarray.GPUArray)
    assert isinstance(v, gpuarray.GPUArray)


def test_piv_gpu_get_next_iteration_predictions(piv_gpu, gpu_array):
    # Need to test different dimensions.
    shape = (22, 30)

    u = v = gpu_array(shape, center=0.0, half_width=1.0)
    mask = gpu_array(shape, d_type=DTYPE_i)
    piv_gpu._piv_field_k._mask = mask
    piv_gpu._piv_field_k = piv_gpu._get_piv_fields()[0]
    piv_gpu._k = 0
    dp_u, dp_v = piv_gpu._get_next_iteration_predictions(u, v)

    assert isinstance(dp_u, gpuarray.GPUArray)
    assert isinstance(dp_v, gpuarray.GPUArray)


def test_piv_gpu_get_residual(piv_gpu, array_pair):
    shape = (16, 16)

    i_peak, i_peak_d = array_pair(shape, center=0.0, half_width=1.0)
    residual = piv_gpu._get_residual(i_peak_d, i_peak_d)

    assert residual == sqrt(int(np.sum(i_peak**2 + i_peak**2)) / i_peak.size) / 0.5


@pytest.mark.parametrize("frame_shape", [(64, 64), (63, 63)])
@pytest.mark.parametrize("window_size", [16])
@pytest.mark.parametrize("spacing", [8, 7])
def test_get_field_shape(frame_shape: tuple, window_size, spacing):
    ht, wd = frame_shape
    m, n = gpu_process.field_shape(frame_shape, window_size, spacing)

    assert m == int((ht - window_size) // spacing) + 1
    assert n == int((wd - window_size) // spacing) + 1


# @pytest.mark.parametrize("frame_shape", [(64, 64), (63, 63)])
# @pytest.mark.parametrize("window_size", [16, 15])
# @pytest.mark.parametrize("spacing", [8, 7])
# @pytest.mark.parametrize("center_field", [True, False])
@pytest.mark.parametrize("frame_shape", [(63, 63)])
@pytest.mark.parametrize("window_size", [16])
@pytest.mark.parametrize("spacing", [8])
@pytest.mark.parametrize("center_field", [False])
def test_get_field_coords(frame_shape: tuple, window_size, spacing, center_field):
    half_width = window_size // 2
    buffer_x = 0
    buffer_y = 0
    if center_field:
        buffer_x, buffer_y = gpu_process._center_buffer(
            frame_shape, window_size, spacing
        )
    x, y = gpu_process.field_coords(frame_shape, window_size, spacing, center_field)
    m, n = x.shape

    assert x[0, 0] == half_width + buffer_x
    assert x[0, -1] == half_width + buffer_x + spacing * (n - 1)
    assert y[0, 0] == half_width + buffer_y + spacing * (m - 1)
    assert y[-1, 0] == half_width + buffer_y


def test_gpu_strain(array_pair):
    shape = (16, 16)

    u, u_d = array_pair(shape, center=0.0, half_width=1.0)
    v, v_d = array_pair(shape, center=0.0, half_width=1.0, seed=1)

    u_y, u_x = np.gradient(u)
    v_y, v_x = np.gradient(v)
    strain_gpu = (gpu_process.gpu_strain(u_d, v_d)).get()

    assert np.array_equal(u_x, strain_gpu[0])
    assert np.array_equal(u_y, strain_gpu[1])
    assert np.array_equal(v_x, strain_gpu[2])
    assert np.array_equal(v_y, strain_gpu[3])


@pytest.mark.parametrize("shape", [(16, 16, 16), (15, 12, 14), (15, 11, 13)])
def test_gpu_fft_shift(shape, array_pair):
    shape = (16, 16, 16)

    correlation, correlation_d = array_pair(shape, center=0.0, half_width=1.0)

    correlation_shifted_np = fft.fftshift(correlation, axes=(1, 2))
    correlation_shifted_gpu = gpu_process.gpu_fft_shift(correlation_d).get()

    assert np.array_equal(correlation_shifted_gpu, correlation_shifted_np)


@pytest.mark.parametrize("spacing", [(1, 2), (2, 3)])
def test_gpu_interpolate_mask(spacing, array_pair, boolean_array_pair):
    shape = (16, 16)
    window_size0 = 4
    window_size1 = 2
    spacing0, spacing1 = spacing
    f0_shape = gpu_process.field_shape(shape, window_size0, spacing0)

    f0, f0_d = array_pair(f0_shape, center=0.0, half_width=1.0)
    mask, mask_d = boolean_array_pair(f0_shape, seed=1)

    x0, y0 = grid_coords(shape, window_size0, spacing0)
    x1, y1 = grid_coords(shape, window_size1, spacing1)
    x0_d, y0_d = gpu_arrays(x0, y0)
    x1_d, y1_d = gpu_arrays(x1, y1)

    f1 = interp_mask_np(x0, y0, x1, y1, f0, mask)
    f1_gpu = gpu_process.gpu_interpolate(
        x0_d, y0_d, x1_d, y1_d, f0_d, mask=mask_d
    ).get()

    assert np.allclose(f1_gpu, f1)


def test_window_sizes():
    ws_iters = (2, 1, 3)
    min_window_size = 8
    window_size_l = list(gpu_process._window_sizes(ws_iters, min_window_size))

    assert window_size_l == [32, 32, 16, 8, 8, 8]


@pytest.mark.parametrize("overlap_ratio, spacing", [(0.01, 7), (0.5, 4), (0.99, 1)])
def test_spacing(overlap_ratio, spacing):
    window_size = 8

    assert spacing == gpu_process._spacing(window_size, overlap_ratio)


def test_field_mask():
    ht, wd = 32, 32
    w = np.pi / 2

    x0, y0 = np.meshgrid(np.arange(ht), np.arange(wd))
    x1, y1 = np.meshgrid(np.arange(0, ht, 2), np.arange(0, wd, 2))
    frame_mask = np.round((1 + np.cos(x0 * w) * np.cos(y0 * w)) / 2).astype(int)
    field_mask0 = np.round((1 + np.cos(x1 * w) * np.cos(y1 * w)) / 2).astype(int)
    field_mask1 = gpu_process._field_mask(x1, y1, frame_mask)

    assert np.array_equal(field_mask1, field_mask0)


@pytest.mark.parametrize(
    "frame_size, window_size, spacing, buffer",
    [(64, 8, 4, 0), (66, 8, 4, 1), (66, 7, 4, 2)],
)
def test_center_buffer(frame_size, window_size, spacing, buffer):
    buffer_x, buffer_y = gpu_process._center_buffer(
        (frame_size, frame_size), window_size, spacing
    )

    assert buffer_x == buffer


@pytest.mark.parametrize("window_size", [4, 8])
@pytest.mark.parametrize("spacing", [4, 8])
@pytest.mark.parametrize("buffer", [-1, 1])
@pytest.mark.parametrize("pass_shift", [True, False])
def test_gpu_window_slice(window_size, spacing, buffer, pass_shift, array_pair):
    frame_shape = (16, 16)
    window_size = 8
    spacing = 4
    field_shape = gpu_process.field_shape(frame_shape, window_size, spacing)
    buffer = 0

    frame, frame_d = array_pair(frame_shape, center=0.0, half_width=1.0)
    shift_d = (
        gpuarray.to_gpu(np.ones((2, *field_shape), dtype=DTYPE_f))
        if pass_shift
        else None
    )

    win_np = window_slice_np(frame, field_shape, window_size, spacing, buffer)
    win_gpu = gpu_process._gpu_window_slice(
        frame_d, field_shape, window_size, spacing, buffer, shift=shift_d
    ).get()

    assert np.array_equal(win_gpu, win_np)


@pytest.mark.parametrize("dt", [-1, 0, 1])
def test_gpu_window_slice_shift(dt, array_pair):
    """Test window translation using a single window centered on the frame."""
    frame_shape = (16, 16)
    field_shape = (1, 1)
    ht, wd = frame_shape
    ws = 16
    spacing = 4
    buffer = 0
    u = 1.4
    v = 2.5

    frame, frame_d = array_pair(frame_shape, center=0.0, half_width=1.0)

    shift_d = gpuarray.to_gpu(np.array([u, v], dtype=DTYPE_f).reshape(2, *field_shape))
    # Apply the strain shift directly to the frame.
    win_np = np.empty((1, ws, ws), dtype=DTYPE_f)
    x, y = np.meshgrid(np.arange(ht), np.arange(wd))
    x = x + u * dt
    y = y + v * dt
    coordinates = [y, x]
    win_np[:, :, :] = map_coordinates(frame[:, :], coordinates, order=1)
    win_gpu = gpu_process._gpu_window_slice(
        frame_d, field_shape, ws, spacing, buffer, dt=dt, shift=shift_d
    ).get()

    assert np.allclose(win_gpu, win_np, atol=1e-6)


@pytest.mark.parametrize("dt", [-1, 0, 1])
def test_gpu_window_slice_strain(dt, array_pair):
    """Test window deformation using a single window centered on the frame."""
    frame_shape = (16, 16)
    field_shape = (1, 1)
    ht, wd = frame_shape
    ws = 16
    spacing = 4
    buffer = 0
    u_x = 0.1
    u_y = 0.2
    v_x = 0.3
    v_y = 0.4

    frame, frame_d = array_pair(frame_shape, center=0.0, half_width=1.0)

    shift_d = gpuarray.zeros((2, *field_shape), dtype=DTYPE_f)
    strain_d = gpuarray.to_gpu(
        np.array([u_x, u_y, v_x, v_y], dtype=DTYPE_f).reshape(4, *field_shape)
    )
    # Apply the strain deformation directly to the frame.
    win_np = np.empty((1, ws, ws), dtype=DTYPE_f)
    x, y = np.meshgrid(np.arange(ht), np.arange(wd))
    r_x = (np.arange(wd) - wd / 2 + 0.5).reshape((1, wd))
    r_y = (np.arange(ht) - ht / 2 + 0.5).reshape((ht, 1))
    x = x + (u_x * r_x + u_y * r_y) * dt
    y = y + (v_x * r_x + v_y * r_y) * dt
    coordinates = [y, x]
    win_np[:, :, :] = map_coordinates(frame[:, :], coordinates, order=1)
    win_gpu = gpu_process._gpu_window_slice(
        frame_d, field_shape, ws, spacing, buffer, dt=dt, shift=shift_d, strain=strain_d
    ).get()

    assert np.allclose(win_gpu, win_np, atol=1e-6)


def test_gpu_normalize_intensity(array_pair):
    shape = (16, 16, 16)
    n_windows, ht, wd = shape

    win, win_d = array_pair(shape, center=0.0, half_width=1.0)

    mean = np.mean(win.reshape((n_windows, ht * wd)), axis=1).reshape((n_windows, 1, 1))
    win_norm_np = win - mean
    win_norm_gpu = gpu_process._gpu_normalize_intensity(win_d).get()

    assert np.allclose(win_norm_gpu, win_norm_np)


@pytest.mark.parametrize("offset", [0, 1, 2])
def test_gpu_zero_pad(offset, array_pair):
    shape = (16, 16, 16)
    fft_shape = (32, 32)
    n_windows, ht, wd = shape
    fft_ht, fft_wd = fft_shape

    win, win_d = array_pair(shape, center=0.0, half_width=1.0)

    win_zp_np = np.zeros((n_windows, fft_ht, fft_wd), dtype=DTYPE_f)
    win_zp_np[:, offset : offset + ht, offset : offset + wd] = win
    win_zp_gpu = gpu_process._gpu_zero_pad(win_d, fft_shape, offset).get()

    assert np.array_equal(win_zp_gpu, win_zp_np)


@pytest.mark.parametrize("shape", [(16, 16, 16), (15, 12, 14), (15, 11, 13)])
def test_cross_correlate(shape: tuple, array_pair):
    _, m, n = shape

    win_a, win_a_d = array_pair(shape, center=0.0, half_width=1.0)
    win_b, win_b_d = array_pair(shape, center=0.0, half_width=1.0, seed=1)

    correlation = np.empty(shape, dtype=DTYPE_f)
    for i in range(shape[0]):
        # The scipy definition of the cross-correlation reverses the output.
        correlation[i, :, :] = correlate2d(
            win_b[i, :, :], win_a[i, :, :], mode="full", boundary="wrap"
        )[m - 1 :, n - 1 :]
    correlation_np = correlation
    correlation_d = gpu_process._gpu_cross_correlate(win_a_d, win_b_d)
    correlation_gpu = correlation_d.get()

    assert np.allclose(correlation_gpu, correlation_np, atol=1e-5)


def test_gpu_window_index_f(array_pair):
    n_windows = 16
    index_size = 16 * 16
    shape = (n_windows, index_size)

    data, data_d = array_pair(shape, center=0.0, half_width=1.0)
    indices, indices_d = array_pair(
        n_windows,
        center=index_size / 2,
        half_width=index_size / 2,
        d_type=DTYPE_i,
        seed=1,
    )

    values_np = data[np.arange(n_windows), indices].squeeze()
    values_gpu = gpu_process._gpu_window_index_f(data_d, indices_d).get()

    assert np.array_equal(values_gpu, values_np)


def test_find_peak(array_pair):
    shape = (16, 16, 16)
    n_windows, ht, wd = shape

    correlation, correlation_d = array_pair(shape, center=0.0, half_width=1.0)

    peak_idx_np = np.argmax(correlation.reshape(n_windows, ht * wd), axis=1).astype(
        dtype=DTYPE_i
    )
    peak_idx_gpu = gpu_process._find_peak(correlation_d).get()

    assert np.array_equal(peak_idx_gpu, peak_idx_np)


def test_get_peak(array_pair):
    shape = (16, 16, 16)
    n_windows, ht, wd = shape

    correlation, correlation_d = array_pair(shape, center=0.0, half_width=1.0)
    peak_idx, peak_idx_d = array_pair(
        (n_windows,), center=ht * wd / 2, half_width=ht * wd / 2, d_type=DTYPE_i, seed=1
    )

    corr_peak_np = correlation.reshape(n_windows, ht * wd)[
        np.arange(n_windows), peak_idx
    ]
    corr_peak_gpu = gpu_process._get_peak(correlation_d, peak_idx_d).get()

    assert np.array_equal(corr_peak_gpu, corr_peak_np)


@pytest.mark.parametrize(
    "method, tol", [("gaussian", 0.1), ("parabolic", 0.1), ("centroid", 0.25)]
)
def test_gpu_subpixel_approximation(method, tol, np_array):
    shape = (16, 16, 16)
    n_windows, ht, wd = shape

    row_sp = np_array(n_windows, center=ht / 2, half_width=ht / 2 - 1)
    col_sp = np_array(n_windows, center=wd / 2, half_width=wd / 2 - 1, seed=1)
    row_peak_d, col_peak_d = gpu_arrays(
        np.round(row_sp).astype(DTYPE_i), np.round(col_sp).astype(DTYPE_i)
    )

    # Create the distribution.
    if method == "parabolic":
        correlation_d = parabolic_peak(shape, row_sp, col_sp)
    else:
        correlation_d = gaussian_peak(shape, row_sp, col_sp)

    row_sp_d, col_sp_d = gpu_process._gpu_subpixel_approximation(
        correlation_d, row_peak_d, col_peak_d, method
    )
    row_sp_gpu, col_sp_gpu = np_arrays(row_sp_d, col_sp_d)

    assert np.all(np.abs(row_sp_gpu - row_sp) <= tol)
    assert np.all(np.abs(col_sp_gpu - col_sp) <= tol)


def test_peak2mean(array_pair):
    shape = (16, 16, 16)
    n_windows, ht, wd = shape
    size = ht * wd

    correlation, correlation_d = array_pair(shape)
    corr_peak, corr_peak_d = array_pair((n_windows,), seed=1)

    correlation_masked = correlation * (
        correlation < corr_peak.reshape(n_windows, 1, 1) / 2
    )
    sig2noise_np = 2 * np.log10(
        corr_peak / np.mean(correlation_masked.reshape(n_windows, size), axis=1)
    )
    sig2noise_gpu = gpu_process._peak2mean(correlation_d, corr_peak_d).get()

    assert np.allclose(sig2noise_gpu, sig2noise_np)


def test_peak2energy(array_pair):
    shape = (16, 16, 16)
    n_windows, ht, wd = shape
    size = ht * wd

    correlation, correlation_d = array_pair(shape)
    corr_peak, corr_peak_d = array_pair((n_windows,), seed=1)

    sig2noise_np = 2 * np.log10(
        corr_peak / np.mean(correlation.reshape(n_windows, size), axis=1)
    )
    sig2noise_gpu = gpu_process._peak2energy(correlation_d, corr_peak_d).get()

    assert np.allclose(sig2noise_gpu, sig2noise_np)


def test_peak2peak(array_pair):
    shape = (16, 16, 16)
    n_windows, ht, wd = shape

    corr_peak1, corr_peak1_d = array_pair((n_windows,))
    corr_peak2, corr_peak2_d = array_pair((n_windows,), seed=1)

    sig2noise_np = np.log10(corr_peak1 / corr_peak2)
    sig2noise_gpu = gpu_process._peak2peak(corr_peak1_d, corr_peak2_d).get()

    assert np.allclose(sig2noise_gpu, sig2noise_np)


@pytest.mark.parametrize("width", [1, 2])
def test_gpu_mask_peak(width, array_pair):
    shape = (16, 16, 16)
    n_windows, ht, wd = shape

    correlation, correlation_d = array_pair(shape)
    row_peak, row_peak_d = array_pair(
        (n_windows,), center=ht / 2, half_width=ht / 2, d_type=DTYPE_i, seed=1
    )
    col_peak, col_peak_d = array_pair(
        (n_windows,), center=wd / 2, half_width=wd / 2, d_type=DTYPE_i, seed=2
    )

    correlation_masked_np = mask_peak_np(correlation, row_peak, col_peak, width)
    correlation_masked_gpu = gpu_process._gpu_mask_peak(
        correlation_d, row_peak_d, col_peak_d, width
    ).get()

    assert np.array_equal(correlation_masked_gpu, correlation_masked_np)


def test_gpu_mask_rms(array_pair):
    shape = (16, 16, 16)
    n_windows, ht, wd = shape

    correlation, correlation_d = array_pair(shape)
    corr_peak, corr_peak_d = array_pair((n_windows,), seed=1)

    correlation_masked_np = correlation * (
        correlation < corr_peak.reshape(n_windows, 1, 1) / 2
    )
    correlation_masked_gpu = gpu_process._gpu_mask_rms(correlation_d, corr_peak_d).get()

    assert np.array_equal(correlation_masked_gpu, correlation_masked_np)


def test_field_shift(array_pair):
    shape = (16, 16)

    u, u_d = array_pair(shape, center=0.0, half_width=1.0)
    v, v_d = array_pair(shape, center=0.0, half_width=1.0, seed=1)

    shift_np = np.stack([u, v], axis=0)
    shift_gpu = gpu_process._field_shift(u_d, v_d).get()

    assert np.array_equal(shift_gpu, shift_np)


def test_gpu_update_field(array_pair, boolean_array_pair):
    shape = (16, 16)

    dp, dp_d = array_pair(shape, center=0.0, half_width=1.0)
    peak, peak_d = array_pair(shape, center=0.0, half_width=1.0, seed=1)
    mask, mask_d = boolean_array_pair(shape, seed=2)

    f_np = (dp + peak) * (mask == 0)
    f_gpu = gpu_process._gpu_update_field(dp_d, peak_d, mask_d).get()

    assert np.array_equal(f_np, f_gpu)


def test_interpolate_replace(array_pair, boolean_array_pair):
    shape = (16, 16)
    window_size0 = 4
    spacing0 = 2
    window_size1 = 2
    spacing1 = 1
    f0_shape = gpu_process.field_shape(shape, window_size0, spacing0)
    f1_shape = gpu_process.field_shape(shape, window_size1, spacing1)

    f0, f0_d = array_pair(f0_shape, center=0.0, half_width=1.0)
    f1, f1_d = array_pair(f1_shape, center=0.0, half_width=1.0)
    mask, mask_d = boolean_array_pair(f0_shape, seed=1)
    val_locations, val_locations_d = boolean_array_pair(f1_shape, seed=2)

    x0, y0 = grid_coords(shape, window_size0, spacing0)
    x1, y1 = grid_coords(shape, window_size1, spacing1)
    x0_d, y0_d = gpu_arrays(x0, y0)
    x1_d, y1_d = gpu_arrays(x1, y1)

    f1 = interp_mask_np(x0, y0, x1, y1, f0, mask) * val_locations + f1 * (
        val_locations == 0
    )
    f1_gpu = gpu_process._interpolate_replace(
        x0_d, y0_d, x1_d, y1_d, f0_d, f1_d, val_locations_d, mask=mask_d
    ).get()

    assert np.allclose(f1_gpu, f1)


# BENCHMARKS
@pytest.mark.parametrize("image_size", [(1024, 1024), (2048, 2048)])
@pytest.mark.parametrize(
    "window_size_iters,min_window_size", [((1, 2), 16), ((1, 2, 2), 8)]
)
def test_gpu_piv_benchmark(benchmark, image_size, window_size_iters, min_window_size):
    """Benchmarks the PIV function."""
    frame_a, frame_b = create_pair_shift(image_size, _u_shift, _v_shift)
    args = {
        "mask": None,
        "window_size_iters": window_size_iters,
        "min_window_size": min_window_size,
        "overlap_ratio": 0.5,
        "dt": 1,
        "deform": True,
        "smooth": True,
        "num_validation_iters": 2,
        "validation_method": "median_velocity",
    }

    benchmark(gpu_process.gpu_piv, frame_a, frame_b, **args)


# INTEGRATION TESTS
# TODO check convenience function for various inputs
def gpu_piv():
    pass


@pytest.mark.parametrize("image_size", (_image_size_rectangle, _image_size_square))
def test_gpu_piv_fast(image_size):
    """Quick test of the main piv function."""
    frame_a, frame_b = create_pair_shift(image_size, _u_shift, _v_shift)
    args = {
        "mask": None,
        "window_size_iters": (1, 2),
        "min_window_size": 16,
        "overlap_ratio": 0.5,
        "dt": 1,
        "deform": True,
        "smooth": True,
        "num_validation_iters": 1,
        "validation_method": "median_velocity",
    }

    x, y, u, v, mask, s2n = gpu_process.gpu_piv(frame_a, frame_b, **args)

    assert (
        np.linalg.norm(u[_trim_slice, _trim_slice] - _u_shift) / sqrt(u.size)
        < _accuracy_tolerance
    )
    assert (
        np.linalg.norm(-v[_trim_slice, _trim_slice] - _v_shift) / sqrt(u.size)
        < _accuracy_tolerance
    )


@pytest.mark.parametrize("image_size", (_image_size_rectangle, _image_size_square))
def test_gpu_piv_zero(image_size):
    """Tests that zero-displacement is returned when the images are empty."""
    frame_a = frame_b = np.zeros(image_size, dtype=np.int32)
    args = {
        "mask": None,
        "window_size_iters": (1, 2),
        "min_window_size": 16,
        "overlap_ratio": 0.5,
        "dt": 1,
        "deform": True,
        "smooth": True,
        "num_validation_iters": 1,
        "validation_method": "median_velocity",
    }

    x, y, u, v, mask, s2n = gpu_process.gpu_piv(frame_a, frame_b, **args)

    assert np.allclose(u, 0, _identity_tolerance)
    assert np.allclose(v, 0, _identity_tolerance)


def test_extended_search_area():
    """Inputs every s2n method to ensure they don't error out."""
    frame_a, frame_b = create_pair_shift(_image_size_rectangle, _u_shift, _v_shift)
    args = {
        "mask": None,
        "window_size_iters": (2, 2),
        "min_window_size": 8,
        "overlap_ratio": 0.5,
        "dt": 1,
        "deform": True,
        "smooth": True,
        "num_validation_iters": 2,
        "extend_ratio": 2,
    }

    x, y, u, v, mask, s2n = gpu_process.gpu_piv(frame_a, frame_b, **args)

    assert (
        np.linalg.norm(u[_trim_slice, _trim_slice] - _u_shift) / sqrt(u.size)
        < _accuracy_tolerance
    )
    assert (
        np.linalg.norm(-v[_trim_slice, _trim_slice] - _v_shift) / sqrt(u.size)
        < _accuracy_tolerance
    )


@pytest.mark.parametrize("s2n_method", ("peak2peak", "peak2mean", "peak2energy"))
def test_sig2noise(s2n_method):
    """Inputs every s2n method to ensure they don't error out."""
    frame_a, frame_b = create_pair_shift(_image_size_rectangle, _u_shift, _v_shift)
    args = {
        "mask": None,
        "window_size_iters": (1, 2, 2),
        "min_window_size": 8,
        "overlap_ratio": 0.5,
        "dt": 1,
        "deform": True,
        "smooth": True,
        "num_validation_iters": 2,
        "validation_method": "median_velocity",
        "return_s2n": True,
        "s2n_method": s2n_method,
    }

    _ = gpu_process.gpu_piv(frame_a, frame_b, **args)


@pytest.mark.parametrize("subpixel_method", ("gaussian", "centroid", "parabolic"))
def test_subpixel_peak(subpixel_method):
    """Inputs every s2n method to ensure they don't error out."""
    frame_a, frame_b = create_pair_shift(_image_size_rectangle, _u_shift, _v_shift)
    args = {
        "mask": None,
        "window_size_iters": (1, 2, 2),
        "min_window_size": 8,
        "overlap_ratio": 0.5,
        "dt": 1,
        "deform": True,
        "smooth": True,
        "num_validation_iters": 2,
        "validation_method": "median_velocity",
        "subpixel_method": subpixel_method,
    }

    _ = gpu_process.gpu_piv(frame_a, frame_b, **args)


# s2n must not cause invalid numbers to be passed to smoothn.
@pytest.mark.parametrize(
    "validation_method", ("s2n", "mean_velocity", "median_velocity", "rms_velocity")
)
def test_validation(validation_method):
    """Inputs every s2n method to ensure they don't error out."""
    frame_a, frame_b = create_pair_shift(_image_size_rectangle, _u_shift, _v_shift)
    args = {
        "mask": None,
        "window_size_iters": (1, 2, 2),
        "min_window_size": 8,
        "overlap_ratio": 0.5,
        "dt": 1,
        "deform": True,
        "smooth": True,
        "num_validation_iters": 2,
        "validation_method": validation_method,
    }

    _ = gpu_process.gpu_piv(frame_a, frame_b, **args)


# TODO regression
# sweep the input variables to ensure everything is same
@pytest.mark.parametrize(
    "window_size_iters",
    [1, (1, 1), (1, 1, 1), (1, 1, 2), (1, 2, 2), (2, 2, 2), (1, 2, 1)],
)
@pytest.mark.parametrize("min_window_size", [8, 16])
@pytest.mark.parametrize("num_validation_iters", [0, 1, 2])
def test_gpu_piv_py(
    window_size_iters, min_window_size, num_validation_iters, ndarrays_regression
):
    """This test checks that the output remains the same."""
    frame_a = imread("../../data/test1/exp1_001_a.bmp")
    frame_b = imread("../../data/test1/exp1_001_b.bmp")
    args = {
        "mask": None,
        "window_size_iters": window_size_iters,
        "min_window_size": min_window_size,
        "overlap_ratio": 0.5,
        "dt": 1,
        "deform": True,
        "smooth": True,
        "num_validation_iters": num_validation_iters,
        "validation_method": "median_velocity",
        "smoothing_par": 0.5,
        "center_field": False,
    }

    x, y, u, v, mask, s2n = gpu_process.gpu_piv(frame_a, frame_b, **args)

    ndarrays_regression.check({"u": u, "v": v})


# @pytest.mark.parametrize('image_size', (_image_size_rectangle, _image_size_square))
# def test_gpu_piv_fast0(image_size):
#     """Quick test of the main piv function."""
#     frame_a, frame_b = create_pair_shift(image_size, _u_shift, _v_shift)
#     args = {'mask': None,
#             'window_size_iters': (1, 2),
#             'min_window_size': 16,
#             'overlap_ratio': 0.5,
#             'dt': 1,
#             'deform': True,
#             'smooth': True,
#             'num_validation_iters': 1,
#             'validation_method': 'median_velocity',
#             }
#
#     x, y, u, v, mask, s2n = gpu_process.gpu_piv(frame_a, frame_a, **args)
#
#     assert np.linalg.norm(u[_trim_slice, _trim_slice] - _u_shift)\
#            / sqrt(u.size) < _accuracy_tolerance
#     assert np.linalg.norm(-v[_trim_slice, _trim_slice] - _v_shift)\
#            / sqrt(u.size) < _accuracy_tolerance


@pytest.mark.integtest
def test_gpu_piv_fast():
    """Quick test of the main piv function."""
    frame_size = (512, 512)
    u_shift = 8
    v_shift = -4
    trim_slice = slice(2, -2, 1)
    args = {
        "mask": None,
        "window_size_iters": (1, 2),
        "min_window_size": 16,
        "overlap_ratio": 0.5,
        "dt": 1,
        "deform": True,
        "smooth": True,
        "num_validation_iters": 1,
        "validation_method": "median_velocity",
    }

    frame_a, frame_b = create_pair_shift(frame_size, u_shift, v_shift)

    x, y, u, v, mask, s2n = gpu_process.gpu_piv(frame_a, frame_b, **args)

    assert np.linalg.norm(u[trim_slice, trim_slice] - u_shift) / sqrt(u.size) < 0.1
    assert np.linalg.norm(-v[trim_slice, trim_slice] - v_shift) / sqrt(u.size) < 0.1


# TODO regression
@pytest.mark.integtest
@pytest.mark.parametrize(
    "window_size_iters",
    [1, (1, 1), (1, 1, 1), (1, 1, 2), (1, 2, 2), (2, 2, 2), (1, 2, 1)],
)
@pytest.mark.parametrize("min_window_size", [8, 16])
@pytest.mark.parametrize("num_validation_iters", [0, 1, 2])
def test_gpu_piv_py(
    window_size_iters, min_window_size, num_validation_iters, ndarrays_regression
):
    """This test checks that the output remains the same."""
    frame_a = imread(data_dir + "test1/exp1_001_a.bmp")
    frame_b = imread(data_dir + "test1/exp1_001_b.bmp")
    args = {
        "mask": None,
        "window_size_iters": window_size_iters,
        "min_window_size": min_window_size,
        "overlap_ratio": 0.5,
        "dt": 1,
        "deform": True,
        "smooth": True,
        "num_validation_iters": num_validation_iters,
        "validation_method": "median_velocity",
        "smoothing_par": 0.5,
        "center_field": False,
    }

    x, y, u, v, mask, s2n = gpu_process.gpu_piv(frame_a, frame_b, **args)

    ndarrays_regression.check({"u": u, "v": v})


# TODO
@pytest.mark.integtest
def test_correlation_gpu():
    """"""
    # This tests at a basic level that the gpu-correlation returns a believable result.
    pass


# TODO
@pytest.mark.integtest
def test_extended_search_area():
    """"""
    frame_size = (512, 512)
    u_shift = 8
    v_shift = -4
    trim_slice = slice(2, -2, 1)
    args = {
        "mask": None,
        "window_size_iters": (2, 2),
        "min_window_size": 8,
        "overlap_ratio": 0.5,
        "dt": 1,
        "deform": True,
        "smooth": True,
        "num_validation_iters": 2,
        "extend_ratio": 2,
    }

    frame_a, frame_b = create_pair_shift(frame_size, u_shift, v_shift)

    x, y, u, v, mask, s2n = gpu_process.gpu_piv(frame_a, frame_b, **args)

    assert np.linalg.norm(u[trim_slice, trim_slice] - u_shift) / sqrt(u.size) < 0.1
    assert np.linalg.norm(-v[trim_slice, trim_slice] - v_shift) / sqrt(u.size) < 0.1


# BENCHMARKS
@pytest.mark.parametrize("frame_shape", [(1024, 1024), (2048, 2048)])
@pytest.mark.parametrize(
    "window_size_iters, min_window_size", [((1, 2), 16), ((1, 2, 2), 8)]
)
def test_gpu_piv_benchmark(benchmark, frame_shape, window_size_iters, min_window_size):
    """Benchmarks the PIV function."""
    u_shift = 8
    v_shift = -4
    args = {
        "mask": None,
        "window_size_iters": window_size_iters,
        "min_window_size": min_window_size,
        "overlap_ratio": 0.5,
        "dt": 1,
        "deform": True,
        "smooth": True,
        "num_validation_iters": 2,
        "validation_method": "median_velocity",
    }

    frame_a, frame_b = create_pair_shift(frame_shape, u_shift, v_shift)

    benchmark(gpu_process.gpu_piv, frame_a, frame_b, **args)


def test_gpu_piv_benchmark_oop(benchmark):
    """Benchmarks the PIV speed with the objected-oriented interface."""
    shape = (1024, 1024)
    u_shift = 8
    v_shift = -4
    args = {
        "mask": None,
        "window_size_iters": (1, 2, 2),
        "min_window_size": 8,
        "overlap_ratio": 0.5,
        "dt": 1,
        "deform": True,
        "smooth": True,
        "num_validation_iters": 2,
        "validation_method": "median_velocity",
    }
    frame_a, frame_b = create_pair_shift(shape, u_shift, v_shift)

    piv_gpu = gpu_process.PIVGPU(shape, **args)

    @benchmark
    def repeat_10():
        for i in range(10):
            piv_gpu(frame_a, frame_b)
