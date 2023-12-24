from math import sqrt, log2

import numpy as np
import pytest
import scipy.fft as fft
from scipy.ndimage import distance_transform_edt, shift, map_coordinates
from scipy.signal import correlate2d
from skimage.util import img_as_ubyte
from skimage.util import random_noise
from pycuda import gpuarray

from openpiv.gpu import process, DTYPE_i, DTYPE_f

# dirs
data_dir = "../data/"


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


def gpu_arrays(*f_l):
    return [gpuarray.to_gpu(f) for f in f_l]


def np_arrays(*f_l):
    return [f.get() for f in f_l]


def cross_correlate_np(a, b):
    fft_product = fft.fft2(a).conj() * fft.fft2(b)
    return fft.ifft(fft_product).real


def window_slice_np(frame, field_shape, window_size, spacing, offset):
    m, n = field_shape
    ws = window_size

    win_np = np.zeros((m * n, ws, ws), dtype=DTYPE_f)
    for i in range(m):
        row0 = offset + spacing * i
        row1 = offset + spacing * i + ws
        # Get the beginning and end indexes that are inside the windows.
        row_inside0 = max(0, row0)
        row_inside1 = min(m, row1)

        frame_row_idx = slice(row0, row1)
        win_row_idx = slice(row_inside0 - row0, ws - (row_inside1 - row1))

        for j in range(n):
            col0 = offset + spacing * j
            col1 = offset + spacing * j + ws
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


def _window_size_parameterization(ws_iters, min_window_size):
    if isinstance(ws_iters, int):
        ws_iters = (ws_iters,)
    for i, num_iters in enumerate(ws_iters):
        yield (2 ** (len(ws_iters) - i - 1)) * min_window_size, num_iters


# UNIT TESTS
def test_correlation_gpu_signal_to_noise(s2n_ratio):
    assert isinstance(s2n_ratio, gpuarray.GPUArray)


def test_correlation_gpu_init_fft_shape(correlation_gpu):
    fft_shape = correlation_gpu.fft_shape

    assert round(log2(fft_shape[0])) == log2(fft_shape[0])
    assert round(log2(fft_shape[1])) == log2(fft_shape[1])


def test_correlation_gpu_correlate_windows(correlation_gpu):
    shape = (32, 32)
    fft_shape = correlation_gpu.fft_shape[0]
    a = 1
    c = 4

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    r = (x - (shape[1] - 1) / 2) ** 2 + (y - (shape[0] - 1) / 2) ** 2
    win = a * np.exp(-r / (2 * c**2))
    win_d = gpuarray.to_gpu(win.reshape(1, *shape).astype(DTYPE_f))
    correlation_gpu._correlate_windows(win_d, win_d)
    peak_idx = np.argmax(correlation_gpu._correlation.get())

    assert peak_idx == (fft_shape / 2) * (fft_shape + 1)


def test_get_peak_idx(correlation_gpu):
    assert isinstance(correlation_gpu._peak1_idx, gpuarray.GPUArray)


def test_get_peak_value(correlation_gpu):
    assert isinstance(correlation_gpu._corr_peak1, gpuarray.GPUArray)


def test_correlation_gpu_check_non_positive_correlation(correlation_gpu, piv_field_gpu):
    fft_ht, fft_wd = correlation_gpu.fft_shape

    corr_peak = correlation_gpu._corr_peak1_.get()
    corr_peak[:] = 0
    correlation_gpu._corr_peak1_ = gpuarray.to_gpu(corr_peak)
    correlation_gpu._check_non_positive_correlation()

    assert np.all(
        correlation_gpu._peak1_idx_.get() == (fft_ht // 2) * fft_wd + fft_wd // 2
    )


def test_correlation_gpu_get_displacement(correlation_gpu):
    i_peak, j_peak = correlation_gpu.get_displacement_peaks()
    assert isinstance(j_peak, gpuarray.GPUArray)
    assert isinstance(i_peak, gpuarray.GPUArray)


def test_correlation_gpu_center_displacement(piv_field_gpu, correlation_gpu):
    field_shape = piv_field_gpu.shape
    field_size = piv_field_gpu.size
    fft_shape = correlation_gpu.fft_shape[0]

    row_sp = np.arange(field_size).reshape(field_shape)
    col_sp = np.arange(field_size).reshape(field_shape)
    i_peak, j_peak = correlation_gpu._center_displacement(row_sp, col_sp)

    assert np.array_equal(i_peak, col_sp - fft_shape // 2)
    assert np.array_equal(j_peak, row_sp - fft_shape // 2)


@pytest.mark.parametrize("s2n_method", list(process.ALLOWED_S2N_METHODS))
def test_correlation_gpu_get_s2n(correlation_gpu, s2n_method):
    correlation_gpu.s2n_method = s2n_method
    s2n = correlation_gpu._s2n_ratio

    assert isinstance(s2n, gpuarray.GPUArray)


def test_correlation_gpu_get_peak2peak(correlation_gpu):
    assert isinstance(correlation_gpu._get_peak2peak(), gpuarray.GPUArray)


def test_correlation_gpu_free_gpu_data(correlation_gpu):
    correlation_gpu.free_gpu_data()

    assert correlation_gpu._correlation is None
    assert correlation_gpu._corr_peak1_ is None
    assert correlation_gpu._corr_idx is None


def test_piv_field_gpu_get_mask(piv_field_gpu):
    mask_ = piv_field_gpu.get_gpu_mask(return_array=True)

    assert isinstance(mask_, gpuarray.GPUArray)


@pytest.mark.parametrize("search_size", [16, 32])
def test_piv_field_gpu_stack_iw(search_size):
    # Need to test with center_field
    shape = (48, 48)
    window_size = 16
    offset = (search_size - window_size) // 2

    piv_field = process.PIVField(shape, 16, 8)
    frame = np.zeros(shape, dtype=DTYPE_f)
    frame[4, 5] = 1
    frame = gpuarray.to_gpu(frame)
    win_a, win_b = piv_field.stack_iw(frame, frame, search_size=search_size)
    win_a = win_a.get()
    win_b = win_b.get()

    assert win_a[0][4, 5] == 1
    assert win_b[0][4 + offset, 5 + offset] == 1


def test_piv_field_gpu_free_gpu_data(piv_field_gpu):
    piv_field_gpu.free_gpu_data()

    assert piv_field_gpu._mask_d is None


def test_piv_field_gpu_coords(piv_field_gpu):
    x, y = piv_field_gpu.coords

    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)


def test_piv_field_gpu_grid_coords(piv_field_gpu):
    x_grid, y_grid = piv_field_gpu.coords

    assert isinstance(x_grid, np.ndarray)
    assert isinstance(y_grid, np.ndarray)


def test_piv_field_gpu_center_offset(piv_field_gpu):
    offset_x, offset_y = piv_field_gpu.center_offset

    assert isinstance(offset_x, int)
    assert isinstance(offset_y, int)


@pytest.mark.parametrize("search_size", [16, 31, 32])
def test_piv_field_gpu_get_search_offset(search_size, piv_field_gpu):
    offset_a, offset_b = piv_field_gpu._get_offset(search_size)
    center_offset_x, center_offset_y = piv_field_gpu.center_offset
    search_offset = -(search_size - piv_field_gpu.window_size) // 2

    assert offset_a[0] == center_offset_x
    assert offset_a[1] == center_offset_y
    assert offset_b[0] == center_offset_x + search_offset
    assert offset_b[1] == center_offset_y + search_offset


def test_piv_coords(piv_gpu):
    x, y = piv_gpu.coords

    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)


def test_piv_field_mask(piv_gpu):
    mask_ = piv_gpu.field_mask

    assert isinstance(mask_, np.ndarray)


def test_piv_s2n(s2n_ratio):

    assert isinstance(s2n_ratio, gpuarray.GPUArray)


def test_piv_free_gpu_data(piv_gpu):
    piv_gpu.free_gpu_data()

    assert piv_gpu._piv_fields_ is None
    assert piv_gpu._frame_mask_ is None
    assert piv_gpu._corr_gpu is None


def test_piv_get_piv_fields(piv_gpu):
    s2n = piv_gpu._piv_fields

    assert isinstance(s2n, list)


def test_piv_mask_frame(piv_gpu, frames, frame_mask):
    # Need to test the switch cases.
    frame_a, frame_b = frames

    piv_gpu.mask = frame_mask
    frame_a_masked, frame_b_masked = piv_gpu._frames_to_gpu(frame_a, frame_b)

    assert isinstance(frame_a_masked, gpuarray.GPUArray)
    assert isinstance(frame_a_masked, gpuarray.GPUArray)


def test_piv_frame_mask(piv_gpu, boolean_np_array):
    shape = (16, 16)

    piv_gpu.mask = boolean_np_array(shape)
    piv_gpu._frame_mask_ = None
    frame_mask_ = piv_gpu._frame_mask

    assert isinstance(frame_mask_, gpuarray.GPUArray)


def test_piv_piv_field_k(piv_gpu):
    piv_gpu._k = 1
    piv_field_k = piv_gpu._piv_field_k

    assert piv_field_k == piv_gpu._piv_fields[1]


def test_piv_get_new_velocity(piv_gpu, frames_gpu, peaks_reshape):
    dp_u, dp_v = peaks_reshape

    piv_gpu._k = 0
    u, v = piv_gpu._get_new_velocity(frames_gpu, dp_u, dp_v)

    assert isinstance(u, gpuarray.GPUArray)
    assert isinstance(v, gpuarray.GPUArray)


def test_piv_get_predictions(piv_gpu, gpu_array, boolean_gpu_array):
    # Need to test different dimensions.
    shape = (22, 30)

    u = v = gpu_array(shape, center=0.0, half_width=1.0)
    mask_ = boolean_gpu_array(shape, seed=1)
    piv_gpu._k = 1
    piv_gpu._piv_field_k._mask = mask_
    dp_u, dp_v = piv_gpu._get_predictions(u, v)

    assert isinstance(dp_u, gpuarray.GPUArray)
    assert isinstance(dp_v, gpuarray.GPUArray)


def test_piv_get_displacement_peaks(piv_gpu, frames_gpu, peaks_reshape):
    dp_u, dp_v = peaks_reshape

    piv_gpu._k = 0
    i_peak, j_peak = piv_gpu._get_displacement_peaks(frames_gpu, dp_u, dp_v)

    assert isinstance(i_peak, gpuarray.GPUArray)
    assert isinstance(j_peak, gpuarray.GPUArray)


def test_piv_get_search_size(piv_gpu):
    piv_gpu._k = 0
    piv_gpu._piv_field_k.window_size = 8
    piv_gpu.search_ratio = 2
    search_size = piv_gpu._get_search_size()

    assert search_size == piv_gpu._piv_field_k.window_size * piv_gpu.search_ratio


def test_piv_get_window_deformation(piv_gpu, gpu_array, boolean_gpu_array):
    shape = (16, 16)

    dp_u = gpu_array(shape, center=0.0, half_width=1.0)
    mask_ = boolean_gpu_array(shape, seed=1)
    piv_gpu._piv_field_k._mask_d = mask_
    shift_, strain = piv_gpu._get_window_deformation(dp_u, dp_u)

    assert isinstance(shift_, gpuarray.GPUArray)
    assert isinstance(strain, gpuarray.GPUArray)


@pytest.mark.parametrize("dp_u", [True, False])
def test_piv_update_velocity(
    dp_u, piv_gpu, peaks_reshape, gpu_array, boolean_gpu_array
):
    i_peak, j_peak = peaks_reshape

    dp_u = gpu_array(i_peak.shape, center=0.0, half_width=1.0) if dp_u else None
    mask_ = boolean_gpu_array(i_peak.shape, seed=2)
    piv_gpu._k = 0
    piv_gpu._piv_field_k._mask_d = mask_
    u, v = piv_gpu._update_velocity(dp_u, dp_u, i_peak, i_peak)

    assert isinstance(u, gpuarray.GPUArray)
    assert isinstance(v, gpuarray.GPUArray)


@pytest.mark.parametrize("num_validation_iters", [0, 1, 2])
def test_piv_validate_fields(num_validation_iters, piv_gpu, gpu_array):
    shape = (16, 16)

    u = v = gpu_array(shape, center=0.0, half_width=1.0)
    dp_u = dp_v = gpu_array(shape, center=0.0, half_width=1.0, seed=1)
    piv_gpu.num_validation_iters = num_validation_iters
    u, v, val_locations = piv_gpu._validate_fields(u, v, dp_u, dp_v)

    assert isinstance(u, gpuarray.GPUArray)
    assert isinstance(v, gpuarray.GPUArray)
    if val_locations is not None:
        assert isinstance(val_locations, gpuarray.GPUArray)


@pytest.mark.parametrize("k", [0, 1])
def test_replace_invalid_vectors(k, piv_gpu, validation_gpu, peaks_reshape, gpu_array):
    dp_u, dp_v = peaks_reshape

    piv_gpu._k = k
    piv_gpu._validation_gpu = validation_gpu
    piv_gpu._validation_gpu(dp_u, dp_v)
    u, v = piv_gpu._replace_invalid_vectors(dp_u, dp_v)

    assert isinstance(u, gpuarray.GPUArray)
    assert isinstance(v, gpuarray.GPUArray)


@pytest.mark.parametrize("val_locations", [True, False])
def test_smooth_fields(val_locations, piv_gpu, gpu_array, boolean_np_array):
    shape = (16, 16)

    u = v = gpu_array(shape, center=0.0, half_width=1.0)
    val_locations = boolean_np_array(shape, seed=1) if val_locations else None
    u, v = piv_gpu._smooth_fields(u, v, val_locations)

    assert isinstance(u, gpuarray.GPUArray)
    assert isinstance(v, gpuarray.GPUArray)


def test_piv_get_residual(piv_gpu, array_pair):
    shape = (16, 16)

    i_peak, i_peak_d = array_pair(shape, center=0.0, half_width=1.0)
    residual = piv_gpu._get_residual(i_peak_d, i_peak_d)

    assert residual == sqrt((np.sum(i_peak**2 + i_peak**2)) / i_peak.size) / 0.5


@pytest.mark.parametrize("frame_shape", [(64, 64), (63, 63)])
@pytest.mark.parametrize("window_size", [16])
@pytest.mark.parametrize("spacing", [8, 7])
def test_get_field_shape(frame_shape: tuple, window_size, spacing):
    ht, wd = frame_shape
    m, n = process.field_shape(frame_shape, window_size, spacing)

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
    offset_x = 0
    offset_y = 0
    if center_field:
        offset_x, offset_y = process._center_offset(
            frame_shape, window_size, spacing
        )
    x, y = process.field_coords(frame_shape, window_size, spacing, center_field)
    m, n = x.shape

    assert x[0, 0] == half_width + offset_x
    assert x[0, -1] == half_width + offset_x + spacing * (n - 1)
    assert y[0, 0] == half_width + offset_y + spacing * (m - 1)
    assert y[-1, 0] == half_width + offset_y


def test_gpu_strain(array_pair):
    shape = (16, 16)

    u, u_d = array_pair(shape, center=0.0, half_width=1.0)
    v, v_d = array_pair(shape, center=0.0, half_width=1.0, seed=1)

    u_y, u_x = np.gradient(u)
    v_y, v_x = np.gradient(v)
    strain_gpu = (process.gpu_strain(u_d, v_d)).get()

    assert np.array_equal(u_x, strain_gpu[0])
    assert np.array_equal(u_y, strain_gpu[1])
    assert np.array_equal(v_x, strain_gpu[2])
    assert np.array_equal(v_y, strain_gpu[3])


@pytest.mark.parametrize("shape", [(16, 16, 16), (15, 12, 14), (15, 11, 13)])
def test_gpu_fft_shift(shape, array_pair):
    shape = (16, 16, 16)

    correlation, correlation_d = array_pair(shape, center=0.0, half_width=1.0)

    correlation_shifted_np = fft.fftshift(correlation, axes=(1, 2))
    correlation_shifted_gpu = process.gpu_fft_shift(correlation_d).get()

    assert np.array_equal(correlation_shifted_gpu, correlation_shifted_np)


def test_window_sizes():
    ws_iters = [(32, 2), (16, 1), (8, 3)]
    window_size_l = list(process._window_sizes(ws_iters))

    assert window_size_l == [32, 32, 16, 8, 8, 8]


@pytest.mark.parametrize("overlap_ratio, spacing", [(0.01, 7), (0.5, 4), (0.99, 1)])
def test_spacing(overlap_ratio, spacing):
    window_size = 8

    assert spacing == process._spacing(window_size, overlap_ratio)


def test_field_mask():
    ht, wd = 32, 32
    w = np.pi / 2

    x0, y0 = np.meshgrid(np.arange(ht), np.arange(wd))
    x1, y1 = np.meshgrid(np.arange(0, ht, 2), np.arange(0, wd, 2))
    frame_mask_ = np.round((1 + np.cos(x0 * w) * np.cos(y0 * w)) / 2).astype(int)
    field_mask0 = np.round((1 + np.cos(x1 * w) * np.cos(y1 * w)) / 2).astype(int)
    field_mask1 = process._field_mask(x1, y1, frame_mask_)

    assert np.array_equal(field_mask1, field_mask0)


@pytest.mark.parametrize(
    "frame_size, window_size, spacing, offset",
    [(64, 8, 4, 0), (66, 8, 4, 1), (66, 7, 4, 2)],
)
def test_center_offset(frame_size, window_size, spacing, offset):
    offset_x, offset_y = process._center_offset(
        (frame_size, frame_size), window_size, spacing
    )

    assert offset_x == offset


@pytest.mark.parametrize("window_size", [4, 8])
@pytest.mark.parametrize("spacing", [4, 8])
@pytest.mark.parametrize("offset", [-1, 1])
@pytest.mark.parametrize("pass_shift", [True, False])
def test_gpu_window_slice(window_size, spacing, offset, pass_shift, array_pair):
    frame_shape = (16, 16)
    window_size = 8
    spacing = 4
    field_shape = process.field_shape(frame_shape, window_size, spacing)
    offset = 0

    frame, frame_d = array_pair(frame_shape, center=0.0, half_width=1.0)
    shift_d = (
        gpuarray.to_gpu(np.ones((2, *field_shape), dtype=DTYPE_f))
        if pass_shift
        else None
    )

    win_np = window_slice_np(frame, field_shape, window_size, spacing, offset)
    win_gpu = process._gpu_window_slice(
        frame_d, field_shape, window_size, spacing, offset, shift=shift_d
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
    offset = 0
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
    win_gpu = process._gpu_window_slice(
        frame_d, field_shape, ws, spacing, offset, dt=dt, shift=shift_d
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
    offset = 0
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
    win_gpu = process._gpu_window_slice(
        frame_d, field_shape, ws, spacing, offset, dt=dt, shift=shift_d, strain=strain_d
    ).get()

    assert np.allclose(win_gpu, win_np, atol=1e-6)


def test_zero_pad_offset():
    shape_a = (1, 16, 16)
    shape_b = (1, 31, 32)

    win_a = np.zeros(shape_a)
    win_b = np.zeros(shape_b)
    offset_x, offset_y = process._zero_pad_offset(win_a, win_b)

    assert offset_x == (shape_b[2] - shape_a[2]) // 2
    assert offset_y == (shape_b[1] - shape_a[1]) // 2


def test_gpu_normalize_intensity(array_pair):
    shape = (16, 16, 16)
    n_windows, ht, wd = shape

    win, win_d = array_pair(shape, center=0.0, half_width=1.0)

    mean = np.mean(win.reshape((n_windows, ht * wd)), axis=1).reshape((n_windows, 1, 1))
    win_norm_np = win - mean
    win_norm_gpu = process._gpu_normalize_intensity(win_d).get()

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
    win_zp_gpu = process._gpu_zero_pad(win_d, fft_shape, offset).get()

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
    correlation_d = process._gpu_cross_correlate(win_a_d, win_b_d)
    correlation_gpu_ = correlation_d.get()

    assert np.allclose(correlation_gpu_, correlation_np, atol=1e-5)


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
    values_gpu = process._gpu_window_index_f(data_d, indices_d).get()

    assert np.array_equal(values_gpu, values_np)


def test_find_peak(array_pair):
    shape = (16, 16, 16)
    n_windows, ht, wd = shape

    correlation, correlation_d = array_pair(shape, center=0.0, half_width=1.0)

    peak_idx_np = np.argmax(correlation.reshape(n_windows, ht * wd), axis=1).astype(
        dtype=DTYPE_i
    )
    peak_idx_gpu = process._peak_idx(correlation_d).get()

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
    corr_peak_gpu = process._peak_value(correlation_d, peak_idx_d).get()

    assert np.array_equal(corr_peak_gpu, corr_peak_np)


@pytest.mark.parametrize(
    "method, tol", [("gaussian", 0.1), ("parabolic", 0.1), ("centroid", 0.25)]
)
def test_gpu_subpixel_approximation(method, tol, np_array):
    shape = (16, 16, 16)
    n_windows, ht, wd = shape

    row_sp = np_array(n_windows, center=ht / 2, half_width=ht / 2 - 1)
    col_sp = np_array(n_windows, center=wd / 2, half_width=wd / 2 - 1, seed=1)
    row_idx = np.round(row_sp).astype(DTYPE_i)
    col_idx = np.round(col_sp).astype(DTYPE_i)
    peak_idx = gpuarray.to_gpu(row_idx * wd + col_idx)

    # Create the distribution.
    if method == "parabolic":
        correlation_d = parabolic_peak(shape, row_sp, col_sp)
    else:
        correlation_d = gaussian_peak(shape, row_sp, col_sp)

    row_sp_d, col_sp_d = process._gpu_subpixel_approximation(
        correlation_d, peak_idx, method
    )
    row_sp_gpu, col_sp_gpu = np_arrays(row_sp_d, col_sp_d)

    assert np.all(np.abs(row_sp_gpu - row_sp) <= tol)
    assert np.all(np.abs(col_sp_gpu - col_sp) <= tol)


def test_peak2rms(array_pair):
    shape = (16, 16, 16)
    n_windows, ht, wd = shape
    size = ht * wd

    correlation, correlation_d = array_pair(shape)
    corr_peak, corr_peak_d = array_pair((n_windows,), seed=1)

    correlation_masked = correlation * (
        correlation < corr_peak.reshape(n_windows, 1, 1) / 2
    )
    s2n_np = np.log10(
        corr_peak**2
        / np.mean(correlation_masked.reshape(n_windows, size) ** 2, axis=1)
    )
    s2n_gpu = process._peak2rms(correlation_d, corr_peak_d).get()

    assert np.allclose(s2n_gpu, s2n_np)


def test_peak2energy(array_pair):
    shape = (16, 16, 16)
    n_windows, ht, wd = shape
    size = ht * wd

    correlation, correlation_d = array_pair(shape)
    corr_peak, corr_peak_d = array_pair((n_windows,), seed=1)

    s2n_np = np.log10(
        corr_peak**2 / np.mean(correlation.reshape(n_windows, size) ** 2, axis=1)
    )
    s2n_gpu = process._peak2energy(correlation_d, corr_peak_d).get()

    assert np.allclose(s2n_gpu, s2n_np)


def test_correlation_gpu_get_second_peak(correlation_gpu):
    shape = (32, 32)
    fft_shape = correlation_gpu.fft_shape[0]
    a = 1
    c = 4

    row_peak = (
        np.array(fft_shape // 2)
        .astype(DTYPE_i)
        .reshape(
            1,
        )
    )
    col_peak = (
        np.array(fft_shape // 2)
        .astype(DTYPE_i)
        .reshape(
            1,
        )
    )
    correlation_gpu._peak1_idx_ = gpuarray.to_gpu(row_peak * fft_shape + col_peak)

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    r = (x - (shape[1] - 1) / 2) ** 2 + (y - (shape[0] - 1) / 2) ** 2
    win = a * np.exp(-r / (2 * c**2))
    win_d = gpuarray.to_gpu(np.expand_dims(win, 0).astype(DTYPE_f))
    correlation_gpu._correlate_windows(win_d, win_d)
    corr = correlation_gpu._correlation
    peak1_idx = correlation_gpu._peak1_idx_
    corr_max2 = process._get_second_peak(corr, peak1_idx, 1).get()

    assert corr_max2 < np.amax(corr.get())


def test_peak2peak(array_pair):
    shape = (16, 16, 16)
    n_windows, ht, wd = shape

    corr_peak1, corr_peak1_d = array_pair((n_windows,))
    corr_peak2, corr_peak2_d = array_pair((n_windows,), seed=1)

    s2n_np = np.log10(corr_peak1 / corr_peak2)
    s2n_gpu = process._peak2peak(corr_peak1_d, corr_peak2_d).get()

    assert np.allclose(s2n_gpu, s2n_np)


@pytest.mark.parametrize("width", [1, 2])
def test_gpu_mask_peak(width, array_pair, np_array):
    shape = (16, 16, 16)
    n_windows, ht, wd = shape

    correlation, correlation_d = array_pair(shape)
    row_peak = np_array(
        (n_windows,), center=ht / 2, half_width=ht / 2, d_type=DTYPE_i, seed=1
    )
    col_peak = np_array(
        (n_windows,), center=wd / 2, half_width=wd / 2, d_type=DTYPE_i, seed=2
    )
    peak_idx = gpuarray.to_gpu(row_peak * wd + col_peak)

    correlation_masked_np = mask_peak_np(correlation, row_peak, col_peak, width)
    correlation_masked_gpu = process._gpu_mask_peak(
        correlation_d, peak_idx, width
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
    correlation_masked_gpu = process._gpu_mask_rms(correlation_d, corr_peak_d).get()

    assert np.array_equal(correlation_masked_gpu, correlation_masked_np)


def test_piv_iter():
    window_size_iters = [(32, 2), (16, 2), (8, 3)]

    iters = list(process._piv_iter(window_size_iters))

    assert sum(iters) == 21


def test_field_shift(array_pair):
    shape = (16, 16)

    u, u_d = array_pair(shape, center=0.0, half_width=1.0)
    v, v_d = array_pair(shape, center=0.0, half_width=1.0, seed=1)

    shift_np = np.stack([u, v], axis=0)
    shift_gpu = process._field_shift(u_d, v_d).get()

    assert np.array_equal(shift_gpu, shift_np)


def test_gpu_update_field(array_pair, boolean_array_pair):
    shape = (16, 16)

    dp, dp_d = array_pair(shape, center=0.0, half_width=1.0)
    peak, peak_d = array_pair(shape, center=0.0, half_width=1.0, seed=1)
    mask_, mask_d = boolean_array_pair(shape, seed=2)

    f_np = (dp + peak) * (mask_ == 0)
    f_gpu = process._gpu_update_field(dp_d, peak_d, mask_d).get()

    assert np.array_equal(f_np, f_gpu)


@pytest.mark.parametrize("val_locations", [True, False])
def test_update_validation_locations(val_locations, gpu_array):
    shape = (16, 16)

    val_locations = gpu_array(shape) if val_locations else None
    new_val_locations = gpu_array(shape, seed=1)

    val_locations = process._update_validation_locations(
        val_locations, new_val_locations
    )

    assert isinstance(val_locations, gpuarray.GPUArray)


# INTEGRATION TESTS
@pytest.fixture
def process_shift(request):
    frame_shape = request.param
    u_shift = 8
    v_shift = -4
    frame_a, frame_b = create_pair_shift(frame_shape, u_shift, v_shift)

    def process_shift(**params):
        trim_slice = slice(2, -2, 1)

        _, _, u, v, _, _ = process.gpu_piv(frame_a, frame_b, **params)

        assert np.linalg.norm(u[trim_slice, trim_slice] - u_shift) / sqrt(u.size) < 0.1
        assert np.linalg.norm(-v[trim_slice, trim_slice] - v_shift) / sqrt(u.size) < 0.1

    return process_shift


@pytest.fixture
def correlate():
    shape = (4, 16, 16)

    win_a = np.zeros(shape, dtype=DTYPE_f)
    win_b = np.zeros(shape, dtype=DTYPE_f)
    win_a[0, 4, 5] = 1
    win_b[0, 5, 7] = 1
    win_a = gpuarray.to_gpu(win_a)
    win_b = gpuarray.to_gpu(win_b)

    def correlate(**params):
        corr_gpu = process.Correlation(**params)
        corr_gpu(win_a, win_b)
        i_peak, j_peak = corr_gpu.get_displacement_peaks()

        assert round(j_peak.get()[0]) == 2
        assert round(i_peak.get()[0]) == 1

    return correlate


@pytest.mark.integtest
def test_correlation_gpu(correlate):
    """Check that the gpu-correlation returns a believable result."""
    correlate()


@pytest.mark.integtest
class TestCorrelationParams:
    @pytest.mark.parametrize(
        "subpixel_method", list(process.ALLOWED_SUBPIXEL_METHODS)
    )
    def test_subpixel_method(self, subpixel_method, correlate):
        correlate(subpixel_method=subpixel_method)

    @pytest.mark.parametrize("s2n_method", list(process.ALLOWED_S2N_METHODS))
    def test_subpixel_method(self, s2n_method, correlate):
        correlate(s2n_method=s2n_method)

    @pytest.mark.parametrize("s2n_width", [1, 2])
    def test_subpixel_method(self, s2n_width, correlate):
        correlate(s2n_width=s2n_width)

    @pytest.mark.parametrize("n_fft", [1, 2])
    def test_subpixel_method(self, n_fft, correlate):
        correlate(n_fft=n_fft)


@pytest.mark.integtest
@pytest.mark.parametrize("process_shift"
                         "", [(512, 512)], indirect=True)
@pytest.mark.parametrize("frame_shape", [(512, 512), (512, 1024)])
def test_gpu_piv(frame_shape, process_shift):
    """Quick test of the main piv function."""
    params = {
        "mask": None,
        "window_size_iters": ((32, 1), (16, 2)),
        "overlap_ratio": 0.5,
        "dt": 1,
        "deform": True,
        "smooth": True,
        "num_validation_iters": 1,
        "validation_method": "median_velocity",
    }

    process_shift(**params)


@pytest.mark.integtest
def test_gpu_piv_zero():
    """Tests that zero-displacement is returned when the images are empty."""
    shape = (512, 512)
    frame_a = frame_b = np.zeros(shape, dtype=np.int32)
    args = {
        "mask": None,
        "window_size_iters": ((32, 1), (16, 2)),
        "overlap_ratio": 0.5,
        "dt": 1,
        "deform": True,
        "smooth": True,
        "num_validation_iters": 1,
        "validation_method": "median_velocity",
    }

    x, y, u, v, mask, s2n = process.gpu_piv(frame_a, frame_b, **args)

    assert np.allclose(u, 0, 1e-6)
    assert np.allclose(v, 0, 1e-6)


@pytest.mark.parametrize("process_shift"
                         "", [(512, 512)], indirect=True)
@pytest.mark.integtest
def test_extended_search_area(process_shift):
    """"""
    params = {
        "mask": None,
        "window_size_iters": ((16, 2), (8, 2)),
        "overlap_ratio": 0.5,
        "dt": 1,
        "deform": True,
        "smooth": True,
        "num_validation_iters": 2,
        "search_ratio": 2,
    }

    process_shift(**params)


@pytest.mark.integtest
class TestProcessParams:
    frame_shape = (1024, 1024)

    @pytest.mark.parametrize(
        "process_shift",
        [
            frame_shape,
        ],
        indirect=True,
    )
    @pytest.mark.parametrize(
        "window_size_iters, min_window_size", [((1, 2), 16), ((1, 2, 2), 8)]
    )
    def test_window_size(self, window_size_iters, min_window_size, process_shift):
        window_size_iters = list(
            _window_size_parameterization(window_size_iters, min_window_size)
        )
        process_shift(window_size_iters=window_size_iters)

    @pytest.mark.parametrize(
        "process_shift",
        [
            frame_shape,
        ],
        indirect=True,
    )
    @pytest.mark.parametrize("overlap_ratio", [0.3, 0.5, 0.7])
    def test_overlap_ratio(self, overlap_ratio, process_shift):
        process_shift(overlap_ratio=overlap_ratio)

    @pytest.mark.parametrize(
        "process_shift",
        [
            frame_shape,
        ],
        indirect=True,
    )
    @pytest.mark.parametrize("dt", [0.99, 1])
    def test_dt(self, dt, process_shift):
        process_shift(dt=dt)

    @pytest.mark.parametrize(
        "process_shift",
        [
            frame_shape,
        ],
        indirect=True,
    )
    @pytest.mark.parametrize("mask", [frame_shape, False])
    def mask(self, mask, process_shift, boolean_gpu_array):
        mask = boolean_gpu_array(mask) if mask else None
        process_shift(mask=mask)

    @pytest.mark.parametrize(
        "process_shift",
        [
            frame_shape,
        ],
        indirect=True,
    )
    @pytest.mark.parametrize("deform", [True, False])
    def deform(self, deform, process_shift):
        process_shift(deform=deform)

    @pytest.mark.parametrize(
        "process_shift",
        [
            frame_shape,
        ],
        indirect=True,
    )
    @pytest.mark.parametrize("smooth", [True, False])
    def smooth(self, smooth, process_shift):
        process_shift(smooth=smooth)

    @pytest.mark.parametrize(
        "process_shift",
        [
            frame_shape,
        ],
        indirect=True,
    )
    @pytest.mark.parametrize("num_validation_iters", [0, 1, 2])
    def test_num_validation_iters(self, num_validation_iters, process_shift):
        process_shift(num_validation_iters=num_validation_iters)

    @pytest.mark.parametrize(
        "process_shift",
        [
            frame_shape,
        ],
        indirect=True,
    )
    @pytest.mark.parametrize("center_field", [True, False])
    def test_center_field(self, center_field, process_shift):
        process_shift(center_field=center_field)

    @pytest.mark.parametrize(
        "process_shift",
        [
            frame_shape,
        ],
        indirect=True,
    )
    @pytest.mark.parametrize("search_ratio", [1.5, 2, None])
    def test_num_search_ratio(self, search_ratio, process_shift):
        process_shift(window_size_iters=((16, 2), (8, 2)), search_ratio=search_ratio)


# REGRESSION TESTS
@pytest.mark.regression
@pytest.mark.parametrize(
    "window_size_iters",
    [1, (1, 1), (1, 1, 1), (1, 1, 2), (1, 2, 2), (2, 2, 2), (1, 2, 1)],
)
@pytest.mark.parametrize("min_window_size", [8, 16])
@pytest.mark.parametrize("num_validation_iters", [0, 1, 2])
def test_piv_regression(
    window_size_iters,
    min_window_size,
    num_validation_iters,
    frames,
    ndarrays_regression,
):
    """"""
    frame_a, frame_b = frames
    if isinstance(window_size_iters, int):
        window_size_iters = (window_size_iters,)
    window_size_iters = list(
        _window_size_parameterization(window_size_iters, min_window_size)
    )
    args = {
        "mask": None,
        "window_size_iters": window_size_iters,
        "overlap_ratio": 0.5,
        "dt": 1,
        "deform": True,
        "smooth": True,
        "num_validation_iters": num_validation_iters,
        "validation_method": "median_velocity",
        "smoothing_par": 0.5,
        "center_field": False,
    }

    x, y, u, v, mask, s2n = process.gpu_piv(frame_a, frame_b, **args)

    ndarrays_regression.check({"u": u, "v": v})


@pytest.mark.parametrize("search_ratio", [1, 2])
def test_stack_iw_determinism(search_ratio, frames_gpu, ndarrays_regression):
    frame_a, frame_b = frames_gpu
    shape = frame_a.shape
    params = {"window_size_iters": [(32, 1), (16, 1), (8, 2)]}

    # Process random data
    frame_a_random = np.random.random(shape) * 65535
    frame_b_random = np.random.random(shape) * 65535
    _ = process.gpu_piv(frame_a_random, frame_b_random, **params)

    # Process deterministic data
    piv_field = process.PIVField(shape, 32, 16)
    win_a, win_b = piv_field.stack_iw(frame_a, frame_b)
    ndarrays_regression.check({"win_a": win_a.get(), "win_b": win_b.get()})


# BENCHMARKS
@pytest.mark.parametrize("frame_shape", [(1024, 1024), (2048, 2048)])
@pytest.mark.parametrize(
    "window_size_iters, min_window_size", [((1, 2), 16), ((1, 2, 2), 8)]
)
def test_piv_benchmark(benchmark, frame_shape, window_size_iters, min_window_size):
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

    benchmark(process.gpu_piv, frame_a, frame_b, **args)


def test_piv_benchmark_oop(benchmark):
    """Benchmarks the PIV speed with the objected-oriented interface."""
    shape = (1024, 1024)
    u_shift = 8
    v_shift = -4
    args = {
        "mask": None,
        "window_size_iters": ((32, 1), (16, 2), (8, 2)),
        "overlap_ratio": 0.5,
        "dt": 1,
        "deform": True,
        "smooth": True,
        "num_validation_iters": 2,
        "validation_method": "median_velocity",
    }
    frame_a, frame_b = create_pair_shift(shape, u_shift, v_shift)

    piv = process.PIV(shape, **args)

    @benchmark
    def repeat_10():
        for i in range(10):
            piv(frame_a, frame_b)
