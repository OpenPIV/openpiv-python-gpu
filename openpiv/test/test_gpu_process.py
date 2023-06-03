import numpy as np
import pytest
from math import sqrt

import scipy.fft as fft
from scipy.interpolate import interp2d
from scipy.ndimage import distance_transform_edt, shift, map_coordinates
from scipy.signal import correlate2d
from skimage import img_as_ubyte
from skimage.util import random_noise
from imageio.v2 import imread
import pycuda.autoinit
import pycuda.gpuarray as gpuarray

import openpiv.gpu_process as gpu_process
from openpiv.test.test_gpu_misc import generate_array_pair, generate_gpu_array, generate_np_array

# GLOBAL VARIABLES
# datatypes used in gpu_process
DTYPE_i = np.int32
DTYPE_f = np.float32

# dirs
_temp_dir = './temp/'
data_dir = '../data/'

# synthetic image parameters
_image_size_rectangle = (1024, 1024)
_image_size_square = (1024, 512)
_u_shift = 8
_v_shift = -4
_accuracy_tolerance = 0.1
_identity_tolerance = 1e-6
_trim_slice = slice(2, -2, 1)

# test parameters
_test_size_tiny = (8, 8)
_test_size_small = (16, 16)
_test_size_medium = (64, 64)
_test_size_large = (256, 256)
_test_size_super = (1024, 1024)
_test_size_small_stack = (8, 16, 9)


# UTILS
def create_pair_shift(image_size, u_shift, v_shift):
    """Creates a pair of images with a roll/shift """
    frame_a = np.zeros(image_size, dtype=np.int32)
    frame_a = random_noise(frame_a)
    frame_a = img_as_ubyte(frame_a)
    frame_b = shift(frame_a, (v_shift, u_shift), mode='wrap')

    return frame_a.astype(np.int32), frame_b.astype(np.int32)


def create_pair_roll(image_size, roll_shift):
    """Creates a pair of images with a roll/shift """
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
    nearest_neighbour = distance_transform_edt(mask, sampling=spacing, return_distances=False, return_indices=True)
    if f.ndim == 1:
        neighbour_index = nearest_neighbour.squeeze()
    else:
        neighbour_index = tuple(nearest_neighbour[i] for i in range(nearest_neighbour.shape[0]))
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


def get_grid_coords(shape, window_size, spacing):
    x, y = gpu_process.get_field_coords(shape, window_size, spacing)
    x = x[0, :].astype(DTYPE_f)
    y = y[:, 0].astype(DTYPE_f)

    return x, y


def gpu_array(*f_l):
    return [gpuarray.to_gpu(f) for f in f_l]


def np_array(*f_l):
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

            win_np[i * n + j, win_row_idx, win_col_idx] = frame[frame_row_idx, frame_col_idx]

    return win_np


def gaussian_peak(shape, row_peak, col_peak):
    n_windows, ht, wd = shape
    x, y = np.meshgrid(np.arange(wd), np.arange(ht))
    x0 = col_peak
    y0 = row_peak
    a, b = (1, 1)

    correlation = np.empty(shape, dtype=DTYPE_f)
    for i in range(n_windows):
        correlation[i] = np.exp(-((x - x0[i]) ** 2 / (2 * a ** 2) + (y - y0[i]) ** 2 / (2 * b ** 2)))

    return gpuarray.to_gpu(correlation)


def parabolic_peak(shape, row_peak, col_peak):
    n_windows, ht, wd = shape
    x, y = np.meshgrid(np.arange(wd), np.arange(ht))
    x0 = col_peak
    y0 = row_peak
    a, b = (1, 1)

    correlation = np.empty(shape, dtype=DTYPE_f)
    for i in range(n_windows):
        correlation[i] = ((x - x0[i]) ** 2 / (2 * a ** 2) + (y - y0[i]) ** 2 / (2 * b ** 2))

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
            correlation_masked[np.arange(n_windows, ), row, col] = 0

    return correlation_masked


def piv_error():
    pass


def l2_norm_2d(a, b):
    return np.sqrt(np.sum(a ** 2 + b ** 2)) / sqrt(a.size)


# FIXTURES
@pytest.fixture
def correlation_gpu():
    frame_a = imread(data_dir + 'test1/exp1_001_a.bmp').astype(np.float32)
    frame_b = imread(data_dir + 'test1/exp1_001_b.bmp').astype(np.float32)
    frame_a_d = gpuarray.to_gpu(frame_a)
    frame_b_d = gpuarray.to_gpu(frame_b)

    correlation_gpu = gpu_process.CorrelationGPU(frame_a_d, frame_b_d)

    return correlation_gpu


@pytest.fixture
def piv_field_gpu():
    frame_shape = (512, 512)
    window_size = 32
    spacing = 16

    piv_field_gpu = gpu_process.PIVFieldGPU(frame_shape, window_size, spacing)

    return piv_field_gpu


# UNIT TESTS
def test_gpu_gradient():
    u, u_d = generate_cpu_gpu_pair(_test_size_small)
    v, v_d = generate_cpu_gpu_pair(_test_size_small)

    u_y, u_x = np.gradient(u)
    v_y, v_x = np.gradient(v)
    strain_gpu = (gpu_process.gpu_strain(u_d, v_d)).get()

    assert np.array_equal(u_x, strain_gpu[0])
    assert np.array_equal(u_y, strain_gpu[1])
    assert np.array_equal(v_x, strain_gpu[2])
    assert np.array_equal(v_y, strain_gpu[3])


@pytest.mark.parametrize('mask_d', [None, gpuarray.zeros((7, 7), dtype=DTYPE_i)])
def test_gpu_interpolate(mask_d):
    ws0 = 16
    spacing0 = 8
    ws1 = 8
    spacing1 = 4
    n_row0, n_col0 = gpu_process.get_field_shape(_test_size_medium, ws0, spacing0)
    x0, y0 = gpu_process.get_field_coords(_test_size_medium, ws0, spacing0)
    x1, y1 = gpu_process.get_field_coords(_test_size_medium, ws1, spacing1)
    x0 = x0.astype(DTYPE_f)
    y0 = y0.astype(DTYPE_f)
    x1 = x1.astype(DTYPE_f)
    y1 = y1.astype(DTYPE_f)

    f0, f0_d = generate_cpu_gpu_pair((n_row0, n_col0))
    x0_d = gpuarray.to_gpu(x0[0, :])
    x1_d = gpuarray.to_gpu(x1[0, :])
    y0_d = gpuarray.to_gpu(y0[:, 0])
    y1_d = gpuarray.to_gpu(y1[:, 0])

    interp_2d = interp2d(x0[0, :], y0[:, 0], f0)
    f1 = np.flip(interp_2d(x1[0, :], y1[:, 0]), axis=0)  # interp2d returns interpolation results with increasing y

    f1_d = gpu_process.gpu_interpolate(x0_d, y0_d, x1_d, y1_d, f0_d, mask=mask_d)
    f1_gpu = f1_d.get()

    assert np.allclose(f1, f1_gpu, _identity_tolerance)


def test_gpu_ftt_shift():
    correlation_stack, correlation_stack_d = generate_cpu_gpu_pair(_test_size_small_stack)

    shift_stack_cpu = fft.fftshift(correlation_stack, axes=(1, 2))
    shift_stack_gpu = gpu_process.gpu_fft_shift(correlation_stack_d).get()

    assert np.allclose(shift_stack_cpu, shift_stack_gpu, _identity_tolerance)


def test_mask_peak():
    correlation_stack, correlation_stack_d = generate_cpu_gpu_pair(_test_size_small_stack)

    row_peak_d = gpuarray.to_gpu(np.arange(_test_size_small_stack[0], dtype=DTYPE_i))
    col_peak_d = gpuarray.to_gpu(np.arange(_test_size_small_stack[0], dtype=DTYPE_i))

    correlation_stack_masked_d = gpu_process._gpu_mask_peak(correlation_stack_d, row_peak_d, col_peak_d, 2)

    assert np.all(correlation_stack_masked_d.get()[6, 4:9, 4:9] == 0)


def test_mask_rms():
    n_windows, ht, wd = _test_size_small_stack
    correlation_stack, correlation_stack_d = generate_cpu_gpu_pair(_test_size_small_stack)

    corr_peak = np.random.random(_test_size_small_stack[0]).astype(DTYPE_f)
    corr_peak_d = gpuarray.to_gpu(corr_peak)

    a = correlation_stack.reshape((n_windows, ht * wd))
    correlation_stack_masked_cpu = (a * (a < corr_peak.reshape(n_windows, 1) / 2)).reshape(_test_size_small_stack)
    correlation_stack_masked_gpu = gpu_process._gpu_mask_rms(correlation_stack_d, corr_peak_d).get()

    assert np.allclose(correlation_stack_masked_cpu, correlation_stack_masked_gpu, _identity_tolerance)


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
#     assert np.linalg.norm(u[_trim_slice, _trim_slice] - _u_shift) / sqrt(u.size) < _accuracy_tolerance
#     assert np.linalg.norm(-v[_trim_slice, _trim_slice] - _v_shift) / sqrt(u.size) < _accuracy_tolerance


# INTEGRATION TESTS
@pytest.mark.parametrize('image_size', (_image_size_rectangle, _image_size_square))
def test_gpu_piv_fast(image_size):
    """Quick test of the main piv function."""
    frame_a, frame_b = create_pair_shift(image_size, _u_shift, _v_shift)
    args = {'mask': None,
            'window_size_iters': (1, 2),
            'min_window_size': 16,
            'overlap_ratio': 0.5,
            'dt': 1,
            'deform': True,
            'smooth': True,
            'num_validation_iters': 1,
            'validation_method': 'median_velocity',
            }

    x, y, u, v, mask, s2n = gpu_process.gpu_piv(frame_a, frame_b, **args)

    assert np.linalg.norm(u[_trim_slice, _trim_slice] - _u_shift) / sqrt(u.size) < _accuracy_tolerance
    assert np.linalg.norm(-v[_trim_slice, _trim_slice] - _v_shift) / sqrt(u.size) < _accuracy_tolerance


@pytest.mark.parametrize('image_size', (_image_size_rectangle, _image_size_square))
def test_gpu_piv_zero(image_size):
    """Tests that zero-displacement is returned when the images are empty."""
    frame_a = frame_b = np.zeros(image_size, dtype=np.int32)
    args = {'mask': None,
            'window_size_iters': (1, 2),
            'min_window_size': 16,
            'overlap_ratio': 0.5,
            'dt': 1,
            'deform': True,
            'smooth': True,
            'num_validation_iters': 1,
            'validation_method': 'median_velocity',
            }

    x, y, u, v, mask, s2n = gpu_process.gpu_piv(frame_a, frame_b, **args)

    assert np.allclose(u, 0, _identity_tolerance)
    assert np.allclose(v, 0, _identity_tolerance)


def test_extended_search_area():
    """Inputs every s2n method to ensure they don't error out."""
    frame_a, frame_b = create_pair_shift(_image_size_rectangle, _u_shift, _v_shift)
    args = {'mask': None,
            'window_size_iters': (2, 2),
            'min_window_size': 8,
            'overlap_ratio': 0.5,
            'dt': 1,
            'deform': True,
            'smooth': True,
            'num_validation_iters': 2,
            'extend_ratio': 2
            }

    x, y, u, v, mask, s2n = gpu_process.gpu_piv(frame_a, frame_b, **args)

    assert np.linalg.norm(u[_trim_slice, _trim_slice] - _u_shift) / sqrt(u.size) < _accuracy_tolerance
    assert np.linalg.norm(-v[_trim_slice, _trim_slice] - _v_shift) / sqrt(u.size) < _accuracy_tolerance


@pytest.mark.parametrize('s2n_method', ('peak2peak', 'peak2mean', 'peak2energy'))
def test_sig2noise(s2n_method):
    """Inputs every s2n method to ensure they don't error out."""
    frame_a, frame_b = create_pair_shift(_image_size_rectangle, _u_shift, _v_shift)
    args = {'mask': None,
            'window_size_iters': (1, 2, 2),
            'min_window_size': 8,
            'overlap_ratio': 0.5,
            'dt': 1,
            'deform': True,
            'smooth': True,
            'num_validation_iters': 2,
            'validation_method': 'median_velocity',
            'return_s2n': True,
            's2n_method': s2n_method,
            }

    _ = gpu_process.gpu_piv(frame_a, frame_b, **args)


@pytest.mark.parametrize('subpixel_method', ('gaussian', 'centroid', 'parabolic'))
def test_subpixel_peak(subpixel_method):
    """Inputs every s2n method to ensure they don't error out."""
    frame_a, frame_b = create_pair_shift(_image_size_rectangle, _u_shift, _v_shift)
    args = {'mask': None,
            'window_size_iters': (1, 2, 2),
            'min_window_size': 8,
            'overlap_ratio': 0.5,
            'dt': 1,
            'deform': True,
            'smooth': True,
            'num_validation_iters': 2,
            'validation_method': 'median_velocity',
            'subpixel_method': subpixel_method,
            }

    _ = gpu_process.gpu_piv(frame_a, frame_b, **args)


# s2n must not cause invalid numbers to be passed to smoothn.
@pytest.mark.parametrize('validation_method', ('s2n', 'mean_velocity', 'median_velocity', 'rms_velocity'))
def test_validation(validation_method):
    """Inputs every s2n method to ensure they don't error out."""
    frame_a, frame_b = create_pair_shift(_image_size_rectangle, _u_shift, _v_shift)
    args = {'mask': None,
            'window_size_iters': (1, 2, 2),
            'min_window_size': 8,
            'overlap_ratio': 0.5,
            'dt': 1,
            'deform': True,
            'smooth': True,
            'num_validation_iters': 2,
            'validation_method': validation_method,
            }

    _ = gpu_process.gpu_piv(frame_a, frame_b, **args)


# sweep the input variables to ensure everything is same
@pytest.mark.parametrize('window_size_iters', [1, (1, 1), (1, 1, 1), (1, 1, 2), (1, 2, 2), (2, 2, 2), (1, 2, 1)])
@pytest.mark.parametrize('min_window_size', [8, 16])
@pytest.mark.parametrize('num_validation_iters', [0, 1, 2])
def test_gpu_piv_py(window_size_iters, min_window_size, num_validation_iters, ndarrays_regression):
    """This test checks that the output remains the same."""
    frame_a = imread('../data/test1/exp1_001_a.bmp')
    frame_b = imread('../data/test1/exp1_001_b.bmp')
    args = {'mask': None,
            'window_size_iters': window_size_iters,
            'min_window_size': min_window_size,
            'overlap_ratio': 0.5,
            'dt': 1,
            'deform': True,
            'smooth': True,
            'num_validation_iters': num_validation_iters,
            'validation_method': 'median_velocity',
            'smoothing_par': 0.5,
            'center_field': False
            }

    x, y, u, v, mask, s2n = gpu_process.gpu_piv(frame_a, frame_b, **args)

    ndarrays_regression.check({'u': u, 'v': v})


# BENCHMARKS
@pytest.mark.parametrize('image_size', [(1024, 1024), (2048, 2048)])
@pytest.mark.parametrize('window_size_iters,min_window_size', [((1, 2), 16), ((1, 2, 2), 8)])
def test_gpu_piv_benchmark(benchmark, image_size, window_size_iters, min_window_size):
    """Benchmarks the PIV function."""
    frame_a, frame_b = create_pair_shift(image_size, _u_shift, _v_shift)
    args = {'mask': None,
            'window_size_iters': window_size_iters,
            'min_window_size': min_window_size,
            'overlap_ratio': 0.5,
            'dt': 1,
            'deform': True,
            'smooth': True,
            'num_validation_iters': 2,
            'validation_method': 'median_velocity',
            }

    benchmark(gpu_process.gpu_piv, frame_a, frame_b, **args)


# UNIT TESTS
def test_correlation_gpu_free_frame_data():
    pass


def test_correlation_gpu_sig2noise_d():
    pass


def test_correlation_gpu_init_fft_shape():
    pass


def test_correlation_gpu_stack_iw():
    pass


def test_correlation_gpu_correlate_windows():
    pass


def test_correlation_gpu_check_zero_correlation():
    """Cross-correlation of windows should return center if correlation is zero."""
    pass


def test_correlation_gpu_get_displacement():
    pass


def test_correlation_gpu_get_s2n():
    pass


def test_correlation_gpu_get_second_peak_height():
    pass


def test_correlation_gpu_coords():
    pass


def test_correlation_gpu_s2n():
    pass


def test_correlation_gpu_free_data():
    pass


def test_piv_field_gpu():
    pass


def test_piv_field_gpu_get_mask():
    pass


def test_piv_field_gpu_coords():
    pass


def test_piv_field_gpu_grid_coords_d():
    pass


def test_piv_field_gpu_center_buffer():
    pass


def test_piv_gpu_init_fields():
    pass


def test_piv_gpu_mask_field():
    pass


def test_piv_gpu_mask_frame():
    pass


def test_piv_gpu_get_corr_arguments():
    pass


def test_piv_gpu_update_values():
    pass


def test_piv_gpu_validate_fields():
    pass


def test_piv_gpu_gpu_replace_vectors():
    pass


def test_piv_gpu_get_next_iteration_prediction():
    pass


def test_piv_gpu_get_mask_k():
    pass


def test_piv_gpu_log_residual():
    pass


def test_get_field_shape():
    pass


# @pytest.mark.parametrize('center_field', [True, False])
def test_get_field_coords():
    pass


def test_gpu_strain():
    shape = (16, 16)

    u, u_d = generate_array_pair(shape, magnitude=2, offset=-1)
    v, v_d = generate_array_pair(shape, magnitude=2, offset=-1, seed=1)

    u_y, u_x = np.gradient(u)
    v_y, v_x = np.gradient(v)
    strain_gpu = (gpu_process.gpu_strain(u_d, v_d)).get()

    assert np.array_equal(u_x, strain_gpu[0])
    assert np.array_equal(u_y, strain_gpu[1])
    assert np.array_equal(v_x, strain_gpu[2])
    assert np.array_equal(v_y, strain_gpu[3])


@pytest.mark.parametrize('shape', [(16, 16, 16), (15, 12, 14), (15, 11, 13)])
def test_gpu_fft_shift(shape):
    shape = (16, 16, 16)

    correlation, correlation_d = generate_array_pair(shape, magnitude=2, offset=-1)

    correlation_shifted_np = fft.fftshift(correlation, axes=(1, 2))
    correlation_shifted_gpu = gpu_process.gpu_fft_shift(correlation_d).get()

    assert np.array_equal(correlation_shifted_gpu, correlation_shifted_np)


@pytest.mark.parametrize('spacing', [1])
def test_gpu_interpolate(spacing):
    shape = (16, 16)
    window_size0 = 4
    spacing0 = 2
    window_size1 = 2
    spacing1 = 1
    f0_shape = gpu_process.get_field_shape(shape, window_size0, spacing0)

    f0, f0_d = generate_array_pair(f0_shape, magnitude=2, offset=-1)

    x0, y0 = get_grid_coords(shape, window_size0, spacing0)
    x1, y1 = get_grid_coords(shape, window_size1, spacing1)
    x0_d, y0_d = gpu_array(x0, y0)
    x1_d, y1_d = gpu_array(x1, y1)

    # interp2d returns interpolation results with increasing y.
    interp_2d = interp2d(x0, y0, f0)
    f1 = np.flip(interp_2d(x1, y1), axis=0)

    f1_gpu = gpu_process.gpu_interpolate(x0_d, y0_d, x1_d, y1_d, f0_d).get()

    assert np.allclose(f1_gpu, f1)


# TODO need to test various spacings
# @pytest.param()
def test_gpu_interpolate_mask():
    shape = (16, 16)
    window_size0 = 4
    spacing0 = 2
    window_size1 = 2
    spacing1 = 1
    f0_shape = gpu_process.get_field_shape(shape, window_size0, spacing0)

    f0, f0_d = generate_array_pair(f0_shape, magnitude=2, offset=-1)
    mask, mask_d = generate_array_pair(f0_shape, magnitude=2, d_type=DTYPE_i, seed=1)

    x0, y0 = get_grid_coords(shape, window_size0, spacing0)
    x1, y1 = get_grid_coords(shape, window_size1, spacing1)
    x0_d, y0_d = gpu_array(x0, y0)
    x1_d, y1_d = gpu_array(x1, y1)

    f1 = interp_mask_np(x0, y0, x1, y1, f0, mask)
    f1_gpu = gpu_process.gpu_interpolate(x0_d, y0_d, x1_d, y1_d, f0_d, mask=mask_d).get()

    assert np.allclose(f1_gpu, f1)


@pytest.mark.parametrize('ws_iters', [(1, 2), (1, 1, 2)])
@pytest.mark.parametrize('min_window_size', [8, 16])
def test_get_window_sizes(ws_iters, min_window_size, data_regression):
    window_size_l = list(gpu_process._get_window_sizes(ws_iters, min_window_size))

    data_regression.check({'window_size_l': window_size_l})


@pytest.mark.parametrize('window_size', [16, 8])
@pytest.mark.parametrize('overlap_ratio', [0.5, 0.9])
def test_get_spacing(window_size, overlap_ratio, data_regression):
    spacing = gpu_process._get_spacing(window_size, overlap_ratio)

    data_regression.check({'spacing': spacing})


@pytest.mark.parametrize('frame_mask', [True, False])
def test_get_field_mask(frame_mask, ndarrays_regression):
    shape = (16, 16)
    ht, wd = shape

    x, y = np.meshgrid(np.arange(ht), np.arange(wd))
    frame_mask = generate_np_array(shape, magnitude=2, d_type=DTYPE_i, seed=0) if frame_mask else None
    mask = gpu_process._get_field_mask(x, y, frame_mask)

    ndarrays_regression.check({'mask': mask})


@pytest.mark.parametrize('frame_size, window_size, spacing, buffer', [(64, 8, 4, 0), (66, 8, 4, 1), (66, 7, 4, 2)])
def test_get_center_buffer(frame_size, window_size, spacing, buffer):
    buffer_x, buffer_y = gpu_process._get_center_buffer((frame_size, frame_size), window_size, spacing)

    assert buffer_x == buffer


@pytest.mark.parametrize('window_size', [4, 8])
@pytest.mark.parametrize('spacing', [4, 8])
@pytest.mark.parametrize('buffer', [-1, 1])
@pytest.mark.parametrize('pass_shift', [True, False])
def test_gpu_window_slice(window_size, spacing, buffer, pass_shift):
    frame_shape = (16, 16)
    window_size = 8
    spacing = 4
    field_shape = gpu_process.get_field_shape(frame_shape, window_size, spacing)
    buffer = 0

    frame, frame_d = generate_array_pair(frame_shape, magnitude=2, offset=-1)
    shift_d = gpuarray.to_gpu(np.ones((2, *field_shape), dtype=DTYPE_f)) if pass_shift else None

    win_np = window_slice_np(frame, field_shape, window_size, spacing, buffer)
    win_gpu = gpu_process._gpu_window_slice(frame_d, field_shape, window_size, spacing, buffer, shift=shift_d).get()

    assert np.array_equal(win_gpu, win_np)


@pytest.mark.parametrize('dt', [-1, 0, 1])
def test_gpu_window_slice_shift(dt):
    """Test window translation using a single window centered on the frame."""
    frame_shape = (16, 16)
    field_shape = (1, 1)
    ht, wd = frame_shape
    ws = 16
    spacing = 4
    buffer = 0
    u = 1.4
    v = 2.5

    frame, frame_d = generate_array_pair(frame_shape, magnitude=2, offset=-1)

    shift_d = gpuarray.to_gpu(np.array([u, v], dtype=DTYPE_f).reshape(2, *field_shape))
    # Apply the strain shift directly to the frame.
    win_np = np.empty((1, ws, ws), dtype=DTYPE_f)
    x, y = np.meshgrid(np.arange(ht), np.arange(wd))
    x = x + u * dt
    y = y + v * dt
    coordinates = [y, x]
    win_np[:, :, :] = map_coordinates(frame[:, :], coordinates, order=1)
    win_gpu = gpu_process._gpu_window_slice(frame_d, field_shape, ws, spacing, buffer, dt=dt, shift=shift_d).get()

    assert np.allclose(win_gpu, win_np, atol=1e-6)


@pytest.mark.parametrize('dt', [-1, 0, 1])
def test_gpu_window_slice_strain(dt):
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

    frame, frame_d = generate_array_pair(frame_shape, magnitude=2, offset=-1)

    shift_d = gpuarray.zeros((2, *field_shape), dtype=DTYPE_f)
    strain_d = gpuarray.to_gpu(np.array([u_x, u_y, v_x, v_y], dtype=DTYPE_f).reshape(4, *field_shape))
    # Apply the strain deformation directly to the frame.
    win_np = np.empty((1, ws, ws), dtype=DTYPE_f)
    x, y = np.meshgrid(np.arange(ht), np.arange(wd))
    r_x = (np.arange(wd) - wd / 2 + 0.5).reshape((1, wd))
    r_y = (np.arange(ht) - ht / 2 + 0.5).reshape((ht, 1))
    x = x + (u_x * r_x + u_y * r_y) * dt
    y = y + (v_x * r_x + v_y * r_y) * dt
    coordinates = [y, x]
    win_np[:, :, :] = map_coordinates(frame[:, :], coordinates, order=1)
    win_gpu = gpu_process._gpu_window_slice(frame_d, field_shape, ws, spacing, buffer, dt=dt, shift=shift_d,
                                            strain=strain_d).get()

    assert np.allclose(win_gpu, win_np, atol=1e-6)


def test_gpu_normalize_intensity():
    shape = (16, 16, 16)
    n_windows, ht, wd = shape

    win, win_d = generate_array_pair(shape, magnitude=2, offset=-1)

    mean = np.mean(win.reshape((n_windows, ht * wd)), axis=1).reshape((n_windows, 1, 1))
    win_norm_np = win - mean
    win_norm_gpu = gpu_process._gpu_normalize_intensity(win_d).get()

    assert np.allclose(win_norm_gpu, win_norm_np)


@pytest.mark.parametrize('offset', [0, 1, 2])
def test_gpu_zero_pad(offset):
    shape = (16, 16, 16)
    fft_shape = (32, 32)
    n_windows, ht, wd = shape
    fft_ht, fft_wd = fft_shape

    win, win_d = generate_array_pair(shape, magnitude=2, offset=-1)

    win_zp_np = np.zeros((n_windows, fft_ht, fft_wd), dtype=DTYPE_f)
    win_zp_np[:, offset:offset + ht, offset:offset + wd] = win
    win_zp_gpu = gpu_process._gpu_zero_pad(win_d, fft_shape, offset).get()

    assert np.array_equal(win_zp_gpu, win_zp_np)


@pytest.mark.parametrize('shape', [(16, 16, 16), (15, 12, 14), (15, 11, 13)])
def test_cross_correlate(shape: tuple):
    _, m, n = shape

    win_a, win_a_d = generate_array_pair(shape, magnitude=2, offset=-1)
    win_b, win_b_d = generate_array_pair(shape, magnitude=2, offset=-1, seed=1)

    correlation = np.empty(shape, dtype=DTYPE_f)
    for i in range(shape[0]):
        # The scipy definition of the cross-correlation reverses the output.
        correlation[i, :, :] = correlate2d(win_b[i, :, :], win_a[i, :, :], mode='full', boundary='wrap')[m - 1:, n - 1:]
    correlation_np = correlation
    correlation_d = gpu_process._gpu_cross_correlate(win_a_d, win_b_d)
    correlation_gpu = correlation_d.get()

    assert np.allclose(correlation_gpu, correlation_np, atol=1e-5)


def test_gpu_window_index_f():
    n_windows = 16
    index_size = 16 * 16
    shape = (n_windows, index_size)

    data, data_d = generate_array_pair(shape, magnitude=2, offset=-1)
    indices, indices_d = generate_array_pair(n_windows, magnitude=index_size, d_type=DTYPE_i, seed=1)

    values_np = data[np.arange(n_windows), indices].squeeze()
    values_gpu = gpu_process._gpu_window_index_f(data_d, indices_d).get()

    assert np.array_equal(values_gpu, values_np)


def test_find_peak():
    shape = (16, 16, 16)
    n_windows, ht, wd = shape

    correlation, correlation_d = generate_array_pair(shape, magnitude=2, offset=-1)

    peak_idx_np = np.argmax(correlation.reshape(n_windows, ht * wd), axis=1).astype(dtype=DTYPE_i)
    peak_idx_gpu = gpu_process._find_peak(correlation_d).get()

    assert np.array_equal(peak_idx_gpu, peak_idx_np)


def test_get_peak():
    shape = (16, 16, 16)
    n_windows, ht, wd = shape

    correlation, correlation_d = generate_array_pair(shape, magnitude=2, offset=-1)
    peak_idx, peak_idx_d = generate_array_pair((n_windows,), magnitude=ht * wd, d_type=DTYPE_i, seed=1)

    corr_peak_np = correlation.reshape(n_windows, ht * wd)[np.arange(n_windows), peak_idx]
    corr_peak_gpu = gpu_process._get_peak(correlation_d, peak_idx_d).get()

    assert np.array_equal(corr_peak_gpu, corr_peak_np)


@pytest.mark.parametrize('method, tol', [('gaussian', 0.1), ('parabolic', 0.1), ('centroid', 0.25)])
def test_gpu_subpixel_approximation(method, tol):
    shape = (16, 16, 16)
    n_windows, ht, wd = shape

    row_sp = generate_np_array(n_windows, magnitude=ht - 2, offset=1)
    col_sp = generate_np_array(n_windows, magnitude=wd - 2, offset=1, seed=1)
    row_peak_d, col_peak_d = gpu_array(np.round(row_sp).astype(DTYPE_i), np.round(col_sp).astype(DTYPE_i))

    # Create the distribution.
    if method == 'parabolic':
        correlation_d = parabolic_peak(shape, row_sp, col_sp)
    else:
        correlation_d = gaussian_peak(shape, row_sp, col_sp)

    row_sp_d, col_sp_d = gpu_process._gpu_subpixel_approximation(correlation_d, row_peak_d, col_peak_d, method)
    row_sp_gpu, col_sp_gpu = np_array(row_sp_d, col_sp_d)

    assert np.all(np.abs(row_sp_gpu - row_sp) <= tol)
    assert np.all(np.abs(col_sp_gpu - col_sp) <= tol)


def test_peak2mean():
    shape = (16, 16, 16)
    n_windows, ht, wd = shape
    size = ht * wd

    correlation, correlation_d = generate_array_pair(shape)
    corr_peak, corr_peak_d = generate_array_pair((n_windows,), seed=1)

    correlation_masked = correlation * (correlation < corr_peak.reshape(n_windows, 1, 1) / 2)
    sig2noise_np = 2 * np.log10(corr_peak / np.mean(correlation_masked.reshape(n_windows, size), axis=1))
    sig2noise_gpu = gpu_process._peak2mean(correlation_d, corr_peak_d).get()

    assert np.allclose(sig2noise_gpu, sig2noise_np)


def test_peak2energy():
    shape = (16, 16, 16)
    n_windows, ht, wd = shape
    size = ht * wd

    correlation, correlation_d = generate_array_pair(shape)
    corr_peak, corr_peak_d = generate_array_pair((n_windows,), seed=1)

    sig2noise_np = 2 * np.log10(corr_peak / np.mean(correlation.reshape(n_windows, size), axis=1))
    sig2noise_gpu = gpu_process._peak2energy(correlation_d, corr_peak_d).get()

    assert np.allclose(sig2noise_gpu, sig2noise_np)


def test_peak2peak():
    shape = (16, 16, 16)
    n_windows, ht, wd = shape

    corr_peak1, corr_peak1_d = generate_array_pair((n_windows,))
    corr_peak2, corr_peak2_d = generate_array_pair((n_windows,), seed=1)

    sig2noise_np = np.log10(corr_peak1 / corr_peak2)
    sig2noise_gpu = gpu_process._peak2peak(corr_peak1_d, corr_peak2_d).get()

    assert np.allclose(sig2noise_gpu, sig2noise_np)


@pytest.mark.parametrize('width', [1, 2])
def test_gpu_mask_peak(width):
    shape = (16, 16, 16)
    n_windows, ht, wd = shape

    correlation, correlation_d = generate_array_pair(shape)
    row_peak, row_peak_d = generate_array_pair((n_windows,), magnitude=ht, d_type=DTYPE_i, seed=1)
    col_peak, col_peak_d = generate_array_pair((n_windows,), magnitude=wd, d_type=DTYPE_i, seed=2)

    correlation_masked_np = mask_peak_np(correlation, row_peak, col_peak, width)
    correlation_masked_gpu = gpu_process._gpu_mask_peak(correlation_d, row_peak_d, col_peak_d, width).get()

    assert np.array_equal(correlation_masked_gpu, correlation_masked_np)


def test_gpu_mask_rms():
    shape = (16, 16, 16)
    n_windows, ht, wd = shape

    correlation, correlation_d = generate_array_pair(shape, magnitude=1)
    corr_peak, corr_peak_d = generate_array_pair((n_windows,), magnitude=1, seed=1)

    correlation_masked_np = correlation * (correlation < corr_peak.reshape(n_windows, 1, 1) / 2)
    correlation_masked_gpu = gpu_process._gpu_mask_rms(correlation_d, corr_peak_d).get()

    assert np.array_equal(correlation_masked_gpu, correlation_masked_np)


def test_get_shift():
    shape = (16, 16)

    u, u_d = generate_array_pair(shape, magnitude=2, offset=-1)
    v, v_d = generate_array_pair(shape, magnitude=2, offset=-1, seed=1)

    shift_np = np.stack([u, v], axis=0)
    shift_gpu = gpu_process._get_shift(u_d, v_d).get()

    assert np.array_equal(shift_gpu, shift_np)


def test_gpu_update_field():
    shape = (16, 16)

    dp, dp_d = generate_array_pair(shape, magnitude=2, offset=-1)
    peak, peak_d = generate_array_pair(shape, magnitude=2, offset=-1, seed=1)
    mask, mask_d = generate_array_pair(shape, magnitude=1, d_type=DTYPE_i, seed=2)

    f_np = (dp + peak) * (mask == 0)
    f_gpu = gpu_process._gpu_update_field(dp_d, peak_d, mask_d).get()

    assert np.array_equal(f_np, f_gpu)


def test_interpolate_replace():
    shape = (16, 16)
    window_size0 = 4
    spacing0 = 2
    window_size1 = 2
    spacing1 = 1
    f0_shape = gpu_process.get_field_shape(shape, window_size0, spacing0)
    f1_shape = gpu_process.get_field_shape(shape, window_size1, spacing1)

    f0, f0_d = generate_array_pair(f0_shape, magnitude=2, offset=-1)
    f1, f1_d = generate_array_pair(f1_shape, magnitude=2, offset=-1)
    mask, mask_d = generate_array_pair(f0_shape, magnitude=2, d_type=DTYPE_i, seed=1)
    val_locations, val_locations_d = generate_array_pair(f1_shape, magnitude=2, d_type=DTYPE_i, seed=2)

    x0, y0 = get_grid_coords(shape, window_size0, spacing0)
    x1, y1 = get_grid_coords(shape, window_size1, spacing1)
    x0_d, y0_d = gpu_array(x0, y0)
    x1_d, y1_d = gpu_array(x1, y1)

    f1 = interp_mask_np(x0, y0, x1, y1, f0, mask) * val_locations + f1 * (val_locations == 0)
    f1_gpu = gpu_process._interpolate_replace(x0_d, y0_d, x1_d, y1_d, f0_d, f1_d, val_locations_d, mask=mask_d).get()

    assert np.allclose(f1_gpu, f1)


# TODO check that input validation works


# INTEGRATION TESTS
@pytest.mark.integtest
def test_gpu_piv_fast():
    """Quick test of the main piv function."""
    frame_size = (512, 512)
    u_shift = 8
    v_shift = -4
    trim_slice = slice(2, -2, 1)
    args = {'mask': None,
            'window_size_iters': (1, 2),
            'min_window_size': 16,
            'overlap_ratio': 0.5,
            'dt': 1,
            'deform': True,
            'smooth': True,
            'num_validation_iters': 1,
            'validation_method': 'median_velocity',
            }

    frame_a, frame_b = create_pair_shift(frame_size, u_shift, v_shift)

    x, y, u, v, mask, s2n = gpu_process.gpu_piv(frame_a, frame_b, **args)

    assert np.linalg.norm(u[trim_slice, trim_slice] - u_shift) / sqrt(u.size) < 0.1
    assert np.linalg.norm(-v[trim_slice, trim_slice] - v_shift) / sqrt(u.size) < 0.1


@pytest.mark.integtest
@pytest.mark.parametrize('window_size_iters', [1, (1, 1), (1, 1, 1), (1, 1, 2), (1, 2, 2), (2, 2, 2), (1, 2, 1)])
@pytest.mark.parametrize('min_window_size', [8, 16])
@pytest.mark.parametrize('num_validation_iters', [0, 1, 2])
def test_gpu_piv_py(window_size_iters, min_window_size, num_validation_iters, ndarrays_regression):
    """This test checks that the output remains the same."""
    frame_a = imread(data_dir + 'test1/exp1_001_a.bmp')
    frame_b = imread(data_dir + 'test1/exp1_001_b.bmp')
    args = {'mask': None,
            'window_size_iters': window_size_iters,
            'min_window_size': min_window_size,
            'overlap_ratio': 0.5,
            'dt': 1,
            'deform': True,
            'smooth': True,
            'num_validation_iters': num_validation_iters,
            'validation_method': 'median_velocity',
            'smoothing_par': 0.5,
            'center_field': False
            }

    x, y, u, v, mask, s2n = gpu_process.gpu_piv(frame_a, frame_b, **args)

    ndarrays_regression.check({'u': u, 'v': v})


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
    args = {'mask': None,
            'window_size_iters': (2, 2),
            'min_window_size': 8,
            'overlap_ratio': 0.5,
            'dt': 1,
            'deform': True,
            'smooth': True,
            'num_validation_iters': 2,
            'extend_ratio': 2
            }

    frame_a, frame_b = create_pair_shift(frame_size, u_shift, v_shift)

    x, y, u, v, mask, s2n = gpu_process.gpu_piv(frame_a, frame_b, **args)

    assert np.linalg.norm(u[trim_slice, trim_slice] - u_shift) / sqrt(u.size) < 0.1
    assert np.linalg.norm(-v[trim_slice, trim_slice] - v_shift) / sqrt(u.size) < 0.1


# BENCHMARKS
@pytest.mark.parametrize('frame_shape', [(1024, 1024), (2048, 2048)])
@pytest.mark.parametrize('window_size_iters, min_window_size', [((1, 2), 16), ((1, 2, 2), 8)])
def test_gpu_piv_benchmark(benchmark, frame_shape, window_size_iters, min_window_size):
    """Benchmarks the PIV function."""
    u_shift = 8
    v_shift = -4
    args = {'mask': None,
            'window_size_iters': window_size_iters,
            'min_window_size': min_window_size,
            'overlap_ratio': 0.5,
            'dt': 1,
            'deform': True,
            'smooth': True,
            'num_validation_iters': 2,
            'validation_method': 'median_velocity',
            }

    frame_a, frame_b = create_pair_shift(frame_shape, u_shift, v_shift)

    benchmark(gpu_process.gpu_piv, frame_a, frame_b, **args)


def test_gpu_piv_benchmark_oop(benchmark):
    """Benchmarks the PIV speed with the objected-oriented interface."""
    shape = (1024, 1024)
    u_shift = 8
    v_shift = -4
    args = {'mask': None,
            'window_size_iters': (1, 2, 2),
            'min_window_size': 8,
            'overlap_ratio': 0.5,
            'dt': 1,
            'deform': True,
            'smooth': True,
            'num_validation_iters': 2,
            'validation_method': 'median_velocity',
            }
    frame_a, frame_b = create_pair_shift(shape, u_shift, v_shift)

    piv_gpu = gpu_process.PIVGPU(shape, **args)

    @benchmark
    def repeat_10():
        for i in range(10):
            piv_gpu(frame_a, frame_b)
