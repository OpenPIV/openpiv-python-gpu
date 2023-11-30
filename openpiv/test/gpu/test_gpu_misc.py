"""Test module for misc.py."""

import numpy as np
import pytest

import pycuda.gpuarray as gpuarray

import gpu.misc as gpu_misc
import gpu.process as gpu_process

DTYPE_i = np.int32
DTYPE_f = np.float32


# UTILS
def interp_mask_np(x0, y0, x1, y1, f0, mask):
    # Get the interpolating coordinate
    ht, wd = mask.shape
    m = y1.size
    offset_x = x0[0]
    offset_y = y0[0]
    spacing_x = x0[1] - offset_x
    spacing_y = y0[1] - offset_y
    mask = mask.astype(bool)

    x = ((x1 - offset_x) / spacing_x).astype(DTYPE_f)
    y = ((y1 - offset_y) / spacing_y).astype(DTYPE_f)

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


# UNIT TESTS
@pytest.mark.parametrize("d_type", [DTYPE_i, DTYPE_f])
def test_gpu_logical_or(d_type, array_pair, boolean_array_pair):
    shape = (16, 16)

    f1, f1_d = boolean_array_pair(shape)
    f2, f2_d = boolean_array_pair(shape, seed=1)

    f_out = np.logical_or(f1, f2)
    f_out_gpu = gpu_misc.gpu_logical_or(f1_d, f2_d).get()

    assert np.array_equal(f_out_gpu, f_out)


@pytest.mark.parametrize("d_type", [DTYPE_i, DTYPE_f])
def test_gpu_mask(d_type, array_pair, boolean_array_pair):
    shape = (16, 16)

    f, f_d = array_pair(shape, center=1.0, d_type=d_type)
    mask, mask_d = boolean_array_pair(shape, seed=1)

    f_masked = f * (1 - mask)
    f_masked_gpu = gpu_misc.gpu_mask(f_d, mask_d).get()

    assert np.array_equal(f_masked_gpu, f_masked)


@pytest.mark.parametrize("d_type", [DTYPE_i, DTYPE_f])
def test_gpu_scalar_mod_i(d_type, array_pair):
    shape = (16, 16)
    m = 3

    f, f_d = array_pair(shape, center=5, half_width=5, d_type=DTYPE_i)

    i = f // m
    r = f % m
    i_d, r_d = gpu_misc.gpu_scalar_mod(f_d, m)
    i_gpu = i_d.get()
    r_gpu = r_d.get()

    assert np.array_equal(i_gpu, i)
    assert np.array_equal(r_gpu, r)


def test_gpu_replace_nan_f(np_array):
    shape = (16, 16)

    f = np_array(shape, d_type=DTYPE_f)
    f[f < 0.25] = np.nan
    f[f > 0.75] = np.inf
    f_d = gpuarray.to_gpu(f)

    f_finite = np.nan_to_num(f, nan=0, posinf=np.inf)
    gpu_misc.gpu_remove_nan(f_d)
    f_finite_gpu = f_d.get()

    assert np.array_equal(f_finite_gpu, f_finite)


@pytest.mark.parametrize("d_type", [DTYPE_i, DTYPE_f])
def test_gpu_replace_negative_f(d_type, array_pair):
    shape = (16, 16)

    f, f_d = array_pair(shape, center=0.0, half_width=2, d_type=d_type)

    f[f < 0] = 0
    gpu_misc.gpu_remove_negative(f_d)
    f_positive_gpu = f_d.get()

    assert np.array_equal(f_positive_gpu, f)


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
    f1_gpu = gpu_misc.gpu_interpolate(x0_d, y0_d, x1_d, y1_d, f0_d, mask=mask_d).get()

    assert np.allclose(f1_gpu, f1)


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
    f1_gpu = gpu_misc.interpolate_replace(
        x0_d, y0_d, x1_d, y1_d, f0_d, f1_d, val_locations_d, mask=mask_d
    ).get()

    assert np.allclose(f1_gpu, f1)
