"""Test module for validation.py."""
from math import log10
from functools import partial

import numpy as np
import pytest
from pycuda import gpuarray

from openpiv.gpu import validation, DTYPE_i, DTYPE_f


# UTILS
def median_np(f_neighbours, neighbours_present):
    f_neighbours[neighbours_present == 0] = np.nan
    f_median = np.nanmedian(f_neighbours, axis=2)
    f_median = np.nan_to_num(f_median, copy=False)

    return f_median


def mean_np(f_neighbours, neighbours_present):
    denominator = np.sum(neighbours_present, axis=2).astype(DTYPE_f)
    f_mean = np.sum(f_neighbours, axis=2) / (denominator + (denominator == 0.0))

    return f_mean


def median_residual_np(f_median, f_neighbours, neighbours_present):
    f_neighbours[neighbours_present == 0] = np.nan
    f_median_residual = np.nanmedian(
        np.abs(f_neighbours - f_median.reshape(*f_median.shape, 1)), axis=2
    )
    f_median_residual = np.nan_to_num(f_median_residual, copy=False)

    return f_median_residual


def mean_residual_np(f_mean, f_neighbours, neighbours_present):
    denominator = np.sum(neighbours_present, axis=2).astype(DTYPE_f)
    numerator = np.sum(np.abs(f_neighbours - f_mean.reshape(*f_mean.shape, 1)), axis=2)
    f_mean_residual = numerator / (denominator + (denominator == 0.0))

    return f_mean_residual


def rms_np(f_mean, f_neighbours, neighbours_present):
    denominator = np.sum(neighbours_present, axis=2).astype(DTYPE_f)
    numerator = np.sum((f_neighbours - f_mean.reshape(*f_mean.shape, 1)) ** 2, axis=2)
    f_rms_residual = np.sqrt(numerator / (denominator + (denominator == 0.0)))

    return f_rms_residual


def median_residual_vec2d_np(
    u_median, v_median, u_neighbours, v_neighbours, neighbours_present
):
    u_neighbours[neighbours_present == 0] = np.nan
    v_neighbours[neighbours_present == 0] = np.nan
    median_residual = np.nanmedian(
        np.hypot(
            u_neighbours - u_median.reshape(*u_median.shape, 1),
            v_neighbours - v_median.reshape(*v_median.shape, 1),
        ),
        axis=2,
    )
    median_residual = np.nan_to_num(median_residual, copy=False)

    return median_residual


def mean_residual_vec2d_np(
    u_mean, v_mean, u_neighbours, v_neighbours, neighbours_present
):
    denominator = np.sum(neighbours_present, axis=2).astype(DTYPE_f)
    numerator = np.sum(
        np.hypot(
            u_neighbours - u_mean.reshape(*u_mean.shape, 1),
            v_neighbours - v_mean.reshape(*v_mean.shape, 1),
        ),
        axis=2,
    )
    mean_residual = numerator / (denominator + (denominator == 0.0))

    return mean_residual


def rms_vec2d_np(u_mean, v_mean, u_neighbours, v_neighbours, neighbours_present):
    denominator = np.sum(neighbours_present, axis=2).astype(DTYPE_f)
    numerator = np.sum(
        (u_neighbours - u_mean.reshape(*u_mean.shape, 1)) ** 2
        + (v_neighbours - v_mean.reshape(*v_mean.shape, 1)) ** 2,
        axis=2,
    )
    rms_residual = np.sqrt(numerator / (denominator + (denominator == 0.0)))

    return rms_residual


def find_neighbours_np(shape, mask):
    neighbours_present = np.zeros((*shape, 8), dtype=DTYPE_i)

    neighbours_present[1:, 1:, 0] = mask[:-1, :-1] == 0  # top left
    neighbours_present[1:, :, 1] = mask[:-1, :] == 0  # top center
    neighbours_present[1:, :-1, 2] = mask[:-1, 1:] == 0  # top right
    neighbours_present[:, 1:, 3] = mask[:, :-1] == 0  # center left
    neighbours_present[:, :-1, 4] = mask[:, 1:] == 0  # center right
    neighbours_present[:-1, 1:, 5] = mask[1:, :-1] == 0  # bottom left
    neighbours_present[:-1, :, 6] = mask[1:, :] == 0  # bottom center
    neighbours_present[:-1, :-1, 7] = mask[1:, 1:] == 0  # bottom right

    return neighbours_present


def get_neighbours_np(f, neighbours_present):
    f_neighbours = np.zeros(neighbours_present.shape, dtype=DTYPE_f)

    f_neighbours[1:, 1:, 0] = neighbours_present[1:, 1:, 0] * f[:-1, :-1]  # top left
    f_neighbours[1:, :, 1] = neighbours_present[1:, :, 1] * f[:-1, :]  # top center
    f_neighbours[1:, :-1, 2] = neighbours_present[1:, :-1, 2] * f[:-1, 1:]  # top right
    f_neighbours[:, 1:, 3] = neighbours_present[:, 1:, 3] * f[:, :-1]  # center left
    f_neighbours[:, :-1, 4] = neighbours_present[:, :-1, 4] * f[:, 1:]  # center right
    f_neighbours[:-1, 1:, 5] = (
        neighbours_present[:-1, 1:, 5] * f[1:, :-1]
    )  # bottom left
    f_neighbours[:-1, :, 6] = neighbours_present[:-1, :, 6] * f[1:, :]  # bottom center
    f_neighbours[:-1, :-1, 7] = (
        neighbours_present[:-1, :-1, 7] * f[1:, 1:]
    )  # bottom right

    return f_neighbours


# UNIT TESTS
def test_validation_replace_vectors(gpu_array, boolean_array_pair):
    shape = (16, 16)

    f = gpu_array(shape, center=0.0, half_width=1.0)
    val_locations, val_locations_d = boolean_array_pair(shape, seed=1)
    zeros = gpuarray.zeros(shape, dtype=DTYPE_f)
    validation_gpu = validation.Validation(shape)
    validation_gpu._f = [f]
    validation_gpu._num_fields = 1
    validation_gpu.val_locations = val_locations_d

    f_d = validation_gpu.replace_vectors(zeros)
    f = f_d[0].get()

    assert np.all(f[val_locations.astype(bool)] == 0)


def test_validation_clear_validation_data(validation_gpu, peaks_reshape):
    i_peaks, j_peaks = peaks_reshape

    validation_gpu(i_peaks)
    validation_gpu.free_gpu_data()

    assert all(
        data is None
        for data in [
            validation_gpu.val_locations,
            validation_gpu._f,
            validation_gpu._neighbours_,
            validation_gpu._mean_,
            validation_gpu._median_,
        ]
    )


def test_validation_free_data(validation_gpu, peaks_reshape):
    i_peaks, j_peaks = peaks_reshape

    validation_gpu(i_peaks)
    validation_gpu.free_gpu_data()

    assert all(
        data is None
        for data in [
            validation_gpu.val_locations,
            validation_gpu._f,
            validation_gpu._neighbours_present,
            validation_gpu._neighbours_,
            validation_gpu._mean_,
            validation_gpu._median_,
        ]
    )


@pytest.mark.parametrize("num_fields, type_", [(1, gpuarray.GPUArray), (2, list)])
def test_validation_median_mean(num_fields, type_, peaks_reshape, validation_gpu):
    validation_gpu._f = peaks_reshape[:num_fields]
    validation_gpu._num_fields = num_fields

    assert isinstance(validation_gpu.median, type_)
    assert isinstance(validation_gpu.mean, type_)


def test_validation_median_num_validation_locations(validation_gpu, peaks_reshape):
    validation_gpu(*peaks_reshape)
    val_locations = validation_gpu.val_locations.get()
    n_val = validation_gpu.num_validation_locations

    assert np.sum(val_locations) == n_val


def test_validation_s2n_validation(validation_gpu, s2n_ratio):
    tol = log10(validation.S2N_TOL)

    val_locations = validation._local_validation(s2n_ratio / tol, 1).get()
    validation_gpu._s2n_validation(s2n_ratio)
    val_locations_gpu = validation_gpu.val_locations.get()

    assert np.array_equal(val_locations_gpu, val_locations)


def test_validation_median_validation(peaks_reshape, mask, validation_gpu):
    tol = validation.MEDIAN_TOL
    validation_gpu._f = peaks_reshape
    validation_gpu._num_fields = len(peaks_reshape)
    neighbours_present_d = validation_gpu._neighbours_present
    val_locations_d = None

    for f_d in peaks_reshape:
        f_neighbours_d = validation._gpu_get_neighbours(f_d, neighbours_present_d)
        f_median_d = validation._gpu_average_velocity(
            f_neighbours_d, neighbours_present_d, "median_velocity"
        )
        f_median_residual_d = validation._gpu_residual(
            f_median_d, f_neighbours_d, neighbours_present_d, "median_residual"
        )
        val_locations_d = validation._neighbour_validation(
            f_d, f_median_d, f_median_residual_d, tol, val_locations=val_locations_d
        )
    val_locations = val_locations_d.get()
    validation_gpu._median_validation()
    val_locations_gpu = validation_gpu.val_locations.get()

    assert np.array_equal(val_locations, val_locations_gpu)


def test_validation_median_validation_vec2d(peaks_reshape, mask, validation_gpu):
    u_d, v_d = peaks_reshape
    tol = validation.MEDIAN_TOL
    validation_gpu._f = peaks_reshape
    validation_gpu._num_fields = len(peaks_reshape)
    neighbours_present_d = validation_gpu._neighbours_present

    u_neighbours_d = validation._gpu_get_neighbours(u_d, neighbours_present_d)
    v_neighbours_d = validation._gpu_get_neighbours(v_d, neighbours_present_d)
    u_median_d = validation._gpu_average_velocity(
        u_neighbours_d, neighbours_present_d, "median_velocity"
    )
    v_median_d = validation._gpu_average_velocity(
        v_neighbours_d, neighbours_present_d, "median_velocity"
    )
    median_residual_d = validation._gpu_residual_vec2d(
        u_median_d,
        v_median_d,
        u_neighbours_d,
        v_neighbours_d,
        neighbours_present_d,
        "median_residual_vec2d",
    )
    val_locations_d = validation._neighbour_validation_vec2d(
        u_d, v_d, u_median_d, v_median_d, median_residual_d, tol
    )
    val_locations = val_locations_d.get()
    validation_gpu._median_validation_vec2d()
    val_locations_gpu = validation_gpu.val_locations.get()

    assert np.array_equal(val_locations, val_locations_gpu)


def test_validation_mean_validation(peaks_reshape, mask, validation_gpu):
    tol = validation.MEAN_TOL
    validation_gpu._f = peaks_reshape
    validation_gpu._num_fields = len(peaks_reshape)
    neighbours_present_d = validation_gpu._neighbours_present
    val_locations_d = None

    for f_d in peaks_reshape:
        f_neighbours_d = validation._gpu_get_neighbours(f_d, neighbours_present_d)
        f_mean_d = validation._gpu_average_velocity(
            f_neighbours_d, neighbours_present_d, "mean_velocity"
        )
        f_mean_residual_d = validation._gpu_residual(
            f_mean_d, f_neighbours_d, neighbours_present_d, "mean_residual"
        )
        val_locations_d = validation._neighbour_validation(
            f_d, f_mean_d, f_mean_residual_d, tol, val_locations=val_locations_d
        )
    val_locations = val_locations_d.get()
    validation_gpu._mean_validation()
    val_locations_gpu = validation_gpu.val_locations.get()

    assert np.array_equal(val_locations, val_locations_gpu)


def test_validation_mean_validation_vec2d(peaks_reshape, mask, validation_gpu):
    u_d, v_d = peaks_reshape
    tol = validation.MEAN_TOL
    validation_gpu._f = peaks_reshape
    validation_gpu._num_fields = len(peaks_reshape)
    neighbours_present_d = validation_gpu._neighbours_present

    u_neighbours_d = validation._gpu_get_neighbours(u_d, neighbours_present_d)
    v_neighbours_d = validation._gpu_get_neighbours(v_d, neighbours_present_d)
    u_mean_d = validation._gpu_average_velocity(
        u_neighbours_d, neighbours_present_d, "mean_velocity"
    )
    v_mean_d = validation._gpu_average_velocity(
        v_neighbours_d, neighbours_present_d, "mean_velocity"
    )
    median_residual_d = validation._gpu_residual_vec2d(
        u_mean_d,
        v_mean_d,
        u_neighbours_d,
        v_neighbours_d,
        neighbours_present_d,
        "mean_residual_vec2d",
    )
    val_locations_d = validation._neighbour_validation_vec2d(
        u_d, v_d, u_mean_d, v_mean_d, median_residual_d, tol
    )
    val_locations = val_locations_d.get()
    validation_gpu._mean_validation_vec2d()
    val_locations_gpu = validation_gpu.val_locations.get()

    assert np.array_equal(val_locations, val_locations_gpu)


def test_validation_rms_validation(peaks_reshape, mask, validation_gpu):
    tol = 1.5
    validation_gpu.rms_tol = tol
    validation_gpu._f = peaks_reshape
    validation_gpu._num_fields = len(peaks_reshape)
    neighbours_present_d = validation_gpu._neighbours_present
    val_locations_d = None

    for f_d in peaks_reshape:
        f_neighbours_d = validation._gpu_get_neighbours(f_d, neighbours_present_d)
        f_mean_d = validation._gpu_average_velocity(
            f_neighbours_d, neighbours_present_d, "mean_velocity"
        )
        f_rms_d = validation._gpu_residual(
            f_mean_d, f_neighbours_d, neighbours_present_d, "rms"
        )
        val_locations_d = validation._neighbour_validation(
            f_d, f_mean_d, f_rms_d, tol, val_locations=val_locations_d
        )
    val_locations = val_locations_d.get()
    validation_gpu._rms_validation()
    val_locations_gpu = validation_gpu.val_locations.get()

    assert np.array_equal(val_locations, val_locations_gpu)


def test_validation_rms_validation_vec2d(peaks_reshape, mask, validation_gpu):
    u_d, v_d = peaks_reshape
    tol = 1.5
    validation_gpu.rms_tol = tol
    validation_gpu._f = peaks_reshape
    validation_gpu._num_fields = len(peaks_reshape)
    neighbours_present_d = validation_gpu._neighbours_present

    u_neighbours_d = validation._gpu_get_neighbours(u_d, neighbours_present_d)
    v_neighbours_d = validation._gpu_get_neighbours(v_d, neighbours_present_d)
    u_mean_d = validation._gpu_average_velocity(
        u_neighbours_d, neighbours_present_d, "mean_velocity"
    )
    v_mean_d = validation._gpu_average_velocity(
        v_neighbours_d, neighbours_present_d, "mean_velocity"
    )
    median_residual_d = validation._gpu_residual_vec2d(
        u_mean_d,
        v_mean_d,
        u_neighbours_d,
        v_neighbours_d,
        neighbours_present_d,
        "rms_vec2d",
    )
    val_locations_d = validation._neighbour_validation_vec2d(
        u_d, v_d, u_mean_d, v_mean_d, median_residual_d, tol
    )
    val_locations = val_locations_d.get()
    validation_gpu._rms_validation_vec2d()
    val_locations_gpu = validation_gpu.val_locations.get()

    assert np.array_equal(val_locations, val_locations_gpu)


def test_validation_mask_val_locations(validation_gpu, mask, boolean_array_pair):
    validation_gpu.mask = mask

    val_locations, val_locations_d = boolean_array_pair(mask.shape, seed=1)

    val_locations_np = (mask.get() == 0) * val_locations
    validation_gpu.val_locations = val_locations_d
    validation_gpu._mask_val_locations()
    val_locations_gpu = validation_gpu.val_locations.get()

    assert np.array_equal(val_locations_gpu, val_locations_np)


def test_validation_gpu_get_neighbours(peaks_reshape, validation_gpu):
    # This simply tests that get_neighbours calls _gpu_get_neighbours() when the
    # _f_neighbours attribute is None.
    validation_gpu._f = peaks_reshape
    validation_gpu._num_fields = n = len(peaks_reshape)
    neighbours_present_d = validation_gpu._neighbours_present

    validation_gpu._neighbours_ = None
    f_neighbours_l = [
        validation._gpu_get_neighbours(f_d, neighbours_present_d).get()
        for f_d in peaks_reshape
    ]
    f_neighbours_gpu_l = [
        f_neighbours_d.get() for f_neighbours_d in validation_gpu._neighbours
    ]
    assert all(
        [np.array_equal(f_neighbours_gpu_l[i], f_neighbours_l[i]) for i in range(n)]
    )


def test_validation_gpu_get_median(peaks_reshape, validation_gpu):
    validation_gpu._f = peaks_reshape
    validation_gpu._num_fields = n = len(peaks_reshape)
    neighbours_present_d = validation_gpu._neighbours_present
    f_neighbours_dl = validation_gpu._neighbours

    validation_gpu._neighbours_ = None
    f_median_l = [
        validation._gpu_average_velocity(
            f_neighbours_dl[i], neighbours_present_d, "median_velocity"
        ).get()
        for i in range(n)
    ]
    f_median_gpu_l = [f_median_d.get() for f_median_d in validation_gpu._median]
    assert all([np.array_equal(f_median_gpu_l[i], f_median_l[i]) for i in range(n)])


def test_validation_gpu_get_mean(peaks_reshape, validation_gpu):
    validation_gpu._f = peaks_reshape
    validation_gpu._num_fields = n = len(peaks_reshape)
    neighbours_present_d = validation_gpu._neighbours_present
    f_neighbours_dl = validation_gpu._neighbours

    validation_gpu._neighbours_ = None
    f_mean_l = [
        validation._gpu_average_velocity(
            f_neighbours_dl[i], neighbours_present_d, "mean_velocity"
        ).get()
        for i in range(n)
    ]
    f_mean_gpu_l = [f_mean_d.get() for f_mean_d in validation_gpu._mean]
    assert all([np.array_equal(f_mean_gpu_l[i], f_mean_l[i]) for i in range(n)])


def test_local_validation(array_pair, boolean_array_pair):
    shape = (16, 16)
    tol = 0.5

    f, f_d = array_pair(shape)
    val_locations, val_locations_d = boolean_array_pair(shape, seed=1)

    val_locations_np = (f > tol).astype(DTYPE_i) | val_locations
    val_locations_gpu = validation._local_validation(
        f_d, tol, val_locations=val_locations_d
    ).get()

    assert np.array_equal(val_locations_gpu, val_locations_np)


def test_neighbour_validation(array_pair, boolean_array_pair):
    shape = (16, 16)
    tol = 2

    f, f_d = array_pair(shape, center=0.0, half_width=1.0)
    f_mean, f_mean_d = array_pair(shape, center=0.0, half_width=1.0, seed=1)
    f_mean_residual, f_mean_residual_d = array_pair(shape, seed=2)
    val_locations, val_locations_d = boolean_array_pair(shape, seed=3)

    val_locations_np = (np.abs(f - f_mean) / (f_mean_residual + 0.1) > tol).astype(
        DTYPE_i
    ) | val_locations
    val_locations_gpu = validation._neighbour_validation(
        f_d, f_mean_d, f_mean_residual_d, tol, val_locations=val_locations_d
    ).get()

    assert np.array_equal(val_locations_gpu, val_locations_np)


def test_neighbour_validation_vec2d(array_pair, boolean_array_pair):
    shape = (16, 16)
    tol = 2

    u, u_d = array_pair(shape, center=0.0, half_width=1.0)
    v, v_d = array_pair(shape, center=0.0, half_width=1.0, seed=1)
    u_mean, u_mean_d = array_pair(shape, center=0.0, half_width=1.0, seed=2)
    v_mean, v_mean_d = array_pair(shape, center=0.0, half_width=1.0, seed=3)
    mean_residual, mean_residual_d = array_pair(shape, seed=4)
    val_locations, val_locations_d = boolean_array_pair(shape, seed=5)

    val_locations_np = (
        np.hypot(u - u_mean, v - v_mean) / (mean_residual + 0.1) > tol
    ).astype(DTYPE_i) | val_locations
    val_locations_gpu = validation._neighbour_validation_vec2d(
        u_d,
        v_d,
        u_mean_d,
        v_mean_d,
        mean_residual_d,
        tol,
        val_locations=val_locations_d,
    ).get()

    assert np.array_equal(val_locations_gpu, val_locations_np)


def test_gpu_find_neighbours(boolean_array_pair):
    shape = (16, 16)

    mask, mask_d = boolean_array_pair(shape, seed=1)

    neighbours_present_np = find_neighbours_np(shape, mask)
    neighbours_present_gpu = validation._gpu_find_neighbours(shape, mask_d).get()

    assert np.array_equal(neighbours_present_gpu, neighbours_present_np)


def test_gpu_get_neighbours(array_pair, boolean_gpu_array):
    shape = (16, 16)

    f, f_d = array_pair(shape, center=0.0, half_width=1.0)
    mask = boolean_gpu_array(shape, seed=1)

    neighbours_present_d = validation._gpu_find_neighbours(shape, mask)
    f_neighbours_np = get_neighbours_np(
        f, neighbours_present=neighbours_present_d.get()
    )
    f_neighbours_gpu = validation._gpu_get_neighbours(f_d, neighbours_present_d).get()

    assert np.array_equal(f_neighbours_gpu, f_neighbours_np)


def test_gpu_median_velocity(array_pair, boolean_gpu_array):
    shape = (16, 16)

    f, f_d = array_pair(shape, center=0.0, half_width=1.0)
    mask = boolean_gpu_array(shape, seed=1)

    neighbours_present_d = validation._gpu_find_neighbours(shape, mask)
    f_neighbours_d = validation._gpu_get_neighbours(f_d, neighbours_present_d)
    f_median_np = median_np(f_neighbours_d.get(), neighbours_present_d.get())
    f_median_gpu = validation._gpu_average_velocity(
        f_neighbours_d, neighbours_present_d, "median_velocity"
    ).get()

    assert np.array_equal(f_median_gpu, f_median_np)


def test_gpu_median_residual(array_pair, boolean_gpu_array):
    shape = (16, 16)

    f, f_d = array_pair(shape, center=0.0, half_width=1.0)
    mask = boolean_gpu_array(shape, seed=1)

    neighbours_present_d = validation._gpu_find_neighbours(shape, mask)
    f_neighbours_d = validation._gpu_get_neighbours(f_d, neighbours_present_d)
    f_median_d = validation._gpu_average_velocity(
        f_neighbours_d, neighbours_present_d, "median_velocity"
    )
    f_median_residual_np = median_residual_np(
        f_median_d.get(), f_neighbours_d.get(), neighbours_present_d.get()
    )
    f_median_residual_gpu = validation._gpu_residual(
        f_median_d, f_neighbours_d, neighbours_present_d, "median_residual"
    ).get()

    assert np.array_equal(f_median_residual_gpu, f_median_residual_np)


def test_gpu_mean_velocity(array_pair, boolean_gpu_array):
    shape = (16, 16)

    f, f_d = array_pair(shape, center=0.0, half_width=1.0)
    mask = boolean_gpu_array(shape, seed=1)

    neighbours_present_d = validation._gpu_find_neighbours(shape, mask)
    f_neighbours_d = validation._gpu_get_neighbours(f_d, neighbours_present_d)
    f_mean_np = mean_np(f_neighbours_d.get(), neighbours_present_d.get())
    f_mean_gpu = validation._gpu_average_velocity(
        f_neighbours_d, neighbours_present_d, "mean_velocity"
    ).get()

    assert np.allclose(f_mean_gpu, f_mean_np)


def test_gpu_mean_residual(array_pair, boolean_gpu_array):
    shape = (16, 16)

    f, f_d = array_pair(shape, center=0.0, half_width=1.0)
    mask = boolean_gpu_array(shape, seed=1)

    neighbours_present_d = validation._gpu_find_neighbours(shape, mask)
    f_neighbours_d = validation._gpu_get_neighbours(f_d, neighbours_present_d)
    f_mean_d = validation._gpu_average_velocity(
        f_neighbours_d, neighbours_present_d, "mean_velocity"
    )
    f_mean_residual_np = mean_residual_np(
        f_mean_d.get(), f_neighbours_d.get(), neighbours_present_d.get()
    )
    f_mean_residual_gpu = validation._gpu_residual(
        f_mean_d, f_neighbours_d, neighbours_present_d, "mean_residual"
    ).get()

    assert np.allclose(f_mean_residual_gpu, f_mean_residual_np)


def test_gpu_rms(array_pair, boolean_gpu_array):
    shape = (16, 16)

    f, f_d = array_pair(shape, center=0.0, half_width=1.0)
    mask = boolean_gpu_array(shape, seed=1)

    neighbours_present_d = validation._gpu_find_neighbours(shape, mask)
    f_neighbours_d = validation._gpu_get_neighbours(f_d, neighbours_present_d)
    f_mean_d = validation._gpu_average_velocity(
        f_neighbours_d, neighbours_present_d, "mean_velocity"
    )
    f_rms_residual_np = rms_np(
        f_mean_d.get(), f_neighbours_d.get(), neighbours_present_d.get()
    )
    f_rms_residual_gpu = validation._gpu_residual(
        f_mean_d, f_neighbours_d, neighbours_present_d, "rms"
    ).get()

    assert np.allclose(f_rms_residual_gpu, f_rms_residual_np)


def test_gpu_median_residual_vec2d(array_pair, boolean_gpu_array):
    shape = (16, 16)

    u, u_d = array_pair(shape, center=0.0, half_width=1.0)
    v, v_d = array_pair(shape, center=0.0, half_width=1.0, seed=1)
    mask = boolean_gpu_array(shape, seed=2)

    neighbours_present_d = validation._gpu_find_neighbours(shape, mask)
    u_neighbours_d = validation._gpu_get_neighbours(u_d, neighbours_present_d)
    v_neighbours_d = validation._gpu_get_neighbours(v_d, neighbours_present_d)
    u_median_d = validation._gpu_average_velocity(
        u_neighbours_d, neighbours_present_d, "median_velocity"
    )
    v_median_d = validation._gpu_average_velocity(
        v_neighbours_d, neighbours_present_d, "median_velocity"
    )
    median_residual_np_ = median_residual_vec2d_np(
        u_median_d.get(),
        v_median_d.get(),
        u_neighbours_d.get(),
        v_neighbours_d.get(),
        neighbours_present_d.get(),
    )
    median_residual_gpu = validation._gpu_residual_vec2d(
        u_median_d,
        v_median_d,
        u_neighbours_d,
        v_neighbours_d,
        neighbours_present_d,
        "median_residual_vec2d",
    ).get()

    assert np.allclose(median_residual_gpu, median_residual_np_)


def test_gpu_mean_residual_vec2d(array_pair, boolean_gpu_array):
    shape = (16, 16)

    u, u_d = array_pair(shape, center=0.0, half_width=1.0)
    v, v_d = array_pair(shape, center=0.0, half_width=1.0, seed=1)
    mask = boolean_gpu_array(shape, seed=1)

    neighbours_present_d = validation._gpu_find_neighbours(shape, mask)
    u_neighbours_d = validation._gpu_get_neighbours(u_d, neighbours_present_d)
    v_neighbours_d = validation._gpu_get_neighbours(v_d, neighbours_present_d)
    u_mean_d = validation._gpu_average_velocity(
        u_neighbours_d, neighbours_present_d, "mean_velocity"
    )
    v_mean_d = validation._gpu_average_velocity(
        v_neighbours_d, neighbours_present_d, "mean_velocity"
    )
    mean_residual_np_ = mean_residual_vec2d_np(
        u_mean_d.get(),
        v_mean_d.get(),
        u_neighbours_d.get(),
        v_neighbours_d.get(),
        neighbours_present_d.get(),
    )
    mean_residual_gpu = validation._gpu_residual_vec2d(
        u_mean_d,
        v_mean_d,
        u_neighbours_d,
        v_neighbours_d,
        neighbours_present_d,
        "mean_residual_vec2d",
    ).get()

    assert np.allclose(mean_residual_gpu, mean_residual_np_)


def test_gpu_rms_vec2d(array_pair, boolean_gpu_array):
    shape = (16, 16)

    u, u_d = array_pair(shape, center=0.0, half_width=1.0)
    v, v_d = array_pair(shape, center=0.0, half_width=1.0, seed=1)
    mask = boolean_gpu_array(shape, seed=2)

    neighbours_present_d = validation._gpu_find_neighbours(shape, mask)
    u_neighbours_d = validation._gpu_get_neighbours(u_d, neighbours_present_d)
    v_neighbours_d = validation._gpu_get_neighbours(v_d, neighbours_present_d)
    u_mean_d = validation._gpu_average_velocity(
        u_neighbours_d, neighbours_present_d, "mean_velocity"
    )
    v_mean_d = validation._gpu_average_velocity(
        v_neighbours_d, neighbours_present_d, "mean_velocity"
    )
    rms_residual_np = rms_vec2d_np(
        u_mean_d.get(),
        v_mean_d.get(),
        u_neighbours_d.get(),
        v_neighbours_d.get(),
        neighbours_present_d.get(),
    )
    rms_residual_gpu = validation._gpu_residual_vec2d(
        u_mean_d,
        v_mean_d,
        u_neighbours_d,
        v_neighbours_d,
        neighbours_present_d,
        "rms_vec2d",
    ).get()

    assert np.allclose(rms_residual_gpu, rms_residual_np)


# INTEGRATION TESTS
@pytest.mark.integtest
@pytest.mark.parametrize(
    "validation_method, expected_sum",
    [("s2n", 316), ("median_velocity", 63), ("mean_velocity", 5), ("rms_velocity", 4)],
)
def test_validation_gpu_(
    validation_method,
    expected_sum,
    validation_gpu,
    peaks_reshape,
    mask,
    s2n_ratio_reshape,
):
    validation_gpu.mask = mask
    validation_gpu.validation_method = validation_method
    val_locations = validation_gpu(*peaks_reshape, s2n=s2n_ratio_reshape).get()

    assert np.sum(val_locations) == expected_sum


@pytest.fixture
def validate(peaks_reshape):
    def validate(f, **params):
        val_locations = validation.gpu_validation(*f, **params).get()

        assert np.sum(val_locations) > 0
        assert not np.any(np.isnan(val_locations))

    return partial(validate, peaks_reshape)


@pytest.mark.integtest
class TestValidationParams:
    @pytest.mark.parametrize(
        "mask_",
        [True, False],
    )
    def test_validation_gpu_mask(self, mask_, mask, validate):
        mask_ = mask if mask_ else None
        validate(mask=mask_)

    @pytest.mark.parametrize(
        "validation_method", list(validation.ALLOWED_VALIDATION_METHODS)
    )
    def test_validation_gpu_validation_method(
        self, validation_method, s2n_ratio, validate
    ):
        validate(s2n=s2n_ratio, validation_method=validation_method)

    @pytest.mark.parametrize("s2n_tol", [1.5, validation.S2N_TOL])
    def test_validation_gpu_s2n_tol(self, s2n_tol, validate):
        validate(s2n_tol=s2n_tol)

    @pytest.mark.parametrize("median_tol", [1.5, validation.MEDIAN_TOL])
    def test_validation_gpu_median_tol(self, median_tol, validate):
        validate(median_tol=median_tol)

    @pytest.mark.parametrize("mean_tol", [1.5, validation.MEAN_TOL])
    def test_validation_gpu_mean_tol(self, mean_tol, validate):
        validate(mean_tol=mean_tol)

    @pytest.mark.parametrize("rms_tol", [1.5, validation.RMS_TOL])
    def test_validation_gpu_rms_tol(self, rms_tol, validate):
        validate(rms_tol=rms_tol)


# REGRESSION TESTS
@pytest.mark.regression
@pytest.mark.parametrize("validation_method", validation.ALLOWED_VALIDATION_METHODS)
def test_validation_regression(
    validation_method,
    validation_gpu,
    peaks_reshape,
    mask,
    ndarrays_regression,
    s2n_ratio_reshape,
):
    validation_gpu.mask = mask
    validation_gpu.validation_method = validation_method
    val_locations = validation_gpu(*peaks_reshape, s2n=s2n_ratio_reshape).get()

    ndarrays_regression.check({"val_locations": val_locations})
