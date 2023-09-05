"""Test module for gpu_validation.py.

Still need to test scalar fields and vector fields of higher dimensions.

"""

from math import log10

import numpy as np
import pytest
import pycuda.gpuarray as gpuarray
# noinspection PyUnresolvedReferences
import pycuda.autoinit

import gpu_validation

DTYPE_i = np.int32
DTYPE_f = np.float32

data_dir = "../data/"


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


def median_fluc_np(f_median, f_neighbours, neighbours_present):
    f_neighbours[neighbours_present == 0] = np.nan
    f_median_fluc = np.nanmedian(
        np.abs(f_neighbours - f_median.reshape(*f_median.shape, 1)), axis=2
    )
    f_median_fluc = np.nan_to_num(f_median_fluc, copy=False)

    return f_median_fluc


def mean_fluc_np(f_mean, f_neighbours, neighbours_present):
    denominator = np.sum(neighbours_present, axis=2).astype(DTYPE_f)
    numerator = np.sum(np.abs(f_neighbours - f_mean.reshape(*f_mean.shape, 1)), axis=2)
    f_mean_fluc = numerator / (denominator + (denominator == 0.0))

    return f_mean_fluc


def rms_np(f_mean, f_neighbours, neighbours_present):
    denominator = np.sum(neighbours_present, axis=2).astype(DTYPE_f)
    numerator = np.sum((f_neighbours - f_mean.reshape(*f_mean.shape, 1)) ** 2, axis=2)
    f_rms_fluc = numerator / (denominator + (denominator == 0.0))

    return f_rms_fluc


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
def test_validation_gpu_free_data(validation_gpu, peaks_d):
    i_peaks_d, j_peaks_d = peaks_d

    validation_gpu(i_peaks_d)
    validation_gpu.free_data()

    assert all(
        data is None
        for data in [
            validation_gpu._val_locations,
            validation_gpu._f,
            validation_gpu._f_neighbours,
            validation_gpu._f_mean,
            validation_gpu._f_median,
        ]
    )


@pytest.mark.parametrize("num_fields, type_", [(1, gpuarray.GPUArray), (2, list)])
def test_validation_gpu_median_mean(num_fields, type_, validation_gpu, peaks_d):
    validation_gpu._f = peaks_d[:num_fields]
    validation_gpu._num_fields = num_fields

    assert isinstance(validation_gpu.median, type_)
    assert isinstance(validation_gpu.mean, type_)


def test_validation_gpu_s2n_validation(validation_gpu, sig2noise_d):
    tol = log10(gpu_validation.S2N_TOL)

    val_locations = gpu_validation._local_validation(sig2noise_d / tol, 1).get()
    validation_gpu._s2n_validation(sig2noise_d)
    val_locations_gpu = validation_gpu._val_locations.get()

    assert np.array_equal(val_locations_gpu, val_locations)


def test_validation_gpu_median_validation(validation_gpu, peaks_d, mask_d):
    tol = gpu_validation.MEDIAN_TOL
    validation_gpu._f = peaks_d
    validation_gpu._num_fields = len(peaks_d)
    neighbours_present_d = validation_gpu._neighbours_present
    val_locations_d = None

    for f_d in peaks_d:
        f_neighbours_d = gpu_validation._gpu_get_neighbours(f_d, neighbours_present_d)
        f_median_d = gpu_validation._gpu_median_velocity(
            f_neighbours_d, neighbours_present_d
        )
        f_median_fluc_d = gpu_validation._gpu_median_fluc(
            f_median_d, f_neighbours_d, neighbours_present_d
        )
        val_locations_d = gpu_validation._neighbour_validation(
            f_d, f_median_d, f_median_fluc_d, tol, val_locations=val_locations_d
        )
    val_locations = val_locations_d.get()
    validation_gpu._median_validation()
    val_locations_gpu = validation_gpu._val_locations.get()

    assert np.array_equal(val_locations, val_locations_gpu)


def test_validation_gpu_mean_validation(validation_gpu, peaks_d, mask_d):
    tol = gpu_validation.MEAN_TOL
    validation_gpu._f = peaks_d
    validation_gpu._num_fields = len(peaks_d)
    neighbours_present_d = validation_gpu._neighbours_present
    val_locations_d = None

    for f_d in peaks_d:
        f_neighbours_d = gpu_validation._gpu_get_neighbours(f_d, neighbours_present_d)
        f_mean_d = gpu_validation._gpu_mean_velocity(
            f_neighbours_d, neighbours_present_d
        )
        f_mean_fluc_d = gpu_validation._gpu_mean_fluc(
            f_mean_d, f_neighbours_d, neighbours_present_d
        )
        val_locations_d = gpu_validation._neighbour_validation(
            f_d, f_mean_d, f_mean_fluc_d, tol, val_locations=val_locations_d
        )
    val_locations = val_locations_d.get()
    validation_gpu._mean_validation()
    val_locations_gpu = validation_gpu._val_locations.get()

    assert np.array_equal(val_locations, val_locations_gpu)


def test_validation_gpu_rms_validation(validation_gpu, peaks_d, mask_d):
    tol = gpu_validation.RMS_TOL
    validation_gpu._f = peaks_d
    validation_gpu._num_fields = len(peaks_d)
    neighbours_present_d = validation_gpu._neighbours_present
    val_locations_d = None

    for f_d in peaks_d:
        f_neighbours_d = gpu_validation._gpu_get_neighbours(f_d, neighbours_present_d)
        f_mean_d = gpu_validation._gpu_mean_velocity(
            f_neighbours_d, neighbours_present_d
        )
        f_rms_d = gpu_validation._gpu_rms(
            f_mean_d, f_neighbours_d, neighbours_present_d
        )
        val_locations_d = gpu_validation._neighbour_validation(
            f_d, f_mean_d, f_rms_d, tol, val_locations=val_locations_d
        )
    val_locations = val_locations_d.get()
    validation_gpu._rms_validation()
    val_locations_gpu = validation_gpu._val_locations.get()

    assert np.array_equal(val_locations, val_locations_gpu)


def test_validation_gpu_mask_val_locations(validation_gpu, mask_d, array_pair):
    # Use an example fixture based on the first test case
    validation_gpu.mask = mask_d

    val_locations, val_locations_d = array_pair(
        mask_d.shape, magnitude=2, d_type=DTYPE_i, seed=1
    )

    val_locations_np = (mask_d.get() == 0) * val_locations
    validation_gpu._val_locations = val_locations_d
    validation_gpu._mask_val_locations()
    val_locations_gpu = validation_gpu._val_locations.get()

    assert np.array_equal(val_locations_gpu, val_locations_np)


def test_validation_gpu_get_neighbours(validation_gpu, peaks_d):
    # This simply tests that get_neighbours calls _gpu_get_neighbours() when the
    # _f_neighbours attribute is None.
    validation_gpu._f = peaks_d
    validation_gpu._num_fields = n = len(peaks_d)
    neighbours_present_d = validation_gpu._neighbours_present

    validation_gpu._f_neighbours = None
    f_neighbours_l = [
        gpu_validation._gpu_get_neighbours(f_d, neighbours_present_d).get()
        for f_d in peaks_d
    ]
    f_neighbours_gpu_l = [
        f_neighbours_d.get() for f_neighbours_d in validation_gpu._get_neighbours()
    ]
    assert all(
        [np.array_equal(f_neighbours_gpu_l[i], f_neighbours_l[i]) for i in range(n)]
    )


def test_validation_gpu_get_median(validation_gpu, peaks_d):
    validation_gpu._f = peaks_d
    validation_gpu._num_fields = n = len(peaks_d)
    neighbours_present_d = validation_gpu._neighbours_present
    f_neighbours_dl = validation_gpu._get_neighbours()

    validation_gpu._f_neighbours = None
    f_median_l = [
        gpu_validation._gpu_median_velocity(
            f_neighbours_dl[i], neighbours_present_d
        ).get()
        for i in range(n)
    ]
    f_median_gpu_l = [f_median_d.get() for f_median_d in validation_gpu._get_median()]
    assert all([np.array_equal(f_median_gpu_l[i], f_median_l[i]) for i in range(n)])


def test_validation_gpu_get_mean(validation_gpu, peaks_d):
    validation_gpu._f = peaks_d
    validation_gpu._num_fields = n = len(peaks_d)
    neighbours_present_d = validation_gpu._neighbours_present
    f_neighbours_dl = validation_gpu._get_neighbours()

    validation_gpu._f_neighbours = None
    f_mean_l = [
        gpu_validation._gpu_mean_velocity(
            f_neighbours_dl[i], neighbours_present_d
        ).get()
        for i in range(n)
    ]
    f_mean_gpu_l = [f_mean_d.get() for f_mean_d in validation_gpu._get_mean()]
    assert all([np.array_equal(f_mean_gpu_l[i], f_mean_l[i]) for i in range(n)])


def test_local_validation(array_pair):
    shape = (16, 16)
    tol = 0.5

    f, f_d = array_pair(shape, magnitude=1.0, d_type=DTYPE_f)
    val_locations, val_locations_d = array_pair(
        shape, magnitude=2, d_type=DTYPE_i, seed=1
    )

    val_locations_np = (f > tol).astype(DTYPE_i) | val_locations
    val_locations_gpu = gpu_validation._local_validation(
        f_d, tol, val_locations=val_locations_d
    ).get()

    assert np.array_equal(val_locations_gpu, val_locations_np)


def test_neighbour_validation(array_pair):
    shape = (16, 16)
    tol = 0.5

    f, f_d = array_pair(shape, magnitude=1.0, d_type=DTYPE_f)
    f_mean, f_mean_d = array_pair(shape, magnitude=1.0, d_type=DTYPE_f, seed=1)
    f_mean_fluc, f_mean_fluc_d = array_pair(
        shape, magnitude=1.0, d_type=DTYPE_f, seed=2
    )
    val_locations, val_locations_d = array_pair(
        shape, magnitude=2, d_type=DTYPE_i, seed=3
    )

    val_locations_np = (np.abs(f - f_mean) / (f_mean_fluc + 0.1) > tol).astype(
        DTYPE_i
    ) | val_locations
    val_locations_gpu = gpu_validation._neighbour_validation(
        f_d, f_mean_d, f_mean_fluc_d, tol, val_locations=val_locations_d
    ).get()

    assert np.array_equal(val_locations_gpu, val_locations_np)


def test_gpu_find_neighbours(array_pair):
    shape = (16, 16)

    mask, mask_d = array_pair(shape, magnitude=2, d_type=DTYPE_i, seed=1)

    neighbours_present_np = find_neighbours_np(shape, mask)
    neighbours_present_gpu = gpu_validation._gpu_find_neighbours(shape, mask_d).get()

    assert np.array_equal(neighbours_present_gpu, neighbours_present_np)


def test_gpu_get_neighbours(array_pair, gpu_array):
    shape = (16, 16)

    f, f_d = array_pair(shape, magnitude=2.0, offset=-1.0, d_type=DTYPE_f)
    mask_d = gpu_array(shape, magnitude=2, d_type=DTYPE_i, seed=1)

    neighbours_present_d = gpu_validation._gpu_find_neighbours(shape, mask_d)
    f_neighbours_np = get_neighbours_np(
        f, neighbours_present=neighbours_present_d.get()
    )
    f_neighbours_gpu = gpu_validation._gpu_get_neighbours(
        f_d, neighbours_present_d
    ).get()

    assert np.array_equal(f_neighbours_gpu, f_neighbours_np)


def test_gpu_median_velocity(array_pair, gpu_array):
    shape = (16, 16)

    f, f_d = array_pair(shape, magnitude=2.0, offset=-1.0, d_type=DTYPE_f)
    mask_d = gpu_array(shape, magnitude=2, d_type=DTYPE_i, seed=1)

    neighbours_present_d = gpu_validation._gpu_find_neighbours(shape, mask_d)
    f_neighbours_d = gpu_validation._gpu_get_neighbours(f_d, neighbours_present_d)
    f_median_np = median_np(f_neighbours_d.get(), neighbours_present_d.get())
    f_median_gpu = gpu_validation._gpu_median_velocity(
        f_neighbours_d, neighbours_present_d
    ).get()

    assert np.array_equal(f_median_gpu, f_median_np)


def test_gpu_median_fluc(array_pair, gpu_array):
    shape = (16, 16)

    f, f_d = array_pair(shape, magnitude=2.0, offset=-1.0, d_type=DTYPE_f)
    mask_d = gpu_array(shape, magnitude=2, d_type=DTYPE_i, seed=1)

    neighbours_present_d = gpu_validation._gpu_find_neighbours(shape, mask_d)
    f_neighbours_d = gpu_validation._gpu_get_neighbours(f_d, neighbours_present_d)
    f_median_d = gpu_validation._gpu_median_velocity(
        f_neighbours_d, neighbours_present_d
    )
    f_median_fluc_np = median_fluc_np(
        f_median_d.get(), f_neighbours_d.get(), neighbours_present_d.get()
    )
    f_median_fluc_gpu = gpu_validation._gpu_median_fluc(
        f_median_d, f_neighbours_d, neighbours_present_d
    ).get()

    assert np.array_equal(f_median_fluc_gpu, f_median_fluc_np)


def test_gpu_mean_velocity(array_pair, gpu_array):
    shape = (16, 16)

    f, f_d = array_pair(shape, magnitude=2.0, offset=-1.0, d_type=DTYPE_f)
    mask_d = gpu_array(shape, magnitude=2, d_type=DTYPE_i, seed=1)

    neighbours_present_d = gpu_validation._gpu_find_neighbours(shape, mask_d)
    f_neighbours_d = gpu_validation._gpu_get_neighbours(f_d, neighbours_present_d)
    f_mean_np = mean_np(f_neighbours_d.get(), neighbours_present_d.get())
    f_mean_gpu = gpu_validation._gpu_mean_velocity(
        f_neighbours_d, neighbours_present_d
    ).get()

    assert np.allclose(f_mean_gpu, f_mean_np)


def test_gpu_mean_fluc(array_pair, gpu_array):
    shape = (16, 16)

    f, f_d = array_pair(shape, magnitude=2.0, offset=-1.0, d_type=DTYPE_f)
    mask_d = gpu_array(shape, magnitude=2, d_type=DTYPE_i, seed=1)

    neighbours_present_d = gpu_validation._gpu_find_neighbours(shape, mask_d)
    f_neighbours_d = gpu_validation._gpu_get_neighbours(f_d, neighbours_present_d)
    f_mean_d = gpu_validation._gpu_mean_velocity(f_neighbours_d, neighbours_present_d)
    f_mean_fluc_np = mean_fluc_np(
        f_mean_d.get(), f_neighbours_d.get(), neighbours_present_d.get()
    )
    f_mean_fluc_gpu = gpu_validation._gpu_mean_fluc(
        f_mean_d, f_neighbours_d, neighbours_present_d
    ).get()

    assert np.allclose(f_mean_fluc_gpu, f_mean_fluc_np)


def test_gpu_rms(array_pair, gpu_array):
    shape = (16, 16)

    f, f_d = array_pair(shape, magnitude=2.0, offset=-1.0, d_type=DTYPE_f)
    mask_d = gpu_array(shape, magnitude=2, d_type=DTYPE_i, seed=1)

    neighbours_present_d = gpu_validation._gpu_find_neighbours(shape, mask_d)
    f_neighbours_d = gpu_validation._gpu_get_neighbours(f_d, neighbours_present_d)
    f_mean_d = gpu_validation._gpu_mean_velocity(f_neighbours_d, neighbours_present_d)
    f_rms_fluc_np = mean_fluc_np(
        f_mean_d.get(), f_neighbours_d.get(), neighbours_present_d.get()
    )
    f_rms_fluc_gpu = gpu_validation._gpu_mean_fluc(
        f_mean_d, f_neighbours_d, neighbours_present_d
    ).get()

    assert np.allclose(f_rms_fluc_gpu, f_rms_fluc_np)


# INTEGRATION TESTS
# TODO Remove dependency on data regression
# TODO keep data regressions for development only
@pytest.mark.integtest
@pytest.mark.parametrize("validation_method", gpu_validation.ALLOWED_VALIDATION_METHODS)
def test_gpu_validation(
    validation_method, peaks_d, mask_d, sig2noise_d, ndarrays_regression
):
    val_locations = gpu_validation.gpu_validation(
        *peaks_d,
        sig2noise=sig2noise_d,
        mask=mask_d,
        validation_method=validation_method
    ).get()

    ndarrays_regression.check({"val_locations": val_locations})


@pytest.mark.integtest
@pytest.mark.parametrize("validation_method", gpu_validation.ALLOWED_VALIDATION_METHODS)
def test_validation_gpu(
    validation_method, validation_gpu, peaks_d, mask_d, ndarrays_regression, sig2noise_d
):
    validation_gpu.mask = mask_d
    validation_gpu.validation_method = validation_method
    val_locations = validation_gpu(*peaks_d, sig2noise=sig2noise_d).get()

    ndarrays_regression.check({"val_locations": val_locations})
