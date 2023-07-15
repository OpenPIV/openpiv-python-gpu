"""Test module for gpu_misc.py."""

import numpy as np
import pytest

import pycuda.gpuarray as gpuarray

import openpiv.gpu_misc as gpu_misc

DTYPE_i = np.int32
DTYPE_f = np.float32


# UTILS
def generate_np_array(shape, magnitude=1.0, offset=0.0, d_type=DTYPE_f, seed=0):
    """Returns ndarray with pseudo-random values."""
    np.random.seed(seed)
    f = (np.random.random(shape) * magnitude + offset).astype(d_type)

    return f


def generate_gpu_array(shape, magnitude=1.0, offset=0.0, d_type=DTYPE_f, seed=0):
    """Returns ndarray with pseudo-random values."""
    f = generate_np_array(
        shape, magnitude=magnitude, offset=offset, d_type=d_type, seed=seed
    )
    f_d = gpuarray.to_gpu(f)

    return f_d


def generate_array_pair(shape, magnitude=1.0, offset=0.0, d_type=DTYPE_f, seed=0):
    """Returns a pair of numpy and gpu arrays with identical pseudo-random values."""
    f = generate_np_array(
        shape, magnitude=magnitude, offset=offset, d_type=d_type, seed=seed
    )
    f_d = gpuarray.to_gpu(f)

    return f, f_d


def generate_np_array1(shape, center=0.0, width=1.0, d_type=DTYPE_f, seed=0):
    """Returns ndarray with pseudo-random values."""
    np.random.seed(seed)
    f = ((np.random.random(shape) * 2 - 1) * width + center).astype(d_type)

    return f


def generate_gpu_array1(shape, center=0.0, width=1.0, d_type=DTYPE_f, seed=0):
    """Returns ndarray with pseudo-random values."""
    f = generate_np_array1(shape, center=center, width=width, d_type=d_type, seed=seed)
    f_d = gpuarray.to_gpu(f)

    return f_d


def generate_array_pair1(shape, center=0.0, width=1.0, d_type=DTYPE_f, seed=0):
    """Returns a pair of numpy and gpu arrays with identical pseudo-random values."""
    f = generate_np_array1(shape, center=center, width=width, d_type=d_type, seed=seed)
    f_d = gpuarray.to_gpu(f)

    return f, f_d


# UNIT TESTS
@pytest.mark.parametrize("d_type", [DTYPE_i, DTYPE_f])
def test_gpu_mask(d_type):
    shape = (16, 16)

    f, f_d = generate_array_pair1(shape, center=1, d_type=d_type)
    mask, mask_d = generate_array_pair1(shape, center=1, d_type=DTYPE_i)

    f_masked = f * (1 - mask)
    f_masked_gpu = gpu_misc.gpu_mask(f_d, mask_d).get()

    assert np.array_equal(f_masked_gpu, f_masked)


@pytest.mark.parametrize("d_type", [DTYPE_i, DTYPE_f])
def test_gpu_scalar_mod_i(d_type):
    shape = (16, 16)
    m = 3

    f, f_d = generate_array_pair(shape, magnitude=10, d_type=DTYPE_i)

    i = f // m
    r = f % m
    i_d, r_d = gpu_misc.gpu_scalar_mod(f_d, m)
    i_gpu = i_d.get()
    r_gpu = r_d.get()

    assert np.array_equal(i_gpu, i)
    assert np.array_equal(r_gpu, r)


def test_gpu_replace_nan_f():
    shape = (16, 16)

    f = generate_np_array(shape, d_type=DTYPE_f)
    f[f < 0.25] = np.nan
    f[f > 0.75] = np.inf
    f_d = gpuarray.to_gpu(f)

    f_finite = np.nan_to_num(f, nan=0, posinf=np.inf)
    gpu_misc.gpu_remove_nan(f_d)
    f_finite_gpu = f_d.get()

    assert np.array_equal(f_finite_gpu, f_finite)


@pytest.mark.parametrize("d_type", [DTYPE_i, DTYPE_f])
def test_gpu_replace_negative_f(d_type):
    shape = (16, 16)

    f, f_d = generate_array_pair1(shape, width=2, d_type=d_type)

    f[f < 0] = 0
    gpu_misc.gpu_remove_negative(f_d)
    f_positive_gpu = f_d.get()

    assert np.array_equal(f_positive_gpu, f)
