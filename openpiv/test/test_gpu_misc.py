"""Test module for gpu_misc.py."""

import numpy as np
import pytest

import pycuda.gpuarray as gpuarray

import openpiv.gpu_misc as gpu_misc

DTYPE_i = np.int32
DTYPE_f = np.float32


# SCRIPTS
def generate_array(size, magnitude=1, d_type=DTYPE_f, seed=0):
    """Returns ndarray with pseudo-random values."""
    np.random.seed(seed)
    f = (np.random.random(size) * magnitude).astype(d_type)

    return f


def generate_np_gpu_array_pair(size, magnitude=1, d_type=DTYPE_f, seed=0):
    """Returns a pair of numpy and gpu arrays with identical pseudo-random values."""
    f = generate_array(size, magnitude=magnitude, d_type=d_type, seed=seed)
    f_d = gpuarray.to_gpu(f)

    return f, f_d


# UNIT TESTS
def test_gpu_mask():
    size = (16, 16)

    f, f_d = generate_np_gpu_array_pair(size, magnitude=2, d_type=DTYPE_f)
    mask, mask_d = generate_np_gpu_array_pair(size, magnitude=2, d_type=DTYPE_i)

    f_masked = f * (1 - mask)
    f_masked_gpu = gpu_misc.gpu_mask(f_d, mask_d).get()

    assert np.array_equal(f_masked, f_masked_gpu)


@pytest.mark.parametrize('divisor', [1, 2, 3])
def test_gpu_scalar_mod_i(divisor):
    size = (16, 16)
    m = divisor

    f, f_d = generate_np_gpu_array_pair(size, magnitude=10, d_type=DTYPE_i)

    i_cpu = f // m
    r_cpu = f % m
    i_d, r_d = gpu_misc.gpu_scalar_mod_i(f_d, m)
    i_gpu = i_d.get()
    r_gpu = r_d.get()

    assert np.array_equal(i_cpu, i_gpu)
    assert np.array_equal(r_cpu, r_gpu)


def test_gpu_replace_nan_f():
    size = (16, 16)

    f = generate_array(size, d_type=DTYPE_f)
    f[f < 0.25] = np.nan
    f[f > 0.75] = np.inf
    f_d = gpuarray.to_gpu(f)

    f_finite = np.nan_to_num(f, nan=0, posinf=np.inf)
    gpu_misc.gpu_remove_nan_f(f_d)
    f_finite_gpu = f_d.get()

    assert np.array_equal(f_finite, f_finite_gpu)


def test_gpu_replace_negative_f():
    size = (16, 16)

    f = generate_array(size, magnitude=2, d_type=DTYPE_f) - 1
    f_d = gpuarray.to_gpu(f)

    f[f < 0] = 0
    gpu_misc.gpu_remove_negative_f(f_d)
    f_positive_gpu = f_d.get()

    assert np.array_equal(f, f_positive_gpu)
