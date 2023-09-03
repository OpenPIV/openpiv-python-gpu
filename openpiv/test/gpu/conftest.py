import pytest
import numpy as np

import pycuda.gpuarray as gpuarray

DTYPE_i = np.int32
DTYPE_f = np.float32


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
