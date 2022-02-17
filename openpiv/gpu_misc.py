"""This module contains miscellaneous GPU functions."""
from math import ceil

import numpy as np
# Create the PyCUDA context.
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
# import pycuda.cumath as cumath
from pycuda.compiler import SourceModule

# Define 32-bit types.
DTYPE_i = np.int32
DTYPE_f = np.float32
DTYPE_c = np.complex64


def gpu_scalar_mod_i(f_d, m):
    """Returns the integer and remainder of division of a PyCUDA array by a scalar int.

    Parameters
    ----------
    f_d : GPUArray
        nd int, input to be decomposed.
    m : int
        Modulus.

    Returns
    -------
    i_d, r_d : GPUArray
        Int, integer part of the decomposition.
    r_d : GPUArray
        Int, remainder part of the decomposition.

    """
    _check_inputs(f_d, array_type=gpuarray.GPUArray, dtype=DTYPE_i)
    assert 0 < m == int(m)
    size_i = DTYPE_i(f_d.size)

    i_d = gpuarray.empty_like(f_d, dtype=DTYPE_i)
    r_d = gpuarray.empty_like(f_d, dtype=DTYPE_i)

    mod_scalar_mod = SourceModule("""
    __global__ void scalar_mod(int *i, int *r, int *f, int m, int size)
    {
        // i, r : output arguments
        int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (t_idx >= size) {return;}

        int f_value = f[t_idx];
        i[t_idx] = f_value / m;
        r[t_idx] = f_value % m;
    }
    """)
    block_size = 32
    grid_size = ceil(size_i / block_size)
    mask_frame_gpu = mod_scalar_mod.get_function('scalar_mod')
    mask_frame_gpu(i_d, r_d, f_d, DTYPE_i(m), size_i, block=(block_size, 1, 1), grid=(grid_size, 1))

    return i_d, r_d


def gpu_remove_nan_f(f_d):
    """Replaces all NaN from array with zeros.

    Parameters
    ----------
    f_d : GPUArray
        nd float.

    """
    _check_inputs(f_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f)
    size_i = DTYPE_i(f_d.size)
    mod_replace_nan_f = SourceModule("""
        __global__ void replace_nan_f(float *f, int size)
    {
        int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(t_idx >= size) {return;}
        float value = f[t_idx];

        // Check for NaNs. The comparison is True for NaNs.
        if (value != value) {f[t_idx] = 0.0f;}
    }
    """)
    block_size = 32
    grid_size = ceil(size_i / block_size)
    index_update = mod_replace_nan_f.get_function('replace_nan_f')
    index_update(f_d, size_i, block=(block_size, 1, 1), grid=(grid_size, 1))


def _check_inputs(*arrays, array_type=None, dtype=None, shape=None, ndim=None, size=None):
    """Checks that all array inputs match either each other's or the given array type, dtype, shape and dim."""
    if array_type is not None:
        assert all([isinstance(array, array_type) for array in arrays]), 'Input(s) must be ({}).'.format(array_type)
    if dtype is not None:
        assert all([array.dtype == dtype for array in arrays]), 'Input(s) must have dtype ({}).'.format(dtype)
    if shape is not None:
        assert all(
            [array.shape == shape for array in
             arrays]), 'Input(s) must have shape ({}, all must be same shape).'.format(
            shape)
    if ndim is not None:
        assert all([array.ndim == ndim for array in arrays]), 'Input(s) must have ndim ({}).'.format(ndim)
    if size is not None:
        assert all([array.size == size for array in arrays]), 'Input(s) must have size ({}).'.format(size)
