"""This module contains miscellaneous GPU functions."""
from math import ceil

import numpy as np
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

# Define 32-bit types.
DTYPE_i = np.int32
DTYPE_f = np.float32
DTYPE_c = np.complex64

mod_mask = SourceModule("""
__global__ void gpu_mask_f(float *f_masked, float *f, int *mask, int size)
{
    // frame_masked : output argument
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= size) {return;}

    f_masked[t_idx] = f[t_idx] * (mask[t_idx] == 0.0f);
}

__global__ void gpu_mask_i(int *f_masked, int *f, int *mask, int size)
{
    // frame_masked : output argument
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= size) {return;}

    f_masked[t_idx] = f[t_idx] * (mask[t_idx] == 0.0f);
}
""")


def gpu_mask(f, mask):
    """Mask an array.

    Parameters
    ----------
    f : GPUArray
        nD float, frame to be masked.
    mask : GPUArray or None, optional
        nD int, mask to apply to frame. 0s are values to keep.

    Returns
    -------
    GPUArray
        nD int, masked field.

    """
    _check_arrays(f, array_type=gpuarray.GPUArray)
    _check_arrays(mask, array_type=gpuarray.GPUArray, dtype=DTYPE_i, size=f.size)
    d_type = f.dtype
    size = f.size

    f_masked = gpuarray.empty_like(f)

    block_size = 32
    grid_size = ceil(size / block_size)
    if d_type == DTYPE_f:
        mask_gpu = mod_mask.get_function('gpu_mask_f')
    elif d_type == DTYPE_i:
        mask_gpu = mod_mask.get_function('gpu_mask_i')
    else:
        raise ValueError('Wrong data type for f_d.')
    mask_gpu(f_masked, f, mask, DTYPE_i(size), block=(block_size, 1, 1), grid=(grid_size, 1))

    return f_masked


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


# TODO add support for float
def gpu_scalar_mod(f, m):
    """Returns the integer and remainder of division of a PyCUDA array by a scalar int.

    Parameters
    ----------
    f : GPUArray
        nd int, input to be decomposed.
    m : int
        Modulus.

    Returns
    -------
    i, r : GPUArray
        Int, integer and remainder parts of the decomposition.

    """
    _check_arrays(f, array_type=gpuarray.GPUArray, dtype=DTYPE_i)
    assert 0 < m == int(m)
    size = f.size

    i = gpuarray.empty_like(f, dtype=DTYPE_i)
    r = gpuarray.empty_like(f, dtype=DTYPE_i)

    block_size = 32
    grid_size = ceil(size / block_size)
    mask_frame_gpu = mod_scalar_mod.get_function('scalar_mod')
    mask_frame_gpu(i, r, f, DTYPE_i(m), DTYPE_i(size), block=(block_size, 1, 1), grid=(grid_size, 1))

    return i, r


mod_replace_nan_f = SourceModule("""
#include <math.h>

__global__ void replace_nan_f(float *f, int size)
{
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= size) {return;}
    
    // Check for NaNs.
    if (std::isnan(f[t_idx])) {f[t_idx] = 0.0f;}
}
""")


# TODO add support for int
def gpu_remove_nan(f):
    """Replaces all NaN from array with zeros.

    Parameters
    ----------
    f : GPUArray
        nd float.

    """
    _check_arrays(f, array_type=gpuarray.GPUArray, dtype=DTYPE_f)
    size = f.size

    block_size = 32
    grid_size = ceil(size / block_size)
    replace_nan = mod_replace_nan_f.get_function('replace_nan_f')
    replace_nan(f, DTYPE_i(size), block=(block_size, 1, 1), grid=(grid_size, 1))


mod_replace_negative_f = SourceModule("""
__global__ void replace_negative_f(float *f, int size)
{
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= size) {return;}
    float value = f[t_idx];

    // Check for negative values.
    f[t_idx] = value * (value > 0.0f);
}
""")


# TODO add support for float
def gpu_remove_negative(f):
    """Replaces all negative values from array with zeros.

    Parameters
    ----------
    f : GPUArray
        nd float.

    """
    _check_arrays(f, array_type=gpuarray.GPUArray, dtype=DTYPE_f)
    size = f.size

    block_size = 32
    grid_size = ceil(size / block_size)
    replace_negative = mod_replace_negative_f.get_function('replace_negative_f')
    replace_negative(f, DTYPE_i(size), block=(block_size, 1, 1), grid=(grid_size, 1))


def _check_arrays(*arrays, array_type=None, dtype=None, shape=None, ndim=None, size=None):
    """Checks that all array inputs match either each other's or the given array type, dtype, shape and dim."""
    if not all([array.flags.c_contiguous for array in arrays]):
        raise ValueError('{} input(s) must be C-contiguous.'.format(len(arrays)))
    if array_type is not None:
        if not all([isinstance(array, array_type) for array in arrays]):
            raise TypeError('{} input(s) must be {}.'.format(len(arrays), array_type))
    if dtype is not None:
        if not all([array.dtype == dtype for array in arrays]):
            raise ValueError('{} input(s) must have dtype {}.'.format(len(arrays), dtype))
    if shape is not None:
        if not all([array.shape == shape for array in arrays]):
            raise ValueError('{} input(s) must have shape {}.'.format(len(arrays), shape))
    if ndim is not None:
        if not all([array.ndim == ndim for array in arrays]):
            raise ValueError('{} input(s) must have ndim {}.'.format(len(arrays), ndim))
    if size is not None:
        if not all([array.size == size for array in arrays]):
            raise ValueError('{} input(s) must have size {}.'.format(len(arrays), size))


# TODO compare speed with above function.
# def _check_arrays1(*arrays, array_type=None, dtype=None, shape=None, ndim=None, size=None):
#     """Checks that all array inputs match either each other's or the given array type, dtype, shape and dim."""
#     for array in arrays:
#         if not array.flags.c_contiguous:
#             raise TypeError('{} input(s) must be C-contiguous.'.format(len(arrays)))
#         if array_type is not None:
#             if not isinstance(array, array_type):
#                 raise TypeError('{} input(s) must be {}.'.format(len(arrays), array_type))
#         if dtype is not None:
#             if not array.dtype == dtype:
#                 raise ValueError('{} input(s) must have dtype {}.'.format(len(arrays), dtype))
#         if shape is not None:
#             if not array.shape == shape:
#                 raise ValueError('{} input(s) must have shape {}.'.format(len(arrays), shape))
#         if ndim is not None:
#             if not array.ndim == ndim:
#                 raise ValueError('{} input(s) must have ndim {}.'.format(len(arrays), ndim))
#         if size is not None:
#             if not array.size == size:
#                 raise ValueError('{} input(s) must have size {}.'.format(len(arrays), size))
