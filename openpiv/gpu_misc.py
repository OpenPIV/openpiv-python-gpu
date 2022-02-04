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


def _gpu_array_index(array_d, indices, dtype):
    """Allows for arbitrary index selecting with numpy arrays

    Parameters
    ----------
    array_d : GPUArray
        Float or int, array to be selected from.
    indices : GPUArray
        1D int, list of indexes that you want to index. If you are indexing more than 1 dimension, then make sure that
        this array is flattened.
    dtype : dtype
        Either int32 or float 32. determines the datatype of the returned array.

    Returns
    -------
    GPUArray
        Float or int, values at the specified indexes.

    """
    assert indices.ndim == 1, "Number of dimensions of indices is wrong. Should be equal to 1"
    assert type(array_d) == gpuarray.GPUArray, 'Input must be GPUArray.'
    assert array_d.dtype == DTYPE_f or array_d.dtype == DTYPE_f, 'Input must have dtype float32 or int32.'

    return_values_d = gpuarray.zeros(indices.size, dtype=dtype)

    mod_array_index = SourceModule("""
    __global__ void array_index_float(float *return_values, float *array, int *return_list, int size)
    {
        int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (t_idx >= size) {return;}

        return_values[t_idx] = array[return_list[t_idx]];
    }

    __global__ void array_index_int(float *array, int *return_values, int *return_list, int size)
    {
        int t_idx = blockIdx.x*blockDim.x + threadIdx.x;
        if (t_idx >= size) {return;}

        return_values[t_idx] = (int)array[return_list[t_idx]];
    }
    """)
    block_size = 32
    r_size = DTYPE_i(indices.size)
    x_blocks = int(r_size // block_size + 1)

    if dtype == DTYPE_f:
        array_index = mod_array_index.get_function('array_index_float')
        array_index(return_values_d, array_d, indices, r_size, block=(block_size, 1, 1), grid=(x_blocks, 1))
    elif dtype == DTYPE_i:
        array_index = mod_array_index.get_function('array_index_int')
        array_index(return_values_d, array_d, indices, r_size, block=(block_size, 1, 1), grid=(x_blocks, 1))

    return return_values_d


def _gpu_index_update(dest_d, values_d, indices_d):
    """Allows for arbitrary index selecting with numpy arrays.

    Parameters
    ----------
    dest_d : GPUArray
       nD float, array to be updated with new values.
    values_d : GPUArray
        1D float, values to be updated in the destination array.
    indices_d : GPUArray
        1D int, indices to update.

    Returns
    -------
    GPUArray
        Float, input array with values updated.

    """
    size_i = DTYPE_i(values_d.size)

    mod_index_update = SourceModule("""
    __global__ void index_update(float *dest, float *values, int *indices, int size)
    {
        int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(t_idx >= size) {return;}

        dest[indices[t_idx]] = values[t_idx];
    }
    """)
    block_size = 32
    x_blocks = int(size_i // block_size + 1)
    index_update = mod_index_update.get_function('index_update')
    index_update(dest_d, values_d, indices_d, size_i, block=(block_size, 1, 1), grid=(x_blocks, 1))


def _gpu_window_index_f(src_d, indices_d):
    """Returns the values of the peaks from the 2D correlation.

    Parameters
    ----------
    src_d : GPUArray
        2D float, correlation values.
    indices_d : GPUArray
        1D int, indexes of the peaks.

    Returns
    -------
    GPUArray
        1D float.

    """
    _check_inputs(src_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=2)
    n_windows, size = src_d.shape
    _check_inputs(indices_d, array_type=gpuarray.GPUArray, dtype=DTYPE_i, shape=(n_windows,), ndim=1)

    dest_d = gpuarray.empty(n_windows, dtype=DTYPE_f)

    mod_index_update = SourceModule("""
    __global__ void window_index_f(float *dest, float *src, int *indices, int size, int n_windows)
    {
        int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (t_idx >= n_windows) {return;}

        dest[t_idx] = src[t_idx * size + indices[t_idx]];
    }
    """)
    block_size = 32
    x_blocks = ceil(n_windows / block_size)
    index_update = mod_index_update.get_function('window_index_f')
    index_update(dest_d, src_d, indices_d, DTYPE_i(size), DTYPE_i(n_windows), block=(block_size, 1, 1),
                 grid=(x_blocks, 1))

    return dest_d


def _gpu_remove_nan_f(f_d):
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
        if (value != value) {f[t_idx] = 0;}
    }
    """)
    block_size = 32
    x_blocks = ceil(size_i / block_size)
    index_update = mod_replace_nan_f.get_function('replace_nan_f')
    index_update(f_d, size_i, block=(block_size, 1, 1), grid=(x_blocks, 1))


def _check_inputs(*arrays, array_type=None, dtype=None, shape=None, ndim=None, size=None):
    """Checks that all array inputs match either each other's or the given array type, dtype, shape and dim."""
    if array_type is not None:
        assert all([type(array) == array_type for array in arrays]), 'Inputs must be ({}).'.format(array_type)
    if dtype is not None:
        assert all([array.dtype == dtype for array in arrays]), 'Inputs must have dtype ({}).'.format(dtype)
    if shape is not None:
        assert all(
            [array.shape == shape for array in
             arrays]), 'Inputs must have shape ({}, all must be same shape).'.format(
            shape)
    if ndim is not None:
        assert all([array.ndim == ndim for array in arrays]), 'Inputs must have same ndim ({}).'.format(ndim)
    if size is not None:
        assert all([array.size == size for array in arrays]), 'Inputs must have same size ({}).'.format(size)