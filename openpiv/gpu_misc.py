"""This module contains miscellaneous GPU functions."""
from math import ceil
from abc import ABC, abstractmethod
from numbers import Number

import numpy as np
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
# noinspection PyUnresolvedReferences
import pycuda.autoinit

# Define 32-bit types.
DTYPE_i = np.int32
DTYPE_f = np.float32
DTYPE_c = np.complex64

_BLOCK_SIZE = 64

mod_mask = SourceModule(
    """
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
"""
)


def gpu_mask(f, mask):
    """Mask an array.

    Parameters
    ----------
    f : GPUArray
        nD int or float, frame to be masked.
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

    # This could also just write to f directly.
    f_masked = gpuarray.empty_like(f)

    block_size = _BLOCK_SIZE
    grid_size = ceil(size / block_size)
    if d_type == DTYPE_f:
        mask_gpu = mod_mask.get_function("gpu_mask_f")
    elif d_type == DTYPE_i:
        mask_gpu = mod_mask.get_function("gpu_mask_i")
    else:
        raise ValueError("Wrong data type for f.")
    mask_gpu(
        f_masked, f, mask, DTYPE_i(size), block=(block_size, 1, 1), grid=(grid_size, 1)
    )

    return f_masked


mod_scalar_mod = SourceModule(
    """
__global__ void scalar_mod_f(float *i, float *r, float *f, int m, int size)
{
    // i, r : output arguments
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= size) {return;}

    int f_value = f[t_idx];
    i[t_idx] = f_value / m;
    r[t_idx] = f_value % m;
}

__global__ void scalar_mod_i(int *i, int *r, int *f, int m, int size)
{
    // i, r : output arguments
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= size) {return;}

    int f_value = f[t_idx];
    i[t_idx] = f_value / m;
    r[t_idx] = f_value % m;
}
"""
)


def gpu_scalar_mod(f, m):
    """Returns the integer and remainder of division of a PyCUDA array by a scalar
    integer.

    Parameters
    ----------
    f : GPUArray
        nD int or float, input to be decomposed.
    m : int
        Modulus.

    Returns
    -------
    i, r : GPUArray
        nD int or float, integer and remainder parts of the decomposition.

    """
    _check_arrays(f, array_type=gpuarray.GPUArray)
    assert 0 < m == int(m)
    d_type = f.dtype
    size = f.size

    i = gpuarray.empty_like(f)
    r = gpuarray.empty_like(f)

    block_size = _BLOCK_SIZE
    grid_size = ceil(size / block_size)
    if d_type == DTYPE_f:
        scalar_mod_gpu = mod_scalar_mod.get_function("scalar_mod_f")
    elif d_type == DTYPE_i:
        scalar_mod_gpu = mod_scalar_mod.get_function("scalar_mod_i")
    else:
        raise ValueError("Wrong data type for f.")
    scalar_mod_gpu(
        i,
        r,
        f,
        DTYPE_i(m),
        DTYPE_i(size),
        block=(block_size, 1, 1),
        grid=(grid_size, 1),
    )

    return i, r


mod_replace_nan = SourceModule(
    """
#include <math.h>

__global__ void replace_nan(float *f, int size)
{
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= size) {return;}
    
    // Check for NaNs.
    if (std::isnan(f[t_idx])) {f[t_idx] = 0.0f;}
}
"""
)


def gpu_remove_nan(f):
    """Replaces all NaN from array with zeros.

    Parameters
    ----------
    f : GPUArray
        nD float.

    """
    _check_arrays(f, array_type=gpuarray.GPUArray, dtype=DTYPE_f)
    size = f.size

    block_size = _BLOCK_SIZE
    grid_size = ceil(size / block_size)
    replace_nan = mod_replace_nan.get_function("replace_nan")
    replace_nan(f, DTYPE_i(size), block=(block_size, 1, 1), grid=(grid_size, 1))


mod_replace_negative = SourceModule(
    """
__global__ void replace_negative_f(float *f, int size)
{
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= size) {return;}
    float value = f[t_idx];

    // Check for negative values.
    f[t_idx] = value * (value > 0.0f);
}

__global__ void replace_negative_i(int *f, int size)
{
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= size) {return;}
    float value = f[t_idx];

    // Check for negative values.
    f[t_idx] = value * (value > 0);
}
"""
)


def gpu_remove_negative(f):
    """Replaces all negative values from array with zeros.

    Parameters
    ----------
    f : GPUArray
        nD int or float.

    """
    _check_arrays(f, array_type=gpuarray.GPUArray)
    d_type = f.dtype
    size = f.size

    block_size = _BLOCK_SIZE
    grid_size = ceil(size / block_size)
    if d_type == DTYPE_f:
        replace_negative = mod_replace_negative.get_function("replace_negative_f")
    elif d_type == DTYPE_i:
        replace_negative = mod_replace_negative.get_function("replace_negative_i")
    else:
        raise ValueError("Wrong data type for f.")
    replace_negative(f, DTYPE_i(size), block=(block_size, 1, 1), grid=(grid_size, 1))


class _Validator(ABC):
    """Validates user inputs."""

    def __set_name__(self, owner, name):
        self.public_name = name
        self.private_name = "_" + name

    def __get__(self, obj, objtype=None):
        return getattr(obj, self.private_name)

    def __set__(self, obj, value):
        value = self.validate(value)
        setattr(obj, self.private_name, value)

    @abstractmethod
    def validate(self, value):
        pass


class _Bool(_Validator):
    """Bool from a numeric input."""

    def validate(self, value):
        if not isinstance(value, Number):
            raise TypeError(
                "{} must be either bool or numeric type.".format(self.public_name)
            )
        if value != bool(value):
            raise ValueError("{} must have a boolean value.".format(self.public_name))

        return bool(value) if value is not None else None


class _Number(_Validator):
    """Float from a numeric input.

    Parameters
    ----------
    min_value, max_value : float or None, optional
        Minimum and maximum values, respectively. None for no restriction.
    min-closure, max_closure : float or None, optional
        Closures of the minimum and maximum values, respectively.

    """

    def __init__(
        self,
        min_value=None,
        max_value=None,
        min_closure=True,
        max_closure=True,
        allow_none=False,
    ):
        self.min_value = min_value
        self.max_value = max_value
        self.min_closure = min_closure
        self.max_closure = max_closure
        self.allow_none = allow_none

    def validate(self, value):
        if value is None and self.allow_none:
            return None
        if not isinstance(value, (Number, float, int)):
            or_none = " or None" if self.allow_none else ""
            raise TypeError(
                "{} must be a numeric type{}.".format(self.public_name, or_none)
            )
        value_error = False
        if self.min_value is not None:
            if value < self.min_value or (
                value == self.min_value and not self.min_closure
            ):
                value_error = True
        elif self.max_value is not None:
            if value > self.max_value or (
                value == self.max_value and not self.max_closure
            ):
                value_error = True
        if value_error:
            expected_interval = _get_allowed_interval(
                self.min_value, self.max_value, self.min_closure, self.max_closure
            )
            raise ValueError(
                "{} must be in {}. {}".format(
                    self.public_name, expected_interval, value
                )
            )

        return float(value)


class _Integer(_Number):
    """Int from a numeric, integer input."""

    def validate(self, value):
        if value is None and self.allow_none:
            return None
        if not float.is_integer(_Number.validate(self, value)):
            raise ValueError("{} must have an integer value.".format(self.public_name))

        return int(value)


class _Element(_Validator):
    """Element of a given set.

    Parameters
    ----------
    *allowed_values : str or numeric
        Valid set of values.
    allow_none : bool
        Whether value can be None.

    """

    def __init__(self, *allowed_values, allow_none=False):
        self.allowed_values = set(allowed_values)
        self.allow_none = allow_none

    def validate(self, value):
        if value is None and self.allow_none:
            return None
        if value not in self.allowed_values:
            or_none = " or None" if self.allow_none else ""
            raise ValueError(
                "{} must be one of: {}{}.".format(
                    self.public_name, self.allowed_values, or_none
                )
            )

        return value


class _Subset(_Validator):
    """Subset of a given set.

    Parameters
    ----------
    *allowed_values : str or numeric
        Valid set of values.
    allow_none : bool
        Whether value can be None.

    """

    def __init__(self, *allowed_values, allow_none=False):
        self.allowed_values = set(allowed_values)
        self.allow_none = allow_none

    def validate(self, value):
        if value is None and self.allow_none:
            return None
        if not isinstance(value, (tuple, list, set)):
            value = {value}
        if not self.allowed_values.issuperset(value):
            or_none = " or None" if self.allow_none else ""
            raise ValueError(
                "{} must be in: {}{}.".format(
                    self.public_name, self.allowed_values, or_none
                )
            )

        return value


class _Array(_Validator):
    """Array.

    Parameters
    ----------
    d_type : np.dtype
        dtype of resulting array.
    enforce_type : bool
        Whether the input array must have the desired dtype.
    allow_none : bool
        Whether value can be None.

    """

    def __init__(self, d_type=None, enforce_type=False, allow_none=False):
        self.d_type = d_type
        self.enforce_type = enforce_type
        self.allow_none = allow_none

    def validate(self, array):
        """"""
        if array is None and self.allow_none:
            return
        if not isinstance(array, np.ndarray):
            or_none = " or None" if self.allow_none else ""
            raise TypeError(
                "{} must be an np.ndarray{}.".format(self.public_name, or_none)
            )
        if array.dtype != self.d_type and self.enforce_type:
            raise ValueError(
                "{} must contain {} type.".format(self.public_name, self.d_type)
            )

        return array.astype(self.d_type) if array is not None else None


# TODO
def _get_allowed_interval(min_value, max_value, min_closure, max_closure):
    """Returns a string representation of the allowed interval of the reals."""
    left_value = str(min_value) if min_value is not None else "-∞"
    right_value = str(max_value) if max_value is not None else "+∞"
    left_brace = "[" if min_closure and min_value is not None else "("
    right_brace = "]" if max_closure and max_value is not None else ")"

    return "".join((left_brace, ", ".join((left_value, right_value)), right_brace))


def _check_arrays(
    *arrays,
    array_type=None,
    dtype=None,
    shape=None,
    ndim=None,
    size=None,
    c_contiguous=True
):
    """Checks that all arrays match the given attributes."""
    for array in arrays:
        if array_type is not None:
            _check_array_type(array, array_type)
        if dtype is not None:
            _check_array_dtype(array, dtype)
        if shape is not None:
            _check_array_shape(array, shape)
        if ndim is not None:
            _check_array_ndim(array, ndim)
        if size is not None:
            _check_array_size(array, size)
        if c_contiguous is not None:
            _check_array_c_contiguous(array)


def _check_array_type(array, array_type):
    """Checks that arrays are the correct type."""
    if not isinstance(array, array_type):
        raise TypeError("Array(s) must be {}.")


def _check_array_dtype(array, dtype):
    """Checks that arrays have the correct dtype."""
    if not array.dtype == dtype:
        raise ValueError("Array(s) must have dtype {}.")


def _check_array_shape(array, shape):
    """Checks that arrays have the correct shape."""
    if not array.shape == shape:
        raise ValueError("Array(s) must have shape {}.")


def _check_array_ndim(array, ndim):
    """Checks that arrays have the correct ndim."""
    if not array.ndim == ndim:
        raise ValueError("Array(s) must have ndim {}.")


def _check_array_size(array, size):
    """Checks that arrays have the correct size."""
    if not array.size == size:
        raise ValueError("Array(s) must have size {}.")


def _check_array_c_contiguous(array):
    """Checks that arrays are C-contiguous"""
    if not array.flags.c_contiguous:
        raise TypeError("Array(s) must be C-contiguous.")
