"""This module is for GPU-accelerated validation algorithms."""
# TODO u, v here are too coupled
# TODO there is redundancy in GPU kernels

import time
import logging
from math import ceil

import numpy as np
# Create the PyCUDA context.
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
# import pycuda.cumath as cumath
from pycuda.compiler import SourceModule

from openpiv.gpu_misc import _check_inputs

# Define 32-bit types
DTYPE_i = np.int32
DTYPE_f = np.float32
DTYPE_c = np.complex64

ALLOWED_VALIDATION_METHODS = {'s2n', 'median_velocity', 'mean_velocity', 'rms_velocity'}
DEFAULT_VALIDATION_TOLS = {
    's2n_tol': DTYPE_f(0.1),
    'median_tol': DTYPE_f(2),
    'mean_tol': DTYPE_f(2),
    'rms_tol': DTYPE_f(2),
}


def gpu_validation(u_d, v_d, sig2noise_d=None, validation_method='median_velocity', s2n_tol=None, median_tol=None,
                   mean_tol=None, rms_tol=None):
    """Returns an array indicating which indices need to be validated.

    Parameters
    ----------
    u_d, v_d : GPUArray
        2D float, velocity fields to be validated.
    sig2noise_d : ndarray, optional
        1D or 2D float, signal-to-noise ratio of each velocity.
    validation_method : {tuple, 's2n', 'median_velocity', 'mean_velocity', 'rms_velocity'}, optional
        Method(s) to use for validation.
    s2n_tol : float, optional
        Minimum value for sig2noise.
    median_tol : float, optional
        Tolerance for median velocity validation.
    mean_tol : float, optional
        Tolerance for mean velocity validation.
    rms_tol : float, optional
        Tolerance for rms validation.

    Returns
    -------
    val_list : GPUArray
        2D int, array of indices that need to be validated. 0 indicates that the index needs to be corrected. 1 means
        no correction is needed.
    u_mean_d : GPUArray
        2D float, mean of the velocities surrounding each point in this iteration.
    v_mean_d : GPUArray
        2D float, mean of the velocities surrounding each point in this iteration.

    """
    # 'mean' in this function refers to either the mean or median estimators of the average.
    _check_inputs(u_d, v_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, shape=u_d.shape, ndim=2)
    if sig2noise_d is not None:
        _check_inputs(u_d, v_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, size=u_d.size)
    val_locations_d = None
    u_mean_d = v_mean_d = None

    # Compute the mean velocities to be returned.
    neighbours_d, neighbours_present_d = _gpu_get_neighbours(u_d, v_d)
    u_median_d, v_median_d = _gpu_median_velocity(neighbours_d, neighbours_present_d)
    if 'mean_velocity' in validation_method or 'rms_velocity' in validation_method:
        u_mean_d, v_mean_d = _gpu_mean_velocity(neighbours_d, neighbours_present_d)

    if 's2n' in validation_method:
        assert sig2noise_d is not None, 's2n validation requires sig2noise to be passed.'
        s2n_tol = DTYPE_f(s2n_tol) if s2n_tol is not None else DEFAULT_VALIDATION_TOLS['s2n_tol']

        val_locations_d = _local_validation(sig2noise_d, s2n_tol, val_locations_d)

    if 'median_velocity' in validation_method:
        median_tol = DTYPE_f(median_tol) if median_tol is not None else DEFAULT_VALIDATION_TOLS['median_tol']

        u_median_fluc_d, v_median_fluc_d = _gpu_median_fluc(neighbours_d, neighbours_present_d, u_median_d, v_median_d)

        val_locations_d = _neighbour_validation(u_d, u_median_d, u_median_fluc_d, median_tol, val_locations_d)
        val_locations_d = _neighbour_validation(v_d, v_median_d, v_median_fluc_d, median_tol, val_locations_d)

    if 'mean_velocity' in validation_method:
        mean_tol = DTYPE_f(mean_tol) if mean_tol is not None else DEFAULT_VALIDATION_TOLS['mean_tol']

        u_mean_fluc_d, v_mean_fluc_d = _gpu_mean_fluc(neighbours_d, neighbours_present_d, u_mean_d, v_mean_d)

        val_locations_d = _neighbour_validation(u_d, u_mean_d, u_mean_fluc_d, mean_tol, val_locations_d)
        val_locations_d = _neighbour_validation(v_d, v_mean_d, v_mean_fluc_d, mean_tol, val_locations_d)

    if 'rms_velocity' in validation_method:
        rms_tol = DTYPE_f(rms_tol) if rms_tol is not None else DEFAULT_VALIDATION_TOLS['rms_tol']

        u_rms_d, v_rms_d = _gpu_rms(neighbours_d, neighbours_present_d, u_mean_d, v_mean_d)

        val_locations_d = _neighbour_validation(u_d, u_mean_d, u_rms_d, rms_tol, val_locations_d)
        val_locations_d = _neighbour_validation(v_d, v_mean_d, v_rms_d, rms_tol, val_locations_d)

    return val_locations_d, u_median_d, v_median_d


def _local_validation(f_d, tol, val_locations_d=None):
    """Updates the validation list by checking if the array elements exceed the tolerance."""
    size_i = DTYPE_i(f_d.size)
    tol_f = DTYPE_f(tol)

    if val_locations_d is None:
        val_locations_d = gpuarray.ones_like(f_d)

    mod_validation = SourceModule("""
    __global__ void local_validation(int *val_list, float *sig2noise, float s2n_tol, int size)
    {
        int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (t_idx >= size) {return;}

        val_list[t_idx] = val_list[t_idx] * (sig2noise[t_idx] > s2n_tol);
    }
    """)
    block_size = 32
    grid_size = ceil(size_i / block_size)
    local_validation = mod_validation.get_function("local_validation")
    local_validation(val_locations_d, f_d, tol_f, size_i, block=(block_size, 1, 1), grid=(grid_size, 1))

    return val_locations_d


def _neighbour_validation(f_d, f_mean_d, f_mean_fluc_d, tol, val_locations_d=None):
    """Updates the validation list by checking if the neighbouring elements exceed the tolerance."""
    size_i = DTYPE_i(f_d.size)
    tol_f = DTYPE_f(tol)

    if val_locations_d is None:
        val_locations_d = gpuarray.ones_like(f_d)

    mod_validation = SourceModule("""
    __global__ void neighbour_validation(int *val_list, float *f, float *f_mean, float *f_fluc, float tol, int size)
    {
        int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (t_idx >= size) {return;}

        // a small number is added to prevent singularities in uniform flow (Scarano & Westerweel, 2005)
        int f_validation = fabsf(f[t_idx] - f_mean[t_idx]) / (f_fluc[t_idx] + 0.1) < tol;

        val_list[t_idx] = val_list[t_idx] * f_validation;
    }
    """)
    block_size = 32
    grid_size = ceil(size_i / block_size)
    neighbour_validation = mod_validation.get_function("neighbour_validation")
    neighbour_validation(val_locations_d, f_d, f_mean_d, f_mean_fluc_d, tol_f, size_i, block=(block_size, 1, 1),
                         grid=(grid_size, 1))

    return val_locations_d


def _gpu_find_neighbours(shape, mask_d=None):
    """An array that stores if a point has neighbours in a 3x3 grid surrounding it.

    Parameters
    ----------
    shape : tuple
        Int, number of rows and columns at each iteration.
    mask_d : GPUArray
        2D int, masked values.

    Returns
    -------
    GPUArray
        4D (m, n, 3, 3), whether the point in the field has neighbours.

    """
    m, n = shape
    size_i = DTYPE_i(m * n)
    if mask_d is None:
        pass

    neighbours_present_d = gpuarray.empty((m, n, 3, 3), dtype=DTYPE_i)

    mod_neighbours = SourceModule("""
    __global__ void find_neighbours(int *np, int n, int m, int size)
    {
        // np : neighbours_present
        int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (t_idx >= size) {return;}

        int row_zero = (t_idx >= n);
        int row_max = (t_idx < n * (m - 1));
        int col_zero = (t_idx % n != 0);
        int col_max = (t_idx % n != n - 1);

        // Top Row.
        np[t_idx * 9 + 0] = row_zero * col_zero;
        np[t_idx * 9 + 1] = row_zero;
        np[t_idx * 9 + 2] = row_zero * col_max;

        // Middle row.
        np[t_idx * 9 + 3] = col_zero;
        np[t_idx * 9 + 5] = col_max;
        // Set center to zero--can't be a neighbour for yourself.
        np[t_idx * 9 + 4] = 0;

        // Bottom row.
        np[t_idx * 9 + 6] = row_max * col_zero;
        np[t_idx * 9 + 7] = row_max;
        np[t_idx * 9 + 8] = row_max * col_max;
    }
    """)
    block_size = 32
    grid_size = ceil(size_i / block_size)
    find_neighbours = mod_neighbours.get_function("find_neighbours")
    find_neighbours(neighbours_present_d, DTYPE_i(n), DTYPE_i(m), size_i, block=(block_size, 1, 1), grid=(grid_size, 1))

    return neighbours_present_d


def _gpu_get_neighbours(d_u, d_v):
    """An array that stores the values of the velocity of the neighbours around it.

    Parameters
    ----------
    d_u, d_v : GPUArray
        2D float, u and v velocity.

    Returns
    -------
    GPUArray
        5D float (m, n, 2, 3, 3), values of u and v of the neighbours of a point.

    """
    m, n = d_u.shape
    size_i = DTYPE_i(d_u.size)

    neighbours_d = gpuarray.zeros((m, n, 2, 3, 3), dtype=DTYPE_f)

    neighbours_present_d = _gpu_find_neighbours((m, n))

    mod_get_neighbours = SourceModule("""
    __global__ void get_u_neighbours(float *nb, int *np, float *u, int n, int size)
    {
        // nb - values of the neighbouring points
        // np - 1 if there is a neighbour, 0 if no neighbour
        int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (t_idx >= size) {return;}

        // get neighbouring values
        if (np[t_idx * 9 + 0]) {nb[t_idx * 18 + 0] = u[t_idx - n - 1];}
        if (np[t_idx * 9 + 1]) {nb[t_idx * 18 + 1] = u[t_idx - n];}
        if (np[t_idx * 9 + 2]) {nb[t_idx * 18 + 2] = u[t_idx - n + 1];}

        if (np[t_idx * 9 + 3]) {nb[t_idx * 18 + 3] = u[t_idx - 1];}
        // nb[t_idx * 18 + 4] = 0.0;
        if (np[t_idx * 9 + 5]) {nb[t_idx * 18 + 5] = u[t_idx + 1];}

        if (np[t_idx * 9 + 6]) {nb[t_idx * 18 + 6] = u[t_idx + n - 1];}
        if (np[t_idx * 9 + 7]) {nb[t_idx * 18 + 7] = u[t_idx + n];}
        if (np[t_idx * 9 + 8]) {nb[t_idx * 18 + 8] = u[t_idx + n + 1];}
    }

    __global__ void get_v_neighbours(float *nb, int *np, float *v, int n, int size)
    {
        // nb - values of the neighbouring points
        // np - 1 if there is a neighbour, 0 if no neighbour
        int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (t_idx >= size) {return;}

        // get neighbouring values
        if (np[t_idx * 9 + 0]) {nb[t_idx * 18 + 9] = v[t_idx - n - 1];}
        if (np[t_idx * 9 + 1]) {nb[t_idx * 18 + 10] = v[t_idx - n];}
        if (np[t_idx * 9 + 2]) {nb[t_idx * 18 + 11] = v[t_idx - n + 1];}

        if (np[t_idx * 9 + 3]) {nb[t_idx * 18 + 12] = v[t_idx - 1];}
        // nb[t_idx * 18 + 13] = 0.0;
        if (np[t_idx * 9 + 5]) {nb[t_idx * 18 + 14] = v[t_idx + 1];}

        if (np[t_idx * 9 + 6]) {nb[t_idx * 18 + 15] = v[t_idx + n - 1];}
        if (np[t_idx * 9 + 7]) {nb[t_idx * 18 + 16] = v[t_idx + n];}
        if (np[t_idx * 9 + 8]) {nb[t_idx * 18 + 17] = v[t_idx + n + 1];}
    }
    """)
    block_size = 32
    grid_size = ceil(size_i / block_size)
    get_u_neighbours = mod_get_neighbours.get_function("get_u_neighbours")
    get_v_neighbours = mod_get_neighbours.get_function("get_v_neighbours")
    get_u_neighbours(neighbours_d, neighbours_present_d, d_u, DTYPE_i(n), size_i, block=(block_size, 1, 1),
                     grid=(grid_size, 1))
    get_v_neighbours(neighbours_d, neighbours_present_d, d_v, DTYPE_i(n), size_i, block=(block_size, 1, 1),
                     grid=(grid_size, 1))

    return neighbours_d, neighbours_present_d


def _gpu_median_velocity(neighbours_d, neighbours_present_d):
    """Calculates the median velocity on a 3x3 grid around each point in a velocity field.

    Parameters
    ----------
    neighbours_d: GPUArray
        5D float (m, n, 2, 3, 3), all the neighbouring velocities of every point.
    neighbours_present_d: GPUArray
        4D int (m, n, 3, 3), indicates if a neighbour is present.

    Returns
    -------
    u_median_d, v_median_d : GPUArray
        2D float, mean velocities at each point.

    """
    m, n, _, _, _ = neighbours_d.shape
    size_i = DTYPE_i(m * n)
    u_median_d = gpuarray.zeros((m, n), dtype=DTYPE_f)
    v_median_d = gpuarray.zeros((m, n), dtype=DTYPE_f)

    mod_median_vel = SourceModule("""
    // device-side function to swap elements of two arrays
    __device__ void swap(float *A, int a, int b)
    {
        float tmp_A = A[a];
        A[a] = A[b];
        A[b] = tmp_A;
    }

    // device-side function to compare and swap elements of two arrays
    __device__ void compare(float *A, float *B, int a, int b)
    {
        if (B[a] < B[b])
        {
            swap(A, a, b);
            swap(B, a, b);
        }
        else if (A[a] > A[b] && B[a] == B[b] == 1)
        {
            swap(A, a, b);
            swap(B, a, b);
        }
    }

    // device-side function to do an 8-wire sorting network
    __device__ void sort(float *A, float *B)
    {
        compare(A, B, 0, 1);
        compare(A, B, 2, 3);
        compare(A, B, 4, 5);
        compare(A, B, 6, 7);
        compare(A, B, 0, 2);
        compare(A, B, 1, 3);
        compare(A, B, 4, 6);
        compare(A, B, 5, 7);
        compare(A, B, 1, 2);
        compare(A, B, 5, 6);
        compare(A, B, 0, 4);
        compare(A, B, 3, 7);
        compare(A, B, 1, 5);
        compare(A, B, 2, 6);
        compare(A, B, 1, 4);
        compare(A, B, 3, 6);
        compare(A, B, 2, 4);
        compare(A, B, 3, 5);
        compare(A, B, 3, 4);
    }

    __global__ void u_median_velocity(float *u_median, float *nb, int *np, int size)
    {
        // nb - values of the neighbouring points
        // np - 1 if there is a neighbour, 0 if no neighbour
        int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (t_idx >= size) {return;}

        // loop through neighbours to populate an array to sort
        int i;
        int j = 0;
        float A[8];
        float B[8];
        for (i = 0; i < 9; i++)
        {
            if (i != 4)
            {
                A[j] = nb[t_idx * 18 + i];
                B[j++] = np[t_idx * 9 + i];
            }
        }
        // sort the array
        sort(A, B);

        // count the neighbouring points
        int N = B[0] + B[1] + B[2] + B[3] + B[4] + B[5] + B[6] + B[7];

        // return the median
        if (N % 2 == 0) {u_median[t_idx] = (A[N / 2 - 1] + A[N / 2]) / 2;}
        else {u_median[t_idx] = A[N / 2];}
    }

    __global__ void v_median_velocity(float *v_median, float *nb, int *np, int size)
    {
        // n : value of neighbours
        // np : neighbours present
        int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (t_idx >= size) {return;}

        // loop through neighbours to populate an array to sort
        int i;
        int j = 0;
        float A[8];
        float B[8];
        for (i = 0; i < 9; i++)
        {
            if (i != 4)
            {
                A[j] = nb[t_idx * 18 + 9 + i];
                B[j++] = np[t_idx * 9 + i];
            }
        }
        // sort the array
        sort(A, B);

        // count the neighbouring points
        int N = B[0] + B[1] + B[2] + B[3] + B[4] + B[5] + B[6] + B[7];

        // return the median
        if (N % 2 == 0) {v_median[t_idx] = (A[N / 2 - 1] + A[N / 2]) / 2;}
        else {v_median[t_idx] = A[N / 2];}
    }
    """)
    block_size = 32
    grid_size = ceil(size_i / block_size)
    u_median_velocity = mod_median_vel.get_function("u_median_velocity")
    v_median_velocity = mod_median_vel.get_function("v_median_velocity")
    u_median_velocity(u_median_d, neighbours_d, neighbours_present_d, size_i, block=(block_size, 1, 1),
                      grid=(grid_size, 1))
    v_median_velocity(v_median_d, neighbours_d, neighbours_present_d, size_i, block=(block_size, 1, 1),
                      grid=(grid_size, 1))

    return u_median_d, v_median_d


def _gpu_median_fluc(d_neighbours, d_neighbours_present, d_u_median, d_v_median):
    """Calculates the magnitude of the median velocity fluctuations on a 3x3 grid around each point in a velocity field.

    Parameters
    ----------
    d_neighbours : GPUArray
        5D float, all the neighbouring velocities of every point.
    d_neighbours_present : GPUArray
        4D float, indicates if a neighbour is present.
    d_u_median, d_v_median : GPUArray
        2D float, mean velocities around each point.

    Returns
    -------
    u_median_fluc_d, v_median_fluc_d : GPUArray
        2D float, RMS velocities at each point.

    """
    m, n = d_u_median.shape
    size_i = DTYPE_i(d_u_median.size)

    u_median_fluc_d = gpuarray.zeros((m, n), dtype=DTYPE_f)
    v_median_fluc_d = gpuarray.zeros((m, n), dtype=DTYPE_f)

    mod_median_fluc = SourceModule("""
    // device-side function to swap elements of two arrays
    __device__ void swap(float *A, int a, int b)
    {
        float tmp_A = A[a];
        A[a] = A[b];
        A[b] = tmp_A;
    }

    // device-side function to compare and swap elements of two arrays
    __device__ void compare(float *A, float *B, int a, int b)
    {
        if (B[a] < B[b])
        {
            swap(A, a, b);
            swap(B, a, b);
        }
        else if (A[a] > A[b] && B[a] == B[b] == 1)
        {
            swap(A, a, b);
            swap(B, a, b);
        }
    }

    // device-side function to do an 8-wire sorting network
    __device__ void sort(float *A, float *B)
    {
        compare(A, B, 0, 1);
        compare(A, B, 2, 3);
        compare(A, B, 4, 5);
        compare(A, B, 6, 7);
        compare(A, B, 0, 2);
        compare(A, B, 1, 3);
        compare(A, B, 4, 6);
        compare(A, B, 5, 7);
        compare(A, B, 1, 2);
        compare(A, B, 5, 6);
        compare(A, B, 0, 4);
        compare(A, B, 3, 7);
        compare(A, B, 1, 5);
        compare(A, B, 2, 6);
        compare(A, B, 1, 4);
        compare(A, B, 3, 6);
        compare(A, B, 2, 4);
        compare(A, B, 3, 5);
        compare(A, B, 3, 4);
    }

    __global__ void u_fluc_k(float *u_median_fluc, float *u_median, float *nb, int *np, int size)
    {
        // nb - value of the neighbouring points
        // np - 1 if there is a neighbour, 0 if no neighbour
        int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (t_idx >= size) {return;}

        float u_m = u_median[t_idx];

        // loop through neighbours to populate an array to sort
        int i;
        int j = 0;
        float A[8];
        float B[8];
        for (i = 0; i < 9; i++)
        {
            if (i != 4)
            {
                A[j] = fabsf(nb[t_idx * 18 + i] - u_m);
                B[j++] = np[t_idx * 9 + i];
            }
        }
        // sort the array
        sort(A, B);

        // count the neighbouring points
        int N = B[0] + B[1] + B[2] + B[3] + B[4] + B[5] + B[6] + B[7];

        // return the median
        if (N % 2 == 0) {u_median_fluc[t_idx] = (A[N / 2 - 1] + A[N / 2]) / 2;}
        else {u_median_fluc[t_idx] = A[N / 2];}

    }

    __global__ void v_fluc_k(float *v_median_fluc, float *v_median, float *nb, int *np, int size)
    {
        // nb - value of the neighbouring points
        // np - 1 if there is a neighbour, 0 if no neighbour
        int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (t_idx >= size) {return;}

        float v_m = v_median[t_idx];

        // loop through neighbours to populate an array to sort
        int i;
        int j = 0;
        float A[8];
        float B[8];
        for (i = 0; i < 9; i++)
        {
            if (i != 4)
            {
                A[j] = fabsf(nb[t_idx * 18 + 9 + i] - v_m);
                B[j++] = np[t_idx * 9 + i];
            }
        }
        // sort the array
        sort(A, B);

        // count the neighbouring points
        int N = B[0] + B[1] + B[2] + B[3] + B[4] + B[5] + B[6] + B[7];

        // return the median
        if (N % 2 == 0) {v_median_fluc[t_idx] = (A[N / 2 - 1] + A[N / 2]) / 2;}
        else {v_median_fluc[t_idx] = A[N / 2];}
    }
    """)
    block_size = 32
    grid_size = ceil(size_i / block_size)
    mod_u_fluc = mod_median_fluc.get_function("u_fluc_k")
    mod_v_fluc = mod_median_fluc.get_function("v_fluc_k")
    mod_u_fluc(u_median_fluc_d, d_u_median, d_neighbours, d_neighbours_present, size_i, block=(block_size, 1, 1),
               grid=(grid_size, 1))
    mod_v_fluc(v_median_fluc_d, d_v_median, d_neighbours, d_neighbours_present, size_i, block=(block_size, 1, 1),
               grid=(grid_size, 1))

    return u_median_fluc_d, v_median_fluc_d


def _gpu_mean_velocity(neighbours_d, neighbours_present_d):
    """Calculates the mean velocity on a 3x3 grid around each point in a velocity field.

    Parameters
    ----------
    neighbours_d: GPUArray
        5D float (m, n, 2, 3, 3), all the neighbouring velocities of every point.
    neighbours_present_d: GPUArray
        4D int (m, n, 3, 3), indicates if a neighbour is present.

    Returns
    -------
    u_mean_d, v_mean_d : GPUArray
        2D float, mean velocities at each point.

    """
    m, n, _, _, _ = neighbours_d.shape
    size_i = DTYPE_i(m * n)

    u_mean_d = gpuarray.zeros((m, n), dtype=DTYPE_f)
    v_mean_d = gpuarray.zeros((m, n), dtype=DTYPE_f)

    mod_mean_vel = SourceModule("""
    __global__ void u_mean_velocity(float *u_mean, float *nb, int *np, int size)
    {
        // n : value of neighbours
        // np : neighbours present
        int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (t_idx >= size) {return;}

        // ensure denominator is not zero then compute mean
        float numerator_u = nb[t_idx * 18 + 0] + nb[t_idx * 18 + 1] + nb[t_idx * 18 + 2] + nb[t_idx * 18 + 3]
                            + nb[t_idx * 18 + 5] + nb[t_idx * 18 + 6] + nb[t_idx * 18 + 7] + nb[t_idx * 18 + 8];

        // mean is normalized by number of terms summed
        float denominator = np[t_idx * 9 + 0] + np[t_idx * 9 + 1] + np[t_idx * 9 + 2] + np[t_idx * 9 + 3]
                            + np[t_idx * 9 + 5] + np[t_idx * 9 + 6] + np[t_idx * 9 + 7] + np[t_idx * 9 + 8];

        u_mean[t_idx] = numerator_u / denominator;
    }

    __global__ void v_mean_velocity(float *v_mean, float *nb, int *np, int size)
    {
        // n : value of neighbours
        // np : neighbours present
        int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (t_idx >= size) {return;}

        // ensure denominator is not zero then compute mean
        float numerator_v = nb[t_idx * 18 + 9] + nb[t_idx * 18 + 10] + nb[t_idx * 18 + 11] + nb[t_idx * 18 + 12]
                            + nb[t_idx * 18 + 14] + nb[t_idx * 18 + 15] + nb[t_idx * 18 + 16] + nb[t_idx * 18 + 17];

        // mean is normalized by number of terms summed
        float denominator = np[t_idx * 9 + 0] + np[t_idx * 9 + 1] + np[t_idx * 9 + 2] + np[t_idx * 9 + 3]
                            + np[t_idx * 9 + 5] + np[t_idx * 9 + 6] + np[t_idx * 9 + 7] + np[t_idx * 9 + 8];

        v_mean[t_idx] = numerator_v / denominator;
    }
    """)
    block_size = 32
    grid_size = ceil(size_i / block_size)
    u_mean_velocity = mod_mean_vel.get_function("u_mean_velocity")
    v_mean_velocity = mod_mean_vel.get_function("v_mean_velocity")
    u_mean_velocity(u_mean_d, neighbours_d, neighbours_present_d, size_i, block=(block_size, 1, 1), grid=(grid_size, 1))
    v_mean_velocity(v_mean_d, neighbours_d, neighbours_present_d, size_i, block=(block_size, 1, 1), grid=(grid_size, 1))
    return u_mean_d, v_mean_d


def _gpu_mean_fluc(neighbours_d, neighbours_present_d, u_mean_d, v_mean_d):
    """Calculates the magnitude of the mean velocity fluctuations on a 3x3 grid around each point in a velocity field.

    Parameters
    ----------
    neighbours_d : GPUArray
        5D float, all the neighbouring velocities of every point.
    neighbours_present_d : GPUArray
        4D float, indicates if a neighbour is present.
    u_mean_d, v_mean_d : GPUArray
        2D float, mean velocities around each point.

    Returns
    -------
    u_fluc_d, v_fluc_d : GPUArray
        2D float, rms velocities at each point.

    """
    m, n = u_mean_d.shape
    size_i = DTYPE_i(u_mean_d.size)

    u_fluc_d = gpuarray.zeros((m, n), dtype=DTYPE_f)
    v_fluc_d = gpuarray.zeros((m, n), dtype=DTYPE_f)

    mod_mean_fluc = SourceModule("""
    __global__ void u_fluc_k(float *u_fluc, float *u_mean, float *nb, int *np, int size)
    {
        // nb - value of the neighbouring points
        // np - 1 if there is a neighbour, 0 if no neighbour
        int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (t_idx >= size) {return;}

        // ensure denominator is not zero then compute fluctuations
        float numerator = fabsf(nb[t_idx * 18 + 0] - u_mean[t_idx]) + fabsf(nb[t_idx * 18 + 1] - u_mean[t_idx])
                          + fabsf(nb[t_idx * 18 + 2] - u_mean[t_idx]) + fabsf(nb[t_idx * 18 + 3] - u_mean[t_idx])
                          + fabsf(nb[t_idx * 18 + 5] - u_mean[t_idx]) + fabsf(nb[t_idx * 18 + 6] - u_mean[t_idx])
                          + fabsf(nb[t_idx * 18 + 7] - u_mean[t_idx]) + fabsf(nb[t_idx * 18 + 8] - u_mean[t_idx]);

        // mean is normalized by number of terms summed
        float denominator = np[t_idx * 9 + 0] + np[t_idx * 9 + 1] + np[t_idx * 9 + 2] + np[t_idx * 9 + 3]
                            + np[t_idx * 9 + 5] + np[t_idx * 9 + 6] + np[t_idx * 9 + 7] + np[t_idx * 9 + 8];

        u_fluc[t_idx] = numerator / denominator;
    }

    __global__ void v_fluc_k(float *v_fluc, float *v_mean, float *nb, int *np, int size)
    {
        // nb - value of the neighbouring points
        // np - 1 if there is a neighbour, 0 if no neighbour
        int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (t_idx >= size) {return;}

        // ensure denominator is not zero then compute fluctuations
        float numerator = fabsf(nb[t_idx * 18 + 9] - v_mean[t_idx]) + fabsf(nb[t_idx * 18 + 10] - v_mean[t_idx])
                          + fabsf(nb[t_idx * 18 + 11] - v_mean[t_idx]) + fabsf(nb[t_idx * 18 + 12] - v_mean[t_idx])
                          + fabsf(nb[t_idx * 18 + 14] - v_mean[t_idx]) + fabsf(nb[t_idx * 18 + 15] - v_mean[t_idx])
                          + fabsf(nb[t_idx * 18 + 16] - v_mean[t_idx]) + fabsf(nb[t_idx * 18 + 17] - v_mean[t_idx]);

        // mean is normalized by number of terms summed
        float denominator = np[t_idx * 9 + 0] + np[t_idx * 9 + 1] + np[t_idx * 9 + 2] + np[t_idx * 9 + 3]
                            + np[t_idx * 9 + 5] + np[t_idx * 9 + 6] + np[t_idx * 9 + 7] + np[t_idx * 9 + 8];

        v_fluc[t_idx] = numerator / denominator;
    }
    """)
    block_size = 32
    grid_size = ceil(size_i / block_size)
    mod_u_fluc = mod_mean_fluc.get_function("u_fluc_k")
    mod_v_fluc = mod_mean_fluc.get_function("v_fluc_k")
    mod_u_fluc(u_fluc_d, u_mean_d, neighbours_d, neighbours_present_d, size_i, block=(block_size, 1, 1),
               grid=(grid_size, 1))
    mod_v_fluc(v_fluc_d, v_mean_d, neighbours_d, neighbours_present_d, size_i, block=(block_size, 1, 1),
               grid=(grid_size, 1))

    return u_fluc_d, v_fluc_d


def _gpu_rms(d_neighbours, d_neighbours_present, d_u_mean, d_v_mean):
    """Calculates the rms velocity in a 3x3 grid around each point in a velocity field.

    Parameters
    ----------
    d_neighbours : GPUArray
        5D float (m, n, 2, 3, 3), all the neighbouring velocities of every point.
    d_neighbours_present : GPUArray
        4D int (m, n, 3, 3), indicates if a neighbour is present.
    d_u_mean, d_v_mean : GPUArray
        2D float, mean velocities around each point.

    Returns
    -------
    u_rms_d, v_rms_d : GPUArray
        2D float, RMS velocities at each point.

    """
    m, n = d_u_mean.shape
    size_i = DTYPE_i(d_u_mean.size)

    u_rms_d = gpuarray.zeros((m, n), dtype=DTYPE_f)
    v_rms_d = gpuarray.zeros((m, n), dtype=DTYPE_f)

    mod_rms = SourceModule("""
    __global__ void u_rms_k(float *u_rms, float *u_mean, float *nb, int *np, int size)
    {
        // nb - value of the neighbouring points
        // np - 1 if there is a neighbour, 0 if no neighbour
        int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (t_idx >= size) {return;}

        // rms is normalized by number of terms summed
        float denominator = np[t_idx * 9 + 0] + np[t_idx * 9 + 1] + np[t_idx * 9 + 2] + np[t_idx * 9 + 3]
                            + np[t_idx * 9 + 5] + np[t_idx * 9 + 6] + np[t_idx * 9 + 7] + np[t_idx * 9 + 8];

        // ensure denominator is not zero then compute rms
        if (denominator > 0) {
            float numerator = (powf(nb[t_idx * 18 + 0] - u_mean[t_idx], 2)
                               + powf(nb[t_idx * 18 + 1] - u_mean[t_idx], 2)
                               + powf(nb[t_idx * 18 + 2] - u_mean[t_idx], 2)
                               + powf(nb[t_idx * 18 + 3] - u_mean[t_idx], 2)
                               + powf(nb[t_idx * 18 + 5] - u_mean[t_idx], 2)
                               + powf(nb[t_idx * 18 + 6] - u_mean[t_idx], 2)
                               + powf(nb[t_idx * 18 + 7] - u_mean[t_idx], 2)
                               + powf(nb[t_idx * 18 + 8] - u_mean[t_idx], 2));

            u_rms[t_idx] = sqrtf(numerator / denominator);
        } else {u_rms[t_idx] = 0;}
    }

    __global__ void v_rms_k(float *v_rms, float *v_mean, float *nb, int *np, int size)
    {
        // nb - value of the neighbouring points
        // np - 1 if there is a neighbour, 0 if no neighbour
        int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (t_idx >= size) {return;}

        // rms is normalized by number of terms summed
        float denominator = np[t_idx * 9 + 0] + np[t_idx * 9 + 1] + np[t_idx * 9 + 2] + np[t_idx * 9 + 3]
                            + np[t_idx * 9 + 5] + np[t_idx * 9 + 6] + np[t_idx * 9 + 7] + np[t_idx * 9 + 8];

        // ensure denominator is not zero then compute rms
        if (denominator > 0) {
            float numerator = (powf(nb[t_idx * 18 + 9] - v_mean[t_idx], 2)
                               + powf(nb[t_idx * 18 + 10] - v_mean[t_idx], 2)
                               + powf(nb[t_idx * 18 + 11] - v_mean[t_idx], 2)
                               + powf(nb[t_idx * 18 + 12] - v_mean[t_idx], 2)
                               + powf(nb[t_idx * 18 + 14] - v_mean[t_idx], 2)
                               + powf(nb[t_idx * 18 + 15] - v_mean[t_idx], 2)
                               + powf(nb[t_idx * 18 + 16] - v_mean[t_idx], 2)
                               + powf(nb[t_idx * 18 + 17] - v_mean[t_idx], 2));

            v_rms[t_idx] = sqrtf(numerator / denominator);
        } else {v_rms[t_idx] = 0;}
    }
    """)
    block_size = 32
    grid_size = ceil(size_i / block_size)
    mod_u_rms = mod_rms.get_function("u_rms_k")
    mod_v_rms = mod_rms.get_function("v_rms_k")
    mod_u_rms(u_rms_d, d_u_mean, d_neighbours, d_neighbours_present, size_i, block=(block_size, 1, 1),
              grid=(grid_size, 1))
    mod_v_rms(v_rms_d, d_v_mean, d_neighbours, d_neighbours_present, size_i, block=(block_size, 1, 1),
              grid=(grid_size, 1))

    return u_rms_d, v_rms_d


def __gpu_divergence(u_d, v_d, w):
    """[This function very likely does not work as intended.] Calculates the divergence at each point in a velocity
    field.

    Parameters
    ----------
    u_d, v_d : array
        2D float, velocity field.
    w : int
        Pixel separation between velocity vectors.

    Returns
    -------
    GPUArray
        2D float, divergence at each point.

    """
    w = DTYPE_f(w)
    m, n = DTYPE_i(u_d.shape)
    size_i = DTYPE_i(u_d.size)

    div_d = np.empty_like(u_d, dtype=DTYPE_f)

    mod_div = SourceModule("""
    __global__ void div_k(float *div, float *u, float *v, float ws, int m, int n, int size)
    {
        int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (t_idx >= size) {return;}

        // Avoid the boundary
        if (t_idx >= (m - 1) * n) {return;}
        if (t_idx % n == n - 1) {return;}

        float u1 = u[t_idx + n];
        float v1 = v[t_idx + 1];

        div[t_idx] = (u1 - u[t_idx]) / ws - (v1 - v[t_idx]) / ws;
    }

    __global__ void div_boundary_k(float *div, float *u, float *v, float ws, int m, int n)
    {
        int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (t_idx >= size) {return;}

        // only calculate on the boundary
        if (t_idx < (m - 1) * n && t_idx % n != n - 1) {return;}

        float u1 = u[t_idx - n];
        float v1 = v[t_idx - 1];

        div[t_idx] = (u[t_idx] - u1) / ws - (v[t_idx] - v1) / ws;
    }
    """)
    block_size = 32
    grid_size = ceil(m * n / block_size)
    div_k = mod_div.get_function("div_k")
    div_boundary_k = mod_div.get_function("div_boundary_k")
    div_k(div_d, u_d, v_d, w, m, n, size_i, block=(block_size, 1, 1), grid=(grid_size, 1))
    div_boundary_k(div_d, u_d, v_d, w, m, n, size_i, block=(block_size, 1, 1), grid=(grid_size, 1))

    # Get single case of bottom i = 0, j = n_col - 1.
    div_d[0, int(n - 1)] = (u_d[1, n - 1] - u_d[0, n - 1]) / w - (v_d[0, n - 1] - v_d[0, n - 2]) / w
    div_d[int(m - 1), 0] = (u_d[m - 1, 0] - u_d[m - 2, 0]) / w - (v_d[m - 1, 1] - v_d[m - 1, 0]) / w

    return div_d
