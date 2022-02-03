"""This module is for GPU-accelerated validation algorithms."""

import time
import logging

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


# TODO cleanup this function
def gpu_validation(u_d, v_d, n_row, n_col, spacing, sig2noise=None, s2n_tol=None, median_tol=None, mean_tol=None,
                   rms_tol=None):
    """Returns an array indicating which indices need to be validated.

    Parameters
    ----------
    u_d, v_d : GPUArray
        2D float, velocity fields to be validated.
    n_row, n_col : int
        Number of rows and columns in the velocity field.
    spacing : int
        Number of pixels between each interrogation window center.
    sig2noise : ndarray
        2D float, signal-to-noise ratio of each velocity.
    s2n_tol : float
        Minimum value for sig2noise.
    median_tol : float
        Tolerance for median velocity validation.
    mean_tol : float
        Tolerance for mean velocity validation.
    rms_tol : float
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
    n_row = DTYPE_i(n_row)
    n_col = DTYPE_i(n_col)
    # spacing = DTYPE_f(spacing)
    s2n_tol = DTYPE_f(s2n_tol) if s2n_tol is not None else None
    median_tol = DTYPE_f(median_tol) if median_tol is not None else None
    mean_tol = DTYPE_f(mean_tol) if mean_tol is not None else None
    rms_tol = DTYPE_f(rms_tol) if rms_tol is not None else None

    val_locations_d = gpuarray.ones_like(u_d, dtype=DTYPE_i)

    # Get neighbours information.
    neighbours_d, neighbours_present_d = gpu_get_neighbours(u_d, v_d, n_row, n_col)

    # Compute the mean velocities to be returned.
    u_median_d, v_median_d = gpu_median_vel(neighbours_d, neighbours_present_d, n_row, n_col)

    mod_validation = SourceModule("""
    __global__ void s2n_validation(int *val_list, float *sig2noise, float s2n_tol, int n_row, int n_col)
    {
        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (w_idx >= n_col * n_row) {return;}

        // get the val list
        val_list[w_idx] = val_list[w_idx] * (sig2noise[w_idx] > s2n_tol);
    }

    __global__ void neighbour_validation(int *val_list, float *u, float *v, float *u_nb, float *v_nb, float *u_fluc,
                        float *v_fluc, int n_row, int n_col, float tol)
    {
        // u_nb, v_nb : measurement of velocity of neighbours
        // tol : validation tolerance. usually 2
        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (w_idx >= n_row * n_col) {return;}
        
        // a small number is added to prevent singularities in uniform flow (Scarano & Westerweel, 2005)
        int u_validation = fabsf(u[w_idx] - u_nb[w_idx]) / (u_fluc[w_idx] + 0.1) < tol;
        int v_validation = fabsf(v[w_idx] - v_nb[w_idx]) / (v_fluc[w_idx] + 0.1) < tol;
        
        // get the val list
        val_list[w_idx] = val_list[w_idx] * u_validation * v_validation;
    }
    
    """)
    block_size = 32
    x_blocks = int(n_col * n_row / block_size + 1)

    # S2N VALIDATION
    if s2n_tol is not None:
        d_sig2noise = gpuarray.to_gpu(sig2noise)

        # Launch signal to noise kernel
        s2n = mod_validation.get_function("s2n_validation")
        s2n(val_locations_d, d_sig2noise, s2n_tol, n_row, n_col, block=(block_size, 1, 1), grid=(x_blocks, 1))

    # MEDIAN VALIDATION
    if median_tol is not None:
        # Get median velocity data.
        u_median_fluc_d, v_median_fluc_d = gpu_median_fluc(neighbours_d, neighbours_present_d, u_median_d, v_median_d,
                                                           n_row, n_col)

        # Launch validation kernel.
        neighbour_validation = mod_validation.get_function("neighbour_validation")
        neighbour_validation(val_locations_d, u_d, v_d, u_median_d, v_median_d, u_median_fluc_d, v_median_fluc_d, n_row,
                             n_col, median_tol, block=(block_size, 1, 1), grid=(x_blocks, 1))

    # MEAN VALIDATION
    if mean_tol is not None:
        # Get mean velocity data.
        u_mean_d, v_mean_d = gpu_mean_vel(neighbours_d, neighbours_present_d, n_row, n_col)
        u_mean_fluc_d, v_mean_fluc_d = gpu_mean_fluc(neighbours_d, neighbours_present_d, u_mean_d, v_mean_d, n_row,
                                                     n_col)

        # Launch validation kernel.
        neighbour_validation = mod_validation.get_function("neighbour_validation")
        neighbour_validation(val_locations_d, u_d, v_d, u_mean_d, v_mean_d, u_mean_fluc_d, v_mean_fluc_d, n_row, n_col,
                             mean_tol, block=(block_size, 1, 1), grid=(x_blocks, 1))

    # RMS VALIDATION
    if rms_tol is not None:
        # Get rms velocity data.
        u_mean_d, v_mean_d = gpu_mean_vel(neighbours_d, neighbours_present_d, n_row, n_col)
        u_rms_d, v_rms_d = gpu_rms(neighbours_d, neighbours_present_d, u_mean_d, v_mean_d, n_row, n_col)

        # Launch validation kernel.
        neighbour_validation = mod_validation.get_function("neighbour_validation")
        neighbour_validation(val_locations_d, u_d, v_d, u_mean_d, v_mean_d, u_rms_d, v_rms_d, n_row, n_col, rms_tol,
                             block=(block_size, 1, 1), grid=(x_blocks, 1))

    return val_locations_d, u_median_d, v_median_d


def gpu_find_neighbours(n_row, n_col):
    """An array that stores if a point has neighbours in a 3x3 grid surrounding it.

    Parameters
    ----------
    n_row, n_col : int
        Number of rows and columns at each iteration.

    Returns
    -------
    GPUArray
        4D [n_row, n_col, 3 , 3], whether the point in the field has neighbours.

    """
    neighbours_present_d = gpuarray.to_gpu(np.empty((int(n_row), int(n_col), 3, 3), dtype=DTYPE_i))

    mod_neighbours = SourceModule("""
    __global__ void find_neighbours(int *np, int n_row, int n_col)
    {
        // np = boolean array
        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (w_idx >= n_row * n_col) {return;}

        int row_zero = (w_idx >= n_col);
        int row_max = (w_idx < n_col * (n_row - 1));
        int col_zero = (w_idx % n_col != 0);
        int col_max = (w_idx % n_col != n_col - 1);

        // Top Row
        np[w_idx * 9 + 0] = 1 * row_zero * col_zero;
        np[w_idx * 9 + 1] = 1 * row_zero;
        np[w_idx * 9 + 2] = 1 * row_zero * col_max;

        // Middle row
        np[w_idx * 9 + 3] = 1 * col_zero;
        np[w_idx * 9 + 5] = 1 * col_max;
        // Set center to zero--can't be a neighbour for yourself
        np[w_idx * 9 + 4] = 0;

        // Bottom row
        np[w_idx * 9 + 6] = 1 * row_max * col_zero;
        np[w_idx * 9 + 7] = 1 * row_max;
        np[w_idx * 9 + 8] = 1 * row_max * col_max;
    }
    """)
    block_size = 32
    x_blocks = int(n_col * n_row // block_size + 1)
    find_neighbours = mod_neighbours.get_function("find_neighbours")
    find_neighbours(neighbours_present_d, n_row, n_col, block=(block_size, 1, 1), grid=(x_blocks, 1))

    return neighbours_present_d


def gpu_get_neighbours(d_u, d_v, n_row, n_col):
    """An array that stores the values of the velocity of the neighbours around it.

    Parameters
    ----------
    d_u, d_v : GPUArray
        2D float, u and v velocity.
    n_row, n_col : int
        Number of rows/columns at each iteration.

    Returns
    -------
    GPUArray
        5D [n_row, n_col, 2, 3, 3], values of u and v of the neighbours of a point.

    """
    neighbours_d = gpuarray.zeros((int(n_row), int(n_col), 2, 3, 3), dtype=DTYPE_f)

    # Find neighbours.
    neighbours_present_d = gpu_find_neighbours(n_row, n_col)

    mod_get_neighbours = SourceModule("""
    __global__ void get_u_neighbours(float *n, int *np, float *u, int n_row, int n_col)
    {
        // n - u and v values around each point
        // np - 1 if there is a neighbour, 0 if no neighbour
        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (w_idx >= n_row * n_col) {return;}

        // get velocities
        if (np[w_idx * 9 + 0]) {n[w_idx * 18 + 0] = u[w_idx - n_col - 1];}
        if (np[w_idx * 9 + 1]) {n[w_idx * 18 + 1] = u[w_idx - n_col];}
        if (np[w_idx * 9 + 2]) {n[w_idx * 18 + 2] = u[w_idx - n_col + 1];}

        if (np[w_idx * 9 + 3]) {n[w_idx * 18 + 3] = u[w_idx - 1];}
        // n[w_idx * 18 + 4] = 0.0;
        if (np[w_idx * 9 + 5]) {n[w_idx * 18 + 5] = u[w_idx + 1];}

        if (np[w_idx * 9 + 6]) {n[w_idx * 18 + 6] = u[w_idx + n_col - 1];}
        if (np[w_idx * 9 + 7]) {n[w_idx * 18 + 7] = u[w_idx + n_col];}
        if (np[w_idx * 9 + 8]) {n[w_idx * 18 + 8] = u[w_idx + n_col + 1];}
    }

    __global__ void get_v_neighbours(float *n, int *np, float *v, int n_row, int n_col)
    {
        // n - u and v values around each point
        // np - 1 if there is a neighbour, 0 if no neighbour
        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (w_idx >= n_row * n_col) {return;}

        // get velocities
        if (np[w_idx * 9 + 0]) {n[w_idx * 18 + 9] = v[w_idx - n_col - 1];}
        if (np[w_idx * 9 + 1]) {n[w_idx * 18 + 10] = v[w_idx - n_col];}
        if (np[w_idx * 9 + 2]) {n[w_idx * 18 + 11] = v[w_idx - n_col + 1];}

        if (np[w_idx * 9 + 3]) {n[w_idx * 18 + 12] = v[w_idx - 1];}
        // n[w_idx * 18 + 13] = 0.0;
        if (np[w_idx * 9 + 5]) {n[w_idx * 18 + 14] = v[w_idx + 1];}

        if (np[w_idx * 9 + 6]) {n[w_idx * 18 + 15] = v[w_idx + n_col - 1];}
        if (np[w_idx * 9 + 7]) {n[w_idx * 18 + 16] = v[w_idx + n_col];}
        if (np[w_idx * 9 + 8]) {n[w_idx * 18 + 17] = v[w_idx + n_col + 1];}
    }
    """)
    block_size = 32
    x_blocks = int(n_col * n_row // block_size + 1)
    get_u_neighbours = mod_get_neighbours.get_function("get_u_neighbours")
    get_v_neighbours = mod_get_neighbours.get_function("get_v_neighbours")
    get_u_neighbours(neighbours_d, neighbours_present_d, d_u, n_row, n_col, block=(block_size, 1, 1),
                     grid=(x_blocks, 1))
    get_v_neighbours(neighbours_d, neighbours_present_d, d_v, n_row, n_col, block=(block_size, 1, 1),
                     grid=(x_blocks, 1))

    return neighbours_d, neighbours_present_d


def gpu_mean_vel(neighbours_d, neighbours_present_d, n_row, n_col):
    """Calculates the mean velocity on a 3x3 grid around each point in a velocity field.

    Parameters
    ----------
    neighbours_d: GPUArray
        5D float, all the neighbouring velocities of every point.
    neighbours_present_d: GPUArray
    4D float, indicates if a neighbour is present.
    n_row, n_col : int
        Number of rows and columns of the velocity field.

    Returns
    -------
    u_mean_d, v_mean_d : GPUArray
        2D float, mean velocities at each point.

    """
    u_mean_d = gpuarray.zeros((int(n_row), int(n_col)), dtype=DTYPE_f)
    v_mean_d = gpuarray.zeros((int(n_row), int(n_col)), dtype=DTYPE_f)

    mod_mean_vel = SourceModule("""
    __global__ void u_mean_vel(float *u_mean, float *n, int *np, int n_row, int n_col)
    {
        // n : velocity of neighbours
        // np : neighbours present
        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (w_idx >= n_col * n_row) {return;}
        
        // ensure denominator is not zero then compute mean
        float numerator_u = n[w_idx * 18 + 0] + n[w_idx * 18 + 1] + n[w_idx * 18 + 2] + n[w_idx * 18 + 3]
                            + n[w_idx * 18 + 5] + n[w_idx * 18 + 6] + n[w_idx * 18 + 7] + n[w_idx * 18 + 8];
        
        // mean is normalized by number of terms summed
        float denominator = np[w_idx * 9 + 0] + np[w_idx * 9 + 1] + np[w_idx * 9 + 2] + np[w_idx * 9 + 3]
                            + np[w_idx * 9 + 5] + np[w_idx * 9 + 6] + np[w_idx * 9 + 7] + np[w_idx * 9 + 8];
        
        u_mean[w_idx] = numerator_u / denominator;
    }

    __global__ void v_mean_vel(float *v_mean, float *n, int *np, int n_row, int n_col)
    {
        // n : velocity of neighbours
        // np : neighbours present
        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (w_idx >= n_col * n_row) {return;}
        
        // ensure denominator is not zero then compute mean
        float numerator_v = n[w_idx * 18 + 9] + n[w_idx * 18 + 10] + n[w_idx * 18 + 11] + n[w_idx * 18 + 12]
                            + n[w_idx * 18 + 14] + n[w_idx * 18 + 15] + n[w_idx * 18 + 16] + n[w_idx * 18 + 17];
        
        // mean is normalized by number of terms summed
        float denominator = np[w_idx * 9 + 0] + np[w_idx * 9 + 1] + np[w_idx * 9 + 2] + np[w_idx * 9 + 3]
                            + np[w_idx * 9 + 5] + np[w_idx * 9 + 6] + np[w_idx * 9 + 7] + np[w_idx * 9 + 8];
        
        v_mean[w_idx] = numerator_v / denominator;
    }
    """)
    block_size = 32
    x_blocks = int(n_row * n_col // block_size + 1)
    u_mean_vel = mod_mean_vel.get_function("u_mean_vel")
    v_mean_vel = mod_mean_vel.get_function("v_mean_vel")
    u_mean_vel(u_mean_d, neighbours_d, neighbours_present_d, n_row, n_col, block=(block_size, 1, 1), grid=(x_blocks, 1))
    v_mean_vel(v_mean_d, neighbours_d, neighbours_present_d, n_row, n_col, block=(block_size, 1, 1), grid=(x_blocks, 1))

    return u_mean_d, v_mean_d


def gpu_mean_fluc(neighbours_d, neighbours_present_d, u_mean_d, v_mean_d, n_row, n_col):
    """Calculates the magnitude of the mean velocity fluctuations on a 3x3 grid around each point in a velocity field.

    Parameters
    ----------
    neighbours_d : GPUArray
        5D float, all the neighbouring velocities of every point.
    neighbours_present_d : GPUArray
        4D float, indicates if a neighbour is present.
    u_mean_d, v_mean_d : GPUArray
        2D float, mean velocities around each point.
    n_row, n_col : int
        Number of rows and columns of the velocity field.

    Returns
    -------
    u_fluc_d, v_fluc_d : GPUArray
        2D float, rms velocities at each point.

    """
    u_fluc_d = gpuarray.zeros((int(n_row), int(n_col)), dtype=DTYPE_f)
    v_fluc_d = gpuarray.zeros((int(n_row), int(n_col)), dtype=DTYPE_f)

    mod_mean_fluc = SourceModule("""
    __global__ void u_fluc_k(float *u_fluc, float *u_mean, float *n, int *np, int n_row, int n_col)
    {
        // n : velocity of neighbours
        // np : neighbours present
        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (w_idx >= n_col * n_row) {return;}

        // ensure denominator is not zero then compute fluctuations
        float numerator = fabsf(n[w_idx * 18 + 0] - u_mean[w_idx]) + fabsf(n[w_idx * 18 + 1] - u_mean[w_idx])
                          + fabsf(n[w_idx * 18 + 2] - u_mean[w_idx]) + fabsf(n[w_idx * 18 + 3] - u_mean[w_idx])
                          + fabsf(n[w_idx * 18 + 5] - u_mean[w_idx]) + fabsf(n[w_idx * 18 + 6] - u_mean[w_idx])
                          + fabsf(n[w_idx * 18 + 7] - u_mean[w_idx]) + fabsf(n[w_idx * 18 + 8] - u_mean[w_idx]);
        
        // mean is normalized by number of terms summed
        float denominator = np[w_idx * 9 + 0] + np[w_idx * 9 + 1] + np[w_idx * 9 + 2] + np[w_idx * 9 + 3]
                            + np[w_idx * 9 + 5] + np[w_idx * 9 + 6] + np[w_idx * 9 + 7] + np[w_idx * 9 + 8];

        u_fluc[w_idx] = numerator / denominator;
    }

    __global__ void v_fluc_k(float *v_fluc, float *v_mean, float *n, int *np, int n_row, int n_col)
    {
        // n : velocity of neighbours
        // np : neighbours present
        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (w_idx >= n_col * n_row) {return;}

        // mean is normalized by number of terms summed
        float denominator = np[w_idx * 9 + 0] + np[w_idx * 9 + 1] + np[w_idx * 9 + 2] + np[w_idx * 9 + 3]
                            + np[w_idx * 9 + 5] + np[w_idx * 9 + 6] + np[w_idx * 9 + 7] + np[w_idx * 9 + 8];

        // ensure denominator is not zero then compute fluctuations
        float numerator = fabsf(n[w_idx * 18 + 9] - v_mean[w_idx]) + fabsf(n[w_idx * 18 + 10] - v_mean[w_idx])
                          + fabsf(n[w_idx * 18 + 11] - v_mean[w_idx]) + fabsf(n[w_idx * 18 + 12] - v_mean[w_idx])
                          + fabsf(n[w_idx * 18 + 14] - v_mean[w_idx]) + fabsf(n[w_idx * 18 + 15] - v_mean[w_idx])
                          + fabsf(n[w_idx * 18 + 16] - v_mean[w_idx]) + fabsf(n[w_idx * 18 + 17] - v_mean[w_idx]);

        v_fluc[w_idx] = numerator / denominator;
    }
    """)
    block_size = 32
    x_blocks = int(n_row * n_col // block_size + 1)
    mod_u_fluc = mod_mean_fluc.get_function("u_fluc_k")
    mod_v_fluc = mod_mean_fluc.get_function("v_fluc_k")
    mod_u_fluc(u_fluc_d, u_mean_d, neighbours_d, neighbours_present_d, n_row, n_col, block=(block_size, 1, 1),
               grid=(x_blocks, 1))
    mod_v_fluc(v_fluc_d, v_mean_d, neighbours_d, neighbours_present_d, n_row, n_col, block=(block_size, 1, 1),
               grid=(x_blocks, 1))

    return u_fluc_d, v_fluc_d

# TODO remove syncthreads
def gpu_median_vel(neighbours_d, neighbours_present_d, n_row, n_col):
    """Calculates the median velocity on a 3x3 grid around each point in a velocity field.

    Parameters
    ----------
    neighbours_d: GPUArray
        5D float, all the neighbouring velocities of every point.
    neighbours_present_d: GPUArray
        4D float, indicates if a neighbour is present.
    n_row, n_col : int
        Number of rows and columns of the velocity field.

    Returns
    -------
    u_median_d, v_median_d : GPUArray
        2D float, mean velocities at each point.

    """
    u_median_d = gpuarray.zeros((int(n_row), int(n_col)), dtype=DTYPE_f)
    v_median_d = gpuarray.zeros((int(n_row), int(n_col)), dtype=DTYPE_f)

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

    __global__ void u_median_vel(float *u_median, float *n, int *np, int n_row, int n_col)
    {
        // n : velocity of neighbours
        // np : neighbours present
        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (w_idx >= n_col * n_row) {return;}

        // loop through neighbours to populate an array to sort
        int i;
        int j = 0;
        float A[8];
        float B[8];
        for (i = 0; i < 9; i++)
        {
            if (i != 4)
            {
                A[j] = n[w_idx * 18 + i];
                B[j++] = np[w_idx * 9 + i];
            }
        }

        __syncthreads();

        // sort the array
        sort(A, B);

        __syncthreads();

        // count the neighbouring points
        int N = B[0] + B[1] + B[2] + B[3] + B[4] + B[5] + B[6] + B[7];

        // return the median
        if (N % 2 == 0) {u_median[w_idx] = (A[N / 2 - 1] + A[N / 2]) / 2;}
        else {u_median[w_idx] = A[N / 2];}

        __syncthreads();
    }

    __global__ void v_median_vel(float *v_median, float *n, int *np, int n_row, int n_col)
    {
        // n : velocity of neighbours
        // np : neighbours present
        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (w_idx >= n_col * n_row) {return;}
        
        __syncthreads();

        // loop through neighbours to populate an array to sort
        int i;
        int j = 0;
        float A[8];
        float B[8];
        for (i = 0; i < 9; i++)
        {
            if (i != 4)
            {
                A[j] = n[w_idx * 18 + 9 + i];
                B[j++] = np[w_idx * 9 + i];
            }
        }

        __syncthreads();

        // sort the array
        sort(A, B);
        
        __syncthreads();

        // count the neighbouring points
        int N = B[0] + B[1] + B[2] + B[3] + B[4] + B[5] + B[6] + B[7];

        // return the median
        if (N % 2 == 0) {v_median[w_idx] = (A[N / 2 - 1] + A[N / 2]) / 2;}
        else {v_median[w_idx] = A[N / 2];}

        __syncthreads();
    }
    """)
    block_size = 32
    x_blocks = int(n_row * n_col // block_size + 1)
    u_median_vel = mod_median_vel.get_function("u_median_vel")
    v_median_vel = mod_median_vel.get_function("v_median_vel")
    u_median_vel(u_median_d, neighbours_d, neighbours_present_d, n_row, n_col, block=(block_size, 1, 1),
                 grid=(x_blocks, 1))
    v_median_vel(v_median_d, neighbours_d, neighbours_present_d, n_row, n_col, block=(block_size, 1, 1),
                 grid=(x_blocks, 1))

    return u_median_d, v_median_d

# TODO remove syncthreads
def gpu_median_fluc(d_neighbours, d_neighbours_present, d_u_median, d_v_median, n_row, n_col):
    """Calculates the magnitude of the median velocity fluctuations on a 3x3 grid around each point in a velocity field.

    Parameters
    ----------
    d_neighbours : GPUArray
        5D float, all the neighbouring velocities of every point.
    d_neighbours_present : GPUArray
        4D float, indicates if a neighbour is present.
    d_u_median, d_v_median : GPUArray
        2D float, mean velocities around each point.
    n_row, n_col : int
        Number of rows and columns of the velocity field.

    Returns
    -------
    u_median_fluc_d, v_median_fluc_d : GPUArray
        2D float, RMS velocities at each point.

    """
    u_median_fluc_d = gpuarray.zeros((int(n_row), int(n_col)), dtype=DTYPE_f)
    v_median_fluc_d = gpuarray.zeros((int(n_row), int(n_col)), dtype=DTYPE_f)

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

    __global__ void u_fluc_k(float *u_median_fluc, float *u_median, float *n, int *np, int n_row, int n_col)
    {
        // n : velocity of neighbours
        // np : neighbours present
        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (w_idx >= n_col * n_row) {return;}

        float u_m = u_median[w_idx];

        // loop through neighbours to populate an array to sort
        int i;
        int j = 0;
        float A[8];
        float B[8];
        for (i = 0; i < 9; i++)
        {
            if (i != 4)
            {
                A[j] = fabsf(n[w_idx * 18 + i] - u_m);
                B[j++] = np[w_idx * 9 + i];
            }
        }

        __syncthreads();

        // sort the array
        sort(A, B);

        __syncthreads();

        // count the neighbouring points
        int N = B[0] + B[1] + B[2] + B[3] + B[4] + B[5] + B[6] + B[7];

        // return the median
        if (N % 2 == 0) {u_median_fluc[w_idx] = (A[N / 2 - 1] + A[N / 2]) / 2;}
        else {u_median_fluc[w_idx] = A[N / 2];}

        __syncthreads();
    }

    __global__ void v_fluc_k(float *v_median_fluc, float *v_median, float *n, int *np, int n_row, int n_col)
    {
        // n : velocity of neighbours
        // np : neighbours present
        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (w_idx >= n_col * n_row) {return;}

        float v_m = v_median[w_idx];

        // loop through neighbours to populate an array to sort
        int i;
        int j = 0;
        float A[8];
        float B[8];
        for (i = 0; i < 9; i++)
        {
            if (i != 4)
            {
                A[j] = fabsf(n[w_idx * 18 + 9 + i] - v_m);
                B[j++] = np[w_idx * 9 + i];
            }
        }

        __syncthreads();

        // sort the array
        sort(A, B);

        __syncthreads();

        // count the neighbouring points
        int N = B[0] + B[1] + B[2] + B[3] + B[4] + B[5] + B[6] + B[7];

        // return the median
        if (N % 2 == 0) {v_median_fluc[w_idx] = (A[N / 2 - 1] + A[N / 2]) / 2;}
        else {v_median_fluc[w_idx] = A[N / 2];}

        __syncthreads();
    }
    """)
    block_size = 32
    x_blocks = int(n_row * n_col // block_size + 1)
    mod_u_fluc = mod_median_fluc.get_function("u_fluc_k")
    mod_v_fluc = mod_median_fluc.get_function("v_fluc_k")
    mod_u_fluc(u_median_fluc_d, d_u_median, d_neighbours, d_neighbours_present, n_row, n_col, block=(block_size, 1, 1),
               grid=(x_blocks, 1))
    mod_v_fluc(v_median_fluc_d, d_v_median, d_neighbours, d_neighbours_present, n_row, n_col, block=(block_size, 1, 1),
               grid=(x_blocks, 1))

    return u_median_fluc_d, v_median_fluc_d


def gpu_rms(d_neighbours, d_neighbours_present, d_u_mean, d_v_mean, n_row, n_col):
    """Calculates the rms velocity in a 3x3 grid around each point in a velocity field.

    Parameters
    ----------
    d_neighbours : GPUArray
        5D float, all the neighbouring velocities of every point.
    d_neighbours_present : GPUArray
        4D float, indicates if a neighbour is present.
    d_u_mean, d_v_mean : GPUArray
        2D float, mean velocities around each point.
    n_row, n_col : int
        Number of rows and columns of the velocity field.

    Returns
    -------
    u_rms_d, v_rms_d : GPUArray
        2D float, RMS velocities at each point.

    """
    u_rms_d = gpuarray.zeros((int(n_row), int(n_col)), dtype=DTYPE_f)
    v_rms_d = gpuarray.zeros((int(n_row), int(n_col)), dtype=DTYPE_f)

    mod_rms = SourceModule("""
    __global__ void u_rms_k(float *u_rms, float *u_mean, float *n, int *np, int n_row, int n_col)
    {
        // n : velocity of neighbours
        // np : neighbours present
        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (w_idx >= n_col * n_row) {return;}

        // rms is normalized by number of terms summed
        float denominator = np[w_idx * 9 + 0] + np[w_idx * 9 + 1] + np[w_idx * 9 + 2] + np[w_idx * 9 + 3]
                            + np[w_idx * 9 + 5] + np[w_idx * 9 + 6] + np[w_idx * 9 + 7] + np[w_idx * 9 + 8];

        // ensure denominator is not zero then compute rms
        if(denominator > 0){
            float numerator = (powf(n[w_idx * 18 + 0] - u_mean[w_idx], 2) + powf(n[w_idx * 18 + 1] - u_mean[w_idx], 2)
                              + powf(n[w_idx * 18 + 2] - u_mean[w_idx], 2) + powf(n[w_idx * 18 + 3] - u_mean[w_idx], 2)
                              + powf(n[w_idx * 18 + 5] - u_mean[w_idx], 2) + powf(n[w_idx * 18 + 6] - u_mean[w_idx], 2)
                              + powf(n[w_idx * 18 + 7] - u_mean[w_idx], 2) + powf(n[w_idx * 18 + 8]
                              - u_mean[w_idx], 2));

            u_rms[w_idx] = sqrtf(numerator / denominator);
        }
    }

    __global__ void v_rms_k(float *v_rms, float *v_mean, float *n, int *np, int n_row, int n_col)
    {
        // n : velocity of neighbours
        // np : neighbours present
        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (w_idx >= n_col * n_row) {return;}

        // rms is normalized by number of terms summed
        float denominator = np[w_idx * 9 + 0] + np[w_idx * 9 + 1] + np[w_idx * 9 + 2] + np[w_idx * 9 + 3]
                            + np[w_idx * 9 + 5] + np[w_idx * 9 + 6] + np[w_idx * 9 + 7] + np[w_idx * 9 + 8];

        // ensure denominator is not zero then compute rms
        if (denominator > 0){
            float numerator = (powf(n[w_idx * 18 + 9] - v_mean[w_idx], 2) + powf(n[w_idx * 18 + 10] - v_mean[w_idx], 2)
                              + powf(n[w_idx * 18 + 11] - v_mean[w_idx], 2) + powf(n[w_idx * 18 + 12]
                              - v_mean[w_idx], 2) + powf(n[w_idx * 18 + 14] - v_mean[w_idx], 2)
                              + powf(n[w_idx * 18 + 15] - v_mean[w_idx], 2) + powf(n[w_idx * 18 + 16]
                              - v_mean[w_idx], 2) + powf(n[w_idx * 18 + 17] - v_mean[w_idx], 2));

            v_rms[w_idx] = sqrtf(numerator / denominator);
        }
    }
    """)
    block_size = 32
    x_blocks = int(n_row * n_col // block_size + 1)
    mod_u_rms = mod_rms.get_function("u_rms_k")
    mod_v_rms = mod_rms.get_function("v_rms_k")
    mod_u_rms(u_rms_d, d_neighbours, d_neighbours_present, d_u_mean, n_row, n_col, block=(block_size, 1, 1),
              grid=(x_blocks, 1))
    mod_v_rms(v_rms_d, d_neighbours, d_neighbours_present, d_v_mean, n_row, n_col, block=(block_size, 1, 1),
              grid=(x_blocks, 1))

    return u_rms_d, v_rms_d


def __gpu_divergence(u_d, v_d, w, n_row, n_col):
    """[This function very likely does not work as intended.] Calculates the divergence at each point in a velocity
    field.

    Parameters
    ----------
    u_d, v_d: array
        2D float, velocity field.
    w: int
        Pixel separation between velocity vectors.
    n_row, n_col : int
        Number of rows and columns of the velocity field.

    Returns
    -------
    GPUArray
        2D float, divergence at each point.

    """
    n_row = DTYPE_i(n_row)
    n_col = DTYPE_i(n_col)
    w = DTYPE_f(w)

    div_d = np.empty((int(n_row), int(n_col)), dtype=DTYPE_f)

    mod_div = SourceModule("""
    __global__ void div_k(float *div, float *u, float *v, float w, int n_row, int n_col)
    {
        // w : window size
        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
        int max_idx = n_row * n_col;

        // Avoid the boundary
        if (w_idx >= (n_row - 1) * n_col) {return;}
        if (w_idx % n_col == n_col - 1) {return;}

        float u1 = u[w_idx + n_col];
        float v1 = v[w_idx + 1];

        div[w_idx] = (u1 - u[w_idx]) / w - (v1 - v[w_idx]) / w;
    }

    __global__ void div_boundary_k(float *div, float *u, float *v, float w, int n_row, int n_col)
    {
        // w : window size
        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;

        // only calculate on the boundary
        if (w_idx < (n_row - 1) * n_col && w_idx%n_col != n_col - 1) {return;}

        float u1 = u[w_idx - n_col];
        float v1 = v[w_idx - 1];

        div[w_idx] = (u[w_idx] - u1) / w - (v[w_idx] - v1) / w;
    }
    """)
    block_size = 32
    x_blocks = int(n_row * n_col // block_size + 1)
    div_k = mod_div.get_function("div_k")
    div_boundary_k = mod_div.get_function("div_boundary_k")
    div_k(div_d, u_d, v_d, w, n_row, n_col, block=(block_size, 1, 1), grid=(x_blocks, 1))
    div_boundary_k(div_d, u_d, v_d, w, n_row, n_col, block=(block_size, 1, 1), grid=(x_blocks, 1))

    # Get single case of bottom i = 0, j = n_col-1.
    div_d[0, int(n_col - 1)] = (u_d[1, n_col - 1] - u_d[0, n_col - 1]) / w - (v_d[0, n_col - 1] - v_d[0, n_col - 2]) / w
    div_d[int(n_row - 1), 0] = (u_d[n_row - 1, 0] - u_d[n_row - 2, 0]) / w - (v_d[n_row - 1, 1] - v_d[n_row - 1, 0]) / w

    return div_d
