"""This module is for GPU-accelerated validation algoritms."""

import numpy as np
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

# Define 32-bit types
DTYPE_i = np.int32
DTYPE_f = np.float32


def gpu_validation(d_u, d_v, n_row, n_col, spacing, sig2noise=None, s2n_tol=None, median_tol=None, mean_tol=None, rms_tol=None):
    """Returns an array indicating which indices need to be validated.

    Parameters
    ----------
    d_u, d_v : GPUArray
        2D float, velocity fields to be validated
    n_row, n_col : int
        number of rows and columns in the velocity field
    spacing : int
        number of pixels between each interrogation window center
    sig2noise : ndarray
        2D float, signal to noise ratio of each velocity
    s2n_tol : float
        minimum value for sig2noise
    median_tol : float
        tolerance for median velocity validation
    mean_tol : float
        tolerance for mean velocity validation
    rms_tol : float
        tolerance for rms validation

    Returns
    -------
    val_list : GPUArray
        2D int, array of indices that need to be validated. 0 indicates that the index needs to be corrected. 1 means no correction is needed
    d_u_mean : GPUArray
        2D float, mean of the velocities surrounding each point in this iteration.
    d_v_mean : GPUArray
        2D float, mean of the velocities surrounding each point in this iteration.

    """
    # GPU functions
    mod_validation = SourceModule("""
    __global__ void s2n_validation(int *val_list, float *sig2noise, float s2n_tol, int n_row, int n_col)
    {
        // val_list : list of indices to be validated
        // sig2noise : signal to noise ratio
        // s2n_tol : min sig2noise value
        // n_row, n_col : number of rows and columns in the field

        // indexing
        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (w_idx >= n_col * n_row) {return;}

        // get the val list
        val_list[w_idx] = val_list[w_idx] * (sig2noise[w_idx] > s2n_tol);
    }

    __global__ void neighbour_validation(int *val_list, float *u, float *v, float *u_nb, float *v_nb, float *u_fluc, float *v_fluc, int n_row, int n_col, float tol)
    {
        // val_list: list of locations where validation is needed
        // u, v : velocity at that point
        // u_nb, v_nb : measurement of velocity of neighbours
        // u_fluc, v_fluc : fluctuating velocity measurement of neighbours
        // n_row, n_col : number of rows and columns
        // tol : validation tolerance. usually 2

        // indexing
        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (w_idx >= n_row * n_col) {return;}
        
        // a small number is added to prevent singularities in uniform flow (Scarano & Westerweel, 2005)
        int u_validation = fabsf(u[w_idx] - u_nb[w_idx]) / (u_fluc[w_idx] + 0.1) < tol;
        int v_validation = fabsf(v[w_idx] - v_nb[w_idx]) / (v_fluc[w_idx] + 0.1) < tol;
        
        // get the val list
        val_list[w_idx] = val_list[w_idx] * u_validation * v_validation;
    }
    
    """)

    # create array to store validation list
    d_val_list = gpuarray.ones_like(d_u, dtype=DTYPE_i)

    # cast inputs to appropriate data types
    n_row = DTYPE_i(n_row)
    n_col = DTYPE_i(n_col)
    spacing = DTYPE_f(spacing)
    s2n_tol = DTYPE_f(s2n_tol) if s2n_tol is not None else None
    median_tol = DTYPE_f(median_tol) if median_tol is not None else None
    mean_tol = DTYPE_f(mean_tol) if mean_tol is not None else None
    rms_tol = DTYPE_f(rms_tol) if rms_tol is not None else None

    # GPU settings
    block_size = 32
    x_blocks = int(n_col * n_row / block_size + 1)

    # get neighbours information
    d_neighbours, d_neighbours_present = gpu_get_neighbours(d_u, d_v, n_row, n_col)

    # compute the mean velocities to be returned
    d_u_median, d_v_median = gpu_median_vel(d_neighbours, d_neighbours_present, n_row, n_col)

    # S2N VALIDATION
    if s2n_tol is not None:
        d_sig2noise = gpuarray.to_gpu(sig2noise)

        # Launch signal to noise kernel
        s2n = mod_validation.get_function("s2n_validation")
        s2n(d_val_list, d_sig2noise, s2n_tol, n_row, n_col, block=(block_size, 1, 1), grid=(x_blocks, 1))

        # Free gpu memory
        d_sig2noise.gpudata.free()

    # MEDIAN VALIDATION
    if median_tol is not None:
        # get median velocity data
        d_u_median_fluc, d_v_median_fluc = gpu_median_fluc(d_neighbours, d_neighbours_present, d_u_median, d_v_median, n_row, n_col)

        # launch validation kernel
        neighbour_validation = mod_validation.get_function("neighbour_validation")
        neighbour_validation(d_val_list, d_u, d_v, d_u_median, d_v_median, d_u_median_fluc, d_v_median_fluc, n_row, n_col, median_tol, block=(block_size, 1, 1), grid=(x_blocks, 1))

        # Free gpu memory
        # d_u_median.gpudata.free()
        # d_v_median.gpudata.free()
        d_u_median_fluc.gpudata.free()
        d_v_median_fluc.gpudata.free()

    # MEAN VALIDATION
    if mean_tol is not None:
        # get mean velocity data
        d_u_mean, d_v_mean = gpu_mean_vel(d_neighbours, d_neighbours_present, n_row, n_col)
        d_u_mean_fluc, d_v_mean_fluc = gpu_mean_fluc(d_neighbours, d_neighbours_present, d_u_mean, d_v_mean, n_row, n_col)

        # launch validation kernel
        neighbour_validation = mod_validation.get_function("neighbour_validation")
        neighbour_validation(d_val_list, d_u, d_v, d_u_mean, d_v_mean, d_u_mean_fluc, d_v_mean_fluc, n_row, n_col, mean_tol, block=(block_size, 1, 1), grid=(x_blocks, 1))

        # Free gpu memory
        d_u_mean.gpudata.free()
        d_v_mean.gpudata.free()
        d_u_mean_fluc.gpudata.free()
        d_v_mean_fluc.gpudata.free()

    # RMS VALIDATION
    if rms_tol is not None:
        # get rms velocity data
        d_u_mean, d_v_mean = gpu_mean_vel(d_neighbours, d_neighbours_present, n_row, n_col)
        d_u_rms, d_v_rms = gpu_rms(d_neighbours, d_neighbours_present, d_u_mean, d_v_mean, n_row, n_col)

        # launch validation kernel
        neighbour_validation = mod_validation.get_function("neighbour_validation")
        neighbour_validation(d_val_list, d_u, d_v, d_u_mean, d_v_mean, d_u_rms, d_v_rms, n_row, n_col, rms_tol, block=(block_size, 1, 1), grid=(x_blocks, 1))

        d_u_rms.gpudata.free()
        d_v_rms.gpudata.free()

    # return the final validation list
    val_list = d_val_list.get()

    # Free gpu memory
    d_val_list.gpudata.free()
    d_u.gpudata.free()
    d_v.gpudata.free()
    d_neighbours_present.gpudata.free()
    d_neighbours.gpudata.free()

    return val_list, d_u_median, d_v_median


def gpu_find_neighbours(n_row, n_col):
    """An array that stores if a point has neighbours in a 3x3 grid surrounding it

    Parameters
    ----------
    n_row, n_col : ndarray
        1D int, number of rows and columns at each iteration

    Returns
    -------
    d_neighbours_present : GPUArray
        4D [n_row, n_col, 3 , 3]

    """
    mod_neighbours = SourceModule("""
    __global__ void find_neighbours(int *np, int n_row, int n_col)
    {
        // np = boolean array
        // n_row = number of rows
        // n_col = Number of columns

        // references each IW
        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;

        int row_zero = (w_idx >= n_col);
        int row_max = (w_idx < n_col * (n_row - 1));
        int col_zero = (w_idx % n_col != 0);
        int col_max = (w_idx % n_col != n_col - 1);

        // Top Row
        np[w_idx * 9 + 0] = np[w_idx * 9 + 0] * row_zero;
        np[w_idx * 9 + 1] = np[w_idx * 9 + 1] * row_zero;
        np[w_idx * 9 + 2] = np[w_idx * 9 + 2] * row_zero;

        __syncthreads();

        // Bottom row
        np[w_idx * 9 + 6] = np[w_idx * 9 + 6] * row_max;
        np[w_idx * 9 + 7] = np[w_idx * 9 + 7] * row_max;
        np[w_idx * 9 + 8] = np[w_idx * 9 + 8] * row_max;

        __syncthreads();

        // Left column
        np[w_idx * 9 + 0] = np[w_idx * 9 + 0] * col_zero;
        np[w_idx * 9 + 3] = np[w_idx * 9 + 3] * col_zero;
        np[w_idx * 9 + 6] = np[w_idx * 9 + 6] * col_zero;

        __syncthreads();

        // right column
        np[w_idx * 9 + 2] = np[w_idx * 9 + 2] * col_max;
        np[w_idx * 9 + 5] = np[w_idx * 9 + 5] * col_max;
        np[w_idx * 9 + 8] = np[w_idx * 9 + 8] * col_max;
        
        __syncthreads();
        
        // Set center to zero--can't be a neighbour for yourself
        np[w_idx * 9 + 4] = 0;
    }
    """)

    # GPU settings
    block_size = 32
    x_blocks = int(n_col * n_row // block_size + 1)

    # allocate space for new array
    neighbours_present = np.ones([n_row, n_col, 3, 3], dtype=DTYPE_i)

    assert neighbours_present.dtype == DTYPE_i, "Wrong data type for neighbours present"

    # send data to gpu
    d_neighbours_present = gpuarray.to_gpu(neighbours_present)

    # get and launch kernel
    find_neighbours = mod_neighbours.get_function("find_neighbours")
    find_neighbours(d_neighbours_present, n_row, n_col, block=(block_size, 1, 1), grid=(x_blocks, 1))

    return d_neighbours_present


def gpu_get_neighbours(d_u, d_v, n_row, n_col):
    """An array that stores the values of the velocity of the neighbours around it.

    Parameters
    ----------
    d_u, d_v : GPUArray
        2D float, u and v velocity
    n_row, n_col : ndarray
        1D int, number of rows/columns at each iteration

    Returns
    -------
    neighbours : array - 5D [n_row, n_col, 2, 3, 3]
        stores the values of u and v of the neighbours of a point

    """
    mod_get_neighbours = SourceModule("""
    __global__ void get_u_neighbours(float *n, int *np, float *u, int n_row, int n_col)
    {
        // n - u and v values around each point
        // np - 1 if there is a neighbour, 0 if no neighbour
        // u, v - u and v velocities
        // n_row, n_col - number of rows and columns

        // references each IW
        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (w_idx >= n_row * n_col) {return;}

        // get velocities
        if (np[w_idx * 9 + 0]) {n[w_idx * 18 + 0] = u[w_idx - n_col - 1];}
        if (np[w_idx * 9 + 1]) {n[w_idx * 18 + 1] = u[w_idx - n_col];}
        if (np[w_idx * 9 + 2]) {n[w_idx * 18 + 2] = u[w_idx - n_col + 1];}

        __syncthreads();

        if (np[w_idx * 9 + 3]) {n[w_idx * 18 + 3] = u[w_idx - 1];}
        // n[w_idx * 18 + 4] = 0.0;
        if (np[w_idx * 9 + 5]) {n[w_idx * 18 + 5] = u[w_idx + 1];}

        __syncthreads();

        if (np[w_idx * 9 + 6]) {n[w_idx * 18 + 6] = u[w_idx + n_col - 1];}
        if (np[w_idx * 9 + 7]) {n[w_idx * 18 + 7] = u[w_idx + n_col];}
        if (np[w_idx * 9 + 8]) {n[w_idx * 18 + 8] = u[w_idx + n_col + 1];}

        __syncthreads();
    }

    __global__ void get_v_neighbours(float *n, int *np, float *v, int n_row, int n_col)
    {
        // n - u and v values around each point
        // np - 1 if there is a neighbour, 0 if no neighbour
        // u, v - u and v velocities
        // n_row, n_col - number of rows and columns

        // references each IW
        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (w_idx >= n_row * n_col) {return;}

        // get velocities
        if (np[w_idx * 9 + 0]) {n[w_idx * 18 + 9] = v[w_idx - n_col - 1];}
        if (np[w_idx * 9 + 1]) {n[w_idx * 18 + 10] = v[w_idx - n_col];}
        if (np[w_idx * 9 + 2]) {n[w_idx * 18 + 11] = v[w_idx - n_col + 1];}

        __syncthreads();

        if (np[w_idx * 9 + 3]) {n[w_idx * 18 + 12] = v[w_idx - 1];}
        // n[w_idx * 18 + 13] = 0.0;
        if (np[w_idx * 9 + 5]) {n[w_idx * 18 + 14] = v[w_idx + 1];}

        __syncthreads();

        if (np[w_idx * 9 + 6]) {n[w_idx * 18 + 15] = v[w_idx + n_col - 1];}
        if (np[w_idx * 9 + 7]) {n[w_idx * 18 + 16] = v[w_idx + n_col];}
        if (np[w_idx * 9 + 8]) {n[w_idx * 18 + 17] = v[w_idx + n_col + 1];}

        __syncthreads();
    }
    """)

    # Get GPU grid dimensions and function
    block_size = 32
    x_blocks = int(n_col * n_row // block_size + 1)
    get_u_neighbours = mod_get_neighbours.get_function("get_u_neighbours")
    get_v_neighbours = mod_get_neighbours.get_function("get_v_neighbours")

    # find neighbours
    d_neighbours_present = gpu_find_neighbours(n_row, n_col)
    neighbours = np.zeros((n_row, n_col, 2, 3, 3), dtype=DTYPE_f)

    # send data to the gpu
    d_neighbours = gpuarray.to_gpu(neighbours)

    # Get u and v data
    get_u_neighbours(d_neighbours, d_neighbours_present, d_u, n_row, n_col, block=(block_size, 1, 1), grid=(x_blocks, 1))
    get_v_neighbours(d_neighbours, d_neighbours_present, d_v, n_row, n_col, block=(block_size, 1, 1), grid=(x_blocks, 1))

    return d_neighbours, d_neighbours_present


def gpu_mean_vel(d_neighbours, d_neighbours_present, n_row, n_col):
    """Calculates the mean velocity on a 3x3 grid around each point in a velocity field.

    Parameters
    ----------
    d_neighbours: GPUArray
        5D float, all the neighbouring velocities of every point
    d_neighbours_present: GPUArray
    4D float, indicates if a neighbour is present
    n_row, n_col : int
        number of rows and columns of the velocity field

    Returns
    -------
    u_mean, v_mean : array
        2D float, mean velocities at each point

    """
    mod_mean_vel = SourceModule("""
    __global__ void u_mean_vel(float *u_mean, float *n, int *np, int n_row, int n_col)
    {
        // mean_u : mean velocity of surrounding points
        // n : velocity of neighbours
        // np : neighbours present
        // n_row, n_col : number of rows and columns
        
        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (w_idx >= n_col * n_row) {return;}
        
        // ensure denominator is not zero then compute mean
        float numerator_u = n[w_idx * 18 + 0] + n[w_idx * 18 + 1] + n[w_idx * 18 + 2] + n[w_idx * 18 + 3] + \
                            n[w_idx * 18 + 5] + n[w_idx * 18 + 6] + n[w_idx * 18 + 7] + n[w_idx * 18 + 8];
        
        // mean is normalized by number of terms summed
        float denominator = np[w_idx * 9 + 0] + np[w_idx * 9 + 1] + np[w_idx * 9 + 2] + np[w_idx * 9 + 3] + \
                            np[w_idx * 9 + 5] + np[w_idx * 9 + 6] + np[w_idx * 9 + 7] + np[w_idx * 9 + 8];
        
        u_mean[w_idx] = numerator_u / denominator;
        
        __syncthreads();
    }

    __global__ void v_mean_vel(float *v_mean, float *n, int *np, int n_row, int n_col)
    {
        // mean_v : mean velocity of surrounding points
        // n : velocity of neighbours
        // np : neighbours present
        // n_row, n_col : number of rows and columns
        
        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (w_idx >= n_col * n_row) {return;}
        
        // ensure denominator is not zero then compute mean
        float numerator_v = n[w_idx * 18 + 9] + n[w_idx * 18 + 10] + n[w_idx * 18 + 11] + n[w_idx * 18 + 12] + \
                            n[w_idx * 18 + 14] + n[w_idx * 18 + 15] + n[w_idx * 18 + 16] + n[w_idx * 18 + 17];
        
        // mean is normalized by number of terms summed
        float denominator = np[w_idx * 9 + 0] + np[w_idx * 9 + 1] + np[w_idx * 9 + 2] + np[w_idx * 9 + 3] + \
                            np[w_idx * 9 + 5] + np[w_idx * 9 + 6] + np[w_idx * 9 + 7] + np[w_idx * 9 + 8];
        
        v_mean[w_idx] = numerator_v / denominator;
        
        __syncthreads();
    }
    """)

    # allocate space for arrays
    u_mean = np.zeros((n_row, n_col), dtype=DTYPE_f)
    v_mean = np.zeros((n_row, n_col), dtype=DTYPE_f)

    # define GPU data
    # block_size = 16
    block_size = 32
    x_blocks = int(n_row * n_col // block_size + 1)

    # send data to gpu
    d_u_mean = gpuarray.to_gpu(u_mean)
    d_v_mean = gpuarray.to_gpu(v_mean)

    # get and launch kernel
    u_mean_vel = mod_mean_vel.get_function("u_mean_vel")
    v_mean_vel = mod_mean_vel.get_function("v_mean_vel")
    u_mean_vel(d_u_mean, d_neighbours, d_neighbours_present, n_row, n_col, block=(block_size, 1, 1), grid=(x_blocks, 1))
    v_mean_vel(d_v_mean, d_neighbours, d_neighbours_present, n_row, n_col, block=(block_size, 1, 1), grid=(x_blocks, 1))

    return d_u_mean, d_v_mean


def gpu_mean_fluc(d_neighbours, d_neighbours_present, d_u_mean, d_v_mean, n_row, n_col):
    """Calculates the magnitude of the mean velocity fluctuations on a 3x3 grid around each point in a velocity field.

    Parameters
    ----------
    d_neighbours : GPUArray
        5D float, all the neighbouring velocities of every point
    d_neighbours_present : GPUArray
        4D float, indicates if a neighbour is present
    d_u_mean, d_v_mean : GPUArray
        2D float, mean velocities around each point
    n_row, n_col : int
        number of rows and columns of the velocity field

    Returns
    -------
    d_u_fluc, d_v_fluc : GPUArray
        2D float, rms velocities at each point

    """
    mod_mean_fluc = SourceModule("""
    __global__ void u_fluc_k(float *u_fluc, float *u_mean, float *n, int *np, int n_row, int n_col)
    {
        // u_fluc : velocity fluctuations of surrounding points
        // u_mean : mean velocity of surrounding points
        // n : velocity of neighbours
        // np : neighbours present
        // n_row, n_col : number of rows and columns

        // index
        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (w_idx >= n_col * n_row) {return;}

        // ensure denominator is not zero then compute fluctuations
        float numerator = fabsf(n[w_idx * 18 + 0] - u_mean[w_idx]) + fabsf(n[w_idx * 18 + 1] - u_mean[w_idx]) + \
                          fabsf(n[w_idx * 18 + 2] - u_mean[w_idx]) + fabsf(n[w_idx * 18 + 3] - u_mean[w_idx]) + \
                          fabsf(n[w_idx * 18 + 5] - u_mean[w_idx]) + fabsf(n[w_idx * 18 + 6] - u_mean[w_idx]) + \
                          fabsf(n[w_idx * 18 + 7] - u_mean[w_idx]) + fabsf(n[w_idx * 18 + 8] - u_mean[w_idx]);
        
        // mean is normalized by number of terms summed
        float denominator = np[w_idx * 9 + 0] + np[w_idx * 9 + 1] + np[w_idx * 9 + 2] + np[w_idx * 9 + 3] + \
                            np[w_idx * 9 + 5] + np[w_idx * 9 + 6] + np[w_idx * 9 + 7] + np[w_idx * 9 + 8];

        u_fluc[w_idx] = numerator / denominator;

        __syncthreads();
    }

    __global__ void v_fluc_k(float *v_fluc, float *v_mean, float *n, int *np, int n_row, int n_col)
    {
        // v_fluc : velocity fluctuations of surrounding points
        // v_mean : mean velocity of surrounding points
        // n : velocity of neighbours
        // np : neighbours present
        // n_row, n_col : number of rows and columns

        // index
        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (w_idx >= n_col * n_row) {return;}

        // mean is normalized by number of terms summed
        float denominator = np[w_idx * 9 + 0] + np[w_idx * 9 + 1] + np[w_idx * 9 + 2] + np[w_idx * 9 + 3] + \
                            np[w_idx * 9 + 5] + np[w_idx * 9 + 6] + np[w_idx * 9 + 7] + np[w_idx * 9 + 8];

        // ensure denominator is not zero then compute fluctuations
        float numerator = fabsf(n[w_idx * 18 + 9] - v_mean[w_idx]) + fabsf(n[w_idx * 18 + 10] - v_mean[w_idx]) + \
                          fabsf(n[w_idx * 18 + 11] - v_mean[w_idx]) + fabsf(n[w_idx * 18 + 12] - v_mean[w_idx]) + \
                          fabsf(n[w_idx * 18 + 14] - v_mean[w_idx]) + fabsf(n[w_idx * 18 + 15] - v_mean[w_idx]) + \
                          fabsf(n[w_idx * 18 + 16] - v_mean[w_idx]) + fabsf(n[w_idx * 18 + 17] - v_mean[w_idx]);

        v_fluc[w_idx] = numerator / denominator;

        __syncthreads();
    }
    """)

    # allocate space for data
    u_rms = np.zeros((n_row, n_col), dtype=DTYPE_f)
    v_rms = np.zeros((n_row, n_col), dtype=DTYPE_f)

    # define GPU data
    block_size = 32
    x_blocks = int(n_row * n_col // block_size + 1)

    # send data to gpu
    d_u_fluc = gpuarray.to_gpu(u_rms)
    d_v_fluc = gpuarray.to_gpu(v_rms)

    # get and launch kernel
    mod_u_fluc = mod_mean_fluc.get_function("u_fluc_k")
    mod_v_fluc = mod_mean_fluc.get_function("v_fluc_k")
    mod_u_fluc(d_u_fluc, d_u_mean, d_neighbours, d_neighbours_present, n_row, n_col, block=(block_size, 1, 1), grid=(x_blocks, 1))
    mod_v_fluc(d_v_fluc, d_v_mean, d_neighbours, d_neighbours_present, n_row, n_col, block=(block_size, 1, 1), grid=(x_blocks, 1))

    return d_u_fluc, d_v_fluc


def gpu_median_vel(d_neighbours, d_neighbours_present, n_row, n_col):
    """Calculates the median velocity on a 3x3 grid around each point in a velocity field.

    Parameters
    ----------
    d_neighbours: GPUArray
        5D float, all the neighbouring velocities of every point
    d_neighbours_present: GPUArray
        4D float, indicates if a neighbour is present
    n_row, n_col : int
        number of rows and columns of the velocity field

    Returns
    -------
    u_median, v_median : ndarray
        2D float, mean velocities at each point

    """
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
        // u_median : median velocity of surrounding points
        // n : velocity of neighbours
        // np : neighbours present
        // n_row, n_col : number of rows and columns

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
        // v_median : median velocity of surrounding points
        // n : velocity of neighbours
        // np : neighbours present
        // n_row, n_col : number of rows and columns

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

    # allocate space for arrays
    u_median = np.zeros((n_row, n_col), dtype=DTYPE_f)
    v_median = np.zeros((n_row, n_col), dtype=DTYPE_f)

    # define GPU data
    block_size = 32
    x_blocks = int(n_row * n_col // block_size + 1)

    # send data to gpu
    d_u_median = gpuarray.to_gpu(u_median)
    d_v_median = gpuarray.to_gpu(v_median)

    # get and launch kernel
    u_median_vel = mod_median_vel.get_function("u_median_vel")
    v_median_vel = mod_median_vel.get_function("v_median_vel")
    u_median_vel(d_u_median, d_neighbours, d_neighbours_present, n_row, n_col, block=(block_size, 1, 1), grid=(x_blocks, 1))
    v_median_vel(d_v_median, d_neighbours, d_neighbours_present, n_row, n_col, block=(block_size, 1, 1), grid=(x_blocks, 1))

    return d_u_median, d_v_median


def gpu_median_fluc(d_neighbours, d_neighbours_present, d_u_median, d_v_median, n_row, n_col):
    """Calculates the magnitude of the median velocity fluctuations on a 3x3 grid around each point in a velocity field.

    Parameters
    ----------
    d_neighbours : GPUArray
        5D float, all the neighbouring velocities of every point
    d_neighbours_present : GPUArray
        4D float, indicates if a neighbour is present
    d_u_median, d_v_median : GPUArray
        2D float, mean velocities around each point
    n_row, n_col : int
        number of rows and columns of the velocity field

    Returns
    -------
    d_u_median_fluc, d_v_median_fluc : GPUArray
        2D float, rms velocities at each point

    """
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
        // u_fluc : velocity fluctuations of surrounding points
        // u_median : median velocity of surrounding points
        // n : velocity of neighbours
        // np : neighbours present
        // n_row, n_col : number of rows and columns

        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (w_idx >= n_col * n_row) {return;}

        __syncthreads();

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
        // v_fluc : velocity fluctuations of surrounding points
        // v_median : mean velocity of surrounding points
        // n : velocity of neighbours
        // np : neighbours present
        // n_row, n_col : number of rows and columns

        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (w_idx >= n_col * n_row) {return;}

        __syncthreads();

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

    # allocate space for data
    u_median_fluc = np.zeros((n_row, n_col), dtype=DTYPE_f)
    v_median_fluc = np.zeros((n_row, n_col), dtype=DTYPE_f)

    # define GPU data
    block_size = 32
    x_blocks = int(n_row * n_col // block_size + 1)

    # send data to gpu
    d_u_median_fluc = gpuarray.to_gpu(u_median_fluc)
    d_v_median_fluc = gpuarray.to_gpu(v_median_fluc)

    # get and launch kernel
    mod_u_fluc = mod_median_fluc.get_function("u_fluc_k")
    mod_v_fluc = mod_median_fluc.get_function("v_fluc_k")
    mod_u_fluc(d_u_median_fluc, d_u_median, d_neighbours, d_neighbours_present, n_row, n_col, block=(block_size, 1, 1), grid=(x_blocks, 1))
    mod_v_fluc(d_v_median_fluc, d_v_median, d_neighbours, d_neighbours_present, n_row, n_col, block=(block_size, 1, 1), grid=(x_blocks, 1))

    return d_u_median_fluc, d_v_median_fluc


def gpu_rms(d_neighbours, d_neighbours_present, d_u_mean, d_v_mean, n_row, n_col):
    """Calculates the rms velocity in a 3x3 grid around each point in a velocity field.

    Parameters
    ----------
    d_neighbours : GPUArray
        5D float, all the neighbouring velocities of every point
    d_neighbours_present : GPUArray
        4D float, indicates if a neighbour is present
    d_u_mean, d_v_mean : GPUArray
        2D float, mean velocities around each point
    n_row, n_col : int
        number of rows and columns of the velocity field

    Returns
    -------
    d_u_rms, d_v_rms : GPUArray
        2D float32, rms velocities at each point

    """
    mod_rms = SourceModule("""
    __global__ void u_rms_k(float *u_rms, float *u_mean, float *n, int *np, int n_row, int n_col)
    {
        // u_rms : rms of surrounding points
        // u_mean : mean velocity of surrounding points
        // n : velocity of neighbours
        // np : neighbours present
        // n_row, n_col: number of rows and columns

        // index
        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(w_idx >= n_col * n_row){return;}

        // rms is normalized by number of terms summed
        float denominator = np[w_idx * 9 + 0] + np[w_idx * 9 + 1] + np[w_idx * 9 + 2] + np[w_idx * 9 + 3] + \
                            np[w_idx * 9 + 5] + np[w_idx * 9 + 6] + np[w_idx * 9 + 7] + np[w_idx * 9 + 8];

        __syncthreads();

        // ensure denominator is not zero then compute rms
        if(denominator > 0){
            float numerator = (powf(n[w_idx * 18 + 0] - u_mean[w_idx], 2) + powf(n[w_idx * 18 + 1] - u_mean[w_idx], 2) + powf(n[w_idx * 18 + 2] - u_mean[w_idx], 2) + powf(n[w_idx * 18 + 3] - u_mean[w_idx], 2) + powf(n[w_idx * 18 + 5] - u_mean[w_idx], 2) + powf(n[w_idx * 18 + 6] - u_mean[w_idx], 2) + powf(n[w_idx * 18 + 7] - u_mean[w_idx], 2) + powf(n[w_idx * 18 + 8] - u_mean[w_idx], 2));

            u_rms[w_idx] = sqrtf(numerator / denominator);
        }

        __syncthreads();
    }

    __global__ void v_rms_k(float *v_rms, float *v_mean, float *n, int *np, int n_row, int n_col)
    {
        // v_rms : rms of surrounding points
        // v_mean : mean velocity of surrounding points
        // n : velocity of neighbours
        // np : neighbours present
        // n_row, n_col: number of rows and columns

        // index
        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (w_idx >= n_col * n_row) {return;}

        // rms is normalized by number of terms summed
        float denominator = np[w_idx * 9 + 0] + np[w_idx * 9 + 1] + np[w_idx * 9 + 2] + np[w_idx * 9 + 3] + \
                            np[w_idx * 9 + 5] + np[w_idx * 9 + 6] + np[w_idx * 9 + 7] + np[w_idx * 9 + 8];

        __syncthreads();

        // ensure denominator is not zero then compute rms
        if (denominator > 0){
            float numerator = (powf(n[w_idx * 18 + 9] - v_mean[w_idx], 2) + powf(n[w_idx * 18 + 10] - v_mean[w_idx], 2) + powf(n[w_idx * 18 + 11] - v_mean[w_idx], 2) + powf(n[w_idx * 18 + 12] - v_mean[w_idx], 2) + powf(n[w_idx * 18 + 14] - v_mean[w_idx], 2) + powf(n[w_idx * 18 + 15] - v_mean[w_idx], 2) + powf(n[w_idx * 18 + 16] - v_mean[w_idx], 2) + powf(n[w_idx * 18 + 17] - v_mean[w_idx], 2));

            v_rms[w_idx] = sqrtf(numerator / denominator);
        }

        __syncthreads();
    }
    """)

    # allocate space for data
    u_rms = np.zeros((n_row, n_col), dtype=DTYPE_f)
    v_rms = np.zeros((n_row, n_col), dtype=DTYPE_f)

    # define GPU data
    # block_size = 16
    block_size = 32
    x_blocks = int(n_row * n_col // block_size + 1)

    # send data to gpu
    d_u_rms = gpuarray.to_gpu(u_rms)
    d_v_rms = gpuarray.to_gpu(v_rms)

    # get and launch kernel
    mod_u_rms = mod_rms.get_function("u_rms_k")
    mod_v_rms = mod_rms.get_function("v_rms_k")
    mod_u_rms(d_u_rms, d_neighbours, d_neighbours_present, d_u_mean, n_row, n_col, block=(block_size, 1, 1), grid=(x_blocks, 1))
    mod_v_rms(d_v_rms, d_neighbours, d_neighbours_present, d_v_mean, n_row, n_col, block=(block_size, 1, 1), grid=(x_blocks, 1))

    return d_u_rms, d_v_rms


def __gpu_divergence(d_u, d_v, w, n_row, n_col):
    """[This function very likely does not work as intended.] Calculates the divergence at each point in a velocity
    field.

    Parameters
    ----------
    d_u, d_v: array - 2D float
        velocity field
    w: int
        pixel separation between velocity vectors
    n_row, n_col : int
        number of rows and columns of the velocity field

    Returns
    -------
    div : array - 2D float32
        divergence at each point

    """
    mod_div = SourceModule("""
    __global__ void div_k(float *div, float *u, float *v, float w, int n_row, int n_col)
    {
        // u : u velocity
        // v : v velocity
        // w : window size
        // n_row, n_col : number of rows and columns

        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
        int max_idx = n_row * n_col;

        // Avoid the boundary
        if(w_idx >= (n_row - 1) * n_col){return;}
        if(w_idx%n_col == n_col - 1){return;}

        float u1 = u[w_idx + n_col];
        float v1 = v[w_idx + 1];

        __syncthreads();

        div[w_idx] = (u1 - u[w_idx]) / w - (v1 - v[w_idx]) / w;
    }

    __global__ void div_boundary_k(float *div, float *u, float *v, float w, int n_row, int n_col)
    {
        // u : u velocity
        // v : v velocity
        // w : window size
        // n_row, n_col : number of rows and columns

        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;

        // only calculate on the boundary
        if(w_idx < (n_row - 1) * n_col && w_idx%n_col != n_col - 1){return;}

        float u1 = u[w_idx - n_col];
        float v1 = v[w_idx - 1];

        __syncthreads();

        div[w_idx] = (u[w_idx] - u1) / w - (v[w_idx] - v1) / w;
    }
    """)

    div = np.empty((n_row, n_col), dtype=DTYPE_f)
    n_row = DTYPE_i(n_row)
    n_col = DTYPE_i(n_col)
    w = DTYPE_f(w)

    # define GPU data
    # block_size = 16
    block_size = 32
    x_blocks = int(n_row * n_col // block_size + 1)

    # move data to gpu
    d_div = gpuarray.to_gpu(div)

    # get and launch kernel
    div_k = mod_div.get_function("div_k")
    div_boundary_k = mod_div.get_function("div_boundary_k")
    div_k(d_div, d_u, d_v, w, n_row, n_col, block=(block_size, 1, 1), grid=(x_blocks, 1))
    div_boundary_k(d_div, d_u, d_v, w, n_row, n_col, block=(block_size, 1, 1), grid=(x_blocks, 1))

    # get single case of bottom i = 0, j = n_col-1
    d_div[0, int(n_col - 1)] = (d_u[1, n_col - 1] - d_u[0, n_col - 1]) / w - (d_v[0, n_col - 1] - d_v[0, n_col - 2]) / w
    d_div[int(n_row - 1), 0] = (d_u[n_row - 1, 0] - d_u[n_row - 2, 0]) / w - (d_v[n_row - 1, 1] - d_v[n_row - 1, 0]) / w

    return d_div, d_u, d_v
