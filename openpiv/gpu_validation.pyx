"""This module is for GPU-accelerated validation algoritms."""

import numpy as np
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule


def gpu_validation(d_f, k, sig2noise, n_row, n_col, w, s2n_tol, median_tol, mean_tol, div_tol):
    """Returns an array indicating which indices need to be validated.

    Parameters
    ----------
    d_f : GPUArray - 4D float
        main loop array
    k : int
        iteration number
    sig2noise : array - 2D float
        signal to noise ratio of each velocity
    n_row, n_col : int
        number of rows and columns in the velocity field
    w : float
        number of pixels between each interrogation window center
    s2n_tol : float
        minimum value for sig2noise
    median_tol : float
        tolerance for median velocity validation
    mean_tol : float
        tolerance for mean velocity validation
    div_tol : float
        tolerance for divergence validation

    Returns
    -------
    val_list : GPUArray - 2D int
        array of indices that need to be validated. 0 indicates that the index needs to be corrected. 1 means no correction is needed
    d_u_mean : GPUArray - 2D
        mean of the velocities surrounding each point in this iteration.
    d_v_mean : GPUArray - 2D
        mean of the velocities surrounding each point in this iteration.

    """
    # GPU functions
    mod_validation = SourceModule("""
    __global__ void s2n(int *val_list, float *sig2noise, float s2n_tol, int Nrow, int Ncol)
    {
        // val_list : list of indices to be validated
        // sig2noise : signal to noise ratio
        // s2n_tol : min sig2noise value
        // Ncol : number of columns in the

        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (w_idx >= Ncol * Nrow) {return;}

        val_list[w_idx] = val_list[w_idx] * (sig2noise[w_idx] > s2n_tol);
    }


    __global__ void median_validation(int *val_list, float *u, float *v, float *u_median, float *v_median, float *u_median_fluc, float *v_median_fluc, int Nrow, int Ncol, float tol)
    {
        // val_list: list of locations where validation is needed
        // rms_u : rms u velocity of neighbours
        // rms_v : rms v velocity of neighbours
        // mean_u : mean u velocity of neighbours
        // mean_v : mean v velocity of neighbours
        // u : u velocity at that point
        // v : v velocity at that point
        // Nrow, Ncol : number of rows and columns
        // tol : validation tolerance. usually 1.5

        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (w_idx >= Nrow * Ncol) {return;}
        
        // a small number is added to prevent singularities in uniform flow (Scarano & Westerweel, 2005)
        int u_validation = fabsf(u[w_idx] - u_median[w_idx]) / (u_median_fluc[w_idx] + 0.1) < tol;
        int v_validation = fabsf(v[w_idx] - v_median[w_idx]) / (v_median_fluc[w_idx] + 0.1) < tol;

        val_list[w_idx] = val_list[w_idx] * u_validation * v_validation;
    }

    __global__ void mean_validation(int *val_list, float *u_rms, float *v_rms, float *u_mean, float *v_mean, float *u, float *v, int Nrow, int Ncol, float tol)
    {
        // val_list: list of locations where validation is needed
        // rms_u : rms u velocity of neighbours
        // rms_v : rms v velocity of neighbours
        // mean_u : mean u velocity of neighbours
        // mean_v : mean v velocity of neighbours
        // u : u velocity at that point
        // v : v velocity at that point
        // Nrow, Ncol : number of rows and columns
        // tol : validation tolerance. usually 1.5

        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (w_idx >= Nrow * Ncol) {return;}
        
        // a small number is added to prevent singularities in uniform flow (Scarano & Westerweel, 2005)
        int u_validation = fabsf(u[w_idx] - u_mean[w_idx]) / (u_rms[w_idx] + 0.1) < tol;
        int v_validation = fabsf(v[w_idx] - v_mean[w_idx]) / (v_rms[w_idx] + 0.1) < tol;

        val_list[w_idx] = val_list[w_idx] * u_validation * v_validation;
    }

    __global__ void div_validation(int *val_list, float *div, int Nrow, int Ncol, float div_tol)
    {
        // u : u velocity
        // v : v velocity
        // w : window size
        // Nrow, Ncol : number of rows and columns

        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (w_idx >= Nrow*Ncol) {return;}

        val_list[w_idx] = val_list[w_idx] * (fabsf(div[w_idx]) < div_tol);
    }
    """)

    # create array to store validation list
    val_list = np.ones_like(sig2noise, dtype=np.int32)
    d_val_list = gpuarray.to_gpu(val_list)

    # cast inputs to appropriate data types
    n_row = np.int32(n_row)
    n_col = np.int32(n_col)
    w = np.float32(w)

    # GPU settings
    block_size = 32
    x_blocks = int(n_col * n_row / block_size + 1)

    # send velocity field to GPU
    d_u = d_f[k, 0:n_row, 0:n_col, 2].copy()
    d_v = d_f[k, 0:n_row, 0:n_col, 3].copy()

    # get neighbours information
    d_neighbours, d_neighbours_present = gpu_get_neighbours(d_u, d_v, n_row, n_col)

    # compute the mean velocities to be returned
    d_u_mean, d_v_mean = gpu_mean_vel(d_neighbours, d_neighbours_present, n_row, n_col)

    ######################
    # sig2noise validation
    ######################
    # if s2n_tol is not None:
    #     sig2noise = sig2noise.astype(np.float32)
    #     s2n_tol = np.float32(s2n_tol)
    #     d_sig2noise = gpuarray.to_gpu(sig2noise)
    #
    #     # Launch signal to noise kernel and free sig2noise data
    #     s2n = mod_validation.get_function("s2n")  # disabled by eric
    #     s2n(d_val_list, d_sig2noise, s2n_tol, n_row, n_col, block=(block_size, 1, 1), grid=(x_blocks, 1))
    #
    #     # Free gpu memory
    #     d_sig2noise.gpudata.free()
    #     # del d_sig2noise

    ############################
    # median_velocity validation
    ############################

    if median_tol is not None:
        median_tol = np.float32(median_tol)

        # get rms data and mean velocity data.
        d_u_median, d_v_median = gpu_median_vel(d_neighbours, d_neighbours_present, n_row, n_col)
        d_u_median_fluc, d_v_median_fluc = gpu_median_fluc(d_neighbours, d_neighbours_present, d_u_median, d_v_median, n_row, n_col)

        median_validation = mod_validation.get_function("median_validation")
        median_validation(d_val_list, d_u, d_v, d_u_median, d_v_median, d_u_median_fluc, d_v_median_fluc, n_row, n_col, median_tol, block=(block_size, 1, 1), grid=(x_blocks, 1))

        # Free gpu memory
        d_u_median.gpudata.free()
        d_v_median.gpudata.free()
        d_u_median_fluc.gpudata.free()
        d_v_median_fluc.gpudata.free()
        # del d_u_median, d_v_median, d_u_median_fluc, d_v_median_fluc

    ##########################
    # mean_velocity validation
    ##########################

    # if mean_tol is not None:
    #     mean_tol = np.float32(mean_tol)
    #
    #     # get rms data and mean velocity data.
    #     d_u_rms, d_v_rms = gpu_rms(d_neighbours, d_neighbours_present, n_row, n_col)
    #     d_u_rms, d_v_rms = gpu_mean_fluc(d_neighbours, d_neighbours_present, d_u_mean, d_v_mean, n_row, n_col)
    #
    #     mean_validation = mod_validation.get_function("mean_validation")
    #     mean_validation(d_val_list, d_u_rms, d_v_rms, d_u_mean, d_v_mean, d_u, d_v, n_row, n_col, mean_tol, block=(block_size, 1, 1), grid=(x_blocks, 1))
    #
    #     # Free gpu memory
    #     d_u_rms.gpudata.free()
    #     d_v_rms.gpudata.free()
    #     # del d_u_rms, d_v_rms

    #######################
    # divergence validation
    #######################

    # if div_tol is not None:
    #     div_tol = np.float32(div_tol)
    #     assert True, 'divergence validation code reached!'
    #     d_div, d_u, d_v = gpu_divergence(d_u, d_v, w, n_row, n_col)
    #
    #     # launch divergence validation kernel
    #     div_validation = mod_validation.get_function("div_validation")
    #     div_validation(d_val_list, d_div, n_row, n_col, div_tol, block=(block_size, 1, 1), grid=(x_blocks, 1))
    #
    #     d_u.gpudata.free()
    #     d_v.gpudata.free()
    #     d_div.gpudata.free()
    #     # del d_u, d_v, d_div

    # return the final validation list
    val_list = d_val_list.get()

    # Free gpu memory
    d_val_list.gpudata.free()

    # Free gpu memory
    d_neighbours_present.gpudata.free()
    d_neighbours.gpudata.free()

    # del d_val_list, d_neighbours, d_neighbours_present

    return val_list, d_u_mean, d_v_mean


def gpu_find_neighbours(n_row, n_col):
    """An array that stores if a point has neighbours in a 3x3 grid surrounding it

    Parameters
    ----------
    n_row : array - 1D int
        number of rows at each iteration
    n_col : array - 1D int
        number of columns at each iteration

    Returns
    -------
    d_neighbours_present : GPUArray - 4D [n_row, n_col, 3 , 3]

    """
    mod_neighbours = SourceModule("""
    __global__ void find_neighbours(int *neighbours_present, int Nrow, int Ncol)
    {
        // neighbours_present = boolean array
        // Nrow = number of rows
        // Ncol = Number of columns

        // references each IW
        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;

        int row_zero = (w_idx >= Ncol);
        int row_max = (w_idx < Ncol * (Nrow - 1));
        int col_zero = (w_idx % Ncol != 0);
        int col_max = (w_idx % Ncol != Ncol - 1);

        // Top Row
        neighbours_present[w_idx * 9 + 0] = neighbours_present[w_idx * 9 + 0] * row_zero;
        neighbours_present[w_idx * 9 + 1] = neighbours_present[w_idx * 9 + 1] * row_zero;
        neighbours_present[w_idx * 9 + 2] = neighbours_present[w_idx * 9 + 2] * row_zero;

        __syncthreads();

        // Bottom row
        neighbours_present[w_idx * 9 + 6] = neighbours_present[w_idx * 9 + 6] * row_max;
        neighbours_present[w_idx * 9 + 7] = neighbours_present[w_idx * 9 + 7] * row_max;
        neighbours_present[w_idx * 9 + 8] = neighbours_present[w_idx * 9 + 8] * row_max;

        __syncthreads();

        // Left column
        neighbours_present[w_idx * 9 + 0] = neighbours_present[w_idx * 9 + 0] * col_zero;
        neighbours_present[w_idx * 9 + 3] = neighbours_present[w_idx * 9 + 3] * col_zero;
        neighbours_present[w_idx * 9 + 6] = neighbours_present[w_idx * 9 + 6] * col_zero;

        __syncthreads();

        // right column
        neighbours_present[w_idx * 9 + 2] = neighbours_present[w_idx * 9 + 2] * col_max;
        neighbours_present[w_idx * 9 + 5] = neighbours_present[w_idx * 9 + 5] * col_max;
        neighbours_present[w_idx * 9 + 8] = neighbours_present[w_idx * 9 + 8] * col_max;
        
        __syncthreads();
        
        // Set center to zero--can't be a neighbour for yourself
        neighbours_present[w_idx * 9 + 4] = 0;
    }
    """)

    # GPU settings
    block_size = 32
    x_blocks = int(n_col * n_row // block_size + 1)

    # allocate space for new array
    neighbours_present = np.ones([n_row, n_col, 3, 3], dtype=np.int32)

    assert neighbours_present.dtype == np.int32, "Wrong data type for neighbours present"

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
    d_u, d_v : GPUArray - 2D float32
        u and v velocity
    n_row, n_col : array - 1D int
        number of rows/columns at each iteration

    Returns
    -------
    neighbours : array - 5D [n_row, n_col, 2, 3, 3]
        stores the values of u and v of the neighbours of a point

    """
    # TODO make this multiplicative instead of if statements
    mod_get_neighbours = SourceModule("""
    __global__ void get_u_neighbours(float *neighbours, int *neighbours_present, float *u, int Nrow, int Ncol)
    {
        // neighbours - u and v values around each point
        // neighbours_present - 1 if there is a neighbour, 0 if no neighbour
        // u, v - u and v velocities
        // Nrow, Ncol - number of rows and columns

        // references each IW
        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (w_idx >= Nrow * Ncol) {return;}

        // get velocities
        if (neighbours_present[w_idx * 9 + 0]) {neighbours[w_idx * 18 + 0] = u[w_idx - Ncol - 1];}
        if (neighbours_present[w_idx * 9 + 1]) {neighbours[w_idx * 18 + 1] = u[w_idx - Ncol];}
        if (neighbours_present[w_idx * 9 + 2]) {neighbours[w_idx * 18 + 2] = u[w_idx - Ncol + 1];}

        __syncthreads();

        if (neighbours_present[w_idx * 9 + 3]) {neighbours[w_idx * 18 + 3] = u[w_idx - 1];}
        // neighbours[w_idx * 18 + 4] = 0.0;
        if (neighbours_present[w_idx * 9 + 5]) {neighbours[w_idx * 18 + 5] = u[w_idx + 1];}

        __syncthreads();

        if (neighbours_present[w_idx * 9 + 6]) {neighbours[w_idx * 18 + 6] = u[w_idx + Ncol - 1];}
        if (neighbours_present[w_idx * 9 + 7]) {neighbours[w_idx * 18 + 7] = u[w_idx + Ncol];}
        if (neighbours_present[w_idx * 9 + 8]) {neighbours[w_idx * 18 + 8] = u[w_idx + Ncol + 1];}

        __syncthreads();
    }

    __global__ void get_v_neighbours(float *neighbours, int *neighbours_present, float *v, int Nrow, int Ncol)
    {
        // neighbours - u and v values around each point
        // neighbours_present - 1 if there is a neighbour, 0 if no neighbour
        // u, v - u and v velocities
        // Nrow, Ncol - number of rows and columns

        // references each IW
        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (w_idx >= Nrow * Ncol) {return;}

        // get velocities
        if (neighbours_present[w_idx * 9 + 0]) {neighbours[w_idx * 18 + 9] = v[w_idx - Ncol - 1];}
        if (neighbours_present[w_idx * 9 + 1]) {neighbours[w_idx * 18 + 10] = v[w_idx - Ncol];}
        if (neighbours_present[w_idx * 9 + 2]) {neighbours[w_idx * 18 + 11] = v[w_idx - Ncol + 1];}

        __syncthreads();

        if (neighbours_present[w_idx * 9 + 3]) {neighbours[w_idx * 18 + 12] = v[w_idx - 1];}
        // neighbours[w_idx * 18 + 13] = 0.0;
        if (neighbours_present[w_idx * 9 + 5]) {neighbours[w_idx * 18 + 14] = v[w_idx + 1];}

        __syncthreads();

        if (neighbours_present[w_idx * 9 + 6]) {neighbours[w_idx * 18 + 15] = v[w_idx + Ncol - 1];}
        if (neighbours_present[w_idx * 9 + 7]) {neighbours[w_idx * 18 + 16] = v[w_idx + Ncol];}
        if (neighbours_present[w_idx * 9 + 8]) {neighbours[w_idx * 18 + 17] = v[w_idx + Ncol + 1];}

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
    neighbours = np.zeros((n_row, n_col, 2, 3, 3), dtype=np.float32)

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
    d_neighbours: GPUArray - 5D float32
        all the neighbouring velocities of every point
    d_neighbours_present: GPUArray - 4D float32
        indicates if a neighbour is present
    n_row, n_col : int
        number of rows and columns of the velocity field

    Returns
    -------
    u_mean, v_mean : array - 2D float32
        mean velocities at each point

    """
    mod_mean_vel = SourceModule("""
    __global__ void u_mean_vel(float *u_mean, float *n, int *np, int Nrow, int Ncol)
    {
        // mean_u : mean velocity of surrounding points
        // n : velocity of neighbours
        // np : neighbours present
        // Nrow, Ncol : number of rows and columns
        
        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (w_idx >= Ncol * Nrow) {return;}
        
        // ensure denominator is not zero then compute mean
        float numerator_u = n[w_idx * 18 + 0] + n[w_idx * 18 + 1] + n[w_idx * 18 + 2] + n[w_idx * 18 + 3] + \
                            n[w_idx * 18 + 5] + n[w_idx * 18 + 6] + n[w_idx * 18 + 7] + n[w_idx * 18 + 8];
        
        // mean is normalized by number of terms summed
        float denominator = np[w_idx * 9 + 0] + np[w_idx * 9 + 1] + np[w_idx * 9 + 2] + np[w_idx * 9 + 3] + \
                            np[w_idx * 9 + 5] + np[w_idx * 9 + 6] + np[w_idx * 9 + 7] + np[w_idx * 9 + 8];
        
        __syncthreads();
        
        u_mean[w_idx] = numerator_u / denominator;
        
        __syncthreads();
    }

    __global__ void v_mean_vel(float *v_mean, float *n, int *np, int Nrow, int Ncol)
    {
        // mean_v : mean velocity of surrounding points
        // n : velocity of neighbours
        // np : neighbours present
        // Nrow, Ncol : number of rows and columns
        
        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (w_idx >= Ncol * Nrow) {return;}
        
        // ensure denominator is not zero then compute mean
        float numerator_v = n[w_idx * 18 + 9] + n[w_idx * 18 + 10] + n[w_idx * 18 + 11] + n[w_idx * 18 + 12] + \
                            n[w_idx * 18 + 14] + n[w_idx * 18 + 15] + n[w_idx * 18 + 16] + n[w_idx * 18 + 17];
        
        // mean is normalized by number of terms summed
        float denominator = np[w_idx * 9 + 0] + np[w_idx * 9 + 1] + np[w_idx * 9 + 2] + np[w_idx * 9 + 3] + \
                            np[w_idx * 9 + 5] + np[w_idx * 9 + 6] + np[w_idx * 9 + 7] + np[w_idx * 9 + 8];
        
        __syncthreads();
        
        v_mean[w_idx] = numerator_v / denominator;
        
        __syncthreads();
    }
    """)

    # allocate space for arrays
    u_mean = np.zeros((n_row, n_col), dtype=np.float32)
    v_mean = np.zeros((n_row, n_col), dtype=np.float32)

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
    d_neighbours : GPUArray - 5D float32
        all the neighbouring velocities of every point
    d_neighbours_present : GPUArray - 4D float32
        indicates if a neighbour is present
    d_u_mean, d_v_mean : GPUArray - 2D
        mean velocities around each point
    n_row, n_col : int
        number of rows and columns of the velocity field

    Returns
    -------
    d_u_fluc, d_v_fluc : GPUArray - 2D float32
        rms velocities at each point

    """
    mod_mean_fluc = SourceModule("""
    __global__ void u_fluc_k(float *u_fluc, float *u_mean, float *n, int *np, int Nrow, int Ncol)
    {
        // u_fluc : velocity fluctuations of surrounding points
        // u_mean : mean velocity of surrounding points
        // n : velocity of neighbours
        // np : neighbours present
        // Nrow, Ncol : number of rows and columns

        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (w_idx >= Ncol * Nrow) {return;}

        // ensure denominator is not zero then compute fluctuations
        float numerator = fabsf(np[w_idx * 18 + 0] - u_mean[w_idx]) + fabsf(np[w_idx * 18 + 1] - u_mean[w_idx]) + \
                          fabsf(np[w_idx * 18 + 2] - u_mean[w_idx]) + fabsf(np[w_idx * 18 + 3] - u_mean[w_idx]) + \
                          fabsf(np[w_idx * 18 + 5] - u_mean[w_idx]) + fabsf(np[w_idx * 18 + 6] - u_mean[w_idx]) + \
                          fabsf(np[w_idx * 18 + 7] - u_mean[w_idx]) + fabsf(np[w_idx * 18 + 8] - u_mean[w_idx]);
        
        // rms is normalized by number of terms summed
        float denominator = np[w_idx * 9 + 0] + np[w_idx * 9 + 1] + np[w_idx * 9 + 2] + np[w_idx * 9 + 3] + \
                            np[w_idx * 9 + 5] + np[w_idx * 9 + 6] + np[w_idx * 9 + 7] + np[w_idx * 9 + 8];

        __syncthreads();

        u_fluc[w_idx] = numerator / denominator;

        __syncthreads();
    }

    __global__ void v_fluc_k(float *v_fluc, float *v_mean, float *n, int *np, int Nrow, int Ncol)
    {
        // v_fluc : velocity fluctuations of surrounding points
        // v_mean : mean velocity of surrounding points
        // n : velocity of neighbours
        // np : neighbours present
        // Nrow, Ncol : number of rows and columns

        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (w_idx >= Ncol * Nrow) {return;}

        // rms is normalized by number of terms summed
        float denominator = np[w_idx * 9 + 0] + np[w_idx * 9 + 1] + np[w_idx * 9 + 2] + np[w_idx * 9 + 3] + \
                            np[w_idx * 9 + 5] + np[w_idx * 9 + 6] + np[w_idx * 9 + 7] + np[w_idx * 9 + 8];

        // ensure denominator is not zero then compute fluctuations
        float numerator = fabsf(np[w_idx * 18 + 9] - v_mean[w_idx]) + fabsf(np[w_idx * 18 + 10] - v_mean[w_idx]) + \
                          fabsf(np[w_idx * 18 + 11] - v_mean[w_idx]) + fabsf(np[w_idx * 18 + 12] - v_mean[w_idx]) + \
                          fabsf(np[w_idx * 18 + 14] - v_mean[w_idx]) + fabsf(np[w_idx * 18 + 15] - v_mean[w_idx]) + \
                          fabsf(np[w_idx * 18 + 16] - v_mean[w_idx]) + fabsf(np[w_idx * 18 + 17] - v_mean[w_idx]);

        __syncthreads();

        v_fluc[w_idx] = numerator / denominator;

        __syncthreads();
    }
    """)

    # allocate space for data
    u_rms = np.zeros((n_row, n_col), dtype=np.float32)
    v_rms = np.zeros((n_row, n_col), dtype=np.float32)

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
    d_neighbours: GPUArray - 5D float
        all the neighbouring velocities of every point
    d_neighbours_present: GPUArray - 4D float
        indicates if a neighbour is present
    n_row, n_col : int
        number of rows and columns of the velocity field

    Returns
    -------
    u_median, v_median : array - 2D float
        mean velocities at each point

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

    __global__ void u_median_vel(float *u_median, float *n, int *np, int Nrow, int Ncol)
    {
        // u_median : median velocity of surrounding points
        // n : velocity of neighbours
        // np : neighbours present
        // Nrow, Ncol : number of rows and columns

        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (w_idx >= Ncol * Nrow) {return;}

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

    __global__ void v_median_vel(float *v_median, float *n, int *np, int Nrow, int Ncol)
    {
        // v_median : median velocity of surrounding points
        // n : velocity of neighbours
        // np : neighbours present
        // Nrow, Ncol : number of rows and columns

        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (w_idx >= Ncol * Nrow) {return;}
        
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
    u_median = np.zeros((n_row, n_col), dtype=np.float32)
    v_median = np.zeros((n_row, n_col), dtype=np.float32)

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
    d_neighbours : GPUArray - 5D float32
        all the neighbouring velocities of every point
    d_neighbours_present : GPUArray - 4D float32
        indicates if a neighbour is present
    d_u_median, d_v_median : GPUArray - 2D
        mean velocities around each point
    n_row, n_col : int
        number of rows and columns of the velocity field

    Returns
    -------
    d_u_median_fluc, d_v_median_fluc : GPUArray - 2D float32
        rms velocities at each point

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

    __global__ void u_fluc_k(float *u_median_fluc, float *u_median, float *n, int *np, int Nrow, int Ncol)
    {
        // u_fluc : velocity fluctuations of surrounding points
        // u_median : median velocity of surrounding points
        // n : velocity of neighbours
        // np : neighbours present
        // Nrow, Ncol : number of rows and columns

        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (w_idx >= Ncol * Nrow) {return;}

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

    __global__ void v_fluc_k(float *v_median_fluc, float *v_median, float *n, int *np, int Nrow, int Ncol)
    {
        // v_fluc : velocity fluctuations of surrounding points
        // v_median : mean velocity of surrounding points
        // n : velocity of neighbours
        // np : neighbours present
        // Nrow, Ncol : number of rows and columns

        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (w_idx >= Ncol * Nrow) {return;}

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
    u_median_fluc = np.zeros((n_row, n_col), dtype=np.float32)
    v_median_fluc = np.zeros((n_row, n_col), dtype=np.float32)

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


# def gpu_rms(d_neighbours, d_neighbours_present, n_row, n_col):
#     """Calculates the rms velocity in a 3x3 grid around each point in a velocity field.
#
#     Parameters
#     ----------
#     d_neighbours : GPUArray - 5D float32
#         all the neighbouring velocities of every point
#     d_neighbours_present : GPUArray - 4D float32
#         indicates if a neighbour is present
#     n_row, n_col : int
#         number of rows and columns of the velocity field
#
#     Returns
#     -------
#     d_u_rms, d_v_rms : GPUArray - 2D float32
#         rms velocities at each point
#
#     """
#     mod_rms = SourceModule("""
#     __global__ void u_rms_k(float *u_rms, float *n, int *np, int Nrow, int Ncol)
#     {
#         // u_rms : rms of surrounding points
#         // n : velocity of neighbours
#         // np : neighbours present
#         // Nrow, Ncol: number of rows and columns
#
#         int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
#
#         if(w_idx >= Ncol * Nrow){return;}
#
#         // rms is normalized by number of terms summed
#         float denominator = np[w_idx * 9 + 0] + np[w_idx * 9 + 1] + np[w_idx * 9 + 2] + np[w_idx * 9 + 3] + \
#                             np[w_idx * 9 + 5] + np[w_idx * 9 + 6] + np[w_idx * 9 + 7] + np[w_idx * 9 + 8];
#
#         __syncthreads();
#
#         // ensure denominator is not zero then compute rms
#         if(denominator > 0){
#             float numerator = (powf(n[w_idx * 18 + 0], 2) + powf(n[w_idx * 18 + 1], 2) + powf(n[w_idx * 18 + 2], 2) + \
#                                powf(n[w_idx * 18 + 3], 2) + powf(n[w_idx * 18 + 5], 2) + powf(n[w_idx * 18 + 6], 2) + \
#                                powf(n[w_idx * 18 + 7], 2) + powf(n[w_idx * 18 + 8], 2));
#
#             u_rms[w_idx] = sqrtf(numerator / denominator);
#         }
#
#         __syncthreads();
#     }
#
#     __global__ void v_rms_k(float *v_rms, float *n, int *np, int Nrow, int Ncol)
#     {
#         // v_rms : rms of surrounding points
#         // n : velocity of neighbours
#         // np : neighbours present
#         // Nrow, Ncol: number of rows and columns
#
#         int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
#
#         if(w_idx >= Ncol * Nrow){return;}
#
#         // rms is normalized by number of terms summed
#         float denominator = np[w_idx * 9 + 0] + np[w_idx * 9 + 1] + np[w_idx * 9 + 2] + np[w_idx * 9 + 3] + \
#                             np[w_idx * 9 + 5] + np[w_idx * 9 + 6] + np[w_idx * 9 + 7] + np[w_idx * 9 + 8];
#
#         __syncthreads();
#
#         // ensure denominator is not zero then compute rms
#         if (denominator > 0){
#             float numerator = (powf(n[w_idx * 18 + 9], 2) + powf(n[w_idx * 18 + 10], 2) + powf(n[w_idx * 18 + 11], 2) + \
#                                powf(n[w_idx * 18 + 12], 2) + powf(n[w_idx * 18 + 14], 2) + powf(n[w_idx * 18 + 15], 2) + \
#                                powf(n[w_idx * 18 + 16], 2) + powf(n[w_idx * 18 + 17], 2));
#
#             v_rms[w_idx] = sqrtf(numerator / denominator);
#         }
#
#         __syncthreads();
#     }
#     """)
#
#     # allocate space for data
#     u_rms = np.zeros((n_row, n_col), dtype=np.float32)
#     v_rms = np.zeros((n_row, n_col), dtype=np.float32)
#     n_row = np.int32(n_row)
#     n_col = np.int32(n_col)
#
#     # define GPU data
#     # block_size = 16
#     block_size = 32
#     x_blocks = int(n_row * n_col // block_size + 1)
#
#     # send data to gpu
#     d_u_rms = gpuarray.to_gpu(u_rms)
#     d_v_rms = gpuarray.to_gpu(v_rms)
#
#     # get and launch kernel
#     mod_u_rms = mod_rms.get_function("u_rms_k")
#     mod_v_rms = mod_rms.get_function("v_rms_k")
#     mod_u_rms(d_u_rms, d_neighbours, d_neighbours_present, n_row, n_col, block=(block_size, 1, 1), grid=(x_blocks, 1))
#     mod_v_rms(d_v_rms, d_neighbours, d_neighbours_present, n_row, n_col, block=(block_size, 1, 1), grid=(x_blocks, 1))
#
#     return d_u_rms, d_v_rms


# def gpu_divergence(d_u, d_v, w, n_row, n_col):
#     """Calculates the divergence at each point in a velocity field.
#
#     Parameters
#     ----------
#     d_u, d_v: array - 2D float
#         velocity field
#     w: int
#         pixel separation between velocity vectors
#     n_row, n_col : int
#         number of rows and columns of the velocity field
#
#     Returns
#     -------
#     div : array - 2D float32
#         divergence at each point
#
#     """
#     mod_div = SourceModule("""
#     __global__ void div_k(float *div, float *u, float *v, float w, int Nrow, int Ncol)
#     {
#         // u : u velocity
#         // v : v velocity
#         // w : window size
#         // Nrow, Ncol : number of rows and columns
#
#         int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
#         int max_idx = Nrow * Ncol;
#
#         // Avoid the boundary
#         if(w_idx >= (Nrow - 1) * Ncol){return;}
#         if(w_idx%Ncol == Ncol - 1){return;}
#
#         float u1 = u[w_idx + Ncol];
#         float v1 = v[w_idx + 1];
#
#         __syncthreads();
#
#         div[w_idx] = (u1 - u[w_idx]) / w - (v1 - v[w_idx]) / w;
#     }
#
#     __global__ void div_boundary_k(float *div, float *u, float *v, float w, int Nrow, int Ncol)
#     {
#         // u : u velocity
#         // v : v velocity
#         // w : window size
#         // Nrow, Ncol : number of rows and columns
#
#         int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
#
#         // only calculate on the boundary
#         if(w_idx < (Nrow - 1) * Ncol && w_idx%Ncol != Ncol - 1){return;}
#
#         float u1 = u[w_idx - Ncol];
#         float v1 = v[w_idx - 1];
#
#         __syncthreads();
#
#         div[w_idx] = (u[w_idx] - u1) / w - (v[w_idx] - v1) / w;
#     }
#     """)
#
#     div = np.empty((n_row, n_col), dtype=np.float32)
#     n_row = np.int32(n_row)
#     n_col = np.int32(n_col)
#     w = np.float32(w)
#
#     # define GPU data
#     # block_size = 16
#     block_size = 32
#     x_blocks = int(n_row * n_col // block_size + 1)
#
#     # move data to gpu
#     d_div = gpuarray.to_gpu(div)
#
#     # get and launch kernel
#     div_k = mod_div.get_function("div_k")
#     div_boundary_k = mod_div.get_function("div_boundary_k")
#     div_k(d_div, d_u, d_v, w, n_row, n_col, block=(block_size, 1, 1), grid=(x_blocks, 1))
#     div_boundary_k(d_div, d_u, d_v, w, n_row, n_col, block=(block_size, 1, 1), grid=(x_blocks, 1))
#
#     # get single case of bottom i = 0, j = Ncol-1
#     d_div[0, int(n_col - 1)] = (d_u[1, n_col - 1] - d_u[0, n_col - 1]) / w - (d_v[0, n_col - 1] - d_v[0, n_col - 2]) / w
#     d_div[int(n_row - 1), 0] = (d_u[n_row - 1, 0] - d_u[n_row - 2, 0]) / w - (d_v[n_row - 1, 1] - d_v[n_row - 1, 0]) / w
#
#     return d_div, d_u, d_v
