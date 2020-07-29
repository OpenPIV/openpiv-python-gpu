"""This module is for GPU-accelerated validation algoritms"""

import numpy as np
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule


def gpu_validation(d_f, k, sig2noise, n_row, n_col, w, s2n_tol, mean_tol, div_tol):
    """Returns an array indicating which indices need to be validated.

    Parameters
    ----------
    d_f : 4D gpuarray - float
        main loop array
    k : int
        iteration number
    sig2noise: 2D array - float
        signal to noise ratio of each velocity
    n_row, n_col : int
        number of rows and columns in the velocity field
    w : float
        number of pixels between each interrogation window center
    s2n_tol : float
        minimum value for sig2noise
    mean_tol : float
        tolerance for mean velocity validation
    div_tol : float
        tolerance for divergence validation

    Returns
    -------
    val_list : 2D array - int
        list of indices that need to be validated. 0 indicates that the index needs to be corrected. 1 means no correction is needed
    d_u_mean, d_v_mean : 2D gpuarray
        mean of the velocities surrounding each point in this iteration.

    """
    # GPU functions
    mod_validation = SourceModule("""
    __global__ void s2n(int *val_list, float *sig2noise, float s2n_tol, int Nrow, int Ncol)
    {
        //val_list : list of indices to be validated
        //sig2noise : signal to noise ratio
        // s2n_tol : min sig2noise value
        // Ncol : number of columns in the

        int w_idx = blockIdx.x*blockDim.x + threadIdx.x;

        if(w_idx >= Ncol*Nrow){return;}

        val_list[w_idx] = val_list[w_idx] * (sig2noise[w_idx] > s2n_tol);
    }


    __global__ void mean_validation(int *val_list, float *u_rms, float *v_rms, float *u_mean, float *v_mean, float *u, float *v, int Nrow, int Ncol, float tol)
    {
        // val_list: list of locations where validation is needed
        // rms_u : rms u velocity of neighbours
        // rms_v : rms v velocity of neighbours
        // mean_u: mean u velocity of neigbours
        // mean_v: mean v velocity of neighbours
        // u: u velocity at that point
        // v: v velocity at that point
        // Nrow, Ncol: number of rows and columns
        // tol : validation tolerance. usually 1.5

        int w_idx = blockIdx.x*blockDim.x + threadIdx.x;

        if(w_idx >= Nrow*Ncol){return;}

        int u_validation = ((u[w_idx] - u_mean[w_idx])/u_rms[w_idx] < tol);
        int v_validation = ((v[w_idx] - v_mean[w_idx])/v_rms[w_idx] < tol);

        val_list[w_idx] = val_list[w_idx] * u_validation * v_validation;

    }

    __global__ void div_validation(int *val_list, float *div,  int Nrow, int Ncol, float div_tol)
    {
        // u: u velocity
        // v: v velocity
        // w: window size
        // Nrow, Ncol: number of rows and columns

        int w_idx = blockIdx.x*blockDim.x + threadIdx.x;

        if(w_idx >= Nrow*Ncol){return;}

        val_list[w_idx] = val_list[w_idx] * (fabsf(div[w_idx]) < div_tol);
    }

    """)

    # create array to store validation list
    val_list = np.ones_like(sig2noise, dtype=np.int32)
    d_val_list = gpuarray.to_gpu(val_list)

    # cast inputs to appropriate data types
    sig2noise = sig2noise.astype(np.float32)
    s2n_tol = np.float32(s2n_tol)
    mean_tol = np.float32(mean_tol)
    div_tol = np.float32(div_tol)
    n_row = np.int32(n_row)
    n_col = np.int32(n_col)
    w = np.float32(w)

    # TODO delete these checks
    # assert sig2noise.dtype == np.float32, "dtype of sig2noise is {}. Should be np.float32".format(sig2noise.dtype)
    # assert type(s2n_tol) == np.float32, "type of s2n_tol is {}. Should be np.float32".format(type(s2n_tol))
    # assert type(n_row) == np.int32, "dtype of Nrow is {}. Should be np.int32".format(type(n_row))
    # assert type(n_col) == np.int32, "dtype of Ncol is {}. Should be np.int32".format(type(n_col))
    # assert type(w) == np.float32, "dtype of w is {}. Should be np.float32" .format(type(w))
    # assert d_f.dtype == np.float32, "dtype of d_F is {}. dtype should be np.float32".format(d_f.dtype)

    # GPU settings
    # block_size = 16
    block_size = 32
    x_blocks = int(n_col * n_row / block_size + 1)

    # send velocity field to GPU
    d_u = d_f[k, 0:n_row, 0:n_col, 10].copy()
    d_v = d_f[k, 0:n_row, 0:n_col, 11].copy()

    # get neighbours information
    d_neighbours, d_neighbours_present, d_u, d_v = gpu_get_neighbours(d_u, d_v, n_row, n_col)

    ##########################
    # sig2noise validation
    ##########################

    # # move data to the gpu
    d_sig2noise = 0
    # d_sig2noise = gpuarray.to_gpu(sig2noise)
    #
    # # Launch signal to noise kernel and free sig2noise data
    # s2n = mod_validation.get_function("s2n")
    # s2n(d_val_list, d_sig2noise, s2n_tol, n_row, n_col, block=(block_size, 1, 1), grid=(x_blocks, 1))
    # d_sig2noise.gpudata.free()

    ##########################
    # mean_velocity validation
    ##########################

    # get rms data and mean velocity data.
    d_u_rms, d_v_rms = gpu_rms(d_neighbours, d_neighbours_present, n_row, n_col)
    d_u_mean, d_v_mean = gpu_mean_vel(d_neighbours, d_neighbours_present, n_row, n_col)

    # get and launch rms
    mean_validation = mod_validation.get_function("mean_validation")
    a = d_val_list.get()
    print(a.size)
    d_val_list = gpuarray.to_gpu(a)
    mean_validation(d_val_list, d_u_rms, d_v_rms, d_u_mean, d_v_mean, d_u, d_v, n_row, n_col, mean_tol,
                    block=(block_size, 1, 1), grid=(x_blocks, 1))

    ##########################
    # divergence validation
    ##########################

    d_div = 0
    # d_div, d_u, d_v = gpu_divergence(d_u, d_v, w, n_row, n_col)
    #
    # # launch divergence validation kernel
    # div_validation = mod_validation.get_function("div_validation")
    # div_validation(d_val_list, d_div, n_row, n_col, div_tol, block=(block_size, 1, 1), grid=(x_blocks, 1))

    # return the final validation list
    val_list = d_val_list.get()
    # u_mean = d_u_mean.get()
    # v_mean = d_v_mean.get()

    # Free gpu memory
    d_val_list.gpudata.free()
    d_neighbours_present.gpudata.free()
    d_neighbours.gpudata.free()
    d_u.gpudata.free()
    d_v.gpudata.free()
    # d_u_mean.gpudata.free()
    # d_v_mean.gpudata.free()
    d_u_rms.gpudata.free()
    d_v_rms.gpudata.free()
    # d_div.gpudata.free()

    del d_val_list, d_sig2noise, d_neighbours, d_neighbours_present, d_u, d_v, d_u_rms, d_v_rms, d_div

    return val_list, d_u_mean, d_v_mean


def gpu_find_neighbours(n_row, n_col):
    """An array that stores if a point has neighbours in a 3x3 grid surrounding it

    Parameters
    ----------
    n_row : 1D array - int
        number of rows at each iteration
    n_col : 1D array - int
        number of columns at each iteration

    Returns
    -------
    d_neighbours_present : 4D gpuarray [n_row, n_col, 3 , 3]

    """
    mod_neighbours = SourceModule("""
    __global__ void find_neighbours(int *neighbours_present, int Nrow, int Ncol)
    {
        // neighbours_present = boolean array
        // Nrow = number of rows
        // Ncol = Number of columns

        // references each IW
        int w_idx = blockIdx.x*blockDim.x + threadIdx.x;

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

        // Set center to zero--can't be a neighbour for yourself
        neighbours_present[w_idx*9 + 4] = 0;
    }
    """)

    # GPU settings
    # block_size = 8
    block_size = 32
    x_blocks = int(n_col * n_row // block_size + 1)
    n_row = np.int32(n_row)
    n_col = np.int32(n_col)

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
    d_u, d_v : 2D GPU array - float32
        u and v velocity
    n_row : 1D array - int
        number of rows at each iteration
    n_col : 1D array - int
        number of columns at each iteration

    Returns
    -------
    neighbours : 5D array [n_row, n_col, 2, 3, 3]
        stores the values of u and v of the neighbours of a point

    """
    # TODO delete this redundant code
    # mod_get_neighbours = SourceModule("""
    # __global__ void get_u_neighbours(float *neighbours, float *neighbours_present, float *u, int Nrow, int Ncol)
    # {
    #     // neighbours - u and v values around each point
    #     // neighbours_present - 1 if there is a neighbour, 0 if no neighbour
    #     // u, v - u and v velocities
    #     // Nrow, Ncol - number of rows and columns
    #
    #     // references each IW
    #     int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
    #     int max_idx = Nrow * Ncol;
    #
    #     if(w_idx >= max_idx){return;}
    #
    #     // get velocities
    #     neighbours[w_idx * 18 + 0] = u[max(w_idx - Ncol - 1, 0)] * neighbours_present[w_idx * 9 + 0];
    #     neighbours[w_idx * 18 + 1] = u[max(w_idx - Ncol, 0)] * neighbours_present[w_idx * 9 + 1];
    #     neighbours[w_idx * 18 + 2] = u[max(w_idx - Ncol + 1, 0)] * neighbours_present[w_idx * 9 + 2];
    #
    #     __syncthreads();
    #
    #     neighbours[w_idx * 18 + 3] = u[max(w_idx - 1, 0)] * neighbours_present[w_idx * 9 + 3];
    #     neighbours[w_idx * 18 + 4] = 0.0;
    #     neighbours[w_idx * 18 + 5] = u[min(w_idx + 1, max_idx)] * neighbours_present[w_idx * 9 + 5];
    #
    #     __syncthreads();
    #
    #     neighbours[w_idx * 18 + 6] = u[min(w_idx + Ncol - 1, max_idx)] * neighbours_present[w_idx * 9 + 6];
    #     neighbours[w_idx * 18 + 7] = u[min(w_idx + Ncol, max_idx)] * neighbours_present[w_idx * 9 + 7];
    #     neighbours[w_idx * 18 + 8] = u[min(w_idx + Ncol + 1, max_idx)] * neighbours_present[w_idx * 9 + 8];
    #
    #     __syncthreads();
    # }
    #
    # __global__ void get_v_neighbours(float *neighbours, float *neighbours_present, float *v, int Nrow, int Ncol)
    # {
    #     // neighbours - u and v values around each point
    #     // neighbours_present - 1 if there is a neighbour, 0 if no neighbour
    #     // u, v - u and v velocities
    #     // Nrow, Ncol - number of rows and columns
    #
    #     // references each IW
    #     int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
    #     int max_idx = Nrow * Ncol;
    #
    #     if(w_idx >= max_idx){return;}
    #
    #     // get velocities
    #     neighbours[w_idx * 18 + 9] = v[max(w_idx - Ncol - 1, 0)] * neighbours_present[w_idx * 9 + 0];
    #     neighbours[w_idx * 18 + 10] = v[max(w_idx - Ncol, 0)] * neighbours_present[w_idx * 9 + 1];
    #     neighbours[w_idx * 18 + 11] = v[max(w_idx - Ncol + 1, 0)] * neighbours_present[w_idx * 9 + 2];
    #
    #     __syncthreads();
    #
    #     neighbours[w_idx * 18 + 12] = v[max(w_idx - 1, 0)] * neighbours_present[w_idx * 9 + 3];
    #     neighbours[w_idx * 18 + 13] = 0.0;
    #     neighbours[w_idx * 18 + 14] = v[min(w_idx + 1, max_idx)] * neighbours_present[w_idx * 9 + 5];
    #
    #     __syncthreads();
    #
    #     neighbours[w_idx * 18 + 15] = v[min(w_idx + Ncol - 1, max_idx)] * neighbours_present[w_idx * 9 + 6];
    #     neighbours[w_idx * 18 + 16] = v[min(w_idx + Ncol, max_idx)] * neighbours_present[w_idx * 9 + 7];
    #     neighbours[w_idx * 18 + 17] = v[min(w_idx + Ncol + 1, max_idx)] * neighbours_present[w_idx * 9 + 8];
    #
    #     __syncthreads();
    # }
    # """)

    mod_get_neighbours = SourceModule("""
    __global__ void get_u_neighbours(float *neighbours, float *neighbours_present, float *u, int Nrow, int Ncol)
    {
        // neighbours - u and v values around each point
        // neighbours_present - 1 if there is a neighbour, 0 if no neighbour
        // u, v - u and v velocities
        // Nrow, Ncol - number of rows and columns

        // references each IW
        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
        int max_idx = Nrow * Ncol;

        if(w_idx >= max_idx){return;}

        // get velocities
        if(neighbours_present[w_idx * 9 + 0] == 1){neighbours[w_idx * 18 + 0] = u[w_idx - Ncol - 1];}
        if(neighbours_present[w_idx * 9 + 1] == 1){neighbours[w_idx * 18 + 1] = u[w_idx - Ncol];}
        if(neighbours_present[w_idx * 9 + 2] == 1){neighbours[w_idx * 18 + 2] = u[w_idx - Ncol + 1];}

        __syncthreads();

        if(neighbours_present[w_idx * 9 + 3] == 1){neighbours[w_idx * 18 + 3] = u[w_idx - 1];}
        //neighbours[w_idx * 18 + 4] = 0.0;
        if(neighbours_present[w_idx * 9 + 5] == 1){neighbours[w_idx * 18 + 5] = u[w_idx + 1];}

        __syncthreads();

        if(neighbours_present[w_idx * 9 + 6] == 1){neighbours[w_idx * 18 + 6] = u[w_idx + Ncol - 1];}
        if(neighbours_present[w_idx * 9 + 7] == 1){neighbours[w_idx * 18 + 7] = u[w_idx + Ncol];}
        if(neighbours_present[w_idx * 9 + 8] == 1){neighbours[w_idx * 18 + 8] = u[w_idx + Ncol + 1];}

        __syncthreads();
    }

    __global__ void get_v_neighbours(float *neighbours, float *neighbours_present, float *v, int Nrow, int Ncol)
    {
        // neighbours - u and v values around each point
        // neighbours_present - 1 if there is a neighbour, 0 if no neighbour
        // u, v - u and v velocities
        // Nrow, Ncol - number of rows and columns

        // references each IW
        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
        int max_idx = Nrow * Ncol;

        if(w_idx >= max_idx){return;}

        // get velocities
        if(neighbours_present[w_idx * 9 + 0] == 1){neighbours[w_idx * 18 + 9] = v[w_idx - Ncol - 1];}
        if(neighbours_present[w_idx * 9 + 1] == 1){neighbours[w_idx * 18 + 10] = v[w_idx - Ncol];}
        if(neighbours_present[w_idx * 9 + 2] == 1){neighbours[w_idx * 18 + 11] = v[w_idx - Ncol + 1];}

        __syncthreads();

        if(neighbours_present[w_idx * 9 + 3] == 1){neighbours[w_idx * 18 + 12] = v[w_idx - 1];}
        //neighbours[w_idx * 18 + 13] = 0.0;
        if(neighbours_present[w_idx * 9 + 5] == 1){neighbours[w_idx * 18 + 14] = v[w_idx + 1];}

        __syncthreads();

        if(neighbours_present[w_idx * 9 + 6] == 1){neighbours[w_idx * 18 + 15] = v[w_idx + Ncol - 1];}
        if(neighbours_present[w_idx * 9 + 7] == 1){neighbours[w_idx * 18 + 16] = v[w_idx + Ncol];}
        if(neighbours_present[w_idx * 9 + 8] == 1){neighbours[w_idx * 18 + 17] = v[w_idx + Ncol + 1];}

        __syncthreads();
    }
    """)

    # set dtype of inputs
    n_row = np.int32(n_row)
    n_col = np.int32(n_col)

    # Get GPU grid dimensions and function
    # block_size = 16
    block_size = 32
    x_blocks = int(n_col * n_row // block_size + 1)
    get_u_neighbours = mod_get_neighbours.get_function("get_u_neighbours")
    get_v_neighbours = mod_get_neighbours.get_function("get_v_neighbours")

    # find neighbours
    d_neighbours_present = gpu_find_neighbours(n_row, n_col)  # .astype(np.float32)  # og
    neighbours = np.zeros((n_row, n_col, 2, 3, 3))
    neighbours = neighbours.astype(np.float32)

    # TODO delete this check
    # # assert statements for data
    # assert neighbours.dtype == np.float32, "Wrong data type for neighbours"
    # assert type(n_row) == np.int32, "Wrong data type for Nrow"
    # assert type(n_col) == np.int32, "Wrong data type for Ncol"

    # send data to the gpu
    d_neighbours = gpuarray.to_gpu(neighbours)

    # Get u and v data
    get_u_neighbours(d_neighbours, d_neighbours_present, d_u, n_row, n_col, block=(block_size, 1, 1),
                     grid=(x_blocks, 1))
    get_v_neighbours(d_neighbours, d_neighbours_present, d_v, n_row, n_col, block=(block_size, 1, 1),
                     grid=(x_blocks, 1))

    # TODO delete this check for NaNs
    # return data
    neighbours = d_neighbours.get()
    a = np.isnan(neighbours)
    assert not a.any(), 'NaNs detected in neighbours'
    # if np.sum(a) > 0:
    #     neighbours[a] = 0.0
    #
    d_neighbours = gpuarray.to_gpu(neighbours)

    return d_neighbours, d_neighbours_present, d_u, d_v


def gpu_mean_vel(d_neighbours, d_neighbours_present, n_row, n_col):
    """Calculates the mean velocity in a 3x3 grid around each point in a velocity field.

    Parameters
    ----------
    d_neighbours: 5D gpuarray - float32
        all the neighbouring velocities of every point
    d_neighbours_present: 4D gpuarray - float32
        indicates if a neighbour is present
    n_row, n_col : int
        number of rows and columns of the velocity field

    Returns
    -------
    u_mean, v_mean : 2D array - float32
        mean velocities at each point

    """
    mod_mean_vel = SourceModule("""
    __global__ void u_mean_vel(float *u_mean, float *n, float *np, int Nrow, int Ncol)
    {
        // mean_u : mean velocity of surrounding points
        // n : velocity of neighbours
        // np : neighbours present
        // Nrow, Ncol: number of rows and columns

        int w_idx = blockIdx.x*blockDim.x + threadIdx.x;

        if(w_idx >= Ncol*Nrow){return;}

        float numerator_u = n[w_idx*18] + n[w_idx*18+1] + n[w_idx*18+2] + n[w_idx*18+3] + n[w_idx*18+5] + n[w_idx*18+6] + n[w_idx*18+7] + n[w_idx*18+8];
        float denominator = np[w_idx*9] + np[w_idx*9+1] + np[w_idx*9+2] + np[w_idx*9+3] + np[w_idx*9+5] + np[w_idx*9+6] + np[w_idx*9+7] + np[w_idx*9+8];

        __syncthreads();

        u_mean[w_idx] = numerator_u / denominator;
    }

    __global__ void v_mean_vel(float *v_mean, float *n, float *np, int Nrow, int Ncol)
    {

        int w_idx = blockIdx.x*blockDim.x + threadIdx.x;

        if(w_idx >= Ncol*Nrow){return;}

        float numerator_v = n[w_idx*18+9] + n[w_idx*18+10] + n[w_idx*18+11] + n[w_idx*18+12] + n[w_idx*18+14] + n[w_idx*18+15] + n[w_idx*18+16] + n[w_idx*18+17];
        float denominator = np[w_idx*9] + np[w_idx*9+1] + np[w_idx*9+2] + np[w_idx*9+3] + np[w_idx*9+5] + np[w_idx*9+6] + np[w_idx*9+7] + np[w_idx*9+8];

        __syncthreads();

        v_mean[w_idx] = numerator_v / denominator;
    }
    """)

    # allocate space for arrays
    u_mean = np.empty((n_row, n_col), dtype=np.float32)
    v_mean = np.empty_like(u_mean)
    n_row = np.int32(n_row)
    n_col = np.int32(n_col)

    # define GPU data
    # block_size = 16
    block_size = 32
    x_blocks = int(n_row * n_col // block_size + 1)

    # assert u_mean.dtype == np.float32, "dtype for u_mean is wrong. Should be np.float32"
    # assert v_mean.dtype == np.float32, "dtype for v_mean is wrong. Should be np.float32"

    # send data to gpu
    d_u_mean = gpuarray.to_gpu(u_mean)
    d_v_mean = gpuarray.to_gpu(v_mean)

    # get and launch kernel
    u_mean_vel = mod_mean_vel.get_function("u_mean_vel")
    v_mean_vel = mod_mean_vel.get_function("v_mean_vel")
    u_mean_vel(d_u_mean, d_neighbours, d_neighbours_present, n_row, n_col, block=(block_size, 1, 1), grid=(x_blocks, 1))
    v_mean_vel(d_v_mean, d_neighbours, d_neighbours_present, n_row, n_col, block=(block_size, 1, 1), grid=(x_blocks, 1))

    return d_u_mean, d_v_mean


def gpu_rms(d_neighbours, d_neighbours_present, n_row, n_col):
    """Calculates the mean velocity in a 3x3 grid around each point in a velocity field.

    Parameters
    ----------
    d_neighbours : 5D gpuarray - float32
        all the neighbouring velocities of every point
    d_neighbours_present : 4D gpuarray - float32
        indicates if a neighbour is present
    n_row, n_col : int
        number of rows and columns of the velocity field

    Returns
    -------
    d_u_rms, d_v_rms : 2D gpuarray - float32
        mean velocities at each point

    """
    mod_rms = SourceModule("""
    __global__ void u_rms_k(float *u_rms, float *n, float *np, int Nrow, int Ncol)
    {

        // Ncol : number of columns in the

        int w_idx = blockIdx.x*blockDim.x + threadIdx.x;

        if(w_idx >= Ncol*Nrow){return;}

        float numerator = (powf(n[w_idx*18+0], 2) + powf(n[w_idx*18+1], 2) + powf(n[w_idx*18+2], 2) + \
                           powf(n[w_idx*18+3], 2) + powf(n[w_idx*18+5], 2) + powf(n[w_idx*18+6], 2) + \
                           powf(n[w_idx*18+7], 2) + powf(n[w_idx*18+8], 2) );
        float denominator = np[w_idx*9] + np[w_idx*9+1] + np[w_idx*9+2] + np[w_idx*9+3] + np[w_idx*9+5] + np[w_idx*9+6] + np[w_idx*9+7] + np[w_idx*9+8];

        __syncthreads();

        u_rms[w_idx] =  sqrtf(numerator / denominator);
    }

    __global__ void v_rms_k(float *v_rms, float *n,float *np, int Nrow, int Ncol)
    {

        // Ncol : number of columns in the

        int w_idx = blockIdx.x*blockDim.x + threadIdx.x;

        if(w_idx >= Ncol*Nrow){return;}

        float numerator = (powf(n[w_idx*18+9], 2) + powf(n[w_idx*18+10], 2) + powf(n[w_idx*18+11], 2) + \
                           powf(n[w_idx*18+12], 2) + powf(n[w_idx*18+14], 2) + powf(n[w_idx*18+15], 2) + \
                           powf(n[w_idx*18+16], 2) + powf(n[w_idx*18+17], 2) );
        float denominator = np[w_idx*9] + np[w_idx*9+1] + np[w_idx*9+2] + np[w_idx*9+3] + np[w_idx*9+5] + np[w_idx*9+6] + np[w_idx*9+7] + np[w_idx*9+8];

        __syncthreads();

        v_rms[w_idx] = sqrtf(numerator / denominator);
    }
    """)

    # allocate space for data
    u_rms = np.empty((n_row, n_col), dtype=np.float32)
    v_rms = np.empty((n_row, n_col), dtype=np.float32)
    n_row = np.int32(n_row)
    n_col = np.int32(n_col)

    # define GPU data
    # block_size = 16
    block_size = 32
    x_blocks = int(n_row * n_col // block_size + 1)

    # TODO delete this check
    # assert u_rms.dtype == np.float32, "dtype for u_rms is wrong. Should be np.float32"
    # assert v_rms.dtype == np.float32, "dtype for v_rms is wrong. Should be np.float32"

    # send data to gpu
    d_u_rms = gpuarray.to_gpu(u_rms)
    d_v_rms = gpuarray.to_gpu(v_rms)

    # get and launch kernel
    mod_u_rms = mod_rms.get_function("u_rms_k")
    mod_v_rms = mod_rms.get_function("v_rms_k")
    mod_u_rms(d_u_rms, d_neighbours, d_neighbours_present, n_row, n_col, block=(block_size, 1, 1), grid=(x_blocks, 1))
    mod_v_rms(d_v_rms, d_neighbours, d_neighbours_present, n_row, n_col, block=(block_size, 1, 1), grid=(x_blocks, 1))

    return d_u_rms, d_v_rms


# TODO check to ensure this function does what it should
def gpu_divergence(d_u, d_v, w, n_row, n_col):
    """Calculates the divergence at each point in a velocity field.

    Parameters
    ----------
    d_u, d_v: 2D array - float
        velocity field
    w: int
        pixel separation between velocity vectors
    n_row, n_col : int
        number of rows and columns of the velocity field

    Returns
    -------
    div : 2D array - float32
        divergence at each point

    """
    mod_div = SourceModule("""
    __global__ void div_k(float *div, float *u, float *v, float w, int Nrow, int Ncol)
    {
        // u : u velocity
        // v : v velocity
        // w : window size
        // Nrow, Ncol : number of rows and columns

        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
        int max_idx = Nrow * Ncol;

        // Avoid the boundary
        if(w_idx >= (Nrow - 1) * Ncol){return;}
        if(w_idx%Ncol == Ncol - 1){return;}

        float u1 = u[w_idx + Ncol];
        float v1 = v[w_idx + 1];

        __syncthreads();

        div[w_idx] = (u1 - u[w_idx]) / w - (v1 - v[w_idx]) / w;
    }

    __global__ void div_boundary_k(float *div, float *u, float *v, float w, int Nrow, int Ncol)
    {
        // u : u velocity
        // v : v velocity
        // w : window size
        // Nrow, Ncol : number of rows and columns

        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;

        // only calculate on the boundary
        if(w_idx < (Nrow - 1) * Ncol && w_idx%Ncol != Ncol - 1){return;}

        float u1 = u[w_idx - Ncol];
        float v1 = v[w_idx - 1];

        __syncthreads();

        div[w_idx] = (u[w_idx] - u1) / w - (v[w_idx] - v1) / w;
    }
    """)

    div = np.empty((n_row, n_col), dtype=np.float32)
    n_row = np.int32(n_row)
    n_col = np.int32(n_col)
    w = np.float32(w)

    # define GPU data
    # block_size = 16
    block_size = 32
    x_blocks = int(n_row * n_col // block_size + 1)

    # TODO delete this check
    # assert div.dtype == np.float32, "dtype of div is {}. Should be np.float32".format(div.dtype)

    # move data to gpu
    d_div = gpuarray.to_gpu(div)

    # get and launch kernel
    div_k = mod_div.get_function("div_k")
    div_boundary_k = mod_div.get_function("div_boundary_k")
    div_k(d_div, d_u, d_v, w, n_row, n_col, block=(block_size, 1, 1), grid=(x_blocks, 1))
    div_boundary_k(d_div, d_u, d_v, w, n_row, n_col, block=(block_size, 1, 1), grid=(x_blocks, 1))

    # get single case of bottom i = 0, j = Ncol-1
    d_div[0, int(n_col - 1)] = (d_u[1, n_col - 1] - d_u[0, n_col - 1]) / w - (d_v[0, n_col - 1] - d_v[0, n_col - 2]) / w
    d_div[int(n_row - 1), 0] = (d_u[n_row - 1, 0] - d_u[n_row - 2, 0]) / w - (d_v[n_row - 1, 1] - d_v[n_row - 1, 0]) / w

    return d_div, d_u, d_v
