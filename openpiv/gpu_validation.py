"""This module is for GPU-accelerated validation algorithms."""

from math import ceil

import numpy as np
# Create the PyCUDA context.
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
# import pycuda.cumath as cumath
from pycuda.compiler import SourceModule

from openpiv.gpu_misc import _check_arrays, gpu_mask

# Define 32-bit types
DTYPE_i = np.int32
DTYPE_f = np.float32
DTYPE_c = np.complex64

ALLOWED_VALIDATION_METHODS = {'s2n', 'median_velocity', 'mean_velocity', 'rms_velocity'}
S2N_TOL = 2
MEDIAN_TOL = 2
MEAN_TOL = 2
RMS_TOL = 2


def gpu_validation(*f_dl, sig2noise_d=None, mask_d=None, validation_method='median_velocity',
                   s2n_tol=S2N_TOL, median_tol=MEDIAN_TOL, mean_tol=MEAN_TOL, rms_tol=RMS_TOL):
    """Returns an array indicating which indices need to be validated.

    Parameters
    ----------
    f_dl : GPUArray
        2D float, velocity fields to be validated.
    sig2noise_d : GPUArray, optional
        1D or 2D float, signal-to-noise ratio of each velocity.
    mask_d : GPUArray
        2D float, mask for the velocity field.
    validation_method : {'s2n', 'median_velocity', 'mean_velocity', 'rms_velocity'}, optional
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
    val_locations : GPUArray
        2D int, array of indices that need to be validated. 0 indicates that the index needs to be corrected. 1 means
        no correction is needed.
    f_median_d : GPUArray or list
        2D float, mean of the velocities surrounding each point in this iteration.

    """
    n_y = len(f_dl)
    f_shape = f_dl[0].shape
    f_size = f_dl[0].size
    _check_arrays(*f_dl, array_type=gpuarray.GPUArray, dtype=DTYPE_f, shape=f_shape, ndim=2)
    if sig2noise_d is not None:
        _check_arrays(sig2noise_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, size=f_size)
    if mask_d is not None:
        _check_arrays(mask_d, array_type=gpuarray.GPUArray, dtype=DTYPE_i, shape=f_shape)
    val_locations_d = None

    if 's2n' in validation_method:
        assert sig2noise_d is not None, 's2n validation requires sig2noise to be passed.'
        val_locations_d = _local_validation(sig2noise_d, s2n_tol, val_locations_d)

    # Compute the median velocities to be returned.
    neighbours_present_d = _gpu_find_neighbours(f_dl[0].shape, mask_d)

    f_median_dl = []
    for f_d in f_dl:
        f_neighbours_d = _gpu_get_neighbours(f_d, neighbours_present_d)
        f_median_d = _gpu_median_velocity(f_neighbours_d, neighbours_present_d)
        f_mean_d = None

        if 'median_velocity' in validation_method:
            f_median_fluc_d = _gpu_median_fluc(f_median_d, f_neighbours_d, neighbours_present_d)
            val_locations_d = _neighbour_validation(f_d, f_median_d, f_median_fluc_d, median_tol, val_locations_d)

        # Compute the mean velocities if they are needed.
        if 'mean_velocity' in validation_method or 'rms_velocity' in validation_method:
            f_mean_d = _gpu_mean_velocity(f_neighbours_d, neighbours_present_d)

        if 'mean_velocity' in validation_method:
            f_mean_fluc_d = _gpu_mean_fluc(f_mean_d, f_neighbours_d, neighbours_present_d)
            val_locations_d = _neighbour_validation(f_d, f_mean_d, f_mean_fluc_d, mean_tol, val_locations_d)

        if 'rms_velocity' in validation_method:
            f_rms_d = _gpu_rms(f_mean_d, f_neighbours_d, neighbours_present_d)
            val_locations_d = _neighbour_validation(f_d, f_mean_d, f_rms_d, rms_tol, val_locations_d)

        f_median_dl.append(f_median_d)

    if mask_d is not None:
        val_locations_d = gpu_mask(val_locations_d, mask_d)

    if n_y == 1:
        f_median_dl = f_median_dl[0]

    return val_locations_d, f_median_dl


mod_validation = SourceModule("""
__global__ void neighbour_validation(int *val_locations, float *f, float *f_mean, float *f_fluc, float tol, int size)
{
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= size) {return;}

    // a small number is added to prevent singularities in uniform flow (Scarano & Westerweel, 2005)
    val_locations[t_idx] = val_locations[t_idx] || (fabsf(f[t_idx] - f_mean[t_idx]) / (f_fluc[t_idx] + 0.1f) > tol);
}

__global__ void local_validation(int *val_locations, float *f, float tol, int size)
{
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= size) {return;}

    val_locations[t_idx] = val_locations[t_idx] || (f[t_idx] <= tol);
}
""")


def _local_validation(f_d, tol, val_locations_d=None):
    """Updates the validation list by checking if the array elements exceed the tolerance."""
    size_i = DTYPE_i(f_d.size)
    tol_f = DTYPE_f(tol)

    if val_locations_d is None:
        val_locations_d = gpuarray.zeros_like(f_d, dtype=DTYPE_i)

    block_size = 32
    grid_size = ceil(size_i / block_size)
    local_validation = mod_validation.get_function('local_validation')
    local_validation(val_locations_d, f_d, tol_f, size_i, block=(block_size, 1, 1), grid=(grid_size, 1))

    return val_locations_d


def _neighbour_validation(f_d, f_mean_d, f_mean_fluc_d, tol, val_locations_d=None):
    """Updates the validation list by checking if the neighbouring elements exceed the tolerance."""
    size_i = DTYPE_i(f_d.size)
    tol_f = DTYPE_f(tol)

    if val_locations_d is None:
        val_locations_d = gpuarray.zeros_like(f_d, dtype=DTYPE_i)

    block_size = 32
    grid_size = ceil(size_i / block_size)
    neighbour_validation = mod_validation.get_function('neighbour_validation')
    neighbour_validation(val_locations_d, f_d, f_mean_d, f_mean_fluc_d, tol_f, size_i, block=(block_size, 1, 1),
                         grid=(grid_size, 1))

    return val_locations_d


mod_neighbours = SourceModule("""
__global__ void find_neighbours(int *np, int *mask, int n, int m, int size)
{
    // np : neighbours_present
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= size) {return;}

    int nb_idx = t_idx % 8;
    int row_idx = t_idx / 8 / n - (nb_idx < 3) + (nb_idx > 4); 
    int col_idx = t_idx / 8 % n - ((nb_idx == 0) || (nb_idx == 3) || (nb_idx == 5))
                                + ((nb_idx == 2) || (nb_idx == 4) || (nb_idx == 7));
    int in_bound = (row_idx >= 0) * (row_idx < m) * (col_idx >= 0) * (col_idx < n);

    np[t_idx] = in_bound * (mask[(row_idx * n + col_idx) * in_bound] == 0);
}

__global__ void get_neighbours(float *nb, int *np, float *f, int n, int size)
{
    // nb - values of the neighbouring points
    // np - 1 if there is a neighbour, 0 if no neighbour
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= size) {return;}
    
    int nb_idx = t_idx % 8;
    int row_idx = t_idx / 8 / n - (nb_idx < 3) + (nb_idx > 4); 
    int col_idx = t_idx / 8 % n - ((nb_idx == 0) || (nb_idx == 3) || (nb_idx == 5))
                                + ((nb_idx == 2) || (nb_idx == 4) || (nb_idx == 7));

    // get neighbouring values
    nb[t_idx] = f[(row_idx * n + col_idx) * np[t_idx]] * np[t_idx];
}
""")


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
        4D (m, n, 8), whether the point in the field has neighbours.

    """
    m, n = shape
    size_i = DTYPE_i(m * n * 8)
    if mask_d is None:
        mask_d = gpuarray.zeros((m, n), dtype=DTYPE_i)

    neighbours_present_d = gpuarray.empty((m, n, 8), dtype=DTYPE_i)

    block_size = 32
    grid_size = ceil(size_i / block_size)
    find_neighbours = mod_neighbours.get_function('find_neighbours')
    find_neighbours(neighbours_present_d, mask_d, DTYPE_i(n), DTYPE_i(m), size_i, block=(block_size, 1, 1),
                    grid=(grid_size, 1))

    return neighbours_present_d


def _gpu_get_neighbours(f_d, neighbours_present_d):
    """An array that stores the values of the velocity of the surrounding neighbours.

    Parameters
    ----------
    f_d : GPUArray
        2D float, values from which to get neighbours.
    neighbours_present_d : GPUArray
        4D int (m, n, 8), locations where neighbours exist.

    Returns
    -------
    GPUArray
        4D float (m, n, 8), values of u and v of the neighbours of a point.

    """
    m, n = f_d.shape
    size_i = DTYPE_i(f_d.size * 8)

    neighbours_d = gpuarray.empty((m, n, 8), dtype=DTYPE_f)

    block_size = 32
    grid_size = ceil(size_i / block_size)
    get_neighbours = mod_neighbours.get_function('get_neighbours')
    get_neighbours(neighbours_d, neighbours_present_d, f_d, DTYPE_i(n), size_i, block=(block_size, 1, 1),
                   grid=(grid_size, 1))

    return neighbours_d


mod_median_velocity = SourceModule("""
// device-side function to swap elements of two arrays.
__device__ void swap(float *A, int a, int b)
{
    float tmp_A = A[a];
    A[a] = A[b];
    A[b] = tmp_A;
}

// device-side function to compare and swap elements of two arrays.
__device__ void compare(float *A, float *B, int a, int b)
{
    // Move non-neighbour values to end.
    if (B[a] < B[b])
    {
        swap(A, a, b);
        swap(B, a, b);
    }
    // Move greater values to right.
    else if (A[a] > A[b] && B[a] == B[b] == 1)
    {
        swap(A, a, b);
        swap(B, a, b);
    }
}

// device-side function to do an 8-wire sorting network.
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

__global__ void median_velocity(float *f_median, float *nb, int *np, int size)
{
    // nb - values of the neighbouring points.
    // np - 1 if there is a neighbour, 0 if no neighbour.
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= size) {return;}

    // Loop through neighbours to populate an array to sort.
    int i;
    int j = 0;
    float A[8];
    float B[8];
    for (i = 0; i < 8; i++) {
        A[j] = nb[t_idx * 8 + i];
        B[j++] = np[t_idx * 8 + i];
    }
    // Sort the array.
    sort(A, B);

    // Count the neighbouring points.
    int N = B[0] + B[1] + B[2] + B[3] + B[4] + B[5] + B[6] + B[7];

    // Return the median out of N neighbours.
    if (N % 2 == 0) {f_median[t_idx] = (A[N / 2 - 1] + A[N / 2]) / 2;}
    else {f_median[t_idx] = A[N / 2];}
}

__global__ void median_fluc(float *f_median_fluc, float *f_median, float *nb, int *np, int size)
{
    // nb - value of the neighbouring points
    // np - 1 if there is a neighbour, 0 if no neighbour
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= size) {return;}

    float f_m = f_median[t_idx];

    // Loop through neighbours to populate an array to sort.
    int i;
    int j = 0;
    float A[8];
    float B[8];
    for (i = 0; i < 8; i++) {
        A[j] = fabsf(nb[t_idx * 8 + i] - f_m);
        B[j++] = np[t_idx * 8 + i];
    }
    // Sort the array
    sort(A, B);

    // Count the neighbouring points.
    int N = B[0] + B[1] + B[2] + B[3] + B[4] + B[5] + B[6] + B[7];

    // Return the median out of N neighbours.
    if (N % 2 == 0) {f_median_fluc[t_idx] = (A[N / 2 - 1] + A[N / 2]) / 2;}
    else {f_median_fluc[t_idx] = A[N / 2];}
}
""")


def _gpu_median_velocity(neighbours_d, neighbours_present_d):
    """Calculates the median velocity on a 3x3 grid around each point in a velocity field.

    Parameters
    ----------
    neighbours_d: GPUArray
        4D float (m, n, 8), all the neighbouring velocities of every point.
    neighbours_present_d: GPUArray
        4D int (m, n, 8), indicates if a neighbour is present.

    Returns
    -------
    GPUArray
        2D float, mean velocities at each point.

    """
    m, n, _ = neighbours_d.shape
    size_i = DTYPE_i(m * n)

    f_median_d = gpuarray.empty((m, n), dtype=DTYPE_f)

    block_size = 32
    grid_size = ceil(size_i / block_size)
    median_velocity = mod_median_velocity.get_function('median_velocity')
    median_velocity(f_median_d, neighbours_d, neighbours_present_d, size_i, block=(block_size, 1, 1),
                    grid=(grid_size, 1))

    return f_median_d


def _gpu_median_fluc(f_median_d, neighbours_d, neighbours_present_d):
    """Calculates the magnitude of the median velocity fluctuations on a 3x3 grid around each point in a velocity field.

    Parameters
    ----------
    f_median_d : GPUArray
        2D float, mean velocities around each point.
    neighbours_d : GPUArray
        4D float (m, n, 8), all the neighbouring velocities of every point.
    neighbours_present_d : GPUArray
        4D int  (m, n, 8), indicates if a neighbour is present.

    Returns
    -------
    GPUArray
        2D float, RMS velocities at each point.

    """
    m, n = f_median_d.shape
    size_i = DTYPE_i(f_median_d.size)

    f_median_fluc_d = gpuarray.empty((m, n), dtype=DTYPE_f)

    block_size = 32
    grid_size = ceil(size_i / block_size)
    median_u_fluc = mod_median_velocity.get_function('median_fluc')
    median_u_fluc(f_median_fluc_d, f_median_d, neighbours_d, neighbours_present_d, size_i, block=(block_size, 1, 1),
                  grid=(grid_size, 1))

    return f_median_fluc_d


mod_mean_velocity = SourceModule("""
__global__ void mean_velocity(float *f_mean, float *nb, int *np, int size)
{
    // n : value of neighbours
    // np : neighbours present
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= size) {return;}

    // mean is normalized by number of terms summed
    float denominator = np[t_idx * 8 + 0] + np[t_idx * 8 + 1] + np[t_idx * 8 + 2] + np[t_idx * 8 + 3]
                        + np[t_idx * 8 + 4] + np[t_idx * 8 + 5] + np[t_idx * 8 + 6] + np[t_idx * 8 + 7];

    // ensure denominator is not zero then compute mean
    if (denominator > 0) {
        float numerator = nb[t_idx * 8 + 0] + nb[t_idx * 8 + 1] + nb[t_idx * 8 + 2] + nb[t_idx * 8 + 3]
                            + nb[t_idx * 8 + 4] + nb[t_idx * 8 + 5] + nb[t_idx * 8 + 6] + nb[t_idx * 8 + 7];

        f_mean[t_idx] = numerator / denominator;
    } else {f_mean[t_idx] = 0;}
}

__global__ void mean_fluc(float *f_fluc, float *f_mean, float *nb, int *np, int size)
{
    // nb - value of the neighbouring points
    // np - 1 if there is a neighbour, 0 if no neighbour
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= size) {return;}

    // mean is normalized by number of terms summed
    float denominator = np[t_idx * 8 + 0] + np[t_idx * 8 + 1] + np[t_idx * 8 + 2] + np[t_idx * 8 + 3]
                        + np[t_idx * 8 + 4] + np[t_idx * 8 + 5] + np[t_idx * 8 + 6] + np[t_idx * 8 + 7];

    // ensure denominator is not zero then compute fluctuations
    if (denominator > 0) {
        float f_m = f_mean[t_idx];
        float numerator = fabsf(nb[t_idx * 8 + 0] - f_m) + fabsf(nb[t_idx * 8 + 1] - f_m)
                          + fabsf(nb[t_idx * 8 + 2] - f_m) + fabsf(nb[t_idx * 8 + 3] - f_m)
                          + fabsf(nb[t_idx * 8 + 4] - f_m) + fabsf(nb[t_idx * 8 + 5] - f_m)
                          + fabsf(nb[t_idx * 8 + 6] - f_m) + fabsf(nb[t_idx * 8 + 7] - f_m);

        f_fluc[t_idx] = numerator / denominator;
    } else {f_fluc[t_idx] = 0;}
}
""")


def _gpu_mean_velocity(neighbours_d, neighbours_present_d):
    """Calculates the mean velocity on a 3x3 grid around each point in a velocity field.

    Parameters
    ----------
    neighbours_d: GPUArray
        4D float (m, n, 8), all the neighbouring velocities of every point.
    neighbours_present_d: GPUArray
        4D int (m, n, 8), indicates if a neighbour is present.

    Returns
    -------
    GPUArray
        2D float, mean velocities at each point.

    """
    m, n, _ = neighbours_d.shape
    size_i = DTYPE_i(m * n)

    f_mean_d = gpuarray.empty((m, n), dtype=DTYPE_f)

    block_size = 32
    grid_size = ceil(size_i / block_size)
    mean_velocity = mod_mean_velocity.get_function('mean_velocity')
    mean_velocity(f_mean_d, neighbours_d, neighbours_present_d, size_i, block=(block_size, 1, 1), grid=(grid_size, 1))
    return f_mean_d


mod_mean_fluc = SourceModule("""

""")


def _gpu_mean_fluc(f_mean_d, neighbours_d, neighbours_present_d):
    """Calculates the magnitude of the mean velocity fluctuations on a 3x3 grid around each point in a velocity field.

    Parameters
    ----------
    f_mean_d: GPUArray
        2D float, mean velocities around each point.
    neighbours_d : GPUArray
        4D float (m, n, 8), all the neighbouring velocities of every point.
    neighbours_present_d : GPUArray
        4D int (m, n, 8), indicates if a neighbour is present.

    Returns
    -------
    GPUArray
        2D float, rms velocities at each point.

    """
    m, n = f_mean_d.shape
    size_i = DTYPE_i(f_mean_d.size)

    f_fluc_d = gpuarray.empty((m, n), dtype=DTYPE_f)

    block_size = 32
    grid_size = ceil(size_i / block_size)
    mean_fluc = mod_mean_velocity.get_function('mean_fluc')
    mean_fluc(f_fluc_d, f_mean_d, neighbours_d, neighbours_present_d, size_i, block=(block_size, 1, 1),
              grid=(grid_size, 1))

    return f_fluc_d


mod_rms = SourceModule("""
__global__ void rms(float *f_rms, float *f_mean, float *nb, int *np, int size)
{
    // nb - value of the neighbouring points
    // np - 1 if there is a neighbour, 0 if no neighbour
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= size) {return;}

    // rms is normalized by number of terms summed
    float denominator = np[t_idx * 8 + 0] + np[t_idx * 8 + 1] + np[t_idx * 8 + 2] + np[t_idx * 8 + 3]
                        + np[t_idx * 8 + 4] + np[t_idx * 8 + 5] + np[t_idx * 8 + 6] + np[t_idx * 8 + 7];

    // ensure denominator is not zero then compute rms
    if (denominator > 0) {
        float f_m = f_mean[t_idx];
        float numerator = (powf(nb[t_idx * 8 + 0] - f_m, 2) + powf(nb[t_idx * 8 + 1] - f_m, 2)
                           + powf(nb[t_idx * 8 + 2] - f_m, 2) + powf(nb[t_idx * 8 + 3] - f_m, 2)
                           + powf(nb[t_idx * 8 + 4] - f_m, 2) + powf(nb[t_idx * 8 + 5] - f_m, 2)
                           + powf(nb[t_idx * 8 + 6] - f_m, 2) + powf(nb[t_idx * 8 + 7] - f_m, 2));

        f_rms[t_idx] = sqrtf(numerator / denominator);
    } else {f_rms[t_idx] = 0.0f;}
}
""")


def _gpu_rms(f_mean_d, neighbours_d, neighbours_present_d):
    """Calculates the rms velocity in a 3x3 grid around each point in a velocity field.

    Parameters
    ----------
    f_mean_d : GPUArray
        2D float, mean velocities around each point.
    neighbours_d : GPUArray
        4D float (m, n, 8), all the neighbouring velocities of every point.
    neighbours_present_d : GPUArray
        4D int (m, n, 8), indicates if a neighbour is present.

    Returns
    -------
    GPUArray
        2D float, RMS velocities at each point.

    """
    m, n = f_mean_d.shape
    size_i = DTYPE_i(f_mean_d.size)

    f_rms_d = gpuarray.empty((m, n), dtype=DTYPE_f)

    block_size = 32
    grid_size = ceil(size_i / block_size)
    u_rms = mod_rms.get_function('rms')
    u_rms(f_rms_d, f_mean_d, neighbours_d, neighbours_present_d, size_i, block=(block_size, 1, 1),
          grid=(grid_size, 1))

    return f_rms_d
