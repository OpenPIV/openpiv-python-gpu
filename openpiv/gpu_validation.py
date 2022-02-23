"""This module is for GPU-accelerated validation algorithms."""

from math import ceil

import numpy as np
# Create the PyCUDA context.
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
# import pycuda.cumath as cumath
from pycuda.compiler import SourceModule

from openpiv.gpu_misc import _check_arrays

# Define 32-bit types
DTYPE_i = np.int32
DTYPE_f = np.float32
DTYPE_c = np.complex64

ALLOWED_VALIDATION_METHODS = {'s2n', 'median_velocity', 'mean_velocity', 'rms_velocity'}
S2N_TOL = 2
MEDIAN_TOL = 2
MEAN_TOL = 2
RMS_TOL = 2


def gpu_validation(u_d, v_d, sig2noise_d=None, mask_d=None, validation_method='median_velocity',
                   s2n_tol=S2N_TOL, median_tol=MEDIAN_TOL, mean_tol=MEAN_TOL, rms_tol=RMS_TOL):
    """Returns an array indicating which indices need to be validated.

    Parameters
    ----------
    u_d, v_d : GPUArray
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
    u_median_d, v_median_d : GPUArray
        2D float, mean of the velocities surrounding each point in this iteration.

    """
    # 'mean' in this function refers to either the mean or median estimators of the average.
    _check_arrays(u_d, v_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, shape=u_d.shape, ndim=2)
    if sig2noise_d is not None:
        _check_arrays(sig2noise_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, size=u_d.size)
    if mask_d is not None:
        _check_arrays(sig2noise_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, shape=u_d.shape)
    val_locations_d = None
    u_mean_d = v_mean_d = None

    # Compute the median velocities to be returned.
    neighbours_present_d = _gpu_find_neighbours(u_d.shape)
    u_neighbours_d = _gpu_get_neighbours(u_d, neighbours_present_d)
    v_neighbours_d = _gpu_get_neighbours(v_d, neighbours_present_d)
    u_median_d = _gpu_median_velocity(u_neighbours_d, neighbours_present_d)
    v_median_d = _gpu_median_velocity(v_neighbours_d, neighbours_present_d)

    # Compute the mean velocities if they are needed.
    if 'mean_velocity' in validation_method or 'rms_velocity' in validation_method:
        u_mean_d = _gpu_mean_velocity(u_neighbours_d, neighbours_present_d)
        v_mean_d = _gpu_mean_velocity(v_neighbours_d, neighbours_present_d)

    if 's2n' in validation_method:
        assert sig2noise_d is not None, 's2n validation requires sig2noise to be passed.'

        val_locations_d = _local_validation(sig2noise_d, s2n_tol, val_locations_d)

    if 'median_velocity' in validation_method:
        u_median_fluc_d = _gpu_median_fluc(u_median_d, u_neighbours_d, neighbours_present_d)
        v_median_fluc_d = _gpu_median_fluc(v_median_d, v_neighbours_d, neighbours_present_d)

        val_locations_d = _neighbour_validation(u_d, u_median_d, u_median_fluc_d, median_tol, val_locations_d)
        val_locations_d = _neighbour_validation(v_d, v_median_d, v_median_fluc_d, median_tol, val_locations_d)

    if 'mean_velocity' in validation_method:
        u_mean_fluc_d = _gpu_mean_fluc(u_mean_d, u_neighbours_d, neighbours_present_d)
        v_mean_fluc_d = _gpu_mean_fluc(v_mean_d, v_neighbours_d, neighbours_present_d)

        val_locations_d = _neighbour_validation(u_d, u_mean_d, u_mean_fluc_d, mean_tol, val_locations_d)
        val_locations_d = _neighbour_validation(v_d, v_mean_d, v_mean_fluc_d, mean_tol, val_locations_d)

    if 'rms_velocity' in validation_method:
        u_rms_d = _gpu_rms(u_mean_d, u_neighbours_d, neighbours_present_d)
        v_rms_d = _gpu_rms(v_mean_d, v_neighbours_d, neighbours_present_d)

        val_locations_d = _neighbour_validation(u_d, u_mean_d, u_rms_d, rms_tol, val_locations_d)
        val_locations_d = _neighbour_validation(v_d, v_mean_d, v_rms_d, rms_tol, val_locations_d)

    return val_locations_d, u_median_d, v_median_d


mod_validation = SourceModule("""
__global__ void neighbour_validation(int *val_list, float *f, float *f_mean, float *f_fluc, float tol, int size)
{
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= size) {return;}

    // a small number is added to prevent singularities in uniform flow (Scarano & Westerweel, 2005)
    val_list[t_idx] = val_list[t_idx] * (fabsf(f[t_idx] - f_mean[t_idx]) / (f_fluc[t_idx] + 0.1f) < tol);
}

__global__ void local_validation(int *val_list, float *sig2noise, float tol, int size)
{
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= size) {return;}

    val_list[t_idx] = val_list[t_idx] * (sig2noise[t_idx] > tol);
}
""")


def _local_validation(f_d, tol, val_locations_d=None):
    """Updates the validation list by checking if the array elements exceed the tolerance."""
    size_i = DTYPE_i(f_d.size)
    tol_f = DTYPE_f(tol)

    if val_locations_d is None:
        val_locations_d = gpuarray.ones_like(f_d)

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
        val_locations_d = gpuarray.ones_like(f_d)

    block_size = 32
    grid_size = ceil(size_i / block_size)
    neighbour_validation = mod_validation.get_function('neighbour_validation')
    neighbour_validation(val_locations_d, f_d, f_mean_d, f_mean_fluc_d, tol_f, size_i, block=(block_size, 1, 1),
                         grid=(grid_size, 1))

    return val_locations_d


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

__global__ void get_neighbours(float *nb, int *np, float *f, int n, int size)
{
    // nb - values of the neighbouring points
    // np - 1 if there is a neighbour, 0 if no neighbour
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= size) {return;}

    // get neighbouring values
    if (np[t_idx * 9 + 0]) {nb[t_idx * 9 + 0] = f[t_idx - n - 1];}
    if (np[t_idx * 9 + 1]) {nb[t_idx * 9 + 1] = f[t_idx - n];}
    if (np[t_idx * 9 + 2]) {nb[t_idx * 9 + 2] = f[t_idx - n + 1];}

    if (np[t_idx * 9 + 3]) {nb[t_idx * 9 + 3] = f[t_idx - 1];}
    nb[t_idx * 9 + 4] = 0.0f;
    if (np[t_idx * 9 + 5]) {nb[t_idx * 9 + 5] = f[t_idx + 1];}

    if (np[t_idx * 9 + 6]) {nb[t_idx * 9 + 6] = f[t_idx + n - 1];}
    if (np[t_idx * 9 + 7]) {nb[t_idx * 9 + 7] = f[t_idx + n];}
    if (np[t_idx * 9 + 8]) {nb[t_idx * 9 + 8] = f[t_idx + n + 1];}
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
        4D (m, n, 3, 3), whether the point in the field has neighbours.

    """
    m, n = shape
    size_i = DTYPE_i(m * n)
    if mask_d is None:
        pass

    neighbours_present_d = gpuarray.empty((m, n, 3, 3), dtype=DTYPE_i)

    block_size = 32
    grid_size = ceil(size_i / block_size)
    find_neighbours = mod_neighbours.get_function('find_neighbours')
    find_neighbours(neighbours_present_d, DTYPE_i(n), DTYPE_i(m), size_i, block=(block_size, 1, 1), grid=(grid_size, 1))

    return neighbours_present_d


def _gpu_get_neighbours(f_d, neighbours_present_d):
    """An array that stores the values of the velocity of the surrounding neighbours.

    Parameters
    ----------
    f_d : GPUArray
        2D float, values from which to get neighbours.
    neighbours_present_d : GPUArray
        4D int (m, n, 3, 3), locations where neighbours exist.

    Returns
    -------
    GPUArray
        4D float (m, n, 3, 3), values of u and v of the neighbours of a point.

    """
    m, n = f_d.shape
    size_i = DTYPE_i(f_d.size)

    neighbours_d = gpuarray.empty((m, n, 3, 3), dtype=DTYPE_f)

    block_size = 32
    grid_size = ceil(size_i / block_size)
    get_u_neighbours = mod_neighbours.get_function('get_neighbours')
    get_u_neighbours(neighbours_d, neighbours_present_d, f_d, DTYPE_i(n), size_i, block=(block_size, 1, 1),
                     grid=(grid_size, 1))

    return neighbours_d


mod_median_velocity = SourceModule("""
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

__global__ void median_velocity(float *f_median, float *nb, int *np, int size)
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
            A[j] = nb[t_idx * 9 + i];
            B[j++] = np[t_idx * 9 + i];
        }
    }
    // sort the array
    sort(A, B);

    // count the neighbouring points
    int N = B[0] + B[1] + B[2] + B[3] + B[4] + B[5] + B[6] + B[7];

    // return the median
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

    // loop through neighbours to populate an array to sort
    int i;
    int j = 0;
    float A[8];
    float B[8];
    for (i = 0; i < 9; i++)
    {
        if (i != 4)
        {
            A[j] = fabsf(nb[t_idx * 9 + i] - f_m);
            B[j++] = np[t_idx * 9 + i];
        }
    }
    // sort the array
    sort(A, B);

    // count the neighbouring points
    int N = B[0] + B[1] + B[2] + B[3] + B[4] + B[5] + B[6] + B[7];

    // return the median
    if (N % 2 == 0) {f_median_fluc[t_idx] = (A[N / 2 - 1] + A[N / 2]) / 2;}
    else {f_median_fluc[t_idx] = A[N / 2];}
}
""")


def _gpu_median_velocity(neighbours_d, neighbours_present_d):
    """Calculates the median velocity on a 3x3 grid around each point in a velocity field.

    Parameters
    ----------
    neighbours_d: GPUArray
        4D float (m, n, 2, 3, 3), all the neighbouring velocities of every point.
    neighbours_present_d: GPUArray
        4D int (m, n, 3, 3), indicates if a neighbour is present.

    Returns
    -------
    GPUArray
        2D float, mean velocities at each point.

    """
    m, n, _, _ = neighbours_d.shape
    size_i = DTYPE_i(m * n)

    f_median_d = gpuarray.empty((m, n), dtype=DTYPE_f)

    block_size = 32
    grid_size = ceil(size_i / block_size)
    median_velocity = mod_median_velocity.get_function('median_velocity')
    median_velocity(f_median_d, neighbours_d, neighbours_present_d, size_i, block=(block_size, 1, 1),
                    grid=(grid_size, 1))

    return f_median_d


def _gpu_median_fluc(f_median_d, d_neighbours, d_neighbours_present):
    """Calculates the magnitude of the median velocity fluctuations on a 3x3 grid around each point in a velocity field.

    Parameters
    ----------
    f_median_d : GPUArray
        2D float, mean velocities around each point.
    d_neighbours : GPUArray
        4D float (m, n, 3, 3), all the neighbouring velocities of every point.
    d_neighbours_present : GPUArray
        4D int  (m, n, 3, 3), indicates if a neighbour is present.

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
    median_u_fluc(f_median_fluc_d, f_median_d, d_neighbours, d_neighbours_present, size_i, block=(block_size, 1, 1),
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
    float denominator = np[t_idx * 9 + 0] + np[t_idx * 9 + 1] + np[t_idx * 9 + 2] + np[t_idx * 9 + 3]
                        + np[t_idx * 9 + 5] + np[t_idx * 9 + 6] + np[t_idx * 9 + 7] + np[t_idx * 9 + 8];

    // ensure denominator is not zero then compute mean
    if (denominator > 0) {
        float numerator = nb[t_idx * 9 + 0] + nb[t_idx * 9 + 1] + nb[t_idx * 9 + 2] + nb[t_idx * 9 + 3]
                            + nb[t_idx * 9 + 5] + nb[t_idx * 9 + 6] + nb[t_idx * 9 + 7] + nb[t_idx * 9 + 8];

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
    float denominator = np[t_idx * 9 + 0] + np[t_idx * 9 + 1] + np[t_idx * 9 + 2] + np[t_idx * 9 + 3]
                        + np[t_idx * 9 + 5] + np[t_idx * 9 + 6] + np[t_idx * 9 + 7] + np[t_idx * 9 + 8];

    // ensure denominator is not zero then compute fluctuations
    if (denominator > 0) {
        float f_m = f_mean[t_idx];
        float numerator = fabsf(nb[t_idx * 9 + 0] - f_m) + fabsf(nb[t_idx * 9 + 1] - f_m)
                          + fabsf(nb[t_idx * 9 + 2] - f_m) + fabsf(nb[t_idx * 9 + 3] - f_m)
                          + fabsf(nb[t_idx * 9 + 5] - f_m) + fabsf(nb[t_idx * 9 + 6] - f_m)
                          + fabsf(nb[t_idx * 9 + 7] - f_m) + fabsf(nb[t_idx * 9 + 8] - f_m);

        f_fluc[t_idx] = numerator / denominator;
    } else {f_fluc[t_idx] = 0;}
}
""")


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
    GPUArray
        2D float, mean velocities at each point.

    """
    m, n, _, _ = neighbours_d.shape
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
        4D float (m, n, 3, 3), all the neighbouring velocities of every point.
    neighbours_present_d : GPUArray
        4D int (m, n, 3, 3), indicates if a neighbour is present.

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
    float denominator = np[t_idx * 9 + 0] + np[t_idx * 9 + 1] + np[t_idx * 9 + 2] + np[t_idx * 9 + 3]
                        + np[t_idx * 9 + 5] + np[t_idx * 9 + 6] + np[t_idx * 9 + 7] + np[t_idx * 9 + 8];

    // ensure denominator is not zero then compute rms
    if (denominator > 0) {
        float f_m = f_mean[t_idx];
        float numerator = (powf(nb[t_idx * 9 + 0] - f_m, 2) + powf(nb[t_idx * 9 + 1] - f_m, 2)
                           + powf(nb[t_idx * 9 + 2] - f_m, 2) + powf(nb[t_idx * 9 + 3] - f_m, 2)
                           + powf(nb[t_idx * 9 + 5] - f_m, 2) + powf(nb[t_idx * 9 + 6] - f_m, 2)
                           + powf(nb[t_idx * 9 + 7] - f_m, 2) + powf(nb[t_idx * 9 + 8] - f_m, 2));

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
        4D float (m, n, 3, 3), all the neighbouring velocities of every point.
    neighbours_present_d : GPUArray
        4D int (m, n, 3, 3), indicates if a neighbour is present.

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
