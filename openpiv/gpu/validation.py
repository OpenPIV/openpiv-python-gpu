"""This module is for GPU-accelerated validation algorithms."""

from math import ceil, prod, log10
import numpy as np

import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

# import pycuda.cumath as cumath

# noinspection PyUnresolvedReferences
import pycuda.autoinit

from gpu.misc import _check_arrays, gpu_mask, _Subset, _Number

# Define 32-bit types
DTYPE_i = np.int32
DTYPE_f = np.float32
DTYPE_c = np.complex64

ALLOWED_VALIDATION_METHODS = {"s2n", "median_velocity", "mean_velocity", "rms_velocity"}
S2N_TOL = 2
MEDIAN_TOL = 2
MEAN_TOL = 2
RMS_TOL = 2
_BLOCK_SIZE = 64


def gpu_validation(
    *f,
    s2n=None,
    mask=None,
    validation_method="median_velocity",
    s2n_tol=S2N_TOL,
    median_tol=MEDIAN_TOL,
    mean_tol=MEAN_TOL,
    rms_tol=RMS_TOL
):
    """Validates 2D vector-fields and returns an array indicating which location need to
    be validated.

    Parameters
    ----------
    f : GPUArray
        2D float (m, n), velocity fields to be validated.
    s2n : GPUArray or None, optional
        2D float (m, n), signal-to-noise ratio of each velocity.
    mask : GPUArray or None
        2D int (m, n), mask for the velocity field.
    validation_method : str {'s2n', 'median_velocity', 'mean_velocity', 'rms_velocity'},
        optional
        Method(s) to use for validation.
    s2n_tol : float, optional
        Minimum value for validation by signal-to-noise ratio.
    median_tol : float, optional
        Tolerance for median velocity validation.
    mean_tol : float, optional
        Tolerance for mean velocity validation.
    rms_tol : float, optional
        Tolerance for rms validation.

    Returns
    -------
    GPUArray
        2D int (m, n), array of indices that need to be validated. 1s indicate locations
        of invalid vectors.

    """
    validation_gpu = ValidationGPU(
        f[0], mask, validation_method, s2n_tol, median_tol, mean_tol, rms_tol
    )

    return validation_gpu(*f, s2n=s2n)


class ValidationGPU:
    """Validates vector-fields and returns an array indicating which location need to be
    validated.

    Parameters
    ----------
    f_shape : GPUArray or tuple of ints
        (ht, wd) of the fields to be validated.
    mask : GPUArray or None
        2D float, mask for the velocity field.
    validation_method : str {'s2n', 'median_velocity', 'mean_velocity', 'rms_velocity'},
        optional
        Method(s) to use for validation.
    s2n_tol : float, optional
        Minimum value for validation by signal-to-noise ratio.
    median_tol : float, optional
        Tolerance for median velocity validation.
    mean_tol : float, optional
        Tolerance for mean velocity validation.
    rms_tol : float, optional
        Tolerance for rms validation.

    """

    validation_method = _Subset(*ALLOWED_VALIDATION_METHODS)
    s2n_tol = _Number(min_value=0)
    median_tol = _Number(min_value=0, min_closure=False)
    mean_tol = _Number(min_value=0, min_closure=False)
    rms_tol = _Number(min_value=0, min_closure=False)

    def __init__(
        self,
        f_shape,
        mask=None,
        validation_method="median_velocity",
        s2n_tol=S2N_TOL,
        median_tol=MEDIAN_TOL,
        mean_tol=MEAN_TOL,
        rms_tol=RMS_TOL,
    ):
        self.f_shape = f_shape.shape if hasattr(f_shape, "shape") else tuple(f_shape)
        if mask is not None:
            _check_arrays(
                mask, array_type=gpuarray.GPUArray, dtype=DTYPE_i, shape=self.f_shape
            )
        self.mask = mask
        self.validation_method = validation_method
        self.validation_tols = {
            "s2n": s2n_tol,
            "median": median_tol,
            "mean": mean_tol,
            "rms": rms_tol,
        }

        self._n_val = None
        self.val_locations = None
        self._f = None
        self._neighbours_ = None
        self._median_ = None
        self._mean_ = None

        self._check_validation_methods()
        self._check_validation_tolerances()

        # Compute the median velocities to be returned.
        self._neighbours_present = _gpu_find_neighbours(self.f_shape, mask)

    def __call__(self, *f, s2n=None):
        """Returns an array indicating which indices need to be validated.

        Parameters
        ----------
        f : GPUArray
            2D float (m, n), velocity fields to be validated.
        s2n : GPUArray or None, optional
            2D float (m, n), signal-to-noise ratio of each velocity.

        Returns
        -------
        val_locations : GPUArray
            2D int (m, n), array of indices that need to be validated. 1s indicate
            locations of invalid vectors.

        """
        self._clear_validation_data()
        self._n_val = None
        _check_arrays(
            *f, array_type=gpuarray.GPUArray, dtype=DTYPE_f, shape=self.f_shape
        )
        self._num_fields = len(f)
        self._f = f

        # Do the validations.
        if "s2n" in self.validation_method:
            self._s2n_validation(s2n)
        if "median_velocity" in self.validation_method:
            self._median_validation()
        if "mean_velocity" in self.validation_method:
            self._mean_validation()
        if "rms_velocity" in self.validation_method:
            self._rms_validation()

        # Apply the mask to the final result.
        self._mask_val_locations()

        return self.val_locations

    def replace_vectors(self, *f_replace):
        """Replace spurious vectors by the mean or median of the surrounding points.

        f : GPUArray
            2D float (m, n), velocity fields from which to get replacement vectors.

        Returns
        -------
        GPUArray
            2D float (m, n), velocity fields with invalid vectors replaced.

        """
        assert self.val_locations is not None, (
            "Can only replace vectors after validation is performed. See __call__()"
            "method."
        )
        assert len(f_replace) == self._num_fields
        _check_arrays(
            *f_replace,
            array_type=gpuarray.GPUArray,
            shape=self._f[0].shape,
            dtype=self._f[0].dtype,
        )

        f = [
            gpuarray.if_positive(self.val_locations, f_replace[i], self._f[i])
            for i in range(self._num_fields)
        ]

        return f

    def free_gpu_data(self):
        """Free data from GPU."""
        self._clear_validation_data()
        self._neighbours_present = None

    @property
    def median(self):
        """Local median surrounding 8 velocity vectors."""
        f_median = self._median
        if len(f_median) == 1:
            f_median = f_median[0]

        return f_median

    @property
    def mean(self):
        """Local mean of surrounding 8 velocity vectors."""
        f_mean = self._mean
        if len(f_mean) == 1:
            f_mean = f_mean[0]

        return f_mean

    @property
    def num_validation_locations(self):
        """Local mean of surrounding 8 velocity vectors."""
        if self.val_locations is None:
            return None
        if self._n_val is None:
            self._n_val = int(gpuarray.sum(self.val_locations).get())

        return self._n_val

    @property
    def _neighbours(self):
        """Returns neighbouring values for each field."""
        if self._neighbours_ is None:
            self._neighbours_ = [
                _gpu_get_neighbours(f, self._neighbours_present) for f in self._f
            ]

        return self._neighbours_

    @property
    def _median(self):
        """Returns field containing median of surrounding points for each field."""
        if self._median_ is None:
            self._median_ = [
                _gpu_average_velocity(
                    self._neighbours[k], self._neighbours_present, "median_velocity"
                )
                for k in range(self._num_fields)
            ]

        return self._median_

    @property
    def _mean(self):
        """Returns field containing mean of surrounding points for each field."""
        if self._mean_ is None:
            self._mean_ = [
                _gpu_average_velocity(
                    self._neighbours[k], self._neighbours_present, "mean_velocity"
                )
                for k in range(self._num_fields)
            ]

        return self._mean_

    def _check_validation_methods(self):
        """Checks that input validation methods are allowed."""
        if not all(
            [
                val_method in ALLOWED_VALIDATION_METHODS
                for val_method in self.validation_method
            ]
        ):
            raise ValueError(
                "Invalid validation method(s). Allowed validation methods are: "
                "{}".format(ALLOWED_VALIDATION_METHODS)
            )

    def _check_validation_tolerances(self):
        """Checks that input validation methods are allowed."""
        if not all([val_tol > 0 for val_tol in self.validation_tols.values()]):
            raise ValueError(
                "Invalid validation tolerances(s). Validation tolerances must be "
                "greater than 0."
            )

    def _clear_validation_data(self):
        """Clears previous validation data.."""
        self.val_locations = None
        self._f = None
        self._neighbours_ = None
        self._median_ = None
        self._mean_ = None

    def _s2n_validation(self, s2n_ratio):
        """Performs signal-to-noise validation on each field."""
        if s2n_ratio is None:
            return
        _check_arrays(
            s2n_ratio,
            array_type=gpuarray.GPUArray,
            dtype=DTYPE_f,
            size=prod(self.f_shape),
        )
        s2n_tol = log10(self.validation_tols["s2n"])

        sig2noise_tol = s2n_ratio / DTYPE_f(s2n_tol)
        self.val_locations = _local_validation(sig2noise_tol, 1, self.val_locations)

    def _median_validation(self):
        """Performs median validation on each field."""
        median_tol = self.validation_tols["median"]

        for k in range(self._num_fields):
            f_median_residual = _gpu_residual(
                self._median[k],
                self._neighbours_[k],
                self._neighbours_present,
                "median_residual",
            )
            self.val_locations = _neighbour_validation(
                self._f[k],
                self._median[k],
                f_median_residual,
                median_tol,
                self.val_locations,
            )

    def _median_validation_vec2d(self):
        """Performs median validation on each field."""
        median_tol = self.validation_tols["median"]
        u, v = self._f
        u_neighbours, v_neighbours = self._neighbours
        u_median, v_median = self._median

        if self._num_fields == 2:
            residual = _gpu_residual_vec2d(
                u_median,
                v_median,
                u_neighbours,
                v_neighbours,
                self._neighbours_present,
                "median_residual_vec2d",
            )
            self.val_locations = _neighbour_validation_vec2d(
                u,
                v,
                u_median,
                v_median,
                residual,
                median_tol,
                self.val_locations,
            )
        else:
            raise NotImplementedError(
                "Validation not supported for vector fields with dimension > 2."
            )

    def _mean_validation(self):
        """Performs mean validation on each field."""
        mean_tol = self.validation_tols["mean"]

        for k in range(self._num_fields):
            f_mean_residual = _gpu_residual(
                self._mean[k],
                self._neighbours[k],
                self._neighbours_present,
                "mean_residual",
            )
            self.val_locations = _neighbour_validation(
                self._f[k],
                self._mean[k],
                f_mean_residual,
                mean_tol,
                self.val_locations,
            )

    def _mean_validation_vec2d(self):
        """Performs mean validation on each field."""
        mean_tol = self.validation_tols["mean"]
        u, v = self._f
        u_mean, v_mean = self._mean
        u_neighbours, v_neighbours = self._neighbours_

        if self._num_fields == 2:
            residual = _gpu_residual_vec2d(
                u_mean,
                v_mean,
                u_neighbours,
                v_neighbours,
                self._neighbours_present,
                "mean_residual_vec2d",
            )
            self.val_locations = _neighbour_validation_vec2d(
                u, v, u_mean, v_mean, residual, mean_tol, self.val_locations
            )
        else:
            raise NotImplementedError(
                "Validation not supported for vector fields with dimension > 2."
            )

    def _rms_validation(self):
        """Performs RMS validation on each field."""
        f_neighbours = self._neighbours
        f_mean = self._mean
        rms_tol = self.validation_tols["rms"]

        for k in range(self._num_fields):
            f_rms = _gpu_residual(
                f_mean[k], f_neighbours[k], self._neighbours_present, "rms"
            )
            self.val_locations = _neighbour_validation(
                self._f[k], f_mean[k], f_rms, rms_tol, self.val_locations
            )

    def _rms_validation_vec2d(self):
        """Performs RMS validation on each field."""
        rms_tol = self.validation_tols["rms"]
        u, v = self._f
        u_mean, v_mean = self._mean
        u_neighbours, v_neighbours = self._neighbours_

        if self._num_fields == 2:
            rms = _gpu_residual_vec2d(
                u_mean,
                v_mean,
                u_neighbours,
                v_neighbours,
                self._neighbours_present,
                "rms_vec2d",
            )
            self.val_locations = _neighbour_validation_vec2d(
                u, v, u_mean, v_mean, rms, rms_tol, self.val_locations
            )
        else:
            raise NotImplementedError(
                "Validation not supported for vector fields with dimension != 2."
            )

    def _mask_val_locations(self):
        """Removes masked locations from the validation locations."""
        if self.mask is not None and self.val_locations is not None:
            self.val_locations = gpu_mask(self.val_locations, self.mask)


mod_validation = SourceModule(
    """
__global__ void local_validation(int *val_locations, float *f, float tol, int size)
{
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= size) {return;}

    val_locations[t_idx] = val_locations[t_idx] || (f[t_idx] > tol);
}


__global__ void neighbour_validation(
    int *val_locations,
    float *f,
    float *f_mean,
    float *r,
    float tol,
    int size
)
{
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= size) {return;}

    // Add a small number to prevent singularities in uniform flow.
    val_locations[t_idx] = val_locations[t_idx] || (
        fabsf(f[t_idx] - f_mean[t_idx]) / (r[t_idx] + 0.1f) > tol
    );
}


__global__ void neighbour_validation_vec2d(
    int *val_locations,
    float *u,
    float *v,
    float *u_mean,
    float *v_mean,
    float *r,
    float tol,
    int size
)
{
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= size) {return;}

    // Add a small number to prevent singularities in uniform flow.
    val_locations[t_idx] = val_locations[t_idx] || (
        hypotf(u[t_idx] - u_mean[t_idx], v[t_idx] - v_mean[t_idx]) / (r[t_idx] + 0.11)
        > tol
    );
}
"""
)


def _local_validation(f, tol, val_locations=None):
    """Updates the validation list by checking if the array elements exceed the
    tolerance."""
    size = f.size
    _check_arrays(f, array_type=gpuarray.GPUArray, dtype=DTYPE_f)

    if val_locations is None:
        val_locations = gpuarray.zeros_like(f, dtype=DTYPE_i)

    block_size = _BLOCK_SIZE
    grid_size = ceil(size / block_size)
    local_validation = mod_validation.get_function("local_validation")
    local_validation(
        val_locations,
        f,
        DTYPE_f(tol),
        DTYPE_i(size),
        block=(block_size, 1, 1),
        grid=(grid_size, 1),
    )

    return val_locations


def _neighbour_validation(f, f_mean, residual, tol, val_locations=None):
    """Updates the validation list by checking if the neighbouring elements exceed the
    tolerance."""
    size = f.size
    _check_arrays(
        f,
        f_mean,
        residual,
        array_type=gpuarray.GPUArray,
        dtype=DTYPE_f,
        shape=f.shape,
    )

    if val_locations is None:
        val_locations = gpuarray.zeros_like(f, dtype=DTYPE_i)

    block_size = _BLOCK_SIZE
    grid_size = ceil(size / block_size)
    neighbour_validation = mod_validation.get_function("neighbour_validation")
    neighbour_validation(
        val_locations,
        f,
        f_mean,
        residual,
        DTYPE_f(tol),
        DTYPE_i(size),
        block=(block_size, 1, 1),
        grid=(grid_size, 1),
    )

    return val_locations


def _neighbour_validation_vec2d(
    u, v, u_mean, v_mean, residual, tol, val_locations=None
):
    """Updates the validation list by checking if the neighbouring elements exceed the
    tolerance."""
    size = u.size
    _check_arrays(
        u,
        v,
        u_mean,
        v_mean,
        residual,
        array_type=gpuarray.GPUArray,
        dtype=DTYPE_f,
        shape=u.shape,
    )

    if val_locations is None:
        val_locations = gpuarray.zeros_like(u, dtype=DTYPE_i)

    block_size = _BLOCK_SIZE
    grid_size = ceil(size / block_size)
    neighbour_validation = mod_validation.get_function("neighbour_validation_vec2d")
    neighbour_validation(
        val_locations,
        u,
        v,
        u_mean,
        v_mean,
        residual,
        DTYPE_f(tol),
        DTYPE_i(size),
        block=(block_size, 1, 1),
        grid=(grid_size, 1),
    )

    return val_locations


mod_neighbours = SourceModule(
    """
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

    np[t_idx] = in_bound * (!mask[(row_idx * n + col_idx) * in_bound]);
}


__global__ void get_neighbours(float *nb, int *np, float *f, int n, int size)
{
    // nb : values of the neighbouring points
    // np : 1 if there is a neighbour, 0 if no neighbour
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= size) {return;}
    
    int nb_idx = t_idx % 8;
    int row_idx = t_idx / 8 / n - (nb_idx < 3) + (nb_idx > 4); 
    int col_idx = t_idx / 8 % n - ((nb_idx == 0) || (nb_idx == 3) || (nb_idx == 5))
                                + ((nb_idx == 2) || (nb_idx == 4) || (nb_idx == 7));

    // get neighbouring values
    nb[t_idx] = f[(row_idx * n + col_idx) * np[t_idx]] * np[t_idx];
}
"""
)


def _gpu_find_neighbours(shape, mask=None):
    """An array that stores if a point has neighbours in a 3x3 grid surrounding it.

    Parameters
    ----------
    shape : tuple of ints
        (m, n), shape of the array to find neighbours for.
    mask : GPUArray or None
        2D int (m, n), value of one where masked.

    Returns
    -------
    GPUArray
        4D (m, n, 8), value of one where the point in the field has neighbours.

    """
    m, n = shape
    size = m * n * 8

    if mask is None:
        mask = gpuarray.zeros((m, n), dtype=DTYPE_i)
        _check_arrays(mask, array_type=gpuarray.GPUArray, dtype=DTYPE_i, shape=shape)
    neighbours_present = gpuarray.empty((m, n, 8), dtype=DTYPE_i)

    block_size = _BLOCK_SIZE
    grid_size = ceil(size / block_size)
    find_neighbours = mod_neighbours.get_function("find_neighbours")
    find_neighbours(
        neighbours_present,
        mask,
        DTYPE_i(n),
        DTYPE_i(m),
        DTYPE_i(size),
        block=(block_size, 1, 1),
        grid=(grid_size, 1),
    )

    return neighbours_present


def _gpu_get_neighbours(f, neighbours_present):
    """An array that stores the values of the velocity of the surrounding neighbours.

    Parameters
    ----------
    f : GPUArray
        2D float (m, n), values from which to get neighbours.
    neighbours_present : GPUArray
        4D int (m, n, 8), value of one where a neighbour is present.

    Returns
    -------
    GPUArray
        4D float (m, n, 8), values of u and v of the neighbours of a point.

    """
    m, n = f.shape
    size = f.size * 8
    _check_arrays(f, array_type=gpuarray.GPUArray, dtype=DTYPE_f)
    _check_arrays(
        neighbours_present, array_type=gpuarray.GPUArray, dtype=DTYPE_i, shape=(m, n, 8)
    )

    neighbours = gpuarray.empty((m, n, 8), dtype=DTYPE_f)

    block_size = _BLOCK_SIZE
    grid_size = ceil(size / block_size)
    get_neighbours = mod_neighbours.get_function("get_neighbours")
    get_neighbours(
        neighbours,
        neighbours_present,
        f,
        DTYPE_i(n),
        DTYPE_i(size),
        block=(block_size, 1, 1),
        grid=(grid_size, 1),
    )

    return neighbours


mod_average_velocity = SourceModule(
    """
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
    // Implements 8-wire sorting network.
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


__device__ float median(float *A, float *B)
{
    // Sort the arrays.
    sort(A, B);

    // Count the neighbouring points.
    int N = B[0] + B[1] + B[2] + B[3] + B[4] + B[5] + B[6] + B[7];

    // Return the median out of N neighbours.
    if (N % 2 == 0) {return (A[N / 2 - 1] + A[N / 2]) / 2;}
    else {return A[N / 2];}
}


__global__ void median_velocity(float *f_median, float *nb, int *np, int size)
{
    // nb : values of the neighbouring points.
    // np : 1 if there is a neighbour, 0 if no neighbour.
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

    f_median[t_idx] = median(A, B);
}


__global__ void median_residual(
    float *median_residual,
    float *f_median,
    float *nb,
    int *np,
    int size
)
{
    // nb : value of the neighbouring points
    // np : 1 if there is a neighbour, 0 if no neighbour
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

    median_residual[t_idx] = median(A, B);
}


__global__ void median_residual_vec2d(
    float *median_residual,
    float *u_median,
    float *v_median,
    float *u_nb,
    float *v_nb,
    int *np,
    int size
)
{
    // nb : value of the neighbouring points
    // np : 1 if there is a neighbour, 0 if no neighbour
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= size) {return;}

    float u_m = u_median[t_idx];
    float v_m = v_median[t_idx];

    // Loop through neighbours to populate an array to sort.
    int i;
    int j = 0;
    float A[8];
    float B[8];
    for (i = 0; i < 8; i++) {
        A[j] = hypotf(u_nb[t_idx * 8 + i] - u_m, v_nb[t_idx * 8 + i] - v_m);
        B[j++] = np[t_idx * 8 + i];
    }

    median_residual[t_idx] = median(A, B);
}


__device__ float num_neighbours(int *np, int t_idx)
{
    float denominator = np[t_idx * 8 + 0] + np[t_idx * 8 + 1] + np[t_idx * 8 + 2]
                      + np[t_idx * 8 + 3] + np[t_idx * 8 + 4]
                      + np[t_idx * 8 + 5] + np[t_idx * 8 + 6] + np[t_idx * 8 + 7];
    return denominator + (denominator == 0.0f);
}


__global__ void mean_velocity(float *f_mean, float *nb, int *np, int size)
{
    // n : value of neighbours.
    // np : neighbours present.
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= size) {return;}

    // Sum terms of the mean.
    float numerator = nb[t_idx * 8 + 0] + nb[t_idx * 8 + 1] + nb[t_idx * 8 + 2]
                    + nb[t_idx * 8 + 3] + nb[t_idx * 8 + 4]
                    + nb[t_idx * 8 + 5] + nb[t_idx * 8 + 6] + nb[t_idx * 8 + 7];

    // Mean is normalized by number of terms summed.
    float denominator = num_neighbours(np, t_idx);
    f_mean[t_idx] = numerator / denominator;
}


__global__ void mean_residual(
    float *mean_residual,
    float *f_mean,
    float *nb,
    int *np,
    int size
)
{
    // nb : value of the neighbouring points.
    // np : 1 if there is a neighbour, 0 if no neighbour.
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= size) {return;}

    // Sum terms of the mean fluctuations.
    float f_m = f_mean[t_idx];
    float numerator = fabsf(nb[t_idx * 8 + 0] - f_m) + fabsf(nb[t_idx * 8 + 1] - f_m)
                    + fabsf(nb[t_idx * 8 + 2] - f_m) + fabsf(nb[t_idx * 8 + 3] - f_m)
                    + fabsf(nb[t_idx * 8 + 4] - f_m) + fabsf(nb[t_idx * 8 + 5] - f_m)
                    + fabsf(nb[t_idx * 8 + 6] - f_m) + fabsf(nb[t_idx * 8 + 7] - f_m);

    // Mean fluctuation is normalized by number of terms summed.
    float denominator = num_neighbours(np, t_idx);
    mean_residual[t_idx] = numerator / denominator;
}


__global__ void mean_residual_vec2d(
    float *mean_residual,
    float *u_mean,
    float *v_mean,
    float *u_nb,
    float *v_nb,
    int *np,
    int size
)
{
    // nb : value of the neighbouring points.
    // np : 1 if there is a neighbour, 0 if no neighbour.
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= size) {return;}

    // Sum terms of the mean fluctuations.
    float u_m = u_mean[t_idx];
    float v_m = v_mean[t_idx];
    float numerator = hypotf(u_nb[t_idx * 8 + 0] - u_m, v_nb[t_idx * 8 + 0] - v_m)
                    + hypotf(u_nb[t_idx * 8 + 1] - u_m, v_nb[t_idx * 8 + 1] - v_m)
                    + hypotf(u_nb[t_idx * 8 + 2] - u_m, v_nb[t_idx * 8 + 2] - v_m)
                    + hypotf(u_nb[t_idx * 8 + 3] - u_m, v_nb[t_idx * 8 + 3] - v_m)
                    + hypotf(u_nb[t_idx * 8 + 4] - u_m, v_nb[t_idx * 8 + 4] - v_m)
                    + hypotf(u_nb[t_idx * 8 + 5] - u_m, v_nb[t_idx * 8 + 5] - v_m)
                    + hypotf(u_nb[t_idx * 8 + 6] - u_m, v_nb[t_idx * 8 + 6] - v_m)
                    + hypotf(u_nb[t_idx * 8 + 7] - u_m, v_nb[t_idx * 8 + 7] - v_m);

    // Mean fluctuation is normalized by number of terms summed
    // float denominator = sqrtf(2) * num_neighbours(np, t_idx);.
    float denominator = num_neighbours(np, t_idx);
    mean_residual[t_idx] = numerator / denominator;
}


__global__ void rms(float *f_rms, float *f_mean, float *nb, int *np, int size)
{
    // nb : value of the neighbouring points
    // np : 1 if there is a neighbour, 0 if no neighbour
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= size) {return;}

    // Sum terms of the rms fluctuations.
    float f_m = f_mean[t_idx];
    float numerator = powf(nb[t_idx * 8 + 0] - f_m, 2)
                    + powf(nb[t_idx * 8 + 1] - f_m, 2)
                    + powf(nb[t_idx * 8 + 2] - f_m, 2)
                    + powf(nb[t_idx * 8 + 3] - f_m, 2)
                    + powf(nb[t_idx * 8 + 4] - f_m, 2)
                    + powf(nb[t_idx * 8 + 5] - f_m, 2)
                    + powf(nb[t_idx * 8 + 6] - f_m, 2)
                    + powf(nb[t_idx * 8 + 7] - f_m, 2);

    // RMS is normalized by number of terms summed.
    float denominator = num_neighbours(np, t_idx);
    f_rms[t_idx] = sqrtf(numerator / denominator);

}


__global__ void rms_vec2d(
    float *rms,
    float *u_mean,
    float *v_mean,
    float *u_nb,
    float *v_nb,
    int *np,
    int size
)
{
    // nb : value of the neighbouring points
    // np : 1 if there is a neighbour, 0 if no neighbour
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= size) {return;}

    // Sum terms of the rms fluctuations.
    float u_m = u_mean[t_idx];
    float v_m = v_mean[t_idx];
    float numerator = powf(u_nb[t_idx * 8 + 0] - u_m, 2)
                    + powf(u_nb[t_idx * 8 + 1] - u_m, 2)
                    + powf(u_nb[t_idx * 8 + 2] - u_m, 2)
                    + powf(u_nb[t_idx * 8 + 3] - u_m, 2)
                    + powf(u_nb[t_idx * 8 + 4] - u_m, 2)
                    + powf(u_nb[t_idx * 8 + 5] - u_m, 2)
                    + powf(u_nb[t_idx * 8 + 6] - u_m, 2)
                    + powf(u_nb[t_idx * 8 + 7] - u_m, 2)
                    + powf(v_nb[t_idx * 8 + 0] - v_m, 2)
                    + powf(v_nb[t_idx * 8 + 1] - v_m, 2)
                    + powf(v_nb[t_idx * 8 + 2] - v_m, 2)
                    + powf(v_nb[t_idx * 8 + 3] - v_m, 2)
                    + powf(v_nb[t_idx * 8 + 4] - v_m, 2)
                    + powf(v_nb[t_idx * 8 + 5] - v_m, 2)
                    + powf(v_nb[t_idx * 8 + 6] - v_m, 2)
                    + powf(v_nb[t_idx * 8 + 7] - v_m, 2);

    // RMS is normalized by number of terms summed.
    // float denominator = 2.0f * num_neighbours(np, t_idx);
    float denominator = num_neighbours(np, t_idx);
    rms[t_idx] = sqrtf(numerator / denominator);

}
"""
)


def _gpu_average_velocity(f_neighbours, neighbours_present, method):
    """Calculates the median velocity on a 3x3 grid around each point in a velocity
    field.

    Parameters
    ----------
    f_neighbours: GPUArray
        4D float (m, n, 8), neighbouring velocities of every point.
    neighbours_present: GPUArray
        4D int (m, n, 8), value of one where a neighbour is present.
    str {'median_velocity', 'mean_velocity'}
        Which residual to compute.

    Returns
    -------
    GPUArray
        2D float (m, n), median velocities at each point.

    """
    m, n, _ = f_neighbours.shape
    size = m * n
    _check_arrays(
        f_neighbours, array_type=gpuarray.GPUArray, dtype=DTYPE_f, shape=(m, n, 8)
    )
    _check_arrays(
        neighbours_present, array_type=gpuarray.GPUArray, dtype=DTYPE_i, shape=(m, n, 8)
    )

    f_median = gpuarray.empty((m, n), dtype=DTYPE_f)

    block_size = _BLOCK_SIZE
    grid_size = ceil(size / block_size)
    get_average_velocity = mod_average_velocity.get_function(method)
    get_average_velocity(
        f_median,
        f_neighbours,
        neighbours_present,
        DTYPE_i(size),
        block=(block_size, 1, 1),
        grid=(grid_size, 1),
    )

    return f_median


def _gpu_residual(f_median, f_neighbours, neighbours_present, method):
    """Returns the mean/median/rms of normalized residual velocities on a 3x3 grid around
    each point in a velocity field.

    Parameters
    ----------
    f_median : GPUArray
        2D float (m, n), mean velocities around each point.
    f_neighbours : GPUArray
        4D float (m, n, 8), neighbouring velocities of every point.
    neighbours_present : GPUArray
        4D int  (m, n, 8), value of one where a neighbour is present.
    method : str {'mean_residual', 'median_residual', 'rms'}
        Which residual to compute.

    Returns
    -------
    GPUArray
        2D float (m, n), median of normalized residual velocities at each point.

    """
    m, n = f_median.shape
    size = f_median.size
    _check_arrays(f_median, array_type=gpuarray.GPUArray, dtype=DTYPE_f)
    _check_arrays(
        f_neighbours, array_type=gpuarray.GPUArray, dtype=DTYPE_f, shape=(m, n, 8)
    )
    _check_arrays(
        neighbours_present, array_type=gpuarray.GPUArray, dtype=DTYPE_i, shape=(m, n, 8)
    )

    f_median_residual = gpuarray.empty((m, n), dtype=DTYPE_f)

    block_size = _BLOCK_SIZE
    grid_size = ceil(size / block_size)
    get_residual = mod_average_velocity.get_function(method)
    get_residual(
        f_median_residual,
        f_median,
        f_neighbours,
        neighbours_present,
        DTYPE_i(size),
        block=(block_size, 1, 1),
        grid=(grid_size, 1),
    )

    return f_median_residual


def _gpu_residual_vec2d(
    u_median, v_median, u_neighbours, v_neighbours, neighbours_present, method
):
    """Returns the median of normalized residual velocities on a 3x3 grid around each
    point in a velocity field.

    Parameters
    ----------
    u_median, v_median : GPUArray
        2D float (m, n), mean velocity components around each point.
    u_neighbours, v_neighbours : GPUArray
        4D float (m, n, 8), neighbouring velocities of every point.
    neighbours_present : GPUArray
        4D int  (m, n, 8), value of one where a neighbour is present.
    method : str {'mean_residual_vec2d', 'median_residual_vec2d', 'rms_vec2d'}
        Which residual to compute.

    Returns
    -------
    GPUArray
        2D float (m, n), median of residual velocities at each point.

    """
    m, n = u_median.shape
    size = u_median.size
    _check_arrays(
        u_median,
        v_median,
        array_type=gpuarray.GPUArray,
        dtype=DTYPE_f,
        shape=u_median.shape,
    )
    _check_arrays(
        u_neighbours,
        v_neighbours,
        array_type=gpuarray.GPUArray,
        dtype=DTYPE_f,
        shape=(m, n, 8),
    )
    _check_arrays(
        neighbours_present, array_type=gpuarray.GPUArray, dtype=DTYPE_i, shape=(m, n, 8)
    )

    residual = gpuarray.empty((m, n), dtype=DTYPE_f)

    block_size = _BLOCK_SIZE
    grid_size = ceil(size / block_size)
    get_residual = mod_average_velocity.get_function(method)
    get_residual(
        residual,
        u_median,
        v_median,
        u_neighbours,
        v_neighbours,
        neighbours_present,
        DTYPE_i(size),
        block=(block_size, 1, 1),
        grid=(grid_size, 1),
    )

    return residual
