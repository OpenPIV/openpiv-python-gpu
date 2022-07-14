import logging
import warnings
from math import sqrt, log2, log10, ceil

import numpy as np
from scipy.optimize.lbfgsb import fmin_l_bfgs_b
from scipy.fft import dct, idct
from scipy.ndimage.morphology import distance_transform_edt
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
from pycuda.compiler import SourceModule

from openpiv.gpu_misc import _check_arrays

# scikit-cuda gives an annoying warning everytime it's imported.
with warnings.catch_warnings():
    warnings.simplefilter('ignore', UserWarning)
    import skcuda.fft as cufft
    from skcuda import misc as cumisc
cumisc.init()

DTYPE_i = np.int32
DTYPE_f = np.float32
DTYPE_c = np.complex64

MAX_ROBUST_STEPS = 3
ERR_P = 0.1
N_P0 = 10
COARSE_COEFFICIENTS = 10
WEIGHT_METHODS = {'bisquare', 'talworth', 'cauchy'}


def gpu_smoothn(*f_dl, **kwargs):
    """Smooths a scalar field stored as a GPUArray.

    Parameters
    ----------
    f_dl : GPUArray
        nD float, field to be smoothed.

    Returns
    -------
    GPUArray
        Float, same size as f_d. Smoothed field.

    """
    _check_arrays(*f_dl, array_type=gpuarray.GPUArray, dtype=DTYPE_f)
    n = len(f_dl)

    # Get data from GPUArrays.
    f = [f_d.get() for f_d in f_dl]
    for key, value in kwargs.items():
        if isinstance(value, gpuarray.GPUArray):
            kwargs[key] = value.get()

    f_smooth = smoothn(*f, **kwargs)[0]
    if n == 1:
        f_smooth_dl = gpuarray.to_gpu(f_smooth)
    else:
        f_smooth_dl = [gpuarray.to_gpu(array) for array in f_smooth]

    return f_smooth_dl


def smoothn(*y, mask=None, w=None, s=None, robust=False, z0=None, max_iter=100, tol_z=1e-3, weight_method='bisquare',
            smooth_order=2, spacing=None):
    """Robust spline smoothing for 1-D to n-D data.

    smoothn provides a fast, automatized and robust discretized smoothing spline for data of any dimension. z =
    smoothn(y) automatically smooths the uniformly-sampled array y. y can be any n-D noisy array (time series, images,
    3D data,...). Non-finite data (NaN or Inf) are treated as missing values. z, s = smoothn(...) also returns the
    calculated value for S so that you can fine-tune the smoothing subsequently if needed. An iteration process is used
    in the presence of weighted and/or missing values. z = smoothn(...,option) smooths with the termination parameters
    specified by option.

    Parameters
    ----------
    y : ndarray
        Input array can be numeric or logical. The returned array is of type double.
    mask : ndarray, optional
        Locations where the data should be masked.
    w : ndarray, optional
        Specifies a weighting array w of real positive values, that must have the same size as y. Note that a nil weight
        corresponds to a missing value.
    s : float, optional
        Smooths the array y using the smoothing parameter s. s must be a real positive scalar. The larger s is, the
        smoother the output will be. If the smoothing parameter s is omitted (see previous option) or empty (i.e. s =
        None), it is automatically determined using the generalized cross-validation (GCV) method.
    robust : bool, optional
        Carries out a robust smoothing that minimizes the influence of outlying data.
    max_iter : int, optional
        Maximum number of iterations allowed (default = 100).
    tol_z : float, optional
        Termination tolerance on Z (default = 1e-3). TolZ must be in [0,1].
    z0 : ndarray or list, optional
        Initial value for the iterative process (default = original data)
    weight_method : {'bisquare', 'talworth', 'cauchy'}, optional
        Weight function for robust smoothing.
    smooth_order : {0, 1, 2}, optional
        Order criterion.
    spacing : ndarray or iterable, optional
        Spacing between points in each dimension.

    Returns
    -------
    z : ndarray or list
    s : float
    w_tot : ndarray

    Example
    -------
    z = smoothn(y)

    References
    ----------
    Garcia D. Robust smoothing of gridded data in one and higher dimensions with missing values. Computational
    Statistics & Data Analysis. 2010.
    https://www.biomecardio.com/pageshtm/publi/csda10.pdf
    https://www.biomecardio.com/matlab/smoothn.html
    https://www.biomecardio.com/matlab/dctn.html
    https://www.biomecardio.com/matlab/idctn.html

    """
    n_y = len(y)
    y0 = y[0]
    y_shape = y0.shape
    y_size = y0.size
    y_ndim = y0.ndim
    d_type = y0.dtype
    if d_type != np.float32:
        d_type = np.float64
    if y_size < 2:
        z = y
        w_tot = np.array(0)
        return z, s, w_tot
    if not all([y[i].shape == y_shape for i in range(n_y)]):
        raise ValueError('All arrays in y must have same shape.')
    if not bool(robust) == robust * 1:
        raise ValueError('bool must be a boolean.')
    if not 0 < max_iter == int(max_iter):
        raise ValueError('max_iter must be an integer.')
    if not 0 < tol_z == float(tol_z):
        raise ValueError('max_iter must be a number greater than zero.')
    weight_method = weight_method.lower()
    if weight_method not in WEIGHT_METHODS:
        raise ValueError('max_iter must be a number.')
    if int(smooth_order) != smooth_order not in {0, 1, 2}:
        raise ValueError('smooth_order must be 0, 1 or 2.')

    # Get mask.
    is_masked_array = False
    if isinstance(y0, np.ma.masked_array):
        is_masked_array = True
        mask = y0.mask if mask is not None else mask
        y = [y[i].data for i in range(n_y)]

    # Arbitrary values for missing y-data.
    is_finite = np.isfinite(y[0])
    for i in range(1, n_y):
        is_finite = is_finite * np.isfinite(y[i])
    num_finite = is_finite.sum()
    for i in range(n_y):
        replace_non_finite(y[i], finite=is_finite, spacing=spacing)

    # Weights. Zero weights are assigned to not finite values (Inf/NaN values = missing data).
    if w is not None:
        w *= is_finite
        if w.shape != y_shape:
            raise ValueError('w must be an ndarray with same shape as arrays in y.')
        if np.any(w < 0):
            raise ValueError('Weights must all be greater than or equal to zero.')
        w = DTYPE_f(w)
        w = (w / np.amax(w))
    else:
        w = np.ones(y_shape, dtype=d_type) * is_finite

    # Apply mask to weights
    if mask is not None:
        if np.any(np.round(mask) != mask):
            'mask must have boolean values.'
        if mask.shape != y_shape:
            raise ValueError('mask must be an ndarray with same shape as arrays in y.')
        w *= (1 - mask)

    # Initial conditions for z.
    is_weighted = np.any(w != 1)
    if is_weighted:
        # With weighted/missing data, an initial guess is provided to ensure faster convergence. For that purpose, a
        # nearest neighbor interpolation followed by a coarse smoothing are performed.
        if z0 is not None:  # an initial guess (z0) has been provided
            if not isinstance(z0, list):
                z0 = [z0]
            if not len(z0) == n_y:
                raise ValueError('z0 must have same number of components as y.')
            if not all([z0[i].shape == y_shape for i in range(n_y)]):
                raise ValueError('Arrays in z0 must all have same shape as arrays in y.')
            z = z0
        else:
            z = _initial_guess(y)
    else:
        z = [np.zeros(y_shape, dtype=d_type)] * n_y

    # Get spacing.
    if spacing is not None:
        spacing = np.array(spacing)
        if spacing.size != y_ndim:
            raise ValueError('spacing must be either None or an array-like with size == y.ndim')
        if np.any(spacing < 0):
            raise ValueError('spacing must all be greater than zero.')
        spacing = np.array(spacing) / np.amax(spacing)
    else:
        spacing = np.ones(y_ndim)

    # With s given as an argument, it will not be found by GCV-optimization.
    is_auto = not s
    p = p_max = p_min = None
    if not is_auto:
        if not 0 < s == float(s):
            raise ValueError('s must be a number greater than zero or None.')
        p = log10(s)
    else:
        p_min, p_max = _p_bounds(y_shape, smooth_order)

    # Relaxation factor to speedup convergence.
    relaxation_factor = 1 + 0.75 * is_weighted

    # Create the eigenvalues.
    lambda_ = _lambda(y0, spacing) ** smooth_order

    # Main iterative process.
    iter_n = 0
    w_tot = w
    for robust_step in range(MAX_ROBUST_STEPS):
        w_mean = np.mean(w)

        for iter_n in range(max_iter):
            z0 = z
            y_dct = [_dct_nd(w_tot * (y[i] - z[i]) + z[i], f=dct) for i in range(n_y)]

            # The generalized cross-validation (GCV) method is used. We seek the smoothing parameter s that
            # minimizes the GCV score i.e. s = Argmin(GCV_score). Because this process is time-consuming, it is
            # performed from time to time (when iter_n is a power of 2).
            if is_auto and not np.remainder(log2(iter_n + 1), 1):
                # If no initial guess for s, span the possible range to get a reasonable starting point. Only need to do
                # it once though. N_S0 is the number of samples used.
                if not p:
                    p_span = np.arange(N_P0) * (1 / (N_P0 - 1)) * (p_max - p_min) + p_min
                    g = np.zeros_like(p_span)
                    for i, p_i in enumerate(p_span):
                        g[i] = _gcv(p_i, y, y_dct, w_tot, lambda_, is_finite, w_mean, num_finite)

                    p = p_span[np.argmin(g)]

                # Estimate the smoothing parameter.
                p, _, _ = fmin_l_bfgs_b(_gcv, np.array([p]), fprime=None, factr=10, approx_grad=True,
                                        bounds=[(p_min, p_max)],
                                        args=(y, y_dct, w_tot, lambda_, is_finite, w_mean, num_finite))
                p = p[0]

            # Update z using the gamma coefficients.
            s = 10 ** p
            gamma = 1 / (1 + s * lambda_)
            z = [relaxation_factor * _dct_nd(gamma * y_dct[i], f=idct) + (1 - relaxation_factor) * z[i]
                 for i in range(n_y)]

            # If not weighted/missing data => tol=0 (no iteration).
            tol = is_weighted * np.linalg.norm([z0[i] - z[i] for i in range(n_y)]) / np.linalg.norm(z)

            if tol <= tol_z:
                break

        # Robust Smoothing: iteratively re-weighted process.
        if robust:
            h = _leverage(s, spacing, smooth_order)
            w_tot = w * _robust_weights(y, z, is_finite, h, weight_method)
            is_weighted = True
        else:
            break

    # Log warning messages.
    if is_auto:
        _s_bound_warning(s, p_min, p_max)
    if iter_n == max_iter - 1:
        _max_iter_warning(max_iter)

    # Re-mask the array.
    if is_masked_array:
        z = [np.ma.masked_array(array, mask=mask) for array in z]
    elif mask is not None:
        for i in range(n_y):
            z[i] *= (1 - mask)

    if d_type == np.float32:
        assert z[0].dtype == np.float32

    if n_y == 1:
        z = z[0]

    return z, s


def gpu_fft(f_d, norm='backward', full_frequency=False):
    """Returns the 1D FFT of the input.

    Parameters
    ----------
    f_d : GPUArray
        1D data to be transformed, with serial data along rows.
    norm : {'forward', 'backward', 'ortho'}
        Normalization of the forward-backward transform pair.
    full_frequency : bool
        Whether to return the full or non-redundant Fourier coefficients.

    Returns
    -------
    GPUArray

    """
    assert f_d.dtype == DTYPE_f
    if f_d.ndim == 1:
        f_d = f_d.reshape(1, f_d.size)
    m, n = f_d.shape
    assert n >= 2
    scale = norm == 'forward'
    fft_d = gpuarray.empty((m, n // 2 + 1), dtype=DTYPE_c)

    plan_forward = cufft.Plan((n,), DTYPE_f, DTYPE_c, batch=m)
    cufft.fft(f_d, fft_d, plan_forward, scale=scale)

    if norm == 'ortho':
        fft_d = fft_d * DTYPE_f(1 / sqrt(n))

    if full_frequency:
        fft_d = _reflect_frequency_comp(fft_d, n)

    return fft_d


def gpu_ifft(f_d, norm='backward', inverse_width=None, full_frequency=False):
    """Returns the 1D, inverse FFT of the input.

    Parameters
    ----------
    f_d : GPUArray
        1D data to be transformed, with serial data along rows.
    norm : {'forward', 'backward', 'ortho'}
        Normalization of the forward-backward transform pair.
    inverse_width : int
        Size of the inverse transform. This ensures that the forward-backward pair is the same size.
    full_frequency : bool
        Whether the transform data includes redundant frequencies.

    Returns
    -------
    GPUArray

    """
    assert f_d.dtype == DTYPE_c
    if f_d.ndim == 1:
        f_d = f_d.reshape(1, f_d.size)
    m, n = f_d.shape
    assert n >= 2
    scale = norm == 'backward'
    if inverse_width is None:
        inverse_width = (n - 1) * 2
    if full_frequency:
        frequency_width = inverse_width // 2 + 1
        f_d = f_d[:, :frequency_width].copy()

    ifft_d = gpuarray.empty((m, inverse_width), dtype=DTYPE_f)

    plan_inverse = cufft.Plan((inverse_width,), DTYPE_c, DTYPE_f, batch=m)
    cufft.ifft(f_d, ifft_d, plan_inverse, scale=scale)

    if norm == 'ortho':
        ifft_d = ifft_d * DTYPE_f(1 / sqrt(inverse_width))

    return ifft_d


def gpu_dct(f_d, norm='backward'):
    assert f_d.dtype == DTYPE_f
    if f_d.ndim == 1:
        f_d = f_d.reshape(1, f_d.size)
    m, n = f_d.shape
    assert n >= 2

    normal_factor = DTYPE_f(1 / n) if norm == 'forward' else DTYPE_f(2)

    freq_width = n // 2 + 1

    # W-coefficients from Makhoul.
    w_real_d = gpuarray.to_gpu(np.cos(DTYPE_f(-np.pi) * np.arange(freq_width, dtype=DTYPE_f) / DTYPE_f(2 * n))
                               * normal_factor)
    w_imag_d = gpuarray.to_gpu(np.sin(DTYPE_f(np.pi) * np.arange(freq_width, dtype=DTYPE_f) / DTYPE_f(2 * n))
                               * normal_factor)

    # Extend the fft output rather than zero-pad (Mahkoul).
    data_d = _sift(f_d, 'forward')
    fft_data_d = gpu_fft(data_d, norm='backward', full_frequency=False)
    fft_data_real_d = fft_data_d.real
    fft_data_imag_d = fft_data_d.imag

    # This could be done by a kernel to save two array definitions.
    dct_positive_d = (cumisc.multiply(fft_data_real_d, w_real_d) + cumisc.multiply(fft_data_imag_d, w_imag_d))
    dct_negative_d2 = (cumisc.multiply(fft_data_real_d, w_imag_d) - cumisc.multiply(fft_data_imag_d, w_real_d))
    dct_d = gpuarray.empty((m, n), dtype=DTYPE_f)
    dct_d[:, :freq_width] = dct_positive_d
    dct_d[:, freq_width:] = _flip_frequency_real(dct_negative_d2, n - freq_width, offset=(n % 2) - 1)

    # Use this after the next release of PyCUDA (2022).
    # dct_d = gpuarray.concatenate(dct_positive_d, dct_negative_d2, axis=0)

    if norm == 'ortho':
        a = np.empty((n,), dtype=DTYPE_f)
        a.fill(DTYPE_f(1 / sqrt(2 * n)))
        a[0] = 1 / sqrt(4 * n)
        a_d = gpuarray.to_gpu(a)

        dct_d = cumisc.multiply(dct_d, a_d)

    return dct_d


def gpu_idct(f_d, norm='backward'):
    """Returns the 1D, type-II, inverse DCT of the input.

    Parameters
    ----------
    f_d : GPUArray
        1D data to be transformed, with serial data along rows.
    norm : {'forward', 'backward', 'ortho'}
        Normalization of the forward-backward transform pair.

    Returns
    -------
    GPUArray

    """
    assert f_d.dtype == DTYPE_f
    if f_d.ndim == 1:
        f_d = f_d.reshape(1, f_d.size)
    m, n = f_d.shape
    assert n >= 2
    scale = norm == 'backward'
    freq_width = n // 2 + 1
    normal_factor = DTYPE_f(1) if norm == 'forward' else DTYPE_f(0.5)

    ifft_output_d = gpuarray.empty((m, n), dtype=DTYPE_f)
    w_d = gpuarray.to_gpu(np.exp(DTYPE_c(1j * np.pi) * np.arange(freq_width, dtype=DTYPE_f) / DTYPE_f(2 * n))
                          * normal_factor)

    fft_data_flip = _flip_frequency_real(f_d, freq_width, left_pad=1)
    ifft_input_d = cumisc.multiply(f_d[:, :freq_width].copy() - DTYPE_c(1j) * fft_data_flip, w_d)

    plan_inverse = cufft.Plan((n,), DTYPE_c, DTYPE_f, batch=m)
    cufft.ifft(ifft_input_d, ifft_output_d, plan_inverse, scale=scale)
    idct_d = _sift(ifft_output_d, 'backward')

    if norm == 'ortho':
        x_0 = f_d[:, 0].copy().reshape(m, 1)
        idct_d = cumisc.add(DTYPE_f(sqrt(2 / n)) * idct_d, (DTYPE_f((sqrt(2) - 1) / sqrt(2 * n))) * x_0)

    return idct_d


def replace_non_finite(y, finite=None, spacing=None):
    """Returns array with non-finite values replaced using nearest-neighbour interpolation.

    Parameters
    ----------
    y : ndarray
        1D data to be transformed, with serial data along rows.
    finite : ndarray, optional
        nD bool, locations in y to replace values.
    spacing: float or iterable, optional
        Spacing between grid points along each axis.

    """
    assert finite.dtype == bool
    y_ndim = y.ndim

    if finite is None:
        finite = np.isfinite(y)
    missing = ~finite

    if np.any(missing):
        nearest_neighbour = distance_transform_edt(missing, sampling=spacing, return_distances=False,
                                                   return_indices=True)
        if y_ndim == 1:
            neighbour_index = nearest_neighbour.squeeze()
        else:
            neighbour_index = tuple(nearest_neighbour[i] for i in range(nearest_neighbour.shape[0]))
        y[missing] = y[neighbour_index][missing]


def _initial_guess(y):
    """Returns initial guess for z using coarse, fast smoothing."""
    assert isinstance(y, list) or isinstance(y, tuple)
    n_y = len(y)

    # Forward transform.
    z_dct = [_dct_nd(y[i], f=dct) for i in range(n_y)]
    num_dct = np.ceil(np.array(z_dct[0].shape) / COARSE_COEFFICIENTS).astype(int) + 1

    # Keep one-tenth of data.
    coefficient_indexes = tuple([slice(n, None) for n in num_dct])
    for i in range(n_y):
        z_dct[i][coefficient_indexes] = 0

    # Backwards transform.
    z = [_dct_nd(z_dct[i], f=idct) for i in range(n_y)]

    return z


def _p_bounds(y_shape, smooth_order=2):
    """Returns upper and lower bound for the smoothness parameter."""
    h_min = 1e-3
    h_max = 1 - h_min

    # Tensor rank of the y-array.
    rank_y = np.sum(np.array(y_shape) != 1)
    # rank_y = np.array(y_shape).squeeze().ndim

    if smooth_order == 0:  # Not recommended--only for numerical purposes.
        p_min = log10(1 / h_max ** (1 / rank_y) - 1)
        p_max = log10(1 / h_min ** (1 / rank_y) - 1)
    elif smooth_order == 1:
        p_min = log10((1 / (h_max ** (2 / rank_y)) - 1) / 4)
        p_max = log10((1 / (h_min ** (2 / rank_y)) - 1) / 4)
    else:
        p_min = log10((((1 + sqrt(1 + 8 * h_max ** (2 / rank_y))) / 4 / h_max ** (2 / rank_y)) ** 2 - 1) / 16)
        p_max = log10((((1 + sqrt(1 + 8 * h_min ** (2 / rank_y))) / 4 / h_min ** (2 / rank_y)) ** 2 - 1) / 16)

    return p_min, p_max


def _lambda(y, spacing):
    """Returns the lambda tensor. lambda_ contains the eigenvalues of the difference matrix used in this
    penalized-least-squares process."""
    y_shape = y.shape
    y_ndim = y.ndim
    d_type = y.dtype

    lambda_ = np.zeros(y_shape, dtype=d_type)
    for k in range(y_ndim):
        shape_k = np.ones(y_ndim, dtype=int)
        shape_k[k] = y_shape[k]
        lambda_ += (np.cos(np.pi * (np.arange(1, y_shape[k] + 1) - 1) / y_shape[k]) / spacing[k] ** 2).reshape(shape_k)
    lambda_ = 2 * (y_ndim - lambda_)

    return lambda_


def _dct_nd(data, f=dct):
    """Returns the nD dct.

    Output must be C-contiguous to transfer back to GPU properly.

    """
    ndim = data.ndim

    if ndim == 1:
        data_dct = f(data, norm='ortho', type=2)
    elif ndim == 2:
        data_dct = np.ascontiguousarray(f(f(data, norm='ortho', type=2).T, norm='ortho', type=2).T)
    else:
        data_dct = data.copy()
        for dim in range(ndim):
            data_dct = f(data_dct, norm='ortho', type=2, axis=dim)

    return data_dct


def _gcv(p, y, y_dct, w, lambda_, is_finite, w_mean, nof):
    """Returns the GCV score for given p-value and y-data."""
    assert isinstance(y, list)
    n_y = len(y)
    y_size = y[0].size
    s = 10 ** p
    gamma = 1 / (1 + s * lambda_)

    # w_mean == 1 means that all the data are equally weighted.
    residual_sum_squares = 0
    if w_mean > 0.9:
        # Very much faster: does not require any inverse DCT.
        for i in range(n_y):
            residual_sum_squares += np.linalg.norm(y_dct[i] * (gamma - 1)) ** 2
    else:
        # Take account of the weights to calculate residual_sum_squares.
        for i in range(n_y):
            y_hat = _dct_nd(gamma * y_dct[i], f=idct)
            residual_sum_squares += np.linalg.norm(np.sqrt(w[is_finite]) * (y[i][is_finite] - y_hat[is_finite])) ** 2

    tr_h = np.sum(gamma)
    # Divide by n_y to match score of scalar field.
    gcv_score = residual_sum_squares / nof / (1 - tr_h / y_size) ** 2 / n_y

    return float(gcv_score)


def _leverage(s, spacing, smooth_order=2):
    """Returns average leverage.

    The average leverage (h) is by definition in [0 1]. Weak smoothing occurs if h is close to 1, while over-smoothing
    occurs when h is near 0. Upper and lower bounds for h are given to avoid under- or over-smoothing. See equation
    relating h to the smoothness parameter (Equation #12 in the referenced CSDA paper).

    """
    assert s > 0
    ndim = len(spacing)
    h = 1

    if smooth_order == 0:  # Not recommended--only for numerical purposes.
        for i in range(ndim):
            h *= 1 / (1 + s / spacing[i])
    elif smooth_order == 1:
        for i in range(ndim):
            h *= 1 / sqrt(1 + 4 * s / spacing[i] ** 2)
    else:
        for i in range(ndim):
            h0 = sqrt(1 + 16 * s / spacing[i] ** 4)
            h *= sqrt(1 + h0) / sqrt(2) / h0

    return float(h)


def _robust_weights(y, z, is_finite, h, weight_str='bisquare'):
    """Returns the weights for robust smoothing."""
    assert weight_str in WEIGHT_METHODS

    r = np.stack(y, axis=0) - np.stack(z, axis=0)
    marginal_median = np.median(r[:, is_finite])
    vector_norm = _rms(r, axis=0)
    median_absolute_deviation = np.median(_rms(r[:, is_finite] - marginal_median))

    # Compute studentized residuals.
    u = vector_norm / (1.4826 * median_absolute_deviation) / np.sqrt(1 - h)

    if weight_str == 'cauchy':
        c = 2.385
        w = 1 / (1 + (u / c) ** 2)
    elif weight_str == 'talworth':
        c = 2.795
        w = u < c
    else:  # Bisquare weights.
        c = 4.685
        w = (1 - (u / c) ** 2) ** 2 * ((u / c) < 1)

    return w


def _norm(x, axis=0):
    return np.sqrt(np.sum(x ** 2, axis=axis))


def _rms(x, axis=0):
    return np.sqrt(np.sum(x ** 2, axis=axis) / x.shape[axis])


def _s_bound_warning(s, p_min, p_max):
    if np.abs(log10(s) - p_min) < ERR_P:
        logging.info('The lower bound for s ({:.3f}) has been reached. '
                     'Put s as an input variable if required.'.format(10 ** p_min))
    elif np.abs(log10(s) - p_max) < ERR_P:
        logging.info('The upper bound for s ({:.3f}) has been reached. '
                     'Put s as an input variable if required.'.format(10 ** p_max))


def _max_iter_warning(max_iter):
    logging.info('The maximum number of iterations ({:d}) has been exceeded. '
                 'Increase max_iter option or decrease tol_z value.'.format(max_iter))


mod_sift = SourceModule("""
    __global__ void forward_sift(float *dest, float *src, int wd, int size)
{
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= size) {return;}
    int row = t_idx / wd;
    int col = t_idx % wd;

    dest[row * wd + col] = src[(row * wd + (col <= (wd - 1) / 2) * (2 * col)
                                         + (col >= (wd + 1) / 2) * (2 * wd - 2 * col - 1))];
}

    __global__ void backward_sift(float *dest, float *src, int wd, int size)
{
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= size) {return;}
    int row = t_idx / wd;
    int col = t_idx % wd;

    dest[row * wd + col] = src[row * wd + (col % 2 == 0) * (col / 2) + (col % 2 == 1) * (wd - 1 - col / 2)];
}
""")


def _sift(f_d, direction):
    """Returns sifted frequencies from inverse transform data (Makhoul)."""
    m, n = f_d.shape
    size_i = DTYPE_i(f_d.size)

    f_sifted = gpuarray.empty((m, n), dtype=DTYPE_f)

    block_size = 32
    x_blocks = ceil(size_i / block_size)
    if direction == 'forward':
        sift = mod_sift.get_function('forward_sift')
    else:
        sift = mod_sift.get_function('backward_sift')
    sift(f_sifted, f_d, DTYPE_i(n), size_i, block=(block_size, 1, 1), grid=(x_blocks, 1))

    return f_sifted


mod_flip = SourceModule("""
__global__ void flip_frequency(float *dest, float *src, int offset, int left_pad, int f_width, int wd, int size)
{
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= size) {return;}
    int row = t_idx / (f_width - left_pad);
    int col = t_idx % (f_width - left_pad);

    dest[row * f_width + col + left_pad] = src[row * wd + wd - col - 1 + offset];
}
""")


def _flip_frequency_real(f_d, flip_width, offset=0, left_pad=0):
    """Returns the frequency spectrum with indexing flipped along rows."""
    m, n = f_d.shape
    assert flip_width == int(flip_width) and left_pad == int(left_pad)
    size_i = DTYPE_i(m * (flip_width - left_pad))

    f_flip_d = gpuarray.zeros((m, flip_width), dtype=DTYPE_f)

    block_size = 32
    x_blocks = ceil(size_i / block_size)
    flip_frequency = mod_flip.get_function('flip_frequency')
    flip_frequency(f_flip_d, f_d, DTYPE_i(offset), DTYPE_i(left_pad), DTYPE_i(flip_width), DTYPE_i(n), size_i,
                   block=(block_size, 1, 1), grid=(x_blocks, 1))

    return f_flip_d


def _flip_frequency_comp(f_d, flip_width, offset=0, left_pad=0):
    """Returns the frequency spectrum with indexing flipped along rows."""
    f_flip_real_d = _flip_frequency_real(f_d.real, flip_width, offset, left_pad)
    f_flip_imag_d = _flip_frequency_real(f_d.imag, flip_width, offset, left_pad)

    f_flip_d = f_flip_real_d - DTYPE_c(1j) * f_flip_imag_d

    return f_flip_d


def _reflect_frequency_comp(f_d, full_width):
    """Returns the full series of transform coefficients from a non-redundant series."""
    m, n = f_d.shape
    assert full_width > n
    flip_width = full_width - n
    offset = (full_width % 2) - 1

    f_reflect_d = gpuarray.empty((m, full_width), dtype=DTYPE_c)
    f_reflect_d[:, :n] = f_d
    f_reflect_d[:, n:] = _flip_frequency_comp(f_d, flip_width, offset=offset)

    # Use this for next release of PyCUDA (2022), after testing.
    # f_flip_d = _flip_frequency_complex(f_d, flip_width)
    # f_reflect_d = gpuarray.concatenate(f_d, f_flip_d, axis=0)

    return f_reflect_d
