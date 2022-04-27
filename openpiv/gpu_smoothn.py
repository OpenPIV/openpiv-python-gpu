"""This module will contain a GPU-accelerated implementation of smoothn."""

import logging
import warnings
from math import sqrt, pi, ceil

import numpy as np
import numpy.linalg as linalg
import scipy.optimize.lbfgsb as lbfgsb
from scipy.fftpack.realtransforms import dct, idct
import skcuda.fft as cufft
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
from pycuda.compiler import SourceModule

# scikit-cuda gives an annoying warning everytime it's imported.
with warnings.catch_warnings():
    warnings.simplefilter('ignore', UserWarning)
    from skcuda import misc as cumisc

from openpiv.gpu_misc import _check_arrays

cumisc.init()
DTYPE_i = np.int32
DTYPE_f = np.float32
DTYPE_c = np.complex64


def gpu_smoothn(f_d, **kwargs):
    """Smooths a scalar field stored as a GPUArray.

    Parameters
    ----------
    f_d : GPUArray
        Field to be smoothed.

    Returns
    -------
    GPUArray
        Float, same size as f_d. Smoothed field.

    """
    _check_arrays(f_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=2)

    f = f_d.get()
    f_smooth_d = gpuarray.to_gpu(smoothn(f, **kwargs)[0].astype(DTYPE_f, order='C'))  # Smoothn returns F-ordered array.

    return f_smooth_d


# TODO implement smooth order (m)
def smoothn(
        y,
        w=None,
        s=None,
        robust=False,
        max_iter=100,
        tol_z=1e-3,
        z0=None,
        weight_str="bisquare",
        spacing=None,
        mask=None,
        axis=None,
        s0=None,
        n_s0=10,
        smooth_order=2,
        **kwargs
):
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
    w : optional
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
    z0 : ndarray or None, optional
        Initial value for the iterative process (default = original data)
    weight_str : {'bisquare', 'talworth', 'cauchy'}, optional
        Weight function for robust smoothing.
    smooth_order : float, optional
    spacing : ndarray, optional
    mask : ndarray, optional
        Locations where the data should be masked.
    axis : optional
        Axes along with to smooth.
    s0 : float, optional
    n_s0 : optional


    Returns
    -------
    z : ndarray
    s : float
    w_tot : ndarray

    Example
    -------
    z = smoothn(y)

    References
    ----------
    Garcia D, Robust smoothing of gridded data in one and higher dimensions with missing values. Computational
    Statistics & Data Analysis, 2010.
    https://www.biomecardio.com/pageshtm/publi/csda10.pdf
    https://www.biomecardio.com/matlab/smoothn.html
    https://www.biomecardio.com/matlab/dctn.html
    https://www.biomecardio.com/matlab/idctn.html

    """
    assert smooth_order == 2, 'smooth_order values other than 2 are not yet implemented.'
    is_auto = not s
    verbose = kwargs['verbose'] if 'verbose' in kwargs.items() else False

    if isinstance(y, np.ma.masked_array):
        mask = y.mask
        y = np.array(y)

    if mask is not None:
        if w is not None:
            w = np.array(w)
            w[mask] = 0.0
        y[mask] = np.nan

    # Smoothness parameter and weights
    if w is not None:
        if np.any(w < 0):
            raise ValueError('Weights must all be greater than or equal to 0.')
        w = w / np.amax(w)
    else:
        w = np.ones(y.shape)

    # sort axis
    if axis is None:
        axis = tuple(np.arange(y.ndim))

    if y.size < 2:
        z = y
        w_tot = 0
        return z, s, w_tot

    # "Weighting function" criterion
    weight_str = weight_str.lower()

    # Weights. Zero weights are assigned to not finite values (Inf or NaN),
    # (Inf/NaN values = missing data).
    is_finite = np.array(np.isfinite(y)).astype(bool)
    nof = np.sum(is_finite)  # number of finite elements
    w = w * is_finite

    # Weighted or missing data?
    is_weighted = np.any(w != 1)

    if spacing is not None:
        spacing = spacing / np.amax(spacing)
    else:
        spacing = np.ones(len(axis))

    # Creation of the lambda_ tensor.
    # lambda_ contains the eingenvalues of the difference matrix used in this
    # penalized least squares process.
    axis = tuple(np.array(axis).flatten())
    lambda_ = np.zeros(y.shape)
    for i in axis:
        # create a 1 x d array (so e.g. [1,1] for a 2D case
        siz0 = np.ones(y.ndim, dtype=int)
        siz0[i] = y.shape[i]
        # cos(pi*(reshape(1:y_shape(i),siz0)-1)/y_shape(i)))
        # (arange(1,y_shape[i]+1).reshape(siz0) - 1.)/y_shape[i]
        lambda_ = lambda_ + (np.cos(np.pi * (np.arange(1, y.shape[i] + 1) - 1.0) / y.shape[i]
                                    ).reshape(siz0)) / spacing[i] ** 2
    lambda_ = -2.0 * (len(axis) - lambda_)
    # if not is_auto:
    #     gamma = 1.0 / (1 + (s * abs(lambda_)) ** smooth_order)

    # Upper and lower bound for the smoothness parameter
    # The average leverage (h) is by definition in [0 1]. Weak smoothing occurs
    # if h is close to 1, while over-smoothing appears when h is near 0. Upper
    # and lower bounds for h are given to avoid under- or over-smoothing. See
    # equation relating h to the smoothness parameter (Equation #12 in the
    # referenced CSDA paper).
    n = sum(y.shape) != 1  # tensor rank of the y-array
    h_min = 1e-6
    h_max = 0.99
    # (h/n)**2 = (1 + a)/( 2 a)
    # a = 1/(2 (h/n)**2 -1)
    # where a = sqrt(1 + 16 s)
    # (a**2 -1)/16
    s_min_bnd = np.sqrt((((1 + np.sqrt(1 + 8 * h_max ** (2.0 / n))) / 4.0 / h_max ** (2.0 / n)) ** 2 - 1) / 16.0)
    s_max_bnd = np.sqrt((((1 + np.sqrt(1 + 8 * h_min ** (2.0 / n))) / 4.0 / h_min ** (2.0 / n)) ** 2 - 1) / 16.0)

    if is_auto:
        xpost = np.array([(0.9 * np.log10(s_min_bnd) + np.log10(s_max_bnd) * 0.1)])
    else:
        xpost = np.array([np.log10(s)])

    # Initialize before iterating.
    w_tot = w

    # Arbitrary values for missing y-data.
    y[~is_finite] = 0

    # Initial conditions for z.
    if is_weighted:
        # With weighted/missing data an initial guess is provided to ensure faster convergence. For that purpose, a
        # nearest neighbor interpolation followed by a coarse smoothing are performed.
        if z0 is not None:  # an initial guess (z0) has been provided
            z = z0
        else:
            z = y  # InitialGuess(y,is_finite);
    else:
        z = np.zeros(y.shape)

    # Error on p. Smoothness parameter s = 10^p.
    errp = 0.1
    # Relaxation factor rf: to speedup convergence.
    rf = 1 + 0.75 * is_weighted

    # Main iterative process
    robust_iterative_process = True
    robust_step = 0
    iter_n = 0
    while robust_iterative_process:
        # Amount of weights (see the function GCVscore).
        aow = np.sum(w_tot) / y.size  # 0 < aow <= 1
        tol = 1.0
        iter_n = 0

        while tol > tol_z and iter_n < max_iter:
            iter_n += 1
            z0 = z
            dct_y = dct_nd(w_tot * (y - z) + z, f=dct)
            if verbose:
                logging.info('tol {} iter_n {}'.format(tol, iter_n))

            if is_auto and not np.remainder(np.log2(iter_n), 1):
                # The generalized cross-validation (GCV) method is used.
                # We seek the smoothing parameter s that minimizes the GCV
                # score i.e. s = Argmin(GCVscore).
                # Because this process is time-consuming, it is performed from
                # time to time (when iter_n is a power of 2)

                # xpost,f,d = lbfgsb.fmin_l_bfgs_b(gcv,xpost,fprime=None,factr=10.,\
                #   approx_grad=True,bounds=[(log10(s_min_bnd),log10(s_max_bnd))],\
                #   args=(lambda_,aow,dct_y,is_finite,w_tot,y,nof,y_size))

                # if we have no clue what value of s to use, better span the
                # possible range to get a reasonable starting point ...
                # only need to do it once though. nS0 is the number of samples used
                if not s0:
                    ss = np.arange(n_s0) * (1.0 / (n_s0 - 1.0)) * (
                            np.log10(s_max_bnd) - np.log10(s_min_bnd)
                    ) + np.log10(s_min_bnd)
                    g = np.zeros_like(ss)
                    for i, p in enumerate(ss):
                        g[i] = gcv(p, lambda_, aow, dct_y, is_finite, w_tot, y, nof, y.size, smooth_order)
                        # print 10**p,g[i]
                    xpost = [ss[g == np.amin(g)]]
                    # logging.log('{} {} {} {} {}'.format(iter_n, tol, np.amin(g), xpost[0], s))
                else:
                    xpost = [s0]
                xpost, f, d = lbfgsb.fmin_l_bfgs_b(
                    gcv,
                    np.asarray(xpost),
                    fprime=None,
                    factr=10.0,
                    approx_grad=True,
                    bounds=[(np.log10(s_min_bnd), np.log10(s_max_bnd))],
                    args=(lambda_, aow, dct_y, is_finite, w_tot, y, nof, y.size, smooth_order)
                )
            s = 10 ** xpost[0]
            # Update the value we use for the initial s estimate.
            s0 = xpost[0]

            gamma = 1.0 / (1 + (s * np.abs(lambda_)) ** smooth_order)

            z = rf * dct_nd(gamma * dct_y, f=idct) + (1 - rf) * z
            # If no weighted/missing data => tol=0 (no iteration).
            tol = is_weighted * linalg.norm(z0 - z) / linalg.norm(z)

        # Robust Smoothing: iteratively re-weighted process
        if robust:
            h = 1.0
            for i in range(n):
                # Average leverage.
                h0 = np.sqrt(1 + 16.0 * s / spacing[i] ** (2 ** smooth_order))
                h0 = np.sqrt(1 + h) / np.sqrt(2) / h0
                h *= h0

            # Take robust weights into account.
            w_tot = w * robust_weights(y - z, is_finite, h, weight_str)

            # Re-initialize for another iterative weighted process.
            is_weighted = True
            robust_step = robust_step + 1
            robust_iterative_process = robust_step < 2  # 3 robust steps are enough.
        else:
            robust_iterative_process = False

    # Warning messages.
    if is_auto:
        if np.abs(np.log10(s) - np.log10(s_min_bnd)) < errp:
            logging.warning('smoothn:SLowerBound\ns = {.3f} : the lower bound for s has been reached.'
                            'Put s as an input variable if required.'.format(s))
        elif np.abs(np.log10(s) - np.log10(s_max_bnd)) < errp:
            logging.warning('smoothn:SUpperBound\ns = {.3f} : the lower bound for s has been reached.'
                            'Put s as an input variable if required.'.format(s))
    if iter_n == max_iter:
        logging.warning('smoothn:MaxIter\nMaximum number of iterations ({:d}) has been exceeded.'
                        'Increase max_iter option or decrease tol_z value.'.format(max_iter))
    return z, s, w_tot


def gcv(p, lambda_, aow, dct_y, is_finite, w_tot, y, nof, noe, smooth_order):
    # Search the smoothing parameter s that minimizes the GCV score
    # ---
    s = 10 ** p
    gamma = 1.0 / (1 + (s * np.abs(lambda_)) ** smooth_order)
    # --- rss = Residual sum-of-squares
    if aow > 0.9:  # aow = 1 means that all the data are equally weighted
        # very much faster: does not require any inverse DCT
        rss = linalg.norm(dct_y * (gamma - 1.0)) ** 2
    else:
        # take account of the weights to calculate rss:
        y_hat = dct_nd(gamma * dct_y, f=idct)
        rss = linalg.norm(np.sqrt(w_tot[is_finite]) * (y[is_finite] - y_hat[is_finite])) ** 2
    # ---
    tr_h = np.sum(gamma)
    gcv_score = rss / float(nof) / (1.0 - tr_h / float(noe)) ** 2
    return gcv_score


def robust_weights(r, i, h, w_str):
    # weights for robust smoothing.
    mad = np.median(np.abs(r[i] - np.median(r[i])))  # median absolute deviation
    u = np.abs(r / (1.4826 * mad) / np.sqrt(1 - h))  # studentized residuals
    if w_str == "cauchy":
        c = 2.385
        w = 1.0 / (1 + (u / c) ** 2)  # Cauchy weights
    elif w_str == "talworth":
        c = 2.795
        w = u < c  # Talworth weights
    else:
        c = 4.685
        w = (1 - (u / c) ** 2) ** 2.0 * ((u / c) < 1)  # bisquare weights

    w[np.isnan(w)] = 0
    return w


# TODO write InitialGuess function


# NB: filter is 2*I - (np.roll(I,-1) + np.roll(I,1))


def dct_nd(data, f=dct):
    nd = len(data.shape)
    if nd == 1:
        return f(data, norm='ortho', type=2)
    elif nd == 2:
        return f(f(data, norm='ortho', type=2).T, norm='ortho', type=2).T
    elif nd == 3:
        return f(
            f(f(data, norm='ortho', type=2, axis=0), norm='ortho', type=2, axis=1), norm='ortho', type=2, axis=2)


def gpu_forward_fft(data_d, norm='backward'):
    assert data_d.dtype == DTYPE_f
    if data_d.ndim == 1:
        data_d = data_d.reshape(1, data_d.size)
    m, n = data_d.shape
    assert n >= 2
    scale = norm == 'forward'
    forward_fft_d = gpuarray.empty((m, n // 2 + 1), dtype=DTYPE_c)

    plan_forward = cufft.Plan((n,), DTYPE_f, DTYPE_c, batch=m)
    cufft.fft(data_d, forward_fft_d, plan_forward, scale=scale)

    if norm == 'ortho':
        forward_fft_d = forward_fft_d * (1 / sqrt(n))

    return forward_fft_d


def gpu_inverse_fft(fft_data_d, inverse_length=None, norm='backward'):
    assert fft_data_d.dtype == DTYPE_c
    if fft_data_d.ndim == 1:
        fft_data_d = fft_data_d.reshape(1, fft_data_d.size)
    m, n = fft_data_d.shape
    assert n >= 2
    scale = norm == 'backward'
    if inverse_length is None:
        inverse_length = (n - 1) * 2

    inverse_fft_d = gpuarray.empty((m, inverse_length), dtype=DTYPE_f)

    plan_inverse = cufft.Plan((inverse_length,), DTYPE_c, DTYPE_f, batch=m)
    cufft.ifft(fft_data_d, inverse_fft_d, plan_inverse, scale=scale)

    if norm == 'ortho':
        inverse_fft_d = inverse_fft_d * (1 / sqrt(inverse_length))

    return inverse_fft_d


def gpu_forward_dct(data_d, norm='backward'):
    assert data_d.dtype == DTYPE_f
    if data_d.ndim == 1:
        data_d = data_d.reshape(1, data_d.size)
    m, n = data_d.shape
    scale = norm == 'forward'
    assert n >= 2

    # could extend the fft output rather than zero-pad (Mahkoul)
    data_zp_d = gpuarray.zeros((m, 2 * n), dtype=DTYPE_f)
    data_zp_d[:, :n] = data_d

    output_d = gpuarray.empty((m, n + 1), dtype=DTYPE_c)
    w_d = 2 * cumath.exp(-1j * pi * gpuarray.arange(n, dtype=DTYPE_f) / (2 * n))

    plan_forward = cufft.Plan(2 * n, DTYPE_f, DTYPE_c, batch=m)
    cufft.fft(data_zp_d, output_d, plan_forward, scale=scale)

    forward_dct_d = cumisc.multiply(output_d[:, :n].copy(), w_d).real

    if norm == 'ortho':
        a_d = gpuarray.zeros((n,), dtype=DTYPE_f) + (1 / sqrt(2 * n))
        a_d[0] = np.array(1 / sqrt(4 * n), dtype=DTYPE_f)
        forward_dct_d = cumisc.multiply(forward_dct_d, a_d)

    return forward_dct_d


mod_dct = SourceModule("""
__global__ void fft_flip(float *dest, float *src, int wd, int fl, int size)
{
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= size) {return;}
    int row = t_idx / fl;
    int col = t_idx % fl + 1;

    dest[row * (fl + 1) + col] = src[row * wd + wd - col];
}
    __global__ void dct_sift(float *dest, float *src, int wd, int size)
{
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= size) {return;}
    int row = t_idx / wd;
    int col = t_idx % wd;

    if (col % 2 == 0) {dest[row * wd + col] = src[row * wd + col / 2];}
    else {dest[row * wd + col] = src[row * wd + wd - 1 - col / 2];}
}
""")


def gpu_inverse_dct(dct_data_d, norm='backward'):
    assert dct_data_d.dtype == DTYPE_f
    if dct_data_d.ndim == 1:
        dct_data_d = dct_data_d.reshape(1, dct_data_d.size)
    m, n = dct_data_d.shape
    assert n >= 2
    scale = norm == 'backward'
    frequency_width = n // 2
    size_i_flip = DTYPE_i(m * frequency_width)
    size_i = DTYPE_i(m * n)

    idct_output_d = gpuarray.empty((m, n), dtype=DTYPE_f)
    inverse_dct_d = gpuarray.empty((m, n), dtype=DTYPE_f)
    w_d = 0.5 * cumath.exp(1j * pi * gpuarray.arange(frequency_width + 1, dtype=DTYPE_f) / (2 * n))
    fft_data_flip = gpuarray.zeros((m, frequency_width + 1), dtype=DTYPE_f)

    block_size = 32
    x_blocks_flip = ceil(size_i_flip / block_size)
    x_blocks = ceil(size_i / block_size)
    fft_flip = mod_dct.get_function('fft_flip')
    dct_sift = mod_dct.get_function('dct_sift')

    fft_flip(fft_data_flip, dct_data_d, DTYPE_i(n), DTYPE_i(frequency_width), size_i_flip, block=(block_size, 1, 1),
             grid=(x_blocks_flip, 1))

    idct_input_d = cumisc.multiply(dct_data_d[:, :frequency_width + 1].copy() - 1j * fft_data_flip, w_d)

    plan_inverse = cufft.Plan((n,), DTYPE_c, DTYPE_f, batch=m)
    cufft.ifft(idct_input_d, idct_output_d, plan_inverse, scale=scale)

    dct_sift(inverse_dct_d, idct_output_d, DTYPE_i(n), size_i, block=(block_size, 1, 1), grid=(x_blocks, 1))

    if norm == 'forward':
        inverse_dct_d = inverse_dct_d * 2
    if norm == 'ortho':
        x_0 = cumisc.multiply(dct_data_d[:, 0].copy().reshape(m, 1), gpuarray.ones_like(inverse_dct_d, dtype=DTYPE_f))

        inverse_dct_d = (inverse_dct_d * 2 - x_0) / sqrt(2 * n) + x_0 / sqrt(n)

    return inverse_dct_d