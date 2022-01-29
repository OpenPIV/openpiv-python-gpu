import logging

import numpy as np
import numpy.linalg as linalg
# import numpy.ma as ma
import scipy.optimize.lbfgsb as lbfgsb
from scipy.fftpack.realtransforms import dct, idct
# import pycuda.autoinit
# import pycuda.gpuarray as gpuarray
# from pycuda.compiler import SourceModule


def smoothn(
        y,
        n_s0=10,
        axis=None,
        smooth_order=2.0,
        sd=None,
        s0=None,
        z0=None,
        robust=False,
        w=None,
        s=None,
        max_iter=100,
        tol_z=1e-3,
        weight_str="bisquare",
        # **kwargs
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
    n_s0 : optional
    axis : optional
    smooth_order : float, optional
    sd : optional
    s0 : float, optional
    z0 : ndarray or None, optional
        Initial value for the iterative process (default = original data)
    robust : bool, optional
        Carries out a robust smoothing that minimizes the influence of outlying data.
    w : optional
        Specifies a weighting array w of real positive values, that must have the same size as y. Note that a nil weight
        corresponds to a missing value.
    s : float, optional
        Smooths the array y using the smoothing parameter s. s must be a real positive scalar. The larger s is, the
        smoother the output will be. If the smoothing parameter s is omitted (see previous option) or empty (i.e. s =
        None), it is automatically determined using the generalized cross-validation (GCV) method.
    max_iter : int, optional
        Maximum number of iterations allowed (default = 100).
    tol_z : float, optional
        Termination tolerance on Z (default = 1e-3). TolZ must be in [0,1].
    weight_str : optional

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
    # verbose = kwargs['verbose'] if 'verbose' in kwargs.items() else False
    # if type(y) == ma.core.MaskedArray:  # masked array
    #     is_masked = True
    #     mask = y.mask
    #     y = np.array(y)
    #     y[mask] = 0.0
    #     if w is not None:
    #         w = np.array(w)
    #         w[mask] = 0.0
    #     if sd is not None:
    #         w = np.array(1.0 / sd ** 2)
    #         w[mask] = 0.0
    #         sd = None
    #     y[mask] = np.nan

    if sd is not None:
        sd_ = np.array(sd)
        mask = sd > 0.0
        w = np.zeros_like(sd_)
        w[mask] = 1.0 / sd_[mask] ** 2
        # sd = None

    if w is not None:
        w = w / np.amax(w)

    y_shape = y.shape

    # sort axis
    if axis is None:
        axis = tuple(np.arange(y.ndim))

    y_size = y.size  # number of elements
    if y_size < 2:
        z = y
        # exitflag = 0
        # w_tot = 0
        return z, s

    # Smoothness parameter and weights
    # if s != None:
    #  s = []
    if w is None:
        w = np.ones(y_shape)

    # if z0 == None:
    #  z0 = y.copy()

    # "Weighting function" criterion
    weight_str = weight_str.lower()

    # Weights. Zero weights are assigned to not finite values (Inf or NaN),
    # (Inf/NaN values = missing data).
    is_finite = np.array(np.isfinite(y)).astype(np.bool)
    nof = is_finite.sum()  # number of finite elements
    w = w * is_finite
    if np.any(w < 0):
        raise Exception("Weights must all be >=0")
    else:
        # W = W/np.max(W)
        pass

    # Weighted or missing data?
    is_weighted = np.any(w != 1)

    # Robust smoothing?
    # is_robust

    # Automatic smoothing?
    is_auto = not s
    # ---
    # DCTN and IDCTN are required

    # Creation of the lambda_ tensor.
    # lambda_ contains the eingenvalues of the difference matrix used in this
    # penalized least squares process.
    axis = tuple(np.array(axis).flatten())
    # d = y.ndim
    lambda_ = np.zeros(y_shape)
    for i in axis:
        # create a 1 x d array (so e.g. [1,1] for a 2D case
        siz0 = np.ones((1, y.ndim), dtype=np.int)[0]
        siz0[i] = y_shape[i]
        # cos(pi*(reshape(1:y_shape(i),siz0)-1)/y_shape(i)))
        # (arange(1,y_shape[i]+1).reshape(siz0) - 1.)/y_shape[i]
        lambda_ = lambda_ + (
            np.cos(np.pi * (np.arange(1, y_shape[i] + 1) - 1.0) / y_shape[i]).reshape(siz0)
        )
        # else:
        #  lambda_ = lambda_ + siz0
    lambda_ = -2.0 * (len(axis) - lambda_)
    # if not is_auto:
    #     gamma = 1.0 / (1 + (s * abs(lambda_)) ** smooth_order)

    # Upper and lower bound for the smoothness parameter
    # The average leverage (h) is by definition in [0 1]. Weak smoothing occurs
    # if h is close to 1, while over-smoothing appears when h is near 0. Upper
    # and lower bounds for h are given to avoid under- or over-smoothing. See
    # equation relating h to the smoothness parameter (Equation #12 in the
    # referenced CSDA paper).
    n = np.sum(np.array(y_shape) != 1)  # tensor rank of the y-array
    h_min = 1e-6
    h_max = 0.99
    # (h/n)**2 = (1 + a)/( 2 a)
    # a = 1/(2 (h/n)**2 -1)
    # where a = sqrt(1 + 16 s)
    # (a**2 -1)/16
    if n == 0:
        s_min_bnd = np.sqrt((((1 + np.sqrt(1 + 8 * h_max ** (2.0 / n))) / 4.0 / h_max ** (2.0 / n)) ** 2 - 1) / 16.0)
        s_max_bnd = np.sqrt((((1 + np.sqrt(1 + 8 * h_min ** (2.0 / n))) / 4.0 / h_min ** (2.0 / n)) ** 2 - 1) / 16.0)
    else:
        s_min_bnd = None
        s_max_bnd = None
    # try:
    #     s_min_bnd = np.sqrt(
    #         (((1 + np.sqrt(1 + 8 * h_max ** (2.0 / n))) / 4.0 / h_max ** (2.0 / n)) ** 2 - 1)
    #         / 16.0
    #     )
    #     s_max_bnd = np.sqrt(
    #         (((1 + np.sqrt(1 + 8 * h_min ** (2.0 / n))) / 4.0 / h_min ** (2.0 / n)) ** 2 - 1)
    #         / 16.0
    #     )
    # except:
    #     s_min_bnd = None
    #     s_max_bnd = None
    # Initialize before iterating
    w_tot = w

    # --- Initial conditions for z
    if is_weighted:
        # --- With weighted/missing data
        # An initial guess is provided to ensure faster convergence. For that
        # purpose, a nearest neighbor interpolation followed by a coarse
        # smoothing are performed.
        # ---
        if z0 is not None:  # an initial guess (z0) has been provided
            z = z0
        else:
            z = y  # InitialGuess(y,is_finite);
            z[~is_finite] = 0.0
    else:
        z = np.zeros(y_shape)
    # ---
    z0 = z
    y[~is_finite] = 0  # arbitrary values for missing y-data
    # ---
    tol = 1.0
    robust_iterative_process = True
    robust_step = 1
    iter_n = 0
    # --- Error on p. Smoothness parameter s = 10^p
    errp = 0.1
    # opt = optimset('TolX',errp);
    # --- Relaxation factor rf: to speedup convergence
    rf = 1 + 0.75 * is_weighted

    # Main iterative process
    if is_auto:
        if s_min_bnd is not None and s_max_bnd is not None:
            xpost = np.array([(0.9 * np.log10(s_min_bnd) + np.log10(s_max_bnd) * 0.1)])
        else:
            xpost = np.array([100.0])
        # try:
        #     xpost = np.array([(0.9 * np.log10(s_min_bnd) + np.log10(s_max_bnd) * 0.1)])
        # except:
        #     xpost = np.array([100.0])
    else:
        xpost = np.array([np.log10(s)])
    while robust_iterative_process:
        # --- "amount" of weights (see the function GCVscore)
        aow = np.sum(w_tot) / y_size  # 0 < aow <= 1
        # ---
        while tol > tol_z and iter_n < max_iter:
            # if verbose:
            #     logging.log('tol {} iter_n {}'.format(tol, iter_n)
            iter_n += 1
            dct_y = dct_nd(w_tot * (y - z) + z, f=dct)
            if is_auto and not np.remainder(np.log2(iter_n), 1):
                # The generalized cross-validation (GCV) method is used.
                # We seek the smoothing parameter s that minimizes the GCV
                # score i.e. s = Argmin(GCVscore).
                # Because this process is time-consuming, it is performed from
                # time to time (when iter_n is a power of 2)

                # errp in here somewhere

                # xpost,f,d = lbfgsb.fmin_l_bfgs_b(gcv,xpost,fprime=None,factr=10.,\
                #   approx_grad=True,bounds=[(log10(s_min_bnd),log10(s_max_bnd))],\
                #   args=(lambda_,aow,dct_y,is_finite,w_tot,y,nof,y_size))

                # if we have no clue what value of s to use, better span the
                # possible range to get a reasonable starting point ...
                # only need to do it once though. nS0 is teh number of samples used
                if not s0:
                    ss = np.arange(n_s0) * (1.0 / (n_s0 - 1.0)) * (
                            np.log10(s_max_bnd) - np.log10(s_min_bnd)
                    ) + np.log10(s_min_bnd)
                    g = np.zeros_like(ss)
                    for i, p in enumerate(ss):
                        g[i] = gcv(p, lambda_, aow, dct_y, is_finite, w_tot, y, nof, y_size, smooth_order)
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
                    args=(lambda_, aow, dct_y, is_finite, w_tot, y, nof, y_size, smooth_order)
                )
            s = 10 ** xpost[0]
            # update the value we use for the initial s estimate
            s0 = xpost[0]

            gamma = 1.0 / (1 + (s * np.abs(lambda_)) ** smooth_order)

            z = rf * dct_nd(gamma * dct_y, f=idct) + (1 - rf) * z
            # if no weighted/missing data => tol=0 (no iteration)
            tol = is_weighted * linalg.norm(z0 - z) / linalg.norm(z)

            z0 = z  # re-initialization
        # exitflag = iter_n < max_iter

        if robust:  # -- Robust Smoothing: iteratively re-weighted process
            # --- average leverage
            h = np.sqrt(1 + 16.0 * s)
            h = np.sqrt(1 + h) / np.sqrt(2) / h
            h = h ** n
            # --- take robust weights into account
            w_tot = w * robust_weights(y - z, is_finite, h, weight_str)
            # --- re-initialize for another iterative weighted process
            is_weighted = True
            tol = 1
            iter_n = 0
            # ---
            robust_step = robust_step + 1
            robust_iterative_process = robust_step < 3  # 3 robust steps are enough.
        else:
            robust_iterative_process = False  # stop the whole process

    # Warning messages
    if is_auto:
        if np.abs(np.log10(s) - np.log10(s_min_bnd)) < errp:
            warn('smoothn:SLowerBound\ns = {.3f} : the lower bound for s has been reached. Put s as an input variable '
                 'if required.'.format(s))
        elif np.abs(np.log10(s) - np.log10(s_max_bnd)) < errp:
            warn('smoothn:SUpperBound\ns = {.3f} : the lower bound for s has been reached. Put s as an input variable '
                 'if required.'.format(s))
        # warn('MATLAB:smoothn:MaxIter\nMaximum number of iterations ({:d}) has been exceeded. Increase max_iter '
        #      'option or decrease tol_z value.'.format(max_iter))
    return z, s, w_tot


def warn(statement):
    logging.warning(statement)


# GCV score
# function GCVscore = gcv(p)
def gcv(p, lambda_, aow, dct_y, is_finite, w_tot, y, nof, noe, smooth_order):
    # Search the smoothing parameter s that minimizes the GCV score
    # ---
    s = 10 ** p
    gamma = 1.0 / (1 + (s * np.abs(lambda_)) ** smooth_order)
    # --- rss = Residual sum-of-squares
    if aow > 0.9:  # aow = 1 means that all of the data are equally weighted
        # very much faster: does not require any inverse DCT
        rss = linalg.norm(dct_y * (gamma - 1.0)) ** 2
    else:
        # take account of the weights to calculate rss:
        yhat = dct_nd(gamma * dct_y, f=idct)
        rss = linalg.norm(np.sqrt(w_tot[is_finite]) * (y[is_finite] - yhat[is_finite])) ** 2
    # ---
    tr_h = np.sum(gamma)
    gcv_score = rss / np.float(nof) / (1.0 - tr_h / np.float(noe)) ** 2
    return gcv_score


# Robust weights
# function W = RobustWeights(r,I,h,wstr)
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


# NB: filter is 2*I - (np.roll(I,-1) + np.roll(I,1))


def dct_nd(data, f=dct):
    nd = len(data.shape)
    if nd == 1:
        return f(data, norm="ortho", type=2)
    elif nd == 2:
        return f(f(data, norm="ortho", type=2).T, norm="ortho", type=2).T
    elif nd == 3:
        return f(
            f(f(data, norm="ortho", type=2, axis=0), norm="ortho", type=2, axis=1),
            norm="ortho",
            type=2,
            axis=2,
        )
