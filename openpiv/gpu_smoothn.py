
import scipy.optimize.lbfgsb as lbfgsb
from scipy.fftpack.realtransforms import dct, idct
import numpy as np
import numpy.linalg as linalg
import numpy.ma as ma
import logging


def smoothn(
    y,
    nS0=10,
    axis=None,
    smoothOrder=2.0,
    sd=None,
    verbose=False,
    s0=None,
    z0=None,
    isrobust=False,
    W=None,
    s=None,
    MaxIter=100,
    TolZ=1e-3,
    weightstr="bisquare",
):
    """SMOOTHN Robust spline smoothing for 1-D to N-D data.

    SMOOTHN provides a fast, automatized and robust discretized smoothing
    spline for data of any dimension.
    Z = SMOOTHN(Y) automatically smoothes the uniformly-sampled array Y. Y
    can be any N-D noisy array (time series, images, 3D data,...). Non
    finite data (NaN or Inf) are treated as missing values.
    Z = SMOOTHN(Y,S) smoothes the array Y using the smoothing parameter S.
    S must be a real positive scalar. The larger S is, the smoother the
    output will be. If the smoothing parameter S is omitted (see previous
    option) or empty (i.e. S = []), it is automatically determined using
    the generalized cross-validation (GCV) method.
    Z = SMOOTHN(Y,W) or Z = SMOOTHN(Y,W,S) specifies a weighting array W of
    real positive values, that must have the same size as Y. Note that a
    nil weight corresponds to a missing value.

    Parameters
    ----------
    y : ndarray

    nS0 :

    axis :

    smoothOrder float, optional

    sd :

    verbose : bool

    s0 :

    z0 :

    isrobust :

    W :

    s : float
        smoothing parameter
    MaxIter :

    TolZ :

    weightstr :


    Returns
    -------
    z : ndarray

    s :

    exitflag :

    Wtot :


    Robust smoothing
    ----------------
    Z = SMOOTHN(...,'robust') carries out a robust smoothing that minimizes
    the influence of outlying data.
    [Z,S] = SMOOTHN(...) also returns the calculated value for S so that
    you can fine-tune the smoothing subsequently if needed.
    An iteration process is used in the presence of weighted and/or missing
    values. Z = SMOOTHN(...,OPTION_NAME,OPTION_VALUE) smoothes with the
    termination parameters specified by OPTION_NAME and OPTION_VALUE. They
    can contain the following criteria:
       -----------------
       TolZ:       Termination tolerance on Z (default = 1e-3)
                   TolZ must be in ]0,1[
       MaxIter:    Maximum number of iterations allowed (default = 100)
       Initial:    Initial value for the iterative process (default =
                   original data)
       -----------------
    Syntax: [Z,...] = SMOOTHN(...,'MaxIter',500,'TolZ',1e-4,'Initial',Z0);
    [Z,S,EXITFLAG] = SMOOTHN(...) returns a boolean value EXITFLAG that
    describes the exit condition of SMOOTHN:
       1       SMOOTHN converged.
       0       Maximum number of iterations was reached.

    Class Support
    -------------
    Input array can be numeric or logical. The returned array is of class
    double.
    Notes
    -----
    The N-D (inverse) discrete cosine transform functions <a
    href="matlab:web('http://www.biomecardio.com/matlab/dctn.html')"
    >DCTN</a> and <a
    href="matlab:web('http://www.biomecardio.com/matlab/idctn.html')"
    >IDCTN</a> are required.

    Reference
    ---------
    Garcia D, Robust smoothing of gridded data in one and higher dimensions
    with missing values. Computational Statistics & Data Analysis, 2010.
    <a
    href="matlab:web('http://www.biomecardio.com/pageshtm/publi/csda10.pdf')">PDF download</a>
    """
    if type(y) == ma.core.MaskedArray:  # masked array
        is_masked = True
        mask = y.mask_return
        y = np.array(y)
        y[mask] = 0.0
        if W is not None:
            W = np.array(W)
            W[mask] = 0.0
        if sd is not None:
            W = np.array(1.0 / sd ** 2)
            W[mask] = 0.0
            sd = None
        y[mask] = np.nan

    if sd is not None:
        sd_ = np.array(sd)
        mask = sd > 0.0
        W = np.zeros_like(sd_)
        W[mask] = 1.0 / sd_[mask] ** 2
        sd = None

    if W is not None:
        W = W / W.max()

    sizy = y.shape

    # sort axis
    if axis is None:
        axis = tuple(np.arange(y.ndim))

    noe = y.size  # number of elements
    if noe < 2:
        z = y
        exitflag = 0
        Wtot = 0
        return z, s, exitflag, Wtot
    # ---
    # Smoothness parameter and weights
    # if s != None:
    #  s = []
    if W is None:
        W = np.ones(sizy)

    # if z0 == None:
    #  z0 = y.copy()

    # ---
    # "Weighting function" criterion
    weightstr = weightstr.lower()
    # ---
    # Weights. Zero weights are assigned to not finite values (Inf or NaN),
    # (Inf/NaN values = missing data).
    is_finite = np.array(np.isfinite(y)).astype(bool)
    nof = is_finite.sum()  # number of finite elements
    W = W * is_finite
    if np.any(W < 0):
        raise Exception("Weights must all be >=0")
    else:
        # W = W/np.max(W)
        pass
    # ---
    # Weighted or missing data?
    isweighted = np.any(W != 1)
    # ---
    # Robust smoothing?
    # isrobust
    # ---
    # Automatic smoothing?
    isauto = not s
    # ---
    # DCTN and IDCTN are required
    try:
        from scipy.fftpack.realtransforms import dct, idct
    except:
        z = y
        exitflag = -1
        Wtot = 0
        return z, s, exitflag, Wtot

    ## Creation of the Lambda tensor
    # ---
    # Lambda contains the eingenvalues of the difference matrix used in this
    # penalized least squares process.
    axis = tuple(np.array(axis).flatten())
    d = y.ndim
    Lambda = np.zeros(sizy)
    for i in axis:
        # create a 1 x d array (so e.g. [1,1] for a 2D case
        siz0 = np.ones((1, y.ndim), dtype=np.int)[0]
        siz0[i] = sizy[i]
        # cos(pi*(reshape(1:sizy(i),siz0)-1)/sizy(i)))
        # (arange(1,sizy[i]+1).reshape(siz0) - 1.)/sizy[i]
        Lambda = Lambda + (
            np.cos(np.pi * (np.arange(1, sizy[i] + 1) - 1.0) / sizy[i]).reshape(siz0)
        )
        # else:
        #  Lambda = Lambda + siz0
    Lambda = -2.0 * (len(axis) - Lambda)
    if not isauto:
        Gamma = 1.0 / (1 + (s * abs(Lambda)) ** smoothOrder)

    ## Upper and lower bound for the smoothness parameter
    # The average leverage (h) is by definition in [0 1]. Weak smoothing occurs
    # if h is close to 1, while over-smoothing appears when h is near 0. Upper
    # and lower bounds for h are given to avoid under- or over-smoothing. See
    # equation relating h to the smoothness parameter (Equation #12 in the
    # referenced CSDA paper).
    N = sum(np.array(sizy) != 1)  # tensor rank of the y-array
    hMin = 1e-6
    hMax = 0.99
    # (h/n)**2 = (1 + a)/( 2 a)
    # a = 1/(2 (h/n)**2 -1)
    # where a = sqrt(1 + 16 s)
    # (a**2 -1)/16
    try:
        sMinBnd = np.sqrt(
            (((1 + np.sqrt(1 + 8 * hMax ** (2.0 / N))) / 4.0 / hMax ** (2.0 / N)) ** 2 - 1)
            / 16.0
        )
        sMaxBnd = np.sqrt(
            (((1 + np.sqrt(1 + 8 * hMin ** (2.0 / N))) / 4.0 / hMin ** (2.0 / N)) ** 2 - 1)
            / 16.0
        )
    except:
        sMinBnd = None
        sMaxBnd = None
    ## Initialize before iterating
    # ---
    Wtot = W
    # --- Initial conditions for z
    if isweighted:
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
        z = np.zeros(sizy)
    # ---
    z0 = z
    y[~is_finite] = 0  # arbitrary values for missing y-data
    # ---
    tol = 1.0
    RobustIterativeProcess = True
    RobustStep = 1
    nit = 0
    # --- Error on p. Smoothness parameter s = 10^p
    errp = 0.1
    # opt = optimset('TolX',errp);
    # --- Relaxation factor RF: to speedup convergence
    RF = 1 + 0.75 * isweighted
    # ??
    ## Main iterative process
    # ---
    if isauto:
        try:
            xpost = np.array([(0.9 * np.log10(sMinBnd) + np.log10(sMaxBnd) * 0.1)])
        except:
            np.array([100.0])
    else:
        xpost = np.array([np.log10(s)])
    while RobustIterativeProcess:
        # --- "amount" of weights (see the function GCVscore)
        aow = sum(Wtot) / noe  # 0 < aow <= 1
        # ---
        while tol > TolZ and nit < MaxIter:
            if verbose:
                print("tol", tol, "nit", nit)
            nit = nit + 1
            DCTy = dctND(Wtot * (y - z) + z, f=dct)
            if isauto and not np.remainder(np.log2(nit), 1):
                # ---
                # The generalized cross-validation (GCV) method is used.
                # We seek the smoothing parameter s that minimizes the GCV
                # score i.e. s = Argmin(GCVscore).
                # Because this process is time-consuming, it is performed from
                # time to time (when nit is a power of 2)
                # ---
                # errp in here somewhere

                # xpost,f,d = lbfgsb.fmin_l_bfgs_b(gcv,xpost,fprime=None,factr=10.,\
                #   approx_grad=True,bounds=[(log10(sMinBnd),log10(sMaxBnd))],\
                #   args=(Lambda,aow,DCTy,is_finite,Wtot,y,nof,noe))

                # if we have no clue what value of s to use, better span the
                # possible range to get a reasonable starting point ...
                # only need to do it once though. nS0 is teh number of samples used
                if not s0:
                    ss = np.arange(nS0) * (1.0 / (nS0 - 1.0)) * (
                            np.log10(sMaxBnd) - np.log10(sMinBnd)
                    ) + np.log10(sMinBnd)
                    g = np.zeros_like(ss)
                    for i, p in enumerate(ss):
                        g[i] = gcv(
                            p,
                            Lambda,
                            aow,
                            DCTy,
                            is_finite,
                            Wtot,
                            y,
                            nof,
                            noe,
                            smoothOrder,
                        )
                        # print 10**p,g[i]
                    xpost = [ss[g == g.min()]]
                    # print '==============='
                    # print nit,tol,g.min(),xpost[0],s
                    # print '==============='
                else:
                    xpost = [s0]
                xpost, f, d = lbfgsb.fmin_l_bfgs_b(
                    gcv,
                    xpost,
                    fprime=None,
                    factr=10.0,
                    approx_grad=True,
                    bounds=[(np.log10(sMinBnd), np.log10(sMaxBnd))],
                    args=(Lambda, aow, DCTy, is_finite, Wtot, y, nof, noe, smoothOrder),
                )
            s = 10 ** xpost[0]
            # update the value we use for the initial s estimate
            s0 = xpost[0]

            Gamma = 1.0 / (1 + (s * abs(Lambda)) ** smoothOrder)

            z = RF * dctND(Gamma * DCTy, f=idct) + (1 - RF) * z
            # if no weighted/missing data => tol=0 (no iteration)
            tol = isweighted * linalg.norm(z0 - z) / linalg.norm(z)

            z0 = z  # re-initialization
        exitflag = nit < MaxIter

        if isrobust:  # -- Robust Smoothing: iteratively re-weighted process
            # --- average leverage
            h = np.sqrt(1 + 16.0 * s)
            h = np.sqrt(1 + h) / np.sqrt(2) / h
            h = h ** N
            # --- take robust weights into account
            Wtot = W * RobustWeights(y - z, is_finite, h, weightstr)
            # --- re-initialize for another iterative weighted process
            isweighted = True
            tol = 1
            nit = 0
            # ---
            RobustStep = RobustStep + 1
            RobustIterativeProcess = RobustStep < 3  # 3 robust steps are enough.
        else:
            RobustIterativeProcess = False  # stop the whole process

    ## Warning messages
    # ---
    if isauto:
        if abs(np.log10(s) - np.log10(sMinBnd)) < errp:
            warning(
                "MATLAB:smoothn:SLowerBound",
                [
                    "s = %.3f " % (s)
                    + ": the lower bound for s "
                    + "has been reached. Put s as an input variable if required."
                ],
            )
        elif abs(np.log10(s) - np.log10(sMaxBnd)) < errp:
            warning(
                "MATLAB:smoothn:SUpperBound",
                [
                    "s = %.3f " % (s)
                    + ": the upper bound for s "
                    + "has been reached. Put s as an input variable if required."
                ],
            )
        # warning('MATLAB:smoothn:MaxIter',\
        #    ['Maximum number of iterations (%d'%(MaxIter) + ') has '\
        #    + 'been exceeded. Increase MaxIter option or decrease TolZ value.'])
    return z, s, exitflag, Wtot


def warning(s1, s2):
    logging.warning(s1)
    logging.warning(s2[0])


## GCV score
# ---
# function GCVscore = gcv(p)
def gcv(p, Lambda, aow, DCTy, IsFinite, Wtot, y, nof, noe, smoothOrder):
    # Search the smoothing parameter s that minimizes the GCV score
    # ---
    s = 10 ** p
    Gamma = 1.0 / (1 + (s * abs(Lambda)) ** smoothOrder)
    # --- RSS = Residual sum-of-squares
    if aow > 0.9:  # aow = 1 means that all of the data are equally weighted
        # very much faster: does not require any inverse DCT
        RSS = linalg.norm(DCTy * (Gamma - 1.0)) ** 2
    else:
        # take account of the weights to calculate RSS:
        yhat = dctND(Gamma * DCTy, f=idct)
        RSS = linalg.norm(np.sqrt(Wtot[IsFinite]) * (y[IsFinite] - yhat[IsFinite])) ** 2
    # ---
    TrH = sum(Gamma)
    GCVscore = RSS / float(nof) / (1.0 - TrH / float(noe)) ** 2
    return GCVscore


## Robust weights
# function W = RobustWeights(r,I,h,wstr)
def RobustWeights(r, I, h, wstr):
    # weights for robust smoothing.
    MAD = np.median(abs(r[I] - np.median(r[I])))  # median absolute deviation
    u = abs(r / (1.4826 * MAD) / np.sqrt(1 - h))  # studentized residuals
    if wstr == "cauchy":
        c = 2.385
        W = 1.0 / (1 + (u / c) ** 2)  # Cauchy weights
    elif wstr == "talworth":
        c = 2.795
        W = u < c  # Talworth weights
    else:
        c = 4.685
        W = (1 - (u / c) ** 2) ** 2.0 * ((u / c) < 1)  # bisquare weights

    W[np.isnan(W)] = 0
    return W


# NB: filter is 2*I - (np.roll(I,-1) + np.roll(I,1))


def dctND(data, f=dct):
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