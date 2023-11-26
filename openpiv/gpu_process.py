"""This module is dedicated to advanced algorithms for PIV image analysis with NVIDIA
GPU Support.

Note that all data must 32-bit at most to be stored on GPUs. Numpy types must always be
32-bit for compatibility with CUDA. Scalars should be python types in general to work as
function arguments. The block-size argument to GPU kernels should be multiples of 32 to
avoid wasting GPU resources--e.g. (32, 1, 1), (8, 8, 1), etc.

"""
import logging
import warnings
from math import sqrt, ceil, log2, prod
from numbers import Number

import numpy as np
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
from pycuda.compiler import SourceModule

# noinspection PyUnresolvedReferences
import pycuda.autoinit

from openpiv.gpu_validation import ValidationGPU, S2N_TOL, MEAN_TOL, MEDIAN_TOL, RMS_TOL
from openpiv.gpu_smoothn import gpu_smoothn
import gpu_misc
from openpiv.gpu_misc import (
    _check_arrays,
    _Validator,
    _Bool,
    _Number,
    _Integer,
    _Element,
    _Array,
)

# Initialize the scikit-cuda library. This is necessary when certain cumisc calls happen
# that don't autoinit.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    import skcuda.fft as cufft
    from skcuda import misc as cumisc
cumisc.init()

# Define 32-bit types.
DTYPE_i = np.int32
DTYPE_f = np.float32
DTYPE_c = np.complex64

ALLOWED_SUBPIXEL_METHODS = {"gaussian", "parabolic", "centroid"}
ALLOWED_S2N_METHODS = {"peak2peak", "peak2energy", "peak2rms"}
SMOOTHING_PAR = None
N_FFT = 2
SUBPIXEL_METHOD = "gaussian"
S2N_METHOD = "peak2peak"
S2N_WIDTH = 2
_BLOCK_SIZE = 64


class _NFFT(_Integer):
    """Tuple of integers."""

    def validate(self, value):
        if not isinstance(value, (Number, tuple, list)):
            raise TypeError(
                "{} must be a number or sequence of integers.".format(self.public_name)
            )
        if isinstance(value, (tuple, list)):
            if len(value) != 2:
                raise ValueError(
                    "{} must have length of 2 if sequence.".format(self.public_name)
                )
            value = value[0]
            logging.warning(
                "Using the same n_fft_x for both directions. ({})".format(value)
            )
        _Integer.validate(self, value)

        return int(value), int(value)


class CorrelationGPU:
    """Performs the cross-correlation of interrogation windows.

    Can perform correlation by extended search area, where the first window is larger
    than the first window, allowing a for displacements larger than the nominal window
    size to be found.

    Parameters
    ----------
    subpixel_method : {'gaussian', 'centroid', 'parabolic'}, optional
        Method to approximate the subpixel location of the peaks.
    n_fft : int or tuple of ints, optional
        (n_fft_x, n_fft_y), Window size multiplier for fft. Pass a tuple of length 2 for
        asymmetric multipliers -- this is not yet implemented.
    s2n_method : str {'peak2peak', 'peak2energy', 'peak2rms'}, optional
        Method for evaluating the signal-to-noise ratio value from the correlation map.
    s2n_width : int, optional
        Half size of the region around the first correlation peak to ignore for finding
        the second peak. Only used if 's2n_method == peak2peak'.

    """

    subpixel_method = _Element(*ALLOWED_SUBPIXEL_METHODS)
    s2n_method = _Element(*ALLOWED_S2N_METHODS)
    s2n_width = _Integer(min_value=1)
    n_fft = _NFFT(min_value=1)

    def __init__(
        self,
        subpixel_method=SUBPIXEL_METHOD,
        s2n_method=S2N_METHOD,
        s2n_width=S2N_WIDTH,
        n_fft=N_FFT,
    ):
        self.subpixel_method = subpixel_method
        self.s2n_method = s2n_method
        self.s2n_width = s2n_width
        self.n_fft = n_fft

        self._correlation = None
        self._corr_peak1_ = None
        self._corr_idx = None

    def __call__(self, win_a, win_b, shift=None, strain=None):
        """Returns the pixel peaks using the specified correlation method.

        Parameters
        ----------
        win_a, win_b : GPUArray
            3D float (n_windows, ht, wd), interrogation windows stacked in the first
            dimension.
        shift : GPUArray or None, optional
            2D float [du, dv], du and dv are 1D arrays of the x-y shift at each
            interrogation window of the second frame. This is using the x-y convention
            of this code where x is the row and y is the column.
        strain : GPUArray or None, optional
            2D float [u_x, u_y, v_x, v_y], strain tensor.

        Returns
        -------
        i_peak, j_peak : ndarray
            3D float, locations of the subpixel peaks.

        """
        self._s2n_ratio_ = None

        self._init_fft_shape(win_a)

        # Correlate the windows.
        self._correlate_windows(win_a, win_b)

    def get_displacement_peaks(self):
        """Returns the subpixel locations of the displacement peaks.

        Returns
        -------
        i_peak, j_peak : GPUArray
            1D int (n_windows)

        """
        assert (
            self._correlation is not None
        ), "Can only return displacement after correlation peaks have been computed."

        # Get the subpixel location.
        row_sp, col_sp = _gpu_subpixel_approximation(
            self._correlation, self._peak1_idx_, self.subpixel_method
        )

        # Center the peak displacement.
        i_peak, j_peak = self._center_displacement(row_sp, col_sp)

        return i_peak, j_peak

    def free_gpu_data(self):
        """Frees data from GPU."""
        self._correlation = None
        self._corr_peak1_ = None
        self._corr_idx = None

    @property
    def s2n_ratio(self):
        """Signal-to-noise ratio of the cross-correlation.

        Returns
        -------
        GPUArray or None
            2D float

        """
        if self._correlation is None:
            return None

        return self._s2n_ratio

    @property
    def _peak1_idx(self):
        """Returns the row and column of the highest peak in the cross--correlation."""
        if self._peak1_idx_ is None:
            self._peak1_idx_ = _peak_idx(self._correlation)

        return self._peak1_idx_

    @property
    def _corr_peak1(self):
        """Returns the value of the highest peak in the cross-correlation."""
        if self._corr_peak1_ is None:
            self._corr_peak1_ = _peak_value(self._correlation, self._peak1_idx)

        return self._corr_peak1_

    @property
    def _s2n_ratio(self):
        """Computes the signal-to-noise ratio using one of three available methods.

        The signal-to-noise ratio is computed from the correlation and is a measure of
        the quality of the matching between two interrogation windows. Note that this
        method returns the base-10 logarithm of the signal-to-noise ratio.
        The signal-to-noise field takes +np.Inf values where there is no noise.

        Returns
        -------
        ndarray
            2D float (m, n), the base-10 logarithm of the signal-to-noise ratio from the
            correlation map for each vector.

        """
        if self._s2n_ratio_ is None:
            # Compute signal-to-noise ratio by the elected method.
            if self.s2n_method == "peak2energy":
                s2n_ratio = _peak2energy(self._correlation, self._corr_peak1)
            elif self.s2n_method == "peak2rms":
                s2n_ratio = _peak2rms(self._correlation, self._corr_peak1)
            else:
                assert 0 <= self.s2n_width < int(min(self.fft_shape) / 2), (
                    "Mask width must be integer from 0 and to less than half the"
                    "correlation window height or width. Recommended value is 2."
                )
                s2n_ratio = self._get_peak2peak()

            self._s2n_ratio_ = s2n_ratio

        return self._s2n_ratio_

    def _init_fft_shape(self, win):
        """Creates the shape of the fft windows padded up to power of 2 to boost
        speed."""
        n_windows, ht, wd = win.shape

        self.n_windows = n_windows
        self.fft_wd = 2 ** ceil(log2(wd * self.n_fft[0]))
        self.fft_ht = 2 ** ceil(log2(ht * self.n_fft[1]))
        self.fft_shape = (self.fft_ht, self.fft_wd)
        self.fft_size = self.fft_wd * self.fft_ht

    def _correlate_windows(self, win_a, win_b):
        """Computes the cross-correlation of the window stacks with zero-padding.

        Parameters
        ----------
        win_a, win_b : GPUArray
            3D float (n_windows, ht, wd), interrogation windows.

        Returns
        -------
        GPUArray
            3D (n_window, fft_ht, fft_wd), outputs of the correlation function.

        """
        _check_arrays(win_a, win_b, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=3)
        self._peak1_idx_ = None
        self._corr_peak1_ = None

        # Normalize array by computing the norm of each IW.
        win_a_norm = _gpu_normalize_intensity(win_a)
        win_b_norm = _gpu_normalize_intensity(win_b)

        # Zero pad windows to fft shape.
        offset = _zero_pad_offset(win_a, win_b)
        win_a_zp = _gpu_zero_pad(win_a_norm, self.fft_shape, offset=offset)
        win_b_zp = _gpu_zero_pad(win_b_norm, self.fft_shape)

        # Get the cross-correlation.
        corr = _gpu_cross_correlate(win_a_zp, win_b_zp)

        # Correlation is shifted to be relative to center of array.
        self._correlation = gpu_fft_shift(corr)

        # Non-positive correlations are defaulted to center.
        self._check_non_positive_correlation()

    def _check_non_positive_correlation(self):
        """Sets the row and column to the center if the correlation peak is near
        zero."""
        center = gpuarray.ones_like(self._peak1_idx, dtype=DTYPE_i) * DTYPE_i(
            (self.fft_ht // 2) * self.fft_wd + self.fft_wd // 2
        )
        self._peak1_idx_ = gpuarray.if_positive(
            self._corr_peak1, self._peak1_idx, center
        )

    def _center_displacement(self, row_sp, col_sp):
        """Returns the relative position of the peaks with respect to the center of the
        interrogation window.

        Returns
        -------
        i_peak, j_peak : GPUArray
            1D int (n_windows)

        """
        i_peak = row_sp - DTYPE_f(self.fft_ht // 2)
        j_peak = col_sp - DTYPE_f(self.fft_wd // 2)

        return i_peak, j_peak

    def _get_peak2peak(self):
        """Returns the signal-to-noise ratio computed using the peak-to-peak method."""
        assert self._correlation is not None

        corr_peak2 = _get_second_peak(
            self._correlation, self._peak1_idx, self.s2n_width
        )
        s2n_ratio = _peak2peak(self._corr_peak1, corr_peak2)

        return s2n_ratio


class PIVFieldGPU:
    """Geometric information of PIV windows.

    Parameters
    ----------
    frame_shape : tuple of ints
        (ht, wd), shape of the piv frame.
    frame_mask : ndarray
        Int, (ht, wd), mask on the frame coordinates.
    window_size : int
        Size of the interrogation window.
    spacing : int
        Number of pixels between interrogation windows.
    search_ratio : int or None, optional
        Ratio the extended search area to use on the first iteration. If not specified,
        extended search will not be used.
    """

    def __init__(
        self,
        frame_shape,
        window_size,
        spacing,
        frame_mask=None,
        center_field=True,
        search_ratio=None,
    ):
        assert (
            window_size >= 8 and window_size % 8 == 0
        ), "Window size must be a multiple of 8."
        self.frame_shape = frame_shape
        self.window_size = window_size
        self.spacing = spacing
        self.center_field = center_field
        self.search_ratio = search_ratio
        self.shape = field_shape(frame_shape, window_size, spacing)
        self.size = prod(self.shape)

        self._x, self._y = field_coords(
            frame_shape, window_size, spacing, center_field=center_field
        )
        self._x_grid = gpuarray.to_gpu(self._x[0, :].astype(DTYPE_f))
        self._y_grid = gpuarray.to_gpu(self._y[:, 0].astype(DTYPE_f))
        self.is_masked = frame_mask is not None
        self.mask = _field_mask(self._x, self._y, frame_mask)
        self._mask_d = gpuarray.to_gpu(self.mask)

    def get_gpu_mask(self, return_array=False):
        """Returns GPUArray containing field mask if frame is masked, None otherwise.

        Parameters
        ----------
        return_array : bool
            Whether to return an array of zeros if field when mask is None. False
            returns None when mask is None.

        Returns
        -------
        GPUArray or None

        """
        if self.is_masked or return_array:
            return self._mask_d

        return None

    def stack_iw(self, frame_a, frame_b, shift=None, strain=None, search_size=None):
        """Returns 3D arrays of stacked interrogation windows.

        This is necessary to do the FFTs all at once on the GPU. This populates
        interrogation windows from the origin of the image. The implementation requires
        that the window sizes are multiples of 8.

        Parameters
        -----------
        frame_a, frame_b : GPUArray
            2D int (ht, wd), image pair.
        shift : GPUArray or None, optional
            3D float (2, m, n), ([du, dv]), shift of the second window.
        strain : GPUArray or None, optional
            3D float (4, m, n), [u_x, u_y, v_x, v_y], strain rate tensor.
        search_size : int or None, optional
            Size of the second interrogation window for use in extended search method.

        Returns
        -------
        win_a, win_b : GPUArray
            3D float (n_windows, ht, wd), interrogation windows stacked in the first
            dimension.

        """
        _check_arrays(
            frame_a,
            frame_b,
            array_type=gpuarray.GPUArray,
            shape=frame_a.shape,
            dtype=DTYPE_f,
            ndim=2,
        )
        if search_size is not None:
            if not 0 < search_size == int(search_size):
                raise ValueError("search_size must be a positive integer or None.")
        else:
            search_size = self.window_size
        offset_a, offset_b = self._get_search_offset(search_size)

        win_a = _gpu_window_slice(
            frame_a,
            self.shape,
            self.window_size,
            self.spacing,
            offset_a,
            dt=-0.5,
            shift=shift,
            strain=strain,
        )
        win_b = _gpu_window_slice(
            frame_b,
            self.shape,
            search_size,
            self.spacing,
            offset_b,
            dt=0.5,
            shift=shift,
            strain=strain,
        )

        return win_a, win_b

    def free_gpu_data(self):
        """Frees data from GPU."""
        self._mask_d = None

    @property
    def coords(self):
        """Full coordinates of the PIV field.

        Returns
        -------
        x, y : ndarray
            2D float

        """
        return self._x, self._y

    @property
    def grid_coords(self):
        """Vectors containing the grid coordinates of the PIV field.

        Returns
        -------
        x, y : ndarray
            1D float

        """
        return self._x_grid, self._y_grid

    @property
    def center_offset(self):
        """Offsets in pixel units to the window positions to center the velocity
        field on the frame.

        Returns
        -------
        tuple of ints
            [offset_x, offset_y]

        """
        return _center_offset(self.frame_shape, self.window_size, self.spacing)

    def _get_search_offset(self, search_size):
        """Returns offset to center each window with extended search.

        Parameters
        ----------
        search_size : int
            Size of the second interrogation window for use in extended search method.

        Returns
        -------
        offset_a, offset_b
            tuple of ints [offset_x, offset_y]

        """
        offset_a = 0
        offset_b = -(search_size - self.window_size) // 2

        if self.center_field:
            center_offset = self.center_offset
            offset_a = center_offset
            offset_b = (center_offset[0] + offset_b, center_offset[1] + offset_b)

        return offset_a, offset_b


def gpu_piv(frame_a, frame_b, return_s2n=False, **kwargs):
    """Convenience wrapper-function for PIVGPU.

    Parameters
    ----------
    frame_a, frame_b : ndarray
        2D int (ht, wd), grey levels of the first and second frames.
    return_s2n : bool
        Sets whether to return the signal-to-noise ratio. Not returning the
        signal-to-noise speeds up computation significantly, which is default behaviour.
    **kwargs
        PIV settings. See PIVGPU.

    Returns
    -------
    x : ndarray
        2D float (m, n), x-coordinates where the velocity field has been computed.
    y : ndarray
        2D float (m, n), y-coordinates where the velocity field has been computed.
    u : ndarray
        2D float (m, n), horizontal component of velocity in pixel/time units.
    v : ndarray
        2D float (m, n), vertical component of velocity in pixel/time units.
    mask : ndarray
        2D int (m, n), boolean values (True for vectors interpolated from previous
        iteration).
    s2n : ndarray
        2D float (m, n), signal-to-noise ratio of the final velocity field.

    Example
    -------
    x, y, u, v, mask, s2n = gpu_piv(frame_a, frame_b, mask=None,
        window_size_iters=(1, 2), min_window_size=16,
    overlap_ratio=0.5, dt=1, deform=True, smooth=True, nb_validation_iter=2,
        validation_method='median_velocity', median_tol=2)

    """
    piv_gpu = PIVGPU(frame_a.shape, **kwargs)

    x, y = piv_gpu.coords
    u, v = piv_gpu(frame_a, frame_b)
    mask = piv_gpu.field_mask
    s2n = piv_gpu.s2n_ratio if return_s2n else None

    return x, y, u, v, mask, s2n


class _FrameShape(_Validator):
    """2-tuple of integers."""

    def validate(self, frame_shape):
        frame_shape = (
            frame_shape.shape if hasattr(frame_shape, "shape") else tuple(frame_shape)
        )
        if (
            int(frame_shape[0]) != frame_shape[0]
            or int(frame_shape[1]) != frame_shape[1]
        ):
            raise TypeError(
                "frame_shape must be either a tuple of integers or array-like."
            )
        if len(frame_shape) != 2:
            raise ValueError("frame_shape must be 2D.")

        return frame_shape


class _WindowSizeIters(_Validator):
    """Sequence of 2-tuple of integers, corresponding to window size and iterations."""

    def validate(self, window_size_iters):
        # Check is a sequence.
        if not isinstance(window_size_iters, (list, tuple)):
            raise TypeError(
                "{} must be a 2-tuple [window_size, num_iterations] or a sequence of"
                "such 2-tuples.".format(self.public_name)
            )

        # Check nesting of sequence.
        if all([isinstance(ws_iter, (list, tuple)) for ws_iter in window_size_iters]):
            window_size_iters = list(window_size_iters)
        else:
            window_size_iters = [window_size_iters]

        # Check values of sequence.
        for i, values in enumerate(window_size_iters):
            # Check is 2-tuple.
            if len(values) != 2:
                raise ValueError(
                    "{} must contain 2-tuples [window_size, num_iterations].".format(
                        self.public_name
                    )
                )
            ws, num_iters = values

            # Check window size.
            if not isinstance(ws, (Number, float, int)):
                raise TypeError(
                    "Window sizes in {} must be a numeric type.".format(
                        self.public_name
                    )
                )
            if ws % 8 != 0:
                raise ValueError(
                    "Window sizes in {} must be a multiple of 8.".format(
                        self.public_name
                    )
                )

            # Check iterations.
            if not isinstance(num_iters, (Number, float, int)):
                raise TypeError(
                    "Number of iterations in {} must be a numeric type.".format(
                        self.public_name
                    )
                )
            if not 1 <= ws == int(ws):
                raise ValueError(
                    "Number of iterations in {} must be integers greater than or equal"
                    "to 1.".format(self.public_name)
                )

        return tuple([(ws, num_iters) for ws, num_iters in window_size_iters])


class PIVGPU:
    """Iterative GPU-accelerated algorithm that uses translation and deformation of
    interrogation windows.

    At every iteration, the estimate of the displacement and gradient are used to shift
    and deform the interrogation windows used during the next iteration. One or more
    iterations can be performed before the estimated velocity is interpolated onto a
    finer mesh. This is done until the final mesh and number of iterations is met.

    Algorithm Details
    -----------------
    Only window sizes that are multiples of 8 are supported now, and the minimum window
    size is 8.
    Windows are shifted symmetrically to reduce bias errors.
    The displacement obtained after each correlation is the residual displacement dc.
    The new displacement is computed by dx = dpx + dcx and dy = dpy + dcy.
    Validation is done by any combination of signal-to-noise ratio, mean, median and rms
    velocities.
    Smoothn can be used between iterations to improve the estimate and replace missing
    values.

    References
    ----------
    Scarano F, Riethmuller ML (1999) Iterative multigrid approach in PIV image
        processing with discrete window offset.
        Exp Fluids 26:513–523
    Meunier, P., & Leweke, T. (2003). Analysis and treatment of errors due to high
        velocity gradients in particle image
        velocimetry.
        Experiments in fluids, 35(5), 408-421.
    Garcia, D. (2010). Robust smoothing of gridded data in one and higher dimensions
        with missing values.
        Computational statistics & data analysis, 54(4), 1167-1178.

    Parameters
    ----------
    frame_shape : ndarray or tuple of ints
        (ht, wd), size of the images in pixels.
    window_size_iters : tuple, optional
        2-tuples of ints [[window_size0, num_size0_iters], ...], the window sizes and
        number of iterations to perform at each size.
        e.g. [(32, 1), (16, 2), (8, 2)]
    overlap_ratio : float, optional
        Ratio of overlap between two windows (between 0 and 1).
    dt : float, optional
        Time delay separating the two frames.
    mask : ndarray or None, optional
        2D, float, array with values 0 for the background, 1 for the flow-field. If the
        center of a window is on a 0 value the velocity is set to 0.
    deform : bool, optional
        Whether to deform the windows by the velocity gradient at each iteration.
    smooth : bool, optional
        Whether to smooth the intermediate fields.
    num_validation_iters : int, optional
        Number of iterations per validation cycle.
    validation_method : str {'s2n', 'median_velocity', 'mean_velocity', 'rms_velocity'},
        tuple or None, optional
        Method(s) to use for validation.
    s2n_tol, median_tol, mean_tol, rms_tol : float, optional
        Tolerance of the validation methods.
    smoothing_par : float or None, optional
        Smoothing parameter to pass to smoothn to apply to the intermediate velocity
        fields.
    search_ratio : float or None, optional
        Ratio the extended search area to use on the first iteration. If not specified,
        extended search will not be used.
    center_field : bool, optional
        Whether to center the vector field on the image.
    subpixel_method : str {'gaussian', 'centroid', 'parabolic'}, optional
        Method to estimate subpixel location of the peak.
    s2n_method : str {'peak2peak', 'peak2energy', 'peakrms'}, optional
        Method of signal-to-noise-ratio measurement.
    s2n_width : int, optional
        Half size of the region around the first correlation peak to ignore for finding
        the second peak. Default is 2. Only used if s2n_method == 'peak2peak'.
    n_fft : int or tuple of ints, optional
        (n_fft_x, n_fft_y), factor of size of the 2D FFT in x and y-directions. The
        default of 2 is recommended.

    """

    frame_shape = _FrameShape()
    window_size_iters = _WindowSizeIters()
    overlap_ratio = _Number(
        min_value=0, max_value=1, min_closure=False, max_closure=False
    )
    dt = _Number()
    mask = _Array(allow_none=True)
    deform = _Bool()
    smooth = _Bool()
    num_validation_iters = _Integer(min_value=0)
    smoothing_par = _Number(min_value=0, min_closure=False, allow_none=True)
    search_ratio = _Number(min_value=1, min_closure=False, allow_none=True)
    center_field = _Bool()

    def __init__(
        self,
        frame_shape,
        window_size_iters=((32, 1), (16, 2)),
        overlap_ratio=0.5,
        dt=1,
        mask=None,
        deform=True,
        smooth=True,
        num_validation_iters=1,
        validation_method="median_velocity",
        s2n_tol=S2N_TOL,
        median_tol=MEDIAN_TOL,
        mean_tol=MEAN_TOL,
        rms_tol=RMS_TOL,
        smoothing_par=SMOOTHING_PAR,
        search_ratio=None,
        center_field=True,
        subpixel_method=SUBPIXEL_METHOD,
        s2n_method=S2N_METHOD,
        s2n_width=S2N_WIDTH,
        n_fft=N_FFT,
    ):
        self.frame_shape = frame_shape
        self.window_size_iters = window_size_iters
        self.overlap_ratio = overlap_ratio
        self.dt = dt
        self.mask = mask
        self.deform = deform
        self.smooth = smooth
        self.num_validation_iters = num_validation_iters
        self.validation_method = validation_method
        self.s2n_tol = s2n_tol
        self.median_tol = median_tol
        self.mean_tol = mean_tol
        self.rms_tol = rms_tol
        self.center_field = center_field
        self.search_ratio = search_ratio
        self.smoothing_par = smoothing_par
        self.subpixel_method = subpixel_method
        self.s2n_method = s2n_method
        self.s2n_width = s2n_width
        self.n_fft = n_fft

        self._piv_fields_ = None
        self._frame_mask_ = None
        self._corr_gpu = None

    def __call__(self, frame_a, frame_b):
        """Computes velocity field from an image pair.

        Parameters
        ----------
        frame_a, frame_b : ndarray
            2D int (ht, wd), grey levels of the first and second frames.

        Returns
        -------
        u, v : ndarray
            2D float (m, n), horizontal/vertical components of velocity in pixels/time
            units.

        """
        _check_arrays(frame_a, frame_b, array_type=np.ndarray, ndim=2)
        u = v = None
        dp_u = dp_v = None

        # Send masked frames to device.
        frames = self._frames_to_gpu(frame_a, frame_b)

        # Create the correlation object.
        self._corr_gpu = CorrelationGPU(
            subpixel_method=self.subpixel_method,
            s2n_method=self.s2n_method,
            s2n_width=self.s2n_width,
            n_fft=self.n_fft,
        )

        # MAIN LOOP
        for k in _piv_iter(self.window_size_iters):
            self._k = k
            _log_iteration(k)

            # Compute the predictors dp_x and dp_y from the previous displacements.
            if self._k > 0:
                dp_u, dp_v = self._get_predictions(u, v)

            # Compute new velocity previous velocity.
            (
                u,
                v,
            ) = self._get_new_velocity(frames, dp_u, dp_v)

            # Validate velocity fields.
            u, v, val_locations = self._validate_fields(u, v, dp_u, dp_v)
            u, v = self._smooth_fields(u, v, val_locations)

        # Scale by given time factor.
        u = (u / DTYPE_f(self.dt)).get()
        v = (v / DTYPE_f(-self.dt)).get()

        return u, v

    @property
    def coords(self):
        """Returns coordinates where the velocity field has been computed.

        Returns
        -------
        x, y : ndarray
            2D float (m, n)

        """
        return self._piv_fields[-1].coords

    @property
    def field_mask(self):
        """Mask corresponding to the resulting vector field.

        Returns
        -------
        ndarray
            2D int (m, n), boolean values (True for vectors interpolated from previous
            iteration).
        """
        return self._piv_fields[-1].mask

    @property
    def s2n_ratio(self):
        """Signal-to-noise ratio of the final velocity field.

        Returns
        -------
        ndarray
            2D float (m, n)

        """
        if self._corr_gpu is None:
            return None
        shape = self._piv_fields[-1].shape

        return self._corr_gpu.s2n_ratio.reshape(shape)

    def free_gpu_data(self):
        """Frees data from GPU."""
        self._piv_fields_ = None
        self._frame_mask_ = None
        self._corr_gpu = None

    @property
    def _frame_mask(self):
        """Mask for the frame stored as a GPUArray."""
        if self.mask is not None and self._frame_mask_ is None:
            self._frame_mask_ = gpuarray.to_gpu(self.mask)

        return self._frame_mask_

    @property
    def _piv_field_k(self):
        """PIVField at current iteration."""
        return self._piv_fields[self._k]

    @property
    def _piv_fields(self):
        """PIVField for each iteration."""
        if self._piv_fields_ is None:
            self._piv_fields_ = []
            for window_size in _window_sizes(self.window_size_iters):
                window_size = window_size
                spacing = _spacing(window_size, self.overlap_ratio)
                self._piv_fields_.append(
                    PIVFieldGPU(
                        self.frame_shape,
                        window_size,
                        spacing,
                        frame_mask=self.mask,
                        center_field=self.center_field,
                    )
                )

        return self._piv_fields_

    def _frames_to_gpu(self, frame_a, frame_b):
        """Sends frames to device with masking."""
        _check_arrays(
            frame_a, frame_b, array_type=np.ndarray, shape=frame_a.shape, ndim=2
        )
        frame_mask = self._frame_mask

        frame_a_d = gpuarray.to_gpu(frame_a.astype(DTYPE_f))
        frame_b_d = gpuarray.to_gpu(frame_b.astype(DTYPE_f))

        if frame_mask is not None:
            frame_a_d = gpu_misc.gpu_mask(frame_a, frame_mask)
            frame_b_d = gpu_misc.gpu_mask(frame_b, frame_mask)

        return frame_a_d, frame_b_d

    def _get_new_velocity(self, frames, dp_u, dp_v):
        """Returns velocity fields from prediction."""
        # Get the displacement peaks.
        i_peak, j_peak = self._get_displacement_peaks(frames, dp_u, dp_v)
        residual = self._get_residual(i_peak, j_peak)
        _log_residual(residual)

        # Update the field with new values.
        u, v = self._update_velocity(dp_u, dp_v, i_peak, j_peak)

        return u, v

    def _get_predictions(self, u, v):
        """Returns the predicted velocity field to begin the next iteration."""
        _check_arrays(
            u, v, array_type=gpuarray.GPUArray, dtype=DTYPE_f, shape=u.shape, ndim=2
        )
        x0, y0 = self._piv_fields[self._k - 1].grid_coords
        x1, y1 = self._piv_field_k.grid_coords
        mask = self._piv_fields[self._k - 1].get_gpu_mask()

        # Interpolate if dimensions do not agree.
        if self._piv_fields[self._k - 1].window_size != self._piv_field_k.window_size:
            dp_u = gpu_misc.gpu_interpolate(x0, y0, x1, y1, u, mask=mask)
            dp_v = gpu_misc.gpu_interpolate(x0, y0, x1, y1, v, mask=mask)
        else:
            dp_u = u
            dp_v = v

        return dp_u, dp_v

    def _get_displacement_peaks(self, frames, dp_u, dp_v):
        """Returns displacements peaks from cross-correlation of frames using deforming
        interrogation windows."""
        frame_a, frame_b = frames

        # Get arguments for the correlation class.
        search_size = self._get_search_size()
        shift, strain = self._get_window_deformation(dp_u, dp_v)

        # Get stack of all interrogation windows.
        win_a, win_b = self._piv_field_k.stack_iw(
            frame_a, frame_b, shift, strain, search_size=search_size
        )

        # Get window displacement to subpixel accuracy.
        self._corr_gpu(
            win_a,
            win_b,
            shift=shift,
            strain=strain,
        )
        i_peak, j_peak = self._corr_gpu.get_displacement_peaks()

        return i_peak, j_peak

    def _get_search_size(self):
        """Returns the search size for the current iteration.."""
        search_size = None
        if self._k == 0 and self.search_ratio is not None:
            search_size = int(self._piv_field_k.window_size * self.search_ratio)

        return search_size

    def _get_window_deformation(self, dp_u, dp_v):
        """Returns the shift and strain arguments to the correlation class."""
        mask = self._piv_field_k.get_gpu_mask(return_array=True)
        shift = None
        strain = None

        if self._k > 0:
            shift = _field_shift(dp_u, dp_v)
            if self.deform:
                strain = gpu_strain(dp_u, dp_v, mask, self._piv_field_k.spacing)

        return shift, strain

    def _update_velocity(self, dp_u, dp_v, i_peak, j_peak):
        """Updates the velocity values after each iteration."""
        shape = self._piv_field_k.shape
        mask = self._piv_field_k.get_gpu_mask(return_array=True)

        if dp_u is None:
            # u, v take the shape of the peaks.
            u = gpu_misc.gpu_mask(j_peak, mask).reshape(shape)
            v = gpu_misc.gpu_mask(i_peak, mask).reshape(shape)
        else:
            u = _gpu_update_field(dp_u, j_peak, mask)
            v = _gpu_update_field(dp_v, i_peak, mask)

        return u, v

    def _validate_fields(self, u, v, dp_u, dp_v):
        """Return velocity fields with outliers removed."""
        size = u.size
        mask = self._piv_field_k.get_gpu_mask()
        val_locations = None
        s2n_ratio = None
        if self.num_validation_iters == 0:
            return u, v, val_locations
        if "s2n" in self.validation_method:
            s2n_ratio = self._corr_gpu.s2n_ratio

        # Create the validation object.
        self._validation_gpu = ValidationGPU(
            u.shape,
            mask=mask,
            validation_method=self.validation_method,
            s2n_tol=self.s2n_tol,
            median_tol=self.median_tol,
            mean_tol=self.mean_tol,
            rms_tol=self.rms_tol,
        )
        for i in range(self.num_validation_iters):
            # Do s2n validation for one iteration only.
            if i > 0:
                s2n_ratio = None

            # Do the validation.
            new_val_locations = self._validation_gpu(u, v, s2n=s2n_ratio)

            # Save the validation locations
            val_locations = _update_validation_locations(
                val_locations, new_val_locations
            )

            # Abort if there are no invalid vectors.
            n_val = self._validation_gpu.num_validation_locations
            _log_validation(n_val, size)
            if n_val == 0:
                break

            # Replace invalid vectors.
            u, v = self._replace_invalid_vectors(dp_u, dp_v)

        self._validation_gpu = None

        return u, v, val_locations

    def _replace_invalid_vectors(self, dp_u, dp_v):
        """Returns vector fields with invalid vectors replaced."""
        if self._k == 0:
            # For first iteration, replace invalid vectors with local median.
            u, v = self._validation_gpu.replace_vectors(*self._validation_gpu.median)
        else:
            # For subsequent iterations, replace with prediction.
            u, v = self._validation_gpu.replace_vectors(dp_u, dp_v)

        return u, v

    def _smooth_fields(self, u, v, val_locations):
        """Smoothes the velocity field using smoothn."""
        mask = self._piv_field_k.get_gpu_mask()
        if self.smooth:
            w = (1 - val_locations) if val_locations is not None else None
            u, v = gpu_smoothn(u, v, s=self.smoothing_par, mask=mask, w=w)

        return u, v

    def _get_residual(self, i_peak, j_peak):
        """Normalizes the residual by the maximum quantization error of 0.5 pixel."""
        _check_arrays(
            i_peak,
            j_peak,
            array_type=gpuarray.GPUArray,
            dtype=DTYPE_f,
            shape=i_peak.shape,
        )
        try:
            self.residual = (
                sqrt(gpuarray.sum(i_peak**2 + j_peak**2).get() / i_peak.size) / 0.5
            )
        except OverflowError:
            self.residual = np.nan
            logging.warning("Overflow in residuals.")

        return self.residual


def field_shape(frame_shape, window_size, spacing):
    """Returns the shape of the resulting velocity field.

    Parameters
    ----------
    frame_shape : tuple of ints
        (ht, wd), size of the frame in pixels.
    window_size : int
        Size of the interrogation windows.
    spacing : int
        Spacing between vectors in the resulting field, in pixels.

    Returns
    -------
    tuple of ints
        (m, n)

    """
    assert len(frame_shape) == 2, "frame_shape must have length 2."
    assert int(spacing) == spacing > 0, "spacing must be a positive int."
    ht, wd = frame_shape

    m = int((ht - window_size) // spacing) + 1
    n = int((wd - window_size) // spacing) + 1

    return m, n


def field_coords(frame_shape, window_size, spacing, center_field=True):
    """Returns the coordinates of the resulting velocity field.

    Parameters
    ----------
    frame_shape : tuple of ints
        [ht, wd], size of the frame in pixels.
    window_size : int
        Size of the interrogation windows.
    spacing : int
        Spacing between vectors in the resulting field, in pixels.
    center_field : bool, optional
        Whether the coordinates of the interrogation windows are centered on the image.

    Returns
    -------
    x, y : ndarray
        2D float (m, n)

    """
    assert len(frame_shape) == 2, "frame_shape must have length 2."
    assert int(spacing) == spacing > 0, "spacing must be a positive int."

    m, n = field_shape(frame_shape, window_size, spacing)
    half_width = window_size // 2
    offset_x = 0
    offset_y = 0
    if center_field:
        offset_x, offset_y = _center_offset(frame_shape, window_size, spacing)
    x = np.tile(
        np.linspace(
            half_width + offset_x, half_width + offset_x + spacing * (n - 1), n
        ),
        (m, 1),
    )
    y = np.tile(
        np.linspace(
            half_width + offset_y + spacing * (m - 1), half_width + offset_y, m
        ),
        (n, 1),
    ).T

    return x, y


mod_strain = SourceModule(
    """
__global__ void strain_gpu(float *strain, float *u, float *v, int *mask, float h, int m,
                    int n, int size)
{
    // strain : output argument
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= size * 2) {return;}

    // gradient_axis : d/dx = 0, d/dy = 1
    int gradient_axis = t_idx / size;
    int row = t_idx % size / n;
    int col = t_idx % size % n;
    int idx = row * n + col;

    // Use first order differencing on edges.
    int interior = ((row > 0) * (row < m - 1) || !gradient_axis)
                 * ((col > 0) * (col < n - 1) || gradient_axis);

    // Get the indexes of the neighbouring points.
    int idx0 = idx - (row > 0) * (gradient_axis) * n - (col > 0) * !gradient_axis;
    int idx1 = idx + (row < m - 1) * (gradient_axis) * n
                   + (col < n - 1) * !gradient_axis;

    // Revert to first order differencing where field is masked.
    interior = interior * !mask[idx0] * !mask[idx1];
    idx0 = idx0 * !mask[idx0] + idx * mask[idx0];
    idx1 = idx1 * !mask[idx1] + idx * mask[idx1];

    // Do the differencing.
    strain[size * gradient_axis + idx] = (u[idx1] - u[idx0]) / (1 + interior) / h;
    strain[size * (gradient_axis + 2) + idx] = (v[idx1] - v[idx0]) / (1 + interior) / h;
}
"""
)


def gpu_strain(u, v, mask=None, spacing=1):
    """Computes the full 2D strain rate tensor.

    Parameters
    ----------
    u, v : GPUArray
        2D float, velocity fields.
    mask : GPUArray, optional
        Mask for the vector field.
    spacing : float, optional
        Spacing between nodes.

    Returns
    -------
    GPUArray
        3D float (4, m, n) [(u_x, u_y, v_x, v_y)], full strain tensor of the velocity
        fields.

    """
    _check_arrays(u, v, array_type=gpuarray.GPUArray, shape=u.shape, dtype=DTYPE_f)
    assert spacing > 0, "Spacing must be greater than 0."
    m, n = u.shape
    size = u.size
    if mask is not None:
        _check_arrays(mask, array_type=gpuarray.GPUArray, shape=u.shape, dtype=DTYPE_i)
    else:
        mask = gpuarray.zeros_like(u, dtype=DTYPE_i)

    strain = gpuarray.empty((4, m, n), dtype=DTYPE_f)

    block_size = _BLOCK_SIZE
    n_blocks = ceil(size * 2 / block_size)
    strain_gpu = mod_strain.get_function("strain_gpu")
    strain_gpu(
        strain,
        u,
        v,
        mask,
        DTYPE_f(spacing),
        DTYPE_i(m),
        DTYPE_i(n),
        DTYPE_i(size),
        block=(block_size, 1, 1),
        grid=(n_blocks, 1),
    )

    return strain


mod_fft_shift = SourceModule(
    """
__global__ void fft_shift(float *destination, float *source, int ht, int wd, int ws)
{
    // x blocks are windows; y and z blocks are x and y dimensions, respectively.
    int idx_i = blockIdx.x;
    int idx_x = blockIdx.y * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.z * blockDim.y + threadIdx.y;
    if (idx_x >= wd || idx_y >= ht) {return;}

    // Compute the mapping.
    int row_dest = (idx_y + ht / 2) % ht;
    int col_dest = (idx_x + wd / 2) % wd;

    // Get the source and destination indices.
    int s_idx = ws * idx_i + wd * idx_y + idx_x;
    int d_idx = ws * idx_i + wd * row_dest + col_dest;
    destination[d_idx] = source[s_idx];
}
"""
)


def gpu_fft_shift(correlation):
    """Returns the shifted spectrum of stacked fft output.

    Parameters
    ----------
    correlation : GPUArray
        3D float (n_windows, ht, wd), data from fft.

    Returns
    -------
    GPUArray
        3D float (n_windows, ht, wd), shifted data from fft.

    """
    _check_arrays(correlation, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=3)
    n_windows, ht, wd = correlation.shape
    window_size = ht * wd

    correlation_shift = gpuarray.empty_like(correlation, dtype=DTYPE_f)

    block_size = 8
    grid_size = ceil(max(ht, wd) / block_size)
    fft_shift = mod_fft_shift.get_function("fft_shift")
    fft_shift(
        correlation_shift,
        correlation,
        DTYPE_i(ht),
        DTYPE_i(wd),
        DTYPE_i(window_size),
        block=(block_size, block_size, 1),
        grid=(n_windows, grid_size, grid_size),
    )

    return correlation_shift


def _window_sizes(window_size_iters):
    """Generator for window size at each iteration."""
    for ws_iter in window_size_iters:
        for _ in range(ws_iter[1]):
            yield ws_iter[0]


def _spacing(window_size, overlap_ratio):
    """Returns spacing from window size and overlap ratio."""
    return max(1, int(window_size * (1 - overlap_ratio)))


def _field_mask(x, y, frame_mask=None):
    """Creates field mask from frame mask.

    Works for integer-valued coordinates.

    """
    if frame_mask is not None:
        mask = frame_mask[y.astype(DTYPE_i), x.astype(DTYPE_i)]
    else:
        mask = np.zeros_like(x, dtype=DTYPE_i)

    return mask


def _center_offset(frame_shape, window_size, spacing):
    """Returns the left pad to indexes to center the vector-field coordinates on the
    frame.

    This accounts for non-even windows sizes and field dimensions to make the offsets on
    either side of the field differ by one pixel at most.

    """
    ht, wd = frame_shape
    m, n = field_shape(frame_shape, window_size, spacing)

    offset_x = (wd - (spacing * (n - 1) + window_size)) // 2 + window_size % 2
    offset_y = (ht - (spacing * (m - 1) + window_size)) // 2 + window_size % 2

    return offset_x, offset_y


mod_window_slice = SourceModule(
    """
__global__ void window_slice(float *output, float *input, int ws, int spacing,
                    int offset_x, int offset_y, int n, int wd, int ht)
{
    // x blocks are windows; y and z blocks are x and y dimensions, respectively.
    int idx_i = blockIdx.x;
    int idx_x = blockIdx.y * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.z * blockDim.y + threadIdx.y;
    if (idx_x >= ws || idx_y >= ws) {return;}

    // Do the mapping.
    int x = (idx_i % n) * spacing + offset_x + idx_x;
    int y = (idx_i / n) * spacing + offset_y + idx_y;

    // Indices of new array to map to.
    int w_range = idx_i * ws * ws + ws * idx_y + idx_x;

    // Find limits of domain.
    int inside_domain = (x >= 0 && x < wd && y >= 0 && y < ht);

    if (inside_domain) {
        // Apply the mapping.
        output[w_range] = input[(y * wd + x)];
    } else {output[w_range] = 0.0f;}
}

__global__ void window_slice_deform(float *output, float *input, float *shift,
                    float *strain, float dt, int deform, int ws, int spacing,
                    int offset_x, int offset_y, int n_windows, int n, int wd, int ht)
{
    // dt : factor to apply to the shift and strain tensors
    // wd : width (number of columns in the full image)
    // ht : height (number of rows in the full image)
    // x blocks are windows; y and z blocks are x and y dimensions, respectively.
    int idx_i = blockIdx.x;
    int idx_x = blockIdx.y * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.z * blockDim.y + threadIdx.y;
    if (idx_x >= ws || idx_y >= ws) {return;}

    // Get the shift values.
    float u = shift[idx_i];
    float v = shift[n_windows + idx_i];
    float dx;
    float dy;

    if (deform) {
        // Get the strain tensor values.
        float u_x = strain[idx_i];
        float u_y = strain[n_windows + idx_i];
        float v_x = strain[2 * n_windows + idx_i];
        float v_y = strain[3 * n_windows + idx_i];

        // Compute the window vector.
        float r_x = idx_x - ws / 2 + 0.5f;
        float r_y = idx_y - ws / 2 + 0.5f;

        // Compute the deform.
        float du = r_x * u_x + r_y * u_y;
        float dv = r_x * v_x + r_y * v_y;

        // Apply shift and deformation operations.
        dx = (u + du) * dt;
        dy = (v + dv) * dt;
    } else {
        dx = u * dt;
        dy = v * dt;
    }

    // Do the mapping
    float x = (idx_i % n) * spacing + offset_x + idx_x + dx;
    float y = (idx_i / n) * spacing + offset_y + idx_y + dy;

    // Do bilinear interpolation.
    int x1 = floorf(x) - (x == wd - 1);
    int x2 = x1 + 1;
    int y1 = floorf(y) - (y == ht - 1);
    int y2 = y1 + 1;

    // Indices of image to map to.
    int w_range = idx_i * ws * ws + ws * idx_y + idx_x;

    // Find limits of domain.
    int inside_domain = (x1 >= 0 && x2 < wd && y1 >= 0 && y2 < ht);

    if (inside_domain) {
        // Apply the mapping.
        output[w_range] = ((x2 - x) * (y2 - y) * input[(y1 * wd + x1)]  // f11
                           + (x - x1) * (y2 - y) * input[(y1 * wd + x2)]  // f21
                           + (x2 - x) * (y - y1) * input[(y2 * wd + x1)]  // f12
                           + (x - x1) * (y - y1) * input[(y2 * wd + x2)]);  // f22
    } else {output[w_range] = 0.0f;}
}
"""
)


def _gpu_window_slice(
    frame, field_shape_, window_size, spacing, offset, dt=0, shift=None, strain=None
):
    """Creates a 3D array stack of all the interrogation windows using shift and strain.

    Parameters
    -----------
    frame : GPUArray
        2D int (ht, wd), frame form which to create windows.
    field_shape_ : tuple of ints
        (m, n), shape of the vector field.
    window_size : int
        Side dimension of the square interrogation windows.
    spacing : int
        Spacing between vectors of the velocity field.
    offset : int or tuple of ints
        [offset_x, offset_y], adjustment to the location of the windows relative to the
        edge of frame, for the purpose of centering the vector field on the frame.
    dt : float, optional
        Number between -1 and 1 indicating the level of shifting/deform. E.g. 1
        indicates shift by full amount, 0 is stationary. This is applied to the
        deformation in an analogous way.
    shift : GPUArray, optional
        3D float (2, m, n), [du, dv], shift of the second window.
    strain : GPUArray, optional
        3D float (4, m, n), [u_x, u_y, v_x, v_y], strain rate tensor.

    Returns
    -------
    GPUArray
        3D float (n_windows, ht, wd), interrogation windows stacked in the first
        dimension.

    """
    _check_arrays(frame, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=2)
    assert len(field_shape_) == 2
    assert -1 <= dt <= 1
    assert np.all(offset == DTYPE_i(offset))
    if isinstance(offset, Number):
        offset_x_i = offset_y_i = DTYPE_i(offset)
    else:
        offset_x_i, offset_y_i = DTYPE_i(offset)
    ht, wd = frame.shape
    m, n = field_shape_
    n_windows = m * n

    win = gpuarray.empty((n_windows, window_size, window_size), dtype=DTYPE_f)

    block_size = 8
    grid_size = ceil(window_size / block_size)
    if shift is not None:
        _check_arrays(
            shift, array_type=gpuarray.GPUArray, dtype=DTYPE_f, shape=(2, m, n)
        )
        do_deform = DTYPE_i(strain is not None)
        if not do_deform:
            strain = gpuarray.zeros(1, dtype=DTYPE_i)
        else:
            _check_arrays(strain, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=3)
        window_slice = mod_window_slice.get_function("window_slice_deform")
        window_slice(
            win,
            frame,
            shift,
            strain,
            DTYPE_f(dt),
            do_deform,
            DTYPE_i(window_size),
            DTYPE_i(spacing),
            offset_x_i,
            offset_y_i,
            DTYPE_i(n_windows),
            DTYPE_i(n),
            DTYPE_i(wd),
            DTYPE_i(ht),
            block=(block_size, block_size, 1),
            grid=(int(n_windows), grid_size, grid_size),
        )
    else:
        window_slice = mod_window_slice.get_function("window_slice")
        window_slice(
            win,
            frame,
            DTYPE_i(window_size),
            DTYPE_i(spacing),
            offset_x_i,
            offset_y_i,
            DTYPE_i(n),
            DTYPE_i(wd),
            DTYPE_i(ht),
            block=(block_size, block_size, 1),
            grid=(int(n_windows), grid_size, grid_size),
        )

    return win


def _zero_pad_offset(win_a, win_b):
    """Returns offset required to align the data in both windows.

    Parameters
    ----------
    win_a, win_b : GPUArray
        3D float (n_windows, ht, wd), interrogation windows.

    Returns
    -------
    tuple of ints
        [offset_x, offset_y]

    """
    n_windows_a, ht_a, wd_a = win_a.shape
    n_windows_b, ht_b, wd_b = win_b.shape
    assert n_windows_a == n_windows_b
    assert (
        ht_b >= ht_b and wd_b >= wd_a
    ), "Dimensions of the second window must be equal or larger than the first."

    offset = ((wd_b - wd_a) // 2, (ht_b - ht_a) // 2)

    return offset


mod_norm = SourceModule(
    """
__global__ void normalize(float *array, float *array_norm, float *mean, int window_size,
                    int size)
{
    // global thread id for 1D grid of 2D blocks
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= size) {return;}

    // indices for mean matrix
    int w_idx = t_idx / window_size;

    array_norm[t_idx] = array[t_idx] - mean[w_idx];
}
"""
)


def _gpu_normalize_intensity(win):
    """Remove the mean from each IW of a 3D stack of interrogation windows.

    Parameters
    ----------
    win : GPUArray
        3D float (n_windows, ht, wd), interrogation windows.

    Returns
    -------
    GPUArray
        3D float (n_windows, ht, wd), normalized intensities in the windows.

    """
    _check_arrays(win, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=3)
    n_windows, ht, wd = win.shape
    window_size = ht * wd
    size = win.size

    win_norm = gpuarray.zeros((n_windows, ht, wd), dtype=DTYPE_f)

    mean = cumisc.mean(win.reshape(n_windows, int(window_size)), axis=1)

    block_size = _BLOCK_SIZE
    grid_size = ceil(size / block_size)
    normalize = mod_norm.get_function("normalize")
    normalize(
        win,
        win_norm,
        mean,
        DTYPE_i(window_size),
        DTYPE_i(size),
        block=(block_size, 1, 1),
        grid=(grid_size, 1),
    )

    return win_norm


mod_zp = SourceModule(
    """
__global__ void zero_pad(float *array_zp, float *array, int fft_ht, int fft_wd, int ht,
                    int wd, int dx, int dy)
{
    // index, x blocks are windows; y and z blocks are x and y dimensions, respectively
    int ind_i = blockIdx.x;
    int ind_x = blockIdx.y * blockDim.x + threadIdx.x;
    int ind_y = blockIdx.z * blockDim.y + threadIdx.y;

    // get range of values to map
    int data_range = ind_i * ht * wd + wd * ind_y + ind_x;
    int zp_range = ind_i * fft_ht * fft_wd + fft_wd * (ind_y + dy) + ind_x + dx;

    // apply the map
    array_zp[zp_range] = array[data_range];
}
"""
)


def _gpu_zero_pad(win, fft_shape, offset=0):
    """Zero-pads a 3D stack of interrogation windows for FFT.

    Parameters
    ----------
    win : GPUArray
        3D float (n_windows, ht, wd), interrogation windows.
    fft_shape : tuple of ints
        (ht, wd), shape to zero pad the date to.
    offset: int or tuple of ints, optional
        [offset_x, offset_y], offsets to the destination index in the padded array. Used
        by the extended-search-area PIV method.

    Returns
    -------
    GPUArray
        3D float, windows which have been zero-padded.

    """
    _check_arrays(win, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=3)
    if isinstance(offset, (Number, int, float)):
        assert 0 <= offset == int(offset), "offset(s) must be positive integers(s)."
        offset_x_i = offset_y_i = DTYPE_i(offset)
    else:
        offset_x_i, offset_y_i = DTYPE_i(offset)
    n_windows, wd, ht = win.shape
    fft_ht_i, fft_wd_i = DTYPE_i(fft_shape)

    win_zp = gpuarray.zeros((n_windows, *fft_shape), dtype=DTYPE_f)

    block_size = 8
    grid_size = ceil(max(wd, ht) / block_size)
    zero_pad = mod_zp.get_function("zero_pad")
    zero_pad(
        win_zp,
        win,
        fft_ht_i,
        fft_wd_i,
        DTYPE_i(ht),
        DTYPE_i(wd),
        offset_x_i,
        offset_y_i,
        block=(block_size, block_size, 1),
        grid=(n_windows, grid_size, grid_size),
    )

    return win_zp


def _gpu_cross_correlate(win_a, win_b):
    """Returns circular cross-correlation between two stacks of interrogation windows.

    The correlation function is computed using the correlation theorem.

    Parameters
    ----------
    win_a, win_b : GPUArray
        3D float (n_windows, fft_ht, fft_wd), zero-padded interrogation windows.

    Returns
    -------
    GPUArray
        3D (n_windows, fft_ht, fft_wd), outputs of the cross-correlation function.

    """
    _check_arrays(
        win_a,
        win_b,
        array_type=gpuarray.GPUArray,
        dtype=DTYPE_f,
        shape=win_b.shape,
        ndim=3,
    )
    n_windows, fft_ht, fft_wd = win_a.shape

    win_cross_correlate = gpuarray.empty((n_windows, fft_ht, fft_wd), DTYPE_f)
    win_a_fft = gpuarray.empty((n_windows, fft_ht, fft_wd // 2 + 1), DTYPE_c)
    win_b_fft = gpuarray.empty((n_windows, fft_ht, fft_wd // 2 + 1), DTYPE_c)

    # Get the forward transform.
    plan_forward = cufft.Plan((fft_ht, fft_wd), DTYPE_f, DTYPE_c, batch=n_windows)
    cufft.fft(win_a, win_a_fft, plan_forward)
    cufft.fft(win_b, win_b_fft, plan_forward)

    # Multiply the FFTs.
    win_fft_product = win_b_fft * win_a_fft.conj()

    # Get the inverse transform.
    plan_inverse = cufft.Plan((fft_ht, fft_wd), DTYPE_c, DTYPE_f, batch=n_windows)
    cufft.ifft(win_fft_product, win_cross_correlate, plan_inverse, True)

    return win_cross_correlate


mod_index_update = SourceModule(
    """
__global__ void window_index_f(float *dest, float *src, int *indices, int ws,
                    int n_windows)
{
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= n_windows) {return;}

    dest[t_idx] = src[t_idx * ws + indices[t_idx]];
}
"""
)


def _gpu_window_index_f(correlation, indices):
    """Returns the values of the peaks from the 2D correlation.

    Parameters
    ----------
    correlation : GPUArray
        2D float (n_windows, m * n), correlation values of each window.
    indices : GPUArray
        1D int (n_windows), indexes of the peaks.

    Returns
    -------
    GPUArray
        1D float (m * n)

    """
    _check_arrays(correlation, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=2)
    n_windows, window_size = correlation.shape
    _check_arrays(
        indices, array_type=gpuarray.GPUArray, dtype=DTYPE_i, shape=(n_windows,), ndim=1
    )

    peak = gpuarray.empty(n_windows, dtype=DTYPE_f)

    block_size = _BLOCK_SIZE
    grid_size = ceil(n_windows / block_size)
    index_update = mod_index_update.get_function("window_index_f")
    index_update(
        peak,
        correlation,
        indices,
        DTYPE_i(window_size),
        DTYPE_i(n_windows),
        block=(block_size, 1, 1),
        grid=(grid_size, 1),
    )

    return peak


def _peak_idx(correlation):
    """Returns the row and column of the highest peak in correlation function.

    Parameters
    ----------
    correlation : GPUArray
        3D float, image of the correlation function.

    Returns
    -------
    peak_idx : GPUArray
        1D int, index of peak location in reshaped correlation function.

    """
    n_windows, wd, ht = correlation.shape

    # cumisc.argmax() has different behaviour when n_windows < 2.
    if n_windows > 1:
        corr_reshape = correlation.reshape(n_windows, wd * ht)
        peak_idx = cumisc.argmax(corr_reshape, axis=1).astype(DTYPE_i)
    else:
        corr_reshape = correlation.reshape((wd * ht, 1))
        peak_idx = cumisc.argmax(corr_reshape, axis=0).astype(DTYPE_i)

    return peak_idx


def _peak_value(correlation, peak_idx):
    """Returns the value of the highest peak in correlation function.

    Parameters
    ----------
    peak_idx : GPUArray
        1D int, image of the correlation function.

    Returns
    -------
    GPUArray
        1D int, flattened index of corr peak.

    """
    n_windows, wd, ht = correlation.shape
    corr_reshape = correlation.reshape(n_windows, wd * ht)
    peak_value = _gpu_window_index_f(corr_reshape, peak_idx)

    return peak_value


mod_subpixel_approximation = SourceModule(
    """
__global__ void gaussian(float *row_sp, float *col_sp, int *p_idx, float *corr,
                    int n_windows, int ht, int wd, int ws)
{
    // x blocks are windows; y and z blocks are x and y dimensions, respectively.
    int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (w_idx >= n_windows) {return;}
    const float small = 1e-20f;

    // Compute the index mapping.
    int row = p_idx[w_idx] / wd;
    int col = p_idx[w_idx] % wd;
    float c = corr[ws * w_idx + wd * row + col];
    int non_zero = c > 0;

    if (row > 0 && row < ht - 1) {
        float cd = corr[ws * w_idx + wd * (row - 1) + col];
        float cu = corr[ws * w_idx + wd * (row + 1) + col];
        if (cd > 0 && cu > 0 && non_zero) {
            cd = logf(cd);
            cu = logf(cu);
            row_sp[w_idx] = row + 0.5f * (cd - cu) / (cd - 2.0f * logf(c) + cu + small);
        } else {row_sp[w_idx] = row;}
    } else {row_sp[w_idx] = row;}

    if (col > 0 && col < wd - 1) {
        float cl = corr[ws * w_idx + wd * row + col - 1];
        float cr = corr[ws * w_idx + wd * row + col + 1];
        if (cl > 0 && cr > 0 && non_zero) {
            cl = logf(cl);
            cr = logf(cr);
            col_sp[w_idx] = col + 0.5f * (cl - cr) / (cl - 2.0f * logf(c) + cr + small);
        } else {col_sp[w_idx] = col;}
    } else {col_sp[w_idx] = col;}
}

__global__ void parabolic(float *row_sp, float *col_sp, int *p_idx, float *corr,
                    int n_windows, int ht, int wd, int ws)
{
    // x blocks are windows; y and z blocks are x and y dimensions, respectively.
    int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (w_idx >= n_windows) {return;}
    const float small = 1e-20f;

    // Compute the index mapping.
    int row = p_idx[w_idx] / wd;
    int col = p_idx[w_idx] % wd;
    float c = corr[ws * w_idx + wd * row + col];

    if (row > 0 && row < ht - 1) {
        float cd = corr[ws * w_idx + wd * (row - 1) + col];
        float cu = corr[ws * w_idx + wd * (row + 1) + col];
        row_sp[w_idx] = row + 0.5f * (cd - cu) / (cd - 2.0f * c + cu + small);
    } else {row_sp[w_idx] = row;}

    if (col > 0 && col < wd - 1) {
        float cl = corr[ws * w_idx + wd * row + col - 1];
        float cr = corr[ws * w_idx + wd * row + col + 1];
        col_sp[w_idx] = col + 0.5f * (cl - cr) / (cl - 2.0f * c + cr + small);
    } else {col_sp[w_idx] = col;}
}

__global__ void centroid(float *row_sp, float *col_sp, int *p_idx, float *corr,
                    int n_windows, int ht, int wd, int ws)
{
    // x blocks are windows; y and z blocks are x and y dimensions, respectively.
    int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (w_idx >= n_windows) {return;}
    const float small = 1e-20f;

    // Compute the index mapping.
    int row = p_idx[w_idx] / wd;
    int col = p_idx[w_idx] % wd;
    float c = corr[ws * w_idx + wd * row + col];
    int non_zero = c > 0;

    if (row > 0 && row < ht - 1) {
        float cd = corr[ws * w_idx + wd * (row - 1) + col];
        float cu = corr[ws * w_idx + wd * (row + 1) + col];
        if (cd > 0 && cu > 0 && non_zero) {
            row_sp[w_idx] = row + (cu - cd) / (cd + c + cu + small);
        } else {row_sp[w_idx] = row;}
    } else {row_sp[w_idx] = row;}

    if (col > 0 && col < wd - 1) {
        float cl = corr[ws * w_idx + wd * row + col - 1];
        float cr = corr[ws * w_idx + wd * row + col + 1];
        if (cl > 0 && cr > 0 && non_zero) {
            col_sp[w_idx] = col + (cr - cl) / (cl + c + cr + small);
        } else {col_sp[w_idx] = col;}
    } else {col_sp[w_idx] = col;}
}
"""
)


def _gpu_subpixel_approximation(correlation, peak_idx, method):
    """Returns the subpixel position of the peaks using gaussian approximation.

    Parameters
    ----------
    correlation : GPUArray
       3D float (n_windows, fft_wd, fft_ht), cross-correlation data from each window
       pair.
    peak_idx : GPUArray
        1D int (n_windows), index position of the correlation peaks.
    method : str {'gaussian', 'parabolic', 'centroid'}
        Method of the subpixel approximation.

    Returns
    -------
    row_sp, col_sp : GPUArray
        1D float (n_windows), row and column positions of the subpixel peak.

    """
    _check_arrays(correlation, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=3)
    _check_arrays(
        peak_idx,
        array_type=gpuarray.GPUArray,
        dtype=DTYPE_i,
        shape=(correlation.shape[0],),
        ndim=1,
    )
    assert method in ALLOWED_SUBPIXEL_METHODS
    n_windows, ht, wd = correlation.shape
    window_size = ht * wd

    row_sp = gpuarray.empty_like(peak_idx, dtype=DTYPE_f)
    col_sp = gpuarray.empty_like(peak_idx, dtype=DTYPE_f)

    block_size = _BLOCK_SIZE
    grid_size = ceil(n_windows / block_size)
    if method == "gaussian":
        gaussian_approximation = mod_subpixel_approximation.get_function("gaussian")
        gaussian_approximation(
            row_sp,
            col_sp,
            peak_idx,
            correlation,
            DTYPE_i(n_windows),
            DTYPE_i(ht),
            DTYPE_i(wd),
            DTYPE_i(window_size),
            block=(block_size, 1, 1),
            grid=(grid_size, 1),
        )
    elif method == "parabolic":
        parabolic_approximation = mod_subpixel_approximation.get_function("parabolic")
        parabolic_approximation(
            row_sp,
            col_sp,
            peak_idx,
            correlation,
            DTYPE_i(n_windows),
            DTYPE_i(ht),
            DTYPE_i(wd),
            DTYPE_i(window_size),
            block=(block_size, 1, 1),
            grid=(grid_size, 1),
        )
    else:
        centroid_approximation = mod_subpixel_approximation.get_function("centroid")
        centroid_approximation(
            row_sp,
            col_sp,
            peak_idx,
            correlation,
            DTYPE_i(n_windows),
            DTYPE_i(ht),
            DTYPE_i(wd),
            DTYPE_i(window_size),
            block=(block_size, 1, 1),
            grid=(grid_size, 1),
        )

    return row_sp, col_sp


def _peak2energy(correlation, corr_peak):
    """Returns the mean-energy measure of the signal-to-noise-ratio."""
    _check_arrays(correlation, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=3)
    _check_arrays(
        corr_peak,
        array_type=gpuarray.GPUArray,
        dtype=DTYPE_f,
        size=correlation.shape[0],
    )
    n_windows, wd, ht = correlation.shape
    size = wd * ht

    # Remove negative correlation values.
    gpu_misc.gpu_remove_negative(corr_peak)
    gpu_misc.gpu_remove_negative(correlation)

    corr_reshape = correlation.reshape(n_windows, size)
    corr_energy = cumisc.mean(corr_reshape**2, axis=1)
    s2n_ratio = cumath.log10(corr_peak**2 / corr_energy)
    gpu_misc.gpu_remove_nan(s2n_ratio)

    return s2n_ratio


def _peak2rms(correlation, corr_peak):
    """Returns the RMS-measure of the signal-to-noise-ratio."""
    correlation_rms = _gpu_mask_rms(correlation, corr_peak)
    s2n_ratio = _peak2energy(correlation_rms, corr_peak)

    return s2n_ratio


def _get_second_peak(correlation_positive, peak1_idx, mask_width):
    """Find the value of the second-largest peak.

    The second-largest peak is the height of the peak in the region outside a
    width * width sub-matrix around the first correlation peak.

    Parameters
    ----------
    correlation_positive : GPUArray
        3D float (n_windows, fft_wd, fft_ht), correlation data with negative values
        removed.
    peak1_idx : GPUArray
        1D int (n_windows) index locations of the first correlation peaks.
    mask_width : int
        Half size of the region around the first correlation peak to ignore for
        finding the second peak.

    Returns
    -------
    GPUArray
        3D float (n_windows, fft_wd, fft_ht), value of the second correlation peak
        for each interrogation window.

    """
    # Set points around the first peak to zero.
    correlation_masked = _gpu_mask_peak(correlation_positive, peak1_idx, mask_width)

    # Get the height of the second peak of correlation.
    peak2_idx = _peak_idx(correlation_masked)
    corr_max2 = _peak_value(correlation_masked, peak2_idx)

    return corr_max2


def _peak2peak(corr_peak1, corr_peak2):
    """Returns the peak-to-peak measure of the signal-to-noise-ratio."""
    _check_arrays(
        corr_peak1,
        corr_peak2,
        array_type=gpuarray.GPUArray,
        dtype=DTYPE_f,
        shape=corr_peak1.shape,
    )

    # Remove negative peaks.
    gpu_misc.gpu_remove_negative(corr_peak1)
    gpu_misc.gpu_remove_negative(corr_peak2)

    s2n_ratio = cumath.log10(corr_peak1 / corr_peak2)
    gpu_misc.gpu_remove_nan(s2n_ratio)

    return s2n_ratio


mod_mask_peak = SourceModule(
    """
__global__ void mask_peak(float *corr, int *p_idx, int mask_w, int ht,
                    int wd, int mask_dim, int size)
{
    // x blocks are windows; y and z blocks are x and y dimensions, respectively.
    int idx_i = blockIdx.x;
    int idx_x = blockIdx.y * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.z * blockDim.y + threadIdx.y;
    if (idx_x >= mask_dim || idx_y >= mask_dim) {return;}

    // Get the mapping.
    int row = p_idx[idx_i] / wd - mask_w + idx_y;
    int col = p_idx[idx_i] % wd - mask_w + idx_x;

    // Mask only if inside window domain.
    if (row >= 0 && row < ht && col >= 0 && col < wd) {
        // Mask the point.
        corr[idx_i * size + row * wd + col] = 0.0f;
    }
}
"""
)


def _gpu_mask_peak(correlation_positive, peak_idx, mask_width):
    """Returns correlation windows with points around peak masked.

    Parameters
    ----------
    correlation_positive : GPUArray.
        3D float (n_windows, fft_wd, fft_ht), correlation data with negative values
        removed.
    peak_idx : GPUArray
        1D int (n_windows), index position of the peaks.
    mask_width : int
        Half size of the region around the first correlation peak to ignore for finding
        the second peak.

    Returns
    -------
    GPUArray
        3D float.

    """
    _check_arrays(
        correlation_positive, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=3
    )
    n_windows, ht, wd = correlation_positive.shape
    _check_arrays(
        peak_idx,
        array_type=gpuarray.GPUArray,
        dtype=DTYPE_i,
        shape=(n_windows,),
    )
    window_size = ht * wd
    assert 0 <= mask_width < int(min(ht, wd) / 2), (
        "Mask width must be integer from 0 and to less than half the correlation "
        "window height or width. Recommended value is 2."
    )
    mask_dim = mask_width * 2 + 1

    correlation_masked = correlation_positive.copy()

    block_size = 8
    grid_size = ceil(mask_dim / block_size)
    mask_peak = mod_mask_peak.get_function("mask_peak")
    mask_peak(
        correlation_masked,
        peak_idx,
        DTYPE_i(mask_width),
        DTYPE_i(ht),
        DTYPE_i(wd),
        DTYPE_i(mask_dim),
        DTYPE_i(window_size),
        block=(block_size, block_size, 1),
        grid=(n_windows, grid_size, grid_size),
    )

    return correlation_masked


mod_correlation_rms = SourceModule(
    """
__global__ void correlation_rms(float *corr, float *corr_p, int ht, int wd, int size)
{
    // x blocks are windows; y and z blocks are x and y dimensions, respectively.
    int idx_i = blockIdx.x;
    int idx_x = blockIdx.y * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.z * blockDim.y + threadIdx.y;
    if (idx_x >= wd || idx_y >= ht) {return;}
    int idx = idx_i * size + idx_y * wd + idx_x;

    // Mask the point if its value greater than the half-peak value.
    if (corr[idx] >= corr_p[idx_i] / 2.0f) {corr[idx] = 0.0f;}
}
"""
)


def _gpu_mask_rms(correlation_positive, corr_peak):
    """Returns correlation windows with values greater than half the primary peak height
    zeroed.

    Parameters
    ----------
    correlation_positive : GPUArray.
        3D float (n_windows, fft_wd, fft_ht), correlation data with negative values
        removed.
    corr_peak : GPUArray
        1D float (n_windows), value of peaks.

    Returns
    -------
    GPUArray
        3D float.

    """
    _check_arrays(
        correlation_positive, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=3
    )
    n_windows, ht, wd = correlation_positive.shape
    _check_arrays(
        corr_peak, array_type=gpuarray.GPUArray, dtype=DTYPE_f, shape=(n_windows,)
    )
    window_size = ht * wd

    correlation_masked = correlation_positive.copy()

    block_size = 8
    grid_size = ceil(max(ht, wd) / block_size)
    fft_shift = mod_correlation_rms.get_function("correlation_rms")
    fft_shift(
        correlation_masked,
        corr_peak,
        DTYPE_i(ht),
        DTYPE_i(wd),
        DTYPE_i(window_size),
        block=(block_size, block_size, 1),
        grid=(n_windows, grid_size, grid_size),
    )

    return correlation_masked


def _piv_iter(window_size_iters):
    """Generator for PIV iterations."""
    i = 0
    for ws_iter in window_size_iters:
        for _ in range(ws_iter[1]):
            yield i
            i += 1


def _field_shift(u, v):
    """Returns the stacked pixel shifts in each direction."""
    _check_arrays(
        u, v, array_type=gpuarray.GPUArray, shape=u.shape, dtype=DTYPE_f, ndim=2
    )
    m, n = u.shape

    shift = gpuarray.empty((2, m, n), dtype=DTYPE_f)
    shift[0, :, :] = u
    shift[1, :, :] = v
    # shift = gpuarray.stack(u, v, axis=0)  # This should work in latest version of
    # PyCUDA.

    return shift


mod_update = SourceModule(
    """
__global__ void update_values(float *f_new, float *f_old, float *peak, int *mask,
                    int size)
{
    // u_new : output argument
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= size) {return;}

    f_new[t_idx] = (f_old[t_idx] + peak[t_idx]) * (1 - mask[t_idx]);
}
"""
)


def _gpu_update_field(dp, peak, mask):
    """Returns updated velocity field values with masking.

    Parameters
    ----------
    dp : GPUArray.
        nD float, predicted displacement.
    peak : GPUArray
        nD float, location of peaks.
    mask : GPUArray
        nD int, mask.

    Returns
    -------
    GPUArray
        nD float.

    """
    _check_arrays(dp, peak, array_type=gpuarray.GPUArray, dtype=DTYPE_f, size=dp.size)
    _check_arrays(mask, array_type=gpuarray.GPUArray, dtype=DTYPE_i, size=dp.size)
    size = dp.size

    f = gpuarray.empty_like(dp, dtype=DTYPE_f)

    block_size = _BLOCK_SIZE
    grid_size = ceil(size / block_size)
    update_values = mod_update.get_function("update_values")
    update_values(
        f, dp, peak, mask, DTYPE_i(size), block=(block_size, 1, 1), grid=(grid_size, 1)
    )

    return f


def _update_validation_locations(val_locations, new_val_locations):
    """Returns combined validations locations."""
    if val_locations is None:
        val_locations = new_val_locations
    else:
        # val_locations = gpuarray.logical_or(val_locations, val_locations)
        val_locations = gpu_misc.gpu_logical_or(new_val_locations, val_locations)

    return val_locations


def _log_iteration(k):
    """Logs the iteration number."""
    logging.info("ITERATION {}".format(k))


def _log_residual(residual):
    """Logs the normalized residual."""
    if residual != np.nan:
        logging.info("Normalized residual : {}.".format(residual))
    else:
        logging.warning("Overflow in residuals.")


def _log_validation(n_val, size):
    """Logs the number of vectors to be validated."""
    if n_val > 0:
        logging.info(
            "Validating {} out of {} vectors ({:.2%}).".format(
                n_val, size, n_val / size
            )
        )
    else:
        logging.info("No invalid vectors.")
