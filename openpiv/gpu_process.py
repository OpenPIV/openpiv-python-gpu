"""This module is dedicated to advanced algorithms for PIV image analysis with NVIDIA GPU Support.

All identifiers ending with '_d' exist on the GPU and not the CPU. The GPU is referred to as the device, and therefore
"_d" signifies that it is a device variable. Please adhere to this standard as it makes developing and debugging much
easier. Note that all data must 32-bit at most to be stored on GPUs. Numpy types should be always 32-bit for
compatibility with GPU. Scalars should be python types in general to work as function arguments. The block-size
argument to GPU kernels should be multiples of 32 to avoid wasting GPU resources--e.g. (32, 1, 1), (8, 8, 1), etc.

"""
import logging
import warnings
from math import sqrt, ceil, log2, prod

import numpy as np
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
from pycuda.compiler import SourceModule

from openpiv.gpu_validation import ValidationGPU, ALLOWED_VALIDATION_METHODS, S2N_TOL, MEAN_TOL, MEDIAN_TOL, RMS_TOL
from openpiv.gpu_smoothn import gpu_smoothn
from openpiv.gpu_misc import _check_arrays, gpu_scalar_mod_i, gpu_remove_nan_f, gpu_remove_negative_f, gpu_mask

# Initialize the scikit-cuda library. This is necessary when certain cumisc calls happen that don't autoinit.
with warnings.catch_warnings():
    warnings.simplefilter('ignore', UserWarning)
    import skcuda.fft as cufft
    from skcuda import misc as cumisc
cumisc.init()

# Define 32-bit types.
DTYPE_i = np.int32
DTYPE_f = np.float32
DTYPE_c = np.complex64

ALLOWED_SUBPIXEL_METHODS = {'gaussian', 'parabolic', 'centroid'}
ALLOWED_S2N_METHODS = {'peak2peak', 'peak2mean', 'peak2energy'}
SMOOTHING_PAR = None
N_FFT = 2
SUBPIXEL_METHOD = 'gaussian'
S2N_METHOD = 'peak2peak'
S2N_WIDTH = 2
_BLOCK_SIZE = 64


class CorrelationGPU:
    """A class that performs the cross-correlation of interrogation windows.

    Can perform correlation by extended search area, where the first window is larger than the first window, allowing a
    for displacements larger than the nominal window size to be found.

    Parameters
    ----------
    frame_a_d, frame_b_d : GPUArray
        2D int (ht, wd), image pair.
    subpixel_method : {'gaussian', 'centroid', 'parabolic'}, optional
        Method to approximate the subpixel location of the peaks.

    Other Parameters
    ----------------
    n_fft : int or tuple
        (n_fft_x, n_fft_y), Window size multiplier for fft. Pass a tuple of length 2 for asymmetric multipliers.
    center_field : bool
        Whether to center the vector field on the image.
    s2n_method : str {'peak2peak', 'peak2mean', 'peak2energy'}, optional
        Method for evaluating the signal-to-noise ratio value from the correlation map.
    mask_width : int, optional
        Half size of the region around the first correlation peak to ignore for finding the second peak. Only used
        if 'sig2noise_method == peak2peak'.

    Attributes
    ----------
    sig2noise_d : GPUArray
        Signal-to-noise ratio of the cross-correlation.

    Methods
    -------
    __call__(window_size, extended_size=None, d_shift=None, d_strain=None)
        Returns the peaks of the correlation windows.
    free_frame_data()
        Clears names associated with frames.

    """

    def __init__(self, frame_a_d, frame_b_d, subpixel_method=SUBPIXEL_METHOD, **kwargs):
        self.frame_a_d = frame_a_d
        self.frame_b_d = frame_b_d
        self.frame_shape = frame_a_d.shape
        self.subpixel_method = subpixel_method
        assert subpixel_method in ALLOWED_SUBPIXEL_METHODS, \
            'subpixel_method is invalid. Must be one of {}.'.format(ALLOWED_SUBPIXEL_METHODS)

        self.n_fft = kwargs['n_fft'] if 'n_fft' in kwargs else N_FFT
        assert np.all(1 <= DTYPE_i(self.n_fft) == self.n_fft)
        if isinstance(self.n_fft, int):
            self.n_fft_x = self.n_fft_y = int(self.n_fft)
        else:
            self.n_fft_x = int(self.n_fft[0])
            self.n_fft_y = self.n_fft_x
            logging.info('For now, n_fft is the same in both directions. ({} is used here.)'.format(self.n_fft_x))
        self.center_field = kwargs['center_field'] if 'center_field' in kwargs else True
        self.s2n_width = kwargs['s2n_width'] if 's2n_width' in kwargs else S2N_WIDTH
        self.s2n_method = kwargs['s2n_method'] if 's2n_method' in kwargs else S2N_METHOD
        assert self.s2n_method in ALLOWED_S2N_METHODS, \
            'subpixel_method_method is invalid. Must be one of {}.'.format(ALLOWED_SUBPIXEL_METHODS)

    def __call__(self, piv_field, extended_size=None, shift_d=None, strain_d=None):
        """Returns the pixel peaks using the specified correlation method.

        Parameters
        ----------
        piv_field : PIVFieldGPU
            Geometric information for the correlation windows.
        extended_size : int or None, optional
            Extended window size to search in the second frame.
        shift_d : GPUArray or None, optional
            2D float ([du, dv]), du and dv are 1D arrays of the x-y shift at each interrogation window of the second
            frame. This is using the x-y convention of this code where x is the row and y is the column.
        strain_d : GPUArray or None, optional
            2D float ([u_x, u_y, v_x, v_y]), strain tensor.

        Returns
        -------
        i_peak, j_peak : ndarray
            3D float, locations of the subpixel peaks.

        """
        self.piv_field = piv_field
        assert piv_field.window_size >= 8 and piv_field.window_size % 8 == 0, 'Window size must be a multiple of 8.'
        self._extended_size = extended_size if extended_size is not None else piv_field.window_size
        assert (self._extended_size & (self._extended_size - 1)) == 0, 'Window size (extended) must be power of 2.'
        self._sig2noise_d = None

        self._init_fft_shape()

        # Get stack of all interrogation windows.
        win_a_d, win_b_d = self._stack_iw(self.frame_a_d, self.frame_b_d, shift_d, strain_d)

        # Correlate the windows.
        self.correlation_d = self._correlate_windows(win_a_d, win_b_d)

        # Get first peak of correlation.
        self.peak_idx_d = _find_peak(self.correlation_d)
        self.corr_peak1_d = _get_peak(self.correlation_d, self.peak_idx_d)

        # Get row and column of peak.
        # TODO Does storing these save time?
        self.row_peak_d, self.col_peak_d = gpu_scalar_mod_i(self.peak_idx_d, self.fft_wd)
        self._check_zero_correlation()

        # Get the subpixel location.z
        row_sp_d, col_sp_d = _gpu_subpixel_approximation(self.correlation_d, self.row_peak_d, self.col_peak_d,
                                                         self.subpixel_method)

        # Center the peak displacement.
        i_peak, j_peak = self._get_displacement(row_sp_d, col_sp_d)

        return i_peak, j_peak

    def free_frame_data(self):
        """Clears names associated with frames."""
        self.frame_a_d = None
        self.frame_b_d = None

    @property
    def sig2noise_d(self):
        return self._get_s2n()

    def _init_fft_shape(self):
        """Creates the shape of the fft windows padded up to power of 2 to boost speed."""
        self.fft_wd = 2 ** ceil(log2(self._extended_size * self.n_fft_x))
        self.fft_ht = 2 ** ceil(log2(self._extended_size * self.n_fft_y))
        self.fft_shape = (self.fft_ht, self.fft_wd)
        self.fft_size = self.fft_wd * self.fft_ht

    def _stack_iw(self, frame_a_d, frame_b_d, shift_d, strain_d=None):
        """Creates a 3D array stack of all the interrogation windows.

        This is necessary to do the FFTs all at once on the GPU. This populates interrogation windows from the origin
        of the image. The implementation requires that the window sizes are multiples of 4.

        Parameters
        -----------
        frame_a_d, frame_b_d : GPUArray
            2D int (ht, wd), image pair.
        shift_d : GPUArray
            3D float (2, m, n), ([du, dv]), shift of the second window.
        strain_d : GPUArray or None
            3D float (4, m, n) ([u_x, u_y, v_x, v_y]), strain rate tensor. First dimension is (u_x, u_y, v_x, v_y).

        Returns
        -------
        win_a_d, win_b_d : GPUArray
            3D float (n_windows, ht, wd), interrogation windows stacked in the first dimension.

        """
        _check_arrays(frame_a_d, frame_b_d, array_type=gpuarray.GPUArray, shape=frame_a_d.shape, dtype=DTYPE_f, ndim=2)

        # buffer_b shifts the centers of the extended windows to match the normal windows.
        buffer_a = -(self._extended_size - self.piv_field.window_size) // 2
        buffer_b = 0
        if self.center_field:
            center_buffer = self.piv_field.center_buffer
            buffer_a = center_buffer
            buffer_b = (center_buffer[0] + buffer_b, center_buffer[1] + buffer_b)

        win_a_d = _gpu_window_slice(frame_a_d, self.piv_field.shape, self.piv_field.window_size, self.piv_field.spacing,
                                    buffer_a, dt=-0.5, shift_d=shift_d, strain_d=strain_d)
        win_b_d = _gpu_window_slice(frame_b_d, self.piv_field.shape, self._extended_size, self.piv_field.spacing,
                                    buffer_b, dt=0.5, shift_d=shift_d, strain_d=strain_d)

        return win_a_d, win_b_d

    def _correlate_windows(self, win_a_d, win_b_d):
        """Computes the cross-correlation of the window stacks with zero-padding.

        Parameters
        ----------
        win_a_d, win_b_d : GPUArray
            3D float (n_windows, ht, wd), interrogation windows.

        Returns
        -------
        GPUArray
            3D (n_window, fft_ht, fft_wd), outputs of the correlation function.

        """
        _check_arrays(win_a_d, win_b_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=3)

        # Normalize array by computing the norm of each IW.
        win_a_norm_d = _gpu_normalize_intensity(win_a_d)
        win_b_norm_d = _gpu_normalize_intensity(win_b_d)

        # Zero pad arrays according to extended size requirements.
        extended_search_offset = (self._extended_size - self.piv_field.window_size) // 2
        win_a_zp_d = _gpu_zero_pad(win_a_norm_d, self.fft_shape, extended_search_offset=extended_search_offset)
        win_b_zp_d = _gpu_zero_pad(win_b_norm_d, self.fft_shape)

        # The second argument in the cross correlation remains stationary.
        corr_d = _gpu_cross_correlate(win_a_zp_d, win_b_zp_d)

        # Correlation is shifted so that the peak is not on near the boundary.
        corr_d = gpu_fft_shift(corr_d)

        return corr_d

    def _check_zero_correlation(self):
        """Sets the row and column to the center if the correlation peak is near zero."""
        center_d = gpuarray.ones_like(self.row_peak_d, dtype=DTYPE_i) * DTYPE_i(self.fft_wd // 2)
        self.row_peak_d = gpuarray.if_positive(self.corr_peak1_d, self.row_peak_d, center_d)
        self.col_peak_d = gpuarray.if_positive(self.corr_peak1_d, self.col_peak_d, center_d)

    def _get_displacement(self, row_sp_d, col_sp_d):
        """Returns the relative position of the peaks with respect to the center of the interrogation window."""
        i_peak = row_sp_d - DTYPE_f(self.fft_ht // 2)
        j_peak = col_sp_d - DTYPE_f(self.fft_wd // 2)

        return i_peak.reshape(self.piv_field.shape), j_peak.reshape(self.piv_field.shape)

    def _get_s2n(self):
        """Computes the signal-to-noise ratio using one of three available methods.

        The signal-to-noise ratio is computed from the correlation and is a measure of the quality of the matching
        between two interrogation windows. Note that this method returns the base-10 logarithm of the sig2noise ratio.
        The sig2noise field contains +np.Inf values where there is no noise.

        Returns
        -------
        ndarray
            2D float (m, n), the base-10 logarithm of the signal-to-noise ratio from the correlation map for each
            vector.

        """
        assert self.correlation_d is not None, 'Can only compute signal-to-noise ratio after correlation peaks' \
                                               'have been computed.'
        assert 0 <= self.s2n_width < int(min(self.fft_shape) / 2), \
            'Mask width must be integer from 0 and to less than half the correlation window height or width. ' \
            'Recommended value is 2.'

        if self._sig2noise_d is None:
            # Compute signal-to-noise ratio by the elected method.
            if self.s2n_method == 'peak2mean':
                sig2noise_d = _peak2mean(self.correlation_d, self.corr_peak1_d)
            elif self.s2n_method == 'peak2energy':
                sig2noise_d = _peak2energy(self.correlation_d, self.corr_peak1_d)
            else:
                corr_peak2_d = self._get_second_peak(self.correlation_d, self.s2n_width)
                sig2noise_d = _peak2peak(self.corr_peak1_d, corr_peak2_d)
            self._sig2noise_d = sig2noise_d.reshape(self.piv_field.shape)

        return self._sig2noise_d

    def _get_second_peak(self, correlation_positive_d, mask_width):
        """Find the value of the second-largest peak.

        The second-largest peak is the height of the peak in the region outside a width * width sub-matrix around
        the first correlation peak.

        Parameters
        ----------
        correlation_positive_d : GPUArray
            3D float (n_windows, fft_wd, fft_ht), correlation data with negative values removed.
        mask_width : int
            Half size of the region around the first correlation peak to ignore for finding the second peak.

        Returns
        -------
        GPUArray
            3D float (n_windows, fft_wd, fft_ht), value of the second correlation peak for each interrogation window.

        """
        assert self.row_peak_d is not None and self.col_peak_d is not None

        # Set points around the first peak to zero.
        correlation_masked_d = _gpu_mask_peak(correlation_positive_d, self.row_peak_d, self.col_peak_d, mask_width)

        # Get the height of the second peak of correlation.
        peak2_idx_d = _find_peak(correlation_masked_d)
        corr_max2_d = _get_peak(correlation_masked_d, peak2_idx_d)

        return corr_max2_d


class PIVFieldGPU:
    """Object storing geometric information of PIV windows.

    Parameters
    ----------
    frame_shape : tuple
        Int (ht, wd), shape of the piv frame.
    frame_mask : ndarray
        Int, (ht, wd), mask on the frame coordinates.
    window_size : int
        Size of the interrogation window.
    spacing : int
        Number of pixels between interrogation windows.

    Attributes
    ----------
    coords : tuple
        2D ndarray float (x, y), full coordinates of the PIV field.
    grid_coords_d : tuple
        1D ndarray float (x, y), vectors containing the grid coordinates of the PIV field.
    window_buffer : tuple
        1D int (buffer_x, buffer_y), offset in pixel units to the window positions to align them with the vector field.

    Methods
    -------
    get_mask(): GPUArray
        Returns field_mask if frame is mask, None otherwise.

    """

    def __init__(self, frame_shape, window_size, spacing, frame_mask=None, center_field=True, **kwargs):
        self.frame_shape = frame_shape
        self.window_size = window_size
        self.spacing = spacing
        self.shape = get_field_shape(frame_shape, window_size, spacing)
        self.size = prod(self.shape)
        self._x, self._y = get_field_coords(frame_shape, window_size, spacing, center_field=center_field)
        assert kwargs == {}

        self.is_masked = frame_mask is not None
        self.mask = _get_field_mask(self._x, self._y, frame_mask)
        self.mask_d = gpuarray.to_gpu(self.mask)

        self._x_grid_d = gpuarray.to_gpu(self._x[0, :].astype(DTYPE_f))
        self._y_grid_d = gpuarray.to_gpu(self._y[:, 0].astype(DTYPE_f))

    def get_mask(self):
        """Returns field_mask if frame is mask, None otherwise."""
        return self.mask_d if self.is_masked else None

    @property
    def coords(self):
        return self._x, self._y

    @property
    def grid_coords_d(self):
        return self._x_grid_d, self._y_grid_d

    @property
    def center_buffer(self):
        return _get_center_buffer(self.frame_shape, self.window_size, self.spacing)


def gpu_piv(frame_a, frame_b,
            mask=None,
            window_size_iters=(1, 2),
            min_window_size=16,
            overlap_ratio=0.5,
            dt=1,
            deform=True,
            smooth=True,
            nb_validation_iter=1,
            validation_method='median_velocity',
            **kwargs):
    """An iterative GPU-accelerated algorithm that uses translation and deformation of interrogation windows.

    At every iteration, the estimate of the displacement and gradient are used to shift and deform the interrogation
    windows used in the next iteration. One or more iterations can be performed before the estimated velocity is
    interpolated onto a finer mesh. This is done until the final mesh and number of iterations is met.

    Algorithm Details
    -----------------
    Only window sizes that are multiples of 8 are supported now, and the minimum window size is 8.
    Windows are shifted symmetrically to reduce bias errors.
    The displacement obtained after each correlation is the residual displacement dc.
    The new displacement is computed by dx = dpx + dcx and dy = dpy + dcy.
    Validation is done by any combination of signal-to-noise ratio, mean, median and rms velocities.
    Smoothn can be used between iterations to improve the estimate and replace missing values.

    References
    ----------
    Scarano F, Riethmuller ML (1999) Iterative multigrid approach in PIV image processing with discrete window offset.
        Exp Fluids 26:513–523
    Meunier, P., & Leweke, T. (2003). Analysis and treatment of errors due to high velocity gradients in particle image
        velocimetry.
        Experiments in fluids, 35(5), 408-421.
    Garcia, D. (2010). Robust smoothing of gridded data in one and higher dimensions with missing values.
        Computational statistics & data analysis, 54(4), 1167-1178.

    Parameters
    ----------
    frame_a, frame_b : ndarray
        2D int (ht, wd), grey levels of the first and second frames.
    mask : ndarray or None, optional
        2D int (ht, wd), array with values 0 for the background, 1 for the flow-field. If the center of a
        window is on a 0 value the velocity is set to 0.
    window_size_iters : tuple or int, optional
        Number of iterations performed at each window size. The length of window_size_iters gives the number of
        different windows sizes to use, while the value of each entry gives the number of times a window size is use.
    min_window_size : tuple or int, optional
        Length of the sides of the square interrogation window. Only supports multiples of 8.
    overlap_ratio : float, optional
        Ratio of overlap between two windows (between 0 and 1).
    dt : float, optional
        Time delay separating the two frames.
    deform : bool, optional
        Whether to deform the windows by the velocity gradient at each iteration.
    smooth : bool, optional
        Whether to smooth the intermediate fields.
    nb_validation_iter : int, optional
        Number of iterations per validation cycle.
    validation_method : str {tuple, 's2n', 'median_velocity', 'mean_velocity', 'rms_velocity'}, optional
        Method used for validation. Only the mean velocity method is implemented now. The default tolerance is 2 for
        median validation.

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
        2D int (m, n), boolean values (True for vectors interpolated from previous iteration).
    s2n : ndarray
        2D float (m, n), signal-to-noise ratio of the final velocity field.

    Other Parameters
    ----------------
    s2n_tol, median_tol, mean_tol, median_tol, rms_tol : float
        Tolerance of the validation methods.
    smoothing_par : float
        Smoothing parameter to pass to smoothn to apply to the intermediate velocity fields.
    extend_ratio : float
        Ratio the extended search area to use on the first iteration. If not specified, extended search will not be
        used.
    subpixel_method : str {'gaussian', 'centroid', 'parabolic'}
        Method to estimate subpixel location of the peak.
    return_sig2noise : bool
        Sets whether to return the signal-to-noise ratio. Not returning the signal-to-noise speeds up computation
        significantly, which is default behaviour.
    sig2noise_method : str {'peak2peak', 'peak2mean', 'peak2energy'}
        Method of signal-to-noise-ratio measurement.
    s2n_width : int
        Half size of the region around the first correlation peak to ignore for finding the second peak. Default is 2.
        Only used if sig2noise_method == 'peak2peak'.
    n_fft : int or tuple
        Size-factor of the 2D FFT in x and y-directions. The default of 2 is recommended.
    center_field : bool
        Whether to center the vector field on the image.

    Example
    -------
    x, y, u, v, mask, s2n = gpu_piv(frame_a, frame_b, mask=None, window_size_iters=(1, 2), min_window_size=16,
    overlap_ratio=0.5, dt=1, deform=True, smooth=True, nb_validation_iter=2, validation_method='median_velocity',
    median_tol=2)

    """
    piv_gpu = PIVGPU(frame_a.shape, window_size_iters, min_window_size, overlap_ratio, dt, mask, deform, smooth,
                     nb_validation_iter, validation_method, **kwargs)

    return_sig2noise = kwargs['return_sig2noise'] if 'return_sig2noise' in kwargs else False
    x, y = piv_gpu.coords
    u, v = piv_gpu(frame_a, frame_b)
    mask = piv_gpu.mask
    s2n = piv_gpu.s2n if return_sig2noise else None
    return x, y, u, v, mask, s2n


class PIVGPU:
    """This class is the object-oriented implementation of the GPU PIV function.

    Parameters
    ----------
    frame_shape : ndarray or tuple
        Int (ht, wd), size of the images in pixels.
    window_size_iters : tuple or int, optional
        Number of iterations performed at each window size. The length of window_size_iters gives the number of
        different windows sizes to use, while the value of each entry gives the number of times a window size is use.
    min_window_size : tuple or int, optional
        Length of the sides of the square interrogation window. Only supports multiples of 8.
    overlap_ratio : float, optional
        Ratio of overlap between two windows (between 0 and 1).
    dt : float, optional
        Time delay separating the two frames.
    mask : ndarray or None, optional
        2D, float, array with values 0 for the background, 1 for the flow-field. If the center of a window
        is on a 0 value the velocity is set to 0.
    deform : bool, optional
        Whether to deform the windows by the velocity gradient at each iteration.
    smooth : bool, optional
        Whether to smooth the intermediate fields.
    nb_validation_iter : int, optional
        Number of iterations per validation cycle.
    validation_method : str {tuple, 's2n', 'median_velocity', 'mean_velocity', 'rms_velocity'}, optional
        Method(s) to use for validation.

    Other Parameters
    ----------------
    s2n_tol, median_tol, mean_tol, median_tol, rms_tol : float
        Tolerance of the validation methods.
    smoothing_par : float
        Smoothing parameter to pass to smoothn to apply to the intermediate velocity fields. Default is 0.5.
    extend_ratio : float
        Ratio the extended search area to use on the first iteration. If not specified, extended search will not be
        used.
    subpixel_method : str {'gaussian', 'centroid', 'parabolic'}
        Method to estimate subpixel location of the peak.
    sig2noise_method : str {'peak2peak', 'peak2mean', 'peak2energy'}
        Method of signal-to-noise-ratio measurement.
    sig2noise_width : int
        Half size of the region around the first correlation peak to ignore for finding the second peak. Default is 2.
        Only used if sig2noise_method == 'peak2peak'.
    n_fft : int or tuple
        Size-factor of the 2D FFT in x and y-directions. The default of 2 is recommended.
    center_field : bool
        Whether to center the vector field on the image.

    Attributes
    ----------
    coords : tuple
        2D ndarray (x, y), coordinates where the velocity field has been computed.
    mask : ndarray
        2D float (m, n), boolean values (True for vectors interpolated from previous iteration).
    s2n : ndarray
        2D float (m, n), signal-to-noise ratio of the final velocity field.

    Methods
    -------
    __call__(frame_a, frame_b)
        Main method to process image pairs.

    """

    def __init__(self,
                 frame_shape,
                 window_size_iters=(1, 2),
                 min_window_size=16,
                 overlap_ratio=0.5,
                 dt=1,
                 mask=None,
                 deform=True,
                 smooth=True,
                 nb_validation_iter=1,
                 validation_method='median_velocity',
                 **kwargs):
        self.frame_shape = frame_shape.shape if hasattr(frame_shape, 'shape') else tuple(frame_shape)
        self.min_window_size = min_window_size
        self.ws_iters = (window_size_iters,) if isinstance(window_size_iters, int) else tuple(window_size_iters)
        self.overlap_ratio = float(overlap_ratio)
        self.dt = dt
        self.frame_mask = mask.astype(DTYPE_i) if mask is not None else None
        self.deform = deform
        self.smooth = smooth
        self.nb_validation_iter = nb_validation_iter
        self.validation_method = (validation_method,) if isinstance(validation_method, str) else validation_method

        self.extend_ratio = kwargs['extend_ratio'] if 'extend_ratio' in kwargs else None
        self.s2n_tol = kwargs['s2n_tol'] if 's2n_tol' in kwargs else S2N_TOL
        self.median_tol = kwargs['median_tol'] if 'median_tol' in kwargs else MEDIAN_TOL
        self.mean_tol = kwargs['mean_tol'] if 'mean_tol' in kwargs else MEAN_TOL
        self.rms_tol = kwargs['rms_tol'] if 'rms_tol' in kwargs else RMS_TOL
        self.smoothing_par = kwargs['smoothing_par'] if 'smoothing_par' in kwargs else SMOOTHING_PAR
        self.n_fft = kwargs['n_fft'] if 'n_fft' in kwargs else N_FFT
        self.subpixel_method = kwargs['subpixel_method'] if 'subpixel_method' in kwargs else SUBPIXEL_METHOD
        self.s2n_method = kwargs['sig2noise_method'] if 'sig2noise_method' in kwargs else S2N_METHOD
        self.s2n_width = kwargs['sig2noise_width'] if 'sig2noise_width' in kwargs else S2N_WIDTH
        self.center_field = kwargs['center_field'] if 'center_field' in kwargs else True

        self._nb_iter = sum(self.ws_iters)
        self._corr = None
        self._im_mask_d = gpuarray.to_gpu(self.frame_mask) if mask is not None else None

        self._check_inputs()
        self._init_fields()

    def __call__(self, frame_a, frame_b):
        """Processes an image pair.

        Parameters
        ----------
        frame_a, frame_b : ndarray
            2D int (ht, wd), grey levels of the first and second frames.

        Returns
        -------
        u, v : ndarray
            2D float (m, n), horizontal/vertical components of velocity in pixels/time units.

        """
        _check_arrays(frame_a, frame_b, array_type=np.ndarray, ndim=2)
        u_d = v_d = None
        u_previous_d = v_previous_d = None
        dp_u_d = dp_v_d = None

        # Send masked frames to device.
        frame_a_d, frame_b_d = self._mask_frame(frame_a, frame_b)

        # Create the correlation object.
        self._corr = CorrelationGPU(frame_a_d, frame_b_d, n_fft=self.n_fft, subpixel_method=self.subpixel_method,
                                    center_field=self.center_field, s2n_method=self.s2n_method,
                                    s2n_width=self.s2n_width)

        # MAIN LOOP
        for k in range(self._nb_iter):
            self._k = k
            self._piv_field_k = self._piv_fields[k]
            logging.info('ITERATION {}'.format(k))

            # CROSS-CORRELATION
            # Get arguments for the correlation class.
            extended_size = self._get_extended_size()
            shift_d, strain_d = self._get_window_deformation(dp_u_d, dp_v_d)

            # Get window displacement to subpixel accuracy.
            i_peak_d, j_peak_d = self._corr(self._piv_fields[k], extended_size=extended_size, shift_d=shift_d,
                                            strain_d=strain_d)

            # Update the field with new values.
            u_d, v_d = self._update_values(i_peak_d, j_peak_d, dp_u_d, dp_v_d)
            self._log_residual(i_peak_d, j_peak_d)

            # VALIDATION
            u_d, v_d = self._validate_fields(u_d, v_d, u_previous_d, v_previous_d)

            # NEXT ITERATION
            # Compute the predictors dpx and dpy from the current displacements.
            if k < self._nb_iter - 1:
                u_previous_d = u_d
                v_previous_d = v_d
                dp_u_d, dp_v_d = self._get_next_iteration_predictions(u_d, v_d)

        u = (u_d / DTYPE_f(self.dt)).get()
        v = (v_d / DTYPE_f(-self.dt)).get()

        self._corr.free_frame_data()

        return u, v

    @property
    def coords(self):
        return self._piv_fields[-1].coords

    @property
    def mask(self):
        return self._piv_fields[-1].mask

    @property
    def s2n(self):
        return self._corr.sig2noise_d

    def free_data(self):
        """Frees correlation data from GPU."""
        self._corr = None

    def _init_fields(self):
        """Creates piv-field object at each iteration."""
        self._piv_fields = []
        for window_size in _get_window_sizes(self.ws_iters, self.min_window_size):
            window_size = window_size
            spacing = _get_spacing(window_size, self.overlap_ratio)
            self._piv_fields.append(PIVFieldGPU(self.frame_shape, window_size, spacing, frame_mask=self.frame_mask,
                                                center_field=self.center_field))

    def _mask_frame(self, frame_a, frame_b):
        """Mask the frames before sending to device."""
        _check_arrays(frame_a, frame_b, array_type=np.ndarray, shape=frame_a.shape, ndim=2)

        if self.frame_mask is not None:
            frame_a_d = gpu_mask(gpuarray.to_gpu(frame_a.astype(DTYPE_f)), self._im_mask_d)
            frame_b_d = gpu_mask(gpuarray.to_gpu(frame_b.astype(DTYPE_f)), self._im_mask_d)
        else:
            frame_a_d = gpuarray.to_gpu(frame_a.astype(DTYPE_f))
            frame_b_d = gpuarray.to_gpu(frame_b.astype(DTYPE_f))

        return frame_a_d, frame_b_d

    def _get_extended_size(self):
        """Returns the extended size used during the first iteration."""
        extended_size = None
        if self._k == 0 and self.extend_ratio is not None:
            extended_size = int(self._piv_field_k.window_size * self.extend_ratio)

        return extended_size

    def _get_window_deformation(self, dp_u_d, dp_v_d):
        """Returns the shift and strain arguments to the correlation class."""
        mask_d = self._piv_field_k.mask_d
        shift_d = None
        strain_d = None

        if self._k > 0:
            shift_d = _get_shift(dp_u_d, dp_v_d)
            if self.deform:
                strain_d = gpu_strain(dp_u_d, dp_v_d, mask_d, self._piv_field_k.spacing)

        return shift_d, strain_d

    def _update_values(self, i_peak_d, j_peak_d, dp_x_d, dp_y_d):
        """Updates the velocity values after each iteration."""
        mask_d = self._piv_field_k.mask_d

        if dp_x_d is None:
            u_d = gpu_mask(j_peak_d, mask_d)
            v_d = gpu_mask(i_peak_d, mask_d)
        else:
            u_d = _gpu_update_field(dp_x_d, j_peak_d, mask_d)
            v_d = _gpu_update_field(dp_y_d, i_peak_d, mask_d)

        return u_d, v_d

    def _validate_fields(self, u_d, v_d, u_previous_d, v_previous_d):
        """Return velocity fields with outliers removed."""
        size = u_d.size
        val_locations_d = None
        mask_d = self._piv_field_k.get_mask()
        # Retrieve signal-to-noise ratio only if required for validation.
        sig2noise_d = None
        if 's2n' in self.validation_method and self.nb_validation_iter > 0:
            sig2noise_d = self._corr.sig2noise_d

        # Do the validation.
        validation_gpu = ValidationGPU(u_d.shape, mask_d=mask_d, validation_method=self.validation_method,
                                       s2n_tol=self.s2n_tol, median_tol=self.median_tol,
                                       mean_tol=self.mean_tol, rms_tol=self.rms_tol)
        for i in range(self.nb_validation_iter):
            val_locations_d = validation_gpu(u_d, v_d, sig2noise_d=sig2noise_d)
            u_mean_d, v_mean_d = validation_gpu.median_d

            # Replace invalid vectors.
            n_val = int(gpuarray.sum(val_locations_d).get())
            if n_val > 0:
                logging.info('Validating {} out of {} vectors ({:.2%}).'.format(n_val, size, n_val / size))
                u_d, v_d = self._gpu_replace_vectors(u_d, v_d, u_previous_d, v_previous_d, u_mean_d, v_mean_d,
                                                     val_locations_d)
            else:
                logging.info('No invalid vectors.')
                break

            validation_gpu.free_data()

        # Smooth the validated field.
        if self.smooth:
            w_d = (1 - val_locations_d) if val_locations_d is not None else None
            u_d, v_d = gpu_smoothn(u_d, v_d, s=self.smoothing_par, mask=mask_d, w=w_d)

        return u_d, v_d

    def _gpu_replace_vectors(self, u_d, v_d, u_previous_d, v_previous_d, u_mean_d, v_mean_d, val_locations_d):
        """Replace spurious vectors by the mean or median of the surrounding points."""
        _check_arrays(u_d, v_d, u_mean_d, v_mean_d, val_locations_d, array_type=gpuarray.GPUArray, shape=u_d.shape)

        # First iteration, just replace with mean velocity.
        if self._k == 0:
            u_d = gpuarray.if_positive(val_locations_d, u_mean_d, u_d)
            v_d = gpuarray.if_positive(val_locations_d, v_mean_d, v_d)

        # Case if different dimensions: interpolation using previous iteration.
        elif self._k > 0 and self._piv_field_k.shape != self._piv_fields[self._k - 1].shape:
            x0_d, y0_d = self._piv_fields[self._k - 1].grid_coords_d
            x1_d, y1_d = self._piv_fields[self._k].grid_coords_d
            mask_d = self._piv_fields[self._k - 1].get_mask()

            u_d = _interpolate_replace(x0_d, y0_d, x1_d, y1_d, u_previous_d, u_d, val_locations_d, mask_d=mask_d)
            v_d = _interpolate_replace(x0_d, y0_d, x1_d, y1_d, v_previous_d, v_d, val_locations_d, mask_d=mask_d)

        # Case if same dimensions.
        elif self._k > 0 and self._piv_field_k.shape == self._piv_fields[self._k - 1].shape:
            u_d = gpuarray.if_positive(val_locations_d, u_previous_d, u_d)
            v_d = gpuarray.if_positive(val_locations_d, v_previous_d, v_d)

        return u_d, v_d

    def _get_next_iteration_predictions(self, u_d, v_d):
        """Returns the velocity field to begin the next iteration."""
        _check_arrays(u_d, v_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, shape=u_d.shape, ndim=2)
        x0_d, y0_d = self._piv_field_k.grid_coords_d
        x1_d, y1_d = self._piv_fields[self._k + 1].grid_coords_d
        mask_d = self._piv_field_k.get_mask()

        # Interpolate if dimensions do not agree.
        if self._piv_fields[self._k + 1].window_size != self._piv_field_k.window_size:
            dp_u_d = gpu_interpolate(x0_d, y0_d, x1_d, y1_d, u_d, mask_d=mask_d)
            dp_v_d = gpu_interpolate(x0_d, y0_d, x1_d, y1_d, v_d, mask_d=mask_d)
        else:
            dp_u_d = u_d
            dp_v_d = v_d

        return dp_u_d, dp_v_d

    def _log_residual(self, i_peak_d, j_peak_d):
        """Normalizes the residual by the maximum quantization error of 0.5 pixel."""
        _check_arrays(i_peak_d, j_peak_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, shape=i_peak_d.shape)

        try:
            normalized_residual = sqrt(int(gpuarray.sum(i_peak_d ** 2 + j_peak_d ** 2).get()) / i_peak_d.size) / 0.5
            logging.info('Normalized residual : {}.'.format(normalized_residual))
        except OverflowError:
            logging.warning('Overflow in residuals.')
            normalized_residual = np.nan

        self.normalized_residual = normalized_residual

    def _check_inputs(self):
        if int(self.frame_shape[0]) != self.frame_shape[0] or int(self.frame_shape[1]) != self.frame_shape[1]:
            raise TypeError('frame_shape must be either tuple of integers or array-like.')
        if len(self.frame_shape) != 2:
            raise ValueError('frame_shape must be 2D.')
        if not all([1 <= ws == int(ws) for ws in self.ws_iters]):
            raise ValueError('Window sizes must be integers greater than or equal to 1.')
        if not self._nb_iter >= 1:
            raise ValueError('Sum of window_size_iters must be equal to or greater than 1.')
        if not 0 < self.overlap_ratio < 1:
            raise ValueError('overlap ratio must be between 0 and 1.')
        if self.dt != float(self.dt):
            raise ValueError('dt must be a number.')
        if self.frame_mask is not None:
            if self.frame_mask.shape != self.frame_shape:
                raise ValueError('mask is not same shape as frame.')
        if self.deform != bool(self.deform):
            raise ValueError('deform must have a boolean value.')
        if self.smooth != bool(self.smooth):
            raise ValueError('smooth must have a boolean value.')
        if not 0 <= self.nb_validation_iter == int(self.nb_validation_iter):
            raise ValueError('nb_validation_iter must be 0 or a positive integer.')
        if not all([method in ALLOWED_VALIDATION_METHODS for method in self.validation_method]):
            raise ValueError('validation_method is not allowed. Allowed are: {}'.format(ALLOWED_VALIDATION_METHODS))
        if self.extend_ratio is not None:
            if not 1 < self.extend_ratio == float(self.extend_ratio):
                raise ValueError('extend_ratio must be a number greater than unity.')
        if not all(0 < tol == float(tol) or tol is None for tol in
                   [self.s2n_tol, self.median_tol, self.mean_tol, self.rms_tol]):
            raise ValueError('Validation tolerances must be positive numbers.')
        if not 1 < self.n_fft == float(self.n_fft):
            raise ValueError('n_fft must be an number equal to or greater than 1.')
        if self.s2n_method not in ALLOWED_S2N_METHODS:
            raise ValueError('sig2noise_method is not allowed. Allowed is one of: {}'.format(ALLOWED_S2N_METHODS))
        if self.subpixel_method not in ALLOWED_SUBPIXEL_METHODS:
            raise ValueError('subpixel_method is not allowed. Allowed is one of: {}'.format(ALLOWED_SUBPIXEL_METHODS))
        if not 1 < self.s2n_width == int(self.s2n_width):
            raise ValueError('s2n_width must be an integer.')
        if self.center_field != bool(self.center_field):
            raise ValueError('center_field must have a boolean value.')


def get_field_shape(frame_shape, window_size, spacing):
    """Returns the shape of the resulting velocity field.

    Parameters
    ----------
    frame_shape : tuple
        Int (ht, wd), size of the frame in pixels.
    window_size : int
        Size of the interrogation windows.
    spacing : int
        Spacing between vectors in the resulting field, in pixels.

    Returns
    -------
    tuple
        Int (m, n), shape of the resulting flow field.

    """
    assert len(frame_shape) == 2, 'frame_shape must have length 2.'
    assert int(spacing) == spacing > 0, 'spacing must be a positive int.'
    ht, wd = frame_shape

    m = int((ht - window_size) // spacing) + 1
    n = int((wd - window_size) // spacing) + 1

    return m, n


def get_field_coords(frame_shape, window_size, spacing, center_field=True):
    """Returns the coordinates of the resulting velocity field.

    Parameters
    ----------
    frame_shape : tuple
        Int (ht, wd), size of the frame in pixels.
    window_size : int
        Size of the interrogation windows.
    spacing : int
        Spacing between vectors in the resulting field, in pixels.
    center_field : bool, optional
        Whether the coordinates of the interrogation windows are centered on the image.

    Returns
    -------
    x, y : ndarray
        2D float (m, n), pixel coordinates of the resulting flow field.

    """
    assert len(frame_shape) == 2, 'frame_shape must have length 2.'
    assert int(spacing) == spacing > 0, 'spacing must be a positive int.'

    m, n = get_field_shape(frame_shape, window_size, spacing)
    half_width = window_size // 2
    buffer_x = 0
    buffer_y = 0
    if center_field:
        buffer_x, buffer_y = _get_center_buffer(frame_shape, window_size, spacing)
    x = np.tile(np.linspace(half_width + buffer_x, half_width + buffer_x + spacing * (n - 1), n), (m, 1))
    y = np.tile(np.linspace(half_width + buffer_y + spacing * (m - 1), half_width + buffer_y, m), (n, 1)).T

    return x, y


mod_strain = SourceModule("""
__global__ void strain_gpu(float *strain, float *u, float *v, int *mask, float h, int m, int n, int size)
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
    int interior = ((row > 0) * (row < m - 1) || !gradient_axis) * ((col > 0) * (col < n - 1) || gradient_axis);

    // Get the indexes of the neighbouring points.
    int idx0 = idx - (row > 0) * (gradient_axis) * n - (col > 0) * !gradient_axis;
    int idx1 = idx + (row < m - 1) * (gradient_axis) * n + (col < n - 1) * !gradient_axis;

    // Revert to first order differencing where field is masked.
    interior = interior * !mask[idx0] * !mask[idx1];
    idx0 = idx0 * !mask[idx0] + idx * mask[idx0];
    idx1 = idx1 * !mask[idx1] + idx * mask[idx1];

    // Do the differencing.
    strain[size * gradient_axis + idx] = (u[idx1] - u[idx0]) / (1 + interior) / h;
    strain[size * (gradient_axis + 2) + idx] = (v[idx1] - v[idx0]) / (1 + interior) / h;
}
""")


def gpu_strain(u_d, v_d, mask_d=None, spacing=1):
    """Computes the full 2D strain rate tensor.

    Parameters
    ----------
    u_d, v_d : GPUArray
        2D float, velocity fields.
    mask_d : GPUArray, optional
        Mask for the vector field.
    spacing : float, optional
        Spacing between nodes.

    Returns
    -------
    GPUArray
        3D float (4, m, n) [(u_x, u_y, v_x, v_y)], full strain tensor of the velocity fields.

    """
    _check_arrays(u_d, v_d, array_type=gpuarray.GPUArray, shape=u_d.shape, dtype=DTYPE_f)
    assert spacing > 0, 'Spacing must be greater than 0.'
    m, n = u_d.shape
    size = u_d.size
    if mask_d is not None:
        _check_arrays(mask_d, array_type=gpuarray.GPUArray, shape=u_d.shape, dtype=DTYPE_i)
    else:
        mask_d = gpuarray.zeros_like(u_d, dtype=DTYPE_i)

    strain_d = gpuarray.empty((4, m, n), dtype=DTYPE_f)

    block_size = _BLOCK_SIZE
    n_blocks = ceil(size * 2 / block_size)
    strain_gpu = mod_strain.get_function('strain_gpu')
    strain_gpu(strain_d, u_d, v_d, mask_d, DTYPE_f(spacing), DTYPE_i(m), DTYPE_i(n), DTYPE_i(size),
               block=(block_size, 1, 1), grid=(n_blocks, 1))

    return strain_d


mod_fft_shift = SourceModule("""
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
""")


def gpu_fft_shift(correlation_d):
    """Returns the shifted spectrum of stacked fft output.

    Parameters
    ----------
    correlation_d : GPUArray
        3D float (n_windows, ht, wd), data from fft.

    Returns
    -------
    GPUArray
        3D float (n_windows, ht, wd), shifted data from fft.

    """
    _check_arrays(correlation_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=3)
    n_windows, ht, wd = correlation_d.shape
    window_size = ht * wd

    correlation_shift_d = gpuarray.empty_like(correlation_d, dtype=DTYPE_f)

    block_size = 8
    grid_size = ceil(max(ht, wd) / block_size)
    fft_shift = mod_fft_shift.get_function('fft_shift')
    fft_shift(correlation_shift_d, correlation_d, DTYPE_i(ht), DTYPE_i(wd), DTYPE_i(window_size),
              block=(block_size, block_size, 1), grid=(n_windows, grid_size, grid_size))

    return correlation_shift_d


mod_interpolate = SourceModule("""
__global__ void bilinear_interpolation(float *f1, float *f0, float *x_grid, float *y_grid, float buffer_x,
                    float buffer_y, float spacing_x, float spacing_y, int ht, int wd, int n, int size)
{
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= size) {return;}

    // Map indices to old mesh coordinates.
    int x_idx = t_idx % n;
    int y_idx = t_idx / n;
    float x = (x_grid[x_idx] - buffer_x) / spacing_x;
    float y = (y_grid[y_idx] - buffer_y) / spacing_y;

    // Coerce interpolation point to within limits of domain.
    x = x * (x >= 0.0f && x <= wd - 1) + (wd - 1) * (x > wd - 1);
    y = y * (y >= 0.0f && y <= ht - 1) + (ht - 1) * (y > ht - 1);

    // Get neighbouring points.
    int x1 = floorf(x) - (x == wd - 1);
    int x2 = x1 + 1;
    int y1 = floorf(y) - (y == ht - 1);
    int y2 = y1 + 1;

    // Apply the mapping.
    f1[t_idx] = (x2 - x) * (y2 - y) * f0[y1 * wd + x1]  // f11
                + (x - x1) * (y2 - y) * f0[y1 * wd + x2]  // f21
                + (x2 - x) * (y - y1) * f0[y2 * wd + x1]  // f12
                + (x - x1) * (y - y1) * f0[y2 * wd + x2];  // f22
}

__global__ void bilinear_interpolation_mask(float *f1, float *f0, float *x_grid, float *y_grid, int *mask,
                    float buffer_x, float buffer_y, float spacing_x, float spacing_y, int ht, int wd, int n, int size)
{
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= size) {return;}

    // Map indices to old mesh coordinates.
    int x_idx = t_idx % n;
    int y_idx = t_idx / n;
    float x = (x_grid[x_idx] - buffer_x) / spacing_x;
    float y = (y_grid[y_idx] - buffer_y) / spacing_y;

    // Coerce interpolation point to within limits of domain.
    x = x * (x >= 0.0f && x <= wd - 1) + (wd - 1) * (x > wd - 1);
    y = y * (y >= 0.0f && y <= ht - 1) + (ht - 1) * (y > ht - 1);

    // Get neighbouring points.
    int x1 = floorf(x) - (x == wd - 1);
    int x2 = x1 + 1;
    int y1 = floorf(y) - (y == ht - 1);
    int y2 = y1 + 1;

    // Get masked values.
    int m11 = mask[y1 * wd + x1];
    int m21 = mask[y1 * wd + x2];
    int m12 = mask[y2 * wd + x1];
    int m22 = mask[y2 * wd + x2];
    int m_y1 = m11 * m21;
    int m_y2 = m12 * m22;

    // Apply the mapping along x-axis.
    float f_y1 = ((x2 - x) * (!m11 * !m21) + (!m11 * m21)) * f0[y1 * wd + x1]  // f11
                 + ((x - x1) * (!m11 * !m21) + (m11 * !m21)) * f0[y1 * wd + x2]; // f21
    float f_y2 = ((x2 - x) * (!m12 * !m22) + (!m12 * m22)) * f0[y2 * wd + x1] // f12
                 + ((x - x1) * (!m12 * !m22) + (m12 * !m22)) * f0[y2 * wd + x2]; // f22

    // Apply the mapping along y-axis.
    f1[t_idx] = ((y2 - y) * (!m_y1 * !m_y2) + (!m_y1 * m_y2)) * f_y1
                + ((y - y1) * (!m_y1 * !m_y2) + (m_y1 * !m_y2)) * f_y2;
}
""")


def gpu_interpolate(x0_d, y0_d, x1_d, y1_d, f0_d, mask_d=None):
    """Performs an interpolation of a field from one mesh to another.

    The implementation requires that the mesh spacing is uniform. The spacing can be different in x and y directions.

    Parameters
    ----------
    x0_d, y0_d : GPUArray
        1D float, grid coordinates of the original field
    x1_d, y1_d : GPUArray
        1D float, grid coordinates of the field to be interpolated.
    f0_d : GPUArray
        2D float (y0_d.size, x0_d.size), field to be interpolated.
    mask_d : (y0_d.size, x0_d.size): GPUArray, optional
        2D float, value of one where masked values are.

    Returns
    -------
    GPUArray
        2D float (x1_d.size, y1_d.size), interpolated field.

    """
    _check_arrays(x0_d, y0_d, x1_d, y1_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=1)
    ht = y0_d.size
    wd = x0_d.size
    _check_arrays(f0_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=2, shape=(ht, wd))
    n = x1_d.size
    m = y1_d.size
    size = m * n

    f1_d = gpuarray.empty((m, n), dtype=DTYPE_f)

    # Calculate the relationship between the two grid coordinates.
    buffer_x_f = DTYPE_f(x0_d[0].get())
    buffer_y_f = DTYPE_f(y0_d[0].get())
    spacing_x_f = DTYPE_f((x0_d[1].get() - buffer_x_f))
    spacing_y_f = DTYPE_f((y0_d[1].get() - buffer_y_f))

    block_size = _BLOCK_SIZE
    grid_size = ceil(size / block_size)
    if mask_d is not None:
        _check_arrays(mask_d, array_type=gpuarray.GPUArray, dtype=DTYPE_i, shape=f0_d.shape)
        interpolate_gpu = mod_interpolate.get_function('bilinear_interpolation_mask')
        interpolate_gpu(f1_d, f0_d, x1_d, y1_d, mask_d, buffer_x_f, buffer_y_f, spacing_x_f, spacing_y_f, DTYPE_i(ht),
                        DTYPE_i(wd), DTYPE_i(n), DTYPE_i(size), block=(block_size, 1, 1), grid=(grid_size, 1))
    else:
        interpolate_gpu = mod_interpolate.get_function('bilinear_interpolation')
        interpolate_gpu(f1_d, f0_d, x1_d, y1_d, buffer_x_f, buffer_y_f, spacing_x_f, spacing_y_f, DTYPE_i(ht),
                        DTYPE_i(wd), DTYPE_i(n), DTYPE_i(size), block=(block_size, 1, 1), grid=(grid_size, 1))

    return f1_d


def _get_window_sizes(ws_iters, min_window_size):
    """Returns the window size at each iteration."""
    for i, ws in enumerate(ws_iters):
        for _ in range(ws):
            yield (2 ** (len(ws_iters) - i - 1)) * min_window_size


def _get_spacing(window_size, overlap_ratio):
    """Returns spacing from window size and overlap ratio."""
    return max(1, int(window_size * (1 - overlap_ratio)))


def _get_field_mask(x, y, frame_mask=None):
    """Creates field mask from frame mask."""
    if frame_mask is not None:
        mask = frame_mask[y.astype(DTYPE_i), x.astype(DTYPE_i)]
    else:
        mask = np.zeros_like(x, dtype=DTYPE_i)

    return mask


def _get_center_buffer(frame_shape, window_size, spacing):
    """Returns the left pad to indexes to center the vector-field coordinates on the frame.

    This accounts for non-even windows sizes and field dimensions to make the buffers on either side of the field differ
    by one pixel at most.

    """
    ht, wd = frame_shape
    m, n = get_field_shape(frame_shape, window_size, spacing)

    buffer_x = (wd - (spacing * (n - 1) + window_size)) // 2 + window_size % 2
    buffer_y = (ht - (spacing * (m - 1) + window_size)) // 2 + window_size % 2

    return buffer_x, buffer_y


mod_window_slice = SourceModule("""
__global__ void window_slice(float *output, float *input, int ws, int spacing, int buffer_x, int buffer_y, int n,
                    int wd, int ht)
{
    // x blocks are windows; y and z blocks are x and y dimensions, respectively.
    int idx_i = blockIdx.x;
    int idx_x = blockIdx.y * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.z * blockDim.y + threadIdx.y;
    if (idx_x >= ws || idx_y >= ws) {return;}

    // Do the mapping.
    int x = (idx_i % n) * spacing + buffer_x + idx_x;
    int y = (idx_i / n) * spacing + buffer_y + idx_y;

    // Indices of new array to map to.
    int w_range = idx_i * ws * ws + ws * idx_y + idx_x;

    // Find limits of domain.
    int inside_domain = (x >= 0 && x < wd && y >= 0 && y < ht);

    if (inside_domain) {
    // Apply the mapping.
    output[w_range] = input[(y * wd + x)];
    } else {output[w_range] = 0;}
}

__global__ void window_slice_deform(float *output, float *input, float *shift, float *strain, float dt, int deform,
                    int ws, int spacing, int buffer_x, int buffer_y, int n_windows, int n, int wd, int ht)
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
    float x = (idx_i % n) * spacing + buffer_x + idx_x + dx;
    float y = (idx_i / n) * spacing + buffer_y + idx_y + dy;

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
""")


def _gpu_window_slice(frame_d, field_shape, window_size, spacing, buffer, dt=0, shift_d=None, strain_d=None):
    """Creates a 3D array stack of all the interrogation windows using shift and strain.

    Parameters
    -----------
    frame_d : GPUArray
        2D int (ht, wd), frame form which to create windows.
    field_shape : tuple
        Int (m, n), shape of the vector field.
    window_size : int
        Side dimension of the square interrogation windows.
    spacing : int
        Spacing between vectors of the velocity field.
    buffer : int or tuple
        (buffer_x, buffer_y), adjustment to location of windows from left/top vectors to edge of frame.
    dt : float, optional
        Number between -1 and 1 indicating the level of shifting/deform. E.g. 1 indicates shift by full amount, 0 is
        stationary. This is applied to the deformation in an analogous way.
    shift_d : GPUArray, optional
        3D float (2, m, n) ([du, dv]), shift of the second window.
    strain_d : GPUArray, optional
        3D float (4, m, n) ([u_x, u_y, v_x, v_y]), strain rate tensor.

    Returns
    -------
    GPUArray
        3D float (n_windows, ht, wd), interrogation windows stacked in the first dimension.

    """
    _check_arrays(frame_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=2)
    assert len(field_shape) == 2
    assert -1 <= dt <= 1
    assert np.all(buffer == DTYPE_i(buffer))
    if isinstance(buffer, int):
        buffer_x_i = buffer_y_i = DTYPE_i(buffer)
    else:
        buffer_x_i, buffer_y_i = DTYPE_i(buffer)
    ht, wd = frame_d.shape
    m, n = field_shape
    n_windows = m * n

    win_d = gpuarray.empty((n_windows, window_size, window_size), dtype=DTYPE_f)

    block_size = 8
    grid_size = ceil(window_size / block_size)
    if shift_d is not None:
        _check_arrays(shift_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, shape=(2, m, n))
        do_deform = DTYPE_i(strain_d is not None)
        if not do_deform:
            strain_d = gpuarray.zeros(1, dtype=DTYPE_i)
        else:
            _check_arrays(strain_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=3)
        window_slice = mod_window_slice.get_function('window_slice_deform')
        window_slice(win_d, frame_d, shift_d, strain_d, DTYPE_f(dt), do_deform, DTYPE_i(window_size),
                     DTYPE_i(spacing), buffer_x_i, buffer_y_i, DTYPE_i(n_windows), DTYPE_i(n), DTYPE_i(wd),
                     DTYPE_i(ht), block=(block_size, block_size, 1), grid=(int(n_windows), grid_size, grid_size))
    else:
        window_slice = mod_window_slice.get_function('window_slice')
        window_slice(win_d, frame_d, DTYPE_i(window_size), DTYPE_i(spacing), buffer_x_i, buffer_y_i, DTYPE_i(n),
                     DTYPE_i(wd), DTYPE_i(ht), block=(block_size, block_size, 1),
                     grid=(int(n_windows), grid_size, grid_size))

    return win_d


mod_norm = SourceModule("""
__global__ void normalize(float *array, float *array_norm, float *mean, int window_size, int size)
{
    // global thread id for 1D grid of 2D blocks
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= size) {return;}

    // indices for mean matrix
    int w_idx = t_idx / window_size;

    array_norm[t_idx] = array[t_idx] - mean[w_idx];
}
""")


def _gpu_normalize_intensity(win_d):
    """Remove the mean from each IW of a 3D stack of interrogation windows.

    Parameters
    ----------
    win_d : GPUArray
        3D float (n_windows, ht, wd), interrogation windows.

    Returns
    -------
    GPUArray
        3D float (n_windows, ht, wd), normalized intensities in the windows.

    """
    _check_arrays(win_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=3)
    n_windows, ht, wd = win_d.shape
    window_size = ht * wd
    size = win_d.size

    win_norm_d = gpuarray.zeros((n_windows, ht, wd), dtype=DTYPE_f)

    mean_d = cumisc.mean(win_d.reshape(n_windows, int(window_size)), axis=1)

    block_size = _BLOCK_SIZE
    grid_size = ceil(size / block_size)
    normalize = mod_norm.get_function('normalize')
    normalize(win_d, win_norm_d, mean_d, DTYPE_i(window_size), DTYPE_i(size), block=(block_size, 1, 1),
              grid=(grid_size, 1))

    return win_norm_d


mod_zp = SourceModule("""
__global__ void zero_pad(float *array_zp, float *array, int fft_ht, int fft_wd, int ht, int wd, int dx, int dy)
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
""")


def _gpu_zero_pad(win_d, fft_shape, extended_search_offset=0):
    """Function that zero-pads an 3D stack of arrays for use with the scikit-cuda FFT function.

    Parameters
    ----------
    win_d : GPUArray
        3D float (n_windows, ht, wd), interrogation windows.
    fft_shape : tuple
        Int (ht, wd), shape to zero pad the date to.
    extended_search_offset: int or tuple, optional
        (offset_x, offset_y), offsets to the destination index in the padded array. Used for the extended search area
        PIV method.

    Returns
    -------
    GPUArray
        3D float, windows which have been zero-padded.

    """
    _check_arrays(win_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=3)
    assert 0 <= extended_search_offset == int(extended_search_offset)
    if isinstance(extended_search_offset, int):
        offset_x_i = offset_y_i = DTYPE_i(extended_search_offset)
    else:
        offset_x_i, offset_y_i = DTYPE_i(extended_search_offset)
    n_windows, wd, ht = win_d.shape
    fft_ht_i, fft_wd_i = DTYPE_i(fft_shape)

    win_zp_d = gpuarray.zeros((n_windows, *fft_shape), dtype=DTYPE_f)

    block_size = 8
    grid_size = ceil(max(wd, ht) / block_size)
    zero_pad = mod_zp.get_function('zero_pad')
    zero_pad(win_zp_d, win_d, fft_ht_i, fft_wd_i, DTYPE_i(ht), DTYPE_i(wd), offset_x_i, offset_y_i,
             block=(block_size, block_size, 1), grid=(n_windows, grid_size, grid_size))

    return win_zp_d


def _gpu_cross_correlate(win_a_d, win_b_d):
    """Returns circular cross-correlation between two stacks of interrogation windows.

    The correlation function is computed using the correlation theorem.

    Parameters
    ----------
    win_a_d, win_b_d : GPUArray
        3D float (n_windows, fft_ht, fft_wd), zero-padded interrogation windows.

    Returns
    -------
    GPUArray
        3D (n_windows, fft_ht, fft_wd), outputs of the cross-correlation function.

    """
    _check_arrays(win_a_d, win_b_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, shape=win_b_d.shape, ndim=3)
    n_windows, fft_ht, fft_wd = win_a_d.shape

    win_cross_correlate_d = gpuarray.empty((n_windows, fft_ht, fft_wd), DTYPE_f)
    win_a_fft_d = gpuarray.empty((n_windows, fft_ht, fft_wd // 2 + 1), DTYPE_c)
    win_b_fft_d = gpuarray.empty((n_windows, fft_ht, fft_wd // 2 + 1), DTYPE_c)

    # Forward FFTs.
    plan_forward = cufft.Plan((fft_ht, fft_wd), DTYPE_f, DTYPE_c, batch=n_windows)
    cufft.fft(win_a_d, win_a_fft_d, plan_forward)
    cufft.fft(win_b_d, win_b_fft_d, plan_forward)

    # Multiply the FFTs.
    win_a_fft_d = win_a_fft_d.conj()
    win_fft_product_d = win_b_fft_d * win_a_fft_d
    # win_fft_product_d = win_a_fft_d.conj() * win_b_fft_d

    # Inverse transform.
    plan_inverse = cufft.Plan((fft_ht, fft_wd), DTYPE_c, DTYPE_f, batch=n_windows)
    cufft.ifft(win_fft_product_d, win_cross_correlate_d, plan_inverse, True)

    return win_cross_correlate_d


mod_index_update = SourceModule("""
__global__ void window_index_f(float *dest, float *src, int *indices, int ws, int n_windows)
{
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= n_windows) {return;}

    dest[t_idx] = src[t_idx * ws + indices[t_idx]];
}
""")


def _gpu_window_index_f(correlation_d, indices_d):
    """Returns the values of the peaks from the 2D correlation.

    Parameters
    ----------
    correlation_d : GPUArray
        2D float (n_windows, m * n), correlation values of each window.
    indices_d : GPUArray
        1D int (n_windows,), indexes of the peaks.

    Returns
    -------
    GPUArray
        1D float (m * n)

    """
    _check_arrays(correlation_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=2)
    n_windows, window_size = correlation_d.shape
    _check_arrays(indices_d, array_type=gpuarray.GPUArray, dtype=DTYPE_i, shape=(n_windows,), ndim=1)

    peak_d = gpuarray.empty(n_windows, dtype=DTYPE_f)

    block_size = _BLOCK_SIZE
    grid_size = ceil(n_windows / block_size)
    index_update = mod_index_update.get_function('window_index_f')
    index_update(peak_d, correlation_d, indices_d, DTYPE_i(window_size), DTYPE_i(n_windows), block=(block_size, 1, 1),
                 grid=(grid_size, 1))

    return peak_d


def _find_peak(correlation_d):
    """Returns the row and column of the highest peak in correlation function.

    Parameters
    ----------
    correlation_d : GPUArray
        3D float, image of the correlation function.

    Returns
    -------
    peak_idx_d : GPUArray
        1D int, index of peak location in reshaped correlation function.

    """
    n_windows, wd, ht = correlation_d.shape

    corr_reshape_d = correlation_d.reshape(n_windows, wd * ht)
    peak_idx_d = cumisc.argmax(corr_reshape_d, axis=1).astype(DTYPE_i)

    return peak_idx_d


def _get_peak(correlation_d, peak_idx_d):
    """Returns the value of the highest peak in correlation function.

    Parameters
    ----------
    peak_idx_d : GPUArray
        1D int, image of the correlation function.

    Returns
    -------
    GPUArray
        1D int, flattened index of corr peak.

    """
    n_windows, wd, ht = correlation_d.shape
    corr_reshape_d = correlation_d.reshape(n_windows, wd * ht)
    peak_value_d = _gpu_window_index_f(corr_reshape_d, peak_idx_d)

    return peak_value_d


mod_subpixel_approximation = SourceModule("""
__global__ void gaussian(float *row_sp, float *col_sp, int *row_p, int *col_p, float *corr, int n_windows, int ht,
                    int wd, int ws)
{
    // x blocks are windows; y and z blocks are x and y dimensions, respectively.
    int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (w_idx >= n_windows) {return;}
    const float small = 1e-20f;

    // Compute the index mapping.
    int row = row_p[w_idx];
    int col = col_p[w_idx];
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

__global__ void parabolic(float *row_sp, float *col_sp, int *row_p, int *col_p, float *corr, int n_windows, int ht,
                    int wd, int ws)
{
    // x blocks are windows; y and z blocks are x and y dimensions, respectively.
    int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (w_idx >= n_windows) {return;}
    const float small = 1e-20f;

    // Compute the index mapping.
    int row = row_p[w_idx];
    int col = col_p[w_idx];
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

__global__ void centroid(float *row_sp, float *col_sp, int *row_p, int *col_p, float *corr, int n_windows, int ht,
                    int wd, int ws)
{
    // x blocks are windows; y and z blocks are x and y dimensions, respectively.
    int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (w_idx >= n_windows) {return;}
    const float small = 1e-20f;

    // Compute the index mapping.
    int row = row_p[w_idx];
    int col = col_p[w_idx];
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
""")


def _gpu_subpixel_approximation(correlation_d, row_peak_d, col_peak_d, method):
    """Returns the subpixel position of the peaks using gaussian approximation.

    Parameters
    ----------
    correlation_d : GPUArray
       3D float (n_windows, fft_wd, fft_ht data from the window correlations.
    row_peak_d, col_peak_d : GPUArray
        1D int (n_windows,), location of the correlation peak.
    method : str {'gaussian', 'parabolic', 'centroid'}
        Method of the subpixel approximation.

    Returns
    -------
    row_sp_d, col_sp_d : GPUArray
        1D float (n_windows,), row and column positions of the subpixel peak.

    """
    _check_arrays(correlation_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=3)
    _check_arrays(row_peak_d, col_peak_d, array_type=gpuarray.GPUArray, dtype=DTYPE_i, shape=(correlation_d.shape[0],),
                  ndim=1)
    assert method in ALLOWED_SUBPIXEL_METHODS
    n_windows, ht, wd = correlation_d.shape
    window_size = ht * wd

    row_sp_d = gpuarray.empty_like(row_peak_d, dtype=DTYPE_f)
    col_sp_d = gpuarray.empty_like(col_peak_d, dtype=DTYPE_f)

    block_size = _BLOCK_SIZE
    grid_size = ceil(n_windows / block_size)
    if method == 'gaussian':
        gaussian_approximation = mod_subpixel_approximation.get_function('gaussian')
        gaussian_approximation(row_sp_d, col_sp_d, row_peak_d, col_peak_d, correlation_d, DTYPE_i(n_windows),
                               DTYPE_i(ht), DTYPE_i(wd), DTYPE_i(window_size), block=(block_size, 1, 1),
                               grid=(grid_size, 1))
    elif method == 'parabolic':
        parabolic_approximation = mod_subpixel_approximation.get_function('parabolic')
        parabolic_approximation(row_sp_d, col_sp_d, row_peak_d, col_peak_d, correlation_d, DTYPE_i(n_windows),
                                DTYPE_i(ht), DTYPE_i(wd), DTYPE_i(window_size), block=(block_size, 1, 1),
                                grid=(grid_size, 1))
    else:
        centroid_approximation = mod_subpixel_approximation.get_function('centroid')
        centroid_approximation(row_sp_d, col_sp_d, row_peak_d, col_peak_d, correlation_d, DTYPE_i(n_windows),
                               DTYPE_i(ht), DTYPE_i(wd), DTYPE_i(window_size), block=(block_size, 1, 1),
                               grid=(grid_size, 1))

    return row_sp_d, col_sp_d


def _peak2mean(correlation_d, corr_peak_d):
    """Returns the mean-energy measure of the signal-to-noise-ratio."""
    correlation_rms_d = _gpu_mask_rms(correlation_d, corr_peak_d)
    sig2noise_d = _peak2energy(correlation_rms_d, corr_peak_d)

    return sig2noise_d


def _peak2energy(correlation_d, corr_peak_d):
    """Returns the RMS-measure of the signal-to-noise-ratio."""
    _check_arrays(correlation_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=3)
    _check_arrays(corr_peak_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, size=correlation_d.shape[0])
    n_windows, wd, ht = correlation_d.shape
    size = wd * ht

    # Remove negative correlation values.
    gpu_remove_negative_f(corr_peak_d)
    gpu_remove_negative_f(correlation_d)

    corr_reshape = correlation_d.reshape(n_windows, size)
    corr_mean_d = cumisc.mean(corr_reshape, axis=1)
    sig2noise_d = DTYPE_f(2) * cumath.log10(corr_peak_d / corr_mean_d)
    gpu_remove_nan_f(sig2noise_d)

    return sig2noise_d


def _peak2peak(corr_peak1_d, corr_peak2_d):
    """Returns the peak-to-peak measure of the signal-to-noise-ratio."""
    _check_arrays(corr_peak1_d, corr_peak2_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, shape=corr_peak1_d.shape)

    # Remove negative peaks.
    gpu_remove_negative_f(corr_peak1_d)
    gpu_remove_negative_f(corr_peak2_d)

    sig2noise_d = cumath.log10(corr_peak1_d / corr_peak2_d)
    gpu_remove_nan_f(sig2noise_d)

    return sig2noise_d


mod_mask_peak = SourceModule("""
__global__ void mask_peak(float *corr, int *row_p, int *col_p, int mask_w, int ht, int wd, int mask_dim, int size)
{
    // x blocks are windows; y and z blocks are x and y dimensions, respectively.
    int idx_i = blockIdx.x;
    int idx_x = blockIdx.y * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.z * blockDim.y + threadIdx.y;
    if (idx_x >= mask_dim || idx_y >= mask_dim) {return;}

    // Get the mapping.
    int row = row_p[idx_i] - mask_w + idx_y;
    int col = col_p[idx_i] - mask_w + idx_x;

    // Mask only if inside window domain.
    if (row >= 0 && row < ht && col >= 0 && col < wd) {
        // Mask the point.
        corr[idx_i * size + row * wd + col] = 0.0f;
    }
}
""")


def _gpu_mask_peak(correlation_positive_d, row_peak_d, col_peak_d, mask_width):
    """Returns correlation windows with points around peak masked.

    Parameters
    ----------
    correlation_positive_d : GPUArray.
        3D float (n_windows, fft_wd, fft_ht), correlation data with negative values removed.
    row_peak_d, col_peak_d : GPUArray
        1D int (n_windows,), position of the peaks.
    mask_width : int
        Half size of the region around the first correlation peak to ignore for finding the second peak.

    Returns
    -------
    GPUArray
        3D float.

    """
    _check_arrays(correlation_positive_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=3)
    n_windows, ht, wd = correlation_positive_d.shape
    _check_arrays(row_peak_d, col_peak_d, array_type=gpuarray.GPUArray, dtype=DTYPE_i, shape=(n_windows,))
    window_size = ht * wd
    assert 0 <= mask_width < int(min(ht, wd) / 2), \
        'Mask width must be integer from 0 and to less than half the correlation window height or width.' \
        'Recommended value is 2.'
    mask_dim_i = DTYPE_i(mask_width * 2 + 1)

    correlation_masked_d = correlation_positive_d.copy()

    block_size = 8
    grid_size = ceil(mask_dim_i / block_size)
    fft_shift = mod_mask_peak.get_function('mask_peak')
    fft_shift(correlation_masked_d, row_peak_d, col_peak_d, DTYPE_i(mask_width), DTYPE_i(ht), DTYPE_i(wd), mask_dim_i,
              DTYPE_i(window_size), block=(block_size, block_size, 1), grid=(n_windows, grid_size, grid_size))

    return correlation_masked_d


mod_correlation_rms = SourceModule("""
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
""")


def _gpu_mask_rms(correlation_positive_d, corr_peak_d):
    """Returns correlation windows with values greater than half the primary peak height zeroed.

    Parameters
    ----------
    correlation_positive_d : GPUArray.
        3D float (n_windows, fft_wd, fft_ht), correlation data with negative values removed.
    corr_peak_d : GPUArray
        1D float (n_windows,), value of peaks.

    Returns
    -------
    GPUArray
        3D float.

    """
    _check_arrays(correlation_positive_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=3)
    n_windows, ht, wd = correlation_positive_d.shape
    _check_arrays(corr_peak_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, shape=(n_windows,))
    window_size = ht * wd

    correlation_masked_d = correlation_positive_d.copy()

    block_size = 8
    grid_size = ceil(max(ht, wd) / block_size)
    fft_shift = mod_correlation_rms.get_function('correlation_rms')
    fft_shift(correlation_masked_d, corr_peak_d, DTYPE_i(ht), DTYPE_i(wd), DTYPE_i(window_size),
              block=(block_size, block_size, 1), grid=(n_windows, grid_size, grid_size))

    return correlation_masked_d


def _get_shift(u_d, v_d):
    """Returns the combined shift array."""
    _check_arrays(u_d, v_d, array_type=gpuarray.GPUArray, shape=u_d.shape, dtype=DTYPE_f, ndim=2)
    m, n = u_d.shape

    shift_d = gpuarray.empty((2, m, n), dtype=DTYPE_f)
    shift_d[0, :, :] = u_d
    shift_d[1, :, :] = v_d
    # shift_d = gpuarray.stack(dp_x_d, dp_y_d, axis=0)  # This should work in latest version of PyCUDA.

    return shift_d


mod_update = SourceModule("""
__global__ void update_values(float *f_new, float *f_old, float *peak, int *mask, int size)
{
    // u_new : output argument
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= size) {return;}

    f_new[t_idx] = (f_old[t_idx] + peak[t_idx]) * (1 - mask[t_idx]);
}
""")


def _gpu_update_field(dp_d, peak_d, mask_d):
    """Returns updated velocity field values with masking.

    Parameters
    ----------
    dp_d : GPUArray.
        nD float, predicted displacement.
    peak_d : GPUArray
        nD float, location of peaks.
    mask_d : GPUArray
        nD int, mask.

    Returns
    -------
    GPUArray
        nD float.

    """
    _check_arrays(dp_d, peak_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, size=dp_d.size)
    _check_arrays(mask_d, array_type=gpuarray.GPUArray, dtype=DTYPE_i, size=dp_d.size)
    size = dp_d.size

    f_d = gpuarray.empty_like(dp_d, dtype=DTYPE_f)

    block_size = _BLOCK_SIZE
    grid_size = ceil(size / block_size)
    update_values = mod_update.get_function('update_values')
    update_values(f_d, dp_d, peak_d, mask_d, DTYPE_i(size), block=(block_size, 1, 1), grid=(grid_size, 1))

    return f_d


def _interpolate_replace(x0_d, y0_d, x1_d, y1_d, f0_d, f1_d, val_locations_d, mask_d=None):
    """Replaces the invalid vectors by interpolating another field."""
    _check_arrays(val_locations_d, array_type=gpuarray.GPUArray, shape=f1_d.shape, ndim=2)

    f1_val_d = gpu_interpolate(x0_d, y0_d, x1_d, y1_d, f0_d, mask_d=mask_d)

    # Replace vectors at validation locations.
    f1_val_d = gpuarray.if_positive(val_locations_d, f1_val_d, f1_d)

    return f1_val_d
