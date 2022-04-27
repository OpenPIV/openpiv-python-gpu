"""This module is dedicated to advanced algorithms for PIV image analysis with NVIDIA GPU Support.

All identifiers ending with '_d' exist on the GPU and not the CPU. The GPU is referred to as the device, and therefore
"_d" signifies that it is a device variable. Please adhere to this standard as it makes developing and debugging much
easier. Note that all data must 32-bit at most to be stored on GPUs. Numpy types should be always 32-bit for
compatibility with GPU. Scalars should be python int type in general to work as function arguments. C-type scalars or
arrays that are arguments to GPU kernels should be identified with ending in either _i or _f. The block argument to GPU
kernels should have size of at least 32 to avoid wasting GPU resources. E.g. (32, 1, 1), (8, 8, 1), etc.

"""
import logging
import warnings
from math import sqrt, ceil, log2

import numpy as np
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
from pycuda.compiler import SourceModule

from openpiv.gpu_validation import gpu_validation, ALLOWED_VALIDATION_METHODS, S2N_TOL, MEAN_TOL, MEDIAN_TOL, RMS_TOL
from openpiv.gpu_smoothn import gpu_smoothn
from openpiv.gpu_misc import _check_arrays, gpu_scalar_mod_i, gpu_remove_nan_f

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
ALLOWED_SIG2NOISE_METHODS = {'peak2peak', 'peak2mean', 'peak2energy'}
SMOOTHING_PAR = 0.5
N_FFT = 2
SUBPIXEL_METHOD = 'gaussian'
SIG2NOISE_METHOD = 'peak2peak'
SIG2NOISE_WIDTH = 2


class GPUCorrelation:
    """A class representing the cross correlation function.

    Parameters
    ----------
    frame_a_d, frame_b_d : GPUArray
        2D int, image pair.
    n_fft : int or tuple, optional
        Window size multiplier for fft. Pass a tuple of length 2 for asymmetric multipliers.
    subpixel_method : {'gaussian', 'centroid', 'parabolic'}, optional
        Method to approximate the subpixel location of the peaks.

    Methods
    -------
    __call__(window_size, extended_size=None, d_shift=None, d_strain=None)
        Returns the peaks of the correlation windows.
    sig2noise_ratio(method='peak2peak', width=2)
        Returns the signal-to-noise ratio of the correlation peaks.

    """

    def __init__(self, frame_a_d, frame_b_d, n_fft=N_FFT, subpixel_method=SUBPIXEL_METHOD):
        _check_arrays(frame_a_d, frame_b_d, array_type=gpuarray.GPUArray, shape=frame_a_d.shape, dtype=DTYPE_f, ndim=2)
        self.frame_a_d = frame_a_d
        self.frame_b_d = frame_b_d
        self.frame_shape = frame_a_d.shape
        if isinstance(n_fft, int):
            assert n_fft >= 1
            self.n_fft_x = self.n_fft_y = int(n_fft)
        else:
            assert n_fft[0] >= 1
            self.n_fft_x = int(n_fft[0])
            self.n_fft_y = self.n_fft_x
            logging.info('For now, n_fft is the same in both directions. ({} is used here.)'.format(self.n_fft_x))
        assert subpixel_method in ALLOWED_SUBPIXEL_METHODS, \
            'subpixel_method is invalid. Must be one of {}.'.format(ALLOWED_SUBPIXEL_METHODS)
        self.subpixel_method = subpixel_method

    def __call__(self, window_size, spacing, extended_size=None, shift_d=None, strain_d=None):
        """Returns the pixel peaks using the specified correlation method.

        Parameters
        ----------
        window_size : int
            Size of the interrogation window.
        spacing : int
            Number of pixels between interrogation windows.
        extended_size : int or None, optional
            Extended window size to search in the second frame.
        shift_d : GPUArray or None, optional
            2D ([dx, dy]), dx and dy are 1D arrays of the x-y shift at each interrogation window of the second image.
            This is using the x-y convention of this code where x is the row and y is the column.
        strain_d : GPUArray or None, optional
            2D float, strain tensor. First dimension is (u_x, u_y, v_x, v_y).

        Returns
        -------
        row_sp, col_sp : ndarray
            3D float, locations of the subpixel peaks.

        """
        assert window_size >= 8, 'Window size is too small.'
        assert window_size % 8 == 0, 'Window size must be a multiple of 8.'
        self.window_size = window_size
        self.extended_size = extended_size if extended_size is not None else window_size
        assert (self.extended_size & (self.extended_size - 1)) == 0, 'Window size (extended) must be power of 2.'
        self.spacing = int(spacing)
        self.field_shape = get_field_shape(self.frame_shape, self.window_size, self.spacing)
        self.n_windows = self.field_shape[0] * self.field_shape[1]

        # Pad up to power of 2 to boost fft speed.
        self.fft_wd = 2 ** ceil(log2(self.extended_size * self.n_fft_x))
        self.fft_ht = 2 ** ceil(log2(self.extended_size * self.n_fft_y))
        self.fft_shape = (self.fft_ht, self.fft_wd)
        self.fft_size = self.fft_wd * self.fft_ht

        # Return stack of all IWs.
        win_a_d, win_b_d = self._stack_iw(self.frame_a_d, self.frame_b_d, shift_d, strain_d)

        # Correlate the windows.
        self.correlation_d = self._correlate_windows(win_a_d, win_b_d)

        # Get first peak of correlation.
        self.row_peak_d, self.col_peak_d, self.corr_peak1_d = _find_peak(self.correlation_d)
        self._check_zero_correlation()

        # Get the subpixel location.
        row_sp_d, col_sp_d = _gpu_subpixel_approximation(self.correlation_d, self.row_peak_d, self.col_peak_d,
                                                         self.subpixel_method)

        # Center the peak displacement.
        i_peak, j_peak = self._get_displacement(row_sp_d, col_sp_d)

        return i_peak, j_peak

    def get_sig2noise(self, subpixel_method=SUBPIXEL_METHOD, mask_width=SIG2NOISE_WIDTH):
        """Computes the signal-to-noise ratio using one of three available methods.

        The signal-to-noise ratio is computed from the correlation and is a measure of the quality of the matching
        between two interrogation windows. Note that this method returns the base-10 logarithm of the sig2noise ratio.
        The sig2noise field contains +np.Inf values where there is no noise.

        Parameters
        ----------
        subpixel_method : string, optional
            Method for evaluating the signal-to-noise ratio value from the correlation map. Can be 'peak2peak',
            'peak2mean', 'peak2energy'.
        mask_width : int, optional
            Half size of the region around the first correlation peak to ignore for finding the second peak. Only used
            if 'sig2noise_method == peak2peak'.

        Returns
        -------
        ndarray
            2D float, the base-10 logarithm of the signal-to-noise ratio from the correlation map for each vector.

        """
        assert subpixel_method in ALLOWED_SIG2NOISE_METHODS, \
            'subpixel_method_method is invalid. Must be one of {}.'.format(ALLOWED_SUBPIXEL_METHODS)
        assert 0 <= mask_width < int(min(self.fft_shape) / 2), \
            'Mask width must be integer from 0 and to less than half the correlation window height or width. ' \
            'Recommended value is 2.'

        # Set all negative values in correlation peaks to zero.
        corr_peak1_d = gpu_mask(self.corr_peak1_d)

        # Compute signal-to-noise ratio by the chosen method.
        if subpixel_method == 'peak2mean':
            sig2noise_d = _peak2mean(self.correlation_d, corr_peak1_d)
        elif subpixel_method == 'peak2energy':
            sig2noise_d = _peak2energy(self.correlation_d, corr_peak1_d)
        else:
            corr_peak2_d = self._get_second_peak_height(self.correlation_d, mask_width)
            sig2noise_d = _peak2peak(self.corr_peak1_d, corr_peak2_d)

        gpu_remove_nan_f(sig2noise_d)

        return sig2noise_d.reshape(self.field_shape)

    def _stack_iw(self, frame_a_d, frame_b_d, shift_d, strain_d=None):
        """Creates a 3D array stack of all the interrogation windows.

        This is necessary to do the FFTs all at once on the GPU. This populates interrogation windows from the origin
        of the image. The implementation requires that the window sizes are multiples of 4.

        Parameters
        -----------
        frame_a_d, frame_b_d : GPUArray
            2D int, image pair.
        shift_d : GPUArray
            3D float, shift of the second window.
        strain_d : GPUArray or None
            3D float, strain rate tensor. First dimension is (u_x, u_y, v_x, v_y).

        Returns
        -------
        win_a_d, win_b_d : GPUArray
            3D float, all interrogation windows stacked on each other.

        """
        _check_arrays(frame_a_d, frame_b_d, array_type=gpuarray.GPUArray, shape=frame_b_d.shape, dtype=DTYPE_f, ndim=2)
        # Buffer is used to shift the extended windows an additional amount.
        buffer = -((self.extended_size - self.window_size) // 2)

        if shift_d is not None:
            # Use translating windows.
            win_a_d = _gpu_window_slice_deform(frame_a_d, self.window_size, self.spacing, 0, -0.5, shift_d, strain_d)
            win_b_d = _gpu_window_slice_deform(frame_b_d, self.extended_size, self.spacing, buffer, 0.5, shift_d,
                                               strain_d)
        else:
            # Use non-translating windows.
            win_a_d = _gpu_window_slice(frame_a_d, self.window_size, self.spacing, 0)
            win_b_d = _gpu_window_slice(frame_b_d, self.extended_size, self.spacing, buffer)

        return win_a_d, win_b_d

    def _correlate_windows(self, win_a_d, win_b_d):
        """Computes the cross-correlation of the window stacks with zero-padding.

        Parameters
        ----------
        win_a_d, win_b_d : GPUArray
            3D float, stacked window data.

        Returns
        -------
        GPUArray
            3D, outputs of the correlation function.

        """
        _check_arrays(win_a_d, win_b_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=3)

        # Normalize array by computing the norm of each IW.
        win_a_norm_d = _gpu_normalize_intensity(win_a_d)
        win_b_norm_d = _gpu_normalize_intensity(win_b_d)

        # Zero pad arrays according to extended size requirements.
        offset = (self.extended_size - self.window_size) // 2
        win_a_zp_d = _gpu_zero_pad(win_a_norm_d, self.fft_shape, offset=offset)
        win_b_zp_d = _gpu_zero_pad(win_b_norm_d, self.fft_shape)

        corr_d = _cross_correlate(win_a_zp_d, win_b_zp_d)

        return gpu_fft_shift(corr_d)

    def _check_zero_correlation(self):
        """Sets the row and column to the center if the correlation peak is near zero."""
        center_d = gpuarray.ones_like(self.row_peak_d, dtype=DTYPE_i) * DTYPE_i(self.fft_wd // 2)
        self.row_peak_d = gpuarray.if_positive(self.corr_peak1_d, self.row_peak_d, center_d)
        self.col_peak_d = gpuarray.if_positive(self.corr_peak1_d, self.col_peak_d, center_d)

    def _get_displacement(self, row_sp_d, col_sp_d):
        i_peak = row_sp_d - DTYPE_f(self.fft_ht // 2)
        j_peak = col_sp_d - DTYPE_f(self.fft_wd // 2)

        return i_peak.reshape(self.field_shape), j_peak.reshape(self.field_shape)

    def _get_second_peak_height(self, correlation_positive_d, mask_width):
        """Find the value of the second-largest peak.

        The second-largest peak is the height of the peak in the region outside a width * width sub-matrix around
        the first correlation peak.

        Parameters
        ----------
        correlation_positive_d : GPUArray.
            Correlation data with negative values removed.
        mask_width : int
            Half size of the region around the first correlation peak to ignore for finding the second peak.

        Returns
        -------
        GPUArray
            Value of the second correlation peak for each interrogation window.

        """
        assert self.row_peak_d is not None and self.col_peak_d is not None

        # Set points around the first peak to zero.
        correlation_masked_d = _gpu_mask_peak(correlation_positive_d, self.row_peak_d, self.col_peak_d, mask_width)

        # Get the height of the second peak of correlation.
        _, _, corr_max2_d = _find_peak(correlation_masked_d)

        return corr_max2_d

    def free_data(self):
        """Frees frame data from GPU."""
        self.frame_a_d = None
        self.frame_b_d = None


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
        2D int, integers containing grey levels of the first and second frames.
    mask : ndarray or None, optional
        2D, int, array of integers with values 0 for the background, 1 for the flow-field. If the center of a window is
        on a 0 value the velocity is set to 0.
    window_size_iters : tuple or int, optional
        Number of iterations performed at each window size
    min_window_size : tuple or int, optional
        Length of the sides of the square deformation. Only supports multiples of 8.
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
    validation_method : {tuple, 's2n', 'median_velocity', 'mean_velocity', 'rms_velocity'}, optional
        Method used for validation. Only the mean velocity method is implemented now. The default tolerance is 2 for
        median validation.

    Returns
    -------
    x, y : ndarray
        2D, Coordinates where the PIV-velocity fields have been computed.
    u, v : ndarray
        2D, Velocity fields in pixel/time units.
    mask : ndarray
        2D, the boolean values (True for vectors interpolated from previous iteration).
    s2n : ndarray
        2D, the signal to noise ratio of the final velocity field.

    Other Parameters
    ----------------
    trust_1st_iter : bool
        With a first window size following the 1/4 rule, the 1st iteration can be trusted and the value should be 1.
    s2n_tol, median_tol, mean_tol, median_tol, rms_tol : float
        Tolerance of the validation methods.
    smoothing_par : float
        Smoothing parameter to pass to smoothn to apply to the intermediate velocity fields.
    extend_ratio : float
        Ratio the extended search area to use on the first iteration. If not specified, extended search will not be
        used.
    subpixel_method : {'gaussian'}
        Method to estimate subpixel location of the peak.
    return_sig2noise : bool
        Sets whether to return the signal-to-noise ratio. Not returning the signal-to-noise speeds up computation
        significantly, which is default behaviour.
    sig2noise_method : {'peak2peak', 'peak2mean'}
        Method of signal-to-noise-ratio measurement.
    s2n_width : int
        Half size of the region around the first correlation peak to ignore for finding the second peak. Default is 2.
        Only used if sig2noise_method==peak2peak.
    n_fft : int or tuple
        Size-factor of the 2D FFT in x and y-directions. The default of 2 is recommended.

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
        (ht, wd) of the image series.
    window_size_iters : tuple or int, optional
        Number of iterations performed at each window size.
    min_window_size : tuple or int, optional
        Length of the sides of the square deformation. Only support multiples of 8.
    overlap_ratio : float, optional
        Ratio of overlap between two windows (between 0 and 1).
    dt : float, optional
        Time delay separating the two frames.
    mask : ndarray or None, optional
        2D, float. Array of integers with values 0 for the background, 1 for the flow-field. If the center of a window
        is on a 0 value the velocity is set to 0.
    deform : bool, optional
        Whether to deform the windows by the velocity gradient at each iteration.
    smooth : bool, optional
        Whether to smooth the intermediate fields.
    nb_validation_iter : int, optional
        Number of iterations per validation cycle.
    validation_method : {tuple, 's2n', 'median_velocity', 'mean_velocity', 'rms_velocity'}, optional
        Method(s) to use for validation.

    Other Parameters
    ----------------
    trust_1st_iter : bool
        With a first window size following the 1/4 rule, the 1st iteration can be trusted and the value should be 1.
    s2n_tol, median_tol, mean_tol, median_tol, rms_tol : float
        Tolerance of the validation methods.
    smoothing_par : float
        Smoothing parameter to pass to smoothn to apply to the intermediate velocity fields. Default is 0.5.
    extend_ratio : float
        Ratio the extended search area to use on the first iteration. If not specified, extended search will not be
        used.
    subpixel_method : {'gaussian'}
        Method to estimate subpixel location of the peak.
    sig2noise_method : {'peak2peak', 'peak2mean', 'peak2energy'}
        Method of signal-to-noise-ratio measurement.
    sig2noise_width : int
        Half size of the region around the first correlation peak to ignore for finding the second peak. Default is 2.
        Only used if sig2noise_method==peak2peak.
    n_fft : int or tuple
        Size-factor of the 2D FFT in x and y-directions. The default of 2 is recommended.

    Attributes
    ----------
    coords : ndarray
        2D, Coordinates where the PIV-velocity fields have been computed.
    mask : ndarray
        2D, the boolean values (True for vectors interpolated from previous iteration).
    s2n : ndarray
        2D, the signal-to-noise ratio of the final velocity field.

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
        self.nb_iter_max = sum(self.ws_iters)
        self.overlap_ratio = float(overlap_ratio)
        self.dt = dt
        self.im_mask = mask.astype(DTYPE_f) if mask is not None else None
        self.im_mask_d = gpuarray.to_gpu(self.im_mask) if self.im_mask is not None else None
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
        self.sig2noise_method = kwargs['sig2noise_method'] if 'sig2noise_method' in kwargs else SIG2NOISE_METHOD
        self.sig2noise_width = kwargs['sig2noise_width'] if 'sig2noise_width' in kwargs else SIG2NOISE_WIDTH
        self.trust_1st_iter = kwargs['trust_first_iter'] if 'trust_first_iter' in kwargs else False

        self._check_inputs()

        self.window_size_l = None
        self.spacing_l = None
        self.field_shape_l = None
        self.x = self.y = None
        self.x_dl = self.y_dl = None
        self.field_mask_dl = None
        self.mask = None
        self.corr = None
        self.sig2noise_d = None

        self.set_geometry()

    def __call__(self, frame_a, frame_b):
        """Processes an image pair.

        Parameters
        ----------
        frame_a, frame_b : ndarray
            2D int, integers containing grey levels of the first and second frames.

        Returns
        -------
        u, v : array
            2D, u velocity component, in pixels/time units.

        """
        _check_arrays(frame_a, frame_b, array_type=np.ndarray, ndim=2)
        u_d = v_d = None
        u_previous_d = v_previous_d = None
        dp_x_d = dp_y_d = None

        # Send masked frames to device.
        frame_a_d, frame_b_d = self._mask_image(frame_a, frame_b)

        # Create the correlation object.
        self.corr = GPUCorrelation(frame_a_d, frame_b_d, n_fft=self.n_fft, subpixel_method=self.subpixel_method)

        # MAIN LOOP
        for k in range(self.nb_iter_max):
            self._k = k
            logging.info('ITERATION {}'.format(k))
            extended_size, shift_d, strain_d = self._get_corr_arguments(dp_x_d, dp_y_d)

            # Get window displacement to subpixel accuracy.
            i_peak_d, j_peak_d = self.corr(self.window_size_l[k], self.spacing_l[k], extended_size=extended_size,
                                           shift_d=shift_d, strain_d=strain_d)

            # update the field with new values
            u_d, v_d = self._update_values(i_peak_d, j_peak_d, dp_x_d, dp_y_d)
            self._log_residual(i_peak_d, j_peak_d)

            # VALIDATION
            if k == 0 and self.trust_1st_iter:
                logging.info('No validation--trusting 1st iteration.')
            else:
                u_d, v_d = self._validate_fields(u_d, v_d, u_previous_d, v_previous_d)

            # NEXT ITERATION
            # Compute the predictors dpx and dpy from the current displacements.
            if k < self.nb_iter_max - 1:
                u_previous_d = u_d
                v_previous_d = v_d
                dp_x_d, dp_y_d = self._get_next_iteration_prediction(u_d, v_d)

                logging.info('[DONE]--Going to iteration {}.'.format(k + 1))

        u_last_d = u_d
        v_last_d = v_d
        u = (u_last_d / DTYPE_f(self.dt)).get()
        v = (v_last_d / DTYPE_f(-self.dt)).get()

        logging.info('[DONE.]\n')

        self.corr.free_data()

        return u, v

    @property
    def coords(self):
        return self.x, self.y

    @property
    def s2n(self):
        if self.sig2noise_d is not None:
            sig2noise_d = self.sig2noise_d
        else:
            sig2noise_d = self.corr.get_sig2noise(subpixel_method=self.sig2noise_method)
        return sig2noise_d.get()

    def free_data(self):
        """Frees correlation data from GPU."""
        self.corr = None

    def set_geometry(self):
        """Creates the parameters for the mesh geometry and mask at each iteration."""
        self.spacing_l = []
        self.field_shape_l = []
        self.x_dl = []
        self.y_dl = []
        self.field_mask_dl = []

        self.window_size_l = [(2 ** (len(self.ws_iters) - i - 1)) * self.min_window_size
                              for i, ws in enumerate(self.ws_iters) for _ in range(ws)]

        for k in range(self.nb_iter_max):
            self.spacing_l.append(max(1, int(self.window_size_l[k] * (1 - self.overlap_ratio))))
            self.field_shape_l.append(get_field_shape(self.frame_shape, self.window_size_l[k], self.spacing_l[k]))

            x, y = get_field_coords(self.field_shape_l[k], self.window_size_l[k], self.spacing_l[k])
            self.x_dl.append(gpuarray.to_gpu(x[0, :].astype(DTYPE_f)))
            self.y_dl.append(gpuarray.to_gpu(y[:, 0].astype(DTYPE_f)))

            self._set_mask(x, y)

            if k == self.nb_iter_max - 1:
                self.x = x
                self.y = y

        self.mask = self.field_mask_dl[-1]
        
    def _check_inputs(self):
        if int(self.frame_shape[0]) != self.frame_shape[0] or int(self.frame_shape[1]) != self.frame_shape[1]:
            raise TypeError('frame_shape must be either tuple of integers or array-like.')
        if len(self.frame_shape) != 2:
            raise ValueError('frame_shape must be 2D.')
        if not all([1 <= ws == int(ws) for ws in self.ws_iters]):
            raise ValueError('Window sizes must be integers greater than or equal to 1.')
        if not self.nb_iter_max >= 1:
            raise ValueError('Sum of window_size_iters must be equal to or greater than 1.')
        if not 0 < self.overlap_ratio < 1:
            raise ValueError('overlap ratio must be between 0 and 1.')
        if self.dt != float(self.dt):
            raise ValueError('dt must be a number.')
        if self.im_mask is not None:
            if self.im_mask.shape != self.frame_shape:
                raise ValueError('mask is not same shape as image.')
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
        if self.sig2noise_method not in ALLOWED_SIG2NOISE_METHODS:
            raise ValueError('sig2noise_method is not allowed. Allowed is one of: {}'.format(ALLOWED_SIG2NOISE_METHODS))
        if self.subpixel_method not in ALLOWED_SUBPIXEL_METHODS:
            raise ValueError('subpixel_method is not allowed. Allowed is one of: {}'.format(ALLOWED_SUBPIXEL_METHODS))
        if not 1 < self.sig2noise_width == int(self.sig2noise_width):
            raise ValueError('s2n_width must be an integer.')
        if self.trust_1st_iter != bool(self.trust_1st_iter):
            raise ValueError('trust_1st_iter must have a boolean value.')

    def _set_mask(self, x, y):
        if self.im_mask is not None:
            field_mask = self.im_mask[y.astype(DTYPE_i), x.astype(DTYPE_i)]
        else:
            field_mask = np.ones_like(x, dtype=DTYPE_f)
        self.field_mask_dl.append(gpuarray.to_gpu(field_mask))

    # TODO use same mask convention as numpy etc.
    def _mask_image(self, frame_a, frame_b):
        """Mask the images before sending to device."""
        _check_arrays(frame_a, frame_b, array_type=np.ndarray, shape=frame_a.shape, ndim=2)
        if self.im_mask is not None:
            frame_a_d = gpu_mask(gpuarray.to_gpu(frame_a.astype(DTYPE_f)), self.im_mask_d)
            frame_b_d = gpu_mask(gpuarray.to_gpu(frame_b.astype(DTYPE_f)), self.im_mask_d)
        else:
            frame_a_d = gpuarray.to_gpu(frame_a.astype(DTYPE_f))
            frame_b_d = gpuarray.to_gpu(frame_b.astype(DTYPE_f))

        return frame_a_d, frame_b_d

    def _get_corr_arguments(self, dp_x_d, dp_y_d):
        """Returns the shift and strain arguments to the correlation class."""
        # Check if extended search area is used for first iteration.
        shift_d = None
        strain_d = None
        extended_size = None
        if self._k == 0:
            if self.extend_ratio is not None:
                extended_size = int(self.window_size_l[self._k] * self.extend_ratio)
        else:
            _check_arrays(dp_x_d, dp_y_d, array_type=gpuarray.GPUArray, shape=dp_x_d.shape, dtype=DTYPE_f, ndim=2)
            m, n = dp_x_d.shape

            # Compute the shift.
            shift_d = gpuarray.empty((2, m, n), dtype=DTYPE_f)
            shift_d[0, :, :] = dp_x_d
            shift_d[1, :, :] = dp_y_d
            # shift_d = gpuarray.stack(dp_x_d, dp_y_d, axis=0)

            # Compute the strain rate.
            if self.deform:
                strain_d = gpu_strain(dp_x_d, dp_y_d, self.spacing_l[self._k])

        return extended_size, shift_d, strain_d

    def _update_values(self, i_peak_d, j_peak_d, dp_x_d, dp_y_d):
        """Updates the velocity values after each iteration."""
        if dp_x_d == dp_y_d is None:
            u_d = gpu_mask(j_peak_d, self.field_mask_dl[self._k])
            v_d = gpu_mask(i_peak_d, self.field_mask_dl[self._k])
        else:
            u_d = _gpu_update_field(dp_x_d, j_peak_d, self.field_mask_dl[self._k])
            v_d = _gpu_update_field(dp_y_d, i_peak_d, self.field_mask_dl[self._k])

        return u_d, v_d

    def _validate_fields(self, u_d, v_d, u_previous_d, v_previous_d):
        """Return velocity fields with outliers removed."""
        _check_arrays(u_d, v_d, array_type=gpuarray.GPUArray, shape=u_d.shape, dtype=DTYPE_f, ndim=2)
        size = u_d.size

        if 's2n' in self.validation_method and self.nb_validation_iter > 0:
            self.sig2noise_d = self.corr.get_sig2noise(subpixel_method=self.sig2noise_method,
                                                       mask_width=self.sig2noise_width)

        for i in range(self.nb_validation_iter):
            # Get list of places that need to be validated.
            val_locations_d, u_mean_d, v_mean_d = gpu_validation(u_d, v_d, self.sig2noise_d, None,
                                                                 self.validation_method,
                                                                 s2n_tol=self.s2n_tol, median_tol=self.median_tol,
                                                                 mean_tol=self.mean_tol, rms_tol=self.rms_tol)

            # Do the validation.
            n_val = size - int(gpuarray.sum(val_locations_d).get())
            if n_val > 0:
                logging.info('Validating {} out of {} vectors ({:.2%}).'.format(n_val, size, n_val / size))

                u_d, v_d = self._gpu_replace_vectors(u_d, v_d, u_previous_d, v_previous_d, u_mean_d,
                                                     v_mean_d, val_locations_d)
                logging.info('[DONE.]')
            else:
                logging.info('No invalid vectors.')

        return u_d, v_d

    def _gpu_replace_vectors(self, u_d, v_d, u_previous_d, v_previous_d, u_mean_d, v_mean_d, val_locations_d):
        """Replace spurious vectors by the mean or median of the surrounding points."""
        _check_arrays(u_d, v_d, u_mean_d, v_mean_d, val_locations_d, array_type=gpuarray.GPUArray, shape=u_d.shape)

        # First iteration, just replace with mean velocity.
        if self._k == 0:
            u_d = gpuarray.if_positive(val_locations_d, u_d, u_mean_d)
            v_d = gpuarray.if_positive(val_locations_d, v_d, v_mean_d)

        # Case if different dimensions: interpolation using previous iteration.
        elif self._k > 0 and self.field_shape_l[self._k] != self.field_shape_l[self._k - 1]:
            u_d = _interpolate_replace(self.x_dl[self._k - 1], self.y_dl[self._k - 1], self.x_dl[self._k],
                                       self.y_dl[self._k], u_previous_d, u_d, val_locations_d)
            v_d = _interpolate_replace(self.x_dl[self._k - 1], self.y_dl[self._k - 1], self.x_dl[self._k],
                                       self.y_dl[self._k], v_previous_d, v_d, val_locations_d)

        # Case if same dimensions.
        elif self._k > 0 and self.field_shape_l[self._k] == self.field_shape_l[self._k - 1]:
            u_d = gpuarray.if_positive(val_locations_d, u_d, u_previous_d)
            v_d = gpuarray.if_positive(val_locations_d, v_d, v_previous_d)

        return u_d, v_d

    def _get_next_iteration_prediction(self, u_d, v_d):
        """Returns the velocity field to begin the next iteration."""
        _check_arrays(u_d, v_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, shape=u_d.shape, ndim=2)
        # Interpolate if dimensions do not agree.
        if self.window_size_l[self._k + 1] != self.window_size_l[self._k]:
            u_d = gpu_interpolate(self.x_dl[self._k], self.y_dl[self._k], self.x_dl[self._k + 1],
                                  self.y_dl[self._k + 1], u_d)
            v_d = gpu_interpolate(self.x_dl[self._k], self.y_dl[self._k], self.x_dl[self._k + 1],
                                  self.y_dl[self._k + 1], v_d)

        if self.smooth:
            dp_x_d, dp_y_d = gpu_smoothn(u_d, v_d, s=self.smoothing_par)
        else:
            dp_x_d = u_d.copy()
            dp_y_d = v_d.copy()

        return dp_x_d, dp_y_d

    @staticmethod
    def _log_residual(i_peak_d, j_peak_d):
        """Normalizes the residual by the maximum quantization error of 0.5 pixel."""
        _check_arrays(i_peak_d, j_peak_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, shape=i_peak_d.shape)

        try:
            normalized_residual = sqrt(int(gpuarray.sum(i_peak_d ** 2 + j_peak_d ** 2).get()) / i_peak_d.size) / 0.5
            logging.info('[DONE]--Normalized residual : {}.'.format(normalized_residual))
        except OverflowError:
            logging.warning('[DONE]--Overflow in residuals.')
            normalized_residual = np.nan

        return normalized_residual


def get_field_shape(image_size, window_size, spacing):
    """Returns the shape of the resulting velocity field.

    Given the image size, the interrogation window size and the overlap size, it is possible to calculate the number of
    rows and columns of the resulting flow field.

    Parameters
    ----------
    image_size : tuple
        (ht, wd), pixel size of the image first element is number of rows, second element is the number of columns.
    window_size : int
        Size of the interrogation windows.
    spacing : int
        Spacing between vectors in the field.

    Returns
    -------
    tuple
        Shape of the resulting flow field.

    """
    assert window_size >= 8, 'Window size is too small.'
    assert window_size % 8 == 0, 'Window size must be a multiple of 8.'
    assert int(spacing) == spacing > 0, 'spacing must be a positive int.'

    m = int((image_size[0] - spacing) // spacing)
    n = int((image_size[1] - spacing) // spacing)
    return m, n


def get_field_coords(field_shape, window_size, spacing):
    """Returns the coordinates of the resulting velocity field.

    Parameters
    ----------
    field_shape : tuple
        int (m, n), the shape of the resulting flow field.
    window_size : int
        Size of the interrogation windows.
    spacing : float
        Ratio by which two adjacent interrogation windows overlap.

    Returns
    -------
    x, y : ndarray
        2D float, pixel coordinates of the resulting flow field

    """
    assert window_size >= 8, 'Window size is too small.'
    assert window_size % 8 == 0, 'Window size must be a multiple of 8.'
    assert int(spacing) == spacing > 0, 'spacing must be a positive int.'
    m, n = field_shape

    x = np.tile(np.linspace(window_size / 2, window_size / 2 + spacing * (n - 1), n), (m, 1))
    y = np.tile(np.linspace(window_size / 2 + spacing * (m - 1), window_size / 2, m), (n, 1)).T

    return x, y


mod_mask = SourceModule("""
__global__ void mask_frame_gpu(float *frame_masked, float *frame, float *mask, int size)
{
    // frame_masked : output argument
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= size) {return;}

    frame_masked[t_idx] = frame[t_idx] * (mask[t_idx] > 0.0f);
}
""")


def gpu_mask(f_d, mask_d=None):
    """Mask a float-type array with an int type-array.

    Parameters
    ----------
    f_d : GPUArray
        nD float, frame to be masked.
    mask_d : GPUArray or None, optional
        nD int, mask to apply to frame. 1s are values to keep.

    Returns
    -------
    GPUArray
        nD int, masked field.

    """
    if mask_d is None:
        mask_d = f_d
    _check_arrays(f_d, mask_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, size=f_d.size)
    size_i = DTYPE_i(f_d.size)

    frame_masked_d = gpuarray.empty_like(mask_d, dtype=DTYPE_f)

    block_size = 32
    grid_size = ceil(size_i / block_size)
    mask_frame_gpu = mod_mask.get_function('mask_frame_gpu')
    mask_frame_gpu(frame_masked_d, f_d, mask_d, size_i, block=(block_size, 1, 1), grid=(grid_size, 1))

    return frame_masked_d


mod_strain = SourceModule("""
__global__ void strain_gpu(float *strain, float *u, float *v, float h, int m, int n)
{
    // strain : output argument
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = m * n;
    if (t_idx >= size) {return;}
    int row = t_idx / n;
    int col = t_idx % n;

    // x-axis
    // first column
    if (col == 0) {
        strain[row * n] = (u[row * n + 1] - u[row * n]) / h;  // u_x
        strain[size * 2 + row * n] = (v[row * n + 1] - v[row * n]) / h;  // v_x
    // last column
    } else if (col == n - 1) {
        strain[(row + 1) * n - 1] = (u[(row + 1) * n - 1] - u[(row + 1) * n - 2]) / h;  // u_x
        strain[size * 2 + (row + 1) * n - 1] = (v[(row + 1) * n - 1] - v[(row + 1) * n - 2]) / h;  // v_x
    // main body
    } else {
        strain[row * n + col] = (u[row * n + col + 1] - u[row * n + col - 1]) / 2 / h;  // u_x
        strain[size * 2 + row * n + col] = (v[row * n + col + 1] - v[row * n + col - 1]) / 2 / h;  // v_x
    }

    // y-axis
    // first row
    if (row == 0) {
        strain[size + col] = (u[n + col] - u[col]) / h;  // u_y
        strain[size * 3 + col] = (v[n + col] - v[col]) / h;  // v_y
    // last row
    } else if (row == m - 1) {
        strain[size + n * (m - 1) + col] = (u[n * (m - 1) + col] - u[n * (m - 2) + col]) / h;  // u_y
        strain[size * 3 + n * (m - 1) + col] = (v[n * (m - 1) + col] - v[n * (m - 2) + col]) / h;  // v_y
    // main body
    } else {
        strain[size + row * n + col] = (u[(row + 1) * n + col] - u[(row - 1) * n + col]) / 2 / h;  // u_y
        strain[size * 3 + row * n + col] = (v[(row + 1) * n + col] - v[(row - 1) * n + col]) / 2 / h;  // v_y
    }
}
""")


def gpu_strain(u_d, v_d, spacing=1):
    """Computes the full strain rate tensor.

    Parameters
    ----------
    u_d, v_d : GPUArray
        2D float, velocity fields.
    spacing : float, optional
        Spacing between nodes.

    Returns
    -------
    GPUArray
        3D float, full strain tensor of the velocity fields. (4, m, n) corresponds to (u_x, u_y, v_x and v_y).

    """
    _check_arrays(u_d, v_d, array_type=gpuarray.GPUArray, shape=u_d.shape, dtype=DTYPE_f)
    assert spacing > 0, 'Spacing must be greater than 0.'

    m, n = u_d.shape
    strain_d = gpuarray.empty((4, m, n), dtype=DTYPE_f)

    block_size = 32
    n_blocks = int((m * n) // block_size + 1)
    strain_gpu = mod_strain.get_function('strain_gpu')
    strain_gpu(strain_d, u_d, v_d, DTYPE_f(spacing), DTYPE_i(m), DTYPE_i(n), block=(block_size, 1, 1),
               grid=(n_blocks, 1))

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

    """
    _check_arrays(correlation_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=3)
    n_windows, ht, wd = correlation_d.shape
    window_size_i = DTYPE_i(ht * wd)

    correlation_shift_d = gpuarray.empty_like(correlation_d, dtype=DTYPE_f)

    block_size = 8
    grid_size = ceil(max(ht, wd) / block_size)
    fft_shift = mod_fft_shift.get_function('fft_shift')
    fft_shift(correlation_shift_d, correlation_d, DTYPE_i(ht), DTYPE_i(wd), window_size_i,
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

    // Find limits of domain.
    if (x < 0) {x = 0;
    } else if (x > wd - 1) {x = wd - 1;}
    if (y < 0) {y = 0;
    } else if (y > ht - 1) {y = ht - 1;}

    // Do bilinear interpolation.
    int x1 = floorf(x);
    int x2 = x1 + 1;
    int y1 = floorf(y);
    int y2 = y1 + 1;

    // Terms of the bilinear interpolation.
    float f11 = f0[(y1 * wd + x1)];
    float f21 = f0[(y1 * wd + x2)];
    float f12 = f0[(y2 * wd + x1)];
    float f22 = f0[(y2 * wd + x2)];

    // Apply the mapping. Multiply by outside_range to set values outside the window to zero.
    f1[t_idx] = f11 * (x2 - x) * (y2 - y) + f21 * (x - x1) * (y2 - y) + f12 * (x2 - x) * (y - y1) + f22 * (x - x1)
                * (y - y1);
}
""")


def gpu_interpolate(x0_d, y0_d, x1_d, y1_d, f0_d):
    """Performs an interpolation of a field from one mesh to another.

    The implementation requires that the mesh spacing is uniform. The spacing can be different in x and y directions.

    Parameters
    ----------
    x0_d, y0_d : GPUArray
        1D float, grid coordinates of the original field
    x1_d, y1_d : GPUArray
        1D float, grid coordinates of the field to be interpolated.
    f0_d : GPUArray
        2D float, field to be interpolated.

    Returns
    -------
    GPUArray
        2D float, interpolated field.

    """
    _check_arrays(x0_d, y0_d, x1_d, y1_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=1)
    _check_arrays(f0_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=2)
    ht_i = DTYPE_i(y0_d.size)
    wd_i = DTYPE_i(x0_d.size)
    n = x1_d.size
    m = y1_d.size
    size_i = DTYPE_i(m * n)

    f1_d = gpuarray.empty((m, n), dtype=DTYPE_f)

    # Calculate the relationship between the two grid coordinates.
    buffer_x_f = DTYPE_f((x0_d[0]).get())
    buffer_y_f = DTYPE_f((y0_d[0]).get())
    spacing_x_f = DTYPE_f((x0_d[1].get() - buffer_x_f))
    spacing_y_f = DTYPE_f((y0_d[1].get() - buffer_y_f))

    block_size = 32
    grid_size = ceil(size_i / block_size)
    interpolate_gpu = mod_interpolate.get_function('bilinear_interpolation')
    interpolate_gpu(f1_d, f0_d, x1_d, y1_d, buffer_x_f, buffer_y_f, spacing_x_f, spacing_y_f, ht_i, wd_i, DTYPE_i(n),
                    size_i, block=(block_size, 1, 1), grid=(grid_size, 1))

    return f1_d


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
        float du_dx = strain[idx_i];
        float du_dy = strain[n_windows + idx_i];
        float dv_dx = strain[2 * n_windows + idx_i];
        float dv_dy = strain[3 * n_windows + idx_i];

        // Compute the window vector.
        float r_x = idx_x - ws / 2 + 0.5f;
        float r_y = idx_y - ws / 2 + 0.5f;

        // Compute the deform.
        float du = r_x * du_dx + r_y * du_dy;
        float dv = r_x * dv_dx + r_y * dv_dy;

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
    int x1 = floorf(x);
    int x2 = x1 + 1;
    int y1 = floorf(y);
    int y2 = y1 + 1;

    // Indices of image to map to.
    int w_range = idx_i * ws * ws + ws * idx_y + idx_x;

    // Find limits of domain.
    int inside_domain = (x1 >= 0 && x2 < wd && y1 >= 0 && y2 < ht);

    if (inside_domain) {
    // Terms of the bilinear interpolation.
    float f11 = input[(y1 * wd + x1)];
    float f21 = input[(y1 * wd + x2)];
    float f12 = input[(y2 * wd + x1)];
    float f22 = input[(y2 * wd + x2)];

    // Apply the mapping.
    output[w_range] = (f11 * (x2 - x) * (y2 - y) + f21 * (x - x1) * (y2 - y) + f12 * (x2 - x) * (y - y1) + f22
                      * (x - x1) * (y - y1));
    } else {output[w_range] = 0.0f;}
}
""")


def _gpu_window_slice(frame_d, window_size, spacing, buffer):
    """Creates a 3D array stack of all the interrogation windows.

    Parameters
    -----------
    frame_d : GPUArray
        2D int, frame to create windows from.
    window_size : int
        Side dimension of the square interrogation windows
    spacing : int
        Spacing between vectors of the velocity field.
    buffer : int or tuple
        Adjustment to location of windows from left/top vectors to edge of frame.

    Returns
    -------
    GPUArray
        3D float, interrogation windows stacked on each other.

    """
    _check_arrays(frame_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=2)
    assert buffer == int(buffer)
    if isinstance(buffer, int):
        buffer_x_i = buffer_y_i = DTYPE_i(buffer)
    else:
        buffer_x_i, buffer_y_i = DTYPE_i(buffer)
    ht, wd = frame_d.shape
    m, n = get_field_shape((ht, wd), window_size, spacing)
    n_windows = m * n

    win_d = gpuarray.empty((n_windows, window_size, window_size), dtype=DTYPE_f)

    block_size = 8
    grid_size = ceil(window_size / block_size)
    window_slice_deform = mod_window_slice.get_function('window_slice')
    window_slice_deform(win_d, frame_d, DTYPE_i(window_size), DTYPE_i(spacing), buffer_x_i, buffer_y_i, DTYPE_i(n),
                        DTYPE_i(wd), DTYPE_i(ht), block=(block_size, block_size, 1),
                        grid=(int(n_windows), grid_size, grid_size))

    return win_d


def _gpu_window_slice_deform(frame_d, window_size, spacing, buffer, dt, shift_d, strain_d=None):
    """Creates a 3D array stack of all the interrogation windows using shift and strain.

    Parameters
    -----------
    frame_d : GPUArray
        2D int, frame to create windows from.
    window_size : int
        Side dimension of the square interrogation windows
    spacing : int
        Spacing between vectors of the velocity field.
    buffer : int or tuple
        Adjustment to location of windows from left/top vectors to edge of frame.
    dt : float
        Number between -1 and 1 indicating the level of shifting/deform. E.g. 1 indicates shift by full amount, 0 is
        stationary. This is applied to the deformation in an analogous way.
    shift_d : GPUArray
        3D float, shift of the second window.
    strain_d : GPUArray or None
        3D float, strain rate tensor. First dimension is (u_x, u_y, v_x, v_y).

    Returns
    -------
    GPUArray
        3D float, all interrogation windows stacked on each other.

    """
    _check_arrays(frame_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=2)
    assert 0 <= buffer == int(buffer)
    if isinstance(buffer, int):
        buffer_x_i = buffer_y_i = DTYPE_i(buffer)
    else:
        buffer_x_i, buffer_y_i = DTYPE_i(buffer)
    ht, wd = frame_d.shape
    m, n = get_field_shape((ht, wd), window_size, spacing)
    n_windows = m * n

    win_d = gpuarray.empty((n_windows, window_size, window_size), dtype=DTYPE_f)

    do_deform = DTYPE_i(strain_d is not None)
    if not do_deform:
        strain_d = gpuarray.zeros(1, dtype=DTYPE_i)

    block_size = 8
    grid_size = ceil(window_size / block_size)
    window_slice_deform = mod_window_slice.get_function('window_slice_deform')
    window_slice_deform(win_d, frame_d, shift_d, strain_d, DTYPE_f(dt), do_deform, DTYPE_i(window_size),
                        DTYPE_i(spacing), buffer_x_i, buffer_y_i, DTYPE_i(n_windows), DTYPE_i(n), DTYPE_i(wd),
                        DTYPE_i(ht), block=(block_size, block_size, 1), grid=(int(n_windows), grid_size, grid_size))

    return win_d


mod_norm = SourceModule("""
__global__ void normalize(float *array, float *array_norm, float *mean, int iw_size, int size)
{
    // global thread id for 1D grid of 2D blocks
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= size) {return;}

    // indices for mean matrix
    int w_idx = t_idx / iw_size;

    array_norm[t_idx] = array[t_idx] - mean[w_idx];
}
""")


def _gpu_normalize_intensity(win_d):
    """Remove the mean from each IW of a 3D stack of interrogation windows.

    Parameters
    ----------
    win_d : GPUArray
        3D float, stack of first IWs.

    Returns
    -------
    GPUArray
        3D float, normalized intensities in the windows.

    """
    _check_arrays(win_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=3)
    n_windows, ht, wd = win_d.shape
    iw_size_i = DTYPE_i(ht * wd)
    size_i = DTYPE_i(win_d.size)

    win_norm_d = gpuarray.zeros((n_windows, ht, wd), dtype=DTYPE_f)

    mean_a_d = cumisc.mean(win_d.reshape(n_windows, int(iw_size_i)), axis=1)

    block_size = 32
    grid_size = ceil(size_i / block_size)
    normalize = mod_norm.get_function('normalize')
    normalize(win_d, win_norm_d, mean_a_d, iw_size_i, size_i, block=(block_size, 1, 1), grid=(grid_size, 1))

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


def _gpu_zero_pad(win_d, fft_shape, offset=0):
    """Function that zero-pads an 3D stack of arrays for use with the scikit-cuda FFT function.

    Parameters
    ----------
    win_d : GPUArray
        3D float, arrays to be zero padded.
    fft_shape : tuple
        Int (ht, wd), shape to zero pad the date to.
    offset: int or tuple, optional
        Offsets to the destination index in the padded array. Used for the extended search area PIV method.

    Returns
    -------
    GPUArray
        3D float, windows which have been zero-padded.

    """
    _check_arrays(win_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=3)
    assert 0 <= offset == int(offset)
    if isinstance(offset, int):
        offset_x_i = offset_y_i = DTYPE_i(offset)
    else:
        offset_x_i, offset_y_i = DTYPE_i(offset)
    n_windows, wd, ht = win_d.shape
    fft_ht_i, fft_wd_i = DTYPE_i(fft_shape)

    win_zp_d = gpuarray.zeros((n_windows, *fft_shape), dtype=DTYPE_f)

    block_size = 8
    grid_size = ceil(max(wd, ht) / block_size)
    zero_pad = mod_zp.get_function('zero_pad')
    zero_pad(win_zp_d, win_d, fft_ht_i, fft_wd_i, DTYPE_i(ht), DTYPE_i(wd), offset_x_i, offset_y_i,
             block=(block_size, block_size, 1), grid=(n_windows, grid_size, grid_size))

    return win_zp_d


def _cross_correlate(win_a_d, win_b_d):
    """Returns cross-correlation between two stacks of interrogation windows.

    The correlation function is computed by using the correlation theorem to speed up the computation.

    Parameters
    ----------
    win_a_d, win_b_d : GPUArray
        3D float, stacked window data.

    Returns
    -------
    GPUArray
        3D, outputs of the cross-correlation function.

    """
    _check_arrays(win_a_d, win_b_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, shape=win_b_d.shape, ndim=3)
    n_windows, fft_ht, fft_wd = win_a_d.shape

    win_i_fft_d = gpuarray.empty((n_windows, fft_ht, fft_wd), DTYPE_f)
    win_a_fft_d = gpuarray.empty((n_windows, fft_ht, fft_wd // 2 + 1), DTYPE_c)
    win_b_fft_d = gpuarray.empty((n_windows, fft_ht, fft_wd // 2 + 1), DTYPE_c)

    # Forward FFTs.
    plan_forward = cufft.Plan((fft_ht, fft_wd), DTYPE_f, DTYPE_c, batch=n_windows)
    cufft.fft(win_a_d, win_a_fft_d, plan_forward)
    cufft.fft(win_b_d, win_b_fft_d, plan_forward)

    # Multiply the FFTs.
    win_a_fft_d = win_a_fft_d.conj()
    tmp_d = win_b_fft_d * win_a_fft_d

    # Inverse transform.
    plan_inverse = cufft.Plan((fft_ht, fft_wd), DTYPE_c, DTYPE_f, batch=n_windows)
    cufft.ifft(tmp_d, win_i_fft_d, plan_inverse, True)

    return win_i_fft_d


mod_index_update = SourceModule("""
__global__ void window_index_f(float *dest, float *src, int *indices, int ws, int n_windows)
{
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= n_windows) {return;}

    dest[t_idx] = src[t_idx * ws + indices[t_idx]];
}
""")


def _gpu_window_index_f(src_d, indices_d):
    """Returns the values of the peaks from the 2D correlation.

    Parameters
    ----------
    src_d : GPUArray
        2D float, correlation values.
    indices_d : GPUArray
        1D int, indexes of the peaks.

    Returns
    -------
    GPUArray
        1D float.

    """
    _check_arrays(src_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=2)
    n_windows, window_size = src_d.shape
    _check_arrays(indices_d, array_type=gpuarray.GPUArray, dtype=DTYPE_i, shape=(n_windows,), ndim=1)

    dest_d = gpuarray.empty(n_windows, dtype=DTYPE_f)

    block_size = 32
    grid_size = ceil(n_windows / block_size)
    index_update = mod_index_update.get_function('window_index_f')
    index_update(dest_d, src_d, indices_d, DTYPE_i(window_size), DTYPE_i(n_windows), block=(block_size, 1, 1),
                 grid=(grid_size, 1))

    return dest_d


def _find_peak(correlation_d):
    """Find the row and column of the highest peak in correlation function.

    Parameters
    ----------
    correlation_d : GPUArray
        3D float, image of the correlation function.

    Returns
    -------
    row_peak : GPUArray
        1D int, row position of corr peak.
    col_peak : GPUArray
        1D int, column position of corr peak.
    max_peak : GPUArray
        1D int, flattened index of corr peak.

    """
    n_windows, wd, ht = correlation_d.shape

    # Get index and value of peak.
    corr_reshape_d = correlation_d.reshape(n_windows, wd * ht)
    peak_idx_d = cumisc.argmax(corr_reshape_d, axis=1).astype(DTYPE_i)
    peak_d = _gpu_window_index_f(corr_reshape_d, peak_idx_d)

    # Row and column information of peak.
    row_peak_d, col_peak_d = gpu_scalar_mod_i(peak_idx_d, wd)

    return row_peak_d, col_peak_d, peak_d


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

    if (row <= 0 || row >= ht - 1) {row_sp[w_idx] = row;  // peak on window edges
    } else {
        float cd = corr[ws * w_idx + wd * (row - 1) + col];
        float cu = corr[ws * w_idx + wd * (row + 1) + col];
        if (cd > 0 && cu > 0 && non_zero) {
            cd = logf(cd);
            cu = logf(cu);
            row_sp[w_idx] = row + 0.5f * (cd - cu) / (cd - 2.0f * logf(c) + cu + small);
        } else {row_sp[w_idx] = row;}
    }

    if (col <= 0 || col >= wd - 1) {col_sp[w_idx] = col;  // peak on window edges
    } else {
        float cl = corr[ws * w_idx + wd * row + col - 1];
        float cr = corr[ws * w_idx + wd * row + col + 1];
        if (cl > 0 && cr > 0 && non_zero) {
            cl = logf(cl);
            cr = logf(cr);
            col_sp[w_idx] = col + 0.5f * (cl - cr) / (cl - 2.0f * logf(c) + cr + small);
        } else {col_sp[w_idx] = col;}
    }
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

    if (row <= 0 || row >= ht - 1) {row_sp[w_idx] = row;  // peak on window edges
    } else {
        float cd = corr[ws * w_idx + wd * (row - 1) + col];
        float cu = corr[ws * w_idx + wd * (row + 1) + col];
        row_sp[w_idx] = row + 0.5f * (cd - cu) / (cd - 2.0f * c + cu + small);
    }

    if (col <= 0 || col >= wd - 1) {col_sp[w_idx] = col;  // peak on window edges
    } else {
        float cl = corr[ws * w_idx + wd * row + col - 1];
        float cr = corr[ws * w_idx + wd * row + col + 1];
        col_sp[w_idx] = col + 0.5f * (cl - cr) / (cl - 2.0f * c + cr + small);
    }
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

    if (row <= 0 || row >= ht - 1) {row_sp[w_idx] = row;  // peak on window edges
    } else {
        float cd = corr[ws * w_idx + wd * (row - 1) + col];
        float cu = corr[ws * w_idx + wd * (row + 1) + col];
        if (cd > 0 && cu > 0 && non_zero) {
            row_sp[w_idx] = row + 0.5f * (cu - cd) / (cd + c + cu + small);
        } else {row_sp[w_idx] = row;}
    }

    if (col <= 0 || col >= wd - 1) {col_sp[w_idx] = col;  // peak on window edges
    } else {
        float cl = corr[ws * w_idx + wd * row + col - 1];
        float cr = corr[ws * w_idx + wd * row + col + 1];
        if (cl > 0 && cr > 0 && non_zero) {
            col_sp[w_idx] = col + 0.5f * (cr - cl) / (cl + c + cr + small);
        } else {col_sp[w_idx] = col;}
    }
}
""")


def _gpu_subpixel_approximation(correlation_d, row_peak_d, col_peak_d, method):
    """Returns the subpixel position of the peaks using gaussian approximation.

    Parameters
    ----------
    correlation_d : GPUArray
        3D float, data from the window correlations.
    row_peak_d, col_peak_d : GPUArray
        1D int, location of the correlation peak.
    method : {'gaussian', 'parabolic', 'centroid'}
        Method of the subpixel approximation.

    Returns
    -------
    row_sp_d, col_sp_d : GPUArray
        1D float, row and column positions of the subpixel peak.

    """
    _check_arrays(correlation_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=3)
    _check_arrays(row_peak_d, col_peak_d, array_type=gpuarray.GPUArray, dtype=DTYPE_i, shape=(correlation_d.shape[0],),
                  ndim=1)
    assert method in ALLOWED_SUBPIXEL_METHODS
    n_windows, ht, wd = correlation_d.shape
    window_size_i = DTYPE_i(ht * wd)

    row_sp_d = gpuarray.empty_like(row_peak_d, dtype=DTYPE_f)
    col_sp_d = gpuarray.empty_like(col_peak_d, dtype=DTYPE_f)

    block_size = 32
    grid_size = ceil(n_windows / block_size)
    if method == 'gaussian':
        gaussian_approximation = mod_subpixel_approximation.get_function('gaussian')
        gaussian_approximation(row_sp_d, col_sp_d, row_peak_d, col_peak_d, correlation_d, DTYPE_i(n_windows),
                               DTYPE_i(ht), DTYPE_i(wd), window_size_i, block=(block_size, 1, 1), grid=(grid_size, 1))
    elif method == 'parabolic':
        parabolic_approximation = mod_subpixel_approximation.get_function('parabolic')
        parabolic_approximation(row_sp_d, col_sp_d, row_peak_d, col_peak_d, correlation_d, DTYPE_i(n_windows),
                                DTYPE_i(ht), DTYPE_i(wd), window_size_i, block=(block_size, 1, 1), grid=(grid_size, 1))
    if method == 'centroid':
        centroid_approximation = mod_subpixel_approximation.get_function('centroid')
        centroid_approximation(row_sp_d, col_sp_d, row_peak_d, col_peak_d, correlation_d, DTYPE_i(n_windows),
                               DTYPE_i(ht), DTYPE_i(wd), window_size_i, block=(block_size, 1, 1), grid=(grid_size, 1))

    return row_sp_d, col_sp_d


def _peak2mean(correlation_d, corr_peak1_d):
    """Returns the mean-energy measure of the signal-to-noise-ratio."""
    correlation_rms_d = _gpu_mask_rms(correlation_d, corr_peak1_d)
    return _peak2energy(correlation_rms_d, corr_peak1_d)


def _peak2energy(correlation_d, corr_peak1_d):
    """Returns the RMS-measure of the signal-to-noise-ratio."""
    _check_arrays(correlation_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=3)
    _check_arrays(corr_peak1_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, size=correlation_d.shape[0])
    n_windows, wd, ht = correlation_d.shape
    window_size = wd * ht

    # Remove negative correlation values.
    correlation_d = gpu_mask(correlation_d)

    corr_reshape = correlation_d.reshape(n_windows, window_size)
    corr_mean_d = cumisc.sum(corr_reshape, axis=1) / DTYPE_f(window_size)
    sig2noise_d = DTYPE_f(2) * cumath.log10(corr_peak1_d / corr_mean_d)

    return sig2noise_d


def _peak2peak(corr_peak1_d, corr_peak2_d):
    """Returns the peak-to-peak measure of the signal-to-noise-ratio."""
    _check_arrays(corr_peak1_d, corr_peak2_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, shape=corr_peak1_d.shape)

    # Remove negative peaks.
    corr_peak2_d = gpu_mask(corr_peak2_d)

    sig2noise_d = cumath.log10(corr_peak1_d / corr_peak2_d)

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

    // Return if outside edge of window.
    if (row >= ht || col >= wd) {return;}

    // Mask the point.
    corr[idx_i * size + row * wd + col] = 0.0f;
}
""")


def _gpu_mask_peak(correlation_positive_d, row_peak_d, col_peak_d, mask_width):
    """Returns correlation windows with points around peak masked.

    Parameters
    ----------
    correlation_positive_d : GPUArray.
        Correlation data with negative values removed.
    row_peak_d, col_peak_d : GPUArray
        1D int, position of the peaks.
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
    window_size_i = DTYPE_i(ht * wd)
    assert 0 <= mask_width < int(min(ht, wd) / 2), \
        'Mask width must be integer from 0 and to less than half the correlation window height or width.' \
        'Recommended value is 2.'
    mask_dim_i = DTYPE_i(mask_width * 2 + 1)

    correlation_masked_d = correlation_positive_d.copy()

    block_size = 8
    grid_size = ceil(mask_dim_i / block_size)
    fft_shift = mod_mask_peak.get_function('mask_peak')
    fft_shift(correlation_masked_d, row_peak_d, col_peak_d, DTYPE_i(mask_width), DTYPE_i(ht), DTYPE_i(wd), mask_dim_i,
              window_size_i, block=(block_size, block_size, 1), grid=(n_windows, grid_size, grid_size))

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
        Correlation data with negative values removed.
    corr_peak_d : GPUArray
        1D float, value of peaks.

    Returns
    -------
    GPUArray
        3D float.

    """
    _check_arrays(correlation_positive_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=3)
    n_windows, ht, wd = correlation_positive_d.shape
    _check_arrays(corr_peak_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, shape=(n_windows,))
    window_size_i = DTYPE_i(ht * wd)

    correlation_masked_d = correlation_positive_d.copy()

    block_size = 8
    grid_size = ceil(max(ht, wd) / block_size)
    fft_shift = mod_correlation_rms.get_function('correlation_rms')
    fft_shift(correlation_masked_d, corr_peak_d, DTYPE_i(ht), DTYPE_i(wd), window_size_i,
              block=(block_size, block_size, 1), grid=(n_windows, grid_size, grid_size))

    return correlation_masked_d


mod_update = SourceModule("""
__global__ void update_values(float *f_new, float *f_old, float *peak, float *mask, int size)
{
    // u_new : output argument
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= size) {return;}

    f_new[t_idx] = (f_old[t_idx] + peak[t_idx]) * mask[t_idx];
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
        nD float, mask.

    Returns
    -------
    GPUArray
        3D float.

    """
    _check_arrays(dp_d, peak_d, mask_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, size=dp_d.size)
    size_i = DTYPE_i(dp_d.size)

    f_d = gpuarray.empty_like(dp_d, dtype=DTYPE_f)

    block_size = 32
    grid_size = ceil(size_i / block_size)
    update_values = mod_update.get_function('update_values')
    update_values(f_d, dp_d, peak_d, mask_d, size_i, block=(block_size, 1, 1), grid=(grid_size, 1))

    return f_d


def _interpolate_replace(x0_d, y0_d, x1_d, y1_d, f0_d, f1_d, val_locations_d):
    """Replaces the invalid vectors by interpolating another field."""
    _check_arrays(x0_d, y0_d, x1_d, y1_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=1)
    _check_arrays(f0_d, f1_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=2)
    _check_arrays(val_locations_d, array_type=gpuarray.GPUArray, shape=f1_d.shape, ndim=2)

    f1_val_d = gpu_interpolate(x0_d, y0_d, x1_d, y1_d, f0_d)

    # Replace vectors at validation locations.
    f1_val_d = gpuarray.if_positive(val_locations_d, f1_d, f1_val_d)

    return f1_val_d
