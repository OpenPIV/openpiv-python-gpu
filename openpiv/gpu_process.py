"""This module is dedicated to advanced algorithms for PIV image analysis with NVIDIA GPU Support.

All identifiers ending with '_d' exist on the GPU and not the CPU. The GPU is referred to as the device, and therefore
"_d" signifies that it is a device variable. Please adhere to this standard as it makes developing and debugging much
easier. Note that all data must 32-bit at most to be stored on GPUs. Numpy types should be always 32-bit for
compatibility with GPU. Scalars should be python int type in general to work as function arguments. C-type scalars or
arrays that are arguments to GPU kernels should be identified with ending in either _i or _f. The block argument to GPU
kernels should have size of at least 32 to avoid wasting GPU resources. E.g. (32, 1, 1), (8, 8, 1), etc.
"""

import time
import logging
import warnings
from math import sqrt, ceil

import numpy as np
import skcuda.fft as cufft
# Create the PyCUDA context.
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
from pycuda.compiler import SourceModule

# scikit-cuda gives an annoying warning everytime it's imported.
with warnings.catch_warnings():
    warnings.simplefilter('ignore', UserWarning)
    from skcuda import misc as cumisc

from openpiv.gpu_validation import gpu_validation
from openpiv.gpu_smoothn import smoothn
from openpiv.gpu_misc import _check_inputs, _gpu_window_index_f

# Define 32-bit types.
DTYPE_i = np.int32
DTYPE_f = np.float32
DTYPE_c = np.complex64

# Initialize the scikit-cuda library. This is necessary when certain cumisc calls happen that don't autoinit.
cumisc.init()


class GPUCorrelation:
    """A class representing the cross correlation function.

    Parameters
    ----------
    frame_a_d, frame_b_d : GPUArray
        2D int, image pair.
    nfft_x : int or None, optional
        Window size multiplier for fft.
    subpixel_method : {'gaussian', 'centroid', 'parabolic'}, optional
        Method to approximate the subpixel location of the peaks.

    Methods
    -------
    __call__(window_size, extended_size=None, d_shift=None, d_strain=None)
        Returns the peaks of the correlation windows.
    sig2noise_ratio(method='peak2peak', width=2)
        Returns the signal-to-noise ratio of the correlation peaks.

    """

    def __init__(self, frame_a_d, frame_b_d, nfft_x=None, subpixel_method='gaussian'):
        _check_inputs(frame_a_d, frame_b_d, array_type=gpuarray.GPUArray, shape=frame_a_d.shape, dtype=DTYPE_f, ndim=2)
        # TODO input checks
        self.frame_a_d = frame_a_d
        self.frame_b_d = frame_b_d
        self.frame_shape = DTYPE_i(frame_a_d.shape)
        # TODO fix this definition. Default values should be 2 in arguments.
        if nfft_x is None:
            self.nfft_x = 2
        else:
            assert (self.nfft_x & (self.nfft_x - 1)) == 0, 'nfft must be power of 2'
            self.nfft_x = nfft_x
        self.nfft_y = self.nfft_x
        allowed_methods = {'gaussian', 'centroid', 'parabolic'}
        try:
            assert subpixel_method in allowed_methods
            self.subpixel_method = subpixel_method
        except AssertionError:
            raise ValueError('subpixel_method is invalid. Must be one of {}. ({} was input.)'.format(
                allowed_methods, subpixel_method)) from None

    def __call__(self, window_size, overlap_ratio, extended_size=None, shift_d=None, strain_d=None):
        """Returns the pixel peaks using the specified correlation method.

        Parameters
        ----------
        window_size : int
            Size of the interrogation window.
        overlap_ratio : float
            Overlap between interrogation windows.
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
        assert window_size >= 8, "Window size is too small."
        assert window_size % 8 == 0, "Window size must be a multiple of 8."
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        # TODO remove unnecessary type casts--Python structures should have python types.
        # TODO extended size must be power of 2?
        self.size_extended = DTYPE_i(extended_size) if extended_size is not None else DTYPE_i(window_size)
        self.fft_width_x = DTYPE_i(self.size_extended * self.nfft_x)
        self.fft_width_y = DTYPE_i(self.size_extended * self.nfft_y)
        self.fft_size = self.fft_width_x * self.fft_width_y
        self.spacing = int(self.window_size - (self.window_size * self.overlap_ratio))
        self.n_rows, self.n_cols = DTYPE_i(get_field_shape(self.frame_shape, self.window_size, self.spacing))
        self.n_windows = self.n_rows * self.n_cols

        # Return stack of all IWs.
        win_a_d, win_b_d = self._stack_iw(self.frame_a_d, self.frame_b_d, shift_d, strain_d)

        # Normalize array by computing the norm of each IW.
        win_a_norm_d, win_b_norm_d = self._normalize_intensity(win_a_d, win_b_d)

        # Zero pad arrays.
        win_a_zp_d, win_b_zp_d = self._zero_pad(win_a_norm_d, win_b_norm_d)

        # Correlate the windows.
        self.correlation_d = self._correlate_windows(win_a_zp_d, win_b_zp_d)

        # Get first peak of correlation.
        self.row_peak_d, self.col_peak_d, self.corr_peak1_d = self._find_peak(self.correlation_d)

        # Get the subpixel location.
        row_sp_d, col_sp_d = self._locate_subpixel_peak()

        # Center the peak displacement.
        i_peak = row_sp_d - DTYPE_f(self.fft_width_y / 2)
        j_peak = col_sp_d - DTYPE_f(self.fft_width_x / 2)

        return i_peak, j_peak

    def get_sig2noise(self, method='peak2peak', mask_width=2):
        """Computes the signal-to-noise ratio using one of three available methods.

        The signal-to-noise ratio is computed from the correlation and is a measure of the quality of the matching
        between two interrogation windows. Note that this method returns the base-10 logarithm of the sig2noise ratio.
        The sig2noise field contains +np.Inf values where there is no noise.

        Parameters
        ----------
        method : string, optional
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
        assert method in ['peak2peak', 'peak2mean', 'peak2energy']
        assert 0 <= mask_width < int(min(self.fft_width_x, self.fft_width_y) / 2), \
            'Mask width must be integer from 0 and to less than half the correlation window height or width.' \
            'Recommended value is 2.'

        # Set all negative values to zero.
        correlation_positive_d = self.correlation_d * (self.correlation_d > 1e-3)

        # TODO SRP violation.
        # Compute signal-to-noise ratio by the chosen method.
        if method == 'peak2mean':
            corr_max1_d = self.corr_peak1_d ** 2
            correlation_rms_d = _gpu_mask_rms(correlation_positive_d, self.corr_peak1_d) ** 2
            corr_max2_d = cumisc.sum(correlation_rms_d.reshape(self.n_windows, self.fft_size),
                                     axis=1) / DTYPE_f(self.fft_size)
        elif method == 'peak2energy':
            corr_max1_d = self.corr_peak1_d ** 2
            correlation_energy_d = correlation_positive_d ** 2
            corr_max2_d = cumisc.sum(correlation_energy_d.reshape(self.n_windows, self.fft_size),
                                     axis=1) / DTYPE_f(self.fft_size)
        else:
            corr_max1_d = self.corr_peak1_d
            corr_max2_d = self._find_second_peak_height(correlation_positive_d, width=mask_width)

        # Get signal-to-noise ratio.
        sig2noise_d = cumath.log10(corr_max1_d / corr_max2_d)

        # Remove NaNs.
        sig2noise_d = sig2noise_d * (sig2noise_d == np.NaN)

        return sig2noise_d.reshape(self.n_rows, self.n_cols)

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
        _check_inputs(frame_a_d, frame_b_d, array_type=gpuarray.GPUArray, shape=frame_b_d.shape, dtype=DTYPE_f, ndim=2)
        spacing = DTYPE_i(self.window_size - (self.window_size * self.overlap_ratio))
        buffer = int(spacing - self.size_extended / 2)

        # Use translating windows.
        if shift_d is not None:
            win_a_d = _window_slice_deform(frame_a_d, self.size_extended, spacing, buffer, -0.5, shift_d, strain_d)
            win_b_d = _window_slice_deform(frame_b_d, self.size_extended, spacing, buffer, 0.5, shift_d, strain_d)
        # Use non-translating windows.
        else:
            win_a_d = _window_slice(frame_a_d, self.size_extended, spacing, buffer)
            win_b_d = _window_slice(frame_b_d, self.size_extended, spacing, buffer)

        return win_a_d, win_b_d

    def _normalize_intensity(self, win_a_d, win_b_d):
        """Remove the mean from each IW of a 3D stack of IWs.

        Parameters
        ----------
        win_a_d, win_b_d : GPUArray
            3D float, stack of first IWs.

        Returns
        -------
        win_a_norm_d, win_b_norm_d : GPUArray
            3D float, normalized intensities in the windows.

        """
        iw_size = DTYPE_i(self.size_extended * self.size_extended)

        win_a_norm_d = gpuarray.zeros((self.n_windows, self.size_extended, self.size_extended), dtype=DTYPE_f)
        win_b_norm_d = gpuarray.zeros((self.n_windows, self.size_extended, self.size_extended), dtype=DTYPE_f)

        mean_a_d = cumisc.mean(win_a_d.reshape(self.n_windows, iw_size), axis=1)
        mean_b_d = cumisc.mean(win_b_d.reshape(self.n_windows, iw_size), axis=1)

        mod_norm = SourceModule("""
            __global__ void normalize(float *array, float *array_norm, float *mean, int iw_size)
        {
            // global thread id for 1D grid of 2D blocks
            int thread_idx = blockIdx.x * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;

            // indices for mean matrix
            int mean_idx = thread_idx / iw_size;

            array_norm[thread_idx] = array[thread_idx] - mean[mean_idx];
        }
        """)
        block_size = 8
        grid_size = int(win_a_d.size / block_size ** 2)
        normalize = mod_norm.get_function('normalize')
        normalize(win_a_d, win_a_norm_d, mean_a_d, iw_size, block=(block_size, block_size, 1), grid=(grid_size, 1))
        normalize(win_b_d, win_b_norm_d, mean_b_d, iw_size, block=(block_size, block_size, 1), grid=(grid_size, 1))

        return win_a_norm_d, win_b_norm_d

    def _zero_pad(self, win_a_norm_d, win_b_norm_d):
        """Function that zero-pads an 3D stack of arrays for use with the scikit-cuda FFT function.

        If extended size is passed, then the second window is padded to match the extended size.

        Parameters
        ----------
        win_a_norm_d, win_b_norm_d : GPUArray
            3D float, arrays to be zero padded.

        Returns
        -------
        win_a_zp_d, win_b_zp_d : GPUArray
            3D float, windows which have been zero-padded.

        """
        # compute the window extension
        size0_a = DTYPE_i((self.size_extended - self.window_size) / 2)
        size1_a = DTYPE_i(self.size_extended - size0_a)
        size0_b = DTYPE_i(0)
        size1_b = self.size_extended

        win_a_zp_d = gpuarray.zeros([self.n_windows, self.fft_width_x, self.fft_width_x], dtype=DTYPE_f)
        win_b_zp_d = gpuarray.zeros([self.n_windows, self.fft_width_x, self.fft_width_x], dtype=DTYPE_f)

        mod_zp = SourceModule("""
            __global__ void zero_pad(float *array_zp, float *array, int fft_size, int window_size, int s0, int s1)
            {
                // index, x blocks are windows; y and z blocks are x and y dimensions, respectively
                int ind_i = blockIdx.x;
                int ind_x = blockIdx.y * blockDim.x + threadIdx.x;
                int ind_y = blockIdx.z * blockDim.y + threadIdx.y;
                
                // don't copy if out of range
                if (ind_x < s0 || ind_x >= s1 || ind_y < s0 || ind_y >= s1) {return;}

                // get range of values to map
                int arr_range = ind_i * window_size * window_size + window_size * ind_y + ind_x;
                int zp_range = ind_i * fft_size * fft_size + fft_size * ind_y + ind_x;

                // apply the map
                array_zp[zp_range] = array[arr_range];
            }
        """)
        block_size = 8
        grid_size = int(self.size_extended / block_size)
        zero_pad = mod_zp.get_function('zero_pad')
        zero_pad(win_a_zp_d, win_a_norm_d, self.fft_width_x, self.size_extended, size0_a, size1_a,
                 block=(block_size, block_size, 1), grid=(int(self.n_windows), grid_size, grid_size))
        zero_pad(win_b_zp_d, win_b_norm_d, self.fft_width_x, self.size_extended, size0_b, size1_b,
                 block=(block_size, block_size, 1), grid=(int(self.n_windows), grid_size, grid_size))

        return win_a_zp_d, win_b_zp_d

    def _correlate_windows(self, win_a_zp_d, win_b_zp_d):
        """Compute correlation function between two interrogation windows.

        The correlation function can be computed by using the correlation theorem to speed up the computation.

        Parameters
        ----------
        win_a_zp_d, win_b_zp_d : GPUArray
            3D float, zero-padded correlation windows.

        Returns
        -------
        GPUArray
            2D, output of the correlation function.

        """
        win_h = self.fft_width_y
        win_w = self.fft_width_x

        win_i_fft_d = gpuarray.empty((int(self.n_windows), int(win_h), int(win_w)), DTYPE_f)
        win_a_fft_d = gpuarray.empty((int(self.n_windows), int(win_h), int(win_w // 2 + 1)), DTYPE_c)
        win_b_fft_d = gpuarray.empty((int(self.n_windows), int(win_h), int(win_w // 2 + 1)), DTYPE_c)

        # Forward FFTs.
        plan_forward = cufft.Plan((win_h, win_w), DTYPE_f, DTYPE_c, batch=self.n_windows)
        cufft.fft(win_a_zp_d, win_a_fft_d, plan_forward)
        cufft.fft(win_b_zp_d, win_b_fft_d, plan_forward)

        # Multiply the FFTs.
        win_a_fft_d = win_a_fft_d.conj()
        tmp_d = win_b_fft_d * win_a_fft_d

        # Inverse transform.
        plan_inverse = cufft.Plan((win_h, win_w), DTYPE_c, DTYPE_f, batch=self.n_windows)
        cufft.ifft(tmp_d, win_i_fft_d, plan_inverse, True)
        corr_d = gpu_fft_shift(win_i_fft_d)

        return corr_d

    def _find_peak(self, corr_d):
        """Find the row and column of the highest peak in correlation function.

        Parameters
        ----------
        corr_d : GPUArray
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
        w = DTYPE_i(self.fft_width_x / 2)

        # Get index and value of peak.
        corr_reshape_d = corr_d.reshape(int(self.n_windows), int(self.fft_size))
        peak_idx_d = cumisc.argmax(corr_reshape_d, axis=1).astype(DTYPE_i)
        peak_d = _gpu_window_index_f(corr_reshape_d, peak_idx_d)

        # TODO these array operations are a serious hit to performance.
        # Row and column information of peak.
        # TODO account for the padded space in non-square correlations.
        col_peak_d, row_peak_d = cumath.modf((peak_idx_d.astype(DTYPE_f) / DTYPE_f(self.fft_width_x)))
        row_peak_d = row_peak_d.astype(DTYPE_i)
        col_peak_d = (col_peak_d * DTYPE_i(self.fft_width_x)).astype(DTYPE_i)

        # Return the center if the correlation peak is near zero.
        zero_peak_inverse_d = (peak_d > 1e-3).astype(DTYPE_i)
        zero_peak_d = (1 - zero_peak_inverse_d) * w
        row_peak_d = row_peak_d * zero_peak_inverse_d + zero_peak_d
        col_peak_d = col_peak_d * zero_peak_inverse_d + zero_peak_d

        return row_peak_d, col_peak_d, peak_d

    def _locate_subpixel_peak(self):
        """Find subpixel peak approximation using Gaussian method.

        Returns
        -------
        row_sp, col_sp : GPUArray
            1D float, location of peak to subpixel accuracy for each window.

        """
        if self.subpixel_method == 'gaussian':
            row_sp_d, col_sp_d = _gpu_subpixel_gaussian(self.correlation_d, self.row_peak_d, self.col_peak_d,
                                                        self.fft_width_x, self.fft_width_x)
        elif self.subpixel_method == 'centroid':
            row_sp_d, col_sp_d = _gpu_subpixel_centroid(self.correlation_d, self.row_peak_d, self.col_peak_d,
                                                        self.fft_width_x, self.fft_width_x)
        else:
            row_sp_d, col_sp_d = _gpu_subpixel_parabolic(self.correlation_d, self.row_peak_d, self.col_peak_d,
                                                         self.fft_width_x, self.fft_width_x)

        return row_sp_d, col_sp_d

    def _find_second_peak_height(self, correlation_positive_d, width):
        """Find the value of the second-largest peak.

        The second-largest peak is the height of the peak in the region outside a "width * width" submatrix around
        the first correlation peak.

        Parameters
        ----------
        correlation_positive_d : GPUArray.
            Correlation data with negative values removed.
        width : int
            Half size of the region around the first correlation peak to ignore for finding the second peak.

        Returns
        -------
        GPUArray
            Value of the second correlation peak for each interrogation window.

        """
        # Set points around the first peak to zero.
        correlation_masked_d = _gpu_mask_peak(correlation_positive_d, self.row_peak_d, self.col_peak_d, width)

        # Get the height of the second peak of correlation.
        _, _, corr_max2_d = self._find_peak(correlation_masked_d)

        return corr_max2_d


def gpu_extended_search_area(frame_a, frame_b,
                             window_size,
                             overlap_ratio,
                             dt,
                             search_area_size,
                             **kwargs
                             ):
    """The implementation of the one-step direct correlation with the same size windows.

    This function is meant to be used with an iterative method to cope with the loss of pairs due to particle movement
    out of the search area. It is an adaptation of the original extended_search_area_piv function rewritten with PyCuda
    and CUDA-C to run on an NVIDIA GPU.

    References
    ----------
        Particle-Imaging Techniques for Experimental Fluid Mechanics Annual Review of Fluid Mechanics
            Vol. 23: 261-304 (Volume publication date January 1991)
            DOI: 10.1146/annurev.fl.23.010191.001401

    Parameters
    ----------
    frame_a, frame_b : ndarray
        2D int, grey levels of the first and second frames.
    window_size : int
        Size of the (square) interrogation window for the first frame.
    search_area_size : int
        Size of the (square) interrogation window for the second frame.
    overlap_ratio : float
        Ratio of overlap between two windows (between 0 and 1)
    dt : float
        Time delay separating the two frames.

    Returns
    -------
    u, v : ndarray
        2D, the u and v velocity components, in pixels/seconds.

    Other Parameters
    ----------------
    subpixel_method : {'gaussian', 'centroid', 'parabolic'}
        Method to estimate subpixel location of the peak. Gaussian is default if correlation map is positive. Centroid
        replaces default if correlation map is negative.
    width : int
        Half size of the region around the first correlation peak to ignore for finding the second peak. Default is 2.
        Only used if sig2noise_method==peak2peak.
    nfft_x : int
        Size of the 2D FFT in x-direction. The default of 2 x windows_a.shape[0] is recommended.

    Example
    --------
    u, v = gpu_extended_search_area(frame_a, frame_b, window_size=16, overlap_ratio=0.5, search_area_size=32, dt=1)

    """
    # Extract the parameters
    nfft_x = kwargs['nfft_x'] if 'nfft_x' in kwargs else None

    # cast images as floats and sent to gpu
    frame_a_d = gpuarray.to_gpu(frame_a.astype(DTYPE_f))
    frame_b_d = gpuarray.to_gpu(frame_b.astype(DTYPE_f))

    # Get correlation function
    corr = GPUCorrelation(frame_a_d, frame_b_d, nfft_x)

    # Get window displacement to subpixel accuracy
    sp_i_d, sp_j_d = corr(window_size, overlap_ratio, search_area_size)

    # reshape the peaks
    i_peak = sp_i_d.reshape(corr.n_rows, corr.n_cols)
    j_peak = sp_j_d.reshape(corr.n_rows, corr.n_cols)

    # calculate velocity fields
    u = (j_peak / dt).get()
    v = (-i_peak / dt).get()

    # Free gpu memory
    frame_a_d.gpudata.free()
    frame_b_d.gpudata.free()

    return u, v


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
    windows used in the next iteration. One or more iterations can be performed before the the estimated velocity is
    interpolated onto a finer mesh. This is done until the final mesh and number of iterations is met.

    Algorithm Details
    -----------------
    Only window sizes that are multiples of 8 are supported now, and the minimum window size is 8.
    Windows are shifted symmetrically to reduce bias errors.
    The displacement obtained after each correlation is the residual displacement dc.
    The new displacement is computed by dx = dpx + dcx and dy = dpy + dcy.
    Validation is done by any combination of signal-to-noise ratio, mean, median and rms.
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
        Smoothing parameter to pass to Smoothn to apply to the intermediate velocity fields.
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
    nfftx : int
        Size of the 2D FFT in x-direction. The default of 2 x windows_a.shape[0] is recommended.

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
    mask = piv_gpu.mask_final
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
        Method used for validation. Only the mean velocity method is implemented now. The default tolerance is 2 for
        median validation.

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
    s2n_width : int
        Half size of the region around the first correlation peak to ignore for finding the second peak. Default is 2.
        Only used if sig2noise_method==peak2peak.
    nfft_x : int
        Size of the 2D FFT in x-direction. The default of 2 x windows_a.shape[0] is recommended.

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
        # TODO check all inputs.
        if hasattr(frame_shape, 'shape'):
            self.frame_shape = frame_shape.shape
        else:
            self.frame_shape = frame_shape
        ws_iters = (window_size_iters,) if type(window_size_iters) == int else window_size_iters
        num_ws = len(ws_iters)
        self.overlap_ratio = overlap_ratio
        self.dt = dt
        self.deform = deform
        self.smooth = smooth
        self.nb_iter_max = sum(ws_iters)
        self.nb_validation_iter = nb_validation_iter
        self.spacing = None
        self.n_row = None
        self.n_col = None
        self.x_final = None
        self.y_final = None
        self.x_d = None
        self.y_d = None
        self.field_mask_d = None
        self.mask_final = None

        if mask is not None:
            try:
                assert mask.shape == self.frame_shape
            except AssertionError:
                raise ValueError('Mask is not same shape as image.') from None

        # Set window sizes.
        self.window_size = np.asarray(
            [np.power(2, num_ws - i - 1) * min_window_size for i in range(num_ws) for _ in range(ws_iters[i])],
            dtype=DTYPE_i)

        # TODO These are terrible for understanding. Try a dictionary instead.
        # Validation method.
        self.val_tols = [None, None, None, None]
        val_methods = validation_method if type(validation_method) == str else (validation_method,)
        if 's2n' in val_methods:
            self.val_tols[0] = kwargs['s2n_tol'] if 's2n_tol' in kwargs else 1.2
        if 'median_velocity' in val_methods:
            self.val_tols[1] = kwargs['median_tol'] if 'median_tol' in kwargs else 2
        if 'mean_velocity' in val_methods:
            self.val_tols[2] = kwargs['mean_tol'] if 'mean_tol' in kwargs else 2
        if 'rms_velocity' in val_methods:
            self.val_tols[3] = kwargs['rms_tol'] if 'rms_tol' in kwargs else 2

        # Init other parameters.
        self.trust_1st_iter = kwargs['trust_first_iter'] if 'trust_first_iter' in kwargs else False
        self.smoothing_par = kwargs['smoothing_par'] if 'smoothing_par' in kwargs else 0.5
        self.sig2noise_method = kwargs['sig2noise_method'] if 'sig2noise_method' in kwargs else 'peak2peak'
        self.s2n_width = kwargs['s2n_width'] if 's2n_width' in kwargs else 2
        self.nfft_x = kwargs['nfft_x'] if 'nfft_x' in kwargs else None
        self.extend_ratio = kwargs['extend_ratio'] if 'extend_ratio' in kwargs else None
        self.im_mask = gpuarray.to_gpu(mask.astype(DTYPE_i)) if mask is not None else None
        self.subpixel_method = kwargs['subpixel_method'] if 'subpixel_method' in kwargs else 'gaussian'
        self.corr = None
        self.sig2noise = None

        # Init derived parameters - mesh geometry and masks at each iteration.
        self.set_geometry(mask)

    def __call__(self, frame_a, frame_b):
        """Processes an image pair.

        Parameters
        ----------
        frame_a, frame_b : ndarray
            2D int, integers containing grey levels of the first and second frames.

        Returns
        -------
        u : array
            2D, u velocity component, in pixels/time units.
        v : array
            2D, v velocity component, in pixels/time units.

        """
        u_d = None
        v_d = None
        u_previous_d = None
        v_previous_d = None
        dp_x_d = None
        dp_y_d = None

        # Send masked frames to device.
        frame_a_d, frame_b_d = self._mask_image(frame_a, frame_b)

        # Create the correlation object.
        self.corr = GPUCorrelation(frame_a_d, frame_b_d, self.nfft_x, self.subpixel_method)

        # MAIN LOOP
        for k in range(self.nb_iter_max):
            logging.info('ITERATION {}'.format(k))
            extended_size, shift_d, strain_d = self._get_corr_arguments(dp_x_d, dp_y_d, k)

            # Get window displacement to subpixel accuracy.
            i_peak_d, j_peak_d = self.corr(self.window_size[k], self.overlap_ratio, extended_size=extended_size,
                                           shift_d=shift_d, strain_d=strain_d)

            # update the field with new values
            u_d, v_d = self._update_values(i_peak_d, j_peak_d, dp_x_d, dp_y_d, k)
            self._log_residual(i_peak_d, j_peak_d)

            # VALIDATION
            if k == 0 and self.trust_1st_iter:
                logging.info('No validation--trusting 1st iteration.')
            else:
                u_d, v_d = self._validate_fields(u_d, v_d, u_previous_d, v_previous_d, k)

            # NEXT ITERATION
            # Go to next iteration: compute the predictors dpx and dpy from the current displacements.
            if k < self.nb_iter_max - 1:
                u_previous_d = u_d
                v_previous_d = v_d
                dp_x_d, dp_y_d = self._get_next_iteration_prediction(u_d, v_d, k)

                logging.info('[DONE] -----> going to iteration {}.\n'.format(k + 1))

        u_last_d = u_d
        v_last_d = v_d
        u = (u_last_d / self.dt).get()
        v = (v_last_d / -self.dt).get()

        logging.info('[DONE]\n')

        frame_a_d.gpudata.free()
        frame_b_d.gpudata.free()

        return u, v

    @property
    def coords(self):
        return self.x_final, self.y_final

    @property
    def mask(self):
        return self.mask_final

    @property
    def s2n(self):
        if self.sig2noise is not None:
            s2n = self.sig2noise
        else:
            s2n = self.corr.get_sig2noise(method=self.sig2noise_method)
        return s2n

    def set_geometry(self, mask=None):
        """Creates the parameters for the mesh geometry and mask at each iteration."""
        self.spacing = np.zeros(self.nb_iter_max, dtype=DTYPE_i)
        self.n_row = np.zeros(self.nb_iter_max, dtype=DTYPE_i)
        self.n_col = np.zeros(self.nb_iter_max, dtype=DTYPE_i)
        self.x_d = []
        self.y_d = []
        self.field_mask_d = []
        for k in range(self.nb_iter_max):
            # Init field geometry.
            self.spacing[k] = self.window_size[k] - int(self.window_size[k] * self.overlap_ratio)
            self.n_row[k], self.n_col[k] = get_field_shape(self.frame_shape, self.window_size[k], self.spacing[k])

            # Initialize x, y and mask.
            x, y = get_field_coords((self.n_row[k], self.n_col[k]), self.window_size[k], self.overlap_ratio)
            self.x_d.append(gpuarray.to_gpu(x[0, :]))
            self.y_d.append(gpuarray.to_gpu(y[:, 0]))

            # TODO SRP violation
            if mask is not None:
                field_mask = mask[y.astype(DTYPE_i), x.astype(DTYPE_i)].astype(DTYPE_i)
            else:
                field_mask = np.ones((self.n_row[k], self.n_col[k]), dtype=DTYPE_i)
            self.field_mask_d.append(gpuarray.to_gpu(field_mask))

            if k == self.nb_iter_max - 1:
                self.x_final = x
                self.y_final = y
                self.mask_final = field_mask

    def _mask_image(self, frame_a, frame_b):
        """Mask the images before sending to device."""
        _check_inputs(frame_a, frame_b, shape=frame_a.shape, ndim=2)
        if self.im_mask is not None:
            frame_a_d = gpu_mask(gpuarray.to_gpu(frame_a.astype(DTYPE_f)), self.im_mask)
            frame_b_d = gpu_mask(gpuarray.to_gpu(frame_b.astype(DTYPE_f)), self.im_mask)
        else:
            frame_a_d = gpuarray.to_gpu(frame_a.astype(DTYPE_f))
            frame_b_d = gpuarray.to_gpu(frame_b.astype(DTYPE_f))

        return frame_a_d, frame_b_d

    # TODO can simplify this to not require u_previous
    def _validate_fields(self, u_d, v_d, u_previous_d, v_previous_d, k):
        """Return velocity fields with outliers removed."""
        _check_inputs(u_d, v_d, array_type=gpuarray.GPUArray, shape=u_d.shape, dtype=DTYPE_f, ndim=2)
        m, n = u_d.shape

        if self.val_tols[0] is not None and self.nb_validation_iter > 0:
            self.sig2noise = self.corr.get_sig2noise(method=self.sig2noise_method, mask_width=self.s2n_width)

        for i in range(self.nb_validation_iter):
            # Get list of places that need to be validated.
            # TODO validation should be done on one field at a time
            val_locations_d, u_mean_d, v_mean_d = gpu_validation(u_d, v_d, m, n, self.window_size[k], self.sig2noise,
                                                                 *self.val_tols)

            # Do the validation.
            total_vectors = m * n
            n_val = total_vectors - int(gpuarray.sum(val_locations_d).get())
            if n_val > 0:
                logging.info('Validating {} out of {} vectors ({:.2%}).'.format(n_val, total_vectors, n_val / (m * n)))

                u_d, v_d = _gpu_replace_vectors(self.x_d[k], self.y_d[k], self.x_d[k - 1], self.y_d[k - 1], u_d, v_d,
                                                u_previous_d, v_previous_d, val_locations_d, u_mean_d,
                                                v_mean_d, self.n_row, self.n_col, k)

                logging.info('[DONE]\n')
            else:
                logging.info('No invalid vectors!')

        return u_d, v_d

    def _get_corr_arguments(self, dp_x_d, dp_y_d, k):
        """Returns the shift and strain arguments to the correlation class."""
        # Check if extended search area is used for first iteration.
        shift_d = None
        strain_d = None
        extended_size = None
        if k == 0:
            if self.extend_ratio is not None:
                extended_size = self.window_size[k] * self.extend_ratio
        else:
            _check_inputs(dp_x_d, dp_y_d, array_type=gpuarray.GPUArray, shape=dp_x_d.shape, dtype=DTYPE_f, ndim=2)
            m, n = dp_x_d.shape

            # Compute the shift.
            shift_d = gpuarray.empty((2, m, n), dtype=DTYPE_f)
            shift_d[0, :, :] = dp_x_d
            shift_d[1, :, :] = dp_y_d

            # Compute the strain rate.
            if self.deform:
                strain_d = gpu_strain(dp_x_d, dp_y_d, self.spacing[k])

        return extended_size, shift_d, strain_d

    def _get_next_iteration_prediction(self, u_d, v_d, k):
        """Returns the velocity field to begin the next iteration."""
        _check_inputs(u_d, v_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, shape=u_d.shape, ndim=2)
        # Interpolate if dimensions do not agree
        if self.window_size[k + 1] != self.window_size[k]:
            # Interpolate velocity onto next iterations grid. Then use it as the predictor for the next step.
            u_d = gpu_interpolate(self.x_d[k], self.y_d[k], self.x_d[k + 1], self.y_d[k + 1], u_d)
            v_d = gpu_interpolate(self.x_d[k], self.y_d[k], self.x_d[k + 1], self.y_d[k + 1], v_d)

        if self.smooth:
            dp_x_d = gpu_smooth(u_d, s=self.smoothing_par)
            dp_y_d = gpu_smooth(v_d, s=self.smoothing_par)
        else:
            dp_x_d = u_d.copy()
            dp_y_d = v_d.copy()

        return dp_x_d, dp_y_d

    def _update_values(self, i_peak_d, j_peak_d, dp_x_d, dp_y_d, k):
        """Updates the velocity values after each iteration."""
        _check_inputs(i_peak_d, j_peak_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, shape=i_peak_d.shape)
        if dp_x_d == dp_y_d is None:
            # TODO need variable self.field_shape
            dp_x_d = gpuarray.zeros((int(self.n_row[k]), int(self.n_col[k])), dtype=DTYPE_f)
            dp_y_d = gpuarray.zeros((int(self.n_row[k]), int(self.n_col[k])), dtype=DTYPE_f)
        else:
            _check_inputs(dp_x_d, dp_y_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, shape=dp_x_d.shape)
        size = DTYPE_i(dp_x_d.size)

        u_d = gpuarray.empty_like(dp_x_d, dtype=DTYPE_f)
        v_d = gpuarray.empty_like(dp_y_d, dtype=DTYPE_f)

        mod_update = SourceModule("""
            __global__ void update_values(float *f_new, float *f_old, float *peak, int *mask, int size)
            {
                // u_new : output argument
                int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (t_idx >= size) {return;}
    
                f_new[t_idx] = (f_old[t_idx] + peak[t_idx]) * mask[t_idx];
            }
            """)
        block_size = 32
        x_blocks = int(self.n_col[k] * self.n_row[k] // block_size + 1)
        update_values = mod_update.get_function("update_values")
        update_values(u_d, dp_x_d, j_peak_d, self.field_mask_d[k], size, block=(block_size, 1, 1), grid=(x_blocks, 1))
        update_values(v_d, dp_y_d, i_peak_d, self.field_mask_d[k], size, block=(block_size, 1, 1), grid=(x_blocks, 1))

        return u_d, v_d

    @staticmethod
    def _log_residual(i_peak_d, j_peak_d):
        """Normalizes the residual by the maximum quantization error of 0.5 pixel."""
        _check_inputs(i_peak_d, j_peak_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, shape=i_peak_d.shape)

        try:
            normalized_residual = sqrt(int(gpuarray.sum(i_peak_d ** 2 + j_peak_d ** 2).get()) / i_peak_d.size) / 0.5
            logging.info("[DONE]--Normalized residual : {}.\n".format(normalized_residual))
        except OverflowError:
            logging.warning('[DONE]--Overflow in residuals.\n')
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
    m, n : int
        Shape of the resulting flow field.

    """
    assert window_size >= 8, "Window size is too small."
    assert window_size % 8 == 0, "Window size must be a multiple of 8."
    assert int(spacing) == spacing, 'overlap_ratio must be an int.'

    m = int((image_size[0] - spacing) // spacing)
    n = int((image_size[1] - spacing) // spacing)
    return m, n


def get_field_coords(field_shape, window_size, overlap_ratio):
    """Returns the coordinates of the resulting velocity field.

    Parameters
    ----------
    field_shape : tuple
        (m, n), the shape of the resulting flow field.
    window_size : int
        Size of the interrogation windows.
    overlap_ratio : float
        Ratio by which two adjacent interrogation windows overlap.

    Returns
    -------
    x, y : ndarray
        2D float, Shape of the resulting flow field

    """
    assert window_size >= 8, "Window size is too small."
    assert window_size % 8 == 0, "Window size must be a multiple of 8."
    assert 0 <= overlap_ratio < 1, 'overlap_ratio should be a float between 0 and 1.'
    m, n = field_shape

    spacing = DTYPE_i(window_size * (1 - overlap_ratio))
    x = np.tile(np.linspace(window_size / 2, window_size / 2 + spacing * (n - 1), n, dtype=DTYPE_f), (m, 1))
    y = np.tile(np.linspace(window_size / 2 + spacing * (m - 1), window_size / 2, m, dtype=DTYPE_f), (n, 1)).T

    return x, y


def gpu_mask(frame_d, mask_d):
    """Multiply two integer-type arrays.

    Parameters
    ----------
    frame_d : GPUArray
        2D int, frame to be masked.
    mask_d : GPUArray
        2D int, mask to apply to frame.

    Returns
    -------
    GPUArray
        2D int, masked frame.

    """
    _check_inputs(frame_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, shape=frame_d.shape, ndim=2)
    _check_inputs(mask_d, array_type=gpuarray.GPUArray, dtype=DTYPE_i, shape=frame_d.shape, ndim=2)

    size = DTYPE_f(frame_d.size)
    m, n = frame_d.shape
    frame_masked_d = gpuarray.empty_like(frame_d, dtype=DTYPE_f)

    mod_mask = SourceModule("""
        __global__ void mask_frame_gpu(float *frame_masked, float *frame, int *mask, int size)
        {
            // frame_masked : output argument
            int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (t_idx >= size) {return;}

            frame_masked[t_idx] = frame[t_idx] * mask[t_idx];
        }
        """)
    block_size = 32
    x_blocks = int(n * m // block_size + 1)
    mask_frame_gpu = mod_mask.get_function("mask_frame_gpu")
    mask_frame_gpu(frame_masked_d, frame_d, mask_d, size, block=(block_size, 1, 1), grid=(x_blocks, 1))

    return frame_masked_d


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
    _check_inputs(u_d, v_d, array_type=gpuarray.GPUArray, shape=u_d.shape, dtype=DTYPE_f)
    assert spacing > 0, 'Spacing must be greater than 0.'

    m, n = u_d.shape
    strain_d = gpuarray.empty((4, m, n), dtype=DTYPE_f)

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
        if (col == 0) {strain[row * n] = (u[row * n + 1] - u[row * n]) / h;  // u_x
        strain[size * 2 + row * n] = (v[row * n + 1] - v[row * n]) / h;  // v_x

        // last column
        } else if (col == n - 1) {strain[(row + 1) * n - 1] = (u[(row + 1) * n - 1] - u[(row + 1) * n - 2]) / h;  // u_x
        strain[size * 2 + (row + 1) * n - 1] = (v[(row + 1) * n - 1] - v[(row + 1) * n - 2]) / h;  // v_x

        // main body
        } else {strain[row * n + col] = (u[row * n + col + 1] - u[row * n + col - 1]) / 2 / h;  // u_x
        strain[size * 2 + row * n + col] = (v[row * n + col + 1] - v[row * n + col - 1]) / 2 / h;  // v_x
        }

        // y-axis
        // first row
        if (row == 0) {strain[size + col] = (u[n + col] - u[col]) / h;  // u_y
        strain[size * 3 + col] = (v[n + col] - v[col]) / h;  // v_y

        // last row
        } else if (row == m - 1) {strain[size + n * (m - 1) + col] = (u[n * (m - 1) + col] - u[n * (m - 2) + col]) / h;
        // u_y
        strain[size * 3 + n * (m - 1) + col] = (v[n * (m - 1) + col] - v[n * (m - 2) + col]) / h;  // v_y

        // main body
        } else {strain[size + row * n + col] = (u[(row + 1) * n + col] - u[(row - 1) * n + col]) / 2 / h;  // u_y
        strain[size * 3 + row * n + col] = (v[(row + 1) * n + col] - v[(row - 1) * n + col]) / 2 / h;  // v_y
        }
    }
    """)
    block_size = 32
    n_blocks = int((m * n) // block_size + 1)
    strain_gpu = mod_strain.get_function('strain_gpu')
    strain_gpu(strain_d, u_d, v_d, DTYPE_f(spacing), DTYPE_i(m), DTYPE_i(n), block=(block_size, 1, 1),
               grid=(n_blocks, 1))

    return strain_d


def gpu_smooth(f_d, s=0.5):
    """Smooths a scalar field stored as a GPUArray.

    Parameters
    ----------
    f_d : GPUArray
        Field to be smoothed.
    s : float, optional
        Smoothing parameter in smoothn.

    Returns
    -------
    GPUArray
        Float, same size as f_d. Smoothed field.

    """
    assert type(f_d) == gpuarray.GPUArray, 'Input must a GPUArray.'
    assert f_d.dtype == DTYPE_f, 'Input array must float type.'
    assert len(f_d.shape), 'Inputs must be 2D.'
    assert s > 0, 'Smoothing parameter must be greater than 0.'

    f = f_d.get()
    f_smooth_d = gpuarray.to_gpu(smoothn(f, s=s)[0].astype(DTYPE_f, order='C'))  # Smoothn returns F-ordered array.

    return f_smooth_d


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
    _check_inputs(correlation_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=3)
    n_windows, ht, wd = correlation_d.shape
    window_size_i = DTYPE_i(ht * wd)

    correlation_shift_d = gpuarray.empty_like(correlation_d)

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
    block_size = 8
    grid_size = ceil(max(ht, wd) / block_size)
    fft_shift = mod_fft_shift.get_function('fft_shift')
    fft_shift(correlation_shift_d, correlation_d, DTYPE_i(ht), DTYPE_i(wd), window_size_i,
              block=(block_size, block_size, 1), grid=(n_windows, grid_size, grid_size))

    return correlation_shift_d


def _window_slice(frame_d, window_size, spacing, buffer):
    """Creates a 3D array stack of all the interrogation windows.

    Parameters
    -----------
    frame_d : GPUArray
        2D int, frame to create windows from.
    window_size :

    spacing :

    buffer :


    Returns
    -------
    GPUArray
        3D float, interrogation windows stacked on each other.

    """
    _check_inputs(frame_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=2)
    ht, wd = frame_d.shape
    m, n = get_field_shape((ht, wd), window_size, spacing)
    n_windows = m * n

    win_d = gpuarray.empty((n_windows, window_size, window_size), dtype=DTYPE_f)

    mod_ws = SourceModule("""
        __global__ void window_slice(float *output, float *input, int ws, int spacing, int buffer, int n, int wd,
                            int ht)
    {
        // x blocks are windows; y and z blocks are x and y dimensions, respectively
        int idx_i = blockIdx.x;
        int idx_x = blockIdx.y * blockDim.x + threadIdx.x;
        int idx_y = blockIdx.z * blockDim.y + threadIdx.y;

        // do the mapping
        int x = (idx_i % n) * spacing + buffer + idx_x;
        int y = (idx_i / n) * spacing + buffer + idx_y;

        // find limits of domain
        int outside_range = (x >= 0 && x < wd && y >= 0 && y < ht);

        // indices of new array to map to
        int w_range = idx_i * ws * ws + ws * idx_y + idx_x;

        // apply the mapping
        output[w_range] = input[(y * wd + x) * outside_range] * outside_range;
    }
    """)
    # TODO this may not work anymore
    block_size = 8
    grid_size = int(window_size / block_size)
    window_slice_deform = mod_ws.get_function('window_slice')
    window_slice_deform(win_d, frame_d, DTYPE_i(window_size), DTYPE_i(spacing), DTYPE_i(buffer), DTYPE_i(n),
                        DTYPE_i(wd), DTYPE_i(ht), block=(block_size, block_size, 1),
                        grid=(int(n_windows), grid_size, grid_size))

    return win_d


def _window_slice_deform(frame_d, window_size, spacing, buffer, shift_factor, shift_d, strain_d=None):
    """Creates a 3D array stack of all the interrogation windows using shift and strain.

    Parameters
    -----------
    frame_d : GPUArray
        2D int, frame to create windows from.
    window_size :

    spacing :

    buffer :

    shift_factor : float

    shift_d : GPUArray
        3D float, shift of the second window.
    strain_d : GPUArray or None
        3D float, strain rate tensor. First dimension is (u_x, u_y, v_x, v_y).

    Returns
    -------
    GPUArray
        3D float, all interrogation windows stacked on each other.

    """
    _check_inputs(frame_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=2)
    ht, wd = frame_d.shape
    m, n = get_field_shape((ht, wd), window_size, spacing)
    n_windows = m * n

    win_d = gpuarray.empty((n_windows, window_size, window_size), dtype=DTYPE_f)

    if strain_d is None:
        strain_d = gpuarray.zeros((4, m, n), dtype=DTYPE_f)

    # TODO outside_range is inefficient
    mod_ws = SourceModule("""
        __global__ void window_slice_deform(float *output, float *input, float *shift, float *strain, float f,
                            int ws, int spacing, int buffer, int n_windows, int n, int wd, int ht)
    {
        // f : factor to apply to the shift and strain tensors
        // wd : width (number of columns in the full image)
        // h : height (number of rows in the full image)

        // x blocks are windows; y and z blocks are x and y dimensions, respectively
        int idx_i = blockIdx.x;  // window index
        int idx_x = blockIdx.y * blockDim.x + threadIdx.x;
        int idx_y = blockIdx.z * blockDim.y + threadIdx.y;

        // Loop through each interrogation window and apply the shift and deformation.
        // get the shift values
        float dx = shift[idx_i] * f;
        float dy = shift[n_windows + idx_i] * f;

        // get the strain tensor values
        float u_x = strain[idx_i] * f;
        float u_y = strain[n_windows + idx_i] * f;
        float v_x = strain[2 * n_windows + idx_i] * f;
        float v_y = strain[3 * n_windows + idx_i] * f;

        // compute the window vector
        float r_x = idx_x - ws / 2 + 0.5;  // r_x = x - x_c
        float r_y = idx_y - ws / 2 + 0.5;  // r_y = y - y_c

        // apply deformation operation
        float x_shift = idx_x + dx + r_x * u_x + r_y * u_y;  // r * du + dx
        float y_shift = idx_y + dy + r_x * v_x + r_y * v_y;  // r * dv + dy

        // do the mapping
        float x = (idx_i % n) * spacing + buffer + x_shift;
        float y = (idx_i / n) * spacing + buffer + y_shift;

        // do bilinear interpolation
        int x1 = floorf(x);
        int x2 = x1 + 1;
        int y1 = floorf(y);
        int y2 = y1 + 1;

        // find limits of domain
        int outside_range = (x1 >= 0 && x2 < wd && y1 >= 0 && y2 < ht);

        // terms of the bilinear interpolation. multiply by outside_range to avoid index error.
        float f11 = input[(y1 * wd + x1) * outside_range];
        float f21 = input[(y1 * wd + x2) * outside_range];
        float f12 = input[(y2 * wd + x1) * outside_range];
        float f22 = input[(y2 * wd + x2) * outside_range];

        // indices of image to map to
        int w_range = idx_i * ws * ws + ws * idx_y + idx_x;

        // Apply the mapping. Multiply by outside_range to set values outside the window to zero.
        output[w_range] = (f11 * (x2 - x) * (y2 - y) + f21 * (x - x1) * (y2 - y) + f12 * (x2 - x) * (y - y1) + f22
                          * (x - x1) * (y - y1)) * outside_range;
    }
    """)
    # TODO this may not work anymore -- may not be power of 2.
    block_size = 8
    grid_size = int(window_size / block_size)
    window_slice_deform = mod_ws.get_function('window_slice_deform')
    window_slice_deform(win_d, frame_d, shift_d, strain_d, DTYPE_f(shift_factor), DTYPE_i(window_size),
                        DTYPE_i(spacing), DTYPE_i(buffer), DTYPE_i(n_windows), DTYPE_i(n), DTYPE_i(wd), DTYPE_i(ht),
                        block=(block_size, block_size, 1), grid=(int(n_windows), grid_size, grid_size))

    return win_d


def _gpu_subpixel_gaussian(correlation_d, row_peak_d, col_peak_d, fft_size_x, fft_size_y):
    """Returns the subpixel position of the peaks using gaussian approximation.

    Parameters
    ----------
    correlation_d : GPUArray
        3D float, data from the window correlations.
    row_peak_d, col_peak_d : GPUArray
        1D int, location of the correlation peak.
    fft_size_x, fft_size_y : int
        Size of the fft domain.

    Returns
    -------
    row_sp_d, col_sp_d : GPUArray
        1D float, row and column positions of the subpixel peak.

    """
    _check_inputs(correlation_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=3)
    _check_inputs(row_peak_d, col_peak_d, array_type=gpuarray.GPUArray, dtype=DTYPE_i, shape=(correlation_d.shape[0],),
                  ndim=1)
    n_windows, ht, wd = correlation_d.shape
    window_size_i = DTYPE_i(ht * wd)

    row_sp_d = gpuarray.empty_like(row_peak_d, dtype=DTYPE_f)
    col_sp_d = gpuarray.empty_like(col_peak_d, dtype=DTYPE_f)

    # TODO use one-sided gaussian estimation for edge cases. Or return just the edge peak.
    # TODO simplify the logic in the kernel since it probably doesn't improve the accuracy that much.
    # TODO add machinery to do parabolic approximation if correlation is negative.
    mod_subpixel_approximation = SourceModule("""
        __global__ void subpixel_approximation(float *row_sp, float *col_sp, int *row_p, int *col_p, float *corr,
                            int ht, int wd, int n_windows, int ws, int fft_size_x, int fft_size_y)
    {
        // x blocks are windows; y and z blocks are x and y dimensions, respectively.
        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (w_idx >= n_windows) {return;}
        float small = 1e-20;
        
        // Compute the index mapping.
        int row = row_p[w_idx];
        int col = col_p[w_idx];
        int row_tmp = row;
        int col_tmp = col;
        
        if (row_tmp < 1) {row_tmp = 1;}
        if (row_tmp > fft_size_y - 2) {row_tmp = fft_size_y - 2;}
        if (col_tmp < 1) {col_tmp = 1;}
        if (col_tmp > fft_size_x - 2) {col_tmp = fft_size_x - 2;}
        
        // Get the neighbouring correlation values.
        float c = corr[ws * w_idx + wd * row_tmp + col_tmp];
        float cd = corr[ws * w_idx + wd * (row_tmp - 1) + col_tmp];
        float cu = corr[ws * w_idx + wd * (row_tmp + 1) + col_tmp];
        float cl = corr[ws * w_idx + wd * row_tmp + col_tmp - 1];
        float cr = corr[ws * w_idx + wd * row_tmp + col_tmp + 1];
        
        // Convert negative values to zero
        int non_zero = c > 0;
        if (c <= 0) {c = small;}
        if (cl <= 0) {cl = small;}
        if (cr <= 0) {cr = small;}
        if (cd <= 0) {cd = small;}
        if (cu <= 0) {cu = small;}
        
        // Compute the subpixel value.
        row_sp[w_idx] = row + ((logf(cd) - logf(cu)) / (2 * logf(cd) - 4 * logf(c) + 2 * logf(cu) + small)) * non_zero;
        col_sp[w_idx] = col + ((logf(cl) - logf(cr)) / (2 * logf(cl) - 4 * logf(c) + 2 * logf(cr) + small)) * non_zero;
    }
    """)
    block_size = 32
    x_blocks = ceil(n_windows / block_size)
    subpixel_approximation = mod_subpixel_approximation.get_function('subpixel_approximation')
    subpixel_approximation(row_sp_d, col_sp_d, row_peak_d, col_peak_d, correlation_d, DTYPE_i(ht), DTYPE_i(wd),
                           DTYPE_i(n_windows), window_size_i, DTYPE_i(fft_size_x), DTYPE_i(fft_size_y),
                           block=(block_size, 1, 1), grid=(x_blocks, 1))

    return row_sp_d, col_sp_d


def _gpu_subpixel_centroid(correlation_d, row_peak_d, col_peak_d, fft_size_x, fft_size_y):
    """Returns the subpixel position of the peaks using centroid approximation.

    Parameters
    ----------
    correlation_d : GPUArray
        3D float, data from the window correlations.
    row_peak_d, col_peak_d : GPUArray
        1D int, location of the correlation peak.
    fft_size_x, fft_size_y : int
        Size of the fft domain.

    Returns
    -------
    row_sp_d, col_sp_d : GPUArray
        1D float, row and column positions of the subpixel peak.

    """
    _check_inputs(correlation_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=3)
    _check_inputs(row_peak_d, col_peak_d, array_type=gpuarray.GPUArray, dtype=DTYPE_i, shape=(correlation_d.shape[0],),
                  ndim=1)
    n_windows, ht, wd = correlation_d.shape
    window_size_i = DTYPE_i(ht * wd)

    row_sp_d = gpuarray.empty_like(row_peak_d, dtype=DTYPE_f)
    col_sp_d = gpuarray.empty_like(col_peak_d, dtype=DTYPE_f)

    # TODO use one-sided gaussian estimation for edge cases. Or return just the edge peak.
    # TODO simplify the logic in the kernel since it probably doesn't improve the accuracy that much.
    mod_subpixel_approximation = SourceModule("""
        __global__ void subpixel_approximation(float *row_sp, float *col_sp, int *row_p, int *col_p, float *corr,
                            int ht, int wd, int n_windows, int ws, int fft_size_x, int fft_size_y)
    {
        // x blocks are windows; y and z blocks are x and y dimensions, respectively.
        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (w_idx >= n_windows) {return;}
        float small = 1e-20;

        // Compute the index mapping.
        int row = row_p[w_idx];
        int col = col_p[w_idx];
        int row_tmp = row;
        int col_tmp = col;

        if (row_tmp < 1) {row_tmp = 1;}
        if (row_tmp > fft_size_y - 2) {row_tmp = fft_size_y - 2;}
        if (col_tmp < 1) {col_tmp = 1;}
        if (col_tmp > fft_size_x - 2) {col_tmp = fft_size_x - 2;}

        // Get the neighbouring correlation values.
        float c = corr[ws * w_idx + wd * row_tmp + col_tmp];
        float cd = corr[ws * w_idx + wd * (row_tmp - 1) + col_tmp];
        float cu = corr[ws * w_idx + wd * (row_tmp + 1) + col_tmp];
        float cl = corr[ws * w_idx + wd * row_tmp + col_tmp - 1];
        float cr = corr[ws * w_idx + wd * row_tmp + col_tmp + 1];

        // Convert negative values to zero
        int non_zero = c > 0;
        if (c <= 0) {c = small;}
        if (cl <= 0) {cl = small;}
        if (cr <= 0) {cr = small;}
        if (cd <= 0) {cd = small;}
        if (cu <= 0) {cu = small;}

        // Compute the subpixel value.
        row_sp[w_idx] = (row - 1) * cd + row * c + (row + 1) * cu) / (cd + c + cu + small);
        col_sp[w_idx] = (col - 1) * cl + col * c + (col + 1) * cr) / (cl + c + cr + small);
    }
    """)
    block_size = 32
    x_blocks = ceil(n_windows / block_size)
    subpixel_approximation = mod_subpixel_approximation.get_function('subpixel_approximation')
    subpixel_approximation(row_sp_d, col_sp_d, row_peak_d, col_peak_d, correlation_d, DTYPE_i(ht), DTYPE_i(wd),
                           DTYPE_i(n_windows), window_size_i, DTYPE_i(fft_size_x), DTYPE_i(fft_size_y),
                           block=(block_size, 1, 1), grid=(x_blocks, 1))

    return row_sp_d, col_sp_d


def _gpu_subpixel_parabolic(correlation_d, row_peak_d, col_peak_d, fft_size_x, fft_size_y):
    """Returns the subpixel position of the peaks using parabolic approximation.

    Parameters
    ----------
    correlation_d : GPUArray
        3D float, data from the window correlations.
    row_peak_d, col_peak_d : GPUArray
        1D int, location of the correlation peak.
    fft_size_x, fft_size_y : int
        Size of the fft domain.

    Returns
    -------
    row_sp_d, col_sp_d : GPUArray
        1D float, row and column positions of the subpixel peak.

    """
    _check_inputs(correlation_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=3)
    _check_inputs(row_peak_d, col_peak_d, array_type=gpuarray.GPUArray, dtype=DTYPE_i, shape=(correlation_d.shape[0],),
                  ndim=1)
    n_windows, ht, wd = correlation_d.shape
    window_size_i = DTYPE_i(ht * wd)

    row_sp_d = gpuarray.empty_like(row_peak_d, dtype=DTYPE_f)
    col_sp_d = gpuarray.empty_like(col_peak_d, dtype=DTYPE_f)

    # TODO use one-sided gaussian estimation for edge cases. Or return just the edge peak.
    # TODO simplify the logic in the kernel since it probably doesn't improve the accuracy that much.
    mod_subpixel_approximation = SourceModule("""
        __global__ void subpixel_approximation(float *row_sp, float *col_sp, int *row_p, int *col_p, float *corr,
                            int ht, int wd, int n_windows, int ws, int fft_size_x, int fft_size_y)
    {
        // x blocks are windows; y and z blocks are x and y dimensions, respectively.
        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (w_idx >= n_windows) {return;}
        float small = 1e-20;

        // Compute the index mapping.
        int row = row_p[w_idx];
        int col = col_p[w_idx];
        int row_tmp = row;
        int col_tmp = col;

        if (row_tmp < 1) {row_tmp = 1;}
        if (row_tmp > fft_size_y - 2) {row_tmp = fft_size_y - 2;}
        if (col_tmp < 1) {col_tmp = 1;}
        if (col_tmp > fft_size_x - 2) {col_tmp = fft_size_x - 2;}

        // Get the neighbouring correlation values.
        float c = corr[ws * w_idx + wd * row_tmp + col_tmp];
        float cd = corr[ws * w_idx + wd * (row_tmp - 1) + col_tmp];
        float cu = corr[ws * w_idx + wd * (row_tmp + 1) + col_tmp];
        float cl = corr[ws * w_idx + wd * row_tmp + col_tmp - 1];
        float cr = corr[ws * w_idx + wd * row_tmp + col_tmp + 1];

        // Convert negative values to zero
        int non_zero = c > 0;
        if (c <= 0) {c = small;}
        if (cl <= 0) {cl = small;}
        if (cr <= 0) {cr = small;}
        if (cd <= 0) {cd = small;}
        if (cu <= 0) {cu = small;}

        // Compute the subpixel value.
        row_sp[w_idx] = row + (cd - cu) / (2 * cd - 4 * c + 2 * cu + small)) * non_zero;
        col_sp[w_idx] = col + (cl - cr) / (2 * cl - 4 * c + 2 * cr + small)) * non_zero;
    }
    """)
    block_size = 32
    x_blocks = ceil(n_windows / block_size)
    subpixel_approximation = mod_subpixel_approximation.get_function('subpixel_approximation')
    subpixel_approximation(row_sp_d, col_sp_d, row_peak_d, col_peak_d, correlation_d, DTYPE_i(ht), DTYPE_i(wd),
                           DTYPE_i(n_windows), window_size_i, DTYPE_i(fft_size_x), DTYPE_i(fft_size_y),
                           block=(block_size, 1, 1), grid=(x_blocks, 1))

    return row_sp_d, col_sp_d


def gpu_interpolate_replace(x0_d, y0_d, x1_d, y1_d, f0_d, f1_d, val_locations_d):
    """Replaces the invalid vectors by interpolating another field.

    The implementation requires that the mesh spacing is uniform. The spacing can be different in x and y directions.

    Parameters
    ----------
    x0_d, y0_d : GPUArray
        1D float, grid coordinates of the original field
    x1_d, y1_d : GPUArray
        1D float, grid coordinates of the field to be interpolated.
    f0_d : GPUArray
        2D float, field to be interpolated.
    f1_d : GPUArray
        2D float, field to be validated
    val_locations_d : GPUArray
        Locations of the valid vectors is to be done.

    Returns
    -------
    GPUArray
        2D float, interpolated field.

    """
    _check_inputs(x0_d, y0_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=1)
    _check_inputs(x1_d, y1_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=1)
    _check_inputs(f0_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=2)
    _check_inputs(f1_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=2)
    _check_inputs(val_locations_d, array_type=gpuarray.GPUArray, dtype=DTYPE_i, shape=f1_d.shape, ndim=2)
    val_locations_inverse_d = 1 - val_locations_d

    f1_val_d = gpu_interpolate(x0_d, y0_d, x1_d, y1_d, f0_d)

    # Replace vectors be at validation locations.
    f1_val_d = (f1_d * val_locations_d).astype(DTYPE_f) + (f1_val_d * val_locations_inverse_d).astype(DTYPE_f)

    return f1_val_d


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
    _check_inputs(x0_d, y0_d, x1_d, y1_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=1)
    _check_inputs(f0_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=2)

    ht_i = DTYPE_i(y0_d.size)
    wd_i = DTYPE_i(x0_d.size)
    n = x1_d.size
    m = y1_d.size
    size_i = DTYPE_i(m * n)

    f1_d = gpuarray.empty((m, n), dtype=DTYPE_f)

    # TODO is this most efficient implementation?
    # Calculate the relationship between the two grid coordinates.
    buffer_x_f = DTYPE_f((x0_d[0]).get())
    buffer_y_f = DTYPE_f((y0_d[0]).get())
    spacing_x_f = DTYPE_f((x0_d[1].get() - buffer_x_f))
    spacing_y_f = DTYPE_f((y0_d[1].get() - buffer_y_f))

    # TODO use if statements
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
        int inside_left = x >= 0;
        int inside_right = x < wd - 1;
        int inside_top = y >= 0;
        int inside_bottom = y < ht - 1;
        x = x * inside_left * inside_right + (1 - inside_right) * (wd - 1);
        y = y * inside_top * inside_bottom + (1 - inside_bottom) * (ht - 1);
        
        // Do bilinear interpolation.
        int x1 = floorf(x - (1 - inside_right));
        int x2 = x1 + 1;
        int y1 = floorf(y - (1 - inside_bottom));
        int y2 = y1 + 1;

        // Terms of the bilinear interpolation. Multiply by outside_range to avoid index error.
        float f11 = f0[(y1 * wd + x1)];
        float f21 = f0[(y1 * wd + x2)];
        float f12 = f0[(y2 * wd + x1)];
        float f22 = f0[(y2 * wd + x2)];

        // Apply the mapping. Multiply by outside_range to set values outside the window to zero.
        f1[t_idx] = f11 * (x2 - x) * (y2 - y) + f21 * (x - x1) * (y2 - y) + f12 * (x2 - x) * (y - y1) + f22 * (x - x1)
                    * (y - y1);
    }
    """)
    block_size = 32
    x_blocks = int(size_i // block_size + 1)
    interpolate_gpu = mod_interpolate.get_function('bilinear_interpolation')
    interpolate_gpu(f1_d, f0_d, x1_d, y1_d, buffer_x_f, buffer_y_f, spacing_x_f, spacing_y_f, ht_i, wd_i, DTYPE_i(n),
                    size_i, block=(block_size, 1, 1), grid=(x_blocks, 1))

    return f1_d


# TODO this shouldn't depend on k, or else there should be a public version which doesn't
# TODO do multiple replacement methods actually improve results?
def _gpu_replace_vectors(x1_d, y1_d, x0_d, y0_d, u_d, v_d, u_previous_d, v_previous_d, val_locations_d, u_mean_d,
                         v_mean_d, n_row,
                         n_col, k):
    """Replace spurious vectors by the mean or median of the surrounding points.

    Parameters
    ----------
    x0_d, y0_d : GPUArray
        3D float, grid coordinates of previous iteration.
    x1_d, y1_d : GPUArray
        3D float, grid coordinates of current iteration.
    u_d, v_d : GPUArray
        2D float, velocities at current iteration.
    u_previous_d, v_previous_d
        2D float, velocities at previous iteration.
    val_locations_d : ndarray
        2D int, indicates which values must be validated. 1 indicates no validation needed, 0 indicates validation is
        needed.
    u_mean_d, v_mean_d : GPUArray
        3D float, mean velocity surrounding each point.
    n_row, n_col : ndarray
        int, number of rows and columns in each main loop iteration.
    k : int
        Main loop iteration count.

    Returns
    -------
    u_d, v_d : GPUArray
        2D float, velocity fields with replaced vectors.

    """
    # TODO cast to float beforehand?
    val_locations_inverse_d = 1 - val_locations_d

    # First iteration, just replace with mean velocity.
    if k == 0:
        u_d = (val_locations_inverse_d * u_mean_d).astype(DTYPE_f) + (val_locations_d * u_d).astype(DTYPE_f)
        v_d = (val_locations_inverse_d * v_mean_d).astype(DTYPE_f) + (val_locations_d * v_d).astype(DTYPE_f)

    # Case if different dimensions: interpolation using previous iteration.
    elif k > 0 and (n_row[k] != n_row[k - 1] or n_col[k] != n_col[k - 1]):
        # TODO can avoid slicing here
        u_d = gpu_interpolate_replace(x0_d, y0_d, x1_d, y1_d, u_previous_d, u_d,
                                      val_locations_d=val_locations_d)
        v_d = gpu_interpolate_replace(x0_d, y0_d, x1_d, y1_d, v_previous_d, v_d,
                                      val_locations_d=val_locations_d)

    # Case if same dimensions.
    # TODO cast to float beforehand?
    elif k > 0 and (n_row[k] == n_row[k - 1] or n_col[k] == n_col[k - 1]):
        u_d = (val_locations_inverse_d * u_previous_d).astype(DTYPE_f) + (val_locations_d * u_d).astype(DTYPE_f)
        v_d = (val_locations_inverse_d * v_previous_d).astype(DTYPE_f) + (val_locations_d * v_d).astype(DTYPE_f)

    return u_d, v_d


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
    _check_inputs(correlation_positive_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=3)
    n_windows, ht, wd = correlation_positive_d.shape
    _check_inputs(row_peak_d, col_peak_d, array_type=gpuarray.GPUArray, dtype=DTYPE_i, shape=(n_windows,))
    assert 0 <= mask_width < int(min(ht, wd) / 2), \
        'Mask width must be integer from 0 and to less than half the correlation window height or width.' \
        'Recommended value is 2. '
    size_i = DTYPE_i(ht * wd)
    mask_dim_i = DTYPE_i(mask_width * 2 + 1)

    correlation_masked_d = correlation_positive_d.copy()

    mod_mask_peak = SourceModule("""
        __global__ void mask_peak(float *corr, int *row_p, int *col_p, int mask_w, int ht, int wd, int mask_dim,
                            int size)
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
        corr[idx_i * size + row * wd + col] = 0;
    }
    """)
    block_size = 8
    grid_size = ceil(mask_dim_i / block_size)
    fft_shift = mod_mask_peak.get_function('mask_peak')
    fft_shift(correlation_masked_d, row_peak_d, col_peak_d, DTYPE_i(mask_width), DTYPE_i(ht), DTYPE_i(wd), mask_dim_i,
              size_i, block=(block_size, block_size, 1), grid=(n_windows, grid_size, grid_size))

    return correlation_masked_d


def _gpu_mask_rms(correlation_positive_d, corr_peak_d):
    """Returns correlation windows with values less than the primary peak.

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
    _check_inputs(correlation_positive_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, ndim=3)
    n_windows, ht, wd = correlation_positive_d.shape
    _check_inputs(corr_peak_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, shape=(n_windows,))
    size_i = DTYPE_i(ht * wd)

    correlation_masked_d = correlation_positive_d.copy()

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
        if (corr[idx] >= corr_p[idx_i] / 2) {corr[idx] = 0;}
    }
    """)
    block_size = 8
    grid_size = ceil(max(ht, wd) / block_size)
    fft_shift = mod_correlation_rms.get_function('correlation_rms')
    fft_shift(correlation_masked_d, corr_peak_d, DTYPE_i(ht), DTYPE_i(wd), size_i, block=(block_size, block_size, 1),
              grid=(n_windows, grid_size, grid_size))

    return correlation_masked_d
