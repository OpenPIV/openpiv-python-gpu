"""This module is dedicated to advanced algorithms for PIV image analysis with NVIDIA GPU Support.

All identifiers ending with '_d' exist on the GPU and not the CPU. The GPU is referred to as the device, and therefore
"_d" signifies that it is a device variable. Please adhere to this standard as it makes developing and debugging much
easier.

Note that all data must 32-bit at most to be stored on GPUs. Numpy types should be always 32-bit for compatibility
with GPU. Scalars should be python int type in general to work as function arguments. C-type scalars or arrays that are
arguments to GPU kernels should be identified with ending in either _i or _f.

"""
import time
import numpy as np
import numpy.ma as ma

import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
import skcuda.fft as cu_fft
import skcuda.misc as cu_misc
import logging
import nvidia_smi
from pycuda.compiler import SourceModule
from scipy.fft import fftshift
from math import sqrt, ceil
from openpiv.gpu_validation import gpu_validation
from openpiv.smoothn import smoothn as smoothn

# Define 32-bit types
DTYPE_i = np.int32
DTYPE_f = np.float32

# initialize the skcuda library
cu_misc.init()


class GPUCorrelation:
    """A class representing the cross correlation function.

    Parameters
    ----------
    frame_a_d, frame_b_d : GPUArray
        2D int, image pair.
    nfft_x : int or None, optional
        Window size multiplier for fft.

    Methods
    -------
    __call__(window_size, extended_size=None, d_shift=None, d_strain=None)
        Returns the peaks of the correlation windows.
    sig2noise_ratio(method='peak2peak', width=2)
        Returns the signal-to-noise ratio of the correlation peaks.

    """

    def __init__(self, frame_a_d, frame_b_d, nfft_x=None):
        _check_inputs(frame_a_d, frame_b_d, array_type=gpuarray.GPUArray, shape=frame_a_d.shape, dtype=DTYPE_f, dim=2)
        if nfft_x is None:
            self.nfft = 2
        else:
            assert (self.nfft & (self.nfft - 1)) == 0, 'nfft must be power of 2'
            self.nfft = nfft_x
        self.frame_a_d = frame_a_d
        self.frame_b_d = frame_b_d
        self.peak_row = None
        self.peak_col = None
        self.frame_shape = DTYPE_i(frame_a_d.shape)

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
        self.fft_size = DTYPE_i(self.size_extended * self.nfft)
        self.n_rows, self.n_cols = DTYPE_i(get_field_shape(self.frame_shape, self.window_size, self.overlap_ratio))
        self.n_windows = self.n_rows * self.n_cols

        # Return stack of all IWs
        win_a_d, win_b_d = self._iw_arrange(self.frame_a_d, self.frame_b_d, shift_d, strain_d)

        # normalize array by computing the norm of each IW
        win_a_norm_d, win_b_norm_d = self._normalize_intensity(win_a_d, win_b_d)

        # zero pad arrays
        win_a_zp_d, win_b_zp_d = self._zero_pad(win_a_norm_d, win_b_norm_d)

        # correlate Windows
        self.correlation_data = self._correlate_windows(win_a_zp_d, win_b_zp_d)

        # get first peak of correlation function
        self.peak_row, self.peak_col, self.corr_max1 = self._find_peak(self.correlation_data)

        # get the subpixel location
        row_sp, col_sp = self._subpixel_peak_location()

        # TODO this could be GPU array--would be faster?
        # reshape to field window coordinates
        i_peak = row_sp.reshape((self.n_rows, self.n_cols)) - self.fft_size / 2
        j_peak = col_sp.reshape((self.n_rows, self.n_cols)) - self.fft_size / 2

        return i_peak, j_peak

    def _iw_arrange(self, frame_a_d, frame_b_d, shift_d, strain_d):
        """Creates a 3D array stack of all the interrogation windows.

        This is necessary to do the FFTs all at once on the GPU. This populates interrogation windows from the origin
        of the image. The implementation requires that the window sizes are multiples of 4.

        Parameters
        -----------
        frame_a_d, frame_b_d : GPUArray
            2D int, image pair.
        shift_d : GPUArray
            3D float, shift of the second window.
        strain_d : GPUArray
            3D float, strain rate tensor. First dimension is (u_x, u_y, v_x, v_y).

        Returns
        -------
        win_a_d, win_b_d : GPUArray
            3D float, all interrogation windows stacked on each other.

        """
        _check_inputs(frame_a_d, frame_b_d, array_type=gpuarray.GPUArray, shape=frame_b_d.shape, dtype=DTYPE_f, dim=2)
        ht, wd = self.frame_shape
        spacing = DTYPE_i(self.window_size * (1 - self.overlap_ratio))
        diff_extended = DTYPE_i(spacing - self.size_extended / 2)

        win_a_d = gpuarray.zeros((self.n_windows, self.size_extended, self.size_extended), dtype=DTYPE_f)
        win_b_d = gpuarray.zeros((self.n_windows, self.size_extended, self.size_extended), dtype=DTYPE_f)

        mod_ws = SourceModule("""
            __global__ void window_slice(float *output, float *input, int ws, int spacing, int diff_extended, int n_col, int wd, int ht)
        {
            // x blocks are windows; y and z blocks are x and y dimensions, respectively
            int idx_i = blockIdx.x;
            int idx_x = blockIdx.y * blockDim.x + threadIdx.x;
            int idx_y = blockIdx.z * blockDim.y + threadIdx.y;
            
            // do the mapping
            int x = (idx_i % n_col) * spacing + diff_extended + idx_x;
            int y = (idx_i / n_col) * spacing + diff_extended + idx_y;
            
            // find limits of domain
            int outside_range = (x >= 0 && x < wd && y >= 0 && y < ht);

            // indices of new array to map to
            int w_range = idx_i * ws * ws + ws * idx_y + idx_x;
            
            // apply the mapping
            output[w_range] = input[(y * wd + x) * outside_range] * outside_range;
        }

            __global__ void window_slice_deform(float *output, float *input, float *shift, float *strain, float f, int ws, int spacing, int diff_extended, int n_col, int num_window, int wd, int ht)
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
            float dy = shift[num_window + idx_i] * f;

            // get the strain tensor values
            float u_x = strain[idx_i] * f;
            float u_y = strain[num_window + idx_i] * f;
            float v_x = strain[2 * num_window + idx_i] * f;
            float v_y = strain[3 * num_window + idx_i] * f;

            // compute the window vector
            float r_x = idx_x - ws / 2 + 0.5;  // r_x = x - x_c
            float r_y = idx_y - ws / 2 + 0.5;  // r_y = y - y_c

            // apply deformation operation
            float x_shift = idx_x + dx + r_x * u_x + r_y * u_y;  // r * du + dx
            float y_shift = idx_y + dy + r_x * v_x + r_y * v_y;  // r * dv + dy

            // do the mapping
            float x = (idx_i % n_col) * spacing + diff_extended + x_shift;
            float y = (idx_i / n_col) * spacing + diff_extended + y_shift;

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
            output[w_range] = (f11 * (x2 - x) * (y2 - y) + f21 * (x - x1) * (y2 - y) + f12 * (x2 - x) * (y - y1) + f22 * (x - x1) * (y - y1)) * outside_range;
        }
        """)
        # TODO this may not work anymore
        block_size = 8
        grid_size = int(self.size_extended / block_size)

        # Use translating windows.
        if shift_d is not None:
            # Factors to apply the symmetric shift.
            shift_factor_a = DTYPE_f(-0.5)
            shift_factor_b = DTYPE_f(0.5)

            if strain_d is None:
                strain_d = gpuarray.zeros((4, self.n_rows, self.n_cols), dtype=DTYPE_f)

            window_slice_deform = mod_ws.get_function('window_slice_deform')
            window_slice_deform(win_a_d, frame_a_d, shift_d, strain_d, shift_factor_a, self.size_extended, spacing,
                                diff_extended,
                                self.n_cols, self.n_windows, wd, ht, block=(block_size, block_size, 1),
                                grid=(int(self.n_windows), grid_size, grid_size))
            window_slice_deform(win_b_d, frame_b_d, shift_d, strain_d, shift_factor_b, self.size_extended, spacing,
                                diff_extended,
                                self.n_cols, self.n_windows, wd, ht, block=(block_size, block_size, 1),
                                grid=(int(self.n_windows), grid_size, grid_size))

        # Use non-translating windows.
        else:
            window_slice_deform = mod_ws.get_function('window_slice')
            window_slice_deform(win_a_d, frame_a_d, self.size_extended, spacing, diff_extended, self.n_cols, wd, ht,
                                block=(block_size, block_size, 1), grid=(int(self.n_windows), grid_size, grid_size))
            window_slice_deform(win_b_d, frame_b_d, self.size_extended, spacing, diff_extended, self.n_cols, wd, ht,
                                block=(block_size, block_size, 1), grid=(int(self.n_windows), grid_size, grid_size))

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

        mean_a_d = cu_misc.mean(win_a_d.reshape(self.n_windows, iw_size), axis=1)
        mean_b_d = cu_misc.mean(win_b_d.reshape(self.n_windows, iw_size), axis=1)

        mod_norm = SourceModule("""
            __global__ void normalize(float *array, float *array_norm, float *mean, int iw_size)
        {
            // global thread id for 1D grid of 2D blocks
            int thread_idx = blockIdx.x * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;

            // indices for mean matrix
            int mean_idx = thread_idx / iw_size;

            array_norm[thread_idx] = array[thread_idx] - mean[mean_idx];
        }
        
            __global__ void smart_normalize(float *array, float *array_norm, float *mean, float *mean_ratio, int iw_size)
        {
            // global thread id for 1D grid of 2D blocks
            int thread_idx = blockIdx.x * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;

            // indices for mean matrix
            int mean_idx = thread_idx / iw_size;

            array_norm[thread_idx] = array[thread_idx] * mean_ratio[mean_idx] - mean[mean_idx];
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
        s0_a = DTYPE_i((self.size_extended - self.window_size) / 2)
        s1_a = DTYPE_i(self.size_extended - s0_a)
        s0_b = DTYPE_i(0)
        s1_b = self.size_extended

        win_a_zp_d = gpuarray.zeros([self.n_windows, self.fft_size, self.fft_size], dtype=DTYPE_f)
        win_b_zp_d = gpuarray.zeros([self.n_windows, self.fft_size, self.fft_size], dtype=DTYPE_f)

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
        zero_pad(win_a_zp_d, win_a_norm_d, self.fft_size, self.size_extended, s0_a, s1_a,
                 block=(block_size, block_size, 1), grid=(int(self.n_windows), grid_size, grid_size))
        zero_pad(win_b_zp_d, win_b_norm_d, self.fft_size, self.size_extended, s0_b, s1_b,
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
        win_h = self.fft_size
        win_w = self.fft_size

        win_i_fft_d = gpuarray.empty((int(self.n_windows), int(win_h), int(win_w)), DTYPE_f)
        win_a_fft_d = gpuarray.empty((int(self.n_windows), int(win_h), int(win_w // 2 + 1)), np.complex64)
        win_b_fft_d = gpuarray.empty((int(self.n_windows), int(win_h), int(win_w // 2 + 1)), np.complex64)

        # forward FFTs
        plan_forward = cu_fft.Plan((win_h, win_w), DTYPE_f, np.complex64, self.n_windows)
        cu_fft.fft(win_a_zp_d, win_a_fft_d, plan_forward)
        cu_fft.fft(win_b_zp_d, win_b_fft_d, plan_forward)

        # multiply the FFTs
        win_a_fft_d = win_a_fft_d.conj()
        tmp_d = win_b_fft_d * win_a_fft_d

        # inverse transform
        plan_inverse = cu_fft.Plan((win_h, win_w), np.complex64, DTYPE_f, self.n_windows)
        cu_fft.ifft(tmp_d, win_i_fft_d, plan_inverse, True)
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
        w = DTYPE_i(self.fft_size / 2)

        # Get index and value of peak.
        corr_reshape_d = corr_d.reshape(int(self.n_windows), int(self.fft_size ** 2))
        max_idx_df = cu_misc.argmax(corr_reshape_d, axis=1).astype(DTYPE_f)
        max_peak_d = cu_misc.max(corr_reshape_d, axis=1).astype(DTYPE_f)

        # Row and column information of peak.
        # TODO account for the padded space
        col_peak_d, row_peak_d = cumath.modf((max_idx_df / DTYPE_i(self.fft_size)).astype(DTYPE_f))
        row_peak_d = row_peak_d.astype(DTYPE_i)
        col_peak_d = (col_peak_d * DTYPE_i(self.fft_size)).astype(DTYPE_i)

        # # # Return the center if the correlation peak is zero.
        zero_peak_inverse_d = (max_peak_d > 0.01).astype(DTYPE_i)
        zero_peak_d = ((1 - zero_peak_inverse_d) * w).astype(DTYPE_i)
        row_peak_d = row_peak_d * zero_peak_inverse_d + zero_peak_d
        col_peak_d = col_peak_d * zero_peak_inverse_d + zero_peak_d

        # return row_peak, col_peak, max_peak
        return row_peak_d.get(), col_peak_d.get(), max_peak_d.get()

    def _find_second_peak(self, width):
        """Find the value of the second-largest peak.

        The second-largest peak is the height of the peak in the region outside a "width * width" submatrix around
        the first correlation peak.

        Parameters
        ----------
        width : int
            Half size of the region around the first correlation peak to ignore for finding the second peak.

        Returns
        -------
        corr_max2 : int
            Value of the second correlation peak.

        """
        # create a masked view of the self.data array
        tmp = self.correlation_data.get().view(ma.MaskedArray)

        # set (width x width) square sub-matrix around the first correlation peak as masked
        for i in range(-width, width + 1):
            for j in range(-width, width + 1):
                rot_idx = self.peak_row.get() + i
                col_idx = self.peak_col.get() + j
                idx = (rot_idx >= 0) & (rot_idx < self.fft_size) & (col_idx >= 0) & (col_idx < self.fft_size)
                tmp[idx, rot_idx[idx], col_idx[idx]] = ma.masked

        row2, col2, corr_max2 = self._find_peak(tmp)

        return corr_max2.get()

    # TODO create gpu kernel doing same thing
    def _subpixel_peak_location(self):
        """Find subpixel peak approximation using Gaussian method.

        Returns
        -------
        row_sp, col_sp : ndarray
            2D float, location of peak to subpixel accuracy.

        """
        # Define small number to replace zeros and get rid of warnings in calculations.
        small = 1e-20

        # Cast to float.
        corr_c = self.correlation_data.get()
        row_c = self.peak_row.astype(DTYPE_f)
        col_c = self.peak_col.astype(DTYPE_f)

        # Move boundary peaks inward one node.
        row_tmp = np.copy(self.peak_row)
        row_tmp[row_tmp < 1] = 1
        row_tmp[row_tmp > self.fft_size - 2] = self.fft_size - 2
        col_tmp = np.copy(self.peak_col)
        col_tmp[col_tmp < 1] = 1
        col_tmp[col_tmp > self.fft_size - 2] = self.fft_size - 2

        # Initialize arrays.
        c = corr_c[range(self.n_windows), row_tmp, col_tmp]
        cl = corr_c[range(self.n_windows), row_tmp - 1, col_tmp]
        cr = corr_c[range(self.n_windows), row_tmp + 1, col_tmp]
        cd = corr_c[range(self.n_windows), row_tmp, col_tmp - 1]
        cu = corr_c[range(self.n_windows), row_tmp, col_tmp + 1]

        # Get rid of values that are zero or lower.
        non_zero = np.array(c > 0, dtype=DTYPE_f)
        c[c <= 0] = small
        cl[cl <= 0] = small
        cr[cr <= 0] = small
        cd[cd <= 0] = small
        cu[cu <= 0] = small

        # Do subpixel approximation. Add small to avoid zero divide.
        row_sp = row_c + ((np.log(cl) - np.log(cr))
                          / (2 * np.log(cl) - 4 * np.log(c) + 2 * np.log(cr) + small)) * non_zero
        col_sp = col_c + ((np.log(cd) - np.log(cu))
                          / (2 * np.log(cd) - 4 * np.log(c) + 2 * np.log(cu) + small)) * non_zero

        return row_sp, col_sp

    def sig2noise_ratio(self, method='peak2peak', width=2):
        """Computes the signal-to-noise ratio.

        The signal-to-noise ratio is computed from the correlation map with one of two available method. It is a measure
        of the quality of the matching between two interrogation windows.

        Parameters
        ----------
        method : string, optional
            Method for evaluating the signal to noise ratio value from
            the correlation map. Can be `peak2peak`, `peak2mean` or None
            if no evaluation should be made.
        width : int, optional
            Half size of the region around the first
            correlation peak to ignore for finding the second
            peak. [default: 2]. Only used if ``sig2noise_method==peak2peak``.

        Returns
        -------
        ndarray
            2D float, the signal-to-noise ratio from the correlation map for each vector.

        """
        # compute signal-to-noise ratio by the chosen method
        if method == 'peak2peak':
            corr_max2 = self._find_second_peak(width=width)
        elif method == 'peak2mean':
            corr_max2 = self.correlation_data.mean()
        else:
            raise ValueError('wrong sig2noise_method')

        # get rid on divide by zero
        corr_max2[corr_max2 == 0.0] = 1e-20

        # get signal to noise ratio
        sig2noise = self.corr_max1 / corr_max2

        # get rid of nan values. Set sig2noise to zero
        sig2noise[np.isnan(sig2noise)] = 0.0

        # if the image is lacking particles, it will correlate to very low value, but not zero
        # return zero, since we have no signal.
        sig2noise[self.corr_max1 < 1e-3] = 0.0

        # if the first peak is on the borders, the correlation map is wrong
        # return zero, since we have no signal.
        sig2noise[np.array(self.peak_row == 0) * np.array(self.peak_row == self.correlation_data.shape[1]) * np.array(
            self.peak_col == 0) * np.array(self.peak_col == self.correlation_data.shape[2])] = 0.0

        return sig2noise.reshape(self.n_rows, self.n_cols)


def gpu_extended_search_area(frame_a, frame_b,
                             window_size,
                             overlap_ratio,
                             dt,
                             search_area_size,
                             **kwargs
                             ):
    """The implementation of the one-step direct correlation with the same size windows.

    Support for extended search area of the second window has yet to be implimetned. This module is meant to be used
    with an iterative method to cope with the loss of pairs due to particle movement out of the search area.

    This function is an adaptation of the original extended_search_area_piv function rewritten with PyCuda and CUDA-C to run on an NVIDIA GPU.

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
        Method to estimate subpixel location of the peak. Gaussian is default if correlation map is positive. Centroid replaces default if correlation map is negative.
    width : int
        Half size of the region around the first correlation peak to ignore for finding the second peak. Default is 2. Only used if sig2noise_method==peak2peak.
    nfft_x : int
        Size of the 2D FFT in x-direction. The default of 2 x windows_a.shape[0] is recommended.

    Example
    --------
    >>> u, v = gpu_extended_search_area(frame_a, frame_b, window_size=16, overlap_ratio=0.5, search_area_size=32, dt=1)

    """
    # Extract the parameters
    nfft_x = kwargs['nfft_x'] if 'nfft_x' in kwargs else None

    # cast images as floats and sent to gpu
    frame_a_d = gpuarray.to_gpu(frame_a.astype(DTYPE_f))
    frame_b_d = gpuarray.to_gpu(frame_b.astype(DTYPE_f))

    # Get correlation function
    corr = GPUCorrelation(frame_a_d, frame_b_d, nfft_x)

    # Get window displacement to subpixel accuracy
    sp_i, sp_j = corr(window_size, overlap_ratio, search_area_size)

    # reshape the peaks
    i_peak = np.reshape(sp_i, (corr.n_rows, corr.n_cols))
    j_peak = np.reshape(sp_j, (corr.n_rows, corr.n_cols))

    # calculate velocity fields
    u = j_peak / dt
    v = -i_peak / dt

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
    Validation is done by any combination of signal-to-noise ratio, mean, median
    Smoothn can be used between iterations to improve the estimate and replace missing values.

    References
    ----------
    Scarano F, Riethmuller ML (1999) Iterative multigrid approach in PIV image processing with discrete window offset.
        Exp Fluids 26:513–523
    Meunier, P., & Leweke, T. (2003). Analysis and treatment of errors due to high velocity gradients in particle image velocimetry.
        Experiments in fluids, 35(5), 408-421.
    Garcia, D. (2010). Robust smoothing of gridded data in one and higher dimensions with missing values.
        Computational statistics & data analysis, 54(4), 1167-1178.

    Parameters
    ----------
    frame_a, frame_b : ndarray
        2D int, integers containing grey levels of the first and second frames.
    mask : ndarray or None, optional
        2D, int, array of integers with values 0 for the background, 1 for the flow-field. If the center of a window is on a 0 value the velocity is set to 0.
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
        Method used for validation. Only the mean velocity method is implemented now. The default tolerance is 2 for median validation.

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
        Ratio the extended search area to use on the first iteration. If not specified, extended search will not be used.
    subpixel_method : {'gaussian', 'centroid', 'parabolic'}
        Method to estimate subpixel location of the peak. Gaussian is default if correlation map is positive. Centroid replaces default if correlation map is negative.
    return_sig2noise : bool
        Sets whether to return the signal-to-noise ratio. Not returning the signal-to-noise speeds up computation significantly, which is default behaviour.
    sig2noise_method : {'peak2peak', 'peak2mean'}
        Method of signal-to-noise-ratio measurement.
    s2n_width : int
        Half size of the region around the first correlation peak to ignore for finding the second peak. Default is 2. Only used if sig2noise_method==peak2peak.
    nfftx : int
        Size of the 2D FFT in x-direction. The default of 2 x windows_a.shape[0] is recommended.

    Example
    -------
    >>> x, y, u, v, mask, s2n = gpu_piv(frame_a, frame_b, mask=None, window_size_iters=(1, 2), min_window_size=16, overlap_ratio=0.5, dt=1, deform=True, smooth=True, nb_validation_iter=2, validation_method='median_velocity', median_tol=2)

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
        2D, float. Array of integers with values 0 for the background, 1 for the flow-field. If the center of a window is on a 0 value the velocity is set to 0.
    deform : bool, optional
        Whether to deform the windows by the velocity gradient at each iteration.
    smooth : bool, optional
        Whether to smooth the intermediate fields.
    nb_validation_iter : int, optional
        Number of iterations per validation cycle.
    validation_method : {tuple, 's2n', 'median_velocity', 'mean_velocity', 'rms_velocity'}, optional
        Method used for validation. Only the mean velocity method is implemented now. The default tolerance is 2 for median validation.

    Other Parameters
    ----------------
    trust_1st_iter : bool
        With a first window size following the 1/4 rule, the 1st iteration can be trusted and the value should be 1.
    s2n_tol, median_tol, mean_tol, median_tol, rms_tol : float
        Tolerance of the validation methods.
    smoothing_par : float
        Smoothing parameter to pass to smoothn to apply to the intermediate velocity fields. Default is 0.5.
    extend_ratio : float
        Ratio the extended search area to use on the first iteration. If not specified, extended search will not be used.
    subpixel_method : {'gaussian', 'centroid', 'parabolic'}
        Method to estimate subpixel location of the peak. Gaussian is default if correlation map is positive. Centroid replaces default if correlation map is negative.
    sig2noise_method : {'peak2peak', 'peak2mean'}
        Method of signal-to-noise-ratio measurement.
    s2n_width : int
        Half size of the region around the first correlation peak to ignore for finding the second peak. Default is 2. Only used if sig2noise_method==peak2peak.
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
            ht, wd = frame_shape.shape
        else:
            ht, wd = frame_shape
        ws_iters = (window_size_iters,) if type(window_size_iters) == int else window_size_iters
        num_ws = len(ws_iters)
        self.overlap_ratio = overlap_ratio
        self.dt = dt
        self.deform = deform
        self.smooth = smooth
        self.nb_iter_max = nb_iter_max = sum(ws_iters)
        self.nb_validation_iter = nb_validation_iter

        if mask is not None:
            assert mask.shape == (ht, wd), 'Mask is not same shape as image.'

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

        # Other parameters.
        self.trust_1st_iter = kwargs['trust_first_iter'] if 'trust_first_iter' in kwargs else False
        self.smoothing_par = kwargs['smoothing_par'] if 'smoothing_par' in kwargs else 0.5
        self.sig2noise_method = kwargs['sig2noise_method'] if 'sig2noise_method' in kwargs else 'peak2peak'
        self.s2n_width = kwargs['s2n_width'] if 's2n_width' in kwargs else 2
        self.nfft_x = kwargs['nfft_x'] if 'nfft_x' in kwargs else None
        self.extend_ratio = kwargs['extend_ratio'] if 'extend_ratio' in kwargs else None
        self.im_mask = gpuarray.to_gpu(mask.astype(DTYPE_i)) if mask is not None else None
        self.corr = None
        self.sig2noise = None

        # Init spacing.
        self.spacing = np.zeros(nb_iter_max, dtype=DTYPE_i)
        for k in range(nb_iter_max):
            self.spacing[k] = self.window_size[k] - int(self.window_size[k] * overlap_ratio)

        # Init n_col and n_row.
        self.n_row = np.zeros(nb_iter_max, dtype=DTYPE_i)
        self.n_col = np.zeros(nb_iter_max, dtype=DTYPE_i)
        for k in range(nb_iter_max):
            self.n_row[k], self.n_col[k] = get_field_shape(frame_shape, self.window_size[k], overlap_ratio)

        # Initialize x, y and mask.
        # TODO use a setter to construct these objects
        self.x_d = []
        self.y_d = []
        self.field_mask_d = []
        for k in range(nb_iter_max):
            x, y = get_field_coords((self.n_row[k], self.n_col[k]), self.window_size[k], overlap_ratio)
            self.x_d.append(gpuarray.to_gpu(x))
            self.y_d.append(gpuarray.to_gpu(y))

            # create the mask arrays for each iteration
            if mask is not None:
                field_mask = mask[y.astype(DTYPE_i), x.astype(DTYPE_i)].astype(DTYPE_i)
            else:
                field_mask = np.ones((self.n_row[k], self.n_col[k]), dtype=DTYPE_i)
            self.field_mask_d.append(gpuarray.to_gpu(field_mask))

            if k == nb_iter_max - 1:
                self.x_final = x
                self.y_final = y
                self.mask_final = field_mask

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
        self.corr = GPUCorrelation(frame_a_d, frame_b_d, self.nfft_x)

        # MAIN LOOP
        for k in range(self.nb_iter_max):
            logging.info('ITERATION {}'.format(k))
            extended_size, shift_d, strain_d = self._get_corr_arguments(dp_x_d, dp_y_d, k)

            # Get window displacement to subpixel accuracy.
            i_peak, j_peak = self.corr(self.window_size[k], self.overlap_ratio, extended_size=extended_size,
                                       shift_d=shift_d, strain_d=strain_d)

            # update the field with new values
            u_d, v_d = self._update_values(i_peak, j_peak, dp_x_d, dp_y_d, k)
            self._log_residual(i_peak, j_peak)

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
            s2n = self.corr.sig2noise_ratio(method=self.sig2noise_method)
        return s2n

    def _mask_image(self, frame_a, frame_b):
        """Mask the images before sending to device."""
        _check_inputs(frame_a, frame_b, shape=frame_a.shape, dim=2)
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
        _check_inputs(u_d, v_d, array_type=gpuarray.GPUArray, shape=u_d.shape, dtype=DTYPE_f, dim=2)
        m, n = u_d.shape

        if self.val_tols[0] is not None and self.nb_validation_iter > 0:
            self.sig2noise = self.corr.sig2noise_ratio(method=self.sig2noise_method, width=self.s2n_width)

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
            _check_inputs(dp_x_d, dp_y_d, array_type=gpuarray.GPUArray, shape=dp_x_d.shape, dtype=DTYPE_f, dim=2)
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
        _check_inputs(u_d, v_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, shape=u_d.shape, dim=2)
        # Interpolate if dimensions do not agree
        if self.window_size[k + 1] != self.window_size[k]:
            # Interpolate velocity onto next iterations grid. Then use it as the predictor for the next step.
            u_d = gpu_interpolate(self.x_d[k][0, :], self.y_d[k][:, 0], self.x_d[k + 1][0, :], self.y_d[k + 1][:, 0],
                                  u_d)
            v_d = gpu_interpolate(self.x_d[k][0, :], self.y_d[k][:, 0], self.x_d[k + 1][0, :], self.y_d[k + 1][:, 0],
                                  v_d)

        if self.smooth:
            dp_x_d = gpu_smooth(u_d, s=self.smoothing_par)
            dp_y_d = gpu_smooth(v_d, s=self.smoothing_par)
        else:
            dp_x_d = u_d.copy()
            dp_y_d = v_d.copy()

        return dp_x_d, dp_y_d

    def _update_values(self, i_peak, j_peak, dp_x_d, dp_y_d, k):
        """Updates the velocity values after each iteration."""
        _check_inputs(i_peak, j_peak, array_type=np.ndarray, dtype=DTYPE_f, shape=i_peak.shape, dim=2)
        if dp_x_d == dp_y_d is None:
            # TODO need variable self.field_shape
            dp_x_d = gpuarray.zeros((int(self.n_row[k]), int(self.n_col[k])), dtype=DTYPE_f)
            dp_y_d = gpuarray.zeros((int(self.n_row[k]), int(self.n_col[k])), dtype=DTYPE_f)
        else:
            _check_inputs(dp_x_d, dp_y_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, shape=i_peak.shape, dim=2)
        size = DTYPE_i(dp_x_d.size)

        u_d = gpuarray.empty_like(dp_x_d, dtype=DTYPE_f)
        v_d = gpuarray.empty_like(dp_y_d, dtype=DTYPE_f)
        i_peak_d = gpuarray.to_gpu(i_peak)
        j_peak_d = gpuarray.to_gpu(j_peak)

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

        i_peak_d.gpudata.free()
        j_peak_d.gpudata.free()

        return u_d, v_d

    @staticmethod
    def _log_residual(i_peak, j_peak):
        """Normalizes the residual by the maximum quantization error of 0.5 pixel."""
        _check_inputs(i_peak, j_peak, array_type=np.ndarray, dtype=DTYPE_f, shape=i_peak.shape, dim=2)

        try:
            normalized_residual = sqrt(np.sum(i_peak ** 2 + j_peak ** 2) / i_peak.size) / 0.5
            logging.info("[DONE]--Normalized residual : {}.\n".format(normalized_residual))
        except OverflowError:
            logging.warning('[DONE]--Overflow in residuals.\n')
            normalized_residual = np.nan

        return normalized_residual


def get_field_shape(image_size, window_size, overlap_ratio):
    """Returns the shape of the resulting velocity field.

    Given the image size, the interrogation window size and the overlap size, it is possible to calculate the number of
    rows and columns of the resulting flow field.

    Parameters
    ----------
    image_size : tuple
        (ht, wd), pixel size of the image first element is number of rows, second element is the number of columns.
    window_size : int
        Size of the interrogation windows.
    overlap_ratio : float
        Ratio by which two adjacent interrogation windows overlap.

    Returns
    -------
    m, n : int
        Shape of the resulting flow field.

    """
    assert window_size >= 8, "Window size is too small."
    assert window_size % 8 == 0, "Window size must be a multiple of 8."
    assert 0 <= overlap_ratio < 1, 'overlap_ratio must be a float between 0 and 1.'

    spacing = DTYPE_i(window_size * (1 - overlap_ratio))
    m = DTYPE_i((image_size[0] - spacing) // spacing)
    n = DTYPE_i((image_size[1] - spacing) // spacing)
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
    _check_inputs(frame_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, shape=frame_d.shape, dim=2)
    _check_inputs(mask_d, array_type=gpuarray.GPUArray, dtype=DTYPE_i, shape=frame_d.shape, dim=2)

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
    
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        int size = m * n;
        if (i >= size) {return;}

        int row = i / n;
        int col = i % n;

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
        } else if (row == m - 1) {strain[size + n * (m - 1) + col] = (u[n * (m - 1) + col] - u[n * (m - 2) + col]) / h;  // u_y
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


def gpu_round(f_d):
    """Rounds each element in the gpu array.

    Parameters
    ----------
    f_d : GPUArray
        Array to be rounded.

    Returns
    -------
    GPUArray
        Float, same size as f_d. Rounded values of f_d.

    """
    assert type(f_d) == gpuarray.GPUArray, 'Input must a GPUArray.'
    assert f_d.dtype == DTYPE_f, 'Input array must float type.'

    n = DTYPE_i(f_d.size)
    f_round_d = gpuarray.empty_like(f_d)

    mod_round = SourceModule("""
    __global__ void round_gpu(float *dest, float *src, int n)
    {
        // dest : output argument

        int t_id = blockIdx.x * blockDim.x + threadIdx.x;
        if(t_id >= n){return;}

        dest[t_id] = roundf(src[t_id]);
    }
    """)
    block_size = 32
    x_blocks = int(n // block_size + 1)
    round_gpu = mod_round.get_function("round_gpu")
    round_gpu(f_round_d, f_d, n, block=(block_size, 1, 1), grid=(x_blocks, 1))

    return f_round_d


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
    f_smooth_d = gpuarray.to_gpu(smoothn(f, s=s)[0].astype(DTYPE_f, order='C'))  # Smoothn return F-ordered array.

    return f_smooth_d


def gpu_fft_shift(correlation_d):
    """Shifts the fft to the center of the correlation windows.

    Parameters
    ----------
    correlation_d : GPUArray
        3D float, data from the window correlations.

    Returns
    -------
    GPUArray
        3D float, full strain tensor of the velocity fields. (4, m, n) corresponds to (u_x, u_y, v_x and v_y).

    """
    _check_inputs(correlation_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, dim=3)
    n_windows, ht, wd = correlation_d.shape
    window_size_i = DTYPE_i(ht * wd)

    correlation_shift_d = gpuarray.empty_like(correlation_d)

    mod_fft_shift = SourceModule("""
        __global__ void fft_shift(float *destination, float *source, int ws, int ht, int wd)
    {
        // x blocks are windows; y and z blocks are x and y dimensions, respectively.
        int idx_i = blockIdx.x;  // window index
        int idx_x = blockIdx.y * blockDim.x + threadIdx.x;
        int idx_y = blockIdx.z * blockDim.y + threadIdx.y;
        if(idx_x >= wd || idx_y >= ht){return;}

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
    grid_size = ceil(max(ht, ht) / block_size)
    fft_shift = mod_fft_shift.get_function('fft_shift')
    fft_shift(correlation_shift_d, correlation_d, window_size_i, DTYPE_i(ht), DTYPE_i(wd), block=(block_size, block_size, 1), grid=(n_windows, grid_size, grid_size))

    return correlation_shift_d


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
    _check_inputs(x0_d, y0_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, dim=1)
    _check_inputs(x1_d, y1_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, dim=1)
    _check_inputs(f0_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, dim=2)
    _check_inputs(f1_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, dim=2)
    _check_inputs(val_locations_d, array_type=gpuarray.GPUArray, dtype=DTYPE_i, shape=f1_d.shape, dim=2)
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
    _check_inputs(x0_d, y0_d, x1_d, y1_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, dim=1)
    _check_inputs(f0_d, array_type=gpuarray.GPUArray, dtype=DTYPE_f, dim=2)

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

    mod_interpolate = SourceModule("""
        __global__ void bilinear_interpolation(float *f1, float *f0, float *x_grid, float *y_grid, float buffer_x, float buffer_y, float spacing_x, float spacing_y, int ht, int wd, int n, int size)
    {
        int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(t_idx >= size){return;}

        // Map indices to old mesh coordinates.
        int x_idx = (t_idx % n);
        int y_idx = (t_idx / n);
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
        f1[t_idx] = f11 * (x2 - x) * (y2 - y) + f21 * (x - x1) * (y2 - y) + f12 * (x2 - x) * (y - y1) + f22 * (x - x1) * (y - y1);
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
        2D int, indicates which values must be validated. 1 indicates no validation needed, 0 indicates validation is needed.
    u_mean_d, v_mean_d : GPUArray
        3D float, mean velocity surrounding each point.
    n_row, n_col : ndarray
        int, number of rows and columns in each main loop iteration.
    k : int
        Main loop iteration count.

    """
    val_locations_inverse_d = 1 - val_locations_d

    # First iteration, just replace with mean velocity.
    if k == 0:
        u_d = (val_locations_inverse_d * u_mean_d).astype(DTYPE_f) + (val_locations_d * u_d).astype(DTYPE_f)
        v_d = (val_locations_inverse_d * v_mean_d).astype(DTYPE_f) + (val_locations_d * v_d).astype(DTYPE_f)

    # Case if different dimensions: interpolation using previous iteration.
    elif k > 0 and (n_row[k] != n_row[k - 1] or n_col[k] != n_col[k - 1]):
        # TODO can avoid slicing here
        u_d = gpu_interpolate_replace(x0_d[0, :], y0_d[:, 0], x1_d[0, :], y1_d[:, 0], u_previous_d, u_d,
                                      val_locations_d=val_locations_d)
        v_d = gpu_interpolate_replace(x0_d[0, :], y0_d[:, 0], x1_d[0, :], y1_d[:, 0], v_previous_d, v_d,
                                      val_locations_d=val_locations_d)

    # Case if same dimensions.
    elif k > 0 and (n_row[k] == n_row[k - 1] or n_col[k] == n_col[k - 1]):
        u_d = (val_locations_inverse_d * u_previous_d).astype(DTYPE_f) + (val_locations_d * u_d).astype(DTYPE_f)
        v_d = (val_locations_inverse_d * v_previous_d).astype(DTYPE_f) + (val_locations_d * v_d).astype(DTYPE_f)

    return u_d, v_d


def _gpu_array_index(array_d, indices, dtype):
    """Allows for arbitrary index selecting with numpy arrays

    Parameters
    ----------
    array_d : GPUArray
        Float or int, array to be selected from.
    indices : GPUArray
        1D int, list of indexes that you want to index. If you are indexing more than 1 dimension, then make sure that this array is flattened.
    dtype : dtype
        Either int32 or float 32. determines the datatype of the returned array.

    Returns
    -------
    GPUArray
        Float or int, values at the specified indexes.

    """
    # GPU will automatically flatten the input array. The indexing must reference the flattened GPU array.
    assert indices.ndim == 1, "Number of dimensions of indices is wrong. Should be equal to 1"
    assert type(array_d) == gpuarray.GPUArray, 'Input must be GPUArray.'
    assert array_d.dtype == DTYPE_f or array_d.dtype == DTYPE_f, 'Input must have dtype float32 or int32.'

    # send data to the gpu
    return_values_d = gpuarray.zeros(indices.size, dtype=dtype)

    mod_array_index = SourceModule("""
    __global__ void array_index_float(float *return_values, float *array, int *return_list, int size)
    {
        // 1D grid of 1D blocks
        int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(t_idx >= size){return;}

        return_values[t_idx] = array[return_list[t_idx]];
    }

    __global__ void array_index_int(float *array, int *return_values, int *return_list, int size)
    {
        // 1D grid of 1D blocks
        int t_idx = blockIdx.x*blockDim.x + threadIdx.x;
        if(t_idx >= size){return;}

        return_values[t_idx] = (int)array[return_list[t_idx]];
    }
    """)
    block_size = 32
    r_size = DTYPE_i(indices.size)
    x_blocks = int(r_size // block_size + 1)

    if dtype == DTYPE_f:
        array_index = mod_array_index.get_function('array_index_float')
        array_index(return_values_d, array_d, indices, r_size, block=(block_size, 1, 1), grid=(x_blocks, 1))
    elif dtype == DTYPE_i:
        array_index = mod_array_index.get_function('array_index_int')
        array_index(return_values_d, array_d, indices, r_size, block=(block_size, 1, 1), grid=(x_blocks, 1))

    return return_values_d


def _gpu_index_update(dest_d, values_d, indices_d):
    """Allows for arbitrary index selecting with numpy arrays.

    Parameters
    ----------
    dest_d : GPUArray
       nD float, array to be updated with new values.
    values_d : GPUArray
        1D float, values to be updated in the destination array.
    indices_d : GPUArray
        1D int, indices to update.

    Returns
    -------
    GPUArray
        Float, input array with values updated

    """
    size_i = DTYPE_i(values_d.size)

    mod_index_update = SourceModule("""
    __global__ void index_update(float *dest, float *values, int *indices, int size)
    {
        // 1D grid of 1D blocks
        int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(t_idx >= size){return;}

        dest[indices[t_idx]] = values[t_idx];
    }
    """)
    block_size = 32
    x_blocks = int(size_i // block_size + 1)
    index_update = mod_index_update.get_function('index_update')
    index_update(dest_d, values_d, indices_d, size_i, block=(block_size, 1, 1), grid=(x_blocks, 1))


def _check_inputs(*arrays, array_type=None, dtype=None, shape=None, dim=None):
    """Checks that all array inputs match either each other's or the given array type, dtype, shape and dim."""
    if array_type is not None:
        assert all([type(array) == array_type for array in arrays]), 'Inputs must be ({}).'.format(array_type)
    if dtype is not None:
        assert all([array.dtype == dtype for array in arrays]), 'Inputs must have dtype ({}).'.format(dtype)
    if shape is not None:
        assert all(
            [array.shape == shape for array in arrays]), 'Inputs must have shape ({}, all must be same shape).'.format(
            shape)
    if dim is not None:
        assert all([len(array.shape) == dim for array in arrays]), 'Inputs must have same dim ({}).'.format(dim)
