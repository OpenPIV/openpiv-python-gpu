# cython: profile=True

"""This module is dedicated to advanced algorithms for PIV image analysis with NVIDIA GPU Support.

Note that all data must 32-bit at most to be stored on GPUs.

"""

import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import pycuda.autoinit
import pycuda.cumath as cumath
from pycuda.compiler import SourceModule
import skcuda.fft as cu_fft
import skcuda.misc as cu_misc
import numpy as np
import numpy.ma as ma
from numpy.fft import fftshift
from math import sqrt
from openpiv.gpu_validation import gpu_validation
# import cupy
from openpiv.smoothn import smoothn as smoothn
from time import process_time_ns

cimport numpy as np

DTYPEi = np.int32
ctypedef np.int32_t DTYPEi_t

DTYPEb = np.uint8
ctypedef np.uint8_t DTYPEb_t

# GPU can only hold 32 bit numbers
DTYPEf = np.float32
ctypedef np.float32_t DTYPEf_t


class CorrelationFunction:
    def __init__(self, d_frame_a, d_frame_b, window_size, overlap, nfft_x=None, extended_size=None, d_shift=None, d_strain=None):
        """A class representing a cross correlation function.

        NOTE: All identifiers starting with 'd_' exist on the GPU and not the CPU. The GPU is referred to as the device,
        and therefore "d_" signifies that it is a device variable. Please adhere to this standard as it makes developing
        and debugging much easier.

        Parameters
        ----------
        d_frame_a, d_frame_b : GPUArray - 2D float32
            image pair
        window_size : int
            size of the interrogation window
        overlap : int
            pixel overlap between interrogation windows
        nfft_x : int
            window size for fft
        extended_size : int
            extended window size to search in the second frame
        d_shift : GPUArray
            2D ([dx, dy]), dx and dy are 1D arrays of the x-y shift at each interrogation window of the second image.
            This is using the x-y convention of this code where x is the row and y is the column.
        d_strain : GPUArray
            2D strain tensor. First dimension is (u_x, u_y, v_x, v_y)

        """
        ########################################################################################
        # PARAMETERS FOR CORRELATION FUNCTION
        self.shape = d_frame_a.shape
        self.window_size = np.int32(window_size)
        self.overlap = np.int32(overlap)
        self.n_rows, self.n_cols = np.int32(get_field_shape(self.shape, window_size, overlap))
        self.batch_size = np.int32(self.n_rows * self.n_cols)

        if nfft_x is None:
            self.nfft = np.int32(2 * self.window_size)
        else:
            self.nfft = np.int32(nfft_x)
            assert (self.nfft & (self.nfft - 1)) == 0, 'nfft must be power of 2'

        ########################################################################################
        # START DOING CALCULATIONS

        # Return stack of all IWs
        d_win_a = gpuarray.zeros((self.batch_size, self.window_size, self.window_size), dtype=np.float32)
        d_win_b = gpuarray.zeros((self.batch_size, self.window_size, self.window_size), dtype=np.float32)
        self._iw_arrange(d_frame_a, d_frame_b, d_win_a, d_win_b, d_shift, d_strain)

        # normalize array by computing the norm of each IW
        d_win_a_norm = gpuarray.zeros((self.batch_size, self.window_size, self.window_size), dtype=np.float32)
        d_win_b_norm = gpuarray.zeros((self.batch_size, self.window_size, self.window_size), dtype=np.float32)
        self._normalize_intensity(d_win_a, d_win_b, d_win_a_norm, d_win_b_norm)

        # zero pad arrays
        d_win_a_zp = gpuarray.zeros([self.batch_size, self.nfft, self.nfft], dtype=np.float32)
        d_win_b_zp = gpuarray.zeros_like(d_win_a_zp)
        self._zero_pad(d_win_a_norm, d_win_b_norm, d_win_a_zp, d_win_b_zp)

        # correlate Windows
        self.data = self._correlate_windows(d_win_a_zp, d_win_b_zp)

        # get first peak of correlation function
        self.p_row, self.p_col, self.corr_max1 = self._find_peak(self.data)


    def _iw_arrange(self, d_frame_a, d_frame_b, d_win_a, d_win_b, d_shift, d_strain):
        """Creates a 3D array stack of all of the interrogation windows.

        This is necessary to do the FFTs all at once on the GPU. Parts of this code assumes that the interrogation
        windows are square. This populates interrogation windows from the origin of the image.

        Parameters
        -----------
        d_frame_a, d_frame_b : GPUArray - 2D float32
            PIV image pair
        d_win_a : GPUArray - 3D
            All frame_a interrogation windows stacked on each other
        d_win_b : GPUArray - 3D
            All frame_b interrogation windows stacked on each other
        d_shift : GPUArray
            shift of the second window
        d_strain : GPUArray
            2D strain rate tensor. First dimension is (u_x, u_y, v_x, v_y)

        """
        # define window slice algorithm
        mod_ws = SourceModule("""
            __global__ void window_slice(float *input, float *output, int window_size, int spacing, int n_col, int w)
        {
            int f_range;
            int w_range;
            int IW_size = window_size * window_size;

            // indices of image to map from
            int ind_i = blockIdx.x;
            int ind_x = blockIdx.y * blockDim.x + threadIdx.x;
            int ind_y = blockIdx.z * blockDim.y + threadIdx.y;

            // loop through each interrogation window
            f_range = (ind_i / n_col * spacing + ind_y) * w + (ind_i % n_col) * spacing + ind_x;

            // indices of new array to map to
            w_range = ind_i * IW_size + window_size * ind_y + ind_x;

            output[w_range] = input[f_range];
        }
        
            __global__ void window_slice_deform(float *input, float *output, float *dx, float *dy, float *strain, int window_size, int spacing, int n_col, int num_window, int w, int h)
        {
            // w = width (number of columns in the full image)
            // h = height (number of rows in the full image)

            int w_range;
            float x_shift;
            float y_shift;
            // int num_window = n_row * n_col;

            // x blocks are windows; y and z blocks are x and y dimensions, respectively
            int ind_i = blockIdx.x;  // window index
            int ind_x = blockIdx.y * blockDim.x + threadIdx.x;
            int ind_y = blockIdx.z * blockDim.y + threadIdx.y;

            // Loop through each interrogation window and apply the shift and deformation.
            // get the strain tensor values
            float u_x = strain[ind_i];
            float u_y = strain[num_window + ind_i];
            float v_x = strain[2 * num_window + ind_i];
            float v_y = strain[3 * num_window + ind_i];

            // compute the window vector
            float r_x = ind_x - window_size / 2 + 1;  // r_x = x - x_c // (+ 1) reduces the bias error
            float r_y = ind_y - window_size / 2 + 1;  // r_y = y - y_c

            // apply deformation operation
            x_shift = ind_x + dx[ind_i] + r_x * u_x + r_y * u_y;  // r * du + dx
            y_shift = ind_y + dy[ind_i] + r_x * v_x + r_y * v_y;  // r * dv + dy

            // do the mapping
            float x = ind_i % n_col * spacing + x_shift;
            float y = ind_i / n_col * spacing + y_shift;

            // do bilinear interpolation
            int x2 = ceilf(x);
            int x1 = floorf(x);
            int y2 = ceilf(y);
            int y1 = floorf(y);

            // prevent divide-by-zero
            if (x2 == x1) {x2 = x1 + 1;}
            if (y2 == y1) {y2 = y2 + 1;}

            // find limits of domain
            int outside_range = (x1 >= 0 && x2 < w && y1 >= 0 && y2 < h);

            // terms of the bilinear interpolation
            float f11 = input[(y1 * w + x1) * outside_range];
            float f21 = input[(y1 * w + x2) * outside_range];
            float f12 = input[(y2 * w + x1) * outside_range];
            float f22 = input[(y2 * w + x2) * outside_range];

            // indices of image to map to
            w_range = ind_i * window_size * window_size + window_size * ind_y + ind_x;

            // Apply the mapping. Multiply by outside_range to set values outside the window to zero.
            output[w_range] = (f11 * (x2 - x) * (y2 - y) + f21 * (x - x1) * (y2 - y) + f12 * (x2 - x) * (y - y1) + f22 * (x - x1) * (y - y1)) * outside_range;

        }
        """)

        # get field shapes
        h = np.int32(self.shape[0])
        w = np.int32(self.shape[1])
        spacing = self.window_size - self.overlap

        # for debugging
        assert self.window_size >= 8, "Window size is too small."
        assert self.window_size % 8 == 0, "Window size should be a multiple of 8."

        # gpu parameters
        block_size = 8
        grid_size = int(self.window_size / block_size)

        # slice windows
        if d_shift is not None:
            d_dy = d_shift[1].copy()
            d_dx = d_shift[0].copy()
            if d_strain is not None:
                d_strain_a = -d_strain.copy() / 2
                d_strain_b = d_strain.copy() / 2
            else:
                d_strain_a = gpuarray.zeros((4, h, w), dtype=np.float32)
                d_strain_b = gpuarray.zeros((4, h, w), dtype=np.float32)

            # shift frames and deform
            d_dy_a = -d_dy / 2
            d_dx_a = -d_dx / 2
            d_dy_b = d_dy / 2
            d_dx_b = d_dx / 2
            window_slice_deform = mod_ws.get_function("window_slice_deform")
            window_slice_deform(d_frame_a, d_win_a, d_dx_a, d_dy_a, d_strain_a, self.window_size, spacing, self.n_cols, self.batch_size, w, h, block=(block_size, block_size, 1), grid=(int(self.batch_size), grid_size, grid_size))
            window_slice_deform(d_frame_b, d_win_b, d_dx_b, d_dy_b, d_strain_b, self.window_size, spacing, self.n_cols, self.batch_size, w, h, block=(block_size, block_size, 1), grid=(int(self.batch_size), grid_size, grid_size))

            # free displacement GPU memory
            d_dy_a.gpudata.free()
            d_dx_a.gpudata.free()
            d_dy_b.gpudata.free()
            d_dx_b.gpudata.free()
            d_dx.gpudata.free()
            d_dy.gpudata.free()
            d_strain_a.gpudata.free()
            d_strain_b.gpudata.free()

        else:
            # use non-translating windows
            window_slice_deform = mod_ws.get_function("window_slice")
            window_slice_deform(d_frame_a, d_win_a, self.window_size, spacing, self.n_cols, w, block=(block_size, block_size, 1), grid=(int(self.batch_size), grid_size, grid_size))
            window_slice_deform(d_frame_b, d_win_b, self.window_size, spacing, self.n_cols, w, block=(block_size, block_size, 1), grid=(int(self.batch_size), grid_size, grid_size))


    def _normalize_intensity(self, d_win_a, d_win_b, d_win_a_norm, d_win_b_norm):
        """Remove the mean from each IW of a 3D stack of IWs

        Parameters
        ----------
        d_win_a : GPUArray - 3D float32
            stack of first frame IWs
        d_win_b : GPUArray - 3D float32
            stack of second frame IWs
        d_win_a_norm : GPUArray - 3D float32
            the normalized intensity in the first window
        d_win_b_norm : GPUArray - 3D float32
            the normalized intensity in the second window

        Returns
        -------
        norm : GPUArray - 3D
            stack of IWs with mean removed

        """
        mod_norm = SourceModule("""
            __global__ void normalize(float *array, float *array_norm, float *mean, int iw_size)
        {
            // global thread id for 1D grid of 2D blocks
            int threadId = blockIdx.x * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;

            // indices for mean matrix
            int meanId = threadId / iw_size;

            array_norm[threadId] = array[threadId] - mean[meanId];
        }
        """)

        # must do this for skcuda misc library
        cu_misc.init()

        # number of pixels in each interrogation window
        iw_size = np.int32(d_win_a.shape[1] * d_win_a.shape[2])

        # get mean of each IW using skcuda
        d_mean_a = cu_misc.mean(d_win_a.reshape(self.batch_size, iw_size), axis=1)
        d_mean_b = cu_misc.mean(d_win_b.reshape(self.batch_size, iw_size), axis=1)

        # gpu kernel blocksize parameters
        block_size = 8
        grid_size = int(d_win_a.size / block_size ** 2)

        assert d_win_a.size % (block_size ** 2) == 0, 'Not all windows are being normalized. Something wrong with block or grid size.'

        # get function and norm IWs
        normalize = mod_norm.get_function('normalize')
        normalize(d_win_a, d_win_a_norm, d_mean_a, iw_size, block=(block_size, block_size, 1), grid=(grid_size, 1))
        normalize(d_win_b, d_win_b_norm, d_mean_b, iw_size, block=(block_size, block_size, 1), grid=(grid_size, 1))

        # free GPU memory
        d_mean_a.gpudata.free()
        d_mean_b.gpudata.free()
        d_win_a.gpudata.free()
        d_win_b.gpudata.free()


    def _zero_pad(self, d_win_a_norm, d_win_b_norm, d_win_a_zp, d_win_b_zp):
        """Function that zero-pads an 3D stack of arrays for use with the skcuda FFT function.

        Parameters
        ----------
        d_win_a_norm : GPUArray - 3D float32
            array to be zero padded
        d_win_b_norm : GPUArray - 3D float32
            arrays to be zero padded
        d_win_a_zp : GPUArray - 3D float32
            array to be zero padded
        d_win_b_zp : GPUArray - 3D float32
            arrays to be zero padded

        Returns
        -------
        d_winA_zp : GPUArray - 3D
            initial array that has been zero padded
        d_search_area_zp : GPUArray - 3D
            initial array that has been zero padded

        """
        mod_zp = SourceModule("""
            __global__ void zero_pad(float *array_zp, float *array, int fft_size, int window_size)
            {
                // number of pixels in each IW
                int IW_size = fft_size * fft_size;
                int arr_size = window_size * window_size;

                int zp_range;
                int arr_range;
            
                // indices for each IW
                // x blocks are windows; y and z blocks are x and y dimensions, respectively
                int ind_i = blockIdx.x;
                int ind_x = blockIdx.y * blockDim.x + threadIdx.x;
                int ind_y = blockIdx.z * blockDim.y + threadIdx.y;

                // get range of values to map
                arr_range = ind_i * arr_size + window_size * ind_y + ind_x;
                zp_range = ind_i * IW_size + fft_size * ind_y + ind_x;

                // apply the map
                array_zp[zp_range] = array[arr_range];
            }
        """)

        # gpu parameters
        block_size = 8
        grid_size = int(self.window_size / block_size)

        # get handle and call function
        zero_pad = mod_zp.get_function('zero_pad')
        zero_pad(d_win_a_zp, d_win_a_norm, self.nfft, self.window_size, block=(block_size, block_size, 1), grid=(int(self.batch_size), grid_size, grid_size))
        zero_pad(d_win_b_zp, d_win_b_norm, self.nfft, self.window_size, block=(block_size, block_size, 1), grid=(int(self.batch_size), grid_size, grid_size))

        # Free GPU memory
        d_win_a_norm.gpudata.free()
        d_win_b_norm.gpudata.free()


    def _correlate_windows(self, d_win_a_zp, d_win_b_zp):
        """Compute correlation function between two interrogation windows.

        The correlation function can be computed by using the correlation theorem to speed up the computation.

        Parameters
        ----------
        d_win_a_zp : GPUArray
            first window
        d_win_b_zp : GPUArray
            second window

        Returns
        -------
        corr : array - 2D
            a two dimensional array for the correlation function.

        """
        # FFT size
        win_h = np.int32(self.nfft)
        win_w = np.int32(self.nfft)

        # allocate space on gpu for FFTs
        d_win_i_fft = gpuarray.empty((self.batch_size, win_h, win_w), np.float32)
        d_win_fft = gpuarray.empty((self.batch_size, win_h, win_w // 2 + 1), np.complex64)
        d_search_area_fft = gpuarray.empty((self.batch_size, win_h, win_w // 2 + 1), np.complex64)

        # forward FFTs
        plan_forward = cu_fft.Plan((win_h, win_w), np.float32, np.complex64, self.batch_size)
        cu_fft.fft(d_win_a_zp, d_win_fft, plan_forward)
        cu_fft.fft(d_win_b_zp, d_search_area_fft, plan_forward)

        # multiply the FFTs
        d_win_fft = d_win_fft.conj()
        d_tmp = cu_misc.multiply(d_search_area_fft, d_win_fft)

        # inverse transform
        plan_inverse = cu_fft.Plan((win_h, win_w), np.complex64, np.float32, self.batch_size)
        cu_fft.ifft(d_tmp, d_win_i_fft, plan_inverse, True)

        # transfer back to cpu to do FFTshift
        corr = fftshift(d_win_i_fft.get().real, axes=(1, 2))

        # free gpu memory
        d_win_i_fft.gpudata.free()
        d_win_fft.gpudata.free()
        d_search_area_fft.gpudata.free()
        d_tmp.gpudata.free()
        d_win_a_zp.gpudata.free()
        d_win_b_zp.gpudata.free()

        return corr


    def _find_peak(self, array):
        """Find row and column of highest peak in correlation function

        Parameters
        ----------
        array : ndarray
            array that is image of the correlation function

        Returns
        -------
        ind : array - 1D int
            flattened index of corr peak
        row : array - 1D int
            row position of corr peak
        col : array - 1D int
            column position of corr peak

        """
        cdef Py_ssize_t i
        cdef Py_ssize_t size = self.batch_size

        # Reshape matrix
        array_reshape = array.reshape(self.batch_size, self.nfft ** 2)
        s = self.nfft

        # Get index and value of peak
        max_ind = np.argmax(array_reshape, axis=1)
        maximum = array_reshape[range(self.batch_size), max_ind]
        # maximum = np.amax(array_reshape, axis=1)

        # row and column information of peak
        row = max_ind // s
        col = max_ind % s

        # # return the center if the correlation peak is zero
        # cdef float[:, :] array_view = array_reshape
        # cdef float[:] maximum_view = maximum
        # cdef long[:] row_view = row
        # cdef long[:] col_view = col
        # cdef long w = int(s / 2)
        # cdef int zero_idx = int(self.nfft ** 2 / 2 + s / 2)
        # cdef Py_ssize_t i
        # cdef Py_ssize_t size = self.batch_size
        #
        # for i in range(size):
        #     if array_view[i, zero_idx] == maximum_view[i]:
        #         row_view[i] = w
        #         col_view[i] = w

        # # return the center if the correlation peak is zero (same as cython code above)
        # w = s / 2
        # c_idx = np.asarray((array_reshape[:, int(self.nfft ** 2 / 2 + w)] == maximum)).nonzero()
        # row[c_idx] = w
        # col[c_idx] = w

        # return the center if the correlation peak is zero
        cdef float[:, :] array_view = array_reshape
        cdef float[:] maximum_view = maximum
        cdef long[:] row_view = row
        cdef long[:] col_view = col
        cdef long w = int(s / 2)

        for i in range(size):
            if abs(maximum_view[i]) < 0.1:
                row_view[i] = w
                col_view[i] = w

        return row, col, maximum


    def _find_second_peak(self, int width):
        """Find the value of the second largest peak.

        The second largest peak is the height of the peak in the region outside a ``width * width`` submatrix around
        the first correlation peak.

        Parameters
        ----------
        width : int
            the half size of the region around the first correlation peak to ignore for finding the second peak.

        Returns
        -------
        i, j : two elements tuple
            the row, column index of the second correlation peak.
        corr_max2 : int
            the value of the second correlation peak.

        """
        # create a masked view of the self.data array
        tmp = self.data.view(ma.MaskedArray)

        cdef Py_ssize_t i
        cdef Py_ssize_t j

        # set (width x width) square submatrix around the first correlation peak as masked
        for i in range(-width, width + 1):
            for j in range(-width, width + 1):
                row_idx = self.p_row + i
                col_idx = self.p_col + j
                idx = (row_idx >= 0) & (row_idx < self.nfft) & (col_idx >= 0) & (col_idx < self.nfft)
                tmp[idx, row_idx[idx], col_idx[idx]] = ma.masked

        row2, col2, corr_max2 = self._find_peak(tmp)

        return corr_max2


    def subpixel_peak_location(self):
        """Find subpixel peak approximation using Gaussian method.

        Returns
        -------
        row_sp : array - 1D float
            row max location to subpixel accuracy
        col_sp : array - 1D float
            column max location to subpixel accuracy

        """
        # Define small number to replace zeros and get rid of warnings in calculations
        cdef DTYPEf_t small = 1e-20

        # cast corr and row as a ctype array
        cdef np.ndarray[DTYPEf_t, ndim=3] corr_c = np.array(self.data, dtype=DTYPEf)
        cdef np.ndarray[DTYPEf_t, ndim=1] row_c = np.array(self.p_row, dtype=DTYPEf)
        cdef np.ndarray[DTYPEf_t, ndim=1] col_c = np.array(self.p_col, dtype=DTYPEf)

        # Define arrays to store the data
        cdef np.ndarray[DTYPEf_t, ndim=1] row_sp = np.empty(self.batch_size, dtype=DTYPEf)
        cdef np.ndarray[DTYPEf_t, ndim=1] col_sp = np.empty(self.batch_size, dtype=DTYPEf)

        # Move boundary peaks inward one node. Replace later in sig2noise
        row_tmp = np.copy(self.p_row)
        row_tmp[row_tmp < 1] = 1
        row_tmp[row_tmp > self.nfft - 2] = self.nfft - 2
        col_tmp = np.copy(self.p_col)
        col_tmp[col_tmp < 1] = 1
        col_tmp[col_tmp > self.nfft - 2] = self.nfft - 2

        # Initialize arrays
        cdef np.ndarray[DTYPEf_t, ndim=1] c = corr_c[range(self.batch_size), row_tmp, col_tmp]
        cdef np.ndarray[DTYPEf_t, ndim=1] cl = corr_c[range(self.batch_size), row_tmp - 1, col_tmp]
        cdef np.ndarray[DTYPEf_t, ndim=1] cr = corr_c[range(self.batch_size), row_tmp + 1, col_tmp]
        cdef np.ndarray[DTYPEf_t, ndim=1] cd = corr_c[range(self.batch_size), row_tmp, col_tmp - 1]
        cdef np.ndarray[DTYPEf_t, ndim=1] cu = corr_c[range(self.batch_size), row_tmp, col_tmp + 1]

        # Get rid of values that are zero or lower
        cdef np.ndarray[DTYPEf_t, ndim=1] non_zero = np.array(c > 0, dtype=DTYPEf)
        c[c <= 0] = small
        cl[cl <= 0] = small
        cr[cr <= 0] = small
        cd[cd <= 0] = small
        cu[cu <= 0] = small

        # Do subpixel approximation. Add small to avoid zero divide.
        row_sp = row_c + ((np.log(cl) - np.log(cr)) / (2 * np.log(cl) - 4 * np.log(c) + 2 * np.log(cr) + small)) * non_zero - self.nfft / 2
        col_sp = col_c + ((np.log(cd) - np.log(cu)) / (2 * np.log(cd) - 4 * np.log(c) + 2 * np.log(cu) + small)) * non_zero - self.nfft / 2
        return row_sp, col_sp


    def sig2noise_ratio(self, method='peak2peak', width=2):
        """Computes the signal to noise ratio.

        The signal to noise ratio is computed from the correlation map with one of two available method. It is a measure
        of the quality of the matching between two interrogation windows.

        Parameters
        ----------
        method : string
            the method for evaluating the signal to noise ratio value from
            the correlation map. Can be `peak2peak`, `peak2mean` or None
            if no evaluation should be made.
        width : int, optional
            the half size of the region around the first
            correlation peak to ignore for finding the second
            peak. [default: 2]. Only used if ``sig2noise_method==peak2peak``.

        Returns
        -------
        sig2noise : array - float
            the signal to noise ratio from the correlation map for each vector.

        """
        # compute signal to noise ratio
        if method == 'peak2peak':
            # find second peak height
            corr_max2 = self._find_second_peak(width=width)

        elif method == 'peak2mean':
            # find mean of the correlation map
            corr_max2 = self.data.mean()

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
        sig2noise[np.array(self.p_row == 0) * np.array(self.p_row == self.data.shape[1]) * np.array(self.p_col == 0) * np.array(self.p_col == self.data.shape[2])] = 0.0

        return sig2noise.reshape(self.n_rows, self.n_cols)


    def return_shape(self):
        """
        Return row/column information
        """
        return self.n_rows, self.n_cols


def get_field_shape(image_size, window_size, overlap):
    """Compute the shape of the resulting flow field.

    Given the image size, the interrogation window size and the overlap size, it is possible to calculate the number of
    rows and columns of the resulting flow field.

    Parameters
    ----------
    image_size : two elements tuple
        a two dimensional tuple for the pixel size of the image first element is number of rows, second element is the
        number of columns.
    window_size : int
        the size of the interrogation window.
    overlap : int
        the number of pixel by which two adjacent interrogation windows overlap.

    Returns
    -------
    field_shape : two elements tuple
        the shape of the resulting flow field

    """
    spacing = window_size - overlap
    return DTYPEi((image_size[0] - spacing) // spacing), DTYPEi((image_size[1] - spacing) // spacing)


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
    2D float
        Two-dimensional array of integers containing grey levels of the first and second frames.
    window_size : int
        The size of the (square) interrogation window for the first frame.
    search_area_size : int
        The size of the (square) interrogation window for the second frame.
    overlap_ratio : int
        The ratio of overlap between two windows (between 0 and 1)
    dt : float
        Time delay separating the two frames.

    Returns
    -------
    u : ndarray
        2D, the u velocity component, in pixels/seconds.
    v : ndarray
        2D, the v velocity component, in pixels/seconds.
    mask : ndarray
        2D, the boolean values (True for vectors interpolated from previous iteration).
    s2n : ndarray
        2D, the signal to noise ratio of the final velocity field.


    Other Parameters
    ----------------
    subpixel_method : {'gaussian', 'centroid', 'parabolic'}
        Method to estimate subpixel location of the peak. Gaussian is default if correlation map is positive. Centroid replaces default if correlation map is negative.
    sig2noise : bool
        whether the signal-to-noise ratio should be computed and returned. Setting this to False speeds up the computation significatly.
    sig2noise_method : {'peak2peak', 'peak2mean', None}
        Method of signal-to-noise-ratio measur. If None, no measure is performed.
    width : int
        Half size of the region around the first correlation peak to ignore for finding the second peak. Default is 2. Only used if sig2noise_method==peak2peak.
    nfftx : int
        The size of the 2D FFT in x-direction. The default of 2 x windows_a.shape[0] is recommended.

    Examples
    --------

    >>> u, v = gpu_extended_search_area(frame_a, frame_b, window_size=16, overlap=8, search_area_size=48, dt=0.1)

    """
    # Extract the parameters
    return_sig2noise = kwargs['sig2noise'] if 'sig2noise' in kwargs else True
    sig2noise_method = kwargs['sig2noise_method'] if 'sig2noise_method' in kwargs else 'peak2peak'
    nfftx = kwargs['nfftx'] if 'nfftx' in kwargs else None
    overlap = overlap_ratio * window_size

    # cast images as floats and sent to gpu
    d_frame_a_f = gpuarray.to_gpu(frame_a.astype(np.float32))
    d_frame_b_f = gpuarray.to_gpu(frame_b.astype(np.float32))

    # Get correlation function
    c = CorrelationFunction(d_frame_a_f, d_frame_b_f, search_area_size, overlap, nfftx)

    # Get window displacement to subpixel accuracy
    sp_i, sp_j = c.subpixel_peak_location()

    # vector field shape
    n_row, n_col = c.return_shape()

    # reshape the peaks
    i_peak = np.reshape(sp_i, (n_row, n_col))
    j_peak = np.reshape(sp_j, (n_row, n_col))

    # calculate velocity fields
    u =  (j_peak - (search_area_size - window_size) / 2) / dt
    v = -(i_peak - (search_area_size - window_size) / 2) / dt

    # Free gpu memory
    d_frame_a_f.gpudata.free()
    d_frame_b_f.gpudata.free()

    # get the signal-to-noise ratio
    if return_sig2noise:
        sig2noise = c.sig2noise_ratio(method=sig2noise_method)
        return u, v, sig2noise
    else:
        return u, v


def gpu_piv(frame_a, frame_b,
                window_size_iters=(1, 2),
                min_window_size=16,
                overlap_ratio=0.5,
                dt=1,
                mask=None,
                deform=True,
                smooth=True,
                nb_validation_iter=1,
                validation_method='median_velocity',
                trust_1st_iter=True,
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
        2D float, integers containing grey levels of the first and second frames.
    window_size_iters : tuple or int
        Number of iterations performed at each window size
    min_window_size : tuple or int
        Length of the sides of the square deformation. Only supports multiples of 8.
    overlap_ratio : float
        The ratio of overlap between two windows (between 0 and 1).
    dt : float
        Time delay separating the two frames.
    mask : ndarray
        2D, dtype=np.int32. Array of integers with values 0 for the background, 1 for the flow-field. If the center of a window is on a 0 value the velocity is set to 0.
    deform : bool
        Whether to deform the windows by the velocity gradient at each iteration.
    smooth : bool
        Whether to smooth the intermediate fields.
    nb_validation_iter : int
        Number of iterations per validation cycle.
    validation_method : {tuple, 's2n', 'median_velocity', 'mean_velocity', 'rms_velocity'}
        Method used for validation. Only the mean velocity method is implemented now. The default tolerance is 2 for median validation.
    trust_1st_iter : bool
        With a first window size following the 1/4 rule, the 1st iteration can be trusted and the value should be 1.

    Returns
    -------
    x : ndarray
        2D, the x-axis component of the interpolation locations, in pixels-lengths starting from 0 at left edge.
    y : ndarray
        2D, the y-axis component of the interpolation locations, in pixels starting from 0 at bottom edge.
    u : ndarray
        2D, the u velocity component, in pixels/seconds.
    v : ndarray
        2D, the v velocity component, in pixels/seconds.
    mask : ndarray
        2D, the boolean values (True for vectors interpolated from previous iteration).
    s2n : ndarray
        2D, the signal to noise ratio of the final velocity field.

    Other Parameters
    ----------------
    s2n_tol, median_tol, mean_tol, median_tol, rms_tol : float
        Tolerance of the validation methods.
    smoothing_par : float
        Smoothing parameter to pass to Smoothn to apply to the intermediate velocity fields.
    subpixel_method : {'gaussian', 'centroid', 'parabolic'}
        Method to estimate subpixel location of the peak. Gaussian is default if correlation map is positive. Centroid replaces default if correlation map is negative.
    sig2noise : bool
        whether the signal-to-noise ratio should be computed and returned. Setting this to False speeds up the computation significatly.
    sig2noise_method : {'peak2peak', 'peak2mean', None}
        Method of signal-to-noise-ratio measurement. If None, no measure is performed.
    s2n_width : int
        Half size of the region around the first correlation peak to ignore for finding the second peak. Default is 2. Only used if sig2noise_method==peak2peak.
    nfftx : int
        The size of the 2D FFT in x-direction. The default of 2 x windows_a.shape[0] is recommended.

    Example
    -------
    >>> x, y, u, v, mask = gpu_piv(frame_a, frame_b, mask, min_window_size=16,window_size_iters=(2, 2), overlap_ratio=0.5, coarse_factor=2, dt=1, deform=True, smoothing=True, validation_method='median_velocity', validation_iter=2, trust_1st_iter=True, median_tol=2)

    """
    """
    Method of implementation: to improve the speed of the program, all data have been placed in the same huge
    4-dimensions 'f' array. (this prevent the definition of a new array for each iteration) However, during the coarsening process a large part of the array is not used.
    Structure of array f:
    --The 1st index is the main iteration (K)   --> length is nb_iter_max
        -- 2nd index (I) is row (of the map of the interpolations locations of iteration K) --> length (effectively used) is n_row[K]
            --3rd index (J) is column  --> length (effectively used) is n_col[K]
                --4th index represent the type of data stored at this point:
                    | 0 --> x         |
                    | 1 --> y         |
                    | 2 --> dx        |
                    | 3 --> dy        |
                    | 4 --> dpx       |
                    | 5 --> dpy       |
                    | 6 --> mask      |
    Storage of data with indices is not good for comprehension so its very important to comment on each single operation.
    A python dictionary type could have been used (and would be much more intuitive)
    but its equivalent in c language (type map) is very slow compared to a numpy ndarray.

    """
    piv_gpu = PIVGPU(frame_a.shape, window_size_iters, min_window_size, overlap_ratio, dt, mask, deform, smooth, nb_validation_iter, validation_method, trust_1st_iter, **kwargs)
    return piv_gpu(frame_a, frame_b)


class PIVGPU:
    """This class is the object-oriented implementation of the GPU PIV function.

    Parameters
    ----------
    frame_shape : tuple
        (ht, wd) of the image series
    window_size_iters : tuple or int
        Number of iterations performed at each window size
    min_window_size : tuple or int
        Length of the sides of the square deformation. Only support multiples of 8.
    overlap_ratio : float
        the ratio of overlap between two windows (between 0 and 1).
    dt : float
        Time delay separating the two frames.
    mask : ndarray
        2D, float. Array of integers with values 0 for the background, 1 for the flow-field. If the center of a window is on a 0 value the velocity is set to 0.
    deform : bool
        Whether to deform the windows by the velocity gradient at each iteration.
    smooth : bool
        Whether to smooth the intermediate fields.
    nb_validation_iter : int
        Number of iterations per validation cycle.
    validation_method : {tuple, 's2n', 'median_velocity', 'mean_velocity', 'rms_velocity'}
        Method used for validation. Only the mean velocity method is implemented now. The default tolerance is 2 for median validation.
    trust_1st_iter : bool
        With a first window size following the 1/4 rule, the 1st iteration can be trusted and the value should be 1.

    Other Parameters
    ----------------
    s2n_tol, median_tol, mean_tol, median_tol, rms_tol : float
        Tolerance of the validation methods.
    smoothing_par : float
        Smoothing parameter to pass to Smoothn to apply to the intermediate velocity fields.
    subpixel_method : {'gaussian', 'centroid', 'parabolic'}
        Method to estimate subpixel location of the peak. Gaussian is default if correlation map is positive. Centroid replaces default if correlation map is negative.
    sig2noise : bool
        whether the signal-to-noise ratio should be computed and returned. Setting this to False speeds up the computation significatly.
    sig2noise_method : {'peak2peak', 'peak2mean', None}
        Method of signal-to-noise-ratio measurement. If None, no measure is performed.
    s2n_width : int
        Half size of the region around the first correlation peak to ignore for finding the second peak. Default is 2. Only used if sig2noise_method==peak2peak.
    nfftx : int
        The size of the 2D FFT in x-direction. The default of 2 x windows_a.shape[0] is recommended.

    Methods
    -------
    __call__ :
        main method to process image pairs

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
                 trust_1st_iter=True,
                 **kwargs):
        # type declarations
        cdef Py_ssize_t K, I, J, nb_iter_max
        cdef float diff

        # input checks
        ht, wd = frame_shape
        dt = np.float32(dt)
        ws_iters = (window_size_iters,) if type(window_size_iters) == int else window_size_iters
        num_ws = len(ws_iters)
        self.dt = dt
        self.deform = deform
        self.smooth = smooth
        self.nb_iter_max = nb_iter_max = sum(ws_iters)
        self.nb_validation_iter = nb_validation_iter
        self.trust_1st_iter = trust_1st_iter

        # windows sizes
        ws = [np.power(2, num_ws - i - 1) * min_window_size for i in range(num_ws) for j in range(ws_iters[i])]

        # validation method
        self.val_tols = [None, None, None, None]
        val_methods = validation_method if type(validation_method) == str else (validation_method,)
        if 's2n' in val_methods:
            self.val_tols[1] = kwargs['s2n_tol'] if 's2n_tol' in kwargs else 1.25  # default tolerance
        if 'median_velocity' in val_methods:
            self.val_tols[1] = kwargs['median_tol'] if 'median_tol' in kwargs else 2  # default tolerance
        if 'mean_velocity' in val_methods:
            self.val_tols[2] = kwargs['mean_tol'] if 'mean_tol' in kwargs else 2  # default tolerance
        if 'rms_velocity' in val_methods:
            self.val_tols[2] = kwargs['rms_tol'] if 'mean_tol' in kwargs else 2  # default tolerance

        # other parameters
        self.smoothing_par = kwargs['smoothing_par'] if 'smoothing_par' in kwargs else None
        self.return_sig2noise = kwargs['sig2noise'] if 'sig2noise' in kwargs else True
        self.sig2noise_method = kwargs['sig2noise_method'] if 'sig2noise_method' in kwargs else 'peak2peak'
        self.s2n_width = kwargs['s2n_width'] if 's2n_width' in kwargs else 2
        self.nfftx = kwargs['nfftx'] if 'nfftx' in kwargs else None

        # Cython buffer definitions
        cdef DTYPEi_t[:] n_row = np.zeros(nb_iter_max, dtype=DTYPEi)
        cdef DTYPEi_t[:] n_col = np.zeros(nb_iter_max, dtype=DTYPEi)
        cdef DTYPEi_t[:] w = np.asarray(ws, dtype=DTYPEi)
        cdef DTYPEi_t[:] overlap = np.zeros(nb_iter_max, dtype=DTYPEi)

        # overlap init
        for K in range(nb_iter_max):
            overlap[K] = int(np.floor(overlap_ratio * w[K]))

        # n_col and n_row init
        for K in range(nb_iter_max):
            n_row[K] = (ht - w[K]) // (w[K] - overlap[K]) + 1
            n_col[K] = (wd - w[K]) // (w[K] - overlap[K]) + 1

        self.n_row = np.asarray(n_row)
        self.n_col = np.asarray(n_col)
        self.w = np.asarray(w)
        self.overlap = np.asarray(overlap)

        # # define temporary arrays and reshaped arrays to store the correlation function output
        # cdef np.ndarray[DTYPEf_t, ndim=1] i_tmp = np.zeros(n_row[-1] * n_col[-1], dtype=DTYPEf)  # delete
        # cdef np.ndarray[DTYPEf_t, ndim=1] j_tmp = np.zeros(n_row[-1] * n_col[-1], dtype=DTYPEf)
        # cdef np.ndarray[DTYPEf_t, ndim=2] i_peak = np.zeros([n_row[nb_iter_max - 1], n_col[nb_iter_max - 1]], dtype=DTYPEf)
        # cdef np.ndarray[DTYPEf_t, ndim=2] j_peak = np.zeros([n_row[nb_iter_max - 1], n_col[nb_iter_max - 1]], dtype=DTYPEf)
        self.i_tmp = np.zeros(n_row[-1] * n_col[-1], dtype=DTYPEf)
        self.j_tmp = np.zeros(n_row[-1] * n_col[-1], dtype=DTYPEf)
        self.i_peak = np.zeros([n_row[nb_iter_max - 1], n_col[nb_iter_max - 1]], dtype=DTYPEf)
        self.j_peak = np.zeros([n_row[nb_iter_max - 1], n_col[nb_iter_max - 1]], dtype=DTYPEf)

        # define arrays for signal to noise ratio
        # cdef np.ndarray[DTYPEf_t, ndim=2] sig2noise = np.zeros([n_row[-1], n_col[-1]], dtype=DTYPEf)  # delete
        self.sig2noise = np.zeros([n_row[-1], n_col[-1]], dtype=DTYPEf)

        # define arrays used for the validation process
        # in validation list, a 1 means that the location does not need to be validated. A 0 means that it does need to be validated  # delete
        # cdef DTYPEi_t[:, :] validation_list = np.ones([n_row[-1], n_col[-1]], dtype=DTYPEi)
        self.val_list = np.ones([n_row[-1], n_col[-1]], dtype=DTYPEi)

        # GPU ARRAYS
        # define the main array f that contains all the data
        cdef np.ndarray[DTYPEf_t, ndim=4] f = np.ones([nb_iter_max, n_row[nb_iter_max - 1], n_col[nb_iter_max - 1], 7], dtype=DTYPEf)

        # initialize x and y values
        for K in range(nb_iter_max):
            diff = w[K] - overlap[K]
            # the two lines below are marginally slower (O(1 ms)) than the 6 lines below them
            # f[K, :n_row[K], :n_col[K], 0] = np.tile(np.linspace(w[K] / 2, w[K] / 2 + diff * (n_col[K] - 1), n_col[K], dtype=np.float32), (n_row[K], 1))  # init x on cols
            # f[K, :n_row[K], :n_col[K], 1] = np.tile(np.linspace(w[K] / 2, w[K] / 2 + diff * (n_row[K] - 1), n_row[K], dtype=np.float32), (n_col[K], 1)).T  # init y on rows
            f[K, :, 0, 0] = w[K] / 2  # init x on first column
            f[K, 0, :, 1] = w[K] / 2  # init y on first row

            # init x on subsequent columns
            for J in range(1, n_col[K]):
                f[K, :, J, 0] = f[K, 0, J - 1, 0] + diff
            # init y on subsequent rows
            for I in range(1, n_row[K]):
                f[K, I, :, 1] = f[K, I - 1, 0, 1] + diff

        cdef DTYPEi_t[:, :] x_idx
        cdef DTYPEi_t[:, :] y_idx

        if mask is not None:
            assert mask.shape == (ht, wd), 'Mask is not same shape as image!'

            for K in range(nb_iter_max):
                x_idx = f[K, :, :, 0].astype(DTYPEi)
                y_idx = f[K, :, :, 1].astype(DTYPEi)
                f[K, :, :, 6] = mask[y_idx, x_idx].astype(DTYPEf)

        # Initialize skcuda miscellaneous library
        cu_misc.init()

        # Move f to the GPU for the whole calculation
        self.d_f = gpuarray.to_gpu(f)

        # define arrays to store the displacement vector
        self.d_shift = gpuarray.zeros([2, n_row[-1], n_col[-1]], dtype=DTYPEf)
        self.d_strain = gpuarray.zeros([4, n_row[-1], n_col[-1]], dtype=DTYPEf)

        # define arrays to store all the mean velocity at each point in each iteration
        self.d_u_mean = gpuarray.zeros([nb_iter_max, n_row[-1], n_col[-1]], dtype=DTYPEf)
        self.d_v_mean = gpuarray.zeros([nb_iter_max, n_row[-1], n_col[-1]], dtype=DTYPEf)

    def __call__(self, frame_a, frame_b):
        """Processes an image pair.

        Parameters
        ----------
        frame_a, frame_b : ndarray
            2D float, integers containing grey levels of the first and second frames.

        Returns
        -------
        x : array
            2D, the x-axis component of the interpolation locations, in pixels-lengths starting from 0 at left edge.
        y : array
            2D, the y-axis component of the interpolation locations, in pixels starting from 0 at bottom edge.
        u : array
            2D, the u velocity component, in pixels/seconds.
        v : array
            2D, the v velocity component, in pixels/seconds.
        mask : ndarray
            2D, the boolean values (True for vectors interpolated from previous iteration).
        s2n : ndarray
            2D, the signal to noise ratio of the final velocity field.

        """
        # recover class variables
        dt = self.dt
        deform = self.deform
        smoothing = self.smooth
        nb_iter_max = self.nb_iter_max
        nb_validation_iter = self.nb_validation_iter
        trust_1st_iter = self.trust_1st_iter
        val_tols = self.val_tols
        smoothing_par = self.smoothing_par
        sig2noise_method = self.sig2noise_method
        d_f =  self.d_f
        d_shift = self.d_shift
        d_strain = self.d_strain
        d_u_mean = self.d_u_mean
        d_v_mean = self.d_v_mean
        d_shift_arg = None
        d_strain_arg = None

        # type declarations
        cdef Py_ssize_t K  # main iteration index
        cdef Py_ssize_t i, j  # indices for various works
        cdef int n_val
        cdef float residual

        # Cython buffers
        cdef DTYPEi_t[:] n_row = self.n_row
        cdef DTYPEi_t[:] n_col = self.n_col
        cdef DTYPEi_t[:] w = self.w
        cdef DTYPEi_t[:] overlap = self.overlap
        cdef DTYPEf_t[:] i_tmp = self.i_tmp
        cdef DTYPEf_t[:] j_tmp = self.j_tmp
        cdef DTYPEf_t[:, :] i_peak = self.i_peak
        cdef DTYPEf_t[:, :] j_peak = self.j_peak
        cdef DTYPEf_t[:, :] sig2noise = self.sig2noise
        cdef DTYPEi_t[:, :] val_list = self.val_list

        # cast to 32-bit floats
        d_frame_a_f = gpuarray.to_gpu(frame_a.astype(np.float32))
        d_frame_b_f = gpuarray.to_gpu(frame_b.astype(np.float32))

        # MAIN LOOP
        for K in range(nb_iter_max):
            print("//////////////////////////////////////////////////////////////////")
            print("ITERATION {}".format(K))

            if K > 0:
                # Calculate second frame displacement (shift)
                d_shift[0, :n_row[K], :n_col[K]] = d_f[K, :n_row[K], :n_col[K], 4]  # xb = xa + dpx
                d_shift[1, :n_row[K], :n_col[K]] = d_f[K, :n_row[K], :n_col[K], 5]  # yb = ya + dpy
                d_shift_arg = d_shift[:, :n_row[K], :n_col[K]]

                # calculate the strain rate tensor
                if deform:
                    d_strain_arg = d_strain[:, :n_row[K], :n_col[K]]
                    if w[K] != w[K - 1]:
                        gpu_gradient(d_strain_arg, d_f[K, :n_row[K], :n_col[K], 2].copy(), d_f[K, :n_row[K], :n_col[K], 3].copy(), n_row[K], n_col[K], w[K] - overlap[K])
                    else:
                        gpu_gradient(d_strain_arg, d_f[K - 1, :n_row[K - 1], :n_col[K - 1], 2].copy(), d_f[K - 1, :n_row[K - 1], :n_col[K - 1], 3].copy(), n_row[K], n_col[K], w[K] - overlap[K])

            # Get correlation function
            c = CorrelationFunction(d_frame_a_f, d_frame_b_f, w[K], overlap[K], self.nfftx, d_shift=d_shift_arg, d_strain=d_strain_arg)

            # Get window displacement to subpixel accuracy
            sp_i, sp_j = c.subpixel_peak_location()
            # i_tmp[:n_row[K] * n_col[K]] = sp_i
            # j_tmp[:n_row[K] * n_col[K]] = sp_j
            # i_tmp[:n_row[K] * n_col[K]], j_tmp[:n_row[K] * n_col[K]] = c.subpixel_peak_location()  # delete

            # reshape the peaks
            # i_peak[:n_row[K], :n_col[K]] = np.reshape(i_tmp[:n_row[K] * n_col[K]], (n_row[K], n_col[K]))
            # j_peak[:n_row[K], :n_col[K]] = np.reshape(j_tmp[:n_row[K] * n_col[K]], (n_row[K], n_col[K]))
            self.i_peak[:n_row[K], :n_col[K]] = np.reshape(sp_i, (n_row[K], n_col[K]))
            self.j_peak[:n_row[K], :n_col[K]] = np.reshape(sp_j, (n_row[K], n_col[K]))
            assert not np.isnan(i_peak).any(), 'NaNs in correlation i-peaks!'
            assert not np.isnan(j_peak).any(), 'NaNs in correlation j-peaks!'

            # Get signal to noise ratio
            # sig2noise[0:n_row[K], 0:n_col[K]] = c.sig2noise_ratio(method=sig2noise_method)
            if self.return_sig2noise:
                self.sig2noise = c.sig2noise_ratio(method=sig2noise_method, width=self.s2n_width)

            # update the field with new values
            # gpu_update(d_f, self.i_peak[:n_row[K], :n_col[K]], self.j_peak[:n_row[K], :n_col[K]], n_row[K], n_col[K], K)
            gpu_update(d_f, self.i_peak[:n_row[K], :n_col[K]], self.j_peak[:n_row[K], :n_col[K]], n_row[K], n_col[K], K)

            # normalize the residual by the maximum quantization error of 0.5 pixel
            try:
                residual = np.sum(np.power(i_peak[:n_row[K], :n_col[K]], 2) + np.power(j_peak[:n_row[K], :n_col[K]], 2))
                print("[DONE]--Normalized residual : {}.\n".format(sqrt(residual / (0.5 * n_row[K] * n_col[K]))))
            except OverflowError:
                print('[DONE]--Overflow in residuals.\n')

            # VALIDATION
            if K == 0 and trust_1st_iter:
                print('No validation--trusting 1st iteration.')

            for i in range(nb_validation_iter):
                print('Validation iteration {}:'.format(i))

                # # reset validation list  # delete
                # validation_list = np.ones([n_row[-1], n_col[-1]], dtype=DTYPEi)

                # get list of places that need to be validated
                # validation_list[:n_row[K], :n_col[K]], d_u_mean[K, :n_row[K], :n_col[K]], d_v_mean[K, :n_row[K], :n_col[K]] = gpu_validation(d_f, K, sig2noise[:n_row[K], :n_col[K]], n_row[K], n_col[K], w[K], *val_tols)
                self.val_list[:n_row[K], :n_col[K]], d_u_mean[K, :n_row[K], :n_col[K]], d_v_mean[K, :n_row[K], :n_col[K]] = gpu_validation(d_f, K, sig2noise[:n_row[K], :n_col[K]], n_row[K], n_col[K], w[K], *val_tols)

                # do the validation
                # n_val = n_row[K] * n_col[K] - np.sum(validation_list[:n_row[K], :n_col[K]])
                n_val = n_row[K] * n_col[K] - np.sum(val_list[:n_row[K], :n_col[K]])
                if n_val > 0:
                    print('Validating {} out of {} vectors ({:.2%}).'.format(n_val, n_row[K] * n_col[K], n_val / (n_row[K] * n_col[K])))
                    # gpu_replace_vectors(d_f, validation_list, d_u_mean, d_v_mean, nb_iter_max, K, n_row, n_col, w, overlap, dt)
                    gpu_replace_vectors(d_f, self.val_list, d_u_mean, d_v_mean, nb_iter_max, K, n_row, n_col, w, overlap, dt)
                else:
                    print('No invalid vectors!')

                print("[DONE]\n")

            # NEXT ITERATION
            # go to next iteration: compute the predictors dpx and dpy from the current displacements
            if K < nb_iter_max - 1:
                # interpolate if dimensions do not agree
                if w[K + 1] != w[K]:
                    v_list = np.ones((n_row[-1], n_col[-1]), dtype=bool)

                    # interpolate velocity onto next iterations grid. Then use it as the predictor for the next step
                    gpu_interpolate_surroundings(d_f, v_list, n_row, n_col, w, overlap, K, 2)
                    gpu_interpolate_surroundings(d_f, v_list, n_row, n_col, w, overlap, K, 3)
                    if smoothing:
                        d_f[K + 1, :n_row[K + 1], :n_col[K + 1], 4] = gpu_smooth(d_f[K + 1, :n_row[K + 1], :n_col[K + 1], 2].copy(), s=smoothing_par, retain_input=True)  # check if .copy() is needed
                        d_f[K + 1, :n_row[K + 1], :n_col[K + 1], 5] = gpu_smooth(d_f[K + 1, :n_row[K + 1], :n_col[K + 1], 3].copy(), s=smoothing_par, retain_input=True)

                else:
                    if smoothing:
                        d_f[K + 1, :n_row[K + 1], :n_col[K + 1], 4] = gpu_smooth(d_f[K, :n_row[K], :n_col[K], 2], s=smoothing_par, retain_input=True)
                        d_f[K + 1, :n_row[K + 1], :n_col[K + 1], 5] = gpu_smooth(d_f[K, :n_row[K], :n_col[K], 3], s=smoothing_par, retain_input=True)
                    else:
                        d_f[K + 1, :, :, 4] = d_f[K + 1, :, :, 2].copy()
                        d_f[K + 1, :, :, 5] = d_f[K + 1, :, :, 3].copy()

                print("[DONE] -----> going to iteration {}.\n".format(K + 1))

        # RETURN RESULTS
        print("//////////////////////////////////////////////////////////////////")

        f = d_f.get()

        # assemble the u, v and x, y fields for outputs
        k_f = nb_iter_max - 1
        x = f[k_f, :, :, 0]
        y = f[k_f, ::-1, :, 1]
        u = f[k_f, :, :, 2] / dt
        v = -f[k_f, :, :, 3] / dt
        mask = f[k_f, :, :, 6]

        print("[DONE]\n")

        # # delete images from gpu memory
        d_frame_a_f.gpudata.free()
        d_frame_b_f.gpudata.free()
        # d_f.gpudata.free()
        # d_shift.gpudata.free()
        # d_strain.gpudata.free()
        # d_u_mean.gpudata.free()
        # d_v_mean.gpudata.free()

        if self.return_sig2noise:
            return x, y, u, v, mask, self.sig2noise
        else:
            return x, y, u, v, mask


def gpu_replace_vectors(d_f, validation_list, d_u_mean, d_v_mean, nb_iter_max, k, n_row, n_col, w, overlap, dt):
    """Initiate the full GPU version of the validation and interpolation.

    Parameters
    ----------
    d_f : GPUArray
        3D float, main array that stores all velocity data
    validation_list : ndarray
        2D int, indicates which values must be validate. 1 indicates no validation needed, 0 indicated validation is needed
    d_u_mean, d_v_mean : GPUArray
        3D float, mean velocity surrounding each point
    nb_iter_max : int
        total number of iterations
    k : int
        main loop iteration count
    n_row, n_col : ndarray
        int, number of rows an columns in each main loop iteration
    w : ndarray
        int, pixels between interrogation windows
    overlap : ndarray
        int, ratio of overlap between interrogation windows
    dt : float
        time between image frames

    """
    # check the inputs
    assert validation_list.shape == (n_row[-1], n_col[-1]), "Must pass the full validation list, not just the section for the iteration you are validating."
    assert d_u_mean.shape == (nb_iter_max, n_row[-1], n_col[-1]), "Must pass the entire d_u_mean array, not just the section for the iteration you are validating."
    assert d_v_mean.shape == (nb_iter_max, n_row[-1], n_col[-1]), "Must pass the entire d_v_mean array, not just the section for the iteration you are validating."

    # change validation_list to type boolean and invert it. Now - True indicates that point needs to be validated, False indicates no validation
    validation_location = np.invert(validation_list.astype(bool))

    # first iteration, just replace with mean velocity
    if k == 0:
        # get indices and send them to the gpu
        indices = np.where(validation_location.flatten() == 1)[0].astype(DTYPEi)
        d_indices = gpuarray.to_gpu(indices)

        # get mean velocity at validation points
        d_u_tmp = gpu_array_index(d_u_mean[k, :, :].copy(), d_indices, np.float32, retain_list=True)
        d_v_tmp = gpu_array_index(d_v_mean[k, :, :].copy(), d_indices, np.float32, retain_list=True)

        # update the velocity values
        d_f[k, :, :, 2] = gpu_index_update(d_f[k, :, :, 2].copy(), d_u_tmp, d_indices, retain_indices=True)  # u
        d_f[k, :, :, 3] = gpu_index_update(d_f[k, :, :, 3].copy(), d_v_tmp, d_indices)  # v

        # you don't need to do all these calculations. Could write a function that only does it for the ones that have been validated

    # case if different dimensions: interpolation using previous iteration
    elif k > 0 and (n_row[k] != n_row[k - 1] or n_col[k] != n_col[k - 1]):
        gpu_interpolate_surroundings(d_f, validation_location, n_row, n_col, w, overlap, k - 1, 2)  # u
        gpu_interpolate_surroundings(d_f, validation_location, n_row, n_col, w, overlap, k - 1, 3)  # v

    # case if same dimensions
    elif k > 0 and (n_row[k] == n_row[k - 1] or n_col[k] == n_col[k - 1]):
        # get indices and send them to the gpu
        indices = np.where(validation_location.flatten() == 1)[0].astype(DTYPEi)
        d_indices = gpuarray.to_gpu(indices)

        # update the velocity values with the previous values.
        # This is essentially a bilinear interpolation when the value is right on top of the other.
        # could replace with the mean of the previous values surrounding the point
        d_u_tmp = gpu_array_index(d_f[k - 1, :, :, 2].copy(), d_indices, np.float32, retain_list=True)
        d_v_tmp = gpu_array_index(d_f[k - 1, :, :, 3].copy(), d_indices, np.float32, retain_list=True)
        d_f[k, :, :, 2] = gpu_index_update(d_f[k, :, :, 2].copy(), d_u_tmp, d_indices, retain_indices=True)
        d_f[k, :, :, 3] = gpu_index_update(d_f[k, :, :, 3].copy(), d_v_tmp, d_indices)


def gpu_interpolate_surroundings(d_f, v_list, n_row, n_col, w, overlap, k, dat):
    """Interpolate a point based on the surroundings.

    Parameters
    ----------
    d_f : GPUArray - 4D float
        main array that stores all velocity data
    v_list : array - 2D bool
        indicates which values must be validated. True means it needs to be validated, False means no validation is needed.
    n_row, n_col : array - 1D
        number rows and columns in each iteration
    w : ndarray
       int,  number of pixels between interrogation windows
    overlap : ndarray
        int, overlap of the interrogation windows
    k : int
        current iteration
    dat : int
        data that needs to be interpolated. 4th index in the F array

    """
    #### Separate validation list into multiple lists for each region ####

    # set all sides to false for interior points
    interior_list = np.copy(v_list[:n_row[k + 1], :n_col[k + 1]]).astype('bool')
    interior_list[0,:] = 0
    interior_list[-1,:] = 0
    interior_list[:,0] = 0
    interior_list[:,-1] = 0

    # define array with the indices of the points to be validated
    interior_ind = np.where(interior_list.flatten() == True)[0].astype(DTYPEi)
    if interior_ind.size != 0:
        # get the x and y indices of the interior points that must be validated
        interior_ind_x = interior_ind // n_col[k + 1]
        interior_ind_y = interior_ind % n_col[k + 1]
        d_interior_ind_x = gpuarray.to_gpu(interior_ind_x)
        d_interior_ind_y = gpuarray.to_gpu(interior_ind_y)

        # use this to update the final d_F array after the interpolation
        d_interior_ind = gpuarray.to_gpu(interior_ind)

    # only select sides and remove corners
    top_list = np.copy(v_list[0, :n_col[k + 1]])
    top_list[0] = 0
    top_list[-1] = 0
    top_ind = np.where(top_list.flatten() == True)[0].astype(DTYPEi)
    if top_ind.size != 0:
        d_top_ind = gpuarray.to_gpu(top_ind)

    bottom_list = np.copy(v_list[n_row[k + 1] - 1, :n_col[k + 1]])
    bottom_list[0] = 0
    bottom_list[-1] = 0
    bottom_ind = np.where(bottom_list.flatten() == True)[0].astype(DTYPEi)
    if bottom_ind.size != 0:
        d_bottom_ind = gpuarray.to_gpu(bottom_ind)

    left_list = np.copy(v_list[:n_row[k + 1], 0])
    left_list[0] = 0
    left_list[-1] = 0
    left_ind = np.where(left_list.flatten() == True)[0].astype(DTYPEi)
    if left_ind.size != 0:
        d_left_ind = gpuarray.to_gpu(left_ind)

    right_list = np.copy(v_list[:n_row[k + 1], n_col[k + 1] - 1])
    right_list[0] = 0
    right_list[-1] = 0
    right_ind = np.where(right_list.flatten() == True)[0].astype(DTYPEi)
    if right_ind.size != 0:
        d_right_ind = gpuarray.to_gpu(right_ind)

    drv.Context.synchronize()

    #--------------------------INTERIOR GRID---------------------------------
    if interior_ind.size != 0:
        # get gpu data for position now
        d_low_x, d_high_x = f_dichotomy_gpu(d_f[k:k + 2, :, 0, 0].copy(), k, "x_axis", d_interior_ind_x, w, overlap, n_row, n_col)
        d_low_y, d_high_y = f_dichotomy_gpu(d_f[k:k + 2, 0, :, 1].copy(), k, "y_axis", d_interior_ind_y, w, overlap, n_row, n_col)

        # get indices surrounding the position now
        d_x1 = gpu_array_index(d_f[k, :n_row[k], 0, 0].copy(), d_low_x, np.float32, retain_list=True)
        d_x2 = gpu_array_index(d_f[k, :n_row[k], 0, 0].copy(), d_high_x, np.float32, retain_list=True)
        d_y1 = gpu_array_index(d_f[k, 0, :n_col[k], 1].copy(), d_low_y, np.float32, retain_list=True)
        d_y2 = gpu_array_index(d_f[k, 0, :n_col[k], 1].copy(), d_high_y, np.float32, retain_list=True)
        d_x = gpu_array_index(d_f[k + 1, :n_row[k + 1], 0, 0].copy(), d_interior_ind_x, np.float32)
        d_y = gpu_array_index(d_f[k + 1, 0, :n_col[k + 1], 1].copy(), d_interior_ind_y, np.float32)

        # get indices for the function values at each spot surrounding the validation points.
        d_f1_ind = d_low_x * n_col[k] + d_low_y
        d_f2_ind = d_low_x * n_col[k] + d_high_y
        d_f3_ind = d_high_x * n_col[k] + d_low_y
        d_f4_ind = d_high_x * n_col[k] + d_high_y

        # return the values of the function surrounding the validation point
        d_f1 = gpu_array_index(d_f[k, :n_row[k], :n_col[k], dat].copy(), d_f1_ind, np.float32)
        d_f2 = gpu_array_index(d_f[k, :n_row[k], :n_col[k], dat].copy(), d_f2_ind, np.float32)
        d_f3 = gpu_array_index(d_f[k, :n_row[k], :n_col[k], dat].copy(), d_f3_ind, np.float32)
        d_f4 = gpu_array_index(d_f[k, :n_row[k], :n_col[k], dat].copy(), d_f4_ind, np.float32)

        # Do interpolation
        d_interior_bilinear = bilinear_interp_gpu(d_x1, d_x2, d_y1, d_y2, d_x, d_y, d_f1, d_f2, d_f3, d_f4)

        # Update values. Return a tmp array and destroy after to avoid GPU memory leak.
        d_tmp_ib = gpu_index_update(d_f[k + 1, :n_row[k + 1], :n_col[k + 1], dat].copy(), d_interior_bilinear, d_interior_ind)
        d_f[k + 1, :n_row[k + 1], :n_col[k + 1], dat] = d_tmp_ib

        # free some GPU memory
        d_low_x.gpudata.free()
        d_low_y.gpudata.free()
        d_high_x.gpudata.free()
        d_high_y.gpudata.free()
        d_tmp_ib.gpudata.free()

        drv.Context.synchronize()

    #------------------------------SIDES-----------------------------------
    if top_ind.size > 0:
        # get now position and surrounding points
        d_low_y, d_high_y = f_dichotomy_gpu(d_f[k:k + 2, 0, :, 1].copy(), k, "y_axis", d_top_ind, w, overlap, n_row, n_col)

        # Get values to compute interpolation
        d_y1 = gpu_array_index(d_f[k, 0, :, 1].copy(), d_low_y, np.float32, retain_list=True)
        d_y2 = gpu_array_index(d_f[k, 0, :, 1].copy(), d_high_y, np.float32, retain_list=True)
        d_y = gpu_array_index(d_f[k + 1, 0, :, 1].copy(), d_top_ind, np.float32, retain_list=True)

        # return the values of the function surrounding the validation point
        d_f1 = gpu_array_index(d_f[k, 0, :, dat].copy(), d_low_y, np.float32)
        d_f2 = gpu_array_index(d_f[k, 0, :, dat].copy(), d_high_y, np.float32)

        # do interpolation
        d_top_linear = linear_interp_gpu(d_y1, d_y2, d_y, d_f1, d_f2)

        # Update values. Return a tmp array and destroy after to avoid GPU memory leak.
        d_tmp_tl = gpu_index_update(d_f[k + 1, 0, :n_col[k + 1], dat].copy(), d_top_linear, d_top_ind)
        d_f[k + 1, 0, :n_col[k + 1], dat] = d_tmp_tl

        # free some data
        d_tmp_tl.gpudata.free()

        drv.Context.synchronize()

    # BOTTOM
    if bottom_ind.size > 0:
        # get position data
        d_low_y, d_high_y = f_dichotomy_gpu(d_f[k:k + 2, 0, :, 1].copy(), k, "y_axis", d_bottom_ind, w, overlap, n_row, n_col)

        # Get values to compute interpolation
        d_y1 = gpu_array_index(d_f[k, int(n_row[k] - 1), :, 1].copy(), d_low_y, np.float32, retain_list=True)
        d_y2 = gpu_array_index(d_f[k, int(n_row[k] - 1), :, 1].copy(), d_high_y, np.float32, retain_list=True)
        d_y = gpu_array_index(d_f[k + 1, int(n_row[k + 1] - 1), :, 1].copy(), d_bottom_ind, np.float32, retain_list=True)

        # return the values of the function surrounding the validation point
        d_f1 = gpu_array_index(d_f[k, int(n_row[k] - 1), :, dat].copy(), d_low_y, np.float32)
        d_f2 = gpu_array_index(d_f[k, int(n_row[k] - 1), :, dat].copy(), d_high_y, np.float32)

        # do interpolation
        d_bottom_linear = linear_interp_gpu(d_y1, d_y2, d_y, d_f1, d_f2)

        # Update values. Return a tmp array and destroy after to avoid GPU memory leak.
        d_tmp_bl = gpu_index_update(d_f[k + 1, int(n_row[k + 1] - 1), :n_col[k + 1], dat].copy(), d_bottom_linear, d_bottom_ind)
        d_f[k + 1, int(n_row[k + 1] - 1), :n_col[k + 1], dat] = d_tmp_bl

        # free some data
        d_tmp_bl.gpudata.free()

        drv.Context.synchronize()

    # LEFT
    if left_ind.size > 0:
        # get position data
        d_low_x, d_high_x = f_dichotomy_gpu(d_f[k:k + 2, :, 0, 0].copy(), k, "x_axis", d_left_ind, w, overlap, n_row, n_col)

        # Get values to compute interpolation
        d_x1 = gpu_array_index(d_f[k, :, 0, 0].copy(), d_low_x, np.float32, retain_list=True)
        d_x2 = gpu_array_index(d_f[k, :, 0, 0].copy(), d_high_x, np.float32, retain_list=True)
        d_x = gpu_array_index(d_f[k + 1, :, 0, 0].copy(), d_left_ind, np.float32, retain_list=True)

        # return the values of the function surrounding the validation point
        d_f1 = gpu_array_index(d_f[k, :, 0, dat].copy(), d_low_x, np.float32)
        d_f2 = gpu_array_index(d_f[k, :, 0, dat].copy(), d_high_x, np.float32)

        # do interpolation
        d_left_linear = linear_interp_gpu(d_x1, d_x2, d_x, d_f1, d_f2)

        # Update values. Return a tmp array and destroy after to avoid GPU memory leak.
        d_tmp_ll = gpu_index_update(d_f[k + 1, :n_row[k + 1], 0, dat].copy(), d_left_linear, d_left_ind)
        d_f[k + 1, :n_row[k + 1], 0, dat] = d_tmp_ll

        # free some data
        d_tmp_ll.gpudata.free()

        drv.Context.synchronize()

    # RIGHT
    if right_ind.size > 0:
        # get position data
        d_low_x, d_high_x = f_dichotomy_gpu(d_f[k:k + 2, :, 0, 0].copy(), k, "x_axis", d_right_ind, w, overlap, n_row, n_col)

        # Get values to compute interpolation
        d_x1 = gpu_array_index(d_f[k, :, int(n_col[k] - 1), 0].copy(), d_low_x, np.float32, retain_list=True)
        d_x2 = gpu_array_index(d_f[k, :, int(n_col[k] - 1), 0].copy(), d_high_x, np.float32, retain_list=True)
        d_x = gpu_array_index(d_f[k + 1, :, int(n_col[k + 1] - 1), 0].copy(), d_right_ind, np.float32, retain_list=True)

        # return the values of the function surrounding the validation point
        d_f1 = gpu_array_index(d_f[k, :, int(n_col[k] - 1), dat].copy(), d_low_x, np.float32)
        d_f2 = gpu_array_index(d_f[k, :, int(n_col[k] - 1), dat].copy(), d_high_x, np.float32)

        # do interpolation
        d_right_linear = linear_interp_gpu(d_x1, d_x2, d_x, d_f1, d_f2)

        # Update values. Return a tmp array and destroy after to avoid GPU memory leak.
        d_tmp_rl = gpu_index_update(d_f[k + 1, :n_row[k + 1], int(n_col[k + 1] - 1), dat].copy(), d_right_linear, d_right_ind)
        d_f[k + 1, :n_row[k + 1], int(n_col[k + 1] - 1), dat] = d_tmp_rl

        # free some data
        d_tmp_rl.gpudata.free()


    # ----------------------------CORNERS-----------------------------------
    # top left
    if v_list[0, 0] == 1:
        d_f[k + 1, 0, 0, dat] = d_f[k, 0, 0, dat]
    # top right
    if v_list[0, n_col[k + 1] - 1] == 1:
        d_f[k + 1, 0, int(n_col[k + 1] - 1), dat] = d_f[k, 0, int(n_col[k] - 1), dat]
    # bottom left
    if v_list[n_row[k + 1] - 1, 0] == 1:
        d_f[k + 1, int(n_row[k + 1] - 1), 0, dat] = d_f[k, int(n_row[k] - 1), 0, dat]
    # bottom right
    if v_list[n_row[k + 1] - 1, n_col[k + 1] - 1] == 1:
        d_f[k + 1, int(n_row[k + 1] - 1), int(n_col[k + 1] - 1), dat] = d_f[k, int(n_row[k] - 1), int(n_col[k] - 1), dat]


#  CUDA GPU FUNCTIONS
def gpu_update(d_f, i_peak, j_peak, n_row, n_col, k):
    """Function to update the velocity values after an iteration in the WiDIM algorithm

    Parameters
    ---------
    d_f : GPUArray - 4D float
        main array in WiDIM algorithm
    i_peak, j_peak : array - 2D float
        correlation function peak at each iteration
    n_row, n_col : int
        number of rows and columns in the current iteration
    k : int
        main loop iteration

    """
    mod_update = SourceModule("""
        __global__ void update_values(float *F, float *i_peak, float *j_peak, int fourth_dim)
        {
            // F is where all the data is stored at a particular K
            // i_peak / j_peak is the correlation peak location
            // sig2noise = sig2noise ratio from correlation function
            // cols = number of columns of IWs
            // fourth_dim  = size of the fourth dimension of F

            int w_idx = blockIdx.x * blockDim.x + threadIdx.x;

            // Index for each IW in the F array
            int F_idx = w_idx * fourth_dim;

            // get new displacement prediction
            F[F_idx + 2] = (F[F_idx + 4] + j_peak[w_idx]) * F[F_idx + 6];
            F[F_idx + 3] = (F[F_idx + 5] + i_peak[w_idx]) * F[F_idx + 6];
        }
        """)

    # GPU parameters
    block_size = 32
    x_blocks = int(n_col * n_row // block_size + 1)

    # move data to gpu
    d_i_peak = gpuarray.to_gpu(i_peak)
    d_j_peak = gpuarray.to_gpu(j_peak)
    d_f_tmp = d_f[k, 0:n_row, 0:n_col, :].copy()

    # last dimension of F
    fourth_dim = np.int32(d_f.shape[-1])

    # update the values
    update_values = mod_update.get_function("update_values")
    update_values(d_f_tmp, d_i_peak, d_j_peak, fourth_dim, block=(block_size, 1, 1), grid=(x_blocks, 1))
    d_f[k, 0:n_row, 0:n_col, :] = d_f_tmp

    # Free gpu memory
    d_f_tmp.gpudata.free()
    d_i_peak.gpudata.free()
    d_j_peak.gpudata.free()


# gpu validation algorithms were removed from here


def f_dichotomy_gpu(d_range, k, side, d_pos_index, w, overlap, n_row, n_col):
    """
    Look for the position of the vectors at the previous iteration that surround the current point in the frame
    you want to validate. Returns the low and high index of the points from the previous iteration on either side of
    the point in the current iteration that needs to be validated.

    Parameters
    ----------
    d_range : GPUArray - 2D
        The x or y locations along the grid for the current and next iteration.
        Example:
        For side = x_axis then the input looks like d_range = d_F[K:K+2, :,0,0].copy()
        For side = y_axis then the input looks like d_range = d_F[K:K+2, 0,:,1].copy()
    k : int
        the iteration you want to use to validate. Typically the previous iteration from the
        one that the code is in now. (1st index for F).
    side : string
        the axis of interest : can be either 'x_axis' or 'y_axis'
    d_pos_index : GPUArray
        1D int, index of the point in the frame you want to validate (along the axis 'side').
    w : ndarray
        1D int, array of window sizes
    overlap : ndarray
        1D int, overlap in number of pixels
    n_row, n_col : ndarray
        1D int, number of rows and columns in the F dataset in each iteration

    Returns
    -------
    d_low : GPUArray - 1D int
        largest index at the iteration K along the 'side' axis so that the position of index low in the frame is less than or equal to pos_now.
    d_high : GPUArray - 1D int
        smallest index at the iteration K along the 'side' axis so that the position of index low in the frame is greater than or equal to pos_now.

    """
    # GPU kernel
    mod_f_dichotomy = SourceModule("""
    __global__ void f_dichotomy_x(float *x, int *low, int *high, int K, int *pos_index, float w_a, float w_b, float dxa, float dxb, int Nrow, int NrowMax, int n)
    {
        int w_idx = blockIdx.x*blockDim.x + threadIdx.x;

        if(w_idx >= n){return;}

        // initial guess for low and high values
        low[w_idx] = (int)floorf((w_a/2. - w_b/2. + pos_index[w_idx]*dxa) / dxb);
        high[w_idx] = low[w_idx] + 1*(x[NrowMax + pos_index[w_idx]] != x[low[w_idx]]);

        // if lower than lowest
        low[w_idx] = low[w_idx] * (low[w_idx] >= 0);
        high[w_idx] = high[w_idx] * (low[w_idx] >= 0);

        // if higher than highest
        low[w_idx] = low[w_idx] + (Nrow - 1 - low[w_idx])*(high[w_idx] > Nrow - 1);
        high[w_idx] = high[w_idx] + (Nrow - 1 - high[w_idx])*(high[w_idx] > Nrow - 1);
    }

    __global__ void f_dichotomy_y(float *y, int *low, int *high, int K, int *pos_index, float w_a, float w_b, float dya, float dyb, int Ncol, int NcolMax, int n)
    {
        int w_idx = blockIdx.x*blockDim.x + threadIdx.x;

        if(w_idx >= n){return;}

        low[w_idx] = (int)floorf((w_a/2. - w_b/2. + pos_index[w_idx]*dya) / dyb);
        high[w_idx] = low[w_idx] + 1*(y[NcolMax + pos_index[w_idx]] != y[low[w_idx]]);

        // if lower than lowest
        low[w_idx] = low[w_idx] * (low[w_idx] >= 0);
        high[w_idx] = high[w_idx] * (low[w_idx] >= 0);

        // if higher than highest
        low[w_idx] = low[w_idx] + (Ncol - 1 - low[w_idx])*(high[w_idx] > Ncol - 1);
        high[w_idx] = high[w_idx] + (Ncol - 1 - high[w_idx])*(high[w_idx] > Ncol - 1);
    }
    """)

    # Define values needed for the calculations
    w_a = np.float32(w[k + 1])
    w_b = np.float32(w[k])
    k = np.int32(k)
    n = np.int32(d_pos_index.size)
    n_row = np.int32(n_row)
    n_col = np.int32(n_col)

    # define gpu settings
    block_size = 32
    x_blocks = int(len(d_pos_index)//block_size + 1)

    # create GPU data
    d_low = gpuarray.zeros_like(d_pos_index, dtype=DTYPEi)
    d_high = gpuarray.zeros_like(d_pos_index, dtype=DTYPEi)

    if side == "x_axis":
        assert d_pos_index[-1].get() < n_row[k + 1], "Position index for validation point is outside the grid. Not possible - all points should be on the grid."
        dxa = np.float32(w_a - overlap[k + 1])
        dxb = np.float32(w_b - overlap[k])

        # get gpu kernel
        f_dichotomy_x = mod_f_dichotomy.get_function("f_dichotomy_x")
        f_dichotomy_x(d_range, d_low, d_high, k, d_pos_index, w_a, w_b, dxa, dxb, n_row[k], n_row[-1], n, block=(block_size, 1, 1), grid=(x_blocks, 1))

    elif side == "y_axis":
        assert d_pos_index[-1].get() < n_col[k + 1], "Position index for validation point is outside the grid. Not possible - all points should be on the grid."
        dya = np.float32(w_a - overlap[k + 1])
        dyb = np.float32(w_b - overlap[k])

        # get gpu kernel
        f_dichotomy_y = mod_f_dichotomy.get_function("f_dichotomy_y")
        f_dichotomy_y(d_range, d_low, d_high, k, d_pos_index, w_a, w_b, dya, dyb, n_col[k], n_col[-1], n, block=(block_size, 1, 1), grid=(x_blocks, 1))

    else:
        raise ValueError("Not a proper axis. Choose either x or y axis.")

    # free gpu data
    d_range.gpudata.free()

    return d_low, d_high


def bilinear_interp_gpu(d_x1, d_x2, d_y1, d_y2, d_x, d_y, d_f1, d_f2, d_f3, d_f4):
    """Performs bilinear interpolation on the GPU."""

    mod_bi = SourceModule("""
    __global__ void bilinear_interp(float *f, float *x1, float *x2, float *y1, float *y2, float *x, float *y, float *f1, float *f2, float *f3, float *f4, int n)
    {
        // 1D grid of 1D blocks
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if(idx >= n){return;}

        // avoid the points that are equal to each other

        float n1 = f1[idx] * (x2[idx]-x[idx]) * (y2[idx]-y[idx]);
        n1 = n1 * (float)(y1[idx] != y2[idx]) + f1[idx] * (float)(y1[idx] == y2[idx]) * (x2[idx]-x[idx]);
        n1 = n1 * (float)(x1[idx] != x2[idx]) + f1[idx] * (float)(x1[idx] == x2[idx]) * (y2[idx]-y[idx]);
        n1 = n1 * (float)((y1[idx] != y2[idx]) || (x1[idx] != x2[idx])) + f1[idx] * (float)((y1[idx] == y2[idx]) && (x1[idx] == x2[idx]));

        float n2 = f2[idx] * (x2[idx]-x[idx]) * (y[idx]-y1[idx]);
        n2 = n2 * (float)(x1[idx] != x2[idx]) + f2[idx] * (float)(x1[idx] == x2[idx]) * (y[idx]-y1[idx]);
        n2 = n2 * (float)(y1[idx] != y2[idx]);

        float n3 = f3[idx] * (x[idx]-x1[idx]) * (y2[idx]-y[idx]);
        n3 = n3 * (float)(y1[idx] != y2[idx]) + f3[idx] * (float)(y1[idx] == y2[idx]) * (x[idx] - x1[idx]);
        n3 = n3 * (float)(x1[idx] != x2[idx]) * (x1[idx] != x2[idx]);

        float n4 = f4[idx] * (x[idx]-x1[idx]) * (y[idx]-y1[idx]);
        n4 = n4 * (float)(y1[idx] != y2[idx]) * (float)(x1[idx] != x2[idx]);

        float numerator = n1 + n2 + n3 + n4;

        float denominator = (x2[idx]-x1[idx])*(y2[idx]-y1[idx]);
        denominator = denominator * (float)(x1[idx] != x2[idx]) + (y2[idx] - y1[idx]) * (float)(x1[idx] == x2[idx]);
        denominator = denominator * (float)(y1[idx] != y2[idx]) + (x2[idx] - x1[idx]) * (float)(y1[idx] == y2[idx]);
        denominator = denominator * (float)((y1[idx] != y2[idx]) || (x1[idx] != x2[idx])) + 1.0 * (float)((y1[idx] == y2[idx]) && (x1[idx] == x2[idx]));

        f[idx] = numerator / denominator;
    }
    """)

    # define gpu parameters
    block_size = 32
    x_blocks = int(len(d_x1) // block_size + 1)
    n = np.int32(len(d_x1))

    d_f = gpuarray.zeros_like(d_x1, dtype=np.float32)

    # get kernel
    bilinear_interp = mod_bi.get_function("bilinear_interp")
    bilinear_interp(d_f, d_x1, d_x2, d_y1, d_y2, d_x, d_y, d_f1, d_f2, d_f3, d_f4, n, block=(block_size, 1, 1), grid=(x_blocks, 1))

    # free gpu data
    d_x1.gpudata.free()
    d_x2.gpudata.free()
    d_y1.gpudata.free()
    d_y2.gpudata.free()
    d_x.gpudata.free()
    d_y.gpudata.free()
    d_f1.gpudata.free()
    d_f2.gpudata.free()
    d_f3.gpudata.free()
    d_f4.gpudata.free()

    return d_f


def linear_interp_gpu(d_x1, d_x2, d_x, d_f1, d_f2):
    mod_lin = SourceModule("""
    __global__ void linear_interp(float *f, float *x1, float *x2, float *x, float *f1, float *f2, int n)
    {
        // 1D grid of 1D blocks
        int idx = blockIdx.x*blockDim.x + threadIdx.x;

        if(idx >= n){return;}

        float tmp = ((x2[idx]-x[idx])/(x2[idx]-x1[idx]))*f1[idx] + ((x[idx]-x1[idx])/(x2[idx]-x1[idx]))*f2[idx];
        f[idx] = tmp * (float)(x2[idx] != x1[idx]) + f1[idx]*(float)(x2[idx] == x1[idx]) ;
    }
    """)

    # define gpu parameters
    block_size = 32
    x_blocks = int(len(d_x1)//block_size + 1)
    n = np.int32(len(d_x1))

    d_f = gpuarray.zeros_like(d_x1, dtype=np.float32)

    # get kernel
    linear_interp = mod_lin.get_function("linear_interp")
    linear_interp(d_f, d_x1, d_x2, d_x, d_f1, d_f2, n, block=(block_size, 1, 1), grid=(x_blocks, 1))

    # free gpu data
    d_x1.gpudata.free()
    d_x2.gpudata.free()
    d_x.gpudata.free()
    d_f1.gpudata.free()
    d_f2.gpudata.free()

    return d_f


def gpu_array_index(d_array, d_return_list, data_type, retain_input=False, retain_list=False):
    """Allows for arbitrary index selecting with numpy arrays

    Parameters
    ----------
    d_array : GPUArray - nD float or int
        Array to be selected from
    d_return_list : GPUArray - 1D int
        list of indexes. That you want to index. If you are indexing more than 1 dimension, then make sure that this array is flattened.
    data_type : dtype
        either int32 or float 32. determines the datatype of the returned array
    retain_input : bool
        If true, the input array is kept in memory, otherwise it is deleted.
    retain_list : bool
        If true, d_return_list is kept in memory, otherwise it is deleted.

    Returns
    -------
    d_return_values : nD array
        Values at the specified indexes.

    """
    mod_array_index = SourceModule("""
    __global__ void array_index_float(float *array, float *return_values, int *return_list, int r_size )
    {
        // 1D grid of 1D blocks
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        if(tid >= r_size){return;}

        return_values[tid] = array[return_list[tid]];
    }

    __global__ void array_index_int(float *array, int *return_values, int *return_list, int r_size )
    {
        // 1D grid of 1D blocks
        int tid = blockIdx.x*blockDim.x + threadIdx.x;

        if(tid >= r_size){return;}

        return_values[tid] = (int)array[return_list[tid]];
    }
    """)

    # GPU will automatically flatten the input array. The indexing must reference the flattened GPU array.
    assert d_return_list.ndim == 1, "Number of dimensions of r_list is wrong. Should be equal to 1"

    # define gpu parameters
    block_size = 32
    r_size = np.int32(d_return_list.size)
    x_blocks = int(r_size // block_size + 1)

    # send data to the gpu
    d_return_values = gpuarray.zeros(d_return_list.size, dtype=data_type)

    if data_type == np.float32:
        # get and launch kernel
        array_index = mod_array_index.get_function("array_index_float")
        array_index(d_array, d_return_values, d_return_list, r_size, block=(block_size, 1, 1), grid=(x_blocks, 1))
    elif data_type == DTYPEi:
        # get and launch kernel
        array_index = mod_array_index.get_function("array_index_int")
        array_index(d_array, d_return_values, d_return_list, r_size, block=(block_size, 1, 1), grid=(x_blocks, 1))
    else:
        raise ValueError("Unrecognized data type for this function. Use float32 or int32.")

    # free GPU data unless specified
    if not retain_input:
        d_array.gpudata.free()
    if not retain_list:
        d_return_list.gpudata.free()

    return d_return_values


def gpu_index_update(d_dest, d_values, d_indices, retain_indices=False):
    """Allows for arbitrary index selecting with numpy arrays

    Parameters
    ----------
    d_dest : GPUArray - nD float
        array to be updated with new values
    d_values : GPUArray - 1D float
        array containing the values to be updated in the destination array
    d_indices : GPUArray - 1D int
        array of indices to update
    retain_indices : bool
        whether to return the indices

    Returns
    -------
    d_dest : nD array
        Input array with values updated

    """
    mod_index_update = SourceModule("""
    __global__ void index_update(float *dest, float *values, int *indices, int r_size)
    {
        // 1D grid of 1D blocks
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        if(tid >= r_size){return;}

        dest[indices[tid]] = values[tid];
    }
    """)
    # define gpu parameters
    block_size = 32
    r_size = np.int32(d_values.size)
    x_blocks = int(r_size//block_size + 1)

    # get and launch kernel
    index_update = mod_index_update.get_function("index_update")
    index_update(d_dest, d_values, d_indices, r_size, block=(block_size, 1, 1), grid=(x_blocks, 1))

    # free gpu data
    d_values.gpudata.free()

    if not retain_indices:
        d_indices.gpudata.free()

    return d_dest


def gpu_gradient(d_strain, d_u, d_v, n_row, n_col, spacing):
    """Computes the strain rate gradients.

    Parameters
    ----------
    d_strain : GPUArray
        2D strain tensor
    d_u, d_v : GPUArray
        velocity
    n_row, n_col : int
        number of rows, columns
    spacing : int
        spacing between nodes

    """
    mod = SourceModule("""
    __global__ void gradient(float *strain, float *u, float *v, float h, int m, int n)
    {
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
        } else if (col == n - 1) {strain[row * n + m - 1] = (u[row * n + m - 1] - u[row * n + m - 2]) / h;  // u_x
        strain[size * 2 + row * n + m - 1] = (v[row * n + m - 1] - v[row * n + m - 2]) / h;  // v_x
        
        // main body
        } else {strain[row * n + col] = (u[row * n + col + 1] - u[row * n + col - 1]) / 2 / h;  // u_x
        strain[size * 2 + row * n + col] = (v[row * n + col + 1] - v[row * n + col - 1]) / 2 / h;}  // v_x
        
        // y-axis
        // first row
        if (row == 0) {strain[size + col] = (u[n + col] - u[col]) / h;  // u_y
        strain[size * 3 + col] = (v[n + col] - v[col]) / h;  // v_y

        // last row
        } else if (row == m - 1) {strain[size + n * (m - 1) + col] = (u[n * (m - 1) + col] - u[n * (m - 2) + col]) / h;  // u_y
        strain[size * 3 + n * (m - 1) + col] = (v[n * (m - 1) + col] - v[n * (m - 2) + col]) / h;  // v_y

        // main body
        } else {strain[size + row * n + col] = (u[(row + 1) * n + col] - u[(row - 1) * n + col]) / 2 / h;  // u_y
        strain[size * 3 + row * n + col] = (v[(row + 1) * n + col] - v[(row - 1) * n + col]) / 2 / h;}  // v_y
    }
    """)

    # CUDA kernel implementation
    block_size = 32
    n_blocks = int((n_row * n_col) // 32 + 1)

    d_gradient = mod.get_function('gradient')
    d_gradient(d_strain, d_u, d_v, np.float32(spacing), np.int32(n_row), np.int32(n_col), block=(block_size, 1, 1), grid=(n_blocks, 1))


def gpu_floor(d_src, retain_input=False):
    """Takes the floor of each element in the gpu array.

    Parameters
    ----------
    d_src : GPUArray
        array to take the floor of
    retain_input : bool
        whether to return the input array

    Returns
    -------
    d_dest : GPUArray
        Same size as d_src. Contains the floored values of d_src.

    """
    assert type(retain_input) == bool, "ReturnInput is {}. Must be of type boolean".format(type(retain_input))

    mod_floor = SourceModule("""
    __global__ void floor_gpu(float *dest, float *src, int n)
    {
        // dest : array to store values
        // src : array of values to be floored

        int tid = blockIdx.x*blockDim.x + threadIdx.x;

        // Avoid the boundary
        if(tid >= n){return;}

        dest[tid] = floorf(src[tid]);
    }
    """)

    # create array to store data
    d_dst = gpuarray.empty_like(d_src.copy())

    # get array size for gpu
    n = np.int32(d_src.size)

    # define gpu parameters
    block_size = 32
    x_blocks = int(n // block_size + 1)

    # get and execute kernel
    floor_gpu = mod_floor.get_function("floor_gpu")
    floor_gpu(d_dst, d_src, n, block=(block_size, 1, 1), grid=(x_blocks, 1))

    # free some gpu memory
    if not retain_input:
        d_src.gpudata.free()

    return d_dst


def gpu_round(d_src, retain_input=False):
    """Rounds of each element in the gpu array.

    Parameters
    ----------
    d_src : GPUArray
        array to round
    retain_input : bool
        whether to return the input array

    Returns
    -------
    d_dest : GPUArray
        Same size as d_src. Contains the floored values of d_src.

    """
    assert type(retain_input) == bool, "ReturnInput is {}. Must be of type boolean".format(type(retain_input))

    mod_round = SourceModule("""
    __global__ void round_gpu(float *dest, float *src, int n)
    {
        // dest : array to store values
        // src : array of values to be floored

        int t_id = blockIdx.x * blockDim.x + threadIdx.x;

        // Avoid the boundary
        if(t_id >= n){return;}

        dest[t_id] = roundf(src[t_id]);
    }
    """)

    # create array to store data
    d_dst = gpuarray.empty_like(d_src)

    # get array size for gpu
    n = np.int32(d_src.size)

    # define gpu parameters
    block_size = 32
    x_blocks = int(n // block_size + 1)

    # get and execute kernel
    round_gpu = mod_round.get_function("round_gpu")
    round_gpu(d_dst, d_src, n, block=(block_size, 1, 1), grid=(x_blocks, 1))

    # free gpu memory
    if not retain_input:
        d_src.gpudata.free()

    return d_dst


def gpu_smooth(d_src, s=0.5, retain_input=False):
    """Smoothes a scalar field stored as a GPUArray.

    Parameters
    ----------
    d_src : GPUArray
        field to be smoothed
    s : int
        smoothing parameter
    retain_input : bool
        whether to keep th input in memory

    Returns
    -------
    d_dst : GPUArray
        smoothed field

    """
    array = d_src.get()
    d_dst = gpuarray.to_gpu(smoothn(array, s=s)[0].astype(np.float32, order='C'))

    # free gpu memory
    if not retain_input:
        d_src.gpudata.free()

    return d_dst
