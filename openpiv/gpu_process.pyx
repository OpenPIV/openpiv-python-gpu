#cython: language_level=3

"""This module is dedicated to advanced algorithms for PIV image analysis with NVIDIA GPU Support."""

import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
import skcuda.cublas as cu_cublas
import skcuda.fft as cu_fft
import skcuda.misc as cu_misc
import numpy as np
import numpy.ma as ma
from numpy.fft import fftshift
import time

cimport numpy as np

DTYPEi = np.int32
ctypedef np.int32_t DTYPEi_t

DTYPEb = np.uint8
ctypedef np.uint8_t DTYPEb_t

#GPU can only hold 32 bit numbers
DTYPEf = np.float32
ctypedef np.float32_t DTYPEf_t


# def gpu_piv(np.ndarray[DTYPEi_t, ndim=2] frame_a,
#             np.ndarray[DTYPEi_t, ndim=2] frame_b,
#             int window_size,
#             int overlap,
#             float dt,
#             int search_area_size,
#             str subpixel_method='gaussian',
#             sig2noise_method=None,
#             int width=2,
#             nfftx=None,
#             nffty=None):
#     """
#     The implementation of the one-step direct correlation with the same size
#     windows. Support for extended search area of the second window has yet to
#     be implimetned. This module is meant to be used with a iterative method to
#     cope with the loss of pairs due to particle movement out of the search area.
#
#     This function is an adaptation of the original extended_search_area_piv
#     function. This has been rewritten with PyCuda and CUDA-C to run on
#     an NVIDIA GPU.
#
#     WARNING FOR DEVELOPERS: Only single precision calculations can be done on the GPU,
#     so all data types must be 32-bit or less.
#
#     See:
#
#     Particle-Imaging Techniques for Experimental Fluid Mechanics
#
#     Annual Review of Fluid Mechanics
#     Vol. 23: 261-304 (Volume publication date January 1991)
#     DOI: 10.1146/annurev.fl.23.010191.001401
#
#     Parameters
#     ----------
#     frame_a : 2d np.ndarray, dtype=np.float32
#         an two dimensions array of integers containing grey levels of
#         the first frame.
#     frame_b : 2d np.ndarray, dtype=np.float32
#         an two dimensions array of integers containing grey levels of
#         the second frame.
#     window_size : int
#         the size of the (square) interrogation window.
#     overlap : int
#         the number of pixels by which two adjacent windows overlap.
#     dt : float
#         the time delay separating the two frames.
#     search_area_size : int
#         the size of the (square) interrogation window from the second frame
#     subpixel_method : string
#          one of the following methods to estimate subpixel location of the peak:
#          'centroid' [replaces default if correlation map is negative],
#          'gaussian' [default if correlation map is positive],
#          'parabolic'.
#     sig2noise_method : string
#         defines the method of signal-to-noise-ratio measure,
#         ('peak2peak' or 'peak2mean'. If None, no measure is performed.)
#     width : int
#         the half size of the region around the first
#         correlation peak to ignore for finding the second
#         peak. [default: 2]. Only used if ``sig2noise_method==peak2peak``.
#     nfftx   : int
#         the size of the 2D FFT in x-direction,
#         [default: 2 x windows_a.shape[0] is recommended]
#     nffty   : int
#         the size of the 2D FFT in y-direction,
#         [default: 2 x windows_a.shape[1] is recommended]
#
#     Returns
#     -------
#     u : 2d np.ndarray
#         a two dimensional array containing the u velocity component,
#         in pixels/seconds.
#     v : 2d np.ndarray
#         a two dimensional array containing the v velocity component,
#         in pixels/seconds.
#     sig2noise : 2d np.ndarray, optional
#         a two dimensional array containing the signal to noise ratio
#         from the cross correlation function. This array is returned if
#         sig2noise_method is not None.
#
#     Examples
#     --------
#     u, v = openpiv.gpu_process.gpu_piv(frame_a, frame_b, window_size=16, overlap=8, search_area_size=48, dt=0.1)
#
#     """
#     # cast images as floats
#     #TODO  changing dtype in the function definition gave weird errors. Find out how to change function definition to avoid this step.
#     cdef np.ndarray[DTYPEf_t, ndim=2] frame_a_f = frame_a.astype(np.float32)
#     cdef np.ndarray[DTYPEf_t, ndim=2] frame_b_f = frame_b.astype(np.float32)
#
#     # Send images to the gpu
#     d_frame_a_f = gpuarray.to_gpu(frame_a_f)
#     d_frame_b_f = gpuarray.to_gpu(frame_b_f)
#
#     # Define variables
#     cdef DTYPEi_t n_rows, n_cols
#
#     assert nfftx == nffty, 'fft x and y dimensions must be same size'
#
#     # Get correlation function
#     c = CorrelationFunction(d_frame_a_f, d_frame_b_f, window_size, overlap, nfftx)
#
#     # Free gpu memory
#     d_frame_a_f.gpudata.free()
#     d_frame_b_f.gpudata.free()
#
#     # vector field shape
#     n_rows, n_cols = c.return_shape()
#
#     # Define arrays
#     cdef np.ndarray[DTYPEf_t, ndim=2] u = np.zeros([n_rows, n_cols], dtype=DTYPEf)
#     cdef np.ndarray[DTYPEf_t, ndim=2] v = np.zeros([n_rows, n_cols], dtype=DTYPEf)
#     cdef np.ndarray[DTYPEf_t, ndim=2] i_peak = np.zeros([n_rows, n_cols], dtype=DTYPEf)
#     cdef np.ndarray[DTYPEf_t, ndim=2] j_peak = np.zeros([n_rows, n_cols], dtype=DTYPEf)
#     cdef np.ndarray[DTYPEf_t, ndim=1] i_tmp = np.zeros(n_rows*n_cols, dtype=DTYPEf)
#     cdef np.ndarray[DTYPEf_t, ndim=1] j_tmp = np.zeros(n_rows*n_cols, dtype=DTYPEf)
#
#     # Get window displacement to subpixel accuracy
#     i_tmp, j_tmp = c.subpixel_peak_location()
#
#     # reshape the peaks
#     i_peak = np.reshape(i_tmp, (n_rows, n_cols))
#     j_peak = np.reshape(j_tmp, (n_rows, n_cols))
#
#     # calculate velocity fields
#     v = -( (i_peak - c.nfft/2) - (search_area_size - window_size)/2)/dt
#     u =  ( (j_peak - c.nfft/2) - (search_area_size - window_size)/2)/dt
#
#     if sig2noise_method is not None:
#         sig2noise = c.sig2noise_ratio(method = sig2noise_method)
#         del c
#         return u,v, sig2noise
#     else:
#         del c
#         return u, v


class CorrelationFunction:
    def __init__(self, d_frame_a, d_frame_b, window_size, overlap, nfftx, d_shift=None):
        """A class representing a cross correlation function.

        NOTE: All identifiers starting with 'd_' exist on the GPU and not the CPU.
        The GPU is referred to as the device, and therefore "d_" signifies that it
        is a device variable. Please adhere to this standard as it makes developing
        and debugging much easier.

        Parameters
        ----------
        d_frame_a, d_frame_b: 2d gpu arrays - float32
            image pair
        window_size: int
            size of the interrogation window
        overlap: int
            pixel overlap between interrogation windows
        nfftx : int
            window size for fft
        d_shift : 2D gpuarray ([dx, dy])
            dx and dy are 1D arrays of the x-y shift at each interrogation window of the second image.
            This is using the x-y convention of this code where x is the row and y is the column.

        """
        ########################################################################################
        # PARAMETERS FOR CORRELATION FUNCTION

        self.shape = d_frame_a.shape
        self.window_size = np.int32(window_size)
        self.overlap = np.int32(overlap)
        self.n_rows, self.n_cols = np.int32(get_field_shape( self.shape, window_size, overlap ))
        self.batch_size = np.int32(self.n_rows*self.n_cols)

        if nfftx is None:
            self.nfft = np.int32(2*self.window_size)
            assert (self.nfft&(self.nfft-1)) == 0, 'nfft must be power of 2'
        else:
            self.nfft = np.int32(nfftx)
            assert (self.nfft&(self.nfft-1)) == 0, 'nfft must be power of 2'

        ########################################################################################

        # START DOING CALCULATIONS

        # Return stack of all IW's
        d_win_a = gpuarray.zeros((self.batch_size, self.window_size, self.window_size), np.float32)
        d_search_area = gpuarray.zeros((self.batch_size, self.window_size, self.window_size), np.float32)
        self._iw_arrange(d_frame_a, d_frame_b, d_win_a, d_search_area, d_shift)

        #normalize array
        d_win_a_norm = gpuarray.zeros((self.batch_size, self.window_size, self.window_size), np.float32)
        d_search_area_norm = gpuarray.zeros((self.batch_size, self.window_size, self.window_size), np.float32)
        self._normalize_intensity(d_win_a, d_search_area, d_win_a_norm, d_search_area_norm)

        # zero pad arrays
        # need this to index arrays apparently...
        d_win_a_zp = gpuarray.zeros([self.batch_size, self.nfft, self.nfft], dtype=np.float32)
        d_search_area_zp = gpuarray.zeros_like(d_win_a_zp)
        #d_win_a_zp[:, :end, :end] = d_win_a_norm.copy()
        #d_search_area_zp[:, :end, :end] = d_search_area_norm.copy()
        self._zero_pad(d_win_a_norm, d_search_area_norm, d_win_a_zp, d_search_area_zp)

        # correlate Windows
        self.data = self._correlate_windows(d_win_a_zp, d_search_area_zp)

        # get first peak of correlation function
        self.row, self.col, self.corr_max1 = self._find_peak(self.data)


    def _iw_arrange(self, d_frame_a, d_frame_b, d_win_a, d_search_area, d_shift):
        """Creates a 3D array stack of all of the interrogation windows.

        This is necessary to do the FFTs all at once on the GPU.

        Parameters
        -----------
        d_frame_a, d_frame_b: 2D numpy arrays - float32
            PIV image pair
        d_win_a: 3D numpy array
             All frame_a interrogation windows stacked on each other
        d_search_area: 3D numpy array
             All frame_b interrogation windows stacked on each other
        d_shift

        """
        #define window slice algorithm
        mod_ws = SourceModule("""
            __global__ void window_slice(float *input, float *output, int window_size, int overlap, int n_col, int w, int batch_size)
        {
            int f_range;
            int w_range;
            int IW_size = window_size*window_size;
            int ind_x = blockIdx.x*blockDim.x + threadIdx.x;
            int ind_y = blockIdx.y*blockDim.y + threadIdx.y;
            int diff = window_size - overlap;

            //loop through each interrogation window

            for(int i=0; i<batch_size; i++)
            {
                //indices of image to map from
                f_range = (i/n_col*diff + ind_y)*w + (i%n_col)*diff + ind_x;

                //indices of new array to map to
                w_range = i*IW_size + window_size*ind_y + ind_x;

                output[w_range] = input[f_range];
            }
        }

            __global__ void window_slice_shift(float *input, float *output, int *dx, int *dy, int window_size, int overlap, int n_col, int w, int h, int batch_size)
        {
            // w = width (number of columns in the full image)
            // h = height (number of rows in the image)
            // batch_size = number of interrogations window pairs

            int f_range;
            int w_range;
            int x_shift;
            int y_shift;

            int IW_size = window_size*window_size;
            int ind_x = blockIdx.x*blockDim.x + threadIdx.x;
            int ind_y = blockIdx.y*blockDim.y + threadIdx.y;
            int diff = window_size - overlap;

            //loop through each interrogation window
            for(int i=0; i<batch_size; i++)
            {
                // y index in whole image for shifted pixel
                y_shift = ind_y + dy[i];

                // x index in whole image for shifted pixel
                x_shift = ind_x + dx[i];

                // Get values outside window in a sneeky way. This array is 1 if the value is inside the window,
                // and 0 if it is outside the window. Multiply This with the shifted value at end
                int outside_range = ( y_shift >= 0 && y_shift < h && x_shift >= 0 && x_shift < w);

                // Get rid of values outside the range
                x_shift = x_shift*outside_range;
                y_shift = y_shift*outside_range;

                //indices of image to map from. Apply shift to pixels
                f_range = (i/n_col*diff + y_shift)*w + (i%n_col)*diff + x_shift;

                // indices of image to map to
                w_range = i*IW_size + window_size*ind_y + ind_x;

                // Apply the mapping. Mulitply by outside_range to set values outside the window to zero!
                output[w_range] = input[f_range]*outside_range;
            }
        }
        """)

        # get field shapes
        h = np.int32(self.shape[0])
        w = np.int32(self.shape[1])

        # transfer data to GPU
        #d_frame_a = gpuarray.to_gpu(frame_a)
        #d_frame_b = gpuarray.to_gpu(frame_b)

        # for debugging
        assert self.window_size >= 8, "Window size is too small"
        assert self.window_size%8 == 0, "Window size should be a multiple of 8"

        # gpu parameters
        # TODO this could be optimized
        grid_size = 8  # I tested a bit and found this number to be fastest.
        block_size = int(self.window_size / grid_size)

        # slice up windows
        window_slice = mod_ws.get_function("window_slice")
        window_slice(d_frame_a, d_win_a, self.window_size, self.overlap, self.n_cols, w, self.batch_size, block=(block_size, block_size, 1), grid=(grid_size, grid_size))

        if d_shift is None:
            window_slice(d_frame_b, d_search_area, self.window_size, self.overlap, self.n_cols, w, self.batch_size, block=(block_size, block_size, 1), grid=(grid_size, grid_size))
        else:
            # Define displacement array for second window
            # GPU thread/block architecture uses column major order, so x is the column and y is the row
            # This code is in row major order
            d_dy = d_shift[0].copy()
            d_dx = d_shift[1].copy()

            window_slice_shift = mod_ws.get_function("window_slice_shift")
            window_slice_shift(d_frame_b, d_search_area, d_dx, d_dy, self.window_size, self.overlap, self.n_cols, w, h, self.batch_size, block=(block_size, block_size, 1), grid=(grid_size, grid_size))

            # free displacement GPU memory
            d_shift.gpudata.free()
            d_dx.gpudata.free()
            d_dy.gpudata.free()

    def _normalize_intensity(self, d_win_a, d_search_area, d_win_a_norm, d_search_area_norm):
        """Remove the mean from each IW of a 3D stack of IW's

        Parameters
        ----------
        d_win_a : 3D gpuarray - float32
            stack of first frame IW's
        d_search_area : 3D gpuarray - float32
            stack of second frame IW's
        d_win_a_norm : 3D gpuarray - float32

        d_search_area_norm : 3D gpuarray - float32


        Returns
        -------
        norm : 3D gpuarray
            stack of IW's with mean removed

        """
        mod_norm = SourceModule("""
            __global__ void normalize(float *array, float *array_norm, float *mean, int iw_size)
        {
            //global thread id for 1D grid of 2D blocks
            int threadId = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

            //indices for mean matrix
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
        d_mean_b = cu_misc.mean(d_search_area.reshape(self.batch_size, iw_size), axis=1)

        # gpu kernel blocksize parameters
        if d_win_a.size%(32 ** 2)==0:
            block_size = 32
        else:
            block_size = 8
        grid_size = int(d_win_a.size / block_size ** 2)

        assert d_win_a.size % (block_size ** 2) == 0, 'Not all windows are being normalized. Something wrong with block or grid size.'

        # get function and norm IW's
        normalize = mod_norm.get_function('normalize')
        normalize(d_win_a, d_win_a_norm, d_mean_a, iw_size, block=(block_size, block_size, 1), grid=(grid_size, 1))
        normalize(d_search_area, d_search_area_norm, d_mean_b, iw_size, block=(block_size, block_size, 1), grid=(grid_size,1))

        # free GPU memory
        d_mean_a.gpudata.free()
        d_mean_b.gpudata.free()
        d_win_a.gpudata.free()
        d_search_area.gpudata.free()


    def _zero_pad(self, d_win_a_norm, d_search_area_norm, d_win_a_zp, d_search_area_zp):
        """Function that zero-pads an 3D stack of arrays for use with the skcuda FFT function.

        Parameters
        ----------
        d_win_a_norm : 3D gpuarray - float32
            array to be zero padded
        d_search_area_norm : 3D gpuarray - float32
            arrays to be zero padded
        d_win_a_zp : 3D gpuarray - float32
            array to be zero padded
        d_search_area_zp : 3D gpuarray - float32
            arrays to be zero padded

        Returns
        -------
        d_winA_zp : 3D gpuarray
            initial array that has been zero padded
        d_search_area_zp : 3D gpuarray
            initial array that has been zero padded

        """
        mod_zp = SourceModule("""
            __global__ void zero_pad(float *array_zp, float *array, int fft_size, int window_size, int batch_size)
            {
                //indices for each IW
                int ind_x = blockIdx.x*blockDim.x + threadIdx.x;
                int ind_y = blockIdx.y*blockDim.y + threadIdx.y;

                //number of pixels in each IW
                int IWsize = fft_size*fft_size;
                int arr_size = window_size*window_size;

                int zp_range;
                int arr_range;
                int i;

                for(i=0; i<batch_size; i++)
                {
                    //get range of values to map
                    arr_range = i*arr_size + window_size*ind_y + ind_x;
                    zp_range = i*IWsize + fft_size*ind_y + ind_x;

                    //apply the map
                    array_zp[zp_range] = array[arr_range];
                }
            }
        """)

        # TODO optimize this
        # gpu parameters
        grid_size = 8
        block_size = int(self.window_size / grid_size)

        # get handle and call function
        zero_pad = mod_zp.get_function('zero_pad')
        zero_pad(d_win_a_zp, d_win_a_norm, self.nfft, self.window_size, self.batch_size, block=(block_size, block_size, 1), grid=(grid_size, grid_size))
        zero_pad(d_search_area_zp, d_search_area_norm, self.nfft, self.window_size, self.batch_size, block=(block_size, block_size, 1), grid=(grid_size, grid_size))

        # Free GPU memory
        d_win_a_norm.gpudata.free()
        d_search_area_norm.gpudata.free()


    def _correlate_windows(self, d_win_a_zp, d_search_area_zp):
        """Compute correlation function between two interrogation windows.

        The correlation function can be computed by using the correlation
        theorem to speed up the computation.

        Parameters
        ----------
        d_win_a_zp :

        d_search_area_zp :

        Returns
        -------
        corr : 2d np.ndarray
            a two dimensions array for the correlation function.

        """
        # FFT size
        win_h = np.int32(self.nfft)
        win_w = np.int32(self.nfft)

        # allocate space on gpu for FFTs
        d_win_i_fft = gpuarray.empty((self.batch_size, win_h, win_w), np.float32)
        d_win_fft = gpuarray.empty((self.batch_size, win_h, win_w//2+1), np.complex64)
        d_search_area_fft = gpuarray.empty((self.batch_size, win_h, win_w//2+1), np.complex64)

        # forward FFTs
        plan_forward = cu_fft.Plan((win_h, win_w), np.float32, np.complex64, self.batch_size)
        cu_fft.fft(d_win_a_zp, d_win_fft, plan_forward)
        cu_fft.fft(d_search_area_zp, d_search_area_fft, plan_forward)

        # multiply the FFTs
        d_win_fft = d_win_fft.conj()
        d_tmp = cu_misc.multiply(d_search_area_fft, d_win_fft)

        #inverse transform
        plan_inverse = cu_fft.Plan((win_h, win_w), np.complex64, np.float32, self.batch_size)
        cu_fft.ifft(d_tmp, d_win_i_fft, plan_inverse, True)

        # transfer back to cpu to do FFTshift
        corr = fftshift(d_win_i_fft.get().real, axes=(1, 2))

        #free gpu memory
        d_win_i_fft.gpudata.free()
        d_win_fft.gpudata.free()
        d_search_area_fft.gpudata.free()
        d_tmp.gpudata.free()
        d_win_a_zp.gpudata.free()
        d_search_area_zp.gpudata.free()

        # delete classes for the plan for free any associated memory
        del(plan_forward, plan_inverse)

        return corr


    def _find_peak(self, array):
        """Find row and column of highest peak in correlation function

        Parameters
        ----------
        array : ndarray
            array of intensities

        Returns
        -------
        ind : 1D array - int
            flattened index of corr peak
        row : 1D array - int
            row position of corr peak
        col : 1D array - int
            column position of corr peak

        """
        # Reshape matrix
        array_reshape = array.reshape(self.batch_size, self.nfft**2)
        s = self.nfft

        # Get index and value of peak
        ind = np.argmax(array_reshape, axis=1)
        maximum = np.amax(array_reshape, axis=1)

        # row and column information of peak
        row = ind // s
        col = ind % s

        return row, col, maximum


    def _find_second_peak(self, width):
        """
        Find the value of the second largest peak.

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

        # TODO When the try statement fails, this can leave lot of points unmasked that
        # should be masked. Must find a better way to do the masking.

        # set (width x width) square submatrix around the first correlation peak as masked
        tmp_len = range(self.batch_size)
        for i in range(-width, width+1):
            for j in range(-width, width+1):
                try:
                    tmp[tmp_len, self.row + i, self.col + j ] = ma.masked
                except IndexError:
                    pass

        row2, col2, corr_max2 = self._find_peak( tmp )

        return corr_max2


    def subpixel_peak_location(self):
        """Find subpixel peak Approximation using Gaussian method

        Inputs
        ------
        corr : 3D numpy array - float
            stack of all correlation functions
        row : 1D numpy array - int
            row location of corr max
        col : 1D numpy array - int
            column location of corr max

        Outputs
        -------
        row_sp : 1D numpy array - float
            row max location to subpixel accuracy
        col_sp : 1D numpy array - float
            column max location to subpixel accuracy

        """
        # Define small number to replace zeros and get rid of warnings in calculations
        cdef DTYPEf_t small = 1e-20

        #cast corr and row as a ctype array
        cdef np.ndarray[DTYPEf_t, ndim=3] corr_c = np.array(self.data, dtype=DTYPEf)
        cdef np.ndarray[DTYPEf_t, ndim=1] row_c = np.array(self.row, dtype=DTYPEf)
        cdef np.ndarray[DTYPEf_t, ndim=1] col_c = np.array(self.col, dtype=DTYPEf)

        # Define arrays to store the data
        cdef np.ndarray[DTYPEf_t, ndim=1] row_sp = np.empty(self.batch_size, dtype=DTYPEf)
        cdef np.ndarray[DTYPEf_t, ndim=1] col_sp = np.empty(self.batch_size, dtype=DTYPEf)

        # TODO figure out what this means
        # Move boundary peaks inward one node. Replace later in sig2noise
        row_tmp = np.copy(self.row)
        row_tmp[row_tmp < 1] = 1
        row_tmp[row_tmp > self.nfft - 2] = self.nfft - 2
        col_tmp = np.copy(self.col)
        col_tmp[col_tmp < 1] = 1
        col_tmp[col_tmp > self.nfft - 2] = self.nfft - 2

        # Initialize arrays
        cdef np.ndarray[DTYPEf_t, ndim=1] c = corr_c[range(self.batch_size), row_tmp, col_tmp]
        cdef np.ndarray[DTYPEf_t, ndim=1] cl = corr_c[range(self.batch_size), row_tmp-1, col_tmp]
        cdef np.ndarray[DTYPEf_t, ndim=1] cr = corr_c[range(self.batch_size), row_tmp+1, col_tmp]
        cdef np.ndarray[DTYPEf_t, ndim=1] cd = corr_c[range(self.batch_size), row_tmp, col_tmp-1]
        cdef np.ndarray[DTYPEf_t, ndim=1] cu = corr_c[range(self.batch_size), row_tmp, col_tmp+1]

        # Get rid of values that are zero or lower
        cdef np.ndarray[DTYPEf_t, ndim=1] non_zero = np.array(c > 0, dtype=DTYPEf)
        c[c <= 0] = small
        cl[cl <= 0] = small
        cr[cr <= 0] = small
        cd[cd <= 0] = small
        cu[cu <= 0] = small

        # Do subpixel approximation. Add small to avoid zero divide.
        row_sp = row_c + ( (np.log(cl)-np.log(cr) )/( 2*np.log(cl) - 4*np.log(c) + 2*np.log(cr) + small ))*non_zero
        col_sp = col_c + ( (np.log(cd)-np.log(cu) )/( 2*np.log(cd) - 4*np.log(c) + 2*np.log(cu) + small))*non_zero

        return row_sp, col_sp


    def sig2noise_ratio(self, method='peak2peak', width=2):
        """Computes the signal to noise ratio.

        The signal to noise ratio is computed from the correlation map with
        one of two available method. It is a measure of the quality of the
        matching between two interrogation windows.

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
        sig2noise = self.corr_max1/corr_max2

        # get rid of nan values. Set sig2noise to zero
        sig2noise[np.isnan(sig2noise)] = 0.0

        # if the image is lacking particles, it will correlate to very low value, but not zero
        # return zero, since we have no signal.
        sig2noise[self.corr_max1 <  1e-3] = 0.0

        # if the first peak is on the borders, the correlation map is wrong
        # return zero, since we have no signal.
        sig2noise[np.array(self.row==0)*np.array(self.row==self.data.shape[1])
                 *np.array(self.col==0)*np.array(self.col==self.data.shape[2])] = 0.0

        return sig2noise.reshape(self.n_rows, self.n_cols)


    def return_shape(self):
        """
        Return row/column information
        """
        return self.n_rows, self.n_cols


def get_field_shape(image_size, window_size, overlap):
    """Compute the shape of the resulting flow field.

    Given the image size, the interrogation window size and
    the overlap size, it is possible to calculate the number
    of rows and columns of the resulting flow field.

    Parameters
    ----------
    image_size: two elements tuple
        a two dimensional tuple for the pixel size of the image
        first element is number of rows, second element is
        the number of columns.
    window_size: int
        the size of the interrogation window.
    overlap: int
        the number of pixel by which two adjacent interrogation
        windows overlap.

    Returns
    -------
    field_shape : two elements tuple
        the shape of the resulting flow field

    """
    return ((image_size[0] - window_size)//(window_size-overlap)+1,
             (image_size[1] - window_size)//(window_size-overlap)+1)

# TODO make mask functional
##################################################################
def WiDIM(np.ndarray[DTYPEi_t, ndim=2] frame_a,
          np.ndarray[DTYPEi_t, ndim=2] frame_b,
          np.ndarray[DTYPEi_t, ndim=2] mark,
          int nb_iter_max=2,
          int nb_refinement_iter=1,
          int min_window_size=16,
          float overlap_ratio=0.5,
          float dt=1,
          int trust_1st_iter=True,
          int nb_validation_iter=1,
          str validation_method='mean_velocity',
          int div_validation=True,
          float sig_tol=1.5,
          float tolerance=1.5,
          float div_tolerance=0.1,
          str subpixel_method='gaussian',
          str sig2noise_method='peak2peak',
          int width=2,
          nfftx=None,
          nffty=None):
    """Implementation of the WiDIM algorithm (Window Displacement Iterative Method).

    This is an iterative  method to cope with  the lost of pairs due to particles
    motion and get rid of the limitation in velocity range due to the window size.
    The possibility of window size coarsening is implemented.
    Example : minimum window size of 16*16 pixels and coarse_level of 2 gives a 1st
    iteration with a window size of 64*64 pixels, then 32*32 then 16*16.
        ----Algorithm : At each step, a predictor of the displacement (dp) is applied based on the results of the previous iteration.
                        Each window is correlated with a shifted window.
                        The displacement obtained from this correlation is the residual displacement (dc)
                        The new displacement (d) is obtained with dx = dpx + dcx and dy = dpy + dcy
                        The velocity field is validated and wrong vectors are replaced by mean value of surrounding vectors from the previous iteration (or by bilinear interpolation if the window size of previous iteration was different)
                        The new predictor is obtained by bilinear interpolation of the displacements of the previous iteration:
                            dpx_k+1 = dx_k

    Reference:
        f. Scarano & M. L. Riethmuller, Iterative multigrid approach in PIV image processing with discrete window offset, Experiments in Fluids 26 (1999) 513-523

    Parameters
    ----------
    frame_a : 2d np.ndarray, dtype=np.float32
        an two dimensions array of integers containing grey levels of
        the first frame.
    frame_b : 2d np.ndarray, dtype=np.float32
        an two dimensions array of integers containing grey levels of
        the second frame.
    mark : 2d np.ndarray, dtype=np.int32
        an two dimensions array of integers with values 0 for the background, 1 for the flow-field. If the center of a window is on a 0 value the velocity is set to 0.
    min_window_size : int
        the size of the minimum (final) (square) interrogation window.
    overlap_ratio : float
        the ratio of overlap between two windows (between 0 and 1).
    nb_refinement_iter : int
        how many times the window size refining processes happens.
    dt : float
        the time delay separating the two frames.
    validation_method : string
        the method used for validation (in addition to the sig2noise method). Only the mean velocity method is implemented now
    trust_1st_iter : bool
        With a first window size following the 1/4 rule, the 1st iteration can be trusted and the value should be 1 (Default value)
    nb_validation_iter : int
        number of iterations per validation cycle.
    div_validation : int
        Boolean - if 1 then the data wil be validated by calculating the divergence. If 0 then it will not be done.
    sig_tol : float
        threshold for signal to noise
    tolerance : float
        the threshold for the validation method chosen. This does not concern the sig2noise for which the threshold is 1.5; [nb: this could change in the future]
    div_tolerance : float
        Threshold value for the maximum divergence at each point. Another validation check to make sure the velocity field is acceptable.
    nb_iter_max : int
        global number of iterations.
    subpixel_method : string
         one of the following methods to estimate subpixel location of the peak:
         'centroid' [replaces default if correlation map is negative],
         'gaussian' [default if correlation map is positive],
         'parabolic'.
    sig2noise_method : string
        defines the method of signal-to-noise-ratio measure,
        ('peak2peak' or 'peak2mean'. If None, no measure is performed.)
    width : int
        the half size of the region around the first
        correlation peak to ignore for finding the second
        peak. [default: 2]. Only used if ``sig2noise_method==peak2peak``.
    nfftx : int
        the size of the 2D FFT in x-direction,
        [default: 2 x windows_a.shape[0] is recommended]
    nffty : int
        the size of the 2D FFT in y-direction,
        [default: 2 x windows_a.shape[1] is recommended]

    Returns
    -------
    x : 2d np.ndarray
        a two dimensional array containing the x-axis component of the interpolations locations.
    y : 2d np.ndarray
        a two dimensional array containing the y-axis component of the interpolations locations.
    u : 2d np.ndarray
        a two dimensional array containing the u velocity component,
        in pixels/seconds.
    v : 2d np.ndarray
        a two dimensional array containing the v velocity component,
        in pixels/seconds.
    mask : 2d np.ndarray
        a two dimensional array containing the boolean values (True for vectors interpolated from previous iteration)

    Example
    -------
    x, y, u, v, mask = openpiv.gpu_process.WiDIM(frame_a, frame_b, mark, min_window_size=16, overlap_ratio=0.25, coarse_factor=2, dt=0.02, validation_method='mean_velocity', trust_1st_iter=1, validation_iter=2, tolerance=0.7, nb_iter_max=4, sig2noise_method='peak2peak')

    --------------------------------------
    Method of implementation : to improve the speed of the program, all data have been placed in the same huge
    4-dimensions 'f' array. (this prevent the definition of a new array for each iteration) However, during the coarsening process a large part of the array is not used.
    Structure of array f:
    --The 1st index is the main iteration (K)   --> length is nb_iter_max
        -- 2nd index (I) is row (of the map of the interpolations locations of iteration K) --> length (effectively used) is n_row[K]
            --3rd index (J) is column  --> length (effectively used) is n_col[K]
                --4th index represent the type of data stored at this point:
                            | 0 --> x         |
                            | 1 --> y         |
                            | 2 --> xb        |
                            | 3 --> yb        |
                            | 4 --> dx        |
                            | 5 --> dy        |
                            | 6 --> dpx       |
                            | 7 --> dpy       |
                            | 8 --> dcx       |
                            | 9 --> dcy       |
                            | 10 --> u        |
                            | 11 --> v        |
                            | 12 --> sig2noise|
    Storage of data with indices is not good for comprehension so its very important to comment on each single operation.
    A python dictionary type could have been used (and would be much more intuitive)
    but its equivalent in c language (type map) is very slow compared to a numpy ndarray.

    """
    ####################################################
    # INITIALIZATIONS
    ####################################################

    # Initialize skcuda miscellaneous library
    cu_misc.init()

    # cast images as floats
    #TODO changing dtype in the function definition gave weird errors. Find out how to change function definition to avoid this step.
    cdef np.ndarray[DTYPEf_t, ndim=2] frame_a_f = frame_a.astype(np.float32)
    cdef np.ndarray[DTYPEf_t, ndim=2] frame_b_f = frame_b.astype(np.float32)

    # Send images to the gpu
    d_frame_a_f = gpuarray.to_gpu(frame_a_f)
    d_frame_b_f = gpuarray.to_gpu(frame_b_f)

    # warnings.warn("deprecated", RuntimeWarning)
    if nb_iter_max <= nb_refinement_iter:
        raise ValueError( "Please provide a nb_iter_max that is greater than the coarse_level" )
    cdef int K  # main iteration index
    cdef int I, J  # interrogation locations indices
    cdef int L, M  # inside window indices
    cdef int O, P  # frame indices corresponding to I and J
    cdef int i, j  # dumb indices for various works
    cdef float mean_u, mean_v, rms_u, rms_v, residual_0, div
    cdef int residual, nb_w_ind
    cdef np.ndarray[DTYPEi_t, ndim=1] n_row = np.zeros(nb_iter_max, dtype=DTYPEi)
    cdef np.ndarray[DTYPEi_t, ndim=1] n_col = np.zeros(nb_iter_max, dtype=DTYPEi)
    cdef np.ndarray[DTYPEi_t, ndim=1] w = np.zeros(nb_iter_max, dtype=DTYPEi)
    cdef np.ndarray[DTYPEi_t, ndim=1] overlap = np.zeros(nb_iter_max, dtype=DTYPEi)
    pic_size = frame_a.shape

    # window sizes list initialization
    for K in range(nb_refinement_iter + 1):
        w[K] = np.power(2, nb_refinement_iter - K) * min_window_size
    for K in range(nb_refinement_iter + 1, nb_iter_max):
        w[K] = w[K-1]

    # overlap init
    for K in range(nb_iter_max):
        overlap[K]=int(np.floor(overlap_ratio*w[K]))

    # n_col and n_row init
    for K in range(nb_iter_max):
        n_row[K]=((pic_size[0]-w[K])//(w[K]-overlap[K]))+1
        n_col[K]=((pic_size[1]-w[K])//(w[K]-overlap[K]))+1

    # write the parameters to the screen
    if nb_validation_iter==0:
        validation_method='None'

    cdef float start_time = launch(method='WiDIM', names=['Size of image', 'total number of iterations', 'overlap ratio', 'coarse factor', 'time step', 'validation method', 'number of validation iterations', 'subpixel_method','n_row', 'n_col', 'Window sizes', 'overlaps'], arg=[[pic_size[0], pic_size[1]], nb_iter_max, overlap_ratio, nb_refinement_iter, dt, validation_method, nb_validation_iter, subpixel_method, n_row, n_col, w, overlap])

    # define the main array f that contains all the data
    cdef np.ndarray[DTYPEf_t, ndim=4] f = np.zeros([nb_iter_max, n_row[nb_iter_max-1], n_col[nb_iter_max-1], 13], dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=4] f_check = np.zeros([nb_iter_max, n_row[nb_iter_max-1], n_col[nb_iter_max-1], 13], dtype=DTYPEf)

    # define mask - bool arrays don't exist in cython so we go to lower level with cast
    # you can access mask with (<object>mask)[I,J]
    cdef np.ndarray[DTYPEb_t, ndim=2, cast=True] mask = np.empty([n_row[nb_iter_max-1], n_col[nb_iter_max-1]], dtype=np.bool)  # changed type

    # define u, v & x, y fields (only used as outputs of this program)
    cdef np.ndarray[DTYPEf_t, ndim=2] u = np.zeros([n_row[nb_iter_max-1], n_col[nb_iter_max-1]], dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=2] v = np.zeros([n_row[nb_iter_max-1], n_col[nb_iter_max-1]], dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=2] x = np.zeros([n_row[nb_iter_max-1], n_col[nb_iter_max-1]], dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=2] y = np.zeros([n_row[nb_iter_max-1], n_col[nb_iter_max-1]], dtype=DTYPEf)

    # define temporary arrays and reshaped arrays to store the correlation function output
    cdef np.ndarray[DTYPEf_t, ndim=1] i_tmp = np.zeros(n_row[-1]*n_col[-1], dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=1] j_tmp = np.zeros(n_row[-1]*n_col[-1], dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=2] i_peak = np.zeros([n_row[nb_iter_max-1], n_col[nb_iter_max-1]], dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=2] j_peak = np.zeros([n_row[nb_iter_max-1], n_col[nb_iter_max-1]], dtype=DTYPEf)

    # define array for signal to noise ratio
    cdef np.ndarray[DTYPEf_t, ndim=2] sig2noise = np.zeros([n_row[-1], n_col[-1]], dtype=DTYPEf)

    # define arrays used for the validation process
    # in validation list, a 1 means that the location does not need to be validated. A 0 means that it does need to be validated
    cdef np.ndarray[DTYPEi_t, ndim=2] validation_list = np.ones([n_row[-1], n_col[-1]], dtype=DTYPEi)
    # cdef np.ndarray[DTYPEf_t, ndim=3] u_mean = np.zeros([nb_iter_max, n_row[-1], n_col[-1]], dtype=DTYPEf)
    # cdef np.ndarray[DTYPEf_t, ndim=3] v_mean = np.zeros([nb_iter_max, n_row[-1], n_col[-1]], dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=3] neighbours = np.zeros([2,3,3], dtype=DTYPEf)
    cdef np.ndarray[DTYPEi_t, ndim=2] neighbours_present = np.zeros([3,3], dtype=DTYPEi)

    #### GPU arrays###

    # define arrays to stores the displacement vector in to save displacement information
    d_shift = gpuarray.zeros([2, n_row[-1], n_col[-1]], dtype=DTYPEi)

    # define arrays to store all the mean velocity at each point in each iteration
    d_u_mean = gpuarray.zeros([nb_iter_max, n_row[-1], n_col[-1]], dtype=DTYPEf)
    d_v_mean = gpuarray.zeros([nb_iter_max, n_row[-1], n_col[-1]], dtype=DTYPEf)

    # initialize x and y values
    for K in range(nb_iter_max):
        for I in range(n_row[K]):
            for J in range(n_col[K]):
                # x unit vector corresponds to rows
                # y unit vector corresponds to columns
                if I == 0:
                    f[K,I,J,0] = w[K] / 2 #init x on 1st row
                else:
                    f[K,I,J,0] = f[K,I-1,J,0] + w[K] - overlap[K] #init x
                if J == 0:
                    f[K,I,J,1] = w[K] / 2 #init y on first column
                else:
                    f[K,I,J,1] = f[K,I,J-1,1] + w[K] - overlap[K] #init y

    # Move f to the GPU for the whole calculation
    d_f = gpuarray.to_gpu(f)

    # end of the initializations

    ####################################################
    # MAIN LOOP
    ####################################################
    for K in range(nb_iter_max):
        # print(" ")
        print("//////////////////////////////////////////////////////////////////")
        # print(" ")
        print("ITERATION # {}".format(K))
        # print(" ")

        residual = 0

        #################################################################################
        #  GPU VERSION
        #################################################################################

        # Calculate second frame displacement (shift)
        d_shift[0, :n_row[K], :n_col[K]] = d_f[K, 0:n_row[K], 0:n_col[K], 6].copy().astype(np.int32)  # xb=xa+dpx
        d_shift[1, :n_row[K], :n_col[K]] = d_f[K, 0:n_row[K], 0:n_col[K], 7].copy().astype(np.int32)  # yb=ya+dpy

        # Get correlation function
        c = CorrelationFunction(d_frame_a_f, d_frame_b_f, w[K], overlap[K], nfftx, d_shift=d_shift[:, :n_row[K], :n_col[K]].copy())

        # Get window displacement to subpixel accuracy
        i_tmp[0:n_row[K] * n_col[K]], j_tmp[0:n_row[K] * n_col[K]] = c.subpixel_peak_location()

        # reshape the peaks
        i_peak[0:n_row[K], 0: n_col[K]] = np.reshape(i_tmp[0:n_row[K]*n_col[K]], (n_row[K], n_col[K]))
        j_peak[0:n_row[K], 0: n_col[K]] = np.reshape(j_tmp[0:n_row[K]*n_col[K]], (n_row[K], n_col[K]))

        # Get signal to noise ratio
        sig2noise[0:n_row[K], 0:n_col[K]] = c.sig2noise_ratio(method=sig2noise_method)

        # update the field with new values
        # TODO check for nans in i_peak and j_peak
        gpu_update(d_f, sig2noise[0:n_row[K], 0:n_col[K]], i_peak[0:n_row[K], 0:n_col[K]], j_peak[0:n_row[K], 0:n_col[K]], n_row[K], n_col[K], c.nfft, dt, K)

        #################################################################################

        print("...[DONE]")
        print(" ")
        """
        if K==0:
            residual_0 = residual/np.float(n_row[K]*n_col[K])
            print(residual_0)
        print(" --residual : ", (residual/np.float(n_row[K]*n_col[K]))/residual_0)
        """

        #####################################################
        # validation of the velocity vectors with 3*3 filtering
        #####################################################

        if K == 0 and trust_1st_iter:#1st iteration can generally be trusted if it follows the 1/4 rule
            print("No validation: trusting 1st iteration")
        elif nb_validation_iter > 0:
            print("Starting validation...")

            # init mask to False
            for I in range(n_row[nb_iter_max-1]):
                for J in range(n_col[nb_iter_max-1]):
                    (<object>mask)[I,J] = False

            # real validation starts
            for i in range(nb_validation_iter):
                print("Validation iteration {}.".format(i))
                # print(" ")

                # reset validation list
                validation_list = np.ones([n_row[-1], n_col[-1]], dtype=DTYPEi)

                # get list of places that need to be validated
                validation_list[0:n_row[K], 0:n_col[K]], d_u_mean[K, 0:n_row[K], 0:n_col[K]], d_v_mean[K, 0:n_row[K], 0:n_col[K]] = gpu_validation(d_f, K, sig2noise[0:n_row[K], 0:n_col[K]], n_row[K], n_col[K], w[K], sig_tol, tolerance, div_tolerance)

                # do the validation
                print('Validating {} out of {} vectors...'.format(n_row[-1] * n_col[-1] - np.sum(validation_list), n_row[K] * n_col[K]))
                initiate_validation(d_f, validation_list, d_u_mean, d_v_mean, nb_iter_max, K, n_row, n_col, w, overlap, dt)

            print("...[DONE]")
            print(" ")
        # end of validation

        ##############################################################################
        # stop process if this is the last iteration
        ##############################################################################

        if K == nb_iter_max - 1:
            print("//////////////////////////////////////////////////////////////////")
            print("End of iterative process. Re-arranging vector fields...")

            f = d_f.get()
            d_f.gpudata.free()

            # assemble the u, v and x, y fields for outputs
            x = f[K, :, :, 1]
            y = f[K, ::-1, :, 0]
            u = f[K, :, :, 10]
            v = f[K, :, :, 11]

            print("...[DONE]")
            print(" ")

            # delete images from gpu memory
            d_frame_a_f.gpudata.free()
            d_frame_b_f.gpudata.free()

            # delete old correlation function
            del c, d_f
            # end(start_time)
            return x, y, u, v, (<object>mask)
        #############################################################################

        # go to next iteration: compute the predictors dpx and dpy from the current displacements
        print("Going to next iteration...")
        print("Performing interpolation of the displacement field for next iteration predictors")
        # print(" ")

        if n_row[K+1] == n_row[K] and n_col[K+1] == n_col[K]:
             d_f[K+1, :n_row[K+1], :n_col[K+1], 6] = gpu_round(d_f[K, :n_row[K], :n_col[K], 4].copy()) #dpx_k+1=dx_k
             d_f[K+1, :n_row[K+1], :n_col[K+1], 7] = gpu_round(d_f[K, :n_row[K], :n_col[K], 5].copy()) #dpy_k+1=dy_k
        # interpolate if dimensions do not agree
        else:
            v_list = np.ones((n_row[-1], n_col[-1]), dtype=bool)
            #interpolate velocity onto next iterations grid. Then take the floor as the predictor for the next step
            gpu_interpolate_surroundings(d_f, v_list, n_row, n_col, w, overlap, K, 4)
            gpu_interpolate_surroundings(d_f, v_list, n_row, n_col, w, overlap, K, 5)
            d_f[K + 1, :, :, 6] = gpu_round(d_f[K + 1, :, :, 4].copy())
            d_f[K + 1, :, :, 7] = gpu_round(d_f[K + 1, :, :, 5].copy())

        # delete old correlation function
        del c

        print("...[DONE] -----> going to iteration ", K+1)
        print(" ")


def initiate_validation(d_f, validation_list, d_u_mean, d_v_mean, nb_iter_max, k, n_row, n_col, w, overlap, dt):
    """Initiate the full GPU version of the validation and interpolation.

    Parameters
    ----------
    d_f : 4D gpuarray - float
        main array that stores all velocity data
    validation_list : 2D array - int
        indicates which values must be validate. 1 indicates no validation needed, 0 indicated validation is needed
    d_u_mean, d_v_mean : 3D gpuarray - float
        mean velocity surrounding each point
    nb_iter_max : int
        total number of iterations
    k : int
        main loop iteration count
    n_row, n_col : 1D array
        number of rows an columns in each main loop iteration
    w : int
        pixels between interrogation windows
    overlap : float
        ratio of overlap between interrogation windows
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
        indices = np.where(validation_location.flatten() == 1)[0].astype(np.int32)
        d_indices = gpuarray.to_gpu(indices)

        # get mean velocity at validation points
        d_u_tmp, d_indices = gpu_array_index(d_u_mean[k, :, :].copy(), d_indices, np.float32, return_list=True)
        d_v_tmp, d_indices = gpu_array_index(d_v_mean[k, :, :].copy(), d_indices, np.float32, return_list=True)

        # update the velocity values
        d_f[k, :, :, 10], d_indices = gpu_index_update(d_f[k, :, :, 10].copy(), d_u_tmp, d_indices, return_indices=True)
        d_f[k, :, :, 11] = gpu_index_update(d_f[k, :, :, 11].copy(), d_v_tmp, d_indices)

        #TODO, you don't need to do all these calculations. Could write a function that only does it for the ones that have been validated
        d_f[k, :, :, 4] = - d_f[k, :, :, 11].copy() * dt
        d_f[k, :, :, 5] = d_f[k, :, :, 10].copy() * dt

    # case if different dimensions: interpolation using previous iteration
    elif k > 0 and (n_row[k] != n_row[k - 1] or n_col[k] != n_col[k - 1]):
        gpu_interpolate_surroundings(d_f, validation_location, n_row, n_col, w, overlap, k - 1, 10)
        gpu_interpolate_surroundings(d_f, validation_location, n_row, n_col, w, overlap, k - 1, 11)
        d_f[k, :, :, 4] = - d_f[k, :, :, 11].copy() * dt
        d_f[k, :, :, 5] = d_f[k, :, :, 10].copy() * dt

    # case if same dimensions
    elif k > 0 and (n_row[k] == n_row[k - 1] or n_col[k] == n_col[k - 1]):
        # get indices and send them to the gpu
        indices = np.where(validation_location.flatten() == 1)[0].astype(np.int32)
        d_indices = gpuarray.to_gpu(indices)

        # update the velocity values with the previous values.
        # This is essentially a bilinear interpolation when the value is right on top of the other.
        #TODO - could replace with the mean of the previous values surrounding the point
        d_u_tmp, d_indices = gpu_array_index(d_f[k - 1, :, :, 10].copy(), d_indices, np.float32, return_list=True)  # changed by Eric
        d_v_tmp, d_indices = gpu_array_index(d_f[k - 1, :, :, 11].copy(), d_indices, np.float32, return_list=True)  # changed by Eric

        d_f[k, :, :, 10], d_indices = gpu_index_update(d_f[k, :, :, 10].copy(), d_u_tmp, d_indices, return_indices=True)  # changed by Eric
        d_f[k, :, :, 11] = gpu_index_update(d_f[k, :, :, 11].copy(), d_v_tmp, d_indices)  # changed by Eric

        # d_F[K,:,:,10], d_indices = gpu_index_update(d_F[K,:,:,10].copy(), d_F[K-1,:,:,10].copy(), d_indices, ReturnIndices=True)  # original
        # d_F[K,:,:,11] = gpu_index_update(d_F[K,:,:,11].copy(), d_F[K-1,:,:,11].copy(), d_indices)  # original
        d_f[k, :, :, 4] = - d_f[k, :, :, 11].copy() * dt
        d_f[k, :, :, 5] = d_f[k, :, :, 10].copy() * dt


def gpu_interpolate_surroundings(d_f, v_list, n_row, n_col, w, overlap, k, dat):
    """Interpolate a point based on the surroundings.

    Parameters
    ----------
    d_f : 4D gpuarray - float
        main array that stores all velocity data
    v_list : 2D array - bool
        indicates which values must be validated. True means it needs to be validated, False means no validation is needed.
    n_row, n_col : 1D array
        Number rows and columns in each iteration
    w :

    overlap :

    k : int
        current iteration
    dat : int
        data that needs to be interpolated. 4th index in the F array

    Returns
    -------
    d_F : 4D gpuarray - float
        This must always be returned so the class handle is not lost

    """
    #### Separate validation list into multiple lists for each region ####

    # set all sides to false for interior points
    interior_list = np.copy(v_list[:n_row[k + 1], :n_col[k + 1]]).astype('bool')
    interior_list[0,:] = 0
    interior_list[-1,:] = 0
    interior_list[:,0] = 0
    interior_list[:,-1] = 0

    # define array with the indices of the points to be validated
    interior_ind = np.where(interior_list.flatten() == True)[0].astype(np.int32)
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
    top_ind = np.where(top_list.flatten() == True)[0].astype(np.int32)
    if top_ind.size != 0:
        d_top_ind = gpuarray.to_gpu(top_ind)

    bottom_list = np.copy(v_list[n_row[k + 1] - 1, :n_col[k + 1]])
    bottom_list[0] = 0
    bottom_list[-1] = 0
    bottom_ind = np.where(bottom_list.flatten() == True)[0].astype(np.int32)
    if bottom_ind.size != 0:
        d_bottom_ind = gpuarray.to_gpu(bottom_ind)

    left_list = np.copy(v_list[:n_row[k + 1], 0])
    left_list[0] = 0
    left_list[-1] = 0
    left_ind = np.where(left_list.flatten() == True)[0].astype(np.int32)
    if left_ind.size != 0:
        d_left_ind = gpuarray.to_gpu(left_ind)

    right_list = np.copy(v_list[:n_row[k + 1], n_col[k + 1] - 1])
    right_list[0] = 0
    right_list[-1] = 0
    right_ind = np.where(right_list.flatten() == True)[0].astype(np.int32)
    if right_ind.size != 0:
        d_right_ind = gpuarray.to_gpu(right_ind)

    drv.Context.synchronize()

    #--------------------------INTERIOR GRID---------------------------------

    if interior_ind.size != 0:

        # get gpu data for position now
        d_low_x, d_high_x = f_dichotomy_gpu(d_f[k:k + 2, :, 0, 0].copy(), k, "x_axis", d_interior_ind_x, w, overlap, n_row, n_col)
        d_low_y, d_high_y = f_dichotomy_gpu(d_f[k:k + 2, 0, :, 1].copy(), k, "y_axis", d_interior_ind_y, w, overlap, n_row, n_col)

        # get indices surrounding the position now
        d_x1, d_low_x = gpu_array_index(d_f[k, :n_row[k], 0, 0].copy(), d_low_x, np.float32, return_list=True)
        d_x2, d_high_x = gpu_array_index(d_f[k, :n_row[k], 0, 0].copy(), d_high_x, np.float32, return_list=True)
        d_y1, d_low_y = gpu_array_index(d_f[k, 0, :n_col[k], 1].copy(), d_low_y, np.float32, return_list=True)
        d_y2, d_high_y = gpu_array_index(d_f[k, 0, :n_col[k], 1].copy(), d_high_y, np.float32, return_list=True)
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
        d_y1, d_low_y = gpu_array_index(d_f[k, 0, :, 1].copy(), d_low_y, np.float32, return_list=True)
        d_y2, d_high_y = gpu_array_index(d_f[k, 0, :, 1].copy(), d_high_y, np.float32, return_list=True)
        d_y, d_top_ind = gpu_array_index(d_f[k + 1, 0, :, 1].copy(), d_top_ind, np.float32, return_list=True)

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
    # if(False):
    if bottom_ind.size > 0:

        # get position data
        d_low_y, d_high_y = f_dichotomy_gpu(d_f[k:k + 2, 0, :, 1].copy(), k, "y_axis", d_bottom_ind, w, overlap, n_row, n_col)

        # Get values to compute interpolation
        d_y1, d_low_y = gpu_array_index(d_f[k, int(n_row[k] - 1), :, 1].copy(), d_low_y, np.float32, return_list=True)
        d_y2, d_high_y = gpu_array_index(d_f[k, int(n_row[k] - 1), :, 1].copy(), d_high_y, np.float32, return_list=True)
        d_y, d_bottom_ind = gpu_array_index(d_f[k + 1, int(n_row[k + 1] - 1), :, 1].copy(), d_bottom_ind, np.float32, return_list=True)

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
    # if(False):
    if left_ind.size > 0:

        # get position data
        d_low_x, d_high_x = f_dichotomy_gpu(d_f[k:k + 2, :, 0, 0].copy(), k, "x_axis", d_left_ind, w, overlap, n_row, n_col)

        # Get values to compute interpolation
        d_x1, d_low_x = gpu_array_index(d_f[k, :, 0, 0].copy(), d_low_x, np.float32, return_list=True)
        d_x2, d_high_x = gpu_array_index(d_f[k, :, 0, 0].copy(), d_high_x, np.float32, return_list=True)
        d_x, d_left_ind = gpu_array_index(d_f[k + 1, :, 0, 0].copy(), d_left_ind, np.float32, return_list=True)

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
    # if(False):
    if right_ind.size > 0:

        # get position data
        d_low_x, d_high_x = f_dichotomy_gpu(d_f[k:k + 2, :, 0, 0].copy(), k, "x_axis", d_right_ind, w, overlap, n_row, n_col)

        # Get values to compute interpolation
        d_x1, d_low_x = gpu_array_index(d_f[k, :, int(n_col[k] - 1), 0].copy(), d_low_x, np.float32, return_list=True)
        d_x2, d_high_x = gpu_array_index(d_f[k, :, int(n_col[k] - 1), 0].copy(), d_high_x, np.float32, return_list=True)
        d_x, d_right_ind = gpu_array_index(d_f[k + 1, :, int(n_col[k + 1] - 1), 0].copy(), d_right_ind, np.float32, return_list=True)

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
    #top left
    if v_list[0, 0] == 1:
        d_f[k + 1, 0, 0, dat] = d_f[k, 0, 0, dat]
    #top right
    if v_list[0, n_col[k + 1] - 1] == 1:
        d_f[k + 1, 0, int(n_col[k + 1] - 1), dat] = d_f[k, 0, int(n_col[k] - 1), dat]
    #bottom left
    if v_list[n_row[k + 1] - 1, 0] == 1:
        d_f[k + 1, int(n_row[k + 1] - 1), 0, dat] = d_f[k, int(n_row[k] - 1), 0, dat]
    #bottom right
    if v_list[n_row[k + 1] - 1, n_col[k + 1] - 1] == 1:
        d_f[k + 1, int(n_row[k + 1] - 1), int(n_col[k + 1] - 1), dat] = d_f[k, int(n_row[k] - 1), int(n_col[k] - 1), dat]


def launch(str method, names, arg):
    """A nice launcher for any OpenPIV function, printing a header in terminal with a list of the parameters used.

    Parameters
    ----------
    method : string
        the name of the algorithm used
    names : list of string
        names of the parameters to print
    arg : list
        parameters of different types

    Returns
    -------
    start_time : float
        the current time --> can be used to print the execution time of the programm at the end.

    """
    cdef int i
    print(" ")
    print('----------------------------------------------------------')
    print('|----->     ||   The Open Source  P article              |')
    print('| Open      ||                    I mage                 |')
    print('|     PIV   ||                    V elocimetry  Toolbox  |')
    print('|     <-----||   www.openpiv.net          version 1.0    |')
    print('----------------------------------------------------------')
    print(" ")
    print("Algorithm : ", method)
    print(" ")
    print("Parameters   ")
    print("-----------------------------------")
    for i in range(len(arg)-1):
        print("     ", names[i], " | ", arg[i])
    print(" ")
    print("-----------------------------------")
    print("|           STARTING              |")
    print("-----------------------------------")
    print(" ")
    cdef float start_time = time.time()
    return start_time


def end(float start_time):
    """A function that prints the time since startTime. Used to nicely end the program

    Parameters
    ----------
    start_time : float
        a time

    """
    print("-------------------------------------------------------------")
    print("[DONE] ..after ", (time.time() - start_time), "seconds ")
    print("-------------------------------------------------------------")


################################################################################
#  CUDA GPU FUNCTIONS
################################################################################
def gpu_update(d_f, sig2noise, i_peak, j_peak, n_row, n_col, nfft, dt, k):
    """Function to update the velocity values after an iteration in the WiDIM algorithm

    Parameters
    ---------
    d_f : np.ndarray - 4D - float
        main array in WiDIM algorithm
    sig2noise : 3D array
        signal to noise ratio at each IW at each iteration
    i_peak, j_peak : 2D array - float
        correlation function peak at each iteration
    n_row, n_col : int
        number of rows and columns in the current iteration
    nfft : int
        size of the fft window
    dt : float
        time between images
    k : int
        main loop iteration

    """
    mod_update = SourceModule("""

        __global__ void update_values(float *F, float *i_peak, float *j_peak, float *sig2noise, int fourth_dim, int nfft, float dt)
        {
            // F is where all the data is stored at a particular K
            // i_peak / j_peak is the correlation peak location
            // sig2noise = sig2noise ratio from correlation function
            // cols = number of colums of IW's
            // fourth_dim  = size of the fourth dimension of F
            // nfft = size of the fft window
            // dt = time step between frames
            // leap = 'leaps' to where the F iteration starts

            int w_idx = blockIdx.x*blockDim.x + threadIdx.x;

            //Index for each IW in the F array
            int F_idx = w_idx*fourth_dim;

            F[F_idx + 2] = F[F_idx + 0] + F[F_idx + 6];
            F[F_idx + 3] = F[F_idx + 1] + F[F_idx + 7];

            F[F_idx + 8] = i_peak[w_idx] - nfft / 2;
            F[F_idx + 9] = j_peak[w_idx] - nfft / 2;

            //get new displacement prediction
            F[F_idx + 4] = F[F_idx + 6] + F[F_idx + 8];
            F[F_idx + 5] = F[F_idx + 7] + F[F_idx + 9];

            //get new velocity vectors
            F[F_idx + 10] = F[F_idx + 5] / dt;
            F[F_idx + 11] = -F[F_idx + 4] / dt;

            // get sig2noise ratio
            F[F_idx + 12] = sig2noise[w_idx];
        }
        """)

    # make all arrays the proper data type
    i_peak = i_peak.astype(np.float32)
    j_peak = j_peak.astype(np.float32)
    print(i_peak)
    print(j_peak)
    sig2noise = sig2noise.astype(np.float32)
    nfft = np.int32(nfft)
    dt = np.float32(dt)

    # GPU parameters
    n_col = np.int32(n_col)
    n_row = np.int32(n_row)
    # block_size = 8
    block_size = 32
    x_blocks = int(n_col * n_row // block_size + 1)

    # move data to gpu
    d_i_peak = gpuarray.to_gpu(i_peak)
    d_j_peak = gpuarray.to_gpu(j_peak)
    d_sig2noise = gpuarray.to_gpu(sig2noise)
    d_f_tmp = d_f[k, 0:n_row, 0:n_col, :].copy()

    # last dimension of F
    fourth_dim = np.int32(d_f.shape[-1])

    update_values = mod_update.get_function("update_values")
    update_values(d_f_tmp, d_i_peak, d_j_peak, d_sig2noise, fourth_dim, nfft, dt, block=(block_size, 1, 1), grid=(x_blocks, 1))
    d_f[k, 0:n_row, 0:n_col, :] = d_f_tmp

    #Free gpu memory
    d_i_peak.gpudata.free()
    d_j_peak.gpudata.free()
    d_sig2noise.gpudata.free()
    d_f_tmp.gpudata.free()


def gpu_validation(d_f, k, sig2noise, n_row, n_col, w, s2n_tol, mean_tol, div_tol):
    """Returns an array indicating which indices need to be validated.

    Parameters
    ----------
    d_f : 4D gpuarray - float
        main loop array
    k : int
        iteration number
    sig2noise: 2D array - float
        signal to noise ratio of each velocity
    n_row, n_col : int
        number of rows and columns in the velocity field
    w : float
        number of pixels between each interrogation window center
    s2n_tol : float
        minimum value for sig2noise
    mean_tol : float
        tolerance for mean velocity validation
    div_tol : float
        tolerance for divergence validation

    Returns
    -------
    val_list : 2D array - int
        list of indices that need to be validated. 0 indicates that the index needs to be corrected. 1 means no correction is needed
    d_u_mean, d_v_mean : 2D gpuarray
        mean of the velocities surrounding each point in this iteration.

    """
    # GPU functions
    mod_validation = SourceModule("""
    __global__ void s2n(int *val_list, float *sig2noise, float s2n_tol, int Nrow, int Ncol)
    {
        //val_list : list of indices to be validated
        //sig2noise : signal to noise ratio
        // s2n_tol : min sig2noise value
        // Ncol : number of columns in the

        int w_idx = blockIdx.x*blockDim.x + threadIdx.x;

        if(w_idx >= Ncol*Nrow){return;}

        val_list[w_idx] = val_list[w_idx] * (sig2noise[w_idx] > s2n_tol);
    }


    __global__ void mean_validation(int *val_list, float *u_rms, float *v_rms, float *u_mean, float *v_mean, float *u, float *v, int Nrow, int Ncol, float tol)
    {
        // val_list: list of locations where validation is needed
        // rms_u : rms u velocity of neighbours
        // rms_v : rms v velocity of neighbours
        // mean_u: mean u velocity of neigbours
        // mean_v: mean v velocity of neighbours
        // u: u velocity at that point
        // v: v velocity at that point
        // Nrow, Ncol: number of rows and columns
        // tol : validation tolerance. usually 1.5

        int w_idx = blockIdx.x*blockDim.x + threadIdx.x;

        if(w_idx >= Nrow*Ncol){return;}

        int u_validation = ((u[w_idx] - u_mean[w_idx])/u_rms[w_idx] < tol);
        int v_validation = ((v[w_idx] - v_mean[w_idx])/v_rms[w_idx] < tol);

        val_list[w_idx] = val_list[w_idx] * u_validation * v_validation;

    }

    __global__ void div_validation(int *val_list, float *div,  int Nrow, int Ncol, float div_tol)
    {
        // u: u velocity
        // v: v velocity
        // w: window size
        // Nrow, Ncol: number of rows and columns

        int w_idx = blockIdx.x*blockDim.x + threadIdx.x;

        if(w_idx >= Nrow*Ncol){return;}

        val_list[w_idx] = val_list[w_idx] * (fabsf(div[w_idx]) < div_tol);
    }

    """)

    # create array to store validation list
    val_list = np.ones_like(sig2noise, dtype=np.int32)
    d_val_list = gpuarray.to_gpu(val_list)

    # cast inputs to appropriate data types
    sig2noise = sig2noise.astype(np.float32)
    s2n_tol = np.float32(s2n_tol)
    mean_tol = np.float32(mean_tol)
    div_tol = np.float32(div_tol)
    n_row = np.int32(n_row)
    n_col = np.int32(n_col)
    w = np.float32(w)

    # TODO delete these checks
    # assert sig2noise.dtype == np.float32, "dtype of sig2noise is {}. Should be np.float32".format(sig2noise.dtype)
    # assert type(s2n_tol) == np.float32, "type of s2n_tol is {}. Should be np.float32".format(type(s2n_tol))
    # assert type(n_row) == np.int32, "dtype of Nrow is {}. Should be np.int32".format(type(n_row))
    # assert type(n_col) == np.int32, "dtype of Ncol is {}. Should be np.int32".format(type(n_col))
    # assert type(w) == np.float32, "dtype of w is {}. Should be np.float32" .format(type(w))
    # assert d_f.dtype == np.float32, "dtype of d_F is {}. dtype should be np.float32".format(d_f.dtype)

    # GPU settings
    # block_size = 16
    block_size = 32
    x_blocks = int(n_col * n_row / block_size + 1)

    # send velocity field to GPU
    d_u = d_f[k, 0:n_row, 0:n_col, 10].copy()
    d_v = d_f[k, 0:n_row, 0:n_col, 11].copy()

    # get neighbours information
    d_neighbours, d_neighbours_present, d_u, d_v = gpu_get_neighbours(d_u, d_v, n_row, n_col)

    ##########################
    # sig2noise validation
    ##########################

    # # move data to the gpu
    d_sig2noise = 0
    # d_sig2noise = gpuarray.to_gpu(sig2noise)
    #
    # # Launch signal to noise kernel and free sig2noise data
    # s2n = mod_validation.get_function("s2n")
    # s2n(d_val_list, d_sig2noise, s2n_tol, n_row, n_col, block=(block_size, 1, 1), grid=(x_blocks, 1))
    # d_sig2noise.gpudata.free()

    ##########################
    # mean_velocity validation
    ##########################

    # get rms data and mean velocity data.
    d_u_rms, d_v_rms = gpu_rms(d_neighbours, d_neighbours_present, n_row, n_col)
    d_u_mean, d_v_mean = gpu_mean_vel(d_neighbours, d_neighbours_present, n_row, n_col)

    # get and launch rms
    mean_validation = mod_validation.get_function("mean_validation")
    a = d_val_list.get()
    print(a.size)
    d_val_list = gpuarray.to_gpu(a)
    mean_validation(d_val_list, d_u_rms, d_v_rms, d_u_mean, d_v_mean, d_u, d_v, n_row, n_col, mean_tol,
                    block=(block_size, 1, 1), grid=(x_blocks, 1))

    ##########################
    # divergence validation
    ##########################

    d_div = 0
    # d_div, d_u, d_v = gpu_divergence(d_u, d_v, w, n_row, n_col)
    #
    # # launch divergence validation kernel
    # div_validation = mod_validation.get_function("div_validation")
    # div_validation(d_val_list, d_div, n_row, n_col, div_tol, block=(block_size, 1, 1), grid=(x_blocks, 1))

    # return the final validation list
    val_list = d_val_list.get()
    # u_mean = d_u_mean.get()
    # v_mean = d_v_mean.get()

    # Free gpu memory
    d_val_list.gpudata.free()
    d_neighbours_present.gpudata.free()
    d_neighbours.gpudata.free()
    d_u.gpudata.free()
    d_v.gpudata.free()
    # d_u_mean.gpudata.free()
    # d_v_mean.gpudata.free()
    d_u_rms.gpudata.free()
    d_v_rms.gpudata.free()
    # d_div.gpudata.free()

    del d_val_list, d_sig2noise, d_neighbours, d_neighbours_present, d_u, d_v, d_u_rms, d_v_rms, d_div

    return val_list, d_u_mean, d_v_mean


def gpu_find_neighbours(n_row, n_col):
    """An array that stores if a point has neighbours in a 3x3 grid surrounding it

    Parameters
    ----------
    n_row : 1D array - int
        number of rows at each iteration
    n_col : 1D array - int
        number of columns at each iteration

    Returns
    -------
    d_neighbours_present : 4D gpuarray [n_row, n_col, 3 , 3]

    """
    mod_neighbours = SourceModule("""
    __global__ void find_neighbours(int *neighbours_present, int Nrow, int Ncol)
    {
        // neighbours_present = boolean array
        // Nrow = number of rows
        // Ncol = Number of columns

        // references each IW
        int w_idx = blockIdx.x*blockDim.x + threadIdx.x;

        int row_zero = (w_idx >= Ncol);
        int row_max = (w_idx < Ncol * (Nrow - 1));
        int col_zero = (w_idx % Ncol != 0);
        int col_max = (w_idx % Ncol != Ncol - 1);

        // Top Row
        neighbours_present[w_idx * 9 + 0] = neighbours_present[w_idx * 9 + 0] * row_zero;
        neighbours_present[w_idx * 9 + 1] = neighbours_present[w_idx * 9 + 1] * row_zero;
        neighbours_present[w_idx * 9 + 2] = neighbours_present[w_idx * 9 + 2] * row_zero;

        __syncthreads();

        // Bottom row
        neighbours_present[w_idx * 9 + 6] = neighbours_present[w_idx * 9 + 6] * row_max;
        neighbours_present[w_idx * 9 + 7] = neighbours_present[w_idx * 9 + 7] * row_max;
        neighbours_present[w_idx * 9 + 8] = neighbours_present[w_idx * 9 + 8] * row_max;

        __syncthreads();

        // Left column
        neighbours_present[w_idx * 9 + 0] = neighbours_present[w_idx * 9 + 0] * col_zero;
        neighbours_present[w_idx * 9 + 3] = neighbours_present[w_idx * 9 + 3] * col_zero;
        neighbours_present[w_idx * 9 + 6] = neighbours_present[w_idx * 9 + 6] * col_zero;

        __syncthreads();

        // right column
        neighbours_present[w_idx * 9 + 2] = neighbours_present[w_idx * 9 + 2] * col_max;
        neighbours_present[w_idx * 9 + 5] = neighbours_present[w_idx * 9 + 5] * col_max;
        neighbours_present[w_idx * 9 + 8] = neighbours_present[w_idx * 9 + 8] * col_max;

        // Set center to zero--can't be a neighbour for yourself
        neighbours_present[w_idx*9 + 4] = 0;
    }
    """)

    # GPU settings
    # block_size = 8
    block_size = 32
    x_blocks = int(n_col * n_row // block_size + 1)
    n_row = np.int32(n_row)
    n_col = np.int32(n_col)

    # allocate space for new array
    neighbours_present = np.ones([n_row, n_col, 3, 3], dtype=np.int32)

    assert neighbours_present.dtype == np.int32, "Wrong data type for neighbours present"

    # send data to gpu
    d_neighbours_present = gpuarray.to_gpu(neighbours_present)

    # get and launch kernel
    find_neighbours = mod_neighbours.get_function("find_neighbours")
    find_neighbours(d_neighbours_present, n_row, n_col, block=(block_size, 1, 1), grid=(x_blocks, 1))

    return d_neighbours_present


def gpu_get_neighbours(d_u, d_v, n_row, n_col):
    """An array that stores the values of the velocity of the neighbours around it.

    Parameters
    ----------
    d_u, d_v : 2D GPU array - float32
        u and v velocity
    n_row : 1D array - int
        number of rows at each iteration
    n_col : 1D array - int
        number of columns at each iteration

    Returns
    -------
    neighbours : 5D array [n_row, n_col, 2, 3, 3]
        stores the values of u and v of the neighbours of a point

    """
    # TODO delete this redundant code
    # mod_get_neighbours = SourceModule("""
    # __global__ void get_u_neighbours(float *neighbours, float *neighbours_present, float *u, int Nrow, int Ncol)
    # {
    #     // neighbours - u and v values around each point
    #     // neighbours_present - 1 if there is a neighbour, 0 if no neighbour
    #     // u, v - u and v velocities
    #     // Nrow, Ncol - number of rows and columns
    #
    #     // references each IW
    #     int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
    #     int max_idx = Nrow * Ncol;
    #
    #     if(w_idx >= max_idx){return;}
    #
    #     // get velocities
    #     neighbours[w_idx * 18 + 0] = u[max(w_idx - Ncol - 1, 0)] * neighbours_present[w_idx * 9 + 0];
    #     neighbours[w_idx * 18 + 1] = u[max(w_idx - Ncol, 0)] * neighbours_present[w_idx * 9 + 1];
    #     neighbours[w_idx * 18 + 2] = u[max(w_idx - Ncol + 1, 0)] * neighbours_present[w_idx * 9 + 2];
    #
    #     __syncthreads();
    #
    #     neighbours[w_idx * 18 + 3] = u[max(w_idx - 1, 0)] * neighbours_present[w_idx * 9 + 3];
    #     neighbours[w_idx * 18 + 4] = 0.0;
    #     neighbours[w_idx * 18 + 5] = u[min(w_idx + 1, max_idx)] * neighbours_present[w_idx * 9 + 5];
    #
    #     __syncthreads();
    #
    #     neighbours[w_idx * 18 + 6] = u[min(w_idx + Ncol - 1, max_idx)] * neighbours_present[w_idx * 9 + 6];
    #     neighbours[w_idx * 18 + 7] = u[min(w_idx + Ncol, max_idx)] * neighbours_present[w_idx * 9 + 7];
    #     neighbours[w_idx * 18 + 8] = u[min(w_idx + Ncol + 1, max_idx)] * neighbours_present[w_idx * 9 + 8];
    #
    #     __syncthreads();
    # }
    #
    # __global__ void get_v_neighbours(float *neighbours, float *neighbours_present, float *v, int Nrow, int Ncol)
    # {
    #     // neighbours - u and v values around each point
    #     // neighbours_present - 1 if there is a neighbour, 0 if no neighbour
    #     // u, v - u and v velocities
    #     // Nrow, Ncol - number of rows and columns
    #
    #     // references each IW
    #     int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
    #     int max_idx = Nrow * Ncol;
    #
    #     if(w_idx >= max_idx){return;}
    #
    #     // get velocities
    #     neighbours[w_idx * 18 + 9] = v[max(w_idx - Ncol - 1, 0)] * neighbours_present[w_idx * 9 + 0];
    #     neighbours[w_idx * 18 + 10] = v[max(w_idx - Ncol, 0)] * neighbours_present[w_idx * 9 + 1];
    #     neighbours[w_idx * 18 + 11] = v[max(w_idx - Ncol + 1, 0)] * neighbours_present[w_idx * 9 + 2];
    #
    #     __syncthreads();
    #
    #     neighbours[w_idx * 18 + 12] = v[max(w_idx - 1, 0)] * neighbours_present[w_idx * 9 + 3];
    #     neighbours[w_idx * 18 + 13] = 0.0;
    #     neighbours[w_idx * 18 + 14] = v[min(w_idx + 1, max_idx)] * neighbours_present[w_idx * 9 + 5];
    #
    #     __syncthreads();
    #
    #     neighbours[w_idx * 18 + 15] = v[min(w_idx + Ncol - 1, max_idx)] * neighbours_present[w_idx * 9 + 6];
    #     neighbours[w_idx * 18 + 16] = v[min(w_idx + Ncol, max_idx)] * neighbours_present[w_idx * 9 + 7];
    #     neighbours[w_idx * 18 + 17] = v[min(w_idx + Ncol + 1, max_idx)] * neighbours_present[w_idx * 9 + 8];
    #
    #     __syncthreads();
    # }
    # """)

    mod_get_neighbours = SourceModule("""
    __global__ void get_u_neighbours(float *neighbours, float *neighbours_present, float *u, int Nrow, int Ncol)
    {
        // neighbours - u and v values around each point
        // neighbours_present - 1 if there is a neighbour, 0 if no neighbour
        // u, v - u and v velocities
        // Nrow, Ncol - number of rows and columns

        // references each IW
        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
        int max_idx = Nrow * Ncol;

        if(w_idx >= max_idx){return;}

        // get velocities
        if(neighbours_present[w_idx * 9 + 0] == 1){neighbours[w_idx * 18 + 0] = u[w_idx - Ncol - 1];}
        if(neighbours_present[w_idx * 9 + 1] == 1){neighbours[w_idx * 18 + 1] = u[w_idx - Ncol];}
        if(neighbours_present[w_idx * 9 + 2] == 1){neighbours[w_idx * 18 + 2] = u[w_idx - Ncol + 1];}

        __syncthreads();

        if(neighbours_present[w_idx * 9 + 3] == 1){neighbours[w_idx * 18 + 3] = u[w_idx - 1];}
        //neighbours[w_idx * 18 + 4] = 0.0;
        if(neighbours_present[w_idx * 9 + 5] == 1){neighbours[w_idx * 18 + 5] = u[w_idx + 1];}

        __syncthreads();

        if(neighbours_present[w_idx * 9 + 6] == 1){neighbours[w_idx * 18 + 6] = u[w_idx + Ncol - 1];}
        if(neighbours_present[w_idx * 9 + 7] == 1){neighbours[w_idx * 18 + 7] = u[w_idx + Ncol];}
        if(neighbours_present[w_idx * 9 + 8] == 1){neighbours[w_idx * 18 + 8] = u[w_idx + Ncol + 1];}

        __syncthreads();
    }

    __global__ void get_v_neighbours(float *neighbours, float *neighbours_present, float *v, int Nrow, int Ncol)
    {
        // neighbours - u and v values around each point
        // neighbours_present - 1 if there is a neighbour, 0 if no neighbour
        // u, v - u and v velocities
        // Nrow, Ncol - number of rows and columns

        // references each IW
        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
        int max_idx = Nrow * Ncol;

        if(w_idx >= max_idx){return;}

        // get velocities
        if(neighbours_present[w_idx * 9 + 0] == 1){neighbours[w_idx * 18 + 9] = v[w_idx - Ncol - 1];}
        if(neighbours_present[w_idx * 9 + 1] == 1){neighbours[w_idx * 18 + 10] = v[w_idx - Ncol];}
        if(neighbours_present[w_idx * 9 + 2] == 1){neighbours[w_idx * 18 + 11] = v[w_idx - Ncol + 1];}

        __syncthreads();

        if(neighbours_present[w_idx * 9 + 3] == 1){neighbours[w_idx * 18 + 12] = v[w_idx - 1];}
        //neighbours[w_idx * 18 + 13] = 0.0;
        if(neighbours_present[w_idx * 9 + 5] == 1){neighbours[w_idx * 18 + 14] = v[w_idx + 1];}

        __syncthreads();

        if(neighbours_present[w_idx * 9 + 6] == 1){neighbours[w_idx * 18 + 15] = v[w_idx + Ncol - 1];}
        if(neighbours_present[w_idx * 9 + 7] == 1){neighbours[w_idx * 18 + 16] = v[w_idx + Ncol];}
        if(neighbours_present[w_idx * 9 + 8] == 1){neighbours[w_idx * 18 + 17] = v[w_idx + Ncol + 1];}

        __syncthreads();
    }
    """)

    # set dtype of inputs
    n_row = np.int32(n_row)
    n_col = np.int32(n_col)

    # Get GPU grid dimensions and function
    # block_size = 16
    block_size = 32
    x_blocks = int(n_col * n_row // block_size + 1)
    get_u_neighbours = mod_get_neighbours.get_function("get_u_neighbours")
    get_v_neighbours = mod_get_neighbours.get_function("get_v_neighbours")

    # find neighbours
    d_neighbours_present = gpu_find_neighbours(n_row, n_col)  # .astype(np.float32)  # og
    neighbours = np.zeros((n_row, n_col, 2, 3, 3))
    neighbours = neighbours.astype(np.float32)

    # TODO delete this check
    # # assert statements for data
    # assert neighbours.dtype == np.float32, "Wrong data type for neighbours"
    # assert type(n_row) == np.int32, "Wrong data type for Nrow"
    # assert type(n_col) == np.int32, "Wrong data type for Ncol"

    # send data to the gpu
    d_neighbours = gpuarray.to_gpu(neighbours)

    # Get u and v data
    get_u_neighbours(d_neighbours, d_neighbours_present, d_u, n_row, n_col, block=(block_size, 1, 1),
                     grid=(x_blocks, 1))
    get_v_neighbours(d_neighbours, d_neighbours_present, d_v, n_row, n_col, block=(block_size, 1, 1),
                     grid=(x_blocks, 1))

    # TODO delete this check for NaNs
    # return data
    neighbours = d_neighbours.get()
    a = np.isnan(neighbours)
    assert not a.any(), 'NaNs detected in neighbours'
    # if np.sum(a) > 0:
    #     neighbours[a] = 0.0
    #
    d_neighbours = gpuarray.to_gpu(neighbours)

    return d_neighbours, d_neighbours_present, d_u, d_v


def gpu_mean_vel(d_neighbours, d_neighbours_present, n_row, n_col):
    """Calculates the mean velocity in a 3x3 grid around each point in a velocity field.

    Parameters
    ----------
    d_neighbours: 5D gpuarray - float32
        all the neighbouring velocities of every point
    d_neighbours_present: 4D gpuarray - float32
        indicates if a neighbour is present
    n_row, n_col : int
        number of rows and columns of the velocity field

    Returns
    -------
    u_mean, v_mean : 2D array - float32
        mean velocities at each point

    """
    mod_mean_vel = SourceModule("""
    __global__ void u_mean_vel(float *u_mean, float *n, float *np, int Nrow, int Ncol)
    {
        // mean_u : mean velocity of surrounding points
        // n : velocity of neighbours
        // np : neighbours present
        // Nrow, Ncol: number of rows and columns

        int w_idx = blockIdx.x*blockDim.x + threadIdx.x;

        if(w_idx >= Ncol*Nrow){return;}

        float numerator_u = n[w_idx*18] + n[w_idx*18+1] + n[w_idx*18+2] + n[w_idx*18+3] + n[w_idx*18+5] + n[w_idx*18+6] + n[w_idx*18+7] + n[w_idx*18+8];
        float denominator = np[w_idx*9] + np[w_idx*9+1] + np[w_idx*9+2] + np[w_idx*9+3] + np[w_idx*9+5] + np[w_idx*9+6] + np[w_idx*9+7] + np[w_idx*9+8];

        __syncthreads();

        u_mean[w_idx] = numerator_u / denominator;
    }

    __global__ void v_mean_vel(float *v_mean, float *n, float *np, int Nrow, int Ncol)
    {

        int w_idx = blockIdx.x*blockDim.x + threadIdx.x;

        if(w_idx >= Ncol*Nrow){return;}

        float numerator_v = n[w_idx*18+9] + n[w_idx*18+10] + n[w_idx*18+11] + n[w_idx*18+12] + n[w_idx*18+14] + n[w_idx*18+15] + n[w_idx*18+16] + n[w_idx*18+17];
        float denominator = np[w_idx*9] + np[w_idx*9+1] + np[w_idx*9+2] + np[w_idx*9+3] + np[w_idx*9+5] + np[w_idx*9+6] + np[w_idx*9+7] + np[w_idx*9+8];

        __syncthreads();

        v_mean[w_idx] = numerator_v / denominator;
    }
    """)

    # allocate space for arrays
    u_mean = np.empty((n_row, n_col), dtype=np.float32)
    v_mean = np.empty_like(u_mean)
    n_row = np.int32(n_row)
    n_col = np.int32(n_col)

    # define GPU data
    # block_size = 16
    block_size = 32
    x_blocks = int(n_row * n_col // block_size + 1)

    # assert u_mean.dtype == np.float32, "dtype for u_mean is wrong. Should be np.float32"
    # assert v_mean.dtype == np.float32, "dtype for v_mean is wrong. Should be np.float32"

    # send data to gpu
    d_u_mean = gpuarray.to_gpu(u_mean)
    d_v_mean = gpuarray.to_gpu(v_mean)

    # get and launch kernel
    u_mean_vel = mod_mean_vel.get_function("u_mean_vel")
    v_mean_vel = mod_mean_vel.get_function("v_mean_vel")
    u_mean_vel(d_u_mean, d_neighbours, d_neighbours_present, n_row, n_col, block=(block_size, 1, 1), grid=(x_blocks, 1))
    v_mean_vel(d_v_mean, d_neighbours, d_neighbours_present, n_row, n_col, block=(block_size, 1, 1), grid=(x_blocks, 1))

    return d_u_mean, d_v_mean


def gpu_rms(d_neighbours, d_neighbours_present, n_row, n_col):
    """Calculates the mean velocity in a 3x3 grid around each point in a velocity field.

    Parameters
    ----------
    d_neighbours : 5D gpuarray - float32
        all the neighbouring velocities of every point
    d_neighbours_present : 4D gpuarray - float32
        indicates if a neighbour is present
    n_row, n_col : int
        number of rows and columns of the velocity field

    Returns
    -------
    d_u_rms, d_v_rms : 2D gpuarray - float32
        mean velocities at each point

    """
    mod_rms = SourceModule("""
    __global__ void u_rms_k(float *u_rms, float *n, float *np, int Nrow, int Ncol)
    {

        // Ncol : number of columns in the

        int w_idx = blockIdx.x*blockDim.x + threadIdx.x;

        if(w_idx >= Ncol*Nrow){return;}

        float numerator = (powf(n[w_idx*18+0], 2) + powf(n[w_idx*18+1], 2) + powf(n[w_idx*18+2], 2) + \
                           powf(n[w_idx*18+3], 2) + powf(n[w_idx*18+5], 2) + powf(n[w_idx*18+6], 2) + \
                           powf(n[w_idx*18+7], 2) + powf(n[w_idx*18+8], 2) );
        float denominator = np[w_idx*9] + np[w_idx*9+1] + np[w_idx*9+2] + np[w_idx*9+3] + np[w_idx*9+5] + np[w_idx*9+6] + np[w_idx*9+7] + np[w_idx*9+8];

        __syncthreads();

        u_rms[w_idx] =  sqrtf(numerator / denominator);
    }

    __global__ void v_rms_k(float *v_rms, float *n,float *np, int Nrow, int Ncol)
    {

        // Ncol : number of columns in the

        int w_idx = blockIdx.x*blockDim.x + threadIdx.x;

        if(w_idx >= Ncol*Nrow){return;}

        float numerator = (powf(n[w_idx*18+9], 2) + powf(n[w_idx*18+10], 2) + powf(n[w_idx*18+11], 2) + \
                           powf(n[w_idx*18+12], 2) + powf(n[w_idx*18+14], 2) + powf(n[w_idx*18+15], 2) + \
                           powf(n[w_idx*18+16], 2) + powf(n[w_idx*18+17], 2) );
        float denominator = np[w_idx*9] + np[w_idx*9+1] + np[w_idx*9+2] + np[w_idx*9+3] + np[w_idx*9+5] + np[w_idx*9+6] + np[w_idx*9+7] + np[w_idx*9+8];

        __syncthreads();

        v_rms[w_idx] = sqrtf(numerator / denominator);
    }
    """)

    # allocate space for data
    u_rms = np.empty((n_row, n_col), dtype=np.float32)
    v_rms = np.empty((n_row, n_col), dtype=np.float32)
    n_row = np.int32(n_row)
    n_col = np.int32(n_col)

    # define GPU data
    # block_size = 16
    block_size = 32
    x_blocks = int(n_row * n_col // block_size + 1)

    # TODO delete this check
    # assert u_rms.dtype == np.float32, "dtype for u_rms is wrong. Should be np.float32"
    # assert v_rms.dtype == np.float32, "dtype for v_rms is wrong. Should be np.float32"

    # send data to gpu
    d_u_rms = gpuarray.to_gpu(u_rms)
    d_v_rms = gpuarray.to_gpu(v_rms)

    # get and launch kernel
    mod_u_rms = mod_rms.get_function("u_rms_k")
    mod_v_rms = mod_rms.get_function("v_rms_k")
    mod_u_rms(d_u_rms, d_neighbours, d_neighbours_present, n_row, n_col, block=(block_size, 1, 1), grid=(x_blocks, 1))
    mod_v_rms(d_v_rms, d_neighbours, d_neighbours_present, n_row, n_col, block=(block_size, 1, 1), grid=(x_blocks, 1))

    return d_u_rms, d_v_rms


# TODO check to ensure this function does what it should
def gpu_divergence(d_u, d_v, w, n_row, n_col):
    """Calculates the divergence at each point in a velocity field.

    Parameters
    ----------
    d_u, d_v: 2D array - float
        velocity field
    w: int
        pixel separation between velocity vectors
    n_row, n_col : int
        number of rows and columns of the velocity field

    Returns
    -------
    div : 2D array - float32
        divergence at each point

    """
    mod_div = SourceModule("""
    __global__ void div_k(float *div, float *u, float *v, float w, int Nrow, int Ncol)
    {
        // u : u velocity
        // v : v velocity
        // w : window size
        // Nrow, Ncol : number of rows and columns

        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
        int max_idx = Nrow * Ncol;

        // Avoid the boundary
        if(w_idx >= (Nrow - 1) * Ncol){return;}
        if(w_idx%Ncol == Ncol - 1){return;}

        float u1 = u[w_idx + Ncol];
        float v1 = v[w_idx + 1];

        __syncthreads();

        div[w_idx] = (u1 - u[w_idx]) / w - (v1 - v[w_idx]) / w;
    }

    __global__ void div_boundary_k(float *div, float *u, float *v, float w, int Nrow, int Ncol)
    {
        // u : u velocity
        // v : v velocity
        // w : window size
        // Nrow, Ncol : number of rows and columns

        int w_idx = blockIdx.x * blockDim.x + threadIdx.x;

        // only calculate on the boundary
        if(w_idx < (Nrow - 1) * Ncol && w_idx%Ncol != Ncol - 1){return;}

        float u1 = u[w_idx - Ncol];
        float v1 = v[w_idx - 1];

        __syncthreads();

        div[w_idx] = (u[w_idx] - u1) / w - (v[w_idx] - v1) / w;
    }
    """)

    div = np.empty((n_row, n_col), dtype=np.float32)
    n_row = np.int32(n_row)
    n_col = np.int32(n_col)
    w = np.float32(w)

    # define GPU data
    # block_size = 16
    block_size = 32
    x_blocks = int(n_row * n_col // block_size + 1)

    # TODO delete this check
    # assert div.dtype == np.float32, "dtype of div is {}. Should be np.float32".format(div.dtype)

    # move data to gpu
    d_div = gpuarray.to_gpu(div)

    # get and launch kernel
    div_k = mod_div.get_function("div_k")
    div_boundary_k = mod_div.get_function("div_boundary_k")
    div_k(d_div, d_u, d_v, w, n_row, n_col, block=(block_size, 1, 1), grid=(x_blocks, 1))
    div_boundary_k(d_div, d_u, d_v, w, n_row, n_col, block=(block_size, 1, 1), grid=(x_blocks, 1))

    # get single case of bottom i = 0, j = Ncol-1
    d_div[0, int(n_col - 1)] = (d_u[1, n_col - 1] - d_u[0, n_col - 1]) / w - (d_v[0, n_col - 1] - d_v[0, n_col - 2]) / w
    d_div[int(n_row - 1), 0] = (d_u[n_row - 1, 0] - d_u[n_row - 2, 0]) / w - (d_v[n_row - 1, 1] - d_v[n_row - 1, 0]) / w

    return d_div, d_u, d_v


def f_dichotomy_gpu(d_range, k, side, d_pos_index, w, overlap, n_row, n_col):
    """
    Look for the position of the vectors at the previous iteration that surround the current point in the frame
    you want to validate. Returns the low and high index of the points from the previous iteration on either side of
    the point in the current iteration that needs to be validated.

    Parameters
    ----------
    d_range : 2D gpuarray
        The x or y locations along the grid for the current and next iteration.
        Example:
        For side = x_axis then the input looks like d_range = d_F[K:K+2, :,0,0].copy()
        For side = y_axis then the input looks like d_range = d_F[K:K+2, 0,:,1].copy()
    k : int
        the iteration you want to use to validate. Typically the previous iteration from the
        one that the code is in now. (1st index for F).
    side : string
        the axis of interest : can be either 'x_axis' or 'y_axis'
    d_pos_index : 1D gpuarray - int
        index of the point in the frame you want to validate (along the axis 'side').
    w : 1D array - int
        array of window sizes
    overlap : 1D array - int
        overlap in number of pixels
    n_row, n_col : 1D array
        number of rows and columns in the F dataset in each iteration

    Returns
    -------
    d_F : 4D gpuarray
        Must return the class handle
    d_low : 1D gpuarray - int
        largest index at the iteration K along the 'side' axis so that the position of index low in the frame is less than or equal to pos_now.
    d_high : 1D gpuarray - int
        smallest index at the iteration K along the 'side' axis so that the position of index low in the frame is greater than or equal to pos_now.
    d_pos_index : 1D gpuarray - int
        Must return class handle

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

    # TODO delete this check
    # assert n_row.dtype == np.int32, "Nrow data type is {}. Should be int32".format(n_row.dtype)
    # assert n_col.dtype == np.int32, "Ncol data type is {}. Should be int32".format(n_col.dtype)
    # assert d_range.dtype == np.float32, "d_range data type is {}. Should be np.float32".format(d_range.dtype)
    # assert d_pos_index.dtype == np.int32, "d_pos_index data type is {}. Should be np.int32".format(d_pos_index.dtype)

    # define gpu settings
    # block_size = 8
    block_size = 32
    x_blocks = int(len(d_pos_index)//block_size + 1)

    # create GPU data
    d_low = gpuarray.zeros_like(d_pos_index, dtype=np.int32)
    d_high = gpuarray.zeros_like(d_pos_index, dtype=np.int32)

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
    del d_range

    return d_low, d_high


def bilinear_interp_gpu(d_x1, d_x2, d_y1, d_y2, d_x, d_y, d_f1, d_f2, d_f3, d_f4):
    """
    """
    mod_bi = SourceModule("""
    __global__ void bilinear_interp(float *f, float *x1, float *x2, float *y1, float *y2, float *x, float *y, float *f1, float *f2, float *f3, float *f4, int n)
    {
        // 1D grid of 1D blocks
        int idx = blockIdx.x*blockDim.x + threadIdx.x;

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
    x_blocks = int(len(d_x1)//block_size + 1)
    n = np.int32(len(d_x1))

    d_f = gpuarray.zeros_like(d_x1, dtype=np.float32)

    # get kernel
    bilinear_interp = mod_bi.get_function("bilinear_interp")
    bilinear_interp(d_f, d_x1, d_x2, d_y1, d_y2, d_x, d_y, d_f1, d_f2, d_f3, d_f4, n, block=(block_size, 1, 1), grid=(x_blocks, 1))

    #free gpu data
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
    # block_size = 8
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


def gpu_array_index(d_array, d_return_list, data_type, return_input=False, return_list=False):
    """Allows for arbitrary index selecting with numpy arrays

    Parameters
    ----------
    d_array: nD gpuarray - float or int
        Array to be selected from
    d_return_list: 1 array - int
        list of indexes. That you want to index. If you are indexing more than 1 dimension, then make sure that this array is flattened.
    data_type: dtype
        either int32 or float 32. determines the datatype of the returned array
    return_input: bool
        If true, returns the input array as well so the class handle is preserved.
    return_list: bool
        If true, d_return_list is also returned to preserve the class handle.

    Returns
    -------
    d_return_values : nD array
        Values at the specified indexes.
    d_array : ND gpuarray
        preserves the class handle
    d_return_list : nD gpuarray
        This is not normally returned unless input ReturnList = True

    """
    mod_array_index = SourceModule("""
    __global__ void array_index_float(float *array, float *return_values, int *return_list, int r_size )
    {
        // 1D grid of 1D blocks
        int tid = blockIdx.x*blockDim.x + threadIdx.x;

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
    # block_size = 8
    block_size = 32
    r_size = np.int32(d_return_list.size)
    x_blocks = int(r_size//block_size + 1)

    # send data to the gpu
    d_return_values = gpuarray.zeros(d_return_list.size, dtype=data_type)

    if data_type == np.float32:
        # get and launch kernel
        array_index = mod_array_index.get_function("array_index_float")
        array_index(d_array, d_return_values, d_return_list, r_size, block=(block_size, 1, 1), grid=(x_blocks, 1))
    elif data_type == np.int32:
        # get and launch kernel
        array_index = mod_array_index.get_function("array_index_int")
        array_index(d_array, d_return_values, d_return_list, r_size, block=(block_size, 1, 1), grid=(x_blocks, 1))
    else:
        raise ValueError("Unrecognized data type for this function. Use float32 or int32.")

    if return_input:
        if return_list:
            return d_return_values, d_array, d_return_list
        else:
            d_return_list.gpudata.free()
            return d_return_values, d_array
    else:
        d_array.gpudata.free()
        if return_list:
            return d_return_values, d_return_list
        else:
            d_return_list.gpudata.free()
            return d_return_values


def gpu_index_update(d_dest, d_values, d_indices, return_indices=False):
    """Allows for arbitrary index selecting with numpy arrays

    Parameters
    ----------
    d_dest : nD gpuarray - float
        Array to be updated with new values
    d_values : 1D gpuarray - float
        Array containing the values to be updated in the destination array
    d_indices : 1D gpuarray - int
        Array of indices to update
    return_indices : bool
        whether to return the indices

    Returns
    -------
    d_dest : nD array
        Input array with values updated
    d_indices :

    """
    mod_index_update = SourceModule("""
    __global__ void index_update(float *dest, float *values, int *indices, int r_size )
    {
        // 1D grid of 1D blocks
        int tid = blockIdx.x*blockDim.x + threadIdx.x;

        if(tid >= r_size){return;}

        dest[indices[tid]] = values[tid];
    }
    """)
    # TODO delete these checks
    # # GPU will automatically flatten the input array. The indexing must reference the flattened GPU array.
    # assert d_values.ndim == 1, "Number of dimensions of d_values is {}. It should be equal to 1".format(d_values.ndim)
    # assert d_values.size == d_indices.size, "Inputs d_values (size {}) and d_indices (size {}) should have the same size.".format(d_values.size, d_indices.size)
    # assert d_indices.dtype == np.int32, "d_indices data type is {}. Should be np.int32".format(d_indices.dtype)
    # assert d_values.dtype == np.float32, "d_values data type is {}. Should be np.float32".format(d_values.dtype)
    # assert d_dest.dtype == np.float32, "d_dest data type is {}. Should be np.float32".format(d_dest.dtype)

    # define gpu parameters
    # block_size = 8
    block_size = 32
    r_size = np.int32(d_values.size)
    x_blocks = int(r_size//block_size + 1)

    # get and launch kernel
    index_update = mod_index_update.get_function("index_update")
    index_update(d_dest, d_values, d_indices, r_size, block=(block_size, 1, 1), grid=(x_blocks, 1))

    # free gpu data
    d_values.gpudata.free()

    if return_indices is True:
        return d_dest, d_indices
    else:
        d_indices.gpudata.free()
        return d_dest


def gpu_floor(d_src, return_input=False):
    """Takes the floor of each element in the gpu array

    Parameters
    ----------
    d_src : gpuarray
        array to take the floor of
    return_input : bool
        whether to return the input array

    Returns
    -------
    d_dest : gpuarray
        Same size as d_src. Contains the floored values of d_src.

    """
    assert type(return_input) == bool, "ReturnInput is {}. Must be of type boolean".format(type(return_input))

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
    d_dst = gpuarray.empty_like(d_src)

    # get array size for gpu
    n = np.int32(d_src.size)

    # define gpu parameters
    # block_size = 8
    block_size = 32
    x_blocks = int(n//block_size + 1)

    # get and execute kernel
    floor_gpu = mod_floor.get_function("floor_gpu")
    floor_gpu(d_dst, d_src, n, block=(block_size, 1, 1), grid=(x_blocks, 1))

    if not return_input:
        # free some gpu memory
        d_src.gpudata.free()
        return d_dst
    elif return_input:
        return d_dst, d_src
    else:
        raise ValueError("ReturnInput is currently {}. Must be of type boolean".format(return_input))


def gpu_round(d_src, return_input=False):
    """Rounds of each element in the gpu array

    Parameters
    ----------
    d_src : gpuarray
        array to round
    return_input : bool
        whether to return the input array

    Returns
    -------
    d_dest : gpuarray
        Same size as d_src. Contains the floored values of d_src.

    """
    assert type(return_input) == bool, "ReturnInput is {}. Must be of type boolean".format(type(return_input))

    mod_round = SourceModule("""
    __global__ void round_gpu(float *dest, float *src, int n)
    {
        // dest : array to store values
        // src : array of values to be floored

        int tid = blockIdx.x*blockDim.x + threadIdx.x;

        // Avoid the boundary
        if(tid >= n){return;}

        dest[tid] = roundf(src[tid]);
    }
    """)

    # create array to store data
    d_dst = gpuarray.empty_like(d_src)

    # get array size for gpu
    n = np.int32(d_src.size)

    # define gpu parameters
    # block_size = 8
    block_size = 32
    x_blocks = int(n//block_size + 1)

    # get and execute kernel
    round_gpu = mod_round.get_function("round_gpu")
    round_gpu(d_dst, d_src, n, block=(block_size, 1, 1), grid=(x_blocks, 1))

    if return_input is False:
        # free some gpu memory
        d_src.gpudata.free()
        return d_dst
    elif return_input is True:
        return d_dst, d_src
    else:
        raise ValueError("ReturnInput is currently {}. Must be of type boolean".format(return_input))
