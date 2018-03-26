"""This module is dedicated to advanced algorithms for PIV image analysis with NVIDIA GPU Support."""

import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv

import skcuda.fft as cu_fft
import skcuda.misc as cu_misc

import numpy as np
import numpy.ma as ma
from numpy.fft import rfft2,irfft2,fftshift
from math import log
from scipy.signal import convolve
import time
import openpiv
import warnings
from progressbar import *

cimport numpy as np
cimport cython

DTYPEi = np.int32
ctypedef np.int32_t DTYPEi_t


#GPU can only hold 32 bit numbers
DTYPEf = np.float32
ctypedef np.float32_t DTYPEf_t


def gpu_piv( np.ndarray[DTYPEi_t, ndim=2] frame_a, 
             np.ndarray[DTYPEi_t, ndim=2] frame_b,
             int window_size,
             int overlap,
             float dt,
             int search_area_size,
             str subpixel_method='gaussian',
             sig2noise_method=None,
             int width=2,
             nfftx=None,
             nffty=None):
                              
    """
    The implementation of the one-step direct correlation with the same size 
    windows. Support for extended search area of the second window has yet to
    be implimetned. This module is meant to be used with a iterative method to
    cope with the loss of piars due to particle movement out of the search area.
    
    This function is an adaptation of the original extended_search_area_piv
    function. This has been rewritten with PyCuda and CUDA-C to run on
    an NVIDIA GPU. 
    
    WARNING FOR DEVELOPERS: Only single precision calculations can be done on the GPU,
    so all data types must be 32-bit or less.
    
    See:
    
    Particle-Imaging Techniques for Experimental Fluid Mechanics

    Annual Review of Fluid Mechanics
    Vol. 23: 261-304 (Volume publication date January 1991)
    DOI: 10.1146/annurev.fl.23.010191.001401    
    
    Parameters
    ----------
    frame_a : 2d np.ndarray, dtype=np.float32
        an two dimensions array of integers containing grey levels of 
        the first frame.
        
    frame_b : 2d np.ndarray, dtype=np.float32
        an two dimensions array of integers containing grey levels of 
        the second frame.
        
    window_size : int
        the size of the (square) interrogation window.
        
    overlap : int
        the number of pixels by which two adjacent windows overlap.
        
    dt : float
        the time delay separating the two frames.
    
    search_area_size : int
        the size of the (square) interrogation window from the second frame
    
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
        
    nfftx   : int
        the size of the 2D FFT in x-direction, 
        [default: 2 x windows_a.shape[0] is recommended]
        
    nffty   : int
        the size of the 2D FFT in y-direction, 
        [default: 2 x windows_a.shape[1] is recommended]

    
    Returns
    -------
    u : 2d np.ndarray
        a two dimensional array containing the u velocity component,
        in pixels/seconds.
        
    v : 2d np.ndarray
        a two dimensional array containing the v velocity component,
        in pixels/seconds.
        
    sig2noise : 2d np.ndarray, optional
        a two dimensional array containing the signal to noise ratio
        from the cross correlation function. This array is returned if
        sig2noise_method is not None.
        
    Examples
    --------
    
    >>> u, v = openpiv.process.extended_search_area_piv( frame_a, frame_b, window_size=16, overlap=8, search_area_size=48, dt=0.1)
    """
    
    # cast images as floats
    #TODO  changing dtype in the function definition gave weird errors. Find out how to change function definition to avoid this step.
    cdef np.ndarray[DTYPEf_t, ndim=2] frame_a_f = frame_a.astype(np.float32)
    cdef np.ndarray[DTYPEf_t, ndim=2] frame_b_f = frame_b.astype(np.float32)

    # Send images to the gpu
    d_frame_a_f = gpuarray.to_gpu(frame_a_f)
    d_frame_b_f = gpuarray.to_gpu(frame_b_f)
    
    # Define variables
    cdef DTYPEi_t n_rows, n_cols
    
    assert nfftx == nffty, 'fft x and y dimensions must be same size'
    
    # Get correlation function
    c = CorrelationFunction(d_frame_a_f, d_frame_b_f, window_size, overlap, nfftx)

    # Free gpu memory
    d_frame_a_f.gpudata.free()
    d_frame_b_f.gpudata.free()
    
    # vector field shape
    n_rows, n_cols = c.return_shape()
    
    # Define arrays
    cdef np.ndarray[DTYPEf_t, ndim=2] u = np.zeros([n_rows, n_cols], dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=2] v = np.zeros([n_rows, n_cols], dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=2] i_peak = np.zeros([n_rows, n_cols], dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=2] j_peak = np.zeros([n_rows, n_cols], dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=1] i_tmp = np.zeros(n_rows*n_cols, dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=1] j_tmp = np.zeros(n_rows*n_cols, dtype=DTYPEf)   
   
    # Get window displacement to subpixel accuracy
    i_tmp, j_tmp = c.subpixel_peak_location()

    # reshape the peaks
    i_peak = np.reshape(i_tmp, (n_rows, n_cols))
    j_peak = np.reshape(j_tmp, (n_rows, n_cols)) 

    # calculate velocity fields
    v = -( (i_peak - c.nfft/2) - (search_area_size - window_size)/2)/dt
    u =  ( (j_peak - c.nfft/2) - (search_area_size - window_size)/2)/dt 
    
    if sig2noise_method is not None:
        sig2noise = c.sig2noise_ratio(method = sig2noise_method)
        del(c)
        return u,v, sig2noise
    else:
        del(c)
        return u, v
        
        

        

class CorrelationFunction( ):
    def __init__(self, d_frame_a, d_frame_b, window_size, overlap, nfftx, shift = None):
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
            
        shift : 2d ndarray np.array([dx, dy])
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
        d_winA = gpuarray.zeros((self.batch_size, self.window_size, self.window_size), np.float32)
        d_search_area = gpuarray.zeros((self.batch_size, self.window_size, self.window_size), np.float32)
        self._IWarrange(d_frame_a, d_frame_b, d_winA, d_search_area, shift)
        
        #normalize array
        d_winA_norm = gpuarray.zeros((self.batch_size, self.window_size, self.window_size), np.float32)
        d_search_area_norm = gpuarray.zeros((self.batch_size, self.window_size, self.window_size), np.float32)           
        self._normalize_intensity(d_winA, d_search_area, d_winA_norm, d_search_area_norm)

        # zero pad arrays
        d_winA_zp = gpuarray.zeros([self.batch_size, self.nfft, self.nfft], dtype = np.float32)
        d_search_area_zp = gpuarray.zeros_like(d_winA_zp)
        self._zero_pad(d_winA_norm, d_search_area_norm, d_winA_zp, d_search_area_zp)

        # correlate Windows
        self.data = self._correlate_windows(d_winA_zp, d_search_area_zp)

        # get first peak of correlation function
        self.row, self.col, self.corr_max1 = self._find_peak(self.data)

        
                
    def _IWarrange(self, d_frame_a, d_frame_b, d_winA, d_search_area, shift):
        """
        Creates a 3D array stack of all of the interrogation windows. 
        This is necessary to do the FFTs all at once on the GPU.

        Parameters
        -----------
        frame_a, frame_b: 2D numpy arrays - float32
            PIV image pair

        Returns
        -----------
        d_winA: 3D numpy array
             All frame_a interrogation windows stacked on each other

        d_search_area: 3D numpy array
             All frame_b interrogation windows stacked on each other
        """

        #define window slice algorithm
        mod_ws = SourceModule("""
            __global__ void windowSlice(float *input, float *output, int window_size, int overlap, int n_col, int w, int batch_size)
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
                //indeces of image to map from
                f_range = (i/n_col*diff + ind_y)*w + (i%n_col)*diff + ind_x;
                
                //indeces of new array to map to
                w_range = i*IW_size + window_size*ind_y + ind_x;

                output[w_range] = input[f_range];
            }
        }
            
            __global__ void windowSlice_shift(float *input, float *output, int *dx, int *dy, int window_size, int overlap, int n_col, int w, int h, int batch_size)
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
                
                //indeces of image to map from. Apply shift to pixels
                f_range = (i/n_col*diff + y_shift)*w + (i%n_col)*diff + x_shift;
               
                // indeces of image to map to
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
        grid_size = int(8)  # I tested a bit and found this number to be fastest.
        block_size = int(self.window_size / grid_size)

        # slice up windows
        windowSlice = mod_ws.get_function("windowSlice")
        windowSlice(d_frame_a, d_winA, self.window_size, self.overlap, self.n_cols, w, self.batch_size, block = (block_size,block_size,1), grid=(grid_size,grid_size) )

        
        if(shift is None):
            windowSlice(d_frame_b, d_search_area, self.window_size, self.overlap, self.n_cols, w, self.batch_size, block = (block_size,block_size,1), grid=(grid_size,grid_size) )
        else:
            # Define displacement array for second window
            # GPU thread/block architecture uses column major order, so x is the column and y is the row
            # This code is in row major order
            dy = shift[0]
            dx = shift[1]
            
            # Move displacements to the gpu
            d_dx = gpuarray.to_gpu(dx)
            d_dy = gpuarray.to_gpu(dy)

            windowSlice_shift = mod_ws.get_function("windowSlice_shift")
            windowSlice_shift(d_frame_b, d_search_area, d_dx, d_dy, self.window_size, self.overlap, self.n_cols, w, h, self.batch_size, block = (block_size,block_size,1), grid=(grid_size,grid_size) )

            # free displacement GPU memory
            d_dx.gpudata.free()
            d_dy.gpudata.free()

        # free GPU memory
        #d_frame_a.gpudata.free()
        #d_frame_b.gpudata.free()
        
    def _normalize_intensity(self, d_winA, d_search_area, d_winA_norm, d_search_area_norm):
        """
        Remove the mean from each IW of a 3D stack of IW's
        
        Parameters
        ----------
        d_winA : 3D gpuarray - float32
            stack of first frame IW's
        d_search_area : 3D gpuarray - float32
            stack of second frame IW's
            
        Returns
        -------
        norm : 3D gpuarray
            stack of IW's with mean removed
        """
        
        mod_norm = SourceModule("""
            __global__ void normalize(float *array, float *array_norm, float *mean, int IWsize)
        {
            //global thread id for 1D grid of 2D blocks
            int threadId = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
        
            //indeces for mean matrix
            int meanId = threadId / IWsize;
        
            array_norm[threadId] = array[threadId] - mean[meanId];   
        }
        """)
        
        # must do this for skcuda misc library
        cu_misc.init()      
        
        # number of pixels in each interrogation window
        IWsize = np.int32(d_winA.shape[1]*d_winA.shape[2])
        
        # get mean of each IW using skcuda
        d_mean_a = cu_misc.mean(d_winA.reshape(self.batch_size, IWsize), axis=1)
        d_mean_b = cu_misc.mean(d_search_area.reshape(self.batch_size, IWsize), axis=1)
        
        # gpu kernel blocksize parameters
        if d_winA.size%(32**2)==0:
            block_size = int(32)
        else:
            block_size = int(8)
        grid_size = int(d_winA.size / block_size**2)
        
        assert d_winA.size%(block_size**2)==0, 'Not all windows are being normalized. Something wrong with block or grid size.'
        
        # get function and norm IW's
        normalize = mod_norm.get_function('normalize')
        normalize(d_winA, d_winA_norm, d_mean_a, IWsize, block=(block_size, block_size, 1), grid=(grid_size,1))
        normalize(d_search_area, d_search_area_norm, d_mean_b, IWsize, block=(block_size, block_size, 1), grid=(grid_size,1))
        
        # free GPU memory
        d_mean_a.gpudata.free()
        d_mean_b.gpudata.free()
        d_winA.gpudata.free()
        d_search_area.gpudata.free()
        
        
    def _zero_pad(self, d_winA_norm, d_search_area_norm, d_winA_zp, d_search_area_zp):
        """
        Function that zero-pads an 3D stack of arrays for use with the 
        skcuda FFT function.
        
        Parameters
        ----------
        d_winA : 3D gpuarray - float32
            array to be zero padded
            
        d_search_area : 3D gpuarray - float32
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
                //indeces for each IW
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

        
        #gpu parameters
        grid_size = int(8)
        block_size = int(self.window_size / grid_size)  
        
        # get handle and call function
        zero_pad = mod_zp.get_function('zero_pad')
        zero_pad(d_winA_zp, d_winA_norm, self.nfft, self.window_size, self.batch_size, block=(block_size, block_size,1), grid=(grid_size,grid_size))
        zero_pad(d_search_area_zp, d_search_area_norm, self.nfft    , self.window_size, self.batch_size, block=(block_size, block_size,1), grid=(grid_size,grid_size))
        
        # Free GPU memory
        d_winA_norm.gpudata.free()
        d_search_area_norm.gpudata.free()
        
        
    def _correlate_windows(self, d_winA_zp, d_search_area_zp):
        """Compute correlation function between two interrogation windows.
        
        The correlation function can be computed by using the correlation 
        theorem to speed up the computation.
        
        Parameters
        ----------
        
            
        Returns
        -------
        corr : 2d np.ndarray
            a two dimensions array for the correlation function.      
        """
        # FFT size
        win_h = np.int32(self.nfft)
        win_w = np.int32(self.nfft)
        
        # allocate space on gpu for FFT's
        d_winIFFT = gpuarray.empty((self.batch_size, win_h, win_w), np.float32)
        d_winFFT = gpuarray.empty((self.batch_size, win_h, win_w//2+1), np.complex64)
        d_searchAreaFFT = gpuarray.empty((self.batch_size, win_h, win_w//2+1), np.complex64)
        
        # forward fft's
        plan_forward = cu_fft.Plan((win_h, win_w), np.float32, np.complex64, self.batch_size)
        cu_fft.fft(d_winA_zp, d_winFFT, plan_forward)
        cu_fft.fft(d_search_area_zp, d_searchAreaFFT, plan_forward)
        
        #multiply the ffts
        d_winFFT = d_winFFT.conj()
        d_tmp = cu_misc.multiply(d_searchAreaFFT, d_winFFT)
        
        #inverse transform
        plan_inverse = cu_fft.Plan((win_h, win_w), np.complex64, np.float32, self.batch_size)
        cu_fft.ifft(d_tmp, d_winIFFT, plan_inverse, True)
        
        # transfer back to cpu to do fftshift
        corr = fftshift(d_winIFFT.get().real, axes = (1,2))

        #free gpu memory
        d_winIFFT.gpudata.free()
        d_winFFT.gpudata.free()
        d_searchAreaFFT.gpudata.free()
        d_tmp.gpudata.free()
        d_winA_zp.gpudata.free()
        d_search_area_zp.gpudata.free()
        
        # delete classes for the plan for free any associated memory
        del(plan_forward, plan_inverse)
             
        return(corr)
        
        
    def _find_peak(self, array):
        """
        Find row and column of highest peak in correlation function
      
        Outputs
        -------
        ind: 1D array - int
            flattened index of corr peak
        row: 1D array - int
            row position of corr peak
        col: 1D array - int
            column position of corr peak

        """
        # Reshape matrix        
        array_reshape = array.reshape(self.batch_size, self.nfft**2)
        s = self.nfft
        
        # Get index and value of peak
        ind = np.argmax(array_reshape, axis = 1)
        maximum = np.amax(array_reshape, axis = 1)
   
        # row and column information of peak
        row = ind // s
        col = ind % s

        return(row, col, maximum)
        
        
    def _find_second_peak ( self, width ):
        """
        Find the value of the second largest peak.
        
        The second largest peak is the height of the peak in 
        the region outside a ``width * width`` submatrix around 
        the first correlation peak.
        
        Inputs
        ----------
        width : int
            the half size of the region around the first correlation 
            peak to ignore for finding the second peak.
              
        Outputs
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
        """
        Find subpixel peak Approximation using Gaussian method
        
        Inputs
        ------
            corr: 3D numpy array - float
                stack of all correlation functions
            row: 1D numpy array - int
                row location of corr max
            col: 1D numpy array - int
                column location of corr max
        
        Outputs
        -------
            row_sp: 1D numpy array - float
                row max location to subpixel accuracy
            col_sp: 1D numpy array - float
                column max location to subpixel accuracy 
        """

        # Define small number to replace zeros and get rid of warnings in calculations
        cdef DTYPEf_t SMALL = 1e-20

        #cast corr and row as a ctype array
        cdef np.ndarray[DTYPEf_t, ndim=3] corr_c = np.array(self.data, dtype = DTYPEf)
        cdef np.ndarray[DTYPEf_t, ndim=1] row_c = np.array(self.row, dtype = DTYPEf)
        cdef np.ndarray[DTYPEf_t, ndim=1] col_c = np.array(self.col, dtype = DTYPEf)

        # Define arrays to store the data
        cdef np.ndarray[DTYPEf_t, ndim=1] row_sp = np.empty(self.batch_size, dtype = DTYPEf)
        cdef np.ndarray[DTYPEf_t, ndim=1] col_sp = np.empty(self.batch_size, dtype = DTYPEf)

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
        cdef np.ndarray[DTYPEf_t, ndim=1] non_zero = np.array(c > 0, dtype = DTYPEf)
        c[c <= 0] = SMALL
        cl[cl <= 0] = SMALL
        cr[cr <= 0] = SMALL
        cd[cd <= 0] = SMALL
        cu[cu <= 0] = SMALL
       
        # Do subpixel approximation. Add SMALL to avoid zero divide.
        row_sp = row_c + ( (np.log(cl)-np.log(cr) )/( 2*np.log(cl) - 4*np.log(c) + 2*np.log(cr) + SMALL ))*non_zero
        col_sp = col_c + ( (np.log(cd)-np.log(cu) )/( 2*np.log(cd) - 4*np.log(c) + 2*np.log(cu) + SMALL))*non_zero
        
        return(row_sp, col_sp)
        

    def sig2noise_ratio( self, method='peak2peak', width=2 ):
        """Computes the signal to noise ratio.
        
        The signal to noise ratio is computed from the correlation map with
        one of two available method. It is a measure of the quality of the 
        matching between two interogation windows.
        
        Parameters
        ----------
        sig2noise_method: string
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
            corr_max2 = self._find_second_peak( width=width )
            
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
        sig2noise[np.isnan(sig2noise)] == 0.0
            
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
        return(self.n_rows, self.n_cols)
        
        
        
        
        
        
        
def get_field_shape ( image_size, window_size, overlap ):
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
    
    return ( (image_size[0] - window_size)//(window_size-overlap)+1, 
             (image_size[1] - window_size)//(window_size-overlap)+1 )
        
        
        
        







##################################################################
def WiDIM( np.ndarray[DTYPEi_t, ndim=2] frame_a, 
           np.ndarray[DTYPEi_t, ndim=2] frame_b,
           np.ndarray[DTYPEi_t, ndim=2] mark,
           int min_window_size,
           float overlap_ratio,
           int coarse_factor,
           float dt,
           str validation_method='mean_velocity',
           int div_validation = 1,
           int trust_1st_iter=1,
           int validation_iter = 1,
           float tolerance = 1.5,
           float div_tolerance = 0.1,
           int nb_iter_max=3,
           str subpixel_method='gaussian',
           str sig2noise_method='peak2peak',
           int width=2,
           nfftx=None,
           nffty=None):
    """
    Implementation of the WiDIM algorithm (Window Displacement Iterative Method).
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
        F. Scarano & M. L. Riethmuller, Iterative multigrid approach in PIV image processing with discrete window offset, Experiments in Fluids 26 (1999) 513-523
    
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
    
    coarse_factor : int
        how many times the window size refining processes happens. 
        
    dt : float
        the time delay separating the two frames.
    
    validation_method : string
        the method used for validation (in addition to the sig2noise method). Only the mean velocity method is implemented now
    
    trust_1st_iter : int = 0 or 1
        0 if the first iteration need to be validated. With a first window size following the 1/4 rule, the 1st iteration can be trusted and the value should be 1 (Default value)
     
    validation_iter : int
        number of iterations per validation cycle.

    div_validation : int
        Boolean - if 1 then the data wil be validated by calculating the divergence. If 0 then it will not be done. 
       
    tolerance : float
        the threshold for the validation method chosen. This does not concern the sig2noise for which the threshold is 1.5; [nb: this could change in the future]
   
    div_tolerance : float
        Threshold value for the maximum divergence at each point. Another validation check to make sure the velocity field is acceptible. 

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
        
    nfftx   : int
        the size of the 2D FFT in x-direction, 
        [default: 2 x windows_a.shape[0] is recommended]
        
    nffty   : int
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
    --------
    
    >>> x,y,u,v, mask = openpiv.process.WiDIM( frame_a, frame_b, mark, min_window_size=16, overlap_ratio=0.25, coarse_factor=2, dt=0.02, validation_method='mean_velocity', trust_1st_iter=1, validation_iter=2, tolerance=0.7, nb_iter_max=4, sig2noise_method='peak2peak')

    --------------------------------------
    Method of implementation : to improve the speed of the programm,
    all data have been placed in the same huge 4-dimensions 'F' array.
    (this prevent the definition of a new array for each iteration)
    However, during the coarsening process a large part of the array is not used.
    Structure of array F:
    --The 1st index is the main iteration (K)   --> length is nb_iter_max
        -- 2nd index (I) is row (of the map of the interpolations locations of iteration K) --> length (effectively used) is Nrow[K]
            --3rd index (J) is column  --> length (effectively used) is Ncol[K]
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
    
    
    # cast images as floats
    #TODO  changing dtype in the function definition gave weird errors. Find out how to change function definition to avoid this step.
    cdef np.ndarray[DTYPEf_t, ndim=2] frame_a_f = frame_a.astype(np.float32)
    cdef np.ndarray[DTYPEf_t, ndim=2] frame_b_f = frame_b.astype(np.float32)

    # Send images to the gpu
    d_frame_a_f = gpuarray.to_gpu(frame_a_f)
    d_frame_b_f = gpuarray.to_gpu(frame_b_f)
    
    #warnings.warn("deprecated", RuntimeWarning)
    if nb_iter_max <= coarse_factor:
        raise ValueError( "Please provide a nb_iter_max that is greater than the coarse_level" )
    cdef int K #main iteration index
    cdef int I, J #interrogation locations indices
    cdef int L, M #inside window indices
    cdef int O, P #frame indices corresponding to I and J
    cdef int i, j #dumb indices for various works
    cdef float mean_u, mean_v, rms_u, rms_v, residual_0, div
    cdef int residual, nbwind
    cdef np.ndarray[DTYPEi_t, ndim=1] Nrow = np.zeros(nb_iter_max, dtype=DTYPEi)
    cdef np.ndarray[DTYPEi_t, ndim=1] Ncol = np.zeros(nb_iter_max, dtype=DTYPEi)
    cdef np.ndarray[DTYPEi_t, ndim=1] W = np.zeros(nb_iter_max, dtype=DTYPEi)
    cdef np.ndarray[DTYPEi_t, ndim=1] Overlap = np.zeros(nb_iter_max, dtype=DTYPEi)
    pic_size=frame_a.shape

    
    #window sizes list initialization
    for K in range(coarse_factor+1):
        W[K] = np.power(2,coarse_factor - K)*min_window_size
    for K in range(coarse_factor+1,nb_iter_max):
        W[K] = W[K-1]
        
    #overlap init
    for K in range(nb_iter_max):
        Overlap[K]=int(np.floor(overlap_ratio*W[K]))
        
    #Ncol and Nrow init
    for K in range(nb_iter_max):
        Nrow[K]=((pic_size[0]-W[K])//(W[K]-Overlap[K]))+1
        Ncol[K]=((pic_size[1]-W[K])//(W[K]-Overlap[K]))+1
        
    #writting the parameters to the screen
    if validation_iter==0:
        validation_method='None'

    #cdef float startTime = launch(method='WiDIM', names=['Size of image', 'total number of iterations', 'overlap ratio', 'coarse factor', 'time step', 'validation method', 'number of validation iterations', 'subpixel_method','Nrow', 'Ncol', 'Window sizes', 'overlaps'], arg=[[pic_size[0], pic_size[1]], nb_iter_max, overlap_ratio, coarse_factor, dt, validation_method, validation_iter,  subpixel_method, Nrow, Ncol, W, Overlap])
    
    #define the main array F that contains all the data
    cdef np.ndarray[DTYPEf_t, ndim=4] F = np.zeros([nb_iter_max, Nrow[nb_iter_max-1], Ncol[nb_iter_max-1], 14], dtype=DTYPEf)
    
    #define mask - bool array don't exist in cython so we go to lower level with cast
    #you can access mask with (<object>mask)[I,J]
    cdef np.ndarray[np.uint8_t, ndim=2, cast=True] mask = np.empty([Nrow[nb_iter_max-1], Ncol[nb_iter_max-1]], dtype=np.bool)
    
    #define u,v, x,y fields (only used as outputs of this programm)
    cdef np.ndarray[DTYPEf_t, ndim=2] u = np.zeros([Nrow[nb_iter_max-1], Ncol[nb_iter_max-1]], dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=2] v = np.zeros([Nrow[nb_iter_max-1], Ncol[nb_iter_max-1]], dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=2] x = np.zeros([Nrow[nb_iter_max-1], Ncol[nb_iter_max-1]], dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=2] y = np.zeros([Nrow[nb_iter_max-1], Ncol[nb_iter_max-1]], dtype=DTYPEf)
    
    # define arrays to stores the displacement vector in to save displacement information
    cdef np.ndarray[DTYPEi_t, ndim=2] shift = np.zeros([2,Nrow[-1]*Ncol[-1]], dtype=DTYPEi)
    
    # define temporary arrays and reshaped arrays to store the correlation function output
    cdef np.ndarray[DTYPEf_t, ndim=1] i_tmp = np.zeros(Nrow[-1]*Ncol[-1], dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=1] j_tmp = np.zeros(Nrow[-1]*Ncol[-1], dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=2] i_peak = np.zeros([Nrow[nb_iter_max-1], Ncol[nb_iter_max-1]], dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=2] j_peak = np.zeros([Nrow[nb_iter_max-1], Ncol[nb_iter_max-1]], dtype=DTYPEf)
    
    # define array for signal to noise ratio
    cdef np.ndarray[DTYPEf_t, ndim=2] sig2noise = np.zeros([Nrow[-1], Ncol[-1]], dtype=DTYPEf)
    
    #define two small arrays used for the validation process
    cdef np.ndarray[DTYPEf_t, ndim=3] neighbours = np.zeros([2,3,3], dtype=DTYPEf)
    cdef np.ndarray[DTYPEi_t, ndim=2] neighbours_present = np.zeros([3,3], dtype=DTYPEi)
    
    #initialize x and y values
    for K in range(nb_iter_max):
        for I in range(Nrow[K]):
            for J in range(Ncol[K]):
                #x unit vector corresponds to rows
                #y unit vector corresponds to columns
                if I == 0:
                    F[K,I,J,0] = W[K]/2 #init x on 1st row
                else:
                    F[K,I,J,0] = F[K,I-1,J,0] + W[K] - Overlap[K] #init x
                if J == 0:
                    F[K,I,J,1] = W[K]/2 #init y on first column
                else:
                    F[K,I,J,1] = F[K,I,J-1,1] + W[K] - Overlap[K] #init y
                    
    #end of the initializations
    
    
    ####################################################
    # MAIN LOOP
    ####################################################
    
    for K in range(nb_iter_max):
        #print " "
        #print "//////////////////////////////////////////////////////////////////"
        #print " "
        print "ITERATION # ",K
        #print " "
        
        
        #a simple progress bar
        #widgets = ['Computing the displacements : ', Percentage(), ' ', Bar(marker='-',left='[',right=']'),
        #   ' ', ETA(), ' ', FileTransferSpeed()]
        #pbar = ProgressBar(widgets=widgets, maxval=100)
        #pbar.start()
        residual = 0
        
        #################################################################################
        #  GPU VERSION
        #################################################################################
        
        # Calculate second frame displacement (shift)
        shift[0, 0:Nrow[K]*Ncol[K]] = F[K,0:Nrow[K], 0:Ncol[K], 6].flatten().astype(np.int32)  #xb=xa+dpx
        shift[1, 0:Nrow[K]*Ncol[K]] = F[K,0:Nrow[K], 0:Ncol[K], 7].flatten().astype(np.int32)  #yb=ya+dpy
        
        # Get correlation function
        #if K == 0:
        #    c = CorrelationFunction(frame_a_f, frame_b_f, W[K], Overlap[K], nfftx)
        #else:
        c = CorrelationFunction(d_frame_a_f, d_frame_b_f, W[K], Overlap[K], nfftx, shift = shift[:, 0:Nrow[K]*Ncol[K]])
            
        # Get window displacement to subpixel accuracy
        i_tmp[0:Nrow[K]*Ncol[K]], j_tmp[0:Nrow[K]*Ncol[K]] = c.subpixel_peak_location()

        # reshape the peaks
        i_peak[0:Nrow[K], 0: Ncol[K]] = np.reshape(i_tmp[0:Nrow[K]*Ncol[K]], (Nrow[K], Ncol[K]))
        j_peak[0:Nrow[K], 0: Ncol[K]] = np.reshape(j_tmp[0:Nrow[K]*Ncol[K]], (Nrow[K], Ncol[K])) 
        
        # Get signal to noise ratio
        sig2noise[0:Nrow[K], 0:Ncol[K]] = c.sig2noise_ratio(method = sig2noise_method)
        
        # Loop through F and update values
        for I in range(Nrow[K]):
            #pbar.update(100*I/Nrow[K])#progress update
            for J in range(Ncol[K]):
            
                F[K,I,J,2] = np.floor(F[K,I,J,0] + F[K,I,J,6]) #xb=xa+dpx
                F[K,I,J,3] = np.floor(F[K,I,J,1] + F[K,I,J,7]) #yb=yb+dpy
                
                #prevent 'Not a Number' peak location
                if np.any(np.isnan((i_peak[I,J], j_peak[I,J]))) or mark[int(F[K,I,J,0]), int(F[K,I,J,1])] == 0:
                    F[K,I,J,8] = 0.0
                    F[K,I,J,9] = 0.0
                else:
                    #find residual displacement dcx and dcy
                    F[K,I,J,8] = i_peak[I,J] - c.nfft/2 #dcx
                    F[K,I,J,9] = j_peak[I,J] - c.nfft/2 #dcy
                    
                residual = residual + np.sqrt(F[K,I,J,8]*F[K,I,J,8] + F[K,I,J,9]*F[K,I,J,9])
                
                #get new displacement prediction
                F[K,I,J,4] = F[K,I,J,6] + F[K,I,J,8]  #dx=dpx+dcx
                F[K,I,J,5] = F[K,I,J,7] + F[K,I,J,9]  #dy=dpy+dcy
                #get new velocity vectors
                F[K,I,J,10] = F[K,I,J,5] / dt #u=dy/dt
                F[K,I,J,11] = -F[K,I,J,4] / dt #v=-dx/dt
                
                # get sig2noise ratio
                F[K,I,J,12] = sig2noise[I,J]
        
        #################################################################################
        """
        #run through interpolations locations
        for I in range(Nrow[K]):
            pbar.update(100*I/Nrow[K])#progress update
            for J in range(Ncol[K]):
                
                #compute xb, yb:
                F[K,I,J,2] = np.floor(F[K,I,J,0] + F[K,I,J,6]) #xb=xa+dpx
                F[K,I,J,3] = np.floor(F[K,I,J,1] + F[K,I,J,7]) #yb=yb+dpy
                
                # Look for corrupted window (ie. going outside of the picture) and relocate them with 0 displacement:
                # if corrupted on x-axis do:
                if F[K,I,J,2] + W[K]/2 > pic_size[0]-1 or F[K,I,J,2] - W[K]/2 < 0: 
                    F[K,I,J,2] = F[K,I,J,0] #xb=x
                    F[K,I,J,3] = F[K,I,J,1] #yb=y
                    F[K,I,J,6] = 0.0 #dpx=0
                    F[K,I,J,7] = 0.0 #dpy=0
                # if corrupted on y-axis do the same
                elif F[K,I,J,3] + W[K]/2 > pic_size[1]-1 or F[K,I,J,3] - W[K]/2 < 0: 
                    F[K,I,J,2] = F[K,I,J,0] #xb=x
                    F[K,I,J,3] = F[K,I,J,1] #yb=y
                    F[K,I,J,6] = 0.0 #dpx=0
                    F[K,I,J,7] = 0.0 #dpy=0
                    
                #fill windows a and b
                for L in range(W[K]):
                    for M in range(W[K]):
                        window_a[L,M] = frame_a[F[K,I,J,0] - W[K]/2 + L, F[K,I,J,1] - W[K]/2 + M]
                        window_b[L,M] = frame_b[F[K,I,J,2] - W[K]/2 + L, F[K,I,J,3] - W[K]/2 + M]
                        
                #perform correlation of the two windows
                corr = correlate_windows( window_b, window_a, nfftx=nfftx, nffty=nffty )
                c = CorrelationFunction( corr )
                F[K,I,J,12] = c.sig2noise_ratio( sig2noise_method, width )#compute sig2noise
                i_peak, j_peak = c.subpixel_peak_position( subpixel_method )#get peak position

                #prevent 'Not a Number' peak location
                if np.any(np.isnan((i_peak, j_peak))) or mark[F[K,I,J,0], F[K,I,J,1]] == 0:
                    F[K,I,J,8] = 0.0
                    F[K,I,J,9] = 0.0
                else:
                    #find residual displacement dcx and dcy
                    F[K,I,J,8] = i_peak - corr.shape[0]/2 #dcx
                    F[K,I,J,9] = j_peak - corr.shape[1]/2 #dcy

                residual = residual + np.sqrt(F[K,I,J,8]*F[K,I,J,8] + F[K,I,J,9]*F[K,I,J,9])
                
                #get new displacement prediction
                F[K,I,J,4] = F[K,I,J,6] + F[K,I,J,8]  #dx=dpx+dcx
                F[K,I,J,5] = F[K,I,J,7] + F[K,I,J,9]  #dy=dpy+dcy
                #get new velocity vectors
                F[K,I,J,10] = F[K,I,J,5] / dt #u=dy/dt
                F[K,I,J,11] = -F[K,I,J,4] / dt #v=-dx/dt
                
        """      
        #pbar.finish()#close progress bar
      
        print "..[DONE]"
        if K==0:
            residual_0 = residual/np.float(Nrow[K]*Ncol[K])
            print(residual_0)
        #print " --residual : ", (residual/np.float(Nrow[K]*Ncol[K]))/residual_0
        
        
        #####################################################
        #validation of the velocity vectors with 3*3 filtering
        #####################################################
        
        if K==0 and trust_1st_iter:#1st iteration can generally be trust if it follows the 1/4 rule
            print "no validation : trusting 1st iteration"
        else: 
            print "Starting validation.."

            #init mask to False
            for I in range(Nrow[nb_iter_max-1]):
                for J in range(Ncol[nb_iter_max-1]):
                    (<object>mask)[I,J] = False

            #real validation starts
            for i in range(validation_iter):
                print "Validation, iteration number ",i
                print " "
                #widgets = ['Validation : ', Percentage(), ' ', Bar(marker='-',left='[',right=']'),
                #' ', ETA(), ' ', FileTransferSpeed()]
                #pbar = ProgressBar(widgets=widgets, maxval=100)
                #pbar.start()
                
                #run through locations
                for I in range(Nrow[K]):
                    #pbar.update(100*I/Nrow[K])                    
                    for J in range(Ncol[K]):
                        neighbours_present = find_neighbours(I, J, Nrow[K]-1, Ncol[K]-1)#get a map of the neighbouring locations
                        
                        #get the velocity of the neighbours in a 2*3*3 array
                        
                        for L in range(3):
                            for M in range(3):
                                if neighbours_present[L,M]:
                                    neighbours[0,L,M] = F[K,I+L-1,J+M-1,10]#u
                                    neighbours[1,L,M] = F[K,I+L-1,J+M-1,11]#v
                                else:
                                    neighbours[0,L,M] = 0
                                    neighbours[1,L,M] = 0
                        
                        
                        # If there are neighbours present and no mask, validate the velocity
                        if np.sum(neighbours_present) !=0 and mark[int(F[K,I,J,0]), int(F[K,I,J,1])] == 1:
                        #if np.sum(neighbours_present):

                            #computing the mean velocity
                            mean_u = np.sum(neighbours[0])/np.float(np.sum(neighbours_present))
                            mean_v = np.sum(neighbours[1])/np.float(np.sum(neighbours_present))

                            #validation with the sig2noise ratio, 1.5 is a recommended minimum value
                            if F[K,I,J,12] < 1.5:
                                #if in 1st iteration, no interpolation is needed so just replace by the mean
                                if K==0:
                                    F[K,I,J,10] = mean_u
                                    F[K,I,J,11] = mean_v
                                    (<object>mask)[I,J]=True
                                    F[K,I,J,4] = -F[K,I,J,11]*dt #recompute displacement from velocity
                                    F[K,I,J,5] = F[K,I,J,10]*dt
                                #perform interpolation using previous iteration (which is supposed to be already validated -> this prevents error propagation)
                                elif K>0 and (Nrow[K] != Nrow[K-1] or Ncol[K] != Ncol[K-1]):
                                    F[K,I,J,10] = interpolate_surroundings(F,Nrow,Ncol,K-1,I,J, 10)
                                    F[K,I,J,11] = interpolate_surroundings(F,Nrow,Ncol,K-1,I,J, 11)

                            #add a validation with the mean and rms values. This happens as well as sig2noise vaildation
                            if validation_method == 'mean_velocity':

                                #get rms of u and v
                                rms_u = np.sqrt(sumsquare_array(neighbours[0])/np.float(np.sum(neighbours_present)))
                                rms_v = np.sqrt(sumsquare_array(neighbours[1])/np.float(np.sum(neighbours_present)))

                                if rms_u==0 or rms_v==0:
                                        F[K,I,J,10] = mean_u
                                        F[K,I,J,11] = mean_v
                                elif ((F[K,I,J,10] - mean_u)/rms_u) > tolerance or ((F[K,I,J,11] - mean_v)/rms_v) > tolerance:

                                    initiate_validation(F, Nrow, Ncol, neighbours_present, neighbours, mean_u, mean_v, dt, K, I, J)
                                    (<object>mask)[I,J] = True

                            # Validate based on divergence of the velocity field

                            if div_validation == 1:
                                #check for boundary
                                if I ==  Nrow[K] - 1 or J == Ncol[K] - 1:
                                    # div = du/dy - dv/dx   see paper if you are confused as I was
                                    div = np.abs((F[K,I,J,10] - F[K,I-1,J,10])/W[K] - (F[K,I,J,11] - F[K,I,J-1,11])/W[K])
                                else:
                                    div = np.abs((F[K,I+1,J,10] - F[K,I,J,10])/W[K] - (F[K,I,J+1,11] - F[K,I,J,11])/W[K])

                                # if div is greater than 0.1, interpolate the value. 
                                if div > div_tolerance:
                                    initiate_validation(F, Nrow, Ncol, neighbours_present, neighbours, mean_u, mean_v, dt, K, I, J)
                                    (<object>mask)[I,J] = True
 
            #pbar.finish()                    
            print "..[DONE]"
            print " "
        #end of validation

        ##############################################################################
        #stop process if this is the last iteration
        if K==nb_iter_max-1:
            #print "//////////////////////////////////////////////////////////////////"
            print "end of iterative process.. Re-arranging vector fields.."
            for I in range(Nrow[K]):#assembling the u,v and x,y fields for outputs
                for J in range(Ncol[K]):
                    x[I,J]=F[K,I,J,1]
                    y[I,J]=F[K,Nrow[K]-I-1,J,0]
                    u[I,J]=F[K,I,J,10]
                    v[I,J]=F[K,I,J,11]
    
            print "...[DONE]"

            # delete images from gpu memory
            d_frame_a_f.gpudata.free()
            d_frame_b_f.gpudata.free()
            
            # delete old correlation function
            del(c)
            #end(startTime)
            return x, y, u, v, (<object>mask)
        #############################################################################

        #go to next iteration : compute the predictors dpx and dpy from the current displacements
        print "going to next iteration.. "
        print "performing interpolation of the displacement field"
        print " "
        #widgets = ['Performing interpolations : ', Percentage(), ' ', Bar(marker='-',left='[',right=']'),
        #   ' ', ETA(), ' ', FileTransferSpeed()]
        #pbar = ProgressBar(widgets=widgets, maxval=100)
        #pbar.start()

        for I in range(Nrow[K+1]):
            #pbar.update(100*I/Nrow[K+1])
            for J in range(Ncol[K+1]):

                # If vector field dimensions agree
                # Make sure predictor is an integer number of pixels
                if Nrow[K+1] == Nrow[K] and Ncol[K+1] == Ncol[K]:
                    F[K+1,I,J,6] = np.floor(F[K,I,J,4]) #dpx_k+1 = dx_k 
                    F[K+1,I,J,7] = np.floor(F[K,I,J,5]) #dpy_k+1 = dy_k
                #interpolate if dimensions do not agree
                else:
                    F[K+1,I,J,6] = np.floor(interpolate_surroundings(F,Nrow,Ncol,K,I,J, 4))
                    F[K+1,I,J,7] = np.floor(interpolate_surroundings(F,Nrow,Ncol,K,I,J, 5))
        
        #pbar.finish()
        
        # delete old correlation function
        del(c)
        
        #print "..[DONE] -----> going to iteration ",K+1
        #print " "


def initiate_validation( np.ndarray[DTYPEf_t, ndim=4] F,
                         np.ndarray[DTYPEi_t, ndim=1] Nrow,
                         np.ndarray[DTYPEi_t, ndim=1] Ncol,
                         np.ndarray[DTYPEi_t, ndim=2] neighbours_present,
                         np.ndarray[DTYPEf_t, ndim=3] neighbours,
                         float mean_u,
                         float mean_v,
                         float dt,
                         int K,
                         int I,
                         int J):

    """
    Parameters
    ----------
    F :  4d np.ndarray
        The main array of the WIDIM algorithm.

    Nrow : 1d np.ndarray
        list of the numbers of row for each iteration K
       
    Ncol : 1d np.ndarray
        list of the numbers of column for each iteration K

    neighbours_present : 2d np.ndarray
        3x3 array surrounding the point indicating if the point has neighbouring values

    neighbours : 3d np.ndarray
        the value of the velocity at the neighbouring points

    mean_u, mean_v : float
        mean velocities of the neighbouring points

    dt : float
        time step between image frames
    
    K : int
        The current main loop iteration
    
    I,J : int
        indices of the point that need interpolation 
    """


    # No previous iteration. Replace with mean velocity
    if K==0:
        F[K,I,J,10] = mean_u
        F[K,I,J,11] = mean_v
        F[K,I,J,4] = -F[K,I,J,11]*dt
        F[K,I,J,5] = F[K,I,J,10]*dt
    #case if different dimensions : interpolation using previous iteration
    elif K>0 and (Nrow[K] != Nrow[K-1] or Ncol[K] != Ncol[K-1]):
        F[K,I,J,10] = interpolate_surroundings(F,Nrow,Ncol,K-1,I,J, 10)
        F[K,I,J,11] = interpolate_surroundings(F,Nrow,Ncol,K-1,I,J, 11)
        F[K,I,J,4] = -F[K,I,J,11]*dt
        F[K,I,J,5] = F[K,I,J,10]*dt
    #case if same dimensions
    elif K>0 and (Nrow[K] == Nrow[K-1] or Ncol[K] == Ncol[K-1]):
        for L in range(3):
            for M in range(3):
                if neighbours_present[L,M]:
                    neighbours[0,L,M] = F[K-1,I+L-1,J+M-1,10]#u
                    neighbours[1,L,M] = F[K-1,I+L-1,J+M-1,11]#v
                else:
                    neighbours[0,L,M] = 0
                    neighbours[1,L,M] = 0
        if np.sum(neighbours_present) != 0:
            mean_u = np.sum(neighbours[0])/np.float(np.sum(neighbours_present))
            mean_v = np.sum(neighbours[1])/np.float(np.sum(neighbours_present))
            F[K,I,J,10] = mean_u
            F[K,I,J,11] = mean_v
            F[K,I,J,4] = -F[K,I,J,11]*dt
            F[K,I,J,5] = F[K,I,J,10]*dt


def interpolate_surroundings(np.ndarray[DTYPEf_t, ndim=4] F,
                             np.ndarray[DTYPEi_t, ndim=1] Nrow,
                             np.ndarray[DTYPEi_t, ndim=1] Ncol,
                             int K,
                             int I,
                             int J,
                             int dat):
    """Perform interpolation of between to iterations of the F 4d-array for a specific location I,J and the data type dat.
    
    Parameters
    ----------
    F :  4d np.ndarray
        The main array of the WIDIM algorithm.

    Nrow : 1d np.ndarray
        list of the numbers of row for each iteration K
       
    Ncol : 1d np.ndarray
        list of the numbers of column for each iteration K
    
    K : int
        the iteration that contains the valid data. K+1 will be the iteration at which the interpolation is needed.
    
    I,J : int
        indices of the point that need interpolation (in iteration K+1)
    
    dat : int
        the index of the data to interpolate.
    
    Returns
    -------
    the interpolated data (type float)
    
    """

    #interpolate data dat from previous iteration
    cdef float lower_lim_previous_x = F[K,0,0,0]
    cdef float lower_lim_previous_y = F[K,0,0,1]
    cdef float upper_lim_previous_x = F[K,Nrow[K]-1,Ncol[K]-1,0]
    cdef float upper_lim_previous_y = F[K,Nrow[K]-1,Ncol[K]-1,1]
    cdef float pos_now_x = F[K+1,I,J,0]
    cdef float pos_now_y = F[K+1,I,J,1]
    cdef np.ndarray[DTYPEi_t, ndim=1] Q1 = np.zeros(2, dtype=DTYPEi)
    cdef np.ndarray[DTYPEi_t, ndim=1] Q4 = np.zeros(2, dtype=DTYPEi)

    # Many cases depending on where the vector location is

    # Top row
    if pos_now_x < lower_lim_previous_x:
        #top left corner
        if pos_now_y < lower_lim_previous_y:
            return F[K,0,0,dat]
        #top right corner
        elif pos_now_y > upper_lim_previous_y:
            return F[K,0,Ncol[K]-1,dat]
        #top row no corners
        else:
            low_y, high_y = F_dichotomy(F,K,Ncol,'y_axis',pos_now_y)
            if low_y == high_y:
                return F[K,0,low_y,dat]
            else:
                return linear_interpolation(F[K,0,low_y,1], F[K,0,high_y,1], pos_now_y, F[K,0,low_y,dat], F[K,0,high_y,dat])
    # Bottom row
    elif pos_now_x > upper_lim_previous_x:
        # bottom left corner
        if pos_now_y < lower_lim_previous_y:
            return F[K,Nrow[K]-1,0,dat]
        #bottom right corner
        elif pos_now_y > upper_lim_previous_y:
            return F[K,Nrow[K]-1,Ncol[K]-1,dat]
        #bottom row no corners
        else:
            low_y, high_y = F_dichotomy(F,K,Ncol,'y_axis',pos_now_y)
            #print low_y, high_y
            if low_y == high_y:
                return F[K,Nrow[K]-1,low_y,dat]
            else:
                return linear_interpolation(F[K,0,low_y,1], F[K,0,high_y,1], pos_now_y, F[K,Nrow[K]-1,low_y,dat], F[K,Nrow[K]-1,high_y,dat])
    #left column no corners
    elif pos_now_y < lower_lim_previous_y:
        low_x, high_x = F_dichotomy(F,K,Nrow,'x_axis',pos_now_x)
        if low_x == high_x:
            return F[K,low_x,0,dat]
        else:
            return linear_interpolation(F[K,low_x,0,0], F[K,high_x,0,0], pos_now_x, F[K,low_x,0,dat], F[K,high_x,0,dat])
    #right column no corners
    elif pos_now_y > upper_lim_previous_y:
        low_x, high_x = F_dichotomy(F,K,Nrow,'x_axis',pos_now_x)
        if low_x == high_x:
            return F[K,low_x,Ncol[K]-1,dat]
        else:
            return linear_interpolation(F[K,low_x,0,0], F[K,high_x,0,0], pos_now_x, F[K,low_x,Ncol[K]-1,dat], F[K,high_x,Ncol[K]-1,dat])
    #interior grid
    else:
        low_x, high_x = F_dichotomy(F,K,Nrow,'x_axis',pos_now_x)
        low_y, high_y = F_dichotomy(F,K,Ncol,'y_axis',pos_now_y)
        Q1[0] = F[K,low_x,0,0] 
        Q1[1] = F[K,0,low_y,1]
        Q4[0] = F[K,high_x,0,0]
        Q4[1] = F[K,0,high_y,1]
        if pos_now_x >= Q1[0] and pos_now_x <= Q4[0] and pos_now_y >= Q1[1] and pos_now_y <= Q4[1]:
            return bilinear_interpolation(Q1[0],Q4[0],Q1[1],Q4[1],pos_now_x,pos_now_y,F[K,low_x,low_y,dat],F[K,low_x,high_y,dat],F[K,high_x,low_y,dat],F[K,high_x,high_y,dat])
        else:
            raise ValueError( "cannot perform interpolation, a problem occured" )






def bilinear_interpolation(int x1, int x2, int y1, int y2, int x, int y, float f1, float f2, float f3, float f4):
    """Perform a bilinear interpolation between 4 points 
    
    Parameters
    ----------
    x1,x2,y1,y2 :  int
        x-axis and y-axis locations of the 4 points. (ie. location in the frame) 
        (Note that the x axis is vertical and pointing down while the y-axis is horizontal)

    x,y : int
        locations of the target point for the interpolation (in the frame)
       
    f1,f2,f3,f4 : float
        value at each point : f1=f(x1,y1), f2=f(x1,y2), f3=f(x2, y1), f4=f(x2,y2)
       
    Returns
    -------
    the interpolated data (type float)
    
    """
    if x1 == x2:
        if y1 == y2:
            return f1
        else:
            return linear_interpolation(y1,y2,y,f1,f2)
    elif y1 == y2:
        return linear_interpolation(x1,x2,x,f1,f3)
    else:
        return (f1*(x2-x)*(y2-y)+f2*(x2-x)*(y-y1)+f3*(x-x1)*(y2-y)+f4*(x-x1)*(y-y1))/(np.float(x2-x1)*np.float(y2-y1))





def linear_interpolation(int x1, int x2, int x, float f1, float f2):
    """Perform a linear interpolation between 2 points 
    
    Parameters
    ----------
    x1,x2 :  int
        locations of the 2 points. (along any axis) 

    x : int
        locations of the target point for the interpolation (along the same axis as x1 and x2)
       
    f1,f2 : float
        value at each point : f1=f(x1), f2=f(x2)
       
    Returns
    -------
    the interpolated data (type float)
    
    """
    return ((x2-x)/np.float(x2-x1))*f1 + ((x-x1)/np.float(x2-x1))*f2







def F_dichotomy( np.ndarray[DTYPEf_t, ndim=4] F, int K, np.ndarray[DTYPEi_t, ndim=1] N, str side, int pos_now):
    """Look for the position of the vectors at the previous iteration that surround the current point in the fram
    you want to interpolate. 
    
    Parameters
    ----------
    F :  4d np.ndarray
        The main array of the WIDIM algorithm.

    K : int
        the iteration of interest (1st index for F).
    
    N : 1d np.ndarray
        list of the numbers of row or column (depending on the specified value of 'side') for each iteration K

    side : string
        the axis of interest : can be either 'x_axis' or 'y_axis'    

    pos_now : int
        position of the point in the frame (along the axis 'side').
    
    Returns
    -------
    low : int
        largest index at the iteration K along the 'side' axis so that the position of index low in the frame is less than or equal to pos_now.    

    high : int
        smallest index at the iteration K along the 'side' axis so that the position of index low in the frame is greater than or equal to pos_now.                                                        
    
    """
    #print "starting dichotomy"
    cdef int low
    cdef int high
    cdef int minlow = 0
    cdef int maxhigh = N[K]-1
    cdef int searching
    low = np.floor(maxhigh/2)
    high = low + 1
    searching = 1
    if side == 'x_axis':
        while searching:#start dichotomy
            if pos_now == F[K,low,0,0] or (low == 0 and pos_now < F[K,low,0,0]):
                searching = 0
                high = low
            elif pos_now == F[K,high,0,0] or (high == N[K]-1 and pos_now > F[K,high,0,0]):
                searching = 0
                low = high
            elif pos_now > F[K,low,0,0] and pos_now < F[K,high,0,0]:
                searching = 0
            elif pos_now < F[K,low,0,0]:
                maxhigh = low
                low = np.floor((low-minlow)/2)
                high = low + 1
            else:
                minlow=high
                low = low + np.floor((maxhigh-low)/2)
                high = low + 1
        return low, high
    elif side == 'y_axis':
        while searching:#start dichotomy
            if pos_now == F[K,0,low,1] or (low == 0 and pos_now < F[K,0,low,1]):
                searching = 0
                high = low
            elif pos_now == F[K,0,high,1] or (high == N[K]-1 and pos_now > F[K,0,high,1]):
                searching = 0
                low = high
            elif pos_now > F[K,0,low,1] and pos_now < F[K,0,high,1]:
                searching = 0
            elif pos_now < F[K,0,low,1]:
                maxhigh = low
                low = np.floor((low-minlow)/2)
                high = low + 1
            else:
                minlow=high
                low = low + np.floor((maxhigh-low)/2)
                high = low + 1
        return low, high
    else:
        raise ValueError( "no valid side for F-dichotomy!" )







def define_windows( int size ):
    """Define two windows of a given size (trick to allow the use of cdef during an iterative process)
    
    Parameters
    ----------
    size : int
        size of the two windows
       
    Returns
    -------
    window_a, window_b : two 2d np.ndarray of zero (integer)
    
    """
    cdef np.ndarray[DTYPEi_t, ndim=2] window_a = np.zeros([size, size], dtype=DTYPEi)
    cdef np.ndarray[DTYPEi_t, ndim=2] window_b = np.zeros([size, size], dtype=DTYPEi)
    return window_a, window_b






def find_neighbours(int I, int J, int Imax, int Jmax):
    """Find the neighbours of a point I,J in an array of size Imax+1, Jmax+1
    
    Parameters
    ----------
    I,J : int
        indices of the point of interest
       
    Imax,Jmax : int
        max indices for the neighbours (ie. (size of the array) - 1)
       
    Returns
    -------
    neighbours : 2d np.ndarray of size 3*3 
        containing value 1 if neighbour is present, 0 if not. Value is 0 at the center (corresponds to point I,J).
    
    """
    cdef np.ndarray[DTYPEi_t, ndim=2] neighbours = np.zeros([3,3], dtype=DTYPEi)
    cdef int k,l
    for k in range(3):
        for l in range(3):
            neighbours[k,l]=1
    neighbours[1,1]=0
    if I == 0:
        neighbours[0,0]=0
        neighbours[0,1]=0
        neighbours[0,2]=0
    if J == 0:
        neighbours[0,0]=0
        neighbours[1,0]=0
        neighbours[2,0]=0
    if I == Imax:
        neighbours[2,0]=0
        neighbours[2,1]=0
        neighbours[2,2]=0
    if J == Jmax:
        neighbours[0,2]=0
        neighbours[1,2]=0
        neighbours[2,2]=0
    return neighbours








def sumsquare_array(arr1):
    """Compute the sum of the square of the elements of a given array or list
    
    Parameters
    ----------
    arr1 : array or list of any size

    Returns
    -------
    result : float
        = sum( arr1_i * arr1_i)
    
    """
    cdef float result
    cdef I, J    
    result = 0
    for I in range(arr1.shape[0]):
        for J in range(arr1.shape[1]):
            result = result + arr1[I,J]*arr1[I,J]
    return result






def launch( str method, names, arg ):
    """A nice launcher for any openpiv function, printing a header in terminal with a list of the parameters used.
    
    Parameters
    ----------
    method : string
        the name of the algorithm used

    names : list of string
        names of the parameters to print

    arg : list of parameters of different types

    Returns
    -------
    StartTime : float 
        the current time --> can be used to print the execution time of the programm at the end.
    
    """
    cdef int i
    space = [" "," "," "," "," "," "," "," "," "," "," "," "," "," "," "," "," "," "," "," "," "," "," "," "," "," "]
    for i in range(len(space)):
        print space[i]
    print '----------------------------------------------------------'
    print '|----->     ||   The Open Source  P article              |'
    print '| Open      ||                    I mage                 |'              
    print '|     PIV   ||                    V elocimetry  Toolbox  |'                                                  
    print '|     <-----||   www.openpiv.net          version 1.0    |'                
    print '----------------------------------------------------------' 
    print " "
    print "Algorithm : ", method
    print " "
    print "Parameters   "
    print "-----------------------------------"
    for i in range(len(arg)-1):
        print "     ", names[i], " | ", arg[i]
    print "-----------------------------------"
    print "|           STARTING              |"
    print "-----------------------------------"
    cdef float StartTime= time.time()
    return StartTime





def end( float startTime ):
    """A function that prints the time since startTime. Used to end nicely a programm
    
    Parameters
    ----------
    startTime : float
        a time
    
    """
    print "-------------------------------------------------------------"
    print "[DONE] ..after ", (time.time() - startTime), "seconds "
    print "-------------------------------------------------------------"
    



