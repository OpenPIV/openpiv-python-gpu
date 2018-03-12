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

DTYPEf = np.float64
ctypedef np.float64_t DTYPEf_t


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
    frame_a : 2d np.ndarray, dtype=np.int32
        an two dimensions array of integers containing grey levels of 
        the first frame.
        
    frame_b : 2d np.ndarray, dtype=np.int32
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
    
    # Define variables
    cdef DTYPEi_t n_rows, n_cols
    
    assert nfftx == nffty, 'fft x and y dimensions must be same size'
    
    # Get correlation function
    c = CorrelationFunction(frame_a, frame_b, window_size, overlap, nfftx)
    
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
    def __init__(self, frame_a, frame_b, window_size, overlap, nfftx):
        """A class representing a cross correlation function.
        
        NOTE: All identifiers starting with 'd_' exist on the GPU and not the CPU.
        The GPU is referred to as the device, and therefore "d_" signifies that it
        is a device variable. Please adhere to this standard as it makes developing
        and debugging much easier.
        
        Parameters
        ----------
            frame_a, frame_b: 2d arrays - int32
                image pair
            window_size: int
                size of the interrogation window
            overlap: int
                pixel overlap between interrogation windows
        """

        # change image dtype to float32
        # GPU only does single precision computation
        frame_a = frame_a.astype(np.float32)
        frame_b = frame_b.astype(np.float32)
        
        # parameters for correlation calcualtion
        self.shape = frame_a.shape
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
        
        # Return stack of all IW's
        d_winA, d_search_area = self._IWarrange(frame_a, frame_b)
        
        #normalize array
        d_winA, d_search_area = self._normalize_intensity(d_winA, d_search_area)
       
        # zero pad arrays
        d_winA_zp, d_search_area_zp = self._zero_pad(d_winA, d_search_area)
        
        # correlate Windows
        self.data = self._correlate_windows(d_winA_zp, d_search_area_zp)

        # get first peak of correlation function
        self.row, self.col, self.corr_max1 = self._find_peak(self.data)
                
    def _IWarrange(self, frame_a, frame_b):
        """
        Creates a 3D array stack of all of the interrogation windows. 
        This is necessary to do the FFTs all at once on the GPU.

        Parameters
        -----------
        frame_a, frame_b: 2D numpy arrays
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
            __global__ void window_slice(int *frame_a, int *win_a, int window_size, int overlap, int n_col, int w, int batch_size)
        {
            int f_range;
            int w_range;
            int IW_size = window_size*window_size;
            int ind_x = blockIdx.x*blockDim.x + threadIdx.x;
            int ind_y = blockIdx.y*blockDim.y + threadIdx.y;
            int diff = window_size - overlap;
            int i; 

            //loop through each interrogation window
           
            for(i=0; i<batch_size; i++)
            {   
                //indeces of image to map from
                f_range = (i/n_col*diff + ind_y)*w + (i%n_col)*diff + ind_x;
                
                //indeces of new array to map to
                w_range = i*IW_size + window_size*ind_y + ind_x;

                win_a[w_range] = frame_a[f_range];
            }
        }
        """)

        # get field shapes
        w = np.int32(self.shape[1])
        
        # transfer data to GPU
        d_winA = gpuarray.zeros((self.batch_size, self.window_size, self.window_size), np.float32)
        d_search_area = gpuarray.zeros((self.batch_size, self.window_size, self.window_size), np.float32)
        d_frame_a = gpuarray.to_gpu(frame_a)
        d_frame_b = gpuarray.to_gpu(frame_b)
        
        # for debugging
        assert self.window_size >=8, "Window size is too small"
        assert self.window_size%8 == 0, "Window size should be a multiple of 8"
        
        # gpu parameters
        grid_size = int(8)  # I tested a bit and found this number to be fastest.
        block_size = int(self.window_size / grid_size)

        # slice up windows
        window_slice = mod_ws.get_function("window_slice")
        window_slice(d_frame_a, d_winA, self.window_size, self.overlap, self.n_cols, w, self.batch_size, block = (block_size,block_size,1), grid=(grid_size,grid_size) )
        window_slice(d_frame_b, d_search_area, self.window_size, self.overlap, self.n_cols, w, self.batch_size, block = (block_size,block_size,1), grid=(grid_size,grid_size) )
        
        # free GPU memory
        d_frame_a.gpudata.free()
        d_frame_b.gpudata.free()

        return(d_winA, d_search_area)
        
    def _normalize_intensity(self, d_winA, d_search_area):
        """
        Remove the mean from each IW of a 3D stack of IW's
        
        Parameters
        ----------
        d_winA : 3D gpuarray
            stack of first frame IW's
        d_search_area : 3D gpuarray
            stack of second frame IW's
            
        Returns
        -------
        norm : 3D gpuarray
            stack of IW's with mean removed
        """
        
        mod_norm = SourceModule("""
            __global__ void normalize(float *array, float *mean, int IWsize)
        {
            //global thread id for 1D grid of 2D blocks
            int threadId = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
        
            //indeces for mean matrix
            int meanId = threadId / IWsize;
        
            array[threadId] = array[threadId] - mean[meanId];   
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
        normalize(d_winA, d_mean_a, IWsize, block=(block_size, block_size, 1), grid=(grid_size,1))
        normalize(d_search_area, d_mean_b, IWsize, block=(block_size, block_size, 1), grid=(grid_size,1))

        # free GPU memory
        d_mean_a.gpudata.free()
        d_mean_b.gpudata.free()
        
        return(d_winA, d_search_area)
        
    def _zero_pad(self, d_winA, d_search_area):
        """
        Function that zero-pads an 3D stack of arrays for use with the 
        skcuda FFT function.
        
        Parameters
        ----------
        d_winA_norm : 3D gpuarray
            array to be zero padded
        d_search_area_norm : 3D gpuarray
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
        
        # create gpu array
        d_winA_zp = gpuarray.zeros([self.batch_size, self.nfft, self.nfft], dtype = np.float32)
        d_search_area_zp = gpuarray.zeros_like(d_winA_zp)  
        
        #gpu parameters
        grid_size = int(8)
        block_size = int(self.window_size / grid_size)  
        
        # get handle and call function
        zero_pad = mod_zp.get_function('zero_pad')
        zero_pad(d_winA_zp, d_winA, self.nfft, self.window_size, self.batch_size, block=(block_size, block_size,1), grid=(grid_size,grid_size))
        zero_pad(d_search_area_zp, d_search_area, self.nfft, self.window_size, self.batch_size, block=(block_size, block_size,1), grid=(grid_size,grid_size))
        
        #free gpu data
        d_winA.gpudata.free()
        d_search_area.gpudata.free()
        
        return(d_winA_zp, d_search_area_zp)
        
    def _correlate_windows(self, d_winA_zp, d_search_area_zp):
        """Compute correlation function between two interrogation windows.
        
        The correlation function can be computed by using the correlation 
        theorem to speed up the computation.
            
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

        #cast corr and row as a ctype array
        cdef np.ndarray[DTYPEf_t, ndim=3] corr_c = np.array(self.data, dtype = DTYPEf)
        cdef np.ndarray[DTYPEi_t, ndim=1] row_c = np.array(self.row, dtype = DTYPEi)
        cdef np.ndarray[DTYPEi_t, ndim=1] col_c = np.array(self.col, dtype = DTYPEi)
        
        # Move boundary peaks inward one node. Replace later in sig2noise
        row_tmp = np.copy(self.row)
        row_tmp[row_tmp < 1] = 1
        row_tmp[row_tmp > self.nfft - 2] = self.nfft - 2
        col_tmp = np.copy(self.col)
        col_tmp[col_tmp < 1] = 1
        col_tmp[col_tmp > self.nfft - 2] = self.nfft - 2

        # Initialize arrays
        c = corr_c[range(self.batch_size), row_tmp, col_tmp]
        cl = corr_c[range(self.batch_size), row_tmp-1, col_tmp]
        cr = corr_c[range(self.batch_size), row_tmp+1, col_tmp]
        cd = corr_c[range(self.batch_size), row_tmp, col_tmp-1]
        cu = corr_c[range(self.batch_size), row_tmp, col_tmp+1]
       
        # Do subpixel approximation   
        row_sp = row_c + ( (np.log(cl)-np.log(cr) )/( 2*np.log(cl) - 4*np.log(c) + 2*np.log(cr) ))
        col_sp = col_c + ( (np.log(cd)-np.log(cu) )/( 2*np.log(cd) - 4*np.log(c) + 2*np.log(cu) ))
        
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
   
        # get signal to noise ratio
        sig2noise = self.corr_max1/corr_max2
            
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
        
        
        
        

