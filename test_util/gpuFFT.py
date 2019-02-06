"""
Demonstrates how to use the PyCUDA interface to CUFFT to compute a
batch of 2D FFTs.
"""
from __future__ import print_function

import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv

import numpy as np
from numpy.fft import fftshift, rfft2, irfft2
import matplotlib.pyplot as plt

import skcuda.fft as cu_fft
import skcuda.misc as cu_misc

import openpiv.process

start = drv.Event()
end = drv.Event()


"""---Code Parameters---"""

DTYPEi = np.int32
DTYPEf = np.float32
subpixel_method = 'gaussian'


def fft_gpu(window_a, search_area):
    """
    Do batch of FFT's on on the Jetson 

    Inputs:
        window_a: 3D numpy array
            stack of interrogation windows of the first frame
            output from the window slice function
        search_area: 3D numpy array
            Stack of interrogation windows of the second frame
            output from the window slice function
    Outputs:
        corr_gpu: 3D numpy array
        Stack of correlation functions for each image pair
    """

    batch_size, win_h, win_w = np.array(window_a.shape).astype(np.int32)
    window_a = window_a.astype(np.float32)
    search_area = search_area.astype(np.float32)

    #allocate space on gpu for FFT's
    #d_winA = drv.mem_alloc(window_a.nbytes)
    #drv.memcpy_htod(d_winA, window_a)
    #d_search_area = drv.mem_alloc(search_area.nbytes)
    #drv.memcpy_htod(d_search_area, search_area)
    
    d_winA = gpuarray.to_gpu(window_a) 
    d_search_area = gpuarray.to_gpu(search_area)

    d_winIFFT = gpuarray.empty_like(d_winA)
    d_winFFT = gpuarray.empty((batch_size, win_h, win_w//2+1), np.complex64)
    d_searchAreaFFT = gpuarray.empty((batch_size, win_h, win_w//2+1), np.complex64)

    #frame a fft
    plan_forward = cu_fft.Plan((win_h, win_w), np.float32, np.complex64, batch = batch_size)
    cu_fft.fft(d_winA, d_winFFT, plan_forward)

    #frame b fft
    cu_fft.fft(d_search_area, d_searchAreaFFT, plan_forward)

    #multiply the ffts
    d_winFFT = d_winFFT.conj()
    d_tmp = cu_misc.multiply(d_searchAreaFFT, d_winFFT)

    #inverse transform
    plan_inverse = cu_fft.Plan((win_h, win_w), np.complex64, np.float32, batch = batch_size)
    cu_fft.ifft(d_tmp, d_winIFFT, plan_inverse, True)

    #transfer data back
    corr_gpu = d_winIFFT.get().real
    corr_gpu = fftshift(corr_gpu, axes = (1,2))

    # Free GPU memory
  
    d_winA.gpudata.free()
    d_search_area.gpudata.free()
    d_winFFT.gpudata.free()
    d_winIFFT.gpudata.free()
    d_searchAreaFFT.gpudata.free()
    d_tmp.gpudata.free()

    return(corr_gpu)


def fft_cpu(window_a, search_area):
    """
    Do batch of FFT's on on the CPU only

    Inputs:
        window_a: 3D numpy array
            stack of interrogation windows of the first frame
            output from the window slice function
        search_area: 3D numpy array
            Stack of interrogation windows of the second frame
            output from the window slice function
    Outputs:
        corr_gpu: 3D numpy array
            Stack of correlation functions for each image pair
    """

    batch_size, win_h, win_w = np.array(window_a.shape).astype(np.int32)
    window_a = window_a.astype(np.float32)
    search_area = search_area.astype(np.float32)

    # preallocate space for data
    winFFT = np.empty([batch_size, window_a.shape[1], window_a.shape[2]//2+1], np.complex64)
    search_areaFFT = np.empty([batch_size, window_a.shape[1], window_a.shape[2]//2+1], np.complex64)
    corr_cpu = np.empty_like(window_a)

    winA = np.zeros([win_h, win_w])
    sa = np.zeros([win_h, win_w])

    for i in range(batch_size):
        winA = window_a[i,:,:]# - np.mean(window_a[i,:,:])
        sa = search_area[i,:,:]# - np.mean(search_area[i,:,:])
        winFFT[i,:,:] = rfft2(winA, s = (win_h, win_w))
        search_areaFFT[i,:,:] = rfft2(sa, s = (win_h, win_w))
        tmp = np.conj(winFFT[i,:,:])*search_areaFFT[i,:,:]
        corr_cpu[i,:,:] = fftshift(irfft2(tmp).real, axes = (0,1))

    """
    end.record()
    end.synchronize()
    time = start.time_till(end)*1e-3
    print("CPU time is: %fs sec" %(time))

    print("CPU corr shape is: %s" %(corr_cpu.shape,))
    print("GPU corr shape is: %s" %(corr_gpu.shape,))
    print('Success status: ', np.allclose(corr_gpu, corr_cpu, atol=1e-1))
    """

    return(corr_cpu)



