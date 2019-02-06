"""
INTERROGATION WINDOW SLICING

Use GPU to efficiently slice images into a 3D array of interrogation windows

"""


from __future__ import print_function


import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import pycuda.driver as drv

import numpy as np

import openpiv.process
import openpiv.tools


"""---Define Code Parameters---"""

DTYPEf = np.float32
DTYPEi = np.int32

def IWarrange_gpu(frame_a, frame_b, window_size, overlap, shift = None):
    """
    GPU window slicing

    Parameters
    ----------
    frame_a, frame_b: 2D numpy arrays
        PIV image pair
    window_size: int
        window size in pixels
    overlap: int
        overlap in number of pixels

    Returns
    -------
    winA_gpu: 3D numpy array
        All frame_a interrogation windows stacked on each other
    search_area_gpu: 3D numpy array
        All frame_b interrogation windows stacked on each other
    """

    #define window slice algorithm
    mod = SourceModule("""
        #include <stdio.h>
         
        __global__ void winSlice(float *frame_a, float *win_a, int window_size, int overlap, int n_col, int w, int batch_size)
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
        
        __global__ void winSlice_shift(float *input, float *output, int *dx, int *dy, int window_size, int overlap, int n_col, int w, int h, int batch_size)
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
                
                //indeces of image to map from. Apply shift to pixels
                f_range = (i/n_col*diff + y_shift)*w + (i%n_col)*diff + x_shift;
                
                // Get values outside window in a sneeky way. This array is 1 if the value is inside the window,
                // and 0 if it is outside the window. Multiply This with the shifted value at end
                int outside_range = ( y_shift >= 0 && y_shift < h && x_shift >= 0 && x_shift < w);
                
                // indeces of image to map to
                w_range = i*IW_size + window_size*ind_y + ind_x;      
                
                // Apply the mapping. Mulitply by outside_range to set values outside the window to zero!
                output[w_range] = input[f_range]*outside_range;
            }
        }
        """)

    #ensure images are correct format
    frame_a = frame_a.astype(np.float32)
    frame_b = frame_b.astype(np.float32)

    #get field shape
    n_row, n_col = np.int32(openpiv.process.get_field_shape(frame_a.shape, window_size, overlap ))
    batch_size = np.int32(n_row*n_col)
    h = np.int32(frame_a.shape[0])
    w = np.int32(frame_a.shape[1])

    # Define GPU data
    d_winA = gpuarray.zeros((batch_size, window_size, window_size), dtype = DTYPEf)
    d_winB = gpuarray.zeros((batch_size, window_size, window_size), dtype = DTYPEf)
    d_frame_a = gpuarray.to_gpu(frame_a)
    d_frame_b = gpuarray.to_gpu(frame_b)
    
    # get shift data
    if(shift is not None):
        dy = shift[0].astype(np.int32)
        dx = shift[1].astype(np.int32)
        d_dx = gpuarray.to_gpu(dx)
        d_dy = gpuarray.to_gpu(dy)

    #gpu parameters
    grid_size = int(8)
    block_size = int(window_size / grid_size) 
    window_size = np.int32(window_size)
    overlap = np.int32(overlap)

    #slice up windows
    winSlice = mod.get_function("winSlice")
    winSlice(d_frame_a, d_winA, window_size, overlap, n_col, w, batch_size, block=(block_size,block_size,1), grid=(grid_size, grid_size))
    
    if(shift is None):
        winSlice(d_frame_b, d_winB, window_size, overlap, n_col, w, batch_size, block=(block_size,block_size,1), grid=(grid_size,grid_size) )
    else:
        winSlice_shift = mod.get_function("winSlice_shift")
        winSlice_shift(d_frame_b, d_winB, d_dx, d_dy, window_size, overlap, n_col, w, h, batch_size, block=(block_size,block_size,1), grid=(grid_size,grid_size) )

    #transfer data back   
    winA = d_winA.get()
    winB = d_winB.get()

    #free device memory
    d_winA.gpudata.free()
    d_winB.gpudata.free()
    d_frame_a.gpudata.free()
    d_frame_b.gpudata.free()
    if shift is not None:
        d_dx.gpudata.free()
        d_dy.gpudata.free()
    
    return(winA, winB)


def IWarrange_cpu(frame_a, frame_b, window_size, overlap):

    """
    OpenPIV CPU Window Slicing algorithm


    """

    #ensure images are correct data type
    frame_a = frame_a.astype(np.int32)
    frame_b = frame_b.astype(np.int32)

    #get field shape
    n_row, n_col = np.int32(openpiv.process.get_field_shape(frame_a.shape, window_size, overlap ))
    batch_size = np.int32(n_row*n_col)
    search_area_size = np.int32(window_size)

    winA_cpu = np.zeros((batch_size, window_size, window_size ), dtype = DTYPEi)
    search_area_cpu = np.zeros((batch_size, search_area_size, search_area_size ), dtype = DTYPEi)

    # loop over the interrogation windows
    # i, j are the row, column indices of the top left corner
    N1 = 0
    for i in range( 0, frame_a.shape[0]-window_size + 1, window_size - overlap ):
        for j in range( 0, frame_a.shape[1]-window_size + 1, window_size -overlap ):

            # get interrogation window matrix from frame a
            for k in range( window_size ):
                for l in range( window_size ):
                    winA_cpu[N1,k,l] = frame_a[i+k, j+l]

            # get search area using frame b
            for k in range( search_area_size ):
                for l in range( search_area_size ):

                    # fill with zeros if we are out of the borders
                    if i+window_size/2-search_area_size/2+k < 0 or i+window_size/2-search_area_size/2+k >= frame_b.shape[0]:
                        search_area_cpu[N1,k,l] = 0
                    elif j+window_size/2-search_area_size/2+l < 0 or j+window_size/2-search_area_size/2+l >= frame_b.shape[1]:
                        search_area_cpu[N1,k,l] = 0
                    else:
                        search_area_cpu[N1,k,l] = frame_b[ i+window_size/2-search_area_size/2+k, j+window_size/2-search_area_size/2+l ]

            #incriment to next IW
            N1 += 1

    return(winA_cpu, search_area_cpu)













