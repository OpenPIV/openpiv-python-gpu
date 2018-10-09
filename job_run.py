"""
Launch job on SOSCIP.

process Maria's PIV images from microblower
Use GPU accalerated version of OpenPIV

"""

import glob, os
import numpy as np
import time
import threading
import openpiv.gpu_process
import pycuda
import pycuda.driver as cuda 

t = time.time()
print "\n\nStarting Code"

#==================================================================
# PARAMETERS FOR OPENPIV
#==================================================================
dt = 5e-6
min_window_size = 16
overlap = 0.50
coarse_factor = 2
nb_iter_max = 3
validation_iter = 1
x_scale = 7.45e-6  # m/pixel 
y_scale = 7.41e-6  # m/pixel

# path to input and output directory
im_dir = "/scratch/p/psulliva/chouvinc/maria_PIV_cont/PIV_Cont_Output/"
out_dir = "/scratch/p/psulliva/chouvinc/maria_PIV_cont/output_data/"

# make sure path is correct
if im_dir[-1] != '':
    im_dir = im_dir + '/'
if out_dir[-1] != '/':
    out_dir = out_dir + '/'

# change pattern to your filename patter
imA_list = sorted(glob.glob(im_dir + "Camera_#0_*.npy"))
imB_list = sorted(glob.glob(im_dir + "Camera_#1_*.npy"))

#=================================================================
# BEGIN CODE
#================================================================

units = ["m", "m", "m/s", "m/s" ]
header = "x [{}],\ty [{}],\tu [{}],\tv [{}],\tmask ".format(units[0], units[1], units[2], units[3])
N = 4
  
def thread_gpu(i):
    im_file1 = imA_list[i]
    im_file2 = imB_list[i]
    frame_a = np.load(im_file1).astype(np.int32)
    frame_b = np.load(im_file2).astype(np.int32) 

    x,y,u,v,mask = openpiv.gpu_process.WiDIM(frame_a, frame_b, np.ones_like(frame_a, dtype=np.int32),
                                             min_window_size,
                                             overlap,
                                             coarse_factor,
                                             dt,
                                             validation_iter=validation_iter,
                                             nb_iter_max = nb_iter_max)
    
    if x_scale != 1.0 or y_scale != 1.0: 
        # scale the data
        x = x*x_scale
        u = u*x_scale
        y = y*y_scale
        v = v*y_scale
    
    # save the data
    if i == 0:
        np.save(out_dir + "x.npy", x)
        np.save(out_dir + "y.npy", y)  

    np.save(out_dir + "u_{:05d}.npy".format(i), u)
    np.save(out_dir + "v_{:05d}.npy".format(i), v)

class gpuThread(threading.Thread):
    def __init__(self, gpuid):
        threading.Thread.__init__(self)
        self.gpuid = gpuid

    def run(self):
        # REMEMBER TO SET THIS TO # OF DEVICES INSTEAD OF HARDCODING
        self.dev = cuda.Device(self.gpuid%N)
        self.ctx = self.dev.make_context()
        print self.ctx.get_device()
        thread_gpu(self.gpuid)
        self.ctx.pop()
        del self.ctx

cuda.init()
numgpus = cuda.Device.count()

for i in range(numgpus):
    gpu_thread = gpuThread(i)
    gpu.start()

print "\nDone Processing data."
