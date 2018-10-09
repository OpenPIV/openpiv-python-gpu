"""
Launch job on SOSCIP.

process Maria's PIV images from microblower
Use GPU accalerated version of OpenPIV

"""

import multiprocessing
from multiprocessing import Process
import glob, os
import numpy as np
import time

               

t = time.time()
print "\n\nStarting Code"

class GPUMulti(Process):
    def __init__(self, number, index, frame_a, frame_b):
        Process.__init__(self)
        self.number = number
        self.index = index
        self.frame_a = frame_a
        self.frame_b = frame_b

    def run(self):
        thread_gpu(self.number, self.index, self.frame_a, self.frame_b)
        print "\nProcess %d took %d seconds to finish image pair %d!" % (self.number, time.time() - t, self.index)

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
N = len(imB_list)
  
def thread_gpu(gpuid, i, frame_a, frame_b):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuid)
    import openpiv.gpu_process
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

if __name__ == "__main__":
    
    process_list = []
    
    for i in range(100):
        p = GPUMulti(i%4, i, np.load(imA_list[i]).astype(np.int32), np.load(imB_list[i]).astype(np.int32))
        p.start()
