"""

Refactored version of job_run.py with multiprocessing.
We use multiprocessing here because we don't want shared memory,
which is what happens with python's multithreading.

"""

from multiprocessing import Process
from glob import glob
from time import time

import numpy as np
import os

CUDA_STRING = 'CUDA_VISIBLE_DEVICES'

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

# change pattern to your filename pattern
imA_list = sorted(glob.glob(im_dir + "Camera_#0_*.npy"))
imB_list = sorted(glob.glob(im_dir + "Camera_#1_*.npy"))
num_images = len(imB_list)

start_time = time()

class MPGPU(Process):

    def __init__(self, gpuid,
                 process_num, start_index,
                 frame_list_a, frame_list_b):
        Process.__init__(self)
        self.gpuid = gpuid
        self.process_num = process_num
        self.start_index = start_index
        self.frame_list_a = frame_list_a
        self.frame_list_b = frame_list_b
        self.num_images = len(frame_list_a)
        self.exceptions = 0

    def run(self):
        for i in range(self.num_images):
            try:
                frame_a, frame_b = self.load_images(self.frame_list_a[i], self.frame_list_b)
                self.widim_gpu(self.gpuid, self.start_index + i, frame_a, frame_b)
            except:
                print "\n An exception occured!"
                self.exceptions += 1

        print "\nProcess %d took %d seconds to finish image pair %d!" % (self.process_num, time.time() - process_time, self.start_index + i)
        print "\nNumber of exceptions: %d" % self.exceptions

    def widim_gpu(self, gpuid, start_index, frame_a, frame_b):
        os.environ[CUDA_STRING] = str(gpuid)
        # Import after setting device number, since gpu_process has autoinit enabled.
        import openpiv.gpu_process
        x, y, u, v, mask = openpiv.gpu_process.WiDIM(frame_a, frame_b, np.ones_like(frame_a, dtype=np.in32),
                                                     min_window_size,
                                                     overlap,
                                                     coarse_factor,
                                                     dt,
                                                     validation_iter=validation_iter,
                                                     nb_inter_max=nb_iter_max)

        if x_scale != 1.0 or y_scale != 1.0:
            # scale the data
            x = x * x_scale
            u = u * x_scale
            y = y * y_scale
            v = v * y_scale

        # save the data
        if start_index == 0:
            np.save(out_dir + "x.npy", x)
            np.save(out_dir + "y.npy", y)

        np.save(out_dir + "u_{:05}.npy".format(start_index), u[::-1, :])
        np.save(out_dir + "v_{:05}.npy".format(start_index), v[::-1, :])

    @staticmethod
    def load_images(image_a, image_b):
        return np.load(image_a).astype(np.int32), np.load(image_b).astype(np.int32)


if __name__ == "__main__":

    # TODO make these configurable
    num_processes = 20
    num_images = 100 # Remove this if you want to process the entire image set

    partitions = int(num_images/num_processes)

    if num_images % num_processes != 0:
        partitions += 1

    print "\n Partition Size: %d" % partitions

    process_list = []

    for i in range(num_processes):
        # If we go over array bounds, stop spawning new processes
        if i*partitions > num_images:
            break
        start_index = i*partitions
        imA_sublist = imA_list[start_index: start_index + partitions]
        imB_sublist = imB_list[start_index: start_index + partitions]
        p = MPGPU(i%4, i, start_index, imA_sublist, imB_sublist)
        p.start()

    try:
        for process in process_list:
            process.join()
    except KeyboardInterrupt:
        for process in process_list:
            process.terminate()
            process.join()

