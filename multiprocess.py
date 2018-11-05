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


def parallelize(num_items, num_processes, list_tuple):
    partitions = int(num_items/num_processes)

    if num_items % num_processes != 0:
        partitions += 1

    print "\n Partitions Size: %d" % partitions

    process_list = []

    for i in range(num_items):
        # If we go over array bounds, stop spawning new processes
        if i*partitions > num_items:
            break
        start_index = i*partitions
        subList_A = list_tuple(0)[start_index: start_index + partitions]
        subList_B = list_tuple(1)[start_index: start_index + partitions]
        process = MGPU(i%4, i, start_index, subList_A, subList_B)
        process.start()
        process_list.append(process)

    # Cleanup
    try:
        for process in process_list:
            process.join()
    except KeyboardInterrupt:
        for process in process_list:
            process.terminate()
            process.join()


if __name__ == "__main__":

    # TODO make these configurable
    num_processes = 20
    num_images = 100 # Remove this if you want to process the entire image set

    # Processing images
    parallelize(num_images, num_processes, (imA_list, imB_list))

    # Post-processing
    # > Replace outliers
    parallelize(num_images)

# ===============================================================================
# IMPORT MODULES
# ===============================================================================

from __future__ import division

import glob
import sys

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp

if sys.version_info[0] == 3:
    print("This system is using Python 3, but OpenPIV requires Python 2")

# OpenPIV requires Python 2
if sys.version_info[0] == 2:
    import openpiv.filters

# ===============================================================================
# DEFINE VARIABLES FOR DIRECTORIES
# ===============================================================================

"""
Could directories be created in the project directory in the scratch directory 
when the code is run based on what window size is selected? So there would only
be two directories here for a single window size.

i.e. if the minimum window size is set to 16, then the GPU code would create
the directory "output_16" for the regular data and "replace_16" for the data 
where the outliers are replaced
"""

out16_dir = "multi_16/"
out32_dir = "output_32/"
out64_dir = "output_64/"
rep32_dir = "replace_32/"
rep64_dir = "rep64_test/"
# rep64_dir = "replace_64/"

# ===============================================================================
# Assign which data you want to process
# ===============================================================================

"""
This could be made to be based on the minimum window size chosen - the analysis
directory would be the output directory (output_XX) and the replacement 
directory would be replace_XX. NOTE that the replacement can happen only after
the regular output data has been processed. 
"""

analysis_dir = out64_dir
rep_dir = rep64_dir

# ===============================================================================
# Other parameters
# ===============================================================================

"""
The mask is created in ImageJ based on the raw images. Therefore, the mask is
also flipped across the x-axis. Since the data output from the GPU code (in 
output_XX) is now flipped to the correct orientation, the mask must also be 
flipped to the correct orientation.

Mask can be placed in the project directory in the scratch directory - copy from
local machine onto soscip
"""
# load the mask so that it is flipped in the correct orientation
mask = np.load("mask.npy")[::-1, :]

# Some threshold value for replacing spurious vectors
r_thresh = 2.0

# Number of image pairs / files
num_files = 5000


# ===============================================================================
# FUNCTION DEFINITIONS
# ===============================================================================

def outlier_detection(u, v, r_thresh, mask=None, max_iter=2):
    """
    Outlier detection

    A single pair of output files is taken by this function and the function
    goes through each element of the arrays one by one. The median value of
    all the surrounding elements is taken (not including the element under
    analysis) and the median difference between the surrounding elements and
    the median value is calculated. If the difference between the element under
    analysis and the median value of the surrounding elements is greater than
    the threshold, then the element is an outlier and assigned NaN. After this
    is done for the entire array (u and v), the mask is applied to the array
    and all masked elements are assigned a value of 0. Note that only the
    outliers are assigned Nan. The arrays are then passed into
    openpiv.filters.replace_outliers which replaces all the NaN elements to be
    an average of the surrounding elements
    """

    u_out = np.copy(u).astype(float)
    v_out = np.copy(v).astype(float)

    for n in range(max_iter):
        for i in range(u.shape[0]):
            for j in range(u.shape[1]):

                # check that the element is not a masked element (NaN)
                fin_u = np.isfinite(u_out[i, j])

                if fin_u:

                    if i == 0 and j == 0:
                        # top left
                        Ui = np.delete(u_out[:2, :2].flatten(), 0)
                    elif i == 0 and j == u.shape[1] - 1:
                        # top right
                        Ui = np.delete(u_out[:2, -2:].flatten(), 1)
                    elif i == u.shape[0] - 1 and j == 0:
                        # bottom left
                        Ui = np.delete(u_out[-2:, :2].flatten(), 2)
                    elif i == u.shape[0] - 1 and j == u.shape[1] - 1:
                        # bottom right
                        Ui = np.delete(u_out[-2:, -2:].flatten(), 3)
                    elif i == 0 and j > 0 and j < u.shape[1] - 1:
                        # top boundary
                        Ui = np.delete(u_out[:2, j - 1:j + 2].flatten(), 1)
                    elif i == u.shape[0] - 1 and j > 0 and j < u.shape[1] - 1:
                        # bottom boundary
                        Ui = np.delete(u_out[-2:, j - 1:j + 2].flatten(), 4)
                    elif i > 0 and i < u.shape[0] and j == 0:
                        # left boundary
                        Ui = np.delete(u_out[i - 1:i + 2, :2].flatten(), 2)
                    elif i > 0 and i < u.shape[0] - 1 and j == 0:
                        # right boundary
                        Ui = np.delete(u_out[i - 1:i + 2, -2:].flatten(), 3)
                    else:
                        # interior grid
                        Ui = np.delete(u_out[i - 1:i + 2, j - 1:j + 2].flatten(), 4)

                    Um = np.nanmedian(Ui)
                    rm = np.nanmedian(np.abs(Ui - Um))
                    ru0 = np.abs(u_out[i, j] - Um) / (rm + 0.1)

                    if ru0 > r_thresh:
                        u_out[i, j] = np.nan
                    if not np.isfinite(ru0):
                        u_out[i, j] = np.nan

                # check that the element is not a masked element (NaN)
                fin_v = np.isfinite(v_out[i, j])

                if fin_v:

                    if i == 0 and j == 0:
                        # top left
                        Vi = np.delete(v_out[:2, :2].flatten(), 0)
                    elif i == 0 and j == u.shape[1] - 1:
                        # top right
                        Vi = np.delete(v_out[:2, -2:].flatten(), 1)
                    elif i == u.shape[0] - 1 and j == 0:
                        # bottom left
                        Vi = np.delete(v_out[-2:, :2].flatten(), 2)
                    elif i == u.shape[0] - 1 and j == u.shape[1] - 1:
                        # bottom right
                        Vi = np.delete(v_out[-2:, -2:].flatten(), 3)
                    elif i == 0 and j > 0 and j < u.shape[1] - 1:
                        # top boundary
                        Vi = np.delete(v_out[:2, j - 1:j + 2].flatten(), 1)
                    elif i == u.shape[0] - 1 and j > 0 and j < u.shape[1] - 1:
                        # bottom boundary
                        Vi = np.delete(v_out[-2:, j - 1:j + 2].flatten(), 4)
                    elif i > 0 and i < u.shape[0] and j == 0:
                        # left boundary
                        Vi = np.delete(v_out[i - 1:i + 2, :2].flatten(), 2)
                    elif i > 0 and i < u.shape[0] - 1 and j == 0:
                        # right boundary
                        Vi = np.delete(v_out[i - 1:i + 2, -2:].flatten(), 3)
                    else:
                        Vi = np.delete(v_out[i - 1:i + 2, j - 1:j + 2].flatten(), 4)

                    Vm = np.nanmedian(Vi)
                    rm = np.nanmedian(np.abs(Vi - Vm))
                    rv0 = np.abs(v_out[i, j] - Vm) / (rm + 0.1)

                    if rv0 > r_thresh:
                        v_out[i, j] = np.nan
                    if not np.isfinite(rv0):
                        v_out[i, j] = np.nan

    # set all masked elements to zero so they are not replaced in openpiv.filters
    if mask is not None:
        u_out[mask] = 0.0
        v_out[mask] = 0.0

    print("Number of u outliers: {}".format(np.sum(np.isnan(u_out))))
    print("Percentage: {}".format(np.sum(np.isnan(u_out)) / u.size * 100))
    print("Number of v outliers: {}".format(np.sum(np.isnan(v_out))))
    print("Percentage: {}".format(np.sum(np.isnan(v_out)) / v.size * 100))

    print("Replacing Outliers")
    u_out, v_out = openpiv.filters.replace_outliers(u_out, v_out)

    return (u_out, v_out)


def replace_outliers(data_dir, output_dir, mask, r_thresh):
    """
    This function first loads all the output data from the output directory
    and applies the mask. All masked elements are assign NaN. A single pair of
    output files is then passed to the function "outlier_detection" where the
    outliers are identified and later replaced using openpiv.filters
    """

    # load all output files
    u_list = sorted(glob.glob(data_dir + "u*.npy"))
    v_list = sorted(glob.glob(data_dir + "v*.npy"))

    for i in range(len(u_list)):
        print("processing file {} of {}".format(i, len(u_list)))

        # load one pair of output files at a time
        u = np.load(u_list[i])
        v = np.load(v_list[i])

        # flip the data - NO NEED IF THE OUTPUT DATA IS IN THE RIGHT ORINETATION
        # u = u[::-1,:]
        # v = v[::-1,:]

        # apply mask - MASK IS LOADED IN CORRECT ORIENTATION
        u[mask] = np.nan
        v[mask] = np.nan

        # call outlier_detection (which replaces the outliers)
        u_out, v_out = outlier_detection(u, v, r_thresh, mask=mask)

        # save to the replacement directory
        np.save(output_dir + "u_repout_{:05d}.npy".format(i), u_out)
        np.save(output_dir + "v_repout_{:05d}.npy".format(i), v_out)


def interp_mask(mask, data_dir, exp=0, plot=False):
    """
    Interpolate the mask onto the output data. The mask has dimensions 996x1296
    while the the output data dimensions are much smaller (and depend on the
    minimum window size chosen)
    """

    # load the x and y location arrays from the output directory
    """
    SAVE y.npy IN THE GPU CODE SO THAT IT IS FLIPPED IN THE CORRECT ORIENTATION
    SO IT SHOULD NOT HAVE TO BE FLIPPED AGAIN HERE
    """
    x_r = np.load(data_dir + "x.npy")[0, :]
    y_r = np.load(data_dir + "y.npy")[::-1, 0]  # change this when GPU code is changed

    x_pix = np.linspace(x_r[0], x_r[-1], len(mask[0, :]))
    y_pix = np.linspace(y_r.min(), y_r.max(), len(mask[:, 0]))

    mask_int = np.empty([y_r.size, x_r.size])

    f = interp.interp2d(x_pix, y_pix, mask.astype(float))
    mask_int = f(x_r, y_r)
    mask_int = np.array(mask_int > 0.9)

    # expand the mask
    if exp > 0:
        for i in range(x_r.size):
            valid = np.where(mask_int[:, i] == 0)[0]
            if valid.size == 0:
                continue
            low = valid[0]
            high = valid[-1]
            mask_int[low - exp:low, i] = 0
            mask_int[high:high + exp + 1, i] = 0

    # plot some shit
    if plot:
        mask_plot = mask.astype(float)
        mask_plot[mask] = np.nan
        mask_int_plot = mask_int.astype(float) + 10
        mask_int_plot[mask_int] = np.nan
        plt.pcolormesh(x_r, y_r, mask_int_plot, cmap="jet")
        plt.pcolormesh(x_pix, y_pix, mask_plot, cmap="jet")
        plt.colorbar()
        plt.show()

    return (mask_int)


# ===============================================================================
# FUNCTION CALLS
# ===============================================================================

if __name__ == "__main__":

    # Interpolate the mask onto the PIV grid
    if "mask_int" not in locals():
        mask_int = interp_mask(mask, analysis_dir + "raw_data/", exp=2)

    # Replace outliers
    replace_outliers(analysis_dir + "raw_data/", rep_dir, mask_int, r_thresh)