"""

Refactored version of job_run.py with multiprocessing.
We use multiprocessing here because we don't want shared memory,
which is what happens with python's multithreading.

"""

from __future__ import division

from multiprocessing import Process
from time import time
from skimage import io
from piv_tools import contrast

import numpy as np
import os

import glob
import sys

# import matplotlib.pyplot as plt
import scipy.interpolate as interp

if sys.version_info[0] == 3:
    print("This system is using Python 3, but OpenPIV requires Python 2")

# OpenPIV requires Python 2
if sys.version_info[0] == 2:
    import openpiv.filters

#import traceback
#import warnings
#import sys

# This function just gives numpy-defined warnings a stack-trace.
#def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

 #   log = file if hasattr(file,'write') else sys.stderr
  #  traceback.print_stack(file=log)
   # log.write(warnings.formatwarning(message, category, filename, lineno, line))

#warnings.showwarning = warn_with_traceback

# TODO -- separate the functions into separate module
class MPGPU(Process):

    # Keep all properties that belong to an individual openpiv function within the properties dict
    # to keep the responsibilities of this class clear (multiprocessing, not keeping track of parameters)
    def __init__(self, gpuid,
                 process_num, start_index,
                 frame_list_a, frame_list_b,
                 properties):
        Process.__init__(self)
        self.gpuid = gpuid
        self.process_num = process_num
        self.start_index = start_index
        self.frame_list_a = frame_list_a
        self.frame_list_b = frame_list_b
        self.num_images = len(frame_list_a)
        self.properties = properties
        self.exceptions = 0

    def run(self):
        process_time = time()
        func = self.properties["gpu_func"]

        for i in range(self.num_images):
            frame_a, frame_b = self.frame_list_a[i], self.frame_list_b[i]    
            #try:
            func(self.start_index + i, frame_a, frame_b, self.properties, gpuid=self.gpuid)
            #except Exception as e:
             #   print "\n An exception occurred! %s" % e
              #  print sys.exc_info()[2].tb_lineno
               # self.exceptions += 1

        print "\nProcess %d took %d seconds to finish %d image pairs (%d to %d)!" % (self.process_num,
                                                                                     time() - process_time,
                                                                                     self.num_images,
                                                                                     self.start_index,
                                                                                     self.start_index + self.num_images)
        print "\nNumber of exceptions: %d" % self.exceptions

    @staticmethod
    def load_images(image_a, image_b):
        return np.load(image_a).astype(np.int32), np.load(image_b).astype(np.int32)


def parallelize(num_items, num_processes, list_tuple, properties):
    # Properties contains data relevant to a particular openpiv function

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
        subList_A = list_tuple[0][start_index: start_index + partitions]
        subList_B = list_tuple[1][start_index: start_index + partitions]
        process = MPGPU(i%4, i, start_index, subList_A, subList_B, properties)
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


# self.start_index + i, frame_a, frame_b, properties, gpuid (named)
def replace_outliers(image_pair_num, u_file, v_file, properties, gpuid=0):
    """
    This function first loads all the output data from the output directory
    and applies the mask. All masked elements are assign NaN. A single pair of
    output files is then passed to the function "outlier_detection" where the
    outliers are identified and later replaced using openpiv.filters
    """

    # TODO: just realized **kwargs is a thing, so need to change to that after.
    output_dir = properties["out_dir"]
    mask = properties["mask"]
    r_thresh = properties["r_thresh"]

    u = np.load(u_file)
    v = np.load(v_file)

    u[mask] = np.nan
    v[mask] = np.nan

    # call outlier_detection (which replaces the outliers)
    u_out, v_out = outlier_detection(u, v, r_thresh, mask=mask)

    # save to the replacement directory
    np.save(output_dir + "u_repout_{:05d}.npy".format(image_pair_num), u_out)
    np.save(output_dir + "v_repout_{:05d}.npy".format(image_pair_num), v_out)


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
    y_r = np.load(data_dir + "y.npy")

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
    # if plot:
    #     mask_plot = mask.astype(float)
    #     mask_plot[mask] = np.nan
    #     mask_int_plot = mask_int.astype(float) + 10
    #     mask_int_plot[mask_int] = np.nan
    #     plt.pcolormesh(x_r, y_r, mask_int_plot, cmap="jet")
    #     plt.pcolormesh(x_pix, y_pix, mask_plot, cmap="jet")
    #     plt.colorbar()
    #     plt.show()

    return mask_int

# self.start_index + i, frame_a, frame_b, self.properties, gpuid=self.gpuid
def histogram_adjust(start_index, frame_a_file, frame_b_file, properties, gpuid=0):
    frame_a = np.load(frame_a_file).astype(np.int32)
    frame_b = np.load(frame_b_file).astype(np.int32)

    # FOR TESTING ONLY -- REMOVE ONCE WE FIGURE OUT WHAT'S ACCEPTABLE
    file = open("percentages.txt", "a+")

    pixels_a_old = frame_a.flatten()
    pixels_b_old = frame_b.flatten()
    # % Difference between the two images in pixel sum based on the first image
    p_deviation = np.sum(frame_a.flatten() - frame_b.flatten())/np.sum(pixels_a_old)

    frame_a = contrast(frame_a, r_low = 60, r_high = 90)
    frame_b = contrast(frame_b, r_low = 60, r_high = 90)

    # % Difference in pixel weight between the old and new image a
    pixels_a_new = frame_a.flatten()
    pixels_b_new = frame_b.flatten()

    a_deviation = np.sum(pixels_a_new - pixels_a_old)/np.sum(pixels_a_new)
    b_deviation = np.sum(pixels_b_new - pixels_b_old)/np.sum(pixels_b_new)

    file_string = "Image number %d: P = %f, A = %f, B = %f" % (start_index, p_deviation, a_deviation, b_deviation)
    file.write(file_string)
    file.write('\n')


def widim_gpu(start_index, frame_a_file, frame_b_file, properties, gpuid=0):

    # TODO -- Decouple these parameters from the functions below and pass them in
    # ==================================================================
    # PARAMETERS FOR OPENPIV
    # ==================================================================
    dt = 5e-6
    min_window_size = 16
    overlap = 0.50
    coarse_factor = 2
    nb_iter_max = 3
    validation_iter = 1
    x_scale = 7.45e-6  # m/pixel
    y_scale = 7.41e-6  # m/pixel
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuid)
    out_dir = properties["out_dir"]

    frame_a = np.load(frame_a_file).astype(np.int32)
    frame_b = np.load(frame_b_file).astype(np.int32)

    # Import after setting device number, since gpu_process has autoinit enabled.
    import openpiv.gpu_process
    x, y, u, v, mask = openpiv.gpu_process.WiDIM(frame_a, frame_b, np.ones_like(frame_a, dtype=np.int32),
                                                 min_window_size,
                                                 overlap,
                                                 coarse_factor,
                                                 dt,
                                                 validation_iter=validation_iter,
                                                 nb_iter_max=nb_iter_max)

    if x_scale != 1.0 or y_scale != 1.0:
        # scale the data
        x = x * x_scale
        u = u * x_scale
        y = y * y_scale
        v = v * y_scale

    # save the data
    if start_index == 0:
        np.save(out_dir + "x.npy", x)
        np.save(out_dir + "y.npy", y[::-1, 0])

    # Note: we're reversing u and v here only because the camera input is inverted. If the camera ever takes
    # images in the correct orientations, we'll have to remove u[::-1, :].
    np.save(out_dir + "u_{:05}.npy".format(start_index), u[::-1, :])
    np.save(out_dir + "v_{:05}.npy".format(start_index), v[::-1, :])


def get_input_files(directory, file_name_pattern):
    # get the images (either .tif or npy), converting to npy if .tif

    file_list = sorted(glob.glob(directory + file_name_pattern))
    
    # if list is empty, files could be in tif format
    if not file_list:
        new_file_pattern = os.path.splitext(file_name_pattern)[0] + '.tif'
        file_list = sorted(glob.glob(directory + new_file_pattern))        
        
        prefix = file_name_pattern.split("*")[0]
        tif_to_npy(directory, prefix, file_list)
        file_list = sorted(glob.glob(directory + file_name_pattern))

    return file_list
        
def tif_to_npy(out_dir, prefix, file_list):
    for i in range(len(file_list)):
        file_read = io.imread(file_list[i])
        np.save(out_dir + prefix + "{:05}.npy".format(i), file_read)


if __name__ == "__main__":

    # path to input and output directory
    im_dir = "/scratch/p/psulliva/chouvinc/maria_PIV_cont/PIV_Cont_Output/"
    out_dir = "/scratch/p/psulliva/chouvinc/maria_PIV_cont/output_data2/"
    rep_dir = "/scratch/p/psulliva/chouvinc/maria_PIV_cont/replaced_data2/"

    # make sure path is correct
    if im_dir[-1] != '/':
        im_dir = im_dir + '/'
    if out_dir[-1] != '/':
        out_dir = out_dir + '/'

    # change pattern to your filename pattern
    imA_list = get_input_files(im_dir, "Camera_#0_*.npy")
    imB_list = get_input_files(im_dir, "Camera_#1_*.npy")
    num_images = len(imB_list)

    # TODO make these configurable
    num_processes = 20 
    num_images = 100  # Remove this if you want to process the entire image set

    # Pre-processing contrast
    contrast_properties = {"gpu_func": histogram_adjust, "out_dir": im_dir}
    parallelize(num_images, num_processes, (imA_list, imB_list), contrast_properties)

    # The images are adjusted, now refresh to include them in our lists
    imA_list = sorted(glob.glob(im_dir + "Camera_#0_*.npy"))
    imB_list = sorted(glob.glob(im_dir + "Camera_#1_*.npy"))

    # Pre-processing contrast
    contrast_properties = {"gpu_func": histogram_adjust, "out_dir": im_dir}
    parallelize(num_images, num_processes, (imA_list, imB_list), contrast_properties)

    # The images are adjusted, now refresh to include them in our lists
    imA_list = sorted(glob.glob(im_dir + "Camera_#0_*.npy"))
    imB_list = sorted(glob.glob(im_dir + "Camera_#1_*.npy"))

    # Processing images
    widim_properties = {"gpu_func": widim_gpu, "out_dir": out_dir}
    parallelize(num_images, num_processes, (imA_list, imB_list), widim_properties)

    # Post-processing
    # > Replace outliers
    # TODO: rename directories to say what data they actually contain (eg. out_dir --> vectors)
    u_list = get_input_files(out_dir, "u*.npy")
    v_list = get_input_files(out_dir, "v*.npy")

    # Interpolate the mask onto the PIV grid
    if "mask_int" not in locals():
        mask_int = interp_mask(mask, out_dir + "/", exp=2)

    routliers_properties = {
        "gpu_func": replace_outliers, "out_dir": rep_dir,
        "mask": mask_int, "r_thresh": r_thresh
        }
    parallelize(num_images, num_processes, (u_list, v_list), routliers_properties)
