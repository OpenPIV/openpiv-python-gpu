"""This module performs multiprocessing of the OpenPIV GPU algorithms.

Current state of the document - 7 July 2020
-----------------------------

All non-core multiprocessing features have been commented out. This was done to make the module more generally
applicable outside the Turbulence Research group. The remaining code is focused on multiprocessing user-defined
functions with up two lists of data. The commented-out code will not work without modification.

In the future the obsolete code fragments will be deleted. The number of physical GPUs also needs to be detected so
that the user won't need to input it.

"""
from multiprocessing import Process, Manager, cpu_count
import numpy as np
from math import ceil, floor
from time import time
from skimage import io
from datetime import datetime
from PIL import Image
import os
import functools
import glob
import sys
import scipy.interpolate as interp
import openpiv.filters


# ===================================================================================================================
# WARNING: File read/close is UNSAFE in multiprocessing applications because multiple
# threads are accessing &/or writing to the same file. Please remember to use a queue if doing file I/O concurrently
# ===================================================================================================================


# ===================================================================================================================
# MULTIPROCESSING UTILITY CLASSES & FUNCTIONS
# ===================================================================================================================
class MPGPU(Process):
    """Multiprocessing class for OpenPIV processing algorithms

    Each instance of this class is a process for some OpenPIV algorithm.

    Parameters
    ----------
    func : function
        GPU to use
    items : tuple
        list of items to process
    gpuid : int
        which GPU to use for processing
    index : int
        beginning index number of items to process

    """

    # Keep all properties that belong to an individual openpiv function within the properties dict to keep the
    # responsibilities of this class clear (multiprocessing, not keeping track of parameters)
    def __init__(self, func, *items, gpuid=None, index=None):
        Process.__init__(self)
        self.func = func
        self.gpuid = gpuid
        self.items = items
        self.index = index
        self.num_items = len(items[0])

        if gpuid is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuid)

    def run(self):
        # process_time = time()
        # func = self.properties["gpu_func"]

        # set the starting index
        index = self.index

        for i in range(self.num_items):
            # run the function
            if self.items is not None:
                if self.index is not None:
                    self.func(*self.items, index)
                else:
                    self.func(*self.items)
            else:
                if self.index is not None:
                    self.func()

            index += 1

            # func(self.start_index + i, frame_a, frame_b, self.properties, gpuid=self.gpuid)

        # for i in range(self.num_items):
        #     frame_a, frame_b = self.frame_list_a[i], self.frame_list_b[i]
        #     try:
        #         func(self.start_index + i, frame_a, frame_b, self.properties, gpuid=self.gpuid)
        #     except Exception as e:
        #         print("\n An exception occurred! %s" % e)
        #         print(sys.exc_info()[2].tb_lineno)
        #         self.exceptions += 1

        # print("\nProcess %d took %d seconds to finish %d image pairs (%d to %d)!" % (self.process_num,
        #                                                                              time() - process_time,
        #                                                                              self.num_images,
        #                                                                              self.start_index,
        #                                                                              self.start_index + self.num_images))
        # print("\nNumber of exceptions: %d" % self.exceptions)

    # @staticmethod
    # def load_images(image_a, image_b):
    #     return np.load(image_a).astype(np.int32), np.load(image_b).astype(np.int32)


def parallelize(func, *items, num_processes=None, num_gpus=None, index=None):
    """Parallelizes OpenPIV algorithms

    This helper function spawns instances of the class MGPU to multiprocess up to two sets of corresponding items. It
    assumes that each physical GPU will handle one process. The arguments for func are optional and must only be
    list_a, list_b and index.

    Parameters
    ----------
    func : function
        user-defined function to parallelize
    items: tuple
        1D lists of the items to process
    num_processes : int
        number of parallel processes to run. This may exceed the number of CPU cores, but will not speed up processing.
    num_gpus : int
        number of physical GPUs to use for multiprocessing. This will cause errors if the larger than number of GPUs.
    index : index
        whether to pass the user-defined function an index of the items processed

    """
    # check that the lists provided are the same dimension
    if items is not None:
        assert all([len(item_a) == len(item_b) for item_a in items for item_b in items]), \
            'Input item lists are different lengths. len(items) = {}'.format([len(item) for item in items])

    # default to a number of cores equal to 37.5% or fewer of the available CPU cores (75% of physical cores)
    if num_processes is None:
        num_processes = max(floor(cpu_count() * 0.75), 1)  # minimum 1 in case of low-spec machine

    # size of each partition is computed
    if items[0] is not None:
        partition_size = ceil(len(items[0]) / num_processes)
    else:
        partition_size = None

    # print information about the multiprocessing
    print('Multiprocessing: Number of processes requested =', num_processes,
          '. Number of CPU cores available =', cpu_count())
    print('Multiprocessing: Number of physical GPUs to use =', num_gpus, '. Number of GPUs available =', 'unknown')
    print('Multiprocessing: Size of each partition =', partition_size)

    process_list = []
    gpuid = None

    # loop through each partition to spawn processes
    i = 0  # number of processes spawned
    while True:
        # If we go over array bounds, stop spawning new processes
        start_index = i * partition_size

        # The items to process are divided into partitions
        if items is not None:
            sublist = [[]] * len(items)
            for i in range(len(items)):
                sublist[i] = items[i][start_index: start_index + partition_size]
        else:
            sublist = None

        # determine which GPU to use, if any
        if num_gpus is not None:
            gpuid = i % num_gpus

        # spawn the process
        if index is not None:
            process = MPGPU(func, *sublist, gpuid=gpuid, index=start_index + index)
        else:
            process = MPGPU(func, *sublist, gpuid=gpuid)
        process.start()
        process_list.append(process)

        i += 1

        # check to see if process stops
        if items is not None:
            if i * partition_size >= len(items[0]):
                break
        else:
            if i == num_processes:
                break

    # join the processes to finish the multiprocessing
    for process in process_list:
        process.join()

    # Cleanup  # delete
    # try:
    #     for process in process_list:
    #         process.join()
    # except KeyboardInterrupt:
    #     for process in process_list:
    #         process.terminate()
    #         process.join()

# ===============================================================================
# MULTIPROCESSED FUNCTIONS
# ===============================================================================
# def outlier_detection(u, v, r_thresh=2.0, mask=None, max_iter=2):
#     """Outlier detection on computed PIV fields
#
#     A single pair of output files is taken by this function, and it goes through each element of the arrays
#     one by one. The median value of all the surrounding elements is taken (not including the element under analysis)
#     and the median difference between the surrounding elements and the median value is calculated. If the difference
#     between the element under analysis and the median value of the surrounding elements is greater than the
#     threshold, then the element is an outlier and assigned NaN. After this is done for the entire array (u and v),
#     the mask is applied to the array and all masked elements are assigned a value of 0. Note that only the outliers
#     are assigned Nan. The arrays are then passed into openpiv.filters.replace_outliers which replaces all the NaN
#     elements to be an average of the surrounding elements
#
#     Parameters
#     ----------
#     u, v : ndarray
#         velocity fields
#     r_thresh : float
#         threshold for validation
#     mask : ndarray
#         mask applied to PIV images
#     max_iter
#         maximum validation iterations
#
#     """
#     u_out = np.copy(u).astype(float)
#     v_out = np.copy(v).astype(float)
#
#     for n in range(max_iter):
#         for i in range(u.shape[0]):
#             for j in range(u.shape[1]):
#
#                 # check that the element is not a masked element (NaN)
#                 fin_u = np.isfinite(u_out[i, j])
#
#                 if fin_u:
#
#                     if i == 0 and j == 0:
#                         # top left
#                         Ui = np.delete(u_out[:2, :2].flatten(), 0)
#                     elif i == 0 and j == u.shape[1] - 1:
#                         # top right
#                         Ui = np.delete(u_out[:2, -2:].flatten(), 1)
#                     elif i == u.shape[0] - 1 and j == 0:
#                         # bottom left
#                         Ui = np.delete(u_out[-2:, :2].flatten(), 2)
#                     elif i == u.shape[0] - 1 and j == u.shape[1] - 1:
#                         # bottom right
#                         Ui = np.delete(u_out[-2:, -2:].flatten(), 3)
#                     elif i == 0 and j > 0 and j < u.shape[1] - 1:
#                         # top boundary
#                         Ui = np.delete(u_out[:2, j - 1:j + 2].flatten(), 1)
#                     elif i == u.shape[0] - 1 and j > 0 and j < u.shape[1] - 1:
#                         # bottom boundary
#                         Ui = np.delete(u_out[-2:, j - 1:j + 2].flatten(), 4)
#                     elif i > 0 and i < u.shape[0] and j == 0:
#                         # left boundary
#                         Ui = np.delete(u_out[i - 1:i + 2, :2].flatten(), 2)
#                     elif i > 0 and i < u.shape[0] - 1 and j == 0:
#                         # right boundary
#                         Ui = np.delete(u_out[i - 1:i + 2, -2:].flatten(), 3)
#                     else:
#                         # interior grid
#                         Ui = np.delete(u_out[i - 1:i + 2, j - 1:j + 2].flatten(), 4)
#
#                     Um = np.nanmedian(Ui)
#                     rm = np.nanmedian(np.abs(Ui - Um))
#                     ru0 = np.abs(u_out[i, j] - Um) / (rm + 0.1)
#
#                     if ru0 > r_thresh:
#                         u_out[i, j] = np.nan
#                     if not np.isfinite(ru0):
#                         u_out[i, j] = np.nan
#
#                 # check that the element is not a masked element (NaN)
#                 fin_v = np.isfinite(v_out[i, j])
#
#                 if fin_v:
#
#                     if i == 0 and j == 0:
#                         # top left
#                         Vi = np.delete(v_out[:2, :2].flatten(), 0)
#                     elif i == 0 and j == u.shape[1] - 1:
#                         # top right
#                         Vi = np.delete(v_out[:2, -2:].flatten(), 1)
#                     elif i == u.shape[0] - 1 and j == 0:
#                         # bottom left
#                         Vi = np.delete(v_out[-2:, :2].flatten(), 2)
#                     elif i == u.shape[0] - 1 and j == u.shape[1] - 1:
#                         # bottom right
#                         Vi = np.delete(v_out[-2:, -2:].flatten(), 3)
#                     elif i == 0 and j > 0 and j < u.shape[1] - 1:
#                         # top boundary
#                         Vi = np.delete(v_out[:2, j - 1:j + 2].flatten(), 1)
#                     elif i == u.shape[0] - 1 and j > 0 and j < u.shape[1] - 1:
#                         # bottom boundary
#                         Vi = np.delete(v_out[-2:, j - 1:j + 2].flatten(), 4)
#                     elif i > 0 and i < u.shape[0] and j == 0:
#                         # left boundary
#                         Vi = np.delete(v_out[i - 1:i + 2, :2].flatten(), 2)
#                     elif i > 0 and i < u.shape[0] - 1 and j == 0:
#                         # right boundary
#                         Vi = np.delete(v_out[i - 1:i + 2, -2:].flatten(), 3)
#                     else:
#                         Vi = np.delete(v_out[i - 1:i + 2, j - 1:j + 2].flatten(), 4)
#
#                     Vm = np.nanmedian(Vi)
#                     rm = np.nanmedian(np.abs(Vi - Vm))
#                     rv0 = np.abs(v_out[i, j] - Vm) / (rm + 0.1)
#
#                     if rv0 > r_thresh:
#                         v_out[i, j] = np.nan
#                     if not np.isfinite(rv0):
#                         v_out[i, j] = np.nan
#
#     # set all masked elements to zero so they are not replaced in openpiv.filters
#     if mask is not None:
#         u_out[mask] = 0.0
#         v_out[mask] = 0.0
#
#     print(("Number of u outliers: {}".format(np.sum(np.isnan(u_out)))))
#     print(("Percentage: {}".format(np.sum(np.isnan(u_out)) / u.size * 100)))
#     print(("Number of v outliers: {}".format(np.sum(np.isnan(v_out)))))
#     print(("Percentage: {}".format(np.sum(np.isnan(v_out)) / v.size * 100)))
#
#     print("Replacing Outliers")
#     u_out, v_out = openpiv.filters.replace_outliers(u_out, v_out)
#
#     return u_out, v_out
#
#
# def replace_outliers(image_pair_num, u_file, v_file, properties, gpuid=0):
#     """Replaces outliers
#
#     This function first loads all the output data from the output directory and applies the mask. All masked elements
#     are assign NaN. A single pair of output files is then passed to the function "outlier_detection" where the
#     outliers are identified and later replaced using openpiv.filters
#
#     Parameters
#     ----------
#     image_pair_num : int
#         the image pair to remove outliers on
#     u_file, v_file : ndarray
#         velocity fields
#     properties : dict
#          properties of the outlier detection
#     gpuid : int
#         gpu to use
#
#     """
#
#     output_dir = properties["out_dir"]
#     mask = properties["mask"]
#     r_thresh = properties["r_thresh"]
#
#     u = np.load(u_file)
#     v = np.load(v_file)
#
#     u[mask] = np.nan
#     v[mask] = np.nan
#
#     # call outlier_detection (which replaces the outliers)
#     u_out, v_out = outlier_detection(u, v, r_thresh, mask=mask)
#
#     # save to the replacement directory
#     save_files(output_dir, "u_repout_{:05d}.npy".format(image_pair_num), u_out)
#     save_files(output_dir, "v_repout_{:05d}.npy".format(image_pair_num), v_out)
#
#
# def interp_mask(mask, data_dir, exp=0, plot=False):
#     """
#     Interpolate the mask onto the output data. The mask has dimensions 996x1296 while the the output data dimensions
#     are much smaller (and depend on the minimum window size chosen)
#     """
#
#     # load the x and y location arrays from the output directory
#     """
#     SAVE y.npy IN THE GPU CODE SO THAT IT IS FLIPPED IN THE CORRECT ORIENTATION
#     SO IT SHOULD NOT HAVE TO BE FLIPPED AGAIN HERE
#     """
#     x_r = np.load(data_dir + "x.npy")[0, :]
#     y_r = np.load(data_dir + "y.npy")
#
#     x_pix = np.linspace(x_r[0], x_r[-1], len(mask[0, :]))
#     y_pix = np.linspace(y_r.min(), y_r.max(), len(mask[:, 0]))
#
#     mask_int = np.empty([y_r.size, x_r.size])
#
#     f = interp.interp2d(x_pix, y_pix, mask.astype(float))
#     mask_int = f(x_r, y_r)
#     mask_int = np.array(mask_int > 0.9)
#
#     # expand the mask
#     if exp > 0:
#         for i in range(x_r.size):
#             valid = np.where(mask_int[:, i] == 0)[0]
#             if valid.size == 0:
#                 continue
#             low = valid[0]
#             high = valid[-1]
#             mask_int[low - exp:low, i] = 0
#             mask_int[high:high + exp + 1, i] = 0
#
#     return mask_int
#
#
# def widim_gpu(start_index, frame_a_file, frame_b_file, properties, gpuid=0):
#     # ==================================================================
#     # PARAMETERS FOR OPENPIV
#     # ==================================================================
#     dt = properties["dt"]
#     min_window_size = properties["min_window_size"]
#     overlap = properties["overlap"]
#     coarse_factor = properties["coarse_factor"]
#     nb_iter_max = properties["nb_iter_max"]
#     validation_iter = properties["validation_iter"]
#     x_scale = properties["x_scale"]  # m/pixel
#     y_scale = properties["y_scale"]  # m/pixel
#     os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuid)
#     out_dir = properties["out_dir"]
#
#     frame_a = np.load(frame_a_file).astype(np.int32)
#     frame_b = np.load(frame_b_file).astype(np.int32)
#
#     # Import after setting device number, since gpu_process has autoinit enabled.
#     import openpiv.gpu_process
#     x, y, u, v, mask = openpiv.gpu_process.WiDIM(frame_a, frame_b, np.ones_like(frame_a, dtype=np.int32),
#                                                  min_window_size,
#                                                  overlap,
#                                                  coarse_factor,
#                                                  dt,
#                                                  validation_iter=validation_iter,
#                                                  nb_iter_max=nb_iter_max)
#
#     if x_scale != 1.0 or y_scale != 1.0:
#         # scale the data
#         x = x * x_scale
#         u = u * x_scale
#         y = y * y_scale
#         v = v * y_scale
#
#     # verify the directory exists:
#     if not os.path.exists(out_dir):
#         os.makedirs(out_dir)
#
#     # save the data
#     if start_index == 0:
#         np.save(out_dir + "x.npy", x)
#         np.save(out_dir + "y.npy", y[::-1, 0])
#
#     # Note: we're reversing u and v here only because the camera input is inverted. If the camera ever takes
#     # images in the correct orientations, we'll have to remove u[::-1, :].
#     save_files(out_dir, "u_{:05}.npy".format(start_index), u[::-1, :])
#     save_files(out_dir, "v_{:05}.npy".format(start_index), v[::-1, :])
#
#
# # ===================================================================================================================
# # FILE READ/SAVE UTILITY & MISC
# # ===================================================================================================================
#
# def save_files(out_dir, file_name, file_list):
#     if not os.path.exists(out_dir):
#         os.makedirs(out_dir)
#
#     file_ext = os.path.splitext(file_name)[1]
#     if file_ext == '.tif':
#         im = Image.fromarray(file_list)
#         im.save(out_dir + file_name)
#     else:
#         np.save(out_dir + file_name, file_list)
#
#
# def get_input_files(directory, file_name_pattern):
#     # get the images (either .tif or npy), converting to npy if .tif
#
#     file_list = sorted(glob.glob(directory + file_name_pattern))
#
#     # if list is empty, files could be in tif format
#     if not file_list:
#         new_file_pattern = os.path.splitext(file_name_pattern)[0] + '.tif'
#         file_list = sorted(glob.glob(directory + new_file_pattern))
#         tif_to_npy(directory, file_name_pattern, file_list)
#         file_list = sorted(glob.glob(directory + file_name_pattern))
#
#     return file_list
#
#
# def tif_to_npy(out_dir, pattern, file_list):
#     for i in range(len(file_list)):
#         file_read = io.imread(file_list[i])
#         np.save(os.path.splitext(file_list[i])[0] + '.npy', file_read)
