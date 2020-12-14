"""This module performs multiprocessing of the OpenPIV GPU algorithms.

WARNING: File read/close is UNSAFE in multiprocessing applications because multiple threads are accessing &/or
writing to the same file. Please remember to use a queue if doing file I/O concurrently.

"""
from multiprocessing import Process, Manager, Pool, cpu_count, set_start_method, current_process
import numpy as np
from math import ceil
from time import time
from functools import partial
import os
from contextlib import redirect_stdout as redirect_stdout
import time
import glob


# MULTIPROCESSING UTILITY CLASSES & FUNCTIONS
class MPGPU(Process):
    """Multiprocessing class for OpenPIV processing algorithms

    Each instance of this class is a process for some OpenPIV algorithm.

    Parameters
    ----------
    func : function
        OpenPIV algorithm that is multiprocessed
    items : iterable
        *lists of partitions of items to process. *list is comprised of arguments to be passed to func (e.g. frame_a, frame_b).
    gpu_id : int
        which GPU to use for processing
    index : int
        beginning index number of items to process

    """

    def __init__(self, func, items, gpu_id, index=None, parameters=None):
        Process.__init__(self)
        self.func = func
        self.gpu_id = gpu_id
        self.items = items
        self.index = index
        self.num_items = len(items[0])
        self.parameters = parameters

        if gpu_id is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            # os.environ['CUDA_DEVICE'] = str(gpu_id)

    def run(self):
        # process_time = time()
        # func = self.properties["gpu_func"]

        # this rearranges the items to pick out the corresponding *args
        items = [[item[i] for item in self.items] for i in range(self.num_items)]

        # set the starting index
        index = self.index

        for i in range(self.num_items):
            # run the function
            if self.items is not None:

                if self.index is not None:
                    self.func(*items[i], index=index, **self.parameters)
                else:
                    self.func(*items[i], **self.parameters)
            else:
                if self.index is not None:
                    self.func()

            index += 1


def parallelize(func, *items, num_processes=None, num_gpus=None, index=None, **parameters):
    """Parallelizes OpenPIV algorithms

    This helper function spawns instances of the class MGPU to multiprocess up to two sets of corresponding items. It
    assumes that each physical GPU will handle one process only. The arguments for func must be *args followed by
    **kargs. If index is true, then MGPU will pass the index number of the items as a keyword argument.

    Parameters
    ----------
    func : function user-defined function to parallelize items: tuple *list of the items to
    process. *list is comprised of arguments to be passed to func (e.g. frame_a, frame_b).
    num_processes : int
        number of parallel processes to run. This may exceed the number of CPU cores, but will not speed up processing.
    num_gpus : int
        number of physical GPUs to use for multiprocessing. This will cause errors if the larger than number of GPUs.
        If > 0, a GPU index will be passed to the function as a positional argument following the sublist.
    index :
        bool whether to pass the user-defined function an index of the items processed
    parameters : dict
    other parameters to pass to function as keywords arguments.

    """
    process_list = []
    gpu_id = None
    num_args = len(items)
    num_items = len(items[0])

    # check that each of the lists of input items provided are the same dimension
    if items is not None:
        assert all([len(item_a) == len(item_b) for item_a in items for item_b in items]), \
            'Input item lists are different lengths. len(items) = {}'.format([len(item) for item in items])

    # default to a number of cores equal to 37.5% or fewer of the available CPU cores (75% of physical cores)
    if num_processes is None:
        num_processes = max(cpu_count() - 1, 1)  # minimum 1 in case of low-spec machine

    # size of each partition is computed
    if items[0] is not None:
        partition_size = ceil(num_items / num_processes)
    else:
        partition_size = None

    # print information about the multiprocessing
    print(
        'Multiprocessing: Number of processes requested = {}. Number of CPU cores available = {}'.format(num_processes,
                                                                                                         cpu_count()))
    print('Multiprocessing: Number of physical GPUs to use = {}. Number of GPUs available = {}'.format(num_gpus,
                                                                                                       'unknown'))
    print('Multiprocessing: Size of each partition =', partition_size)

    # loop through each partition to spawn processes
    i = 0  # number of processes spawned
    while True:
        # determine which GPU to use, if any
        if num_gpus is not None:
            gpu_id = i % num_gpus

        # The partition is selected
        start_index = i * partition_size
        if items is not None:
            # create a list of partitions for each of the input items
            sublist = [[]] * num_args
            for j in range(num_args):
                sublist[j] = items[j][start_index:start_index + partition_size]
        else:
            sublist = None

        # spawn the process
        if index is not None:
            process = MPGPU(func, sublist, gpu_id, index=start_index, parameters=parameters)
        else:
            process = MPGPU(func, sublist, gpu_id, parameters=parameters)
        process.start()
        process_list.append(process)

        # update the number of processes
        i += 1

        # check to see if process stops
        if items is not None:
            if i * partition_size >= num_items:
                break
        else:
            if i == num_processes:
                break

    # join the processes to finish the multiprocessing
    for process in process_list:
        process.join()


"""This module handles the front-end of PIV processing. For convenience, this standardizes the file formats and
reduces scripting."""

class PIV:
    """Handles the front-end file management for PIV processing.

    The images names must be of format im_name + #### + suffix + .extension_name . The image set can be subdivided
    and each subset can be run on its own GPU. The output is saved as a npz file with x, y coordinates and the u, v fields.

    Parameters
    ----------
    im_dir : str
        directory where images are stored
    save_name : str
        name of the file to save, none for no save
    save_dir : str
        name of the directory to save to
    im_ext : str
        extension of the image
    suf_a, suf_b : str
        letter suffix of the first/second image
    im_range : tuple
        numeric range indicating how what portion of the images to process
    processes :
        number of parallel processes to run, should be 1 per GPU

    """

    def __init__(self, im_dir, save_name=None, save_dir=None, im_ext='.tiff', suf_a='a', suf_b='b', im_range=None,
                 processes=1):
        self.im_dir = im_dir
        self.save_name = save_name
        self.save_dir = save_dir
        self.im_ext = im_ext
        self.suf_a, self.suf_b = suf_a, suf_b
        self.im_range = im_range
        self.processes = processes
        self.x = None
        self.y = None
        self.u = None
        self.v = None
        self.exists = False

    def begin(self):
        # check if output exists
        self.exists = os.path.exists(self.save_dir + self.save_name + '.npz') if self.save_name is not None else False

        # set the image range
        if self.im_range is not None:
            im_range = self.im_range
        else:
            im_range = (0, 10000)

        # find the images to process
        str_a = self.im_dir + '*' + '[0-9]' * 4 + self.suf_a + self.im_ext
        str_b = self.im_dir + '*' + '[0-9]' * 4 + self.suf_b + self.im_ext
        list_a = sorted(glob.glob(str_a))[im_range[0]:im_range[1]]
        list_b = sorted(glob.glob(str_b))[im_range[0]:im_range[1]]
        assert len(list_a) == len(list_b), 'Image lists are not same size.'
        assert len(list_a) > 0, 'No images found in {} within the range {}.'.format(self.im_dir, im_range)

        # print status
        print('Beginning to process {} fields...'.format(len(list_a)))
        start_time = time.time()

        return list_a, list_b, start_time

    def end(self, x, y, u, v, start_time):
        print("...Finished in {:.3f} seconds.".format(time.time() - start_time))

        # decide whether the file is to be saved
        if self.save_name is None:
            self.x, self.y = x, y
            self.u, self.v = u, v
        else:
            if self.save_dir is not None:
                # create the save directory if it doesn't exist already
                save_dir = self.save_dir
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                    print('Creating directory {} .'.format(save_dir))
            else:
                save_dir = self.im_dir

            # save the data as a structured frame
            print('Saving into file {}.npz in directory {} ...'.format(self.save_name, save_dir))
            np.savez(save_dir + self.save_name, x=x, y=y, u=u, v=v)
            print('...Done.')

    def run_widim_gpu(self, kwargs=None, cuda_device=None, overwrite=False):
        """Uses the widim algorithm of OpenPIV to process images.

        Parameters
        ----------
        kwargs : dict
            arguments to the widim algorithm
        cuda_device : int
            which CUDA device to use
        overwrite : bool
            controls whether old data is overwritten

        """
        list_a, list_b, start_time = self.begin()

        # check if overwrite condition is met
        if overwrite is False:
            if self.exists:
                print("Output file {} already exists--skipping.".format(self.save_dir + self.save_name))
                return

        if cuda_device is not None:
            # set the CUDA device
            os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_device)

        # do the OpenPIV imports
        import openpiv.gpu_process as gpu_process
        import openpiv.tools as tools

        def gpu_func(a, b, num):
            print('Processing pair {}: {} and {}.'.format(num, a.replace(self.im_dir, ''), b.replace(self.im_dir, '')))

            # load the images
            frame_a, frame_b = tools.imread(a), tools.imread(b)

            # GPU process
            x1, y1, u1, v1, _, _ = gpu_process.widim(frame_a, frame_b, **kwargs)

            return x1, y1, u1, v1

        # process one image pair
        x, y, u0, v0 = gpu_func(list_a[0], list_b[0], 0)

        # initiate arrays
        num_files = len(list_a)
        m, n = x.shape
        u, v = np.empty((num_files, m, n)), np.empty((num_files, m, n))
        u[0, :, :], v[0, :, :] = u0, v0

        for i in range(1, num_files):
            _, _, u[i, :, :], v[i, :, :] = gpu_func(list_a[i], list_b[i], i)

        self.end(x, y, u, v, start_time)


def mp_gpu_func(frames, kwargs, gpus):
    # set the CUDA device
    # k = int(current_process().name[-1]) - 1
    cpu_name = current_process().name
    k = (int(cpu_name[cpu_name.find('-') + 1:]) - 1) % gpus
    # k = os.getpid() % processes
    os.environ['CUDA_DEVICE'] = str(k)
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(k)
    time1 = time.time()

    # GPU process
    x, y, u, v, = gpu_func(frames[0, :, :], frames[1, :, :], kwargs)

    print('processed image pair. GPU = {}. dt = {:.3f}.'.format(k, time.time() - time1))

    return u, v


def gpu_func(frame_a, frame_b, kwargs):
    """This function processes a pair of images using the GPU-PIV algorithm.

    Parameters
    ----------
    frame_a, frame_b : ndarray
        PIV images
    kwargs : dict
        Keyword arguments for the PIV-GPU algorithm.
    """
    import openpiv.gpu_process as gpu_process

    # suppress stout to prevent cluttering the console
    with redirect_stdout(None):
        x, y, u, v, _, _ = gpu_process.gpu_piv_def(frame_a, frame_b, **kwargs)

    return x, y, u, v
