"""This module performs multiprocessing of the OpenPIV GPU algorithms.

WARNING: File read/close is UNSAFE in multiprocessing applications because multiple threads are accessing &/or
writing to the same file. Please remember to use a queue if doing file I/O concurrently.

"""
from multiprocessing import (
    Process,
    Manager,
    Pool,
    cpu_count,
    set_start_method,
    current_process,
)
from math import ceil
from time import time
import os
from contextlib import redirect_stdout as redirect_stdout
import time
import warnings


# MULTIPROCESSING UTILITY CLASSES & FUNCTIONS
class MPGPU(Process):
    """Multiprocessing class for OpenPIV processing algorithms

    Each instance of this class is a process for some OpenPIV algorithm.

    Parameters
    ----------
    func : function
        OpenPIV algorithm that is multiprocessed
    items : iterable
        *lists of partitions of items to process. *list is comprised of arguments to be passed to func (e.g. frame_a,
        frame_b).
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
            os.environ["CUDA_DEVICE"] = str(gpu_id)

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


def parallelize(
    func, *items, num_processes=None, num_gpus=None, index=None, **parameters
):
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
        assert all(
            [len(item_a) == len(item_b) for item_a in items for item_b in items]
        ), "Input item lists are different lengths. len(items) = {}".format(
            [len(item) for item in items]
        )

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
        "Multiprocessing: Number of processes requested = {}. Number of CPU cores available = {}".format(
            num_processes, cpu_count()
        )
    )
    print(
        "Multiprocessing: Number of physical GPUs to use = {}. Number of GPUs available = {}".format(
            num_gpus, "unknown"
        )
    )
    print("Multiprocessing: Size of each partition =", partition_size)

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
                sublist[j] = items[j][start_index : start_index + partition_size]
        else:
            sublist = None

        # spawn the process
        if index is not None:
            process = MPGPU(
                func, sublist, gpu_id, index=start_index, parameters=parameters
            )
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


def mp_gpu_func(frame_a, frame_b, num_gpus, kwargs):
    """This function processes a pair of images using the GPU-PIV algorithm.

    Parameters
    ----------
    frame_a, frame_b : ndarray
        PIV images
    kwargs : dict
        Keyword arguments for the PIV-GPU algorithm.
    num_gpus : int
        number of gpus
    """
    # set the CUDA device
    cpu_name = current_process().name
    k = (int(cpu_name[cpu_name.find("-") + 1 :]) - 1) % num_gpus
    os.environ["CUDA_DEVICE"] = str(k)
    time1 = time.time()

    # GPU process
    with redirect_stdout(None):
        x, y, u, v, maks, s2n = gpu_func(frame_a, frame_b, kwargs)

    print("processed image pair. GPU = {}. dt = {:.3f}.".format(k, time.time() - time1))

    return x, y, u, v, maks, s2n


def gpu_func(frame_a, frame_b, kwargs):
    # start a PyCUDA context
    # import pycuda.autoinit
    import openpiv.gpu_process as gpu_process

    # GPU process
    with warnings.catch_warnings():
        x, y, u, v, mask, s2n = gpu_process.gpu_piv(frame_a, frame_b, **kwargs)

    return x, y, u, v, mask, s2n
