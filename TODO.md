# TODO

The items in this document refer only to the GPU-accelerated functions in the openpiv.gpu package.

## Documentation

- Build an API reference on top of the existing openpiv documentation
- Add a tutorial notebook showing how multiple GPUs can be used in an HPC setting.
- Define the mathematical operations of certain CUDA kernels:
  - Subpixel estimation:
  - Validation:
  - Signal-to-noise ratio:


## Filtering

Implement image filtering using CUDA.

Filtering raw PIV images is a process that can take longer than the PIV processing itself.
Histogram normalization has been shown in testing to increase signal-to-noise ratios.
This can possibly be implemented in CUDA to save significant time in the PIV workflow.

## API alignment

Align the public API of the gpu functions with the final API of the pure python implementation to reduce confusion by users.
  - gpu.process
  - gpu.validation
  - Add wrappers to pass GPUArrays to the GPU-accelerated functions, such as validation, so that they can be used in other
  functions where GPU-acceleration is only desired in one part.

## GPU smoothing

Allow for smoothing of vector fields using GPU processing.

In the current implementation, the vector fields are passed from the GPU back to CPU memory to perform smoothing.
For small vector fields this works fine, but for larger fields, it would be done faster on the GPU.
2D discrete cosine transform can be implemented fairly easily with a combination of PyCUDA interface and CUDA kernels.
The remainder of smoothn should be easily portable to GPU arrays as well.
There should be switching between using the CPU or GPU to smooth, depending on the size of the vector field.

## Multiprocessing

Allow concurrent processes to share same CUDA contexts/streams

The current multiprocessing solution is to supply wrapper modules that multiprocess the GPU functions using Python's multiprocessing class.
This work okay, but is hard to adapt to the differing I/O needs of workstation and HPC use-cases.
Each process also needs its own CUDA context, but the GPU must switch between contexts, which incurs overhead.
The solution may be to use multiprocessing for concurrent I/O operations, and use a single process with multithreading to call the GPU functions for all the other processes.
The ability of PyCUDA to specify the stream that CUDA kernels run on should also be used, and a strategy to choose streams that reduce the latency of concurrent PIV processing should be devised.

## Customization of interrogation windows

Allow for wide range of window shapes.

Currently, only square interrogation windows with dimensions that are powers
of 2 and at least 8 are supported. While this is efficient in view of the FFT step, other shapes may be desired by researchers.
  - Support non-integer x, y, overlap.
  - Support non-isotropic x, y, overlap.
  - Support non-power-of-2 window sizes.
  - 2D thread-block shapes (n, (wd * ht)) indexing should be flattened to 2D to more efficiently use resources since wd * ht
might not be a power of 2.

- Use a common framework to call CPU- or GPU-based PIV functions. The API for the gpu_process module was based on the
decommisioned WiDIM algorithm. Because it is significantly different from the API used in the CPU-based modules, it
could be confusing for users of OpenPIV.

## Improved interpolation

Implement bicubic interpolation in the window slicing.

Interpolation of the pixel values is important to PIV accuracy as subpixel estimation with sharp correlation peaks results in peak-locking on the image coordinates.
Because the grey-levels of the particle images is assumed to be gaussian, linear interpolation is not ideal in estimating the values.

## Auto mesh-refinement and iterations

Automate the refinement of the window sizing and the number iterations at each window size, to remove user specification.

The iterative algorith that is implemented operates by adding a correction to a prediction at each iteration.
When the correction approaches zero, either the mesh should be refined or the iterations should be stopped.

## CUDA performance optimization

Use shared device memory which has less latency than global memory.

Most of the execution time of the GPU kernels is
spent accessing device memory in a parallel fashion. Shared memory exists to speed memory accesses by threads in the
same block, but require logic to be written in the kernels that copy and index data from global device memory to shared
memory.

Ensure that memory accesses are coherent.

Future development should ensure that, as much as possible, global device
memory is accessed in contiguous blocks in each thread block of a kernel. This minimizes costly the number of memory
accesses by the thread blocks to global memory.
