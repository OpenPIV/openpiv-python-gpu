# TODO

The items in this document refer only to the GPU-accelerated functions in the modules prefixed by 'gpu_', which provide
full PIV-processing functionality on their own.

## Documentation

- Add detailed external documentation for the public functions so that users have a good reference for all the
parameters and their effects.


- Add a tutorial notebook comparing CPU- and GPU- bound functions.


- Add a tutorial notebook showing how multiple GPUs can be used in an HPC setting.

## Features

- Center the window coordinates with respect to the image coordinates, such that the number of pixels
between the left side of the image and the leftmost window coordinate is the same (to 1 pixel) as that on the right
side. The same should be true for the top/bottom of the image. This can be easily implemented by padding the left side
at the window-slicing stage, which already has padding functionality. The only work required is adding input parameters
and logic to generate the padding, as well as testing.


- Add an option to print the logging input to console. Logging is preferred for status output since its behaviour can be
controlled by scripts, and it doesn't crowd out other console output. However, having an option to output it to the
console could be desirable for testing.


- Improve multiprocessing functionality of the PIV processing. The current multiprocessing solution is to
supply a wrapper modules that multiprocesses the GPU functions using Python's multiprocessing class. This work OK, but
has proven to be unadaptable to the differing I/O needs of workstation and HPC use-cases. It also precludes including
any preprocessing the multiprocessed workflow. Solutions may include creating a tutorial to show users how to write
their multiprocessing wrappers or to extend the flexibility of the multiprocessing module to suit expected use-cases.


- Allow for additional window sizes/shapes. Currently, only square interrogation windows with dimensions that are powers
of 2 and at least 8 are supported. While this is efficient in view of the FFT step, because PIV is a research tool,
other shapes may be desired. Therefore:
  - Support non-integer x, y, overlap.
  - Support non-isotropic x, y, overlap.
  - Support non-power-of-2 window sizes.
  - 2D thread-block shapes (n, (wd * ht)) indexing should be flattened to 2D to more efficiently use resources since wd * ht
might not be a power of 2.
  - Almost all logic governing the above lies in the correlation class.
  - FFT window stacks that are not powers of 2 should be tested to see if padding improves performance.


- Add wrappers to pass GPUArrays to the GPU-accelerated functions, such as validation, so that they can be used in other
functions where GPU-acceleration is only desired in one part.


- Change the interpolation functions so that masked points do not affect the interpolation. Currently, the interpolation
functions aren't aware of masked points so the interpolation will be wrong next to masked them. The solution will be to
edit the kernels to not interpolate using masked points. The performance cost of doing this should be considered,
however.


- Dynamically allocate size of arrays used in PIV if the passed frame is a different size. The PIV class
requires the user to supply the image size to the init() method, so that arrays of the correct size can be predefined.
The flexibility of the class can be slightly improved if there is logic to redefine all the arrays when a
different-sized image is passed. This would suit a potentially uncommon use-case where the data set are different-sized
images, maybe due to cropping.


- Use a common framework to call CPU- or GPU-based PIV functions. The API for the gpu_process module was based on the
decommisioned WiDIM algorithm. Because it is significantly different from the API used in the CPU-based modules, it
could be confusing for users of OpenPIV.

## Optimization

- Test using CUDA thread block sizes of 64. The allowed block size for thread in CUDA kernels are power of 2 from 32
to 1024. For simplicity in development, 32 was used everywhere. However, the ideal number is likely to differ for each 
kernel.


- Use shared device memory which has less latency than global memory. Most of the execution time of the GPU kernels is
spent accessing device memory in a parallel fashion. Shared memory exists to speed memory accesses by threads in the
same block, but require logic to be written in the kernels that copy and index data from global device memory to shared
memory.


- Ensure that memory accesses are coherent. Future development should ensure that, as much as possible, global device
memory is accessed in contiguous blocks in each thread block of a kernel. This minimizes costly the number of memory
accesses by the thread blocks to global memory.
