# TODO

The items in this document refer only to the GPU-accelerated functions in the openpiv.gpu package.

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

## Documentation

- Build an API reference on top of the existing openpiv documentation
- Add a tutorial notebook showing how multiple GPUs can be used in an HPC setting.
- Define the mathematical operations of certain CUDA kernels:
  - Subpixel estimation:
    - Gaussian:
`\begin{equation}
    \delta = \frac{1}{2} \frac{\ln{f(x-1)} - \ln{f(x+1)}} {\ln{f(x-1)} - 2 \ln{f(x)} + \ln{f(x+1)} + \epsilon_)}
\end{equation}
where $\epsilon_0$ is a $\sim 10^{-6}$ residual to prevent divide by zero errors.`
    - Parabolic:
`\begin{equation}
    \delta = \frac{1}{2} \frac{f(x-1) - f(x+1)} {f(x-1) - 2 f(x) + f(x+1) + \epsilon_0}
\end{equation}`
    - Centroid
`\begin{equation}
    \delta = \frac{f(x+1) - f(x-1)} {f(x-1) + f(x) + f(x+1) + \epsilon_0}
\end{equation}`
    - Reference
`R. B. Fisher and D. K. Naidu, ‘A comparison of algorithms for subpixel peak detection’, in Image Technology: Advances in Image Processing, Multimedia and Machine Vision, Springer, 1996, pp. 385–404.`
  - Signal-to-noise ratio:
    - Peak-to-peak
`\begin{equation}
    \mathrm{SNR} = \log_{10}{\frac{|P_1|^2} {E_C}}
\end{equation}
where $P_1$ and $P_2$ are the magnitiudes of the first and second peak, respectively.
The second peak is found from the correlation domain with the neighbourhood of the first peak removed.`
    - Peak-to-mean-energy
`\begin{equation}
    \mathrm{SNR}= \log_{10}{\frac{P_1} {P_2}}
\end{equation}
\begin{equation}
    E_C = \left( \sqrt{\frac{1}{N} \sum_{i \in \Omega_{P_1/2}}^N |C(i)|^2} \right)^2
\end{equation}
where $\Omega_{P_1/2}$ is the part of the correlation domain with $0 \leq C(i) < P_1/2$`
    - Peak-to-RMS-energy
`\begin{equation}
    \mathrm{SNR} = \log_{10}{\frac{|P_1|^2} {P_{\mathrm{RMS}}^2}}
\end{equation}
\begin{equation}
    P_{\mathrm{RMS}} = \sqrt{\frac{1}{N} \sum_{i \in \Omega^+}^{N} |C(i)|^2}
\end{equation}
where $C(i)$ is the value of the correlation of a point in the correlation domain and $\Omega^+$ is the part of the correlation domain with $C(i) \geq 0$.`
    - Reference
`Z. Xue, J. J. Charonko, and P. P. Vlachos, ‘Particle image velocimetry correlation signal-to-noise ratio metrics and measurement uncertainty quantification’, Measurement Science and Technology, vol. 25, no. 11, p. 115301, 2014.`
  - Validation:
    - Signal-to-noise-ratio
`\begin{equation}
    SNR > \epsilon_{\mathrm{SNR}}
\end{equation}`
    - Normalized median
`\begin{equation}
    \left| \frac{u_i-u_{\mathrm{median},i}} {r_{\mathrm{median}, i} + \epsilon_0} \right| < \epsilon_{\mathrm{median}}
\end{equation}
for $i=1, 2$ components of velocity.
where $\epsilon_0$ is a $\sim 10^{-6}$ residual to prevent divide by zero errors.
\begin{equation}
    r_{\mathrm{median}} = \mathrm{median} \left( \left| u_j-u_{\mathrm{median}} \right| \right)
\end{equation}
for $j=0...8$ neighbouring velocity vectors.
$|\cdot|$ is the absolute value or norm operator, depending on whether the validation is done by velocity components or as vectors.`
    - Normalized mean
`\begin{equation}
    \left| \frac{u_i-u_{\mathrm{mean},i}} {r_{\mathrm{mean}, i} + \epsilon_0} \right| < \epsilon_{\mathrm{mean}}
\end{equation}
\begin{equation}
    r_{\mathrm{mean}} = \mathrm{mean} \left( \left| u_j-u_{\mathrm{mean}} \right| \right)
\end{equation}`
    - RMS
`\begin{equation}
    \left| \frac{u_i-u_{\mathrm{mean},i}} {r_{\mathrm{RMS}, i} + \epsilon_0} \right| < \epsilon_{\mathrm{RMS}}
\end{equation}
\begin{equation}
    r_{\mathrm{RMS}} = \sqrt{\sum_{j}^{N} \left( u_j-u_{\mathrm{mean}} \right) ^2}
\end{equation}`
    - References
`J. Westerweel and F. Scarano, ‘Universal outlier detection for PIV data’, Experiments in fluids, vol. 39, pp. 1096–1100, 2005.
R. J. Adrian and J. Westerweel, Particle image velocimetry. Cambridge university press, 2011.`
