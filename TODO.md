# TODO

## Bugs
- [x] very high overlap ratios throw an error
- [x] bias error in non-translating window
- [ ] square window shapes sometimes error
- [ ] extended search area doesn't work for 2048 x 4096 image sizes
- [ ] completely blank frames cause smoothn to error

## Documentation
- [x] complete the Jupyter notebook tutorials
   - [x] basic
   - [x] advanced
- [ ] update the readme to reflect improved functionality
- [ ] reorganize the tutorials

## Optimizations
- [ ] memory optimizations
  - [x] reuse same arrays as previous iterations
  - [x] write an outside class to store the common GPU data
  - [x] reconcile the definition of new arrays with how it was originally coded
  - [x] predefine arrays for u, v, x, y, mask
  - [x] don't calculate x, y for every iteration
  - [ ] use shared memory for GPU kernels
  - [ ] confirm if to_gpu() is faster than gpuarray.zeros()
  - [ ] optimize masking the input frames

- [ ] correlation class
  - [x] reduce operations outside of CUDA kernels in iw_arrange()
  - [x] move methods out of init()
  - [ ] define fewer GPU arrays
  - [ ] use CUDA kernel in subpixel location
  - [ ] return a GPU array for the validation list

- [x] define empty arrays on the GPU directly instead of sending Numpy arrays to the GPU by gpuarray.to_gpu
- [ ] reduce the number of temporary gpu arrays created
- [x] make consistent the strain and shift array arguments in the correlation function

## Features
- [ ] dismantle the F-structure so that others can more easily contribute
- [x] re-enable other validation methods
  - [x] mean validation
  - [x] s2n
  - [x] rms
- [ ] add ROI feature
- [x] Implement the extended search area in the second frame
- [ ] Make the validation functions usable by other algorithms, maybe by wrapping them in a function the does I/O with the GPU
- [ ] Use CuPy or scikit-cuda to implement the cosine transform used in smoothn
- [ ] validate next to masked points
- [ ] fix edge treatment to have fewer errors
  - [ ] validation of edges
  - [ ] validation of points next to mask
  - [ ] interpolation onto edges
- [x] replace vectors with median rather than median of surrounding points to not be affected by outliers
  - [ ] make this an option
- [ ] pass either a frame or a frame size as the first argument to the GPUPIV init method
  - [ ] remove this requirement in lieu of dynamic allocation of array sizes

## Object-oriented features
- [x] return the mask
- [x] return s2n
- [x] return x, y

## Testing
- [ ] cover all gpu functions by tests

## Clean code
- [ ] refactor names to be more readable
- [x] use logging rather than printing to console
- [ ] add input checks to all public functions
- [ ] add error handling exceptions to all gpu functions
- [ ] cleanly separate validation functions from processing functions in gpu_accelerated modules
- [ ] eliminate output arguments in the gpu-accelerated functions
