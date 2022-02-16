# TODO

## Bugs
- [x] very high overlap ratios throw an error
- [x] bias error in non-translating window
- [x] square window shapes sometimes error
- [x] completely blank frames cause smoothn to error
- [x] smoothing parameter doesn't default to 0.5
- [x] validation being done on first iteration even when trust_first_iter is True
- [x] smoothing is not being applied to functions

## Error handling
- [ ] out-of-memory is unhandled
- [ ] 

## Needs testing
- [ ] smoothing par
- [ ] validation tolerances

## Documentation
- [x] complete the Jupyter notebook tutorials
   - [x] basic
   - [x] advanced
- [ ] update the README to reflect improved functionality
- [ ] reorganize the tutorials

## Optimizations
- [ ] memory optimizations
  - [x] reuse same arrays as previous iterations
  - [x] write an outside class to store the common GPU data
  - [x] reconcile the definition of new arrays with how it was originally coded
  - [x] predefine arrays for u, v, x, y, mask
  - [x] don't calculate x, y for every iteration
  - [ ] use shared memory for GPU kernels
  - [x] confirm if to_gpu() is faster than gpuarray.zeros()
  - [x] optimize masking the input frames
  - [x] move kernel compilation out from functions
  - [x] break global functions into device functions

- [ ] correlation class
  - [x] reduce operations outside of CUDA kernels in iw_arrange()
  - [x] move methods out of init()
  - [x] define fewer GPU arrays
  - [x] use CUDA kernel in subpixel location
  - [x] return a GPU array for the validation list

- [x] define empty arrays on the GPU directly instead of sending Numpy arrays to the GPU by gpuarray.to_gpu
- [x] reduce the number of temporary gpu arrays created
- [x] make consistent the strain and shift array arguments in the correlation function
- [ ] flatten the 3D kernel indexing to 2D
- [ ] use at least 64 threads per block
- [ ] limit use of braching execution paths in same warp
- [ ] explicitly define float type literals to avoid casting down from double
- [ ] use const int literals
- [ ] ensure memory accesses are coherent

## Features
- [x] dismantle the F-structure so that others can more easily contribute
  - [x] remove the pre-allocation of GPU memory the main variables since it adds no performance.
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
  - [x] validation of edges
  - [x] validation of points next to mask
  - [ ] interpolation onto edges
- [x] replace vectors with median rather than median of surrounding points to not be affected by outliers
  - [ ] make this an option
- [x] pass either a frame or a frame size as the first argument to the GPUPIV init method
  - [ ] remove this requirement in lieu of dynamic allocation of array sizes
- [ ] add more sophisticated object-oriented features to set algorithm parameters in gpu_piv()
- [x] accept either floats or ints as frame inputs
- [ ] allow non-integer x, y, overlap
- [ ] allow non-isotropic x, y, overlap
- [ ] allow non-power-of-2 window sizes
- [ ] center-on-field functionality
- [ ] GPU kernel to do validation replacement
- [ ] Other subpixel methods
  - [x] Centroid
  - [x] Parabolic
  - [ ] Linear

## Object-oriented features
- [x] return the mask
- [x] return s2n
- [x] return x, y

## Testing
- [ ] cover all gpu functions by tests

## Clean code
- [x] refactor names to be more readable
- [x] use logging rather than printing to console
- [x] separate public functions from private functions in GPU-module
- [ ] add input checks to all public functions
- [ ] add error handling exceptions to all gpu functions
- [ ] cleanly separate validation functions from processing functions in gpu_accelerated modules
- [ ] eliminate output arguments in the gpu-accelerated functions
- [ ] don't pass/return None
- [ ] find a better way to log gpu piv iterations
- [x] evaluate purpose of old code not found in openpiv-python
- [ ] Try to work with python types whenever possible
  - [ ] indicate np types with encoding ending with _*f
