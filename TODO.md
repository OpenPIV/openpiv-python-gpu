# TODO

## Bugs
- [ ] very high overlap ratios throw an error

## Documentation
- [x] complete the Jupyter notebook tutorial
   - [x] basic
   - [ ] advanced

## Optimizations
- [ ] memory optimizations
  - [x] reuse same arrays as previous iterations
  - [x] write an outside class to store the common GPU data
  - [ ] reconcile the definition of new arrays with how it was originally coded
  - [ ] predefine arrays for u, v, x, y, mask
  - [ ] don't calculate x, y for every iteration

- [ ] correlation class
  - [ ] move methods out of init()
  - [ ] define fewer GPU arrays
  - [ ] Use CUDA kernel in subpixel location - this can gain ~10% performance

- [ ] define empty arrays on the GPU directly instead of sending Numpy arrays to the GPU by gpuarray.to_gpu
- [ ] reduce the number of temporary gpu arrays created

## Features
- [ ] dismantle the F-structure so that others can more easily contribute
- [x] re-enable other validation methods
  - [x] mean validation
  - [x] s2n
  - [x] rms
- [ ] Add ROI feature
- [ ] Implement the extended search area in the second frame
- [ ] Make the validation functions usable by other algorithms, maybe by wrapping them in a function the does I/O with the GPU
- [ ] Use CuPy or scikit-cuda to implement the cosine transform used in Smoothn
  - [ ] validate next to masked points

- [ ] fix edge treatment to have fewer errors
  - [ ] validation of edges
  - [ ] validation of points next to mask
  - [ ] interpolation onto edges
