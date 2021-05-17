# TODO
- [x] profile for the slowest parts of the code

- [ ] memory optimizations
    - [x] reuse same arrays as previous iterations
    - [x] write an outside class to store the common GPU data
    - [ ] reconcile the definition of new arrays with how it was originally coded

- [x] complete the Jupyter notebook tutorial
     - [ ] basic
     - [ ] advanced

- [ ] correlation class
    - [ ] move methods out of init()
    - [ ] define fewer GPU arrays
    - [ ] Use CUDA kernel in subpixel location - this can gain ~10% performance

- [ ] push to mother repository

- [ ] define empty arrays on the GPU directly instead of sending Numpy arrays to the GPU by gpuarray.to_gpu

- [ ] reduce the number of temporary gpu arrays created

- [ ] re-enable other validation methods

- [ ] fix edge treatment to have less errors
    - [ ] validation of edges
    - [ ] validation of points next to mask
    - [ ] interpolation onto edges
    
- [ ] Use CuPy or scikit-cuda to implement the cosine transform used in smoothn
    - [ ] validate next to masked points

- [ ] Merge/deprecate the various branches
