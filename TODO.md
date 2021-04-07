#TODO
 [ ] memory optimizations - complete in April
     [ ] reuse same arrays as previous iterations
         [ ] write an outside class to store the common GPU data

 [ ] reconcile the definition of new arrays with how it was originall coded

 [ ] complete the Jupyter notebook tutorial - complete first week of March

 [ ] correlation class
     [ ] move methods out of init()
     [ ] define fewer GPU arrays

 [ ] define empty arrays on the GPU directly instead of sending Numpy arrays to the GPU by gpuarray.to_gpu

 [ ] push to mother repository - complete in March

 [ ] profile for the slowest parts of the code

 [ ] fix edge treatment to have less errors
     [ ] validation of edges
 [ ] Use CuPy or scikit-cuda to implement the cosine transform used in smoothn

 [ ] reduce the number of temporary gpus arrays created
