# OpenPIV GPU
This is a GPU-accelerated version of openpiv-python, which can be found at https://github.com/OpenPIV/openpiv-python.git

## Installation
The requirements for the process.py module are OpenPIV (GPU version) and the standard Python scientific libraries (SciPy, Matplotlib, etc.).
The PIV analysis on velocity fields are not dependent on OpenPIV, however.

OpenPIV requires CUDA, which may be difficult to install. In addition to the instructions at https://github.com/OpenPIV/openpiv-python-gpu, the procedure below might help with installing OpenPIV.

### 0. Nvidia drivers:
Update to the latest supported drivers.

https://www.nvidia.com/Download/index.aspx

### 1. CUDA toolkit:

Download CUDA from Nvidia website:

https://developer.nvidia.com/cuda-downloads

If installing on Windows, Visual Studio C++ compiler with CLI support needs to be installed before CUDA. It can be downloaded from:

https://visualstudio.microsoft.com/visual-cpp-build-tools/

Ensure that cl.exe is on your Windows PATH

If installing on Linux, follow the instructions for Linux at:

https://docs.nvidia.com/cuda/

Ensure that the post-installation instructions are followed and test the install before proceeding to then next step.

Ensure that CUDA is compiled and on the PATH:

	nvcc -V

### 2. scikit-CUDA:
Install scikit-CUDA, which should install PyCUDA as well. If this throws errors, CUDA cwas probably not installed properly in the step above.
        
    pip install scikit-cuda

### 3. OpenPIV:
To install the OpenPIV, it can either be installed to the python environment by:

    pip install git+git://github.com/OpenPIV/openpiv-python-gpu.git
        
or cloned for local development by:

    python setup.py build_ext --inplace

    git clone https://github.com/ericyang125/PIV-GPU.git
    pip install -e /path/to/package
    
## Getting started.
To get started, see the tutorial Jupyter notebook.

##Copyright statement
`smoothn.py` is a Python version of `smoothn.m` originally created by D. Garcia [https://de.mathworks.com/matlabcentral/fileexchange/25634-smoothn], written by Prof. Lewis and available on Github [https://github.com/profLewis/geogg122/blob/master/Chapter5_Interpolation/python/smoothn.py]. We include a version of it in the `openpiv` folder for convenience and preservation. We are thankful to the original authors for releasing their work as an open source. OpenPIV license does not relate to this code. Please communicate with the authors regarding their license.

# Changes from base repository
piv_gpu is no longer supported
median validation is the main method of validation, along with smoothn used in intermediate fields
window deformation is implemented to improve estimation of velocity gradient
API for the function is different for the GPU function, which is now called pif_gpu_def
performance/reliability has been improved by various other changes
