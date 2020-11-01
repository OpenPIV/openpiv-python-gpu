# OpenPIV GPU
GPU-accelerated PIV processing

## Status
Although the goal of this work is to improve processing speed, many of the subroutines are still not fully optimized.

## Intended Features
1. Update to Python 3 for future-proofing
2. Finish some of the the TODO items in the multiprocessed code, if valuable
3. Implement the work of Paul Lizee of the Turbulence Research Lab
4. Rebase ontop of the current OpenPIV-Python development

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
    
