# OpenPIV GPU - Multiprocessed
I will try to continue work on this code that was started by the OpenPIV team and members of the Turbulence Research Lab.

## Intended Features
1. Update to Python 3 for future-proofing
2. Finish some of the the TODO items in the multiprocessed code, if valuable
3. Implement the work of Paul Lizee of the Turbulence Research Lab
4. Rebase ontop of the current OpenPIV-Python development

## Installation
The requirements for the process.py module are OpenPIV (GPU version) and the standard Python scientific libraries (SciPy, Matplotlib, etc.).
The PIV analysis on velocity fields are not dependent on OpenPIV, however.

OpenPIV depends on CUDA, which may be difficult to install. In addition to the instructions at https://github.com/OpenPIV/openpiv-python-gpu, the procedure below might help with installing OpenPIV.

### 0. Nvidia drivers:
Update to the latest drivers.

https://www.nvidia.com/Download/index.aspx

### 1. CUDA toolkit:

Download from Nvidia website:

https://developer.nvidia.com/cuda-downloads

If installing on Windows, Visual Studio C++ compiler needs to be installed before CUDA. It can be downloaded from:

https://visualstudio.microsoft.com/vs/features/cplusplus/

If installing on Linux, CUDA needs to be added to path:

    export PATH=/usr/local/cuda-10.2/bin:/usr/local/cuda-10.2/NsightCompute-2019.1${PATH:+:${PATH}}

Check that the CUDA compiler is callable (Linux):

    $ nvcc -V

For more info, see the Nvidia website:

https://docs.nvidia.com/cuda/

### 2. scikit-CUDA:
Install scikit-CUDA, which should install PyCUDA as well. If this throws errors, the CUDA compiler was probably not installed properly, or was not added to the path.
        
    pip install scikit-cuda

### 3. OpenPIV:
To install the OpenPIV, it can either be installed to the python environment by:

    pip install git+git://github.com/OpenPIV/openpiv-python-gpu.git
        
or cloned for local development by:

    python setup.py build_ext --inplace

    git clone https://github.com/ericyang125/PIV-GPU.git
    pip install -e /path/to/package
    
