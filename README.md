# OpenPIV-Python with GPU support

[![DOI](https://zenodo.org/badge/148214993.svg)](https://zenodo.org/badge/latestdoi/148214993)

Version of [openpiv-python](https://github.com/OpenPIV/openpiv-python) with the addition of GPU-accelerated functions.
Compared to the CPU-bound functions, the GPU-accelerated modules perform much faster, making them suitable for large
datasets.
The GPU-acceleration is done using Nvidia's CUDA platform, so it can run only on machines with Nvidia GPUs.

OpenPIV-Python consists of Python modules for scripting and executing the analysis of a set of PIV image pairs. In
addition, a Qt and Tk graphical user interfaces are in development, to ease the use for those users who don't have
python skills.

## Warning

The GPU functions of OpenPIV-Python is still in *pre-beta* state. This means that
it still might have some bugs and the API may change. However, testing and contributing
is welcome, especially if you can contribute with new algorithms and features.

Validation of the code for instantaneous and time averaged flow has been done, and a 
paper on that topic has been published.

## Test without installation

Try the GPU functionality on Google Colab: [Basic Tutorial](https://colab.research.google.com/github/ericyang125/openpiv-python-gpu/blob/main/openpiv/tutorials/openpiv_python_gpu_tutorial_basic.ipynb).
When viewing the file on GitHub there will be a link to view the notebook with Google Colab.
Click the link to load the notebook into Colaboratory where you can test the GPU capabilities.

## Install from source

The GPU functions require the Nvidia CUDA toolkit, which is available for Windows or [supported Linux kernels only](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements).

Depending on your platform, the CUDA toolkit or the CUDA-related python packages (`PyCUDA, scikit-cuda`) could be
problematic to install.
See the procedure in the last [section](#installing-cuda-toolkit) that might help with installing CUDA.

Clone the repository from GitHub:

    git clone https://github.com/OpenPIV/openpiv-python-gpu.git

To perform a global installation, use:

    python setup.py install

## Documentation

The OpenPIV documentation is available on the project web page at <https://openpiv.readthedocs.org>.
For documentation of the GPU-accelerated modules, see API reference or the tutorial notebooks below.

## Demo notebooks 

Two tutorial notebooks demonstrate the usage of the GPU-accelerated functions:
1. [Basic tutorial](https://colab.research.google.com/github/ericyang125/openpiv-python-gpu/blob/main/openpiv/tutorials/openpiv_python_gpu_tutorial_basic.ipynb)
2. [Advanced tutorial](https://colab.research.google.com/github/ericyang125/openpiv-python-gpu/blob/main/openpiv/tutorials/openpiv_python_gpu_tutorial_advanced.ipynb)

Notebooks containing examples for non-GPU PIV functions are available in the base repository:
[OpenPIV-Python-Examples](https://github.com/OpenPIV/openpiv-python-examples)

## Contributors

1. [OpenPIV team](https://groups.google.com/forum/#!forum/openpiv-users)
2. [Cameron Dallas](https://github.com/CameronDallas5000)
3. [Alex Liberzon](https://github.com/alexlib)
4. [Eric Yang](https://github.com/ericyang125)

Copyright statement: `smoothn.py` is [a Python version of `smoothn.m` originally created by D. Garcia](https://de.mathworks.com/matlabcentral/fileexchange/25634-smoothn), written by Prof. Lewis and [available on Github](https://github.com/profLewis/geogg122/blob/master/Chapter5_Interpolation/python/smoothn.py).
We include a version of it in the `openpiv` folder for convenience and preservation. We are thankful to the original authors for releasing their work as an open source. OpenPIV license does not relate to this code. Please communicate with the authors regarding their license. 

## How to cite this work

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4409178.svg)](https://doi.org/10.5281/zenodo.4409178)

Dallas CA, Wu M, Chou VP, Liberzon A, Sullivan PE. GPU Accelerated Open Source Particle Image Velocimetry Software for
High Performance Computing Systems. ASME. J. Fluids Eng. 2019.
[doi:10.1115/1.4043422](http://fluidsengineering.asmedigitalcollection.asme.org/article.aspx?articleid=2730543).

## Installing CUDA toolkit

### Linux

Follow the instructions for installation at:

https://docs.nvidia.com/cuda/

An easy way to get a working install of CUDA is to use Ubuntu 20.04 LTS, then install CUDA by:

    sudo apt install nvida-cuda-toolkit

Note that CUDA toolkit installed on unsupported Linux kernels (i.e. not on this
[list](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements)) might not work
correctly, even if PyCUDA appears to install without problems.

### Windows

Update to the latest supported drivers:

https://www.nvidia.com/Download/index.aspx

Download CUDA from Nvidia website:

https://developer.nvidia.com/cuda-downloads

Visual Studio C++ compiler with CLI support needs to be installed before CUDA. It can be downloaded from:

https://visualstudio.microsoft.com/visual-cpp-build-tools/

Ensure that cl.exe is on your Windows PATH.

Follow the instructions for installation at:

https://docs.nvidia.com/cuda/

### Installing scikit-cuda and PyCUDA

The CUDA-accelerated modules depend on scikit-cuda and PyCUDA.

First, ensure that CUDA is compiled and on the PATH:

	nvcc -V

Install scikit-CUDA, which should install PyCUDA as well. If this throws errors, the CUDA toolkit was likely not
properly installed.

    pip install scikit-cuda
