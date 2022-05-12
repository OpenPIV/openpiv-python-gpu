# OpenPIV Python version with GPU support

[![DOI](https://zenodo.org/badge/148214993.svg)](https://zenodo.org/badge/latestdoi/148214993)

GPU accelerated version of OpenPIV in Python. The algorithm and functions are mostly the same 
as the CPU version. The main difference is that it runs much faster. The source code has been 
augmented with CUDA, so it will only run on NVIDIA GPUs.


OpenPIV consists in a Python and Cython modules for scripting and executing the analysis of 
a set of PIV image pairs. In addition, a Qt and Tk graphical user interfaces are in 
development, to ease the use for those users who don't have python skills.

## Warning
The OpenPIV-Python GPU version is still in pre-beta state. This means that
it still might have some bugs and the API may change. However, testing and contributing
is very welcome, especially if you can contribute with new algorithms and features.

Validation of the code for instantaneous and time averaged flow has been done, and a 
paper on that topic has been submitted and will be published in the near future

Development is currently done on a Linux/Mac OSX environment, but as soon as possible 
Windows will be tested. If you have access to one of these platforms
please test the code. 


## Test without installation
You can test the code without needing to install anything locally. Included in this 
repository is the IPython Notebook [Openpiv_Python_Cython_GPU_demo.ipynb](Openpiv_Python_Cython_GPU_demo.ipynb). 
When viewing the file on Github there will be a link to view the notebook with Colaboratory. 
Clicking this will load the notebook into Googles free cloud computing service and you can test
the GPU capabilities. 


## Install From Source

Make sure you have installed all the dependancies (`numpy, matplotlib, scipy, cython, skcuda, pycuda`).
The GPU version will only install if it detects both `skcuda` and `pycuda`. Otherwise, only the CPU version will be installed. 

Clone the repository from Github:

    git clone https://github.com/OpenPIV/openpiv-python-gpu.git

Compile the cython and CUDA code (this can take a while):

    python setup.py build_ext --inplace

Or for the global installation, use:

    python setup.py install 


## Documentation

The OpenPIV documentation is available on the project web page at <http://openpiv.readthedocs.org>

## Demo notebooks 

1. [Tutorial Notebook 1](https://nbviewer.jupyter.org/github/OpenPIV/openpiv-python-examples/blob/main/notebooks/tutorial1.ipynb)
2. [Tutorial notebook 2](https://nbviewer.jupyter.org/github/OpenPIV/openpiv-python-examples/blob/main/notebooks/tutorial2.ipynb)
3. [Dynamic masking tutorial](https://nbviewer.jupyter.org/github/OpenPIV/openpiv-python-examples/blob/main/notebooks/masking_tutorial.ipynb)
4. [Multipass with Windows Deformation](https://nbviewer.jupyter.org/github/OpenPIV/openpiv-python-examples/blob/main/notebooks/window_deformation_comparison.ipynb)
5. [Multiple sets in one notebook](https://nbviewer.jupyter.org/github/OpenPIV/openpiv-python-examples/blob/main/notebooks/all_test_cases_sample.ipynb)
6. [3D PIV](https://nbviewer.jupyter.org/github/OpenPIV/openpiv-python-examples/blob/main/notebooks/openpiv_3D_test.ipynb)


These and many additional examples are in another repository: [OpenPIV-Python-Examples](https://github.com/OpenPIV/openpiv-python-examples)


## Contributors

1. [OpenPIV team](https://groups.google.com/forum/#!forum/openpiv-users)
2. [Cameron Dallas](https://github.com/CameronDallas5000)
3. [Alex Liberzon](https://github.com/alexlib)


Copyright statement: `smoothn.py` is a Python version of `smoothn.m` originally created by D. Garcia [https://de.mathworks.com/matlabcentral/fileexchange/25634-smoothn], written by Prof. Lewis and available on Github [https://github.com/profLewis/geogg122/blob/master/Chapter5_Interpolation/python/smoothn.py]. We include a version of it in the `openpiv` folder for convenience and preservation. We are thankful to the original authors for releasing their work as an open source. OpenPIV license does not relate to this code. Please communicate with the authors regarding their license. 

## How to cite this work

Dallas CA, Wu M, Chou VP, Liberzon A, Sullivan PE. GPU Accelerated Open Source Particle Image Velocimetry Software for High Performance Computing Systems. ASME. J. Fluids Eng. 2019;():. [doi:10.1115/1.4043422](http://fluidsengineering.asmedigitalcollection.asme.org/article.aspx?articleid=2730543).
