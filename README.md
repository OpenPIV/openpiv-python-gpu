# openpiv-python-gpu
GPU accelerated version of OpenPIV in Python. The algorithm and functions are mostly the same 
as the CPU version. The main difference is that it runs much faster. The source code has been 
augmented with CUDA, so it will only run on NVIDIA GPUs.


## Warning
The OpenPIV GPU version is still in pre-beta state. This means that
it still might have some bugs and the API may change. However testing and contributing
is very welcome, especially if you can contribute with new algorithms and features.

Validation of the code for instantaneous and time averaged flow has been done, and a 
paper on that topic has been submitted and will be published in the near future

Development is currently done on a Linux/Mac OSX environment, but as soon as possible 
Windows will be tested. If you have access to one of these platforms
please test the code. 

## Test without installation
You can test the code without needing to install it. To test using Google's Colaboratory
follow the link:
https://colab.research.google.com/drive/1FqOOwtn_VWRhVEICDCvQyUg82nj5bMYP
