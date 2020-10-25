from os import path

from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy

# Check for GPU support
GPU_SUPPORT = False
try:
    import pycuda
    import skcuda

    print("GPU support found. Will build GPU extensions.")
    GPU_SUPPORT = True
except ImportError:
    print("No GPU support found. Continuing install.")
    pass

extensions = [
    Extension("openpiv.process", ["./openpiv/process.pyx"], include_dirs=[numpy.get_include()])]

if GPU_SUPPORT:
    gpu_module = Extension(name="openpiv.gpu_process",
                           sources=["openpiv/gpu_process.pyx"],
                           include_dirs=[numpy.get_include()],
                           )
    extensions.append(gpu_module)
    gpu_validation_module = Extension(name="openpiv.gpu_validation",
                                      sources=["openpiv/gpu_validation.pyx"],
                                      include_dirs=[numpy.get_include()],
                                      )
    extensions.append(gpu_validation_module)

extensions = cythonize(extensions, include_path=[numpy.get_include()], compiler_directives={'language_level': "3"})

# read the contents of your README file
this_directory = path.abspath(path.dirname(__file__))
# with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
with open(path.join(this_directory, 'README.md')) as f:
    long_description = f.read()

setup(
    name = "OpenPIV",
    version ='0.22.3',
    cmdclass = {'build_ext': build_ext},
    ext_modules = extensions,
    packages=find_packages(),
    include_package_data=True,
    setup_requires=[
        'setuptools',
        'cython>=0.29.14',
        'numpy'
    ],
    install_requires=[
        'cython>=0.29.14',
        'numpy',
        'imageio',
        'matplotlib>=3',
        'scikit-image',
        'progressbar2',
        'scipy',
        'natsort',
        'GitPython',
        'pytest',
		'tqdm'
    ],
    classifiers = [
        # PyPI-specific version type. The number specified here is a magic constant
        # with no relation to this application's version numbering scheme. *sigh*
        'Development Status :: 4 - Beta',

        # Sublist of all supported Python versions.
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',

        # Sublist of all supported platforms and environments.
        'Environment :: Console',
        'Environment :: MacOS X',
        'Environment :: Win32 (MS Windows)',
        'Environment :: X11 Applications',

        # Miscellaneous metadata.
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
    ],
    long_description=long_description,
    long_description_content_type='text/markdown'
)
