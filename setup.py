from os import path
from setuptools import setup, find_packages


# read the contents of your README file
this_directory = path.abspath(path.dirname(__file__))
# with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="OpenPIV",
    version="0.25.1",
    packages=find_packages(),
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
    setup_requires=[
        "setuptools",
    ],
    install_requires=[
        "numpy",
        "imageio>=2.22.4",
        "matplotlib>=3",
        "scikit-image",
        "scipy",
        "natsort",
        "tqdm",
        "importlib_resources",
        "pycuda",
        "scikit-cuda @ git+https://github.com/lebedov/scikit-cuda.git",
    ],
    extras_require={"tests": ["pytest", "pytest-benchmark", "pytest-regressions"]},
    classifiers=[
        # PyPI-specific version type. The number specified here is a magic
        # constant
        # with no relation to this application's version numbering scheme.
        # *sigh*
        "Development Status :: 4 - Beta",
        # Sublist of all supported Python versions.
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        # Sublist of all supported platforms and environments.
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        # Miscellaneous metadata.
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
    # long_description=long_description,
    # long_description_content_type='text/markdown'
)
