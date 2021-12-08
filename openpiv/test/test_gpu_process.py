# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from skimage.util import random_noise
from skimage import img_as_ubyte
from scipy.ndimage import shift

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level=3)

import openpiv.gpu_process as gpu_process
import openpiv.gpu_validation as gpu_validation

fixture_dir = "./fixtures/"

# synthetic image parameters
_image_size_square = (2048, 2048)
_image_size_rectangle = (2048, 4096)
_u_shift = 8
_v_shift = -4
_threshold = 0.1


def create_pair_shift(image_size, u_shift, v_shift):
    """Creates a pair of images with a roll/shift """
    frame_a = np.zeros(image_size, dtype=np.int32)
    frame_a = random_noise(frame_a)
    frame_a = img_as_ubyte(frame_a)
    frame_b = shift(frame_a, (v_shift, u_shift), mode='wrap')

    return frame_a.astype(np.int32), frame_b.astype(np.int32)


# def create_pair_roll(image_size, u_roll, v_roll):
#     """Creates a pair of images with a roll/shift """
#     frame_a = np.zeros(image_size, dtype=np.int32)
#     frame_a = random_noise(frame_a)
#     frame_a = img_as_ubyte(frame_a)
#     frame_b = np.roll(frame_a)
#
#     return frame_a.astype(np.int32), frame_b.astype(np.int32)


_frame_a_square, _frame_b_square = create_pair_shift(_image_size_square, _u_shift, _v_shift)
_frame_a_rectangle, _frame_b_rectangle = create_pair_shift(_image_size_rectangle, _u_shift, _v_shift)


def test_gpu_extended_search_area_fast():
    """Quick test of the extanded search area function."""
    u, v = gpu_process.gpu_extended_search_area(
        _frame_a_rectangle, _frame_b_rectangle, window_size=16, overlap_ratio=0.5, search_area_size=32, dt=1
    )
    assert np.mean(np.abs(u - _u_shift)) < _threshold
    assert np.mean(np.abs(v + _v_shift)) < _threshold


def test_gpu_piv_fast_rectangle():
    """Quick test of the main piv function."""
    x, y, u, v, mask, s2n = gpu_process.gpu_piv(
        _frame_a_rectangle,
        _frame_b_rectangle,
        mask=None,
        window_size_iters=(1, 2),
        min_window_size=16,
        overlap_ratio=0.5,
        dt=1,
        deform=True,
        smooth=True,
        nb_validation_iter=1,
        validation_method="median_velocity",
        trust_1st_iter=True,
    )
    assert np.mean(np.abs(u - _u_shift)) < _threshold
    assert np.mean(np.abs(v + _v_shift)) < _threshold


def test_gpu_piv_fast_square():
    """Quick test of the main piv function."""
    x, y, u, v, mask, s2n = gpu_process.gpu_piv(
        _frame_a_square,
        _frame_b_square,
        mask=None,
        window_size_iters=(1, 2),
        min_window_size=16,
        overlap_ratio=0.5,
        dt=1,
        deform=True,
        smooth=True,
        nb_validation_iter=1,
        validation_method="median_velocity",
        trust_1st_iter=True,
    )
    assert np.mean(np.abs(u - _u_shift)) < _threshold
    assert np.mean(np.abs(v + _v_shift)) < _threshold


# def test_gpu_piv_square():
#     """Performs the PIV computation on a pair of square images."""
#     pass
#
#
# def gpu_piv_py():
#     """Ensures the results of the GPU algorithm remains unchanged."""
#     pass
#
#
# def test_correlation_function():
#     pass
#
#
# def test_deform():
#     pass
#
#
# def test_gpu_validation_mean():
#     """Validates a field with exactly one spurious vector"""
#     pass
#
#
# def test_gpu_validation_median():
#     """Validates a field with exactly one spurious vector"""
#     pass
#
#
# def test_gpu_validation_():
#     """Validates a field with exactly one spurious vector"""
#     pass
#
#
# def test_gpu_validation_s2n():
#     """Validates a field with exactly one spurious vector"""
#     pass
#
#
# def test_replace_vectors():
#     pass
#
#
# def test_get_field_shape():
#     """Validates a field with exactly one spurious vector"""
#     # test square field
#
#     # test rectangular field
#
#     pass
#
#
# def test_pycuda():
#     # define gpu array
#     a = np.random.randn(4, 4)
#     a = a.astype(np.float32)
#     a_gpu = drv.mem_alloc(a.nbytes)
#     drv.memcpy_htod(a_gpu, a)
#
#     # define the doubling function
#     mod = SourceModule(
#         """
#       __global__ void doublify(float *a)
#       {
#         int idx = threadIdx.x + threadIdx.y*4;
#         a[idx] *= 2;
#       }
#       """
#     )
#     func = mod.get_function("doublify")
#     func(a_gpu, block=(4, 4, 1))
#
#     # double the gpu array
#     a_doubled = np.empty_like(a)
#     drv.memcpy_dtoh(a_doubled, a_gpu)
#
#
# def test_gpu_array():
#     """If this test fails, then the CUDA version is likely not working with PyCUDA."""
#     a = gpuarray.zeros((2, 2, 2), dtype=np.float32)
#     b = gpuarray.zeros((2, 2, 2), dtype=np.int32)
#     c = gpuarray.empty((2, 2, 2), dtype=np.float32)
#     d = gpuarray.empty((2, 2, 2), dtype=np.int32)
