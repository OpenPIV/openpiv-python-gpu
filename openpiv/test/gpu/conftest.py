import pytest
import numpy as np

from imageio.v2 import imread
import pycuda.gpuarray as gpuarray

import gpu_process
import gpu_validation

# GLOBAL VARIABLES
DTYPE_i = np.int32
DTYPE_f = np.float32

# dirs
data_dir = "../data/"

# test data
frame_a = imread(data_dir + "test1/exp1_001_a.bmp").astype(np.float32)
frame_b = imread(data_dir + "test1/exp1_001_b.bmp").astype(np.float32)


# UTILS
def pytest_addoption(parser):
    parser.addoption(
        "--regression", action="store_true", default=False, help="run regression tests"
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "regression: mark test as using " "pytest-regression"
    )


def pytest_collection_modifyitems(config, items):
    """"""
    if config.getoption("--regression"):
        return
    skip_regression = pytest.mark.skip(reason="need --regression option")
    for item in items:
        if "regression" in item.keywords:
            item.add_marker(skip_regression)


def generate_boolean_np_array(shape, d_type=DTYPE_i, seed=0):
    """Returns ndarray with pseudo-random boolean values."""
    return generate_np_array(shape, center=1.0, d_type=DTYPE_i, seed=seed).astype(
        d_type
    )


def generate_boolean_gpu_array(shape, d_type=DTYPE_i, seed=0):
    """Returns GPUArray with pseudo-random boolean values."""
    f = generate_boolean_np_array(shape, d_type=d_type, seed=seed)
    return gpuarray.to_gpu(f)


def generate_boolean_array_pair(shape, d_type=DTYPE_i, seed=0):
    """Returns a pair of numpy and gpu arrays with identical pseudo-random values."""
    f = generate_boolean_np_array(shape, d_type=d_type, seed=seed)
    f_d = gpuarray.to_gpu(f)

    return f, f_d


def generate_np_array(shape, center=0.5, half_width=0.5, d_type=DTYPE_f, seed=0):
    """Returns ndarray with pseudo-random values."""
    np.random.seed(seed)
    return ((np.random.random(shape) - 0.5) * 2 * half_width + center).astype(d_type)


def generate_gpu_array(shape, center=0.5, half_width=0.5, d_type=DTYPE_f, seed=0):
    """Returns GPUArray with pseudo-random values."""
    f = generate_np_array(
        shape, center=center, half_width=half_width, d_type=d_type, seed=seed
    )
    return gpuarray.to_gpu(f)


def generate_array_pair(shape, center=0.5, half_width=0.5, d_type=DTYPE_f, seed=0):
    """Returns a pair of numpy and gpu arrays with identical pseudo-random values."""
    f = generate_np_array(
        shape, center=center, half_width=half_width, d_type=d_type, seed=seed
    )
    f_d = gpuarray.to_gpu(f)

    return f, f_d


def generate_np_array_old(shape, magnitude=1.0, offset=0.0, d_type=DTYPE_f, seed=0):
    """Returns ndarray with pseudo-random values."""
    np.random.seed(seed)
    return (np.random.random(shape) * magnitude + offset).astype(d_type)


def generate_gpu_array_old(shape, magnitude=1.0, offset=0.0, d_type=DTYPE_f, seed=0):
    """Returns GPUArray with pseudo-random values."""
    f = generate_np_array_old(
        shape, magnitude=magnitude, offset=offset, d_type=d_type, seed=seed
    )
    return gpuarray.to_gpu(f)


def generate_array_pair_old(shape, magnitude=1.0, offset=0.0, d_type=DTYPE_f, seed=0):
    """Returns a pair of numpy and gpu arrays with identical pseudo-random values."""
    f = generate_np_array_old(
        shape, magnitude=magnitude, offset=offset, d_type=d_type, seed=seed
    )
    f_d = gpuarray.to_gpu(f)

    return f, f_d


# FIXTURES
@pytest.fixture
def boolean_np_array():
    return generate_boolean_np_array


@pytest.fixture
def boolean_gpu_array():
    return generate_boolean_gpu_array


@pytest.fixture
def boolean_array_pair():
    return generate_boolean_array_pair


@pytest.fixture
def np_array():
    return generate_np_array


@pytest.fixture
def gpu_array():
    return generate_gpu_array


@pytest.fixture
def array_pair():
    return generate_array_pair


@pytest.fixture
def piv_field_gpu():
    frame_shape = (512, 512)
    window_size = 32
    spacing = 16

    piv_field_gpu = gpu_process.PIVFieldGPU(frame_shape, window_size, spacing)

    return piv_field_gpu


@pytest.fixture
def correlation_gpu(piv_field_gpu):
    frame_a_d = gpuarray.to_gpu(frame_a)
    frame_b_d = gpuarray.to_gpu(frame_b)

    correlation_gpu = gpu_process.CorrelationGPU(frame_a_d, frame_b_d)
    correlation_gpu(piv_field_gpu)

    return correlation_gpu


@pytest.fixture
def piv_field(correlation_gpu):
    window_size = 32
    spacing = 16
    frame_shape = correlation_gpu.frame_shape

    return gpu_process.PIVFieldGPU(frame_shape, window_size, spacing)


@pytest.fixture
def peaks(correlation_gpu, piv_field):
    i_peaks, j_peaks = correlation_gpu(piv_field)

    return i_peaks, j_peaks


@pytest.fixture
def mask(peaks, boolean_gpu_array):
    i_peaks, _ = peaks

    mask = boolean_gpu_array(i_peaks.shape, seed=0)

    return mask


@pytest.fixture
def s2n_ratio(correlation_gpu, piv_field):
    _, _ = correlation_gpu(piv_field)
    sig2noise = correlation_gpu.s2n_ratio

    return sig2noise


@pytest.fixture
def validation_gpu(peaks, boolean_gpu_array):
    i_peaks, _ = peaks

    mask = boolean_gpu_array(i_peaks.shape, seed=1)

    validation_gpu = gpu_validation.ValidationGPU(
        i_peaks,
        mask=mask,
        validation_method="median_velocity",
        s2n_tol=2,
        median_tol=2,
        mean_tol=2,
        rms_tol=2,
    )

    return validation_gpu


@pytest.fixture
def piv_gpu():
    piv_gpu = gpu_process.PIVGPU(frame_a)
    piv_gpu(frame_a, frame_b)

    return piv_gpu


@pytest.fixture()
def param_iter(**params_dict):
    """Returns list of input parameters from lists of parameters.

    Each list of parameters is swept while using the default (first) value for all
    other parameters. The resulting list of parameters increase linearly with
    additional parameterization.
    """
    keys = params_dict.keys()

    for i, key_i in enumerate(keys):
        for value in params_dict[key_i]:
            test_params = {}
            for j, key_j in enumerate(keys):
                if i == j:
                    test_params[key_j] = value
                else:
                    test_params[key_j] = params_dict[key_j][0]
            yield test_params
