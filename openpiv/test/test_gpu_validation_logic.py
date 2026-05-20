import importlib.util
import sys
import types
from pathlib import Path


class _Descriptor:
    def __init__(self, *args, **kwargs):
        self.storage_name = None

    def __set_name__(self, owner, name):
        self.storage_name = f"_{name}_value"

    def __get__(self, obj, owner=None):
        if obj is None:
            return self
        return getattr(obj, self.storage_name, None)

    def __set__(self, obj, value):
        setattr(obj, self.storage_name, value)


def load_gpu_validation_module():
    validation_path = Path(__file__).resolve().parents[1] / "gpu" / "validation.py"

    pycuda = types.ModuleType("pycuda")
    pycuda_gpuarray = types.ModuleType("pycuda.gpuarray")
    pycuda_gpuarray.GPUArray = type("GPUArray", (), {})
    pycuda_compiler = types.ModuleType("pycuda.compiler")

    class DummySourceModule:
        def __init__(self, source):
            self.source = source

        def get_function(self, name):
            def _function(*args, **kwargs):
                raise AssertionError(f"Unexpected kernel call: {name}")

            return _function

    pycuda_compiler.SourceModule = DummySourceModule
    pycuda.gpuarray = pycuda_gpuarray

    openpiv_gpu = types.ModuleType("openpiv.gpu")
    openpiv_gpu.DTYPE_i = int
    openpiv_gpu.DTYPE_f = float

    openpiv_gpu_misc = types.ModuleType("openpiv.gpu.misc")
    openpiv_gpu_misc._Subset = _Descriptor
    openpiv_gpu_misc._Number = _Descriptor
    openpiv_gpu_misc._check_arrays = lambda *args, **kwargs: None
    openpiv_gpu_misc.gpu_mask = lambda values, mask: values
    openpiv_gpu.misc = openpiv_gpu_misc

    original_modules = {}
    stubbed_modules = {
        "pycuda": pycuda,
        "pycuda.gpuarray": pycuda_gpuarray,
        "pycuda.compiler": pycuda_compiler,
        "openpiv.gpu": openpiv_gpu,
        "openpiv.gpu.misc": openpiv_gpu_misc,
    }
    for name, module in stubbed_modules.items():
        original_modules[name] = sys.modules.get(name)
        sys.modules[name] = module

    module_name = "openpiv.test._gpu_validation_under_test"
    spec = importlib.util.spec_from_file_location(module_name, validation_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(module_name, None)
        for name, original_module in original_modules.items():
            if original_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original_module


def test_mean_validation_uses_mean_tol():
    validation = load_gpu_validation_module()
    calls = []

    validation._gpu_residual = lambda *args, **kwargs: "residual"

    def neighbour_validation(f, f_mean, residual, tol, val_locations=None):
        calls.append(tol)
        return val_locations

    validation._neighbour_validation = neighbour_validation

    validation_gpu = validation.Validation.__new__(validation.Validation)
    validation_gpu._num_fields = 2
    validation_gpu.mean_tol = 1.5
    validation_gpu.median_tol = 9.5
    validation_gpu._f = ["u", "v"]
    validation_gpu._mean_ = ["u_mean", "v_mean"]
    validation_gpu._neighbours_ = ["u_neighbours", "v_neighbours"]
    validation_gpu._neighbours_present = "neighbours_present"
    validation_gpu.val_locations = None

    validation_gpu._mean_validation()

    assert calls == [1.5, 1.5]
