# (venv) C:\games\nunif>python
# Python 3.10.11 (tags/v3.10.11:7d4cc5a, Apr  5 2023, 00:38:17) [MSC v.1929 64 bit (AMD64)] on win32
# Type "help", "copyright", "credits" or "license" for more information.
# >>> import tensorrt
# >>> print(tensorrt.__version__)
# 10.11.0.33

import torch
import numpy as np
import tensorrt as trt
from iw3.utils import HUB_MODEL_DIR
from nunif.logger import logger
from nunif.utils.ui import TorchHubDir, HiddenPrints
from os import path
import os
import subprocess
import ctypes
from cuda.bindings import runtime as cuda_runtime
from PIL import Image
from torchvision import transforms
from nunif.device import create_device
from nunif.models import load_model
from iw3.end_to_end import End2End

# This script is adapted from the TensorRT Python examples and tailored for this project.
# See: https://leimao.github.io/blog/TensorRT-Python-Inference/
# and https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/python-api-docs.html

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def check_cuda_err(err):
    if isinstance(err, cuda_runtime.cudaError_t):
        if err != cuda_runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA Runtime Error: {err}")
    else:
        raise RuntimeError(f"Unknown error type: {err}")

def cuda_call(call):
    err, *res = call
    check_cuda_err(err)
    if len(res) == 1:
        return res[0]
    return res

class HostDeviceMem:
    """A helper class for managing paired host and device memory."""
    def __init__(self, size: int, dtype: np.dtype, name: str, shape, format):
        nbytes = size * dtype.itemsize
        host_mem = cuda_call(cuda_runtime.cudaMallocHost(nbytes))
        pointer_type = ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype))
        self._host = np.ctypeslib.as_array(ctypes.cast(host_mem, pointer_type), (size, ))
        self._device = cuda_call(cuda_runtime.cudaMalloc(nbytes))
        self._nbytes = nbytes
        self._name = name
        self._shape = shape
        self._format = format
        self._dtype = dtype

    @property
    def host(self) -> np.ndarray:
        return self._host

    @host.setter
    def host(self, arr: np.ndarray):
        if arr.size > self.host.size:
            raise ValueError(
                f"Tried to fit an array of size {arr.size} into host memory of size {self.host.size}"
            )
        np.copyto(self.host[:arr.size], arr.flat, casting='safe')

    @property
    def device(self) -> int:
        return self._device

    @property
    def nbytes(self) -> int:
        return self._nbytes

    @property
    def name(self) -> str:
        return self._name

    @property
    def shape(self) -> tuple:
        return self._shape

    @property
    def format(self):
        return self._format

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def free(self):
        cuda_call(cuda_runtime.cudaFree(self.device))
        cuda_call(cuda_runtime.cudaFreeHost(self.host.ctypes.data))


def allocate_buffers(engine: trt.ICudaEngine, profile_idx: int = 0):
    inputs = []
    outputs = []
    stream = cuda_call(cuda_runtime.cudaStreamCreate())
    tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]

    for binding in tensor_names:
        shape = engine.get_tensor_profile_shape(binding, profile_idx)[-1] # max shape
        size = trt.volume(shape)
        dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(binding)))
        
        mem = HostDeviceMem(size, dtype, binding, shape, engine.get_tensor_format(binding))
        
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            inputs.append(mem)
        else:
            outputs.append(mem)
            
    output_map = {o.name: o for o in outputs}
    # To maintain order
    outputs = [output_map[name] for name in ["left", "right"]]

    return inputs, outputs, stream

def do_inference(context, inputs, outputs, stream):
    # Transfer input data to the GPU.
    kind = cuda_runtime.cudaMemcpyKind.cudaMemcpyHostToDevice
    for inp in inputs:
        cuda_call(cuda_runtime.cudaMemcpyAsync(inp.device, inp.host.ctypes.data, inp.nbytes, kind, stream))

    # Run inference
    context.set_input_shape(inputs[0].name, inputs[0].shape)
    context.execute_async_v3(stream_handle=stream)

    # Transfer predictions back from the GPU.
    kind = cuda_runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost
    for out in outputs:
        cuda_call(cuda_runtime.cudaMemcpyAsync(out.host.ctypes.data, out.device, out.nbytes, kind, stream))
        
    # Synchronize the stream
    cuda_call(cuda_runtime.cudaStreamSynchronize(stream))
    
    return [out.host for out in outputs]

def test_end2end_tensorrt_model():
    """
    Test End2End Stereo TensorRT model output against PyTorch model output
    """
    print(f"Testing End2End Stereo TensorRT model...")

    model_path = path.join(HUB_MODEL_DIR, "iw3", "ZoeD_M12_N_convnext-xlarge-aug-scale-all_end2end_2023-11-20.pth")
    if not path.exists(model_path):
        print(f"PyTorch model not found at {model_path}, skipping test.")
        # NOTE: You may need to run `python -m iw3.download_models` first.
        return

    trt_path = 'stereo_module.trt'
    if not path.exists(trt_path):
        print(f"TensorRT model not found at {trt_path}, skipping test.")
        # NOTE: You may need to run `iw3/export_trt.bat` first.
        return
    img_path1 = "iw3/figure/convergence.png"

    logger.debug(f"load an image")
    img1 = Image.open(img_path1).convert("RGB")

    # Get PyTorch output
    device = create_device("cuda:0")
    model, _ = load_model(model_path, device=device)
    model.eval()
    t = transforms.ToTensor()
    test_input = t(img1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        torch_left, torch_right = model(test_input)
    torch_left = torch_left.squeeze(0).cpu().numpy()
    torch_right = torch_right.squeeze(0).cpu().numpy()
    test_input_np = test_input.cpu().numpy()

    # Get TensorRT output
    runtime = trt.Runtime(TRT_LOGGER)
    with open(trt_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    
    # 1. Allocate input/output buffers for max shape
    inputs, outputs, stream = allocate_buffers(engine)
    assert len(inputs) == 1
    assert len(outputs) == 2
    
    # 2. Set input shape for this inference
    # Unlike the original code, we don't need to call `set_input_shape` here
    # because `execute_async_v3` will handle it. But we need to update the shape property.
    inputs[0]._shape = test_input.shape

    # 3. Query actual output shapes (after setting input shape)
    context.set_input_shape(inputs[0].name, test_input.shape)
    left_shape = context.get_tensor_shape("left")
    right_shape = context.get_tensor_shape("right")
    left_size = trt.volume(left_shape)
    right_size = trt.volume(right_shape)

    # 4. Set tensor addresses for all inputs and outputs
    bindings = [None] * engine.num_io_tensors
    for mem in inputs + outputs:
        idx = engine.get_tensor_location(mem.name)
        bindings[idx] = mem.device
    context.set_tensor_address("input", inputs[0].device)
    context.set_tensor_address("left", outputs[0].device)
    context.set_tensor_address("right", outputs[1].device)

    # 5. Set input data
    inputs[0].host = test_input_np.ravel()

    # 6. Run inference
    trt_raw_outputs = do_inference(context, inputs, outputs, stream)
    
    # 7. Reshape output from the pre-allocated buffers
    trt_left = trt_raw_outputs[0][:left_size].reshape(left_shape)
    trt_right = trt_raw_outputs[1][:right_size].reshape(right_shape)

    # Compare outputs
    # Left eye
    max_diff_left = np.max(np.abs(torch_left - trt_left))
    mean_diff_left = np.mean(np.abs(torch_left - trt_left))

    print(f"Left Eye - Maximum absolute difference: {max_diff_left}")
    print(f"Left Eye - Mean absolute difference: {mean_diff_left}")

    # Right eye
    max_diff_right = np.max(np.abs(torch_right - trt_right))
    mean_diff_right = np.mean(np.abs(torch_right - trt_right))

    print(f"Right Eye - Maximum absolute difference: {max_diff_right}")
    print(f"Right Eye - Mean absolute difference: {mean_diff_right}")

    # Check if the differences are within acceptable range for FP16
    if max_diff_left > 1e-2 or max_diff_right > 1e-2:
        print("WARNING: Large difference detected between PyTorch and TensorRT outputs!")
    else:
        print("TensorRT model output matches PyTorch model output within acceptable range.")

    # Free memory
    for i in inputs: i.free()
    for o in outputs: o.free()


if __name__ == "__main__":
    test_end2end_tensorrt_model()

