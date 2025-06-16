# (venv) C:\games\nunif>python
# Python 3.10.11 (tags/v3.10.11:7d4cc5a, Apr  5 2023, 00:38:17) [MSC v.1929 64 bit (AMD64)] on win32
# Type "help", "copyright", "credits" or "license" for more information.
# >>> import tensorrt
# >>> print(tensorrt.__version__)
# 10.11.0.33

import torch
import numpy as np
import tensorrt as trt
from iw3.export_onnx import export_distill_any_depth_to_onnx, HUB_MODEL_DIR, DistillAnyDepthWithDilation
from nunif.utils.ui import TorchHubDir
from os import path
import os
import subprocess
import ctypes
from cuda import cudart

# This script is adapted from the TensorRT Python examples and tailored for this project.
# See: https://leimao.github.io/blog/TensorRT-Python-Inference/
# and https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/python-api-docs.html

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def check_cuda_err(err):
    if isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
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
        host_mem = cuda_call(cudart.cudaMallocHost(nbytes))
        pointer_type = ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype))
        self._host = np.ctypeslib.as_array(ctypes.cast(host_mem, pointer_type), (size, ))
        self._device = cuda_call(cudart.cudaMalloc(nbytes))
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
        cuda_call(cudart.cudaFree(self.device))
        cuda_call(cudart.cudaFreeHost(self.host.ctypes.data))


def allocate_buffers(engine: trt.ICudaEngine, profile_idx: int = 0):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda_call(cudart.cudaStreamCreate())
    tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]

    for binding in tensor_names:
        shape = engine.get_tensor_profile_shape(binding, profile_idx)[-1] # max shape
        size = trt.volume(shape)
        dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(binding)))
        
        mem = HostDeviceMem(size, dtype, binding, shape, engine.get_tensor_format(binding))
        bindings.append(int(mem.device))
        
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            inputs.append(mem)
        else:
            outputs.append(mem)
            
    return inputs, outputs, bindings, stream

def do_inference(context, inputs, outputs, stream):
    # Transfer input data to the GPU.
    kind = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
    for inp in inputs:
        cuda_call(cudart.cudaMemcpyAsync(inp.device, inp.host.ctypes.data, inp.nbytes, kind, stream))

    # Run inference
    context.execute_async_v3(stream_handle=stream)

    # Transfer predictions back from the GPU.
    kind = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
    for out in outputs:
        cuda_call(cudart.cudaMemcpyAsync(out.host.ctypes.data, out.device, out.nbytes, kind, stream))
        
    # Synchronize the stream
    cuda_call(cudart.cudaStreamSynchronize(stream))
    
    return [out.host for out in outputs]

def reallocate_output_buffers(engine, context, outputs):
    for i, out in enumerate(outputs):
        shape = context.get_tensor_shape(out.name)
        size = trt.volume(shape)
        dtype = out.dtype
        # Free old memory
        out.free()
        # Allocate new memory
        outputs[i] = HostDeviceMem(size, dtype, out.name, shape, out.format)

def test_tensorrt_model(model_size='s', input_size=392):
    """
    Test TensorRT model output against PyTorch model output
    """
    print(f"Testing Distill Any Depth {model_size.upper()} TensorRT model...")
    
    onnx_path = path.join(HUB_MODEL_DIR, f"distill_any_depth_{model_size}.onnx")
    trt_path = path.join(HUB_MODEL_DIR, f"distill_any_depth_{model_size}.trt")

    # Export ONNX model if it doesn't exist
    if not path.exists(onnx_path):
        with TorchHubDir(HUB_MODEL_DIR):
            export_distill_any_depth_to_onnx(model_size, input_size)

    # Build TensorRT engine if it doesn't exist
    if not path.exists(trt_path):
        print("TensorRT engine not found, building...")
        command = [
            "trtexec",
            f"--onnx={onnx_path}",
            f"--saveEngine={trt_path}",
            "--fp16",
            f"--minShapes=input:1x3x{input_size}x{input_size}",
            f"--optShapes=input:1x3x{input_size}x{input_size}",
            "--maxShapes=input:1x3x1024x1024"
        ]
        print(f"Running: {' '.join(command)}")
        subprocess.run(command, check=True)
        print("TensorRT engine built.")

    # Load PyTorch model
    size_to_encoder = {'s': 'v2_vits', 'b': 'v2_vitb', 'l': 'v2_vitl'}
    encoder = size_to_encoder[model_size]
    model = torch.hub.load("nagadomi/Depth-Anything_iw3:main",
                          "DistillAnyDepth", encoder=encoder,
                          verbose=False, trust_repo=True)
    model.eval()
    model = DistillAnyDepthWithDilation(model, edge_dilation=2)
    
    # Create test input
    test_input = torch.randn(1, 3, input_size, input_size)
    
    # Get PyTorch output
    with torch.no_grad():
        torch_output = model(test_input).numpy()

    # Get TensorRT output
    runtime = trt.Runtime(TRT_LOGGER)
    with open(trt_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    
    # Set input shape for dynamic shape execution
    context.set_input_shape("input", test_input.shape)
    
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    
    # Re-allocate output buffers for the actual output shape
    reallocate_output_buffers(engine, context, outputs)
    
    # Set tensor addresses for v3 execution
    for i in range(engine.num_io_tensors):
        context.set_tensor_address(engine.get_tensor_name(i), bindings[i])

    # Set input data
    inputs[0].host = test_input.numpy().ravel()
    
    # Run inference
    trt_raw_output = do_inference(context, inputs, outputs, stream)[0]
    
    # Reshape output
    output_shape = context.get_tensor_shape(outputs[0].name)
    trt_output = trt_raw_output.reshape(output_shape)

    # Compare outputs
    max_diff = np.max(np.abs(torch_output - trt_output))
    mean_diff = np.mean(np.abs(torch_output - trt_output))
    
    print(f"Maximum absolute difference: {max_diff}")
    print(f"Mean absolute difference: {mean_diff}")
    
    # Check if the differences are within acceptable range for FP16
    if max_diff > 1e-2:
        print("WARNING: Large difference detected between PyTorch and TensorRT outputs!")
    else:
        print("TensorRT model output matches PyTorch model output within acceptable range.")

    # Free memory
    for i in inputs: i.free()
    for o in outputs: o.free()


if __name__ == "__main__":
    with TorchHubDir(HUB_MODEL_DIR):
        test_tensorrt_model('s')

