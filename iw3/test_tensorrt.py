# (venv) C:\games\nunif>python
# Python 3.10.11 (tags/v3.10.11:7d4cc5a, Apr  5 2023, 00:38:17) [MSC v.1929 64 bit (AMD64)] on win32
# Type "help", "copyright", "credits" or "license" for more information.
# >>> import tensorrt
# >>> print(tensorrt.__version__)
# 10.11.0.33

import torch
import numpy as np
import tensorrt as trt
from iw3.export_onnx_old import (
    export_end2end_to_onnx, HUB_MODEL_DIR,
    End2EndStereoModel, DistillAnyDepthWithDilation
)
from nunif.utils.ui import TorchHubDir, HiddenPrints
from os import path
import os
import subprocess
import ctypes
from cuda.bindings import runtime as cuda_runtime
from iw3.utils import ROW_FLOW_V3_SYM_URL
from nunif.models import load_model

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
    bindings = []
    stream = cuda_call(cuda_runtime.cudaStreamCreate())
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
            
    output_map = {o.name: o for o in outputs}
    # To maintain order
    outputs = [output_map[name] for name in ["left", "right"]]

    return inputs, outputs, bindings, stream

def do_inference(context, inputs, outputs, stream):
    # Transfer input data to the GPU.
    kind = cuda_runtime.cudaMemcpyKind.cudaMemcpyHostToDevice
    for inp in inputs:
        cuda_call(cuda_runtime.cudaMemcpyAsync(inp.device, inp.host.ctypes.data, inp.nbytes, kind, stream))

    # Run inference
    context.execute_async_v3(stream_handle=stream)

    # Transfer predictions back from the GPU.
    kind = cuda_runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost
    for out in outputs:
        cuda_call(cuda_runtime.cudaMemcpyAsync(out.host.ctypes.data, out.device, out.nbytes, kind, stream))
        
    # Synchronize the stream
    cuda_call(cuda_runtime.cudaStreamSynchronize(stream))
    
    return [out.host for out in outputs]

def test_end2end_tensorrt_model(model_size='s', input_size=392, edge_dilation=2, divergence=2.0, convergence=0.5):
    """
    Test End2End Stereo TensorRT model output against PyTorch model output
    """
    print(f"Testing End2End Stereo {model_size.upper()} TensorRT model...")

    onnx_path = path.join(HUB_MODEL_DIR, f"end2end_stereo_{model_size}.onnx")
    trt_path = path.join(HUB_MODEL_DIR, f"end2end_stereo_{model_size}.trt")

    # Export ONNX model if it doesn't exist
    if not path.exists(onnx_path):
        with TorchHubDir(HUB_MODEL_DIR):
            export_end2end_to_onnx(model_size=model_size, input_size=input_size,
                                   edge_dilation=edge_dilation, divergence=divergence, convergence=convergence)

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
            f"--maxShapes=input:1x3x1024x1024",
            # Also specify output shapes for robustness
            f"--minShapes=left:1x3x{input_size}x{input_size},right:1x3x{input_size}x{input_size}",
            f"--optShapes=left:1x3x{input_size}x{input_size},right:1x3x{input_size}x{input_size}",
            f"--maxShapes=left:1x3x1024x1024,right:1x3x1024x1024"
        ]
        print(f"Running: {' '.join(command)}")
        subprocess.run(command, check=True)
        print("TensorRT engine built.")

    # Load PyTorch model
    # This logic must be the same as in export_end2end_to_onnx
    encoder = {'s': 'v2_vits', 'b': 'v2_vitb', 'l': 'v2_vitl'}[model_size]
    with HiddenPrints(), TorchHubDir(HUB_MODEL_DIR):
        depth_model_pt = torch.hub.load("nagadomi/Depth-Anything_iw3:main",
                                        "DistillAnyDepth", encoder=encoder,
                                        verbose=False, trust_repo=True)
    depth_model_pt.eval()
    depth_model_pt = DistillAnyDepthWithDilation(depth_model_pt, edge_dilation=edge_dilation)

    side_model_path = ROW_FLOW_V3_SYM_URL
    with TorchHubDir(HUB_MODEL_DIR):
        side_model_pt = load_model(side_model_path, weights_only=True, device_ids=[-1])[0].eval()
    side_model_pt.delta_output = True
    side_model_pt.symmetric = True

    model = End2EndStereoModel(depth_model_pt, side_model_pt, divergence, convergence)
    model.eval()

    # Create test input
    test_input = torch.randn(1, 3, input_size, input_size)

    # Get PyTorch output
    with torch.no_grad():
        torch_left, torch_right = model(test_input)
        torch_left = torch_left.numpy()
        torch_right = torch_right.numpy()

    # Get TensorRT output
    runtime = trt.Runtime(TRT_LOGGER)
    with open(trt_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    
    # 1. Allocate input buffers (max shape is fine)
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    # 2. Set input shape
    context.set_input_shape(inputs[0].name, test_input.shape)

    # 3. Infer shapes
    context.infer_shapes()

    # 4. Query actual output shapes
    left_shape = context.get_tensor_shape("left")
    right_shape = context.get_tensor_shape("right")
    left_size = trt.volume(left_shape)
    right_size = trt.volume(right_shape)
    left_dtype = np.dtype(trt.nptype(engine.get_tensor_dtype("left")))
    right_dtype = np.dtype(trt.nptype(engine.get_tensor_dtype("right")))

    # 5. Allocate output buffers for actual shapes
    left_mem = HostDeviceMem(left_size, left_dtype, "left", left_shape, engine.get_tensor_format("left"))
    right_mem = HostDeviceMem(right_size, right_dtype, "right", right_shape, engine.get_tensor_format("right"))
    outputs = [left_mem, right_mem]

    # 6. Set tensor addresses for all inputs and outputs
    for mem in inputs + outputs:
        context.set_tensor_address(mem.name, mem.device)

    # 7. Set input data
    inputs[0].host = test_input.numpy().ravel()

    # 8. Run inference
    trt_raw_outputs = do_inference(context, inputs, outputs, stream)
    
    # 9. Reshape output
    trt_left = trt_raw_outputs[0].reshape(left_shape)
    trt_right = trt_raw_outputs[1].reshape(right_shape)

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
    with TorchHubDir(HUB_MODEL_DIR):
        test_end2end_tensorrt_model('s')

