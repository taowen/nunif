import tensorrt as trt # version 10.11.0.33
import torch
from torchvision.transforms import functional as TF
from PIL import Image
from nunif.logger import logger
import numpy as np
import cuda.cudart as cudart

def check_cuda_error(cuda_ret):
    err, *rest = cuda_ret
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {cudart.cudaGetErrorString(err)[1].decode('utf-8')}")
    if len(rest) == 1:
        return rest[0]
    return rest

img_path1 = "iw3/figure/convergence.png"
img_path2 = "iw3/figure/divergence.png"
img1 = Image.open(img_path1).convert("RGB")
img2 = Image.open(img_path2).convert("RGB")

# The TRT engine was built with a specific input size range.
# Valid range for profile 0: [1,3,392,392]..[1,3,2160,3840].
# Resize to a valid size while maintaining aspect ratio.
w, h = img1.size
if h < 392:
    new_h = 392
    new_w = int(w * new_h / h)
    img1 = img1.resize((new_w, new_h), Image.Resampling.BICUBIC)
    img2 = img2.resize((new_w, new_h), Image.Resampling.BICUBIC)

x1 = TF.to_tensor(img1)
x2 = TF.to_tensor(img2)
logger.debug(f"x1 shape: {x1.shape}, x2 shape: {x2.shape}")
# x1 shape: torch.Size([3, 260, 780]), x2 shape: torch.Size([3, 260, 780])
x_batch = torch.stack([x1, x2], dim=0)  # BCHW, float32, 0-1
logger.debug(f"stacked x shape: {x_batch.shape}")
# stacked x shape: torch.Size([2, 3, 260, 780])

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
engine_path = "stereo_module.trt"
logger.info(f"Loading engine: {engine_path}")
with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

outputs = []
for i in range(x_batch.shape[0]):
    x = x_batch[i:i + 1]
    logger.info(f"Processing image {i}, shape: {x.shape}")

    # I/O bindings
    input_bindings = []
    output_bindings = []
    output_tensors = []
    bindings = [0] * engine.num_io_tensors

    for binding_idx in range(engine.num_io_tensors):
        binding_name = engine.get_tensor_name(binding_idx)
        if engine.get_tensor_mode(binding_name) == trt.TensorIOMode.INPUT:
            if binding_name == 'input':  # from export_trt.bat
                context.set_input_shape(binding_name, x.shape)
                h_input = x.contiguous()
                d_input = check_cuda_error(cudart.cudaMalloc(h_input.nbytes))
                check_cuda_error(cudart.cudaMemcpy(d_input, h_input.data_ptr(), h_input.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))
                bindings[binding_idx] = int(d_input)
                input_bindings.append(d_input)
        else:
            output_shape = context.get_tensor_shape(binding_name)
            dtype = trt.nptype(engine.get_tensor_dtype(binding_name))
            # HACK: convert numpy dtype to torch dtype
            output_dtype = torch.from_numpy(np.array(0, dtype=dtype)).dtype
            h_output = torch.empty(tuple(output_shape), dtype=output_dtype)
            d_output = check_cuda_error(cudart.cudaMalloc(h_output.nbytes))
            bindings[binding_idx] = int(d_output)
            output_bindings.append(d_output)
            output_tensors.append(h_output)

    context.execute_v2(bindings)

    for h_output, d_output in zip(output_tensors, output_bindings):
        check_cuda_error(cudart.cudaMemcpy(h_output.data_ptr(), d_output, h_output.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))

    for d_mem in input_bindings + output_bindings:
        check_cuda_error(cudart.cudaFree(d_mem))

    # Assuming single output for simplicity in this example
    outputs.append(output_tensors[0])

output_batch = torch.cat(outputs, dim=0)
logger.info(f"Final output batch shape: {output_batch.shape}")
logger.debug(f"Output tensor (first 5 elements): {output_batch.flatten()[:5].cpu().numpy()}")
logger.debug(f"Output tensor dtype: {output_batch.dtype}")

# === 读取导出的图片并对比 ===
from PIL import Image
import torchvision.transforms.functional as TF

def load_eye_image(path, size):
    img = Image.open(path).convert("RGB")
    img = img.resize(size, Image.BICUBIC)  # size: (width, height)
    tensor = TF.to_tensor(img)
    return tensor

trt_left_0 = output_batch[0].cpu()
trt_right_0 = output_batch[1].cpu()
trt_h, trt_w = trt_left_0.shape[1], trt_left_0.shape[2]

left_eye_0 = load_eye_image("tmp/left_eye_0.png", (trt_w, trt_h))
right_eye_0 = load_eye_image("tmp/right_eye_0.png", (trt_w, trt_h))

def compare_tensors(t1, t2, name):
    diff = (t1 - t2).abs()
    print(f"{name}: max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f}")

compare_tensors(trt_left_0, left_eye_0, "Left Eye 0")
compare_tensors(trt_right_0, right_eye_0, "Right Eye 0")