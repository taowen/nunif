# input a rgb image from disk: waifu2x/docs/images/miku_128.png
# calc depth map using distll any depth small
# use row_flow_v3 with delta_output=True
# generate left right eye using the delta
# output left right eye nv12 image to disk
# Modified to accept uint8 NV12 input directly from video decoder

import torch
from torchvision.transforms import functional as TF
from PIL import Image
import os
import torch.nn as nn
import torch.onnx
import torch.nn.functional as F

from iw3.mapper import get_mapper, resolve_mapper_name
from nunif.utils.ui import TorchHubDir
from nunif.logger import logger
from iw3.backward_warp import apply_divergence_nn_LR

# NV12 conversion functions - modified to handle uint8 input/output
def rgb_to_nv12_uint8(rgb):
    """Convert RGB tensor to NV12 format (uint8)
    Args:
        rgb: (3, H, W) float tensor in range [0, 1]
    Returns:
        nv12: (H + H//2, W) uint8 tensor in range [0, 255]
    """
    C, H, W = rgb.shape
    
    # RGB to YUV conversion matrix
    rgb_to_yuv = torch.tensor([
        [0.299, 0.587, 0.114],
        [-0.169, -0.331, 0.500],
        [0.500, -0.419, -0.081]
    ], device=rgb.device, dtype=rgb.dtype)
    
    # Reshape for matrix multiplication
    rgb_flat = rgb.view(3, -1)  # (3, H*W)
    yuv_flat = torch.matmul(rgb_to_yuv, rgb_flat)  # (3, H*W)
    yuv = yuv_flat.view(3, H, W)
    
    # Extract Y, U, V and convert to [0, 255] range
    y = torch.clamp(yuv[0, :, :] * 255.0, 0, 255)  # (H, W)
    u = torch.clamp((yuv[1, :, :] + 0.5) * 255.0, 0, 255)  # (H, W)
    v = torch.clamp((yuv[2, :, :] + 0.5) * 255.0, 0, 255)  # (H, W)
    
    # Downsample UV to 4:2:0
    u_downsampled = F.avg_pool2d(u.unsqueeze(0).unsqueeze(0), kernel_size=2, stride=2)[0, 0]  # (H//2, W//2)
    v_downsampled = F.avg_pool2d(v.unsqueeze(0).unsqueeze(0), kernel_size=2, stride=2)[0, 0]  # (H//2, W//2)
    
    # Interleave U and V to create UV plane
    uv_plane = torch.stack([u_downsampled, v_downsampled], dim=-1)  # (H//2, W//2, 2)
    uv_plane = uv_plane.view(H//2, W)  # (H//2, W)
    
    # Concatenate Y and UV planes
    nv12 = torch.cat([y, uv_plane], dim=0)  # (H + H//2, W)
    
    # Convert to uint8
    return nv12.to(torch.uint8)

def nv12_uint8_to_rgb(nv12_uint8):
    """Convert NV12 format (uint8) to RGB tensor
    Args:
        nv12_uint8: (H + H//2, W) uint8 tensor in range [0, 255]
    Returns:
        rgb: (3, H, W) float tensor in range [0, 1]
    """
    # Convert to float and normalize to [0, 1]
    nv12 = nv12_uint8.float() / 255.0
    
    total_H, W = nv12.shape
    H = total_H * 2 // 3  # Original height
    
    # Split Y and UV planes
    y_plane = nv12[:H, :]  # (H, W)
    uv_plane = nv12[H:, :]  # (H//2, W)
    
    # Separate U and V from UV plane
    uv_reshaped = uv_plane.view(H//2, W//2, 2)  # (H//2, W//2, 2)
    u_downsampled = uv_reshaped[:, :, 0] - 0.5  # (H//2, W//2)
    v_downsampled = uv_reshaped[:, :, 1] - 0.5  # (H//2, W//2)
    
    # Upsample U and V to original resolution
    u = F.interpolate(u_downsampled.unsqueeze(0).unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False)[0, 0]
    v = F.interpolate(v_downsampled.unsqueeze(0).unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False)[0, 0]
    
    # YUV to RGB conversion matrix
    yuv_to_rgb = torch.tensor([
        [1.0, 0.0, 1.402],
        [1.0, -0.344136, -0.714136],
        [1.0, 1.772, 0.0]
    ], device=nv12.device, dtype=nv12.dtype)
    
    # Stack YUV channels
    yuv = torch.stack([y_plane, u, v], dim=0)  # (3, H, W)
    
    # Reshape for matrix multiplication
    yuv_flat = yuv.view(3, -1)  # (3, H*W)
    rgb_flat = torch.matmul(yuv_to_rgb, yuv_flat)  # (3, H*W)
    rgb = rgb_flat.view(3, H, W)
    
    # Clamp to valid range
    rgb = torch.clamp(rgb, 0.0, 1.0)
    
    return rgb

# Create test single frame from RGB image and convert to uint8 NV12
img_path1 = "iw3/figure/convergence.png"
logger.debug(f"load test image")
img1 = Image.open(img_path1).convert("RGB")
x1 = TF.to_tensor(img1)
logger.debug(f"x1 shape: {x1.shape}")

# Convert RGB to uint8 NV12 for testing
x_nv12_uint8_single = rgb_to_nv12_uint8(x1)
logger.debug(f"converted x_nv12_uint8_single shape: {x_nv12_uint8_single.shape}, dtype: {x_nv12_uint8_single.dtype}")

from iw3.depth_model_factory import create_depth_model
depth_model = create_depth_model("Distill_Any_S")
depth_model.load(gpu=[0], resolution=392)
from nunif.models import load_model
from iw3.utils import HUB_MODEL_DIR, ROW_FLOW_V3_SYM_URL

side_model_path = ROW_FLOW_V3_SYM_URL
with TorchHubDir(HUB_MODEL_DIR):
    side_model = load_model(side_model_path, weights_only=True, device_ids=[0])[0].eval()
side_model.delta_output = True
side_model.symmetric = True

mapper_name = resolve_mapper_name(mapper=None, foreground_scale=0.9, metric_depth=False)
logger.debug(f"mapper_name: {mapper_name}")
mapper = get_mapper(mapper_name)

class StereoDepthNV12Uint8Module(nn.Module):
    def __init__(self, depth_model, side_model):
        super().__init__()
        self.depth_model_wrapper = depth_model
        self.depth_model = depth_model.model
        self.side_model_wrapper = side_model
        self.side_model = side_model.model if hasattr(side_model, "model") else side_model

    def forward(self, x_nv12_uint8):
        # x_nv12_uint8: (H + H//2, W) uint8, NV12 format for single frame, range [0, 255]
        # Ensure input is 2D
        assert x_nv12_uint8.ndim == 2, f"Expected 2D input, got {x_nv12_uint8.ndim}D: {x_nv12_uint8.shape}"
        
        # Convert uint8 NV12 to RGB for processing
        x_rgb = nv12_uint8_to_rgb(x_nv12_uint8)  # (3, H, W) float [0, 1]
        
        # Add batch dimension for processing
        x_rgb_batch = x_rgb.unsqueeze(0)  # (1, 3, H, W)
        
        # Original stereo processing
        depth = self.depth_model_wrapper.infer(
            x_rgb_batch, tta=False, low_vram=False, enable_amp=False, edge_dilation=1, depth_aa=False
        )
        depth = self.depth_model_wrapper.minmax_normalize_chw(depth)  # (1, 1, H, W)
        depth = mapper(depth)

        left, right = apply_divergence_nn_LR(
            self.side_model_wrapper,
            x_rgb_batch,
            depth,
            divergence=2.0,
            convergence=0.5,
            steps=None,
            mapper='none',
            synthetic_view='both',
            preserve_screen_border=False,
            enable_amp=False
        )
        
        # Remove batch dimension using indexing
        left = left[0]  # (3, H, W)
        right = right[0]  # (3, H, W)
        
        # Resize left/right to half width for side-by-side
        C, H, W = left.shape
        
        # Use indexing instead of squeeze for interpolation results
        left_resized = F.interpolate(left.unsqueeze(0), size=(H, W // 2), mode="bilinear", align_corners=False)
        left_half = left_resized[0]
        
        right_resized = F.interpolate(right.unsqueeze(0), size=(H, W // 2), mode="bilinear", align_corners=False)
        right_half = right_resized[0]
        
        # Concat along width to create side-by-side
        half_sbs_rgb = torch.cat([left_half, right_half], dim=2)  # (3, H, W)
        
        # Convert RGB output back to uint8 NV12
        half_sbs_nv12_uint8 = rgb_to_nv12_uint8(half_sbs_rgb)  # (H + H//2, W) uint8
        
        # Ensure output is 2D uint8
        assert half_sbs_nv12_uint8.ndim == 2, f"Expected 2D output, got {half_sbs_nv12_uint8.ndim}D: {half_sbs_nv12_uint8.shape}"
        assert half_sbs_nv12_uint8.dtype == torch.uint8, f"Expected uint8 output, got {half_sbs_nv12_uint8.dtype}"
        
        return half_sbs_nv12_uint8

# Build model for single frame processing with uint8 input/output
stereo_module = StereoDepthNV12Uint8Module(depth_model, side_model).eval()

with torch.inference_mode():
    x_nv12_uint8_single = x_nv12_uint8_single.to(depth_model.device)
    logger.debug(f"x_nv12_uint8_single moved to device: {x_nv12_uint8_single.device}, shape: {x_nv12_uint8_single.shape}, dtype: {x_nv12_uint8_single.dtype}")
    half_sbs_nv12_uint8_single = stereo_module(x_nv12_uint8_single)
    logger.debug(f"output half_sbs_nv12_uint8_single shape: {half_sbs_nv12_uint8_single.shape}, dtype: {half_sbs_nv12_uint8_single.dtype}")

    # Export to ONNX with uint8 input/output
    output_path = "stereo_module_nv12.onnx"
    logger.info(f"Exporting ONNX model to {output_path}")
    
    # Print actual tensor shapes for debugging
    print(f"Input tensor shape: {x_nv12_uint8_single.shape}, dtype: {x_nv12_uint8_single.dtype}")
    print(f"Input tensor ndim: {x_nv12_uint8_single.ndim}")
    print(f"Output tensor shape: {half_sbs_nv12_uint8_single.shape}, dtype: {half_sbs_nv12_uint8_single.dtype}")
    print(f"Output tensor ndim: {half_sbs_nv12_uint8_single.ndim}")
    
    # Use a conservative ONNX export with uint8 support
    torch.onnx.export(
        stereo_module,
        (x_nv12_uint8_single,),
        output_path,
        opset_version=17,  # Use older opset for better TensorRT compatibility
        input_names=["input_nv12"],
        output_names=["output_nv12"],
        dynamic_axes={
            "input_nv12": {0: "height_plus_half", 1: "width"},
            "output_nv12": {0: "height_plus_half", 1: "width"},
        },
        do_constant_folding=True,
        keep_initializers_as_inputs=False,
        export_params=True,
        verbose=False,
    )
    logger.info(f"ONNX model saved to {output_path}")

# Save output image for verification
os.makedirs("tmp", exist_ok=True)
# Convert uint8 NV12 back to RGB for saving
rgb_output = nv12_uint8_to_rgb(half_sbs_nv12_uint8_single)
logger.debug(f"Saving half_sbs_nv12_uint8_single.png shape: {rgb_output.shape}")
TF.to_pil_image(rgb_output).save(f"tmp/half_sbs_nv12_uint8_single.png")
print('Single frame uint8 NV12 model export done')

# Verify ONNX inference results match PyTorch output
import onnxruntime as ort
import numpy as np

ort_session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
x_nv12_numpy = x_nv12_uint8_single.cpu().numpy()
onnx_outputs = ort_session.run(None, {"input_nv12": x_nv12_numpy})
onnx_half_sbs_nv12_uint8_single = onnx_outputs[0]
torch_half_sbs_nv12_uint8_single = half_sbs_nv12_uint8_single.cpu().numpy()

def compare_outputs(torch_out, onnx_out, name):
    diff = np.abs(torch_out.astype(np.float32) - onnx_out.astype(np.float32))
    max_diff = diff.max()
    mean_diff = diff.mean()
    print(f"{name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

compare_outputs(torch_half_sbs_nv12_uint8_single, onnx_half_sbs_nv12_uint8_single, "Single Frame uint8 NV12 Side-by-Side")