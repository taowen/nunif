# input a rgb image from disk: waifu2x/docs/images/miku_128.png
# calc depth map using distll any depth small
# use row_flow_v3 with delta_output=True
# generate left right eye using the delta
# output left right eye nv12 image to disk

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

# NV12 conversion functions
def rgb_to_nv12(rgb):
    """Convert RGB tensor to NV12 format
    Args:
        rgb: (B, 3, H, W) float tensor in range [0, 1]
    Returns:
        nv12: (B, H + H//2, W) float tensor
    """
    B, C, H, W = rgb.shape
    
    # RGB to YUV conversion matrix
    # Y =  0.299*R + 0.587*G + 0.114*B
    # U = -0.169*R - 0.331*G + 0.500*B + 0.5
    # V =  0.500*R - 0.419*G - 0.081*B + 0.5
    rgb_to_yuv = torch.tensor([
        [0.299, 0.587, 0.114],
        [-0.169, -0.331, 0.500],
        [0.500, -0.419, -0.081]
    ], device=rgb.device, dtype=rgb.dtype)
    
    # Reshape for matrix multiplication
    rgb_flat = rgb.view(B, 3, -1)  # (B, 3, H*W)
    yuv_flat = torch.matmul(rgb_to_yuv, rgb_flat)  # (B, 3, H*W)
    yuv = yuv_flat.view(B, 3, H, W)
    
    # Extract Y, U, V
    y = yuv[:, 0, :, :]  # (B, H, W)
    u = yuv[:, 1, :, :] + 0.5  # (B, H, W)
    v = yuv[:, 2, :, :] + 0.5  # (B, H, W)
    
    # Downsample UV to 4:2:0
    u_downsampled = F.avg_pool2d(u.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)  # (B, H//2, W//2)
    v_downsampled = F.avg_pool2d(v.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)  # (B, H//2, W//2)
    
    # Interleave U and V to create UV plane
    uv_plane = torch.stack([u_downsampled, v_downsampled], dim=-1)  # (B, H//2, W//2, 2)
    uv_plane = uv_plane.view(B, H//2, W)  # (B, H//2, W)
    
    # Concatenate Y and UV planes
    nv12 = torch.cat([y, uv_plane], dim=1)  # (B, H + H//2, W)
    
    return nv12

def nv12_to_rgb(nv12):
    """Convert NV12 format to RGB tensor
    Args:
        nv12: (B, H + H//2, W) float tensor
    Returns:
        rgb: (B, 3, H, W) float tensor in range [0, 1]
    """
    B, total_H, W = nv12.shape
    H = total_H * 2 // 3  # Original height
    
    # Split Y and UV planes
    y_plane = nv12[:, :H, :]  # (B, H, W)
    uv_plane = nv12[:, H:, :]  # (B, H//2, W)
    
    # Separate U and V from UV plane
    uv_reshaped = uv_plane.view(B, H//2, W//2, 2)  # (B, H//2, W//2, 2)
    u_downsampled = uv_reshaped[:, :, :, 0] - 0.5  # (B, H//2, W//2)
    v_downsampled = uv_reshaped[:, :, :, 1] - 0.5  # (B, H//2, W//2)
    
    # Upsample U and V to original resolution
    u = F.interpolate(u_downsampled.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False).squeeze(1)
    v = F.interpolate(v_downsampled.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False).squeeze(1)
    
    # YUV to RGB conversion matrix
    yuv_to_rgb = torch.tensor([
        [1.0, 0.0, 1.402],
        [1.0, -0.344136, -0.714136],
        [1.0, 1.772, 0.0]
    ], device=nv12.device, dtype=nv12.dtype)
    
    # Stack YUV channels
    yuv = torch.stack([y_plane, u, v], dim=1)  # (B, 3, H, W)
    
    # Reshape for matrix multiplication
    yuv_flat = yuv.view(B, 3, -1)  # (B, 3, H*W)
    rgb_flat = torch.matmul(yuv_to_rgb, yuv_flat)  # (B, 3, H*W)
    rgb = rgb_flat.view(B, 3, H, W)
    
    # Clamp to valid range
    rgb = torch.clamp(rgb, 0.0, 1.0)
    
    return rgb

# Load test images and convert to NV12
img_path1 = "iw3/figure/convergence.png"
img_path2 = "iw3/figure/divergence.png"
logger.debug(f"load two images")
img1 = Image.open(img_path1).convert("RGB")
img2 = Image.open(img_path2).convert("RGB")
x1 = TF.to_tensor(img1)
x2 = TF.to_tensor(img2)
logger.debug(f"x1 shape: {x1.shape}, x2 shape: {x2.shape}")
x_rgb = torch.stack([x1, x2], dim=0)  # BCHW, float32, 0-1
logger.debug(f"stacked x_rgb shape: {x_rgb.shape}")

# Convert RGB to NV12
x_nv12 = rgb_to_nv12(x_rgb)
logger.debug(f"converted x_nv12 shape: {x_nv12.shape}")

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

class StereoDepthNV12Module(nn.Module):
    def __init__(self, depth_model, side_model):
        super().__init__()
        self.depth_model_wrapper = depth_model
        self.depth_model = depth_model.model
        self.side_model_wrapper = side_model
        self.side_model = side_model.model if hasattr(side_model, "model") else side_model

    def forward(self, x_nv12):
        # x_nv12: (B, H + H//2, W) float32, NV12 format
        
        # Convert NV12 to RGB for processing
        x_rgb = nv12_to_rgb(x_nv12)
        
        # Original stereo processing
        depth = self.depth_model_wrapper.infer(
            x_rgb, tta=False, low_vram=False, enable_amp=True, edge_dilation=1, depth_aa=False
        )
        depth = self.depth_model_wrapper.minmax_normalize_chw(depth)  # BCHW
        depth = mapper(depth)

        left, right = apply_divergence_nn_LR(
            self.side_model_wrapper,
            x_rgb,
            depth,
            divergence=2.0,
            convergence=0.5,
            steps=None,
            mapper='none',
            synthetic_view='both',
            preserve_screen_border=False,
            enable_amp=True
        )
        
        # Resize left/right to half width for side-by-side
        B, C, H, W = left.shape
        left_half = F.interpolate(left, size=(H, W // 2), mode="bilinear", align_corners=False)
        right_half = F.interpolate(right, size=(H, W // 2), mode="bilinear", align_corners=False)
        
        # Concat along width to create side-by-side
        half_sbs_rgb = torch.cat([left_half, right_half], dim=3)  # (B, C, H, W)
        
        # Convert RGB output back to NV12
        half_sbs_nv12 = rgb_to_nv12(half_sbs_rgb)
        
        return half_sbs_nv12

# Build model
stereo_module = StereoDepthNV12Module(depth_model, side_model).eval()

with torch.inference_mode():
    x_nv12 = x_nv12.to(depth_model.device)
    logger.debug(f"x_nv12 moved to device: {x_nv12.device}, shape: {x_nv12.shape}")
    half_sbs_nv12 = stereo_module(x_nv12)
    logger.debug(f"output half_sbs_nv12 shape: {half_sbs_nv12.shape}")

    # Export to ONNX
    output_path = "stereo_module_nv12.onnx"
    logger.info(f"Exporting ONNX model to {output_path}")
    torch.onnx.export(
        stereo_module,
        (x_nv12,),
        output_path,
        opset_version=18,
        input_names=["input_nv12"],
        output_names=["output_nv12"],
        dynamic_axes={
            "input_nv12": {0: "batch_size", 1: "height_plus_half", 2: "width"},
            "output_nv12": {0: "batch_size", 1: "height_plus_half", 2: "width"},
        },
    )
    logger.info(f"ONNX model saved to {output_path}")

# Save output images for verification
os.makedirs("tmp", exist_ok=True)
for idx in range(half_sbs_nv12.shape[0]):
    # Convert NV12 back to RGB for saving
    nv12_single = half_sbs_nv12[idx:idx+1]  # Keep batch dimension
    rgb_single = nv12_to_rgb(nv12_single)[0]  # Remove batch dimension
    logger.debug(f"Saving half_sbs_nv12_{idx}.png shape: {rgb_single.shape}")
    TF.to_pil_image(rgb_single).save(f"tmp/half_sbs_nv12_{idx}.png")
print('NV12 model export done')

# Verify ONNX inference results match PyTorch output
import onnxruntime as ort
import numpy as np

ort_session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
x_nv12_numpy = x_nv12.cpu().numpy()
onnx_outputs = ort_session.run(None, {"input_nv12": x_nv12_numpy})
onnx_half_sbs_nv12 = onnx_outputs[0]
torch_half_sbs_nv12 = half_sbs_nv12.cpu().numpy()

def compare_outputs(torch_out, onnx_out, name):
    diff = np.abs(torch_out - onnx_out)
    max_diff = diff.max()
    mean_diff = diff.mean()
    print(f"{name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

compare_outputs(torch_half_sbs_nv12, onnx_half_sbs_nv12, "NV12 Half Side-by-Side")