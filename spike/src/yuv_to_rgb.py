# a python module to convert nv12 format to rgb format
# then export it to onnx

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import numpy as np
import os
from nunif.logger import logger

class NV12ToRGBModule(nn.Module):
    """Convert NV12 format to RGB format using PyTorch operations"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, nv12_data):
        """
        Convert NV12 to RGB
        Args:
            nv12_data: (B, H*3//2, W) tensor containing NV12 data
                      First H rows are Y plane (luminance)
                      Next H//2 rows are UV plane (chrominance, interleaved)
        Returns:
            rgb: (B, 3, H, W) tensor in RGB format, range [0, 1]
        """
        B, total_H, W = nv12_data.shape
        H = total_H * 2 // 3  # Original height
        
        # Extract Y plane (luminance)
        y_plane = nv12_data[:, :H, :]  # (B, H, W)
        
        # Extract UV plane (chrominance, interleaved)
        uv_plane = nv12_data[:, H:, :]  # (B, H//2, W)
        
        # Separate U and V components
        u_plane = uv_plane[:, :, ::2]   # (B, H//2, W//2) - U components
        v_plane = uv_plane[:, :, 1::2]  # (B, H//2, W//2) - V components
        
        # Upsample U and V to full resolution
        u_full = F.interpolate(u_plane.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False).squeeze(1)
        v_full = F.interpolate(v_plane.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False).squeeze(1)
        
        # Convert from YUV to RGB using standard conversion matrix
        # RGB = [Y + 1.402 * (V - 128)]
        #       [Y - 0.344 * (U - 128) - 0.714 * (V - 128)]
        #       [Y + 1.772 * (U - 128)]
        
        # Normalize Y, U, V from [0, 255] to [0, 1] if needed
        # Assuming input is already in [0, 255] range
        y_norm = y_plane / 255.0
        u_norm = u_full / 255.0 - 0.5  # Center around 0
        v_norm = v_full / 255.0 - 0.5  # Center around 0
        
        # YUV to RGB conversion
        r = y_norm + 1.402 * v_norm
        g = y_norm - 0.344136 * u_norm - 0.714136 * v_norm
        b = y_norm + 1.772 * u_norm
        
        # Clamp to [0, 1] range
        r = torch.clamp(r, 0, 1)
        g = torch.clamp(g, 0, 1)
        b = torch.clamp(b, 0, 1)
        
        # Stack to create RGB tensor (B, 3, H, W)
        rgb = torch.stack([r, g, b], dim=1)
        
        return rgb

def create_test_nv12_data(batch_size=2, height=256, width=256):
    """Create test NV12 data for testing"""
    # Create Y plane (luminance) - full resolution
    y_plane = torch.randint(0, 256, (batch_size, height, width), dtype=torch.float32)
    
    # Create UV plane (chrominance) - half resolution, interleaved
    u_plane = torch.randint(0, 256, (batch_size, height//2, width//2), dtype=torch.float32)
    v_plane = torch.randint(0, 256, (batch_size, height//2, width//2), dtype=torch.float32)
    
    # Interleave U and V
    uv_plane = torch.zeros(batch_size, height//2, width, dtype=torch.float32)
    uv_plane[:, :, ::2] = u_plane   # U components at even positions
    uv_plane[:, :, 1::2] = v_plane  # V components at odd positions
    
    # Concatenate Y and UV planes
    nv12_data = torch.cat([y_plane, uv_plane], dim=1)  # (B, H*3//2, W)
    
    return nv12_data

def main():
    logger.info("Creating NV12 to RGB conversion module")
    
    # Create the conversion module
    nv12_to_rgb = NV12ToRGBModule().eval()
    
    # Create test data
    test_nv12 = create_test_nv12_data(batch_size=2, height=256, width=256)
    logger.info(f"Test NV12 data shape: {test_nv12.shape}")
    
    # Test the conversion
    with torch.inference_mode():
        rgb_output = nv12_to_rgb(test_nv12)
        logger.info(f"RGB output shape: {rgb_output.shape}")
        logger.info(f"RGB output range: [{rgb_output.min():.4f}, {rgb_output.max():.4f}]")
    
    # Export to ONNX
    output_path = "nv12_to_rgb.onnx"
    logger.info(f"Exporting ONNX model to {output_path}")
    
    torch.onnx.export(
        nv12_to_rgb,
        (test_nv12,),
        output_path,
        opset_version=18,
        input_names=["nv12_data"],
        output_names=["rgb_output"],
        dynamic_axes={
            "nv12_data": {0: "batch_size", 1: "total_height", 2: "width"},
            "rgb_output": {0: "batch_size", 2: "height", 3: "width"},
        },
    )
    logger.info(f"ONNX model saved to {output_path}")
    
    # Verify ONNX inference matches PyTorch output
    try:
        import onnxruntime as ort
        
        ort_session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
        nv12_numpy = test_nv12.numpy()
        onnx_outputs = ort_session.run(None, {"nv12_data": nv12_numpy})
        onnx_rgb = onnx_outputs[0]
        torch_rgb = rgb_output.numpy()
        
        # Compare outputs
        diff = np.abs(torch_rgb - onnx_rgb)
        max_diff = diff.max()
        mean_diff = diff.mean()
        logger.info(f"ONNX vs PyTorch: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
        
        if max_diff < 1e-5:
            logger.info("✓ ONNX model verification passed")
        else:
            logger.warning(f"⚠ ONNX model verification failed with max difference: {max_diff}")
            
    except ImportError:
        logger.warning("onnxruntime not available, skipping ONNX verification")
    
    logger.info("NV12 to RGB conversion module export completed")

if __name__ == "__main__":
    main()