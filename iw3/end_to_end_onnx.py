import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
from torchvision.transforms import functional as TF
from PIL import Image
import argparse
import numpy as np
from .models.row_flow_v3 import RowFlowV3
from nunif.utils.ui import TorchHubDir
from nunif.models import load_model

HUB_MODEL_DIR = os.path.join(os.path.dirname(__file__), "pretrained_models", "hub")
ROW_FLOW_V3_SYM_URL = "https://github.com/nagadomi/nunif/releases/download/0.0.0/iw3_row_flow_v3_sym_20240424.pth"


def batch_preprocess_depth(x, lower_bound=392, max_aspect_ratio=4):
    """预处理图像用于深度估计 - 适配ONNX导出"""
    B, C, H, W = x.shape
    
    # resize
    ensure_multiple_of = 14
    if W < H:
        scale_factor = lower_bound / W
    else:
        scale_factor = lower_bound / H
    new_h = int(H * scale_factor)
    new_w = int(W * scale_factor)
    
    # Limit aspect ratio to avoid OOM
    if new_h < new_w:
        new_w = min(new_w, int(max_aspect_ratio * new_h))
    else:
        new_h = min(new_h, int(max_aspect_ratio * new_w))
    
    new_h -= new_h % ensure_multiple_of
    new_w -= new_w % ensure_multiple_of
    if new_h < lower_bound:
        new_h = lower_bound
    if new_w < lower_bound:
        new_w = lower_bound
    
    x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=True)
    x = torch.clamp(x, 0, 1)
    
    # normalize
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    stdv = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    x = (x - mean) / stdv
    return x


def make_grid(batch, width, height):
    """创建网格坐标 - ONNX兼容版本"""
    # 创建y和x坐标
    y_coords = torch.linspace(-1, 1, height).unsqueeze(1).expand(height, width)
    x_coords = torch.linspace(-1, 1, width).unsqueeze(0).expand(height, width)
    
    # 组合成网格并添加batch维度
    mesh_x = x_coords.unsqueeze(0).unsqueeze(0).expand(batch, 1, height, width)
    mesh_y = y_coords.unsqueeze(0).unsqueeze(0).expand(batch, 1, height, width)
    grid = torch.cat((mesh_x, mesh_y), dim=1)
    return grid


def backward_warp(c, grid, delta, delta_scale):
    """后向变形 - ONNX兼容版本"""
    grid = grid + delta * delta_scale
    if c.shape[2] != grid.shape[2] or c.shape[3] != grid.shape[3]:
        grid = F.interpolate(grid, size=c.shape[-2:],
                             mode="bilinear", align_corners=True, antialias=False)
    grid = grid.permute(0, 2, 3, 1)
    
    z = F.grid_sample(c, grid, mode="bicubic", padding_mode="border", align_corners=True)
    z = torch.clamp(z, 0, 1)
    return z


def make_input_tensor_for_symmetric(depth, divergence, convergence, image_width):
    """为对称模型创建输入张量 - ONNX兼容版本"""
    H, W = depth.shape[-2:]
    
    # 创建散度和收敛特征
    divergence_value = divergence * 0.01
    convergence_value = convergence
    
    divergence_feat = torch.full((H, W), divergence_value, dtype=depth.dtype)
    convergence_feat = torch.full((H, W), convergence_value, dtype=depth.dtype)
    
    # 创建网格
    y_coords = torch.linspace(-1, 1, H).unsqueeze(1).expand(H, W)
    x_coords = torch.linspace(-1, 1, W).unsqueeze(0).expand(H, W)
    grid = torch.stack((x_coords, y_coords), 0)  # CHW
    
    # 组合输入
    input_tensor = torch.cat([
        depth,
        divergence_feat.unsqueeze(0),
        convergence_feat.unsqueeze(0),
        grid,
    ], dim=0)
    
    return input_tensor


class EndToEndStereoModel(nn.Module):
    """端到端立体视觉生成模型，包含深度估计和立体视觉生成"""
    
    def __init__(self, depth_model, stereo_model, depth_encoder="v2_vits"):
        super().__init__()
        self.depth_model = depth_model
        self.stereo_model = stereo_model
        self.depth_encoder = depth_encoder
        
        # 固定参数
        self.register_buffer("divergence", torch.tensor(2.0))
        self.register_buffer("convergence", torch.tensor(0.5))
        
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入RGB图像 (B, 3, H, W)
        Returns:
            left_eye: 左眼图像 (B, 3, H, W)
            right_eye: 右眼图像 (B, 3, H, W)
            depth: 深度图 (B, 1, H, W)
        """
        B, C, H, W = x.shape
        original_rgb = x.clone()
        
        # 1. 深度估计
        # 预处理用于深度估计
        x_depth = batch_preprocess_depth(x, lower_bound=392)
        
        # 深度推理
        depth = self.depth_model(x_depth).unsqueeze(dim=1)
        depth = -depth  # 反转用于兼容
        depth = depth.squeeze(0)  # 移除不必要的维度
        
        # 归一化深度到0-1
        depth_min = torch.min(depth)
        depth_max = torch.max(depth)
        depth = (depth - depth_min) / (depth_max - depth_min + 1e-8)
        
        # 调整深度图尺寸匹配原始RGB
        if depth.shape[1:] != original_rgb.shape[1:]:
            depth = F.interpolate(
                depth.unsqueeze(0), 
                size=original_rgb.shape[1:], 
                mode="bilinear", 
                align_corners=True, 
                antialias=True
            ).squeeze(0)
        
        # 2. 立体视觉生成
        # 准备输入
        depth_batch = depth.unsqueeze(0)  # 添加batch维度
        x_stereo = torch.stack([make_input_tensor_for_symmetric(
            depth_batch[i], self.divergence, self.convergence, W)
                     for i in range(depth_batch.shape[0])])
        
        # 立体推理
        delta = self.stereo_model(x_stereo)
        
        # 创建网格和变形
        grid = make_grid(B, W, H)
        delta_scale = 1.0 / (W // 2 - 1)
        
        left_eye = backward_warp(original_rgb, grid, delta, delta_scale)
        right_eye = backward_warp(original_rgb, grid, -delta, delta_scale)
        
        return left_eye, right_eye, depth.unsqueeze(0)


def load_depth_model():
    """加载深度估计模型"""
    encoder = "v2_vits"  # 使用小模型以减少ONNX文件大小
    
    if not os.getenv("IW3_DEBUG"):
        model = torch.hub.load("nagadomi/Depth-Anything_iw3:main",
                               "DistillAnyDepth", encoder=encoder,
                               verbose=False, trust_repo=True)
    else:
        model = torch.hub.load("../Depth-Anything_iw3",
                               "DistillAnyDepth", encoder=encoder, source="local",
                               verbose=False, trust_repo=True)
    
    return model.eval()


def load_stereo_model():
    """加载立体视觉模型"""
    with TorchHubDir(HUB_MODEL_DIR):
        side_model = load_model(ROW_FLOW_V3_SYM_URL, weights_only=True, device_ids=[0])[0].eval()
        side_model.symmetric = True
        side_model.delta_output = True
    return side_model


def export_end_to_end_model(output_path="./end_to_end_stereo.onnx", 
                           input_size=(3, 512, 512),
                           dynamic_batch=True):
    """
    导出端到端立体视觉ONNX模型
    
    Args:
        output_path: 输出ONNX文件路径
        input_size: 输入图像尺寸 (C, H, W)
        dynamic_batch: 是否支持动态batch size
    """
    print("Loading models...")
    
    # 加载模型
    depth_model = load_depth_model()
    stereo_model = load_stereo_model()
    
    # 创建端到端模型
    model = EndToEndStereoModel(depth_model, stereo_model)
    model.eval()
    
    print(f"Creating dummy input with size: {input_size}")
    
    # 创建虚拟输入
    C, H, W = input_size
    dummy_input = torch.randn(1, C, H, W)
    
    # 动态轴配置
    if dynamic_batch:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'left_eye': {0: 'batch_size'},
            'right_eye': {0: 'batch_size'},
            'depth': {0: 'batch_size'}
        }
    else:
        dynamic_axes = {}
    
    print("Exporting to ONNX...")
    
    # 导出ONNX
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['left_eye', 'right_eye', 'depth'],
            dynamic_axes=dynamic_axes,
            verbose=False
        )
    
    print(f"ONNX model exported to: {output_path}")
    
    # 验证导出的模型
    print("Validating exported model...")
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model validation passed!")
        
        # 显示模型信息
        print(f"Model input: {onnx_model.graph.input[0].name}")
        print(f"Model outputs: {[output.name for output in onnx_model.graph.output]}")
        
    except ImportError:
        print("ONNX package not found, skipping validation")
    except Exception as e:
        print(f"ONNX validation failed: {e}")


def test_model_compatibility():
    """测试模型兼容性"""
    print("Testing model compatibility...")
    
    try:
        # 加载和测试
        depth_model = load_depth_model()
        stereo_model = load_stereo_model()
        
        model = EndToEndStereoModel(depth_model, stereo_model)
        model.eval()
        
        # 测试前向传播
        test_input = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            left_eye, right_eye, depth = model(test_input)
            
        print(f"Input shape: {test_input.shape}")
        print(f"Left eye shape: {left_eye.shape}")
        print(f"Right eye shape: {right_eye.shape}")
        print(f"Depth shape: {depth.shape}")
        
        print("Model compatibility test passed!")
        return True
        
    except Exception as e:
        print(f"Model compatibility test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Export end-to-end stereo model to ONNX")
    parser.add_argument("--output", type=str, default="./end_to_end_stereo.onnx", 
                       help="Output ONNX file path")
    parser.add_argument("--input-height", type=int, default=512, 
                       help="Input image height")
    parser.add_argument("--input-width", type=int, default=512, 
                       help="Input image width")
    parser.add_argument("--dynamic-batch", action="store_true", 
                       help="Enable dynamic batch size")
    parser.add_argument("--test-only", action="store_true", 
                       help="Only test model compatibility")
    
    args = parser.parse_args()
    
    if args.test_only:
        test_model_compatibility()
    else:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        
        # 先测试兼容性
        if test_model_compatibility():
            export_end_to_end_model(
                output_path=args.output,
                input_size=(3, args.input_height, args.input_width),
                dynamic_batch=args.dynamic_batch
            )
        else:
            print("Model compatibility test failed, aborting export")


if __name__ == "__main__":
    with TorchHubDir(HUB_MODEL_DIR):
        main() 