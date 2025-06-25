# 导出一个 onnx 模型来处理输入格式转换
# 输入是 D3D11VA 的单帧图片
# 输出是 export_depth 计算所需要的格式，因为只有单张图片，所以 batch size总是 1
# 输入的图片尺寸和输出的图片尺寸不一样，因为 D3D11VA 可能会有一些 padding

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import numpy as np
from nunif.logger import logger


class D3D11VAPreprocessModule(nn.Module):
    """
    预处理模块：将 D3D11VA 解码的视频帧转换为深度估计模型的输入格式
    
    输入：NV12 格式的视频帧 (来自D3D11VA解码后传输到系统内存)
    输出：RGB 格式的 BCHW tensor，float32，值范围 0-1
    """
    
    def __init__(self):
        super().__init__()
        
        # YUV to RGB 转换矩阵 (BT.709)
        # RGB = [Y, U-128, V-128] * conversion_matrix^T
        self.register_buffer('yuv_to_rgb_matrix', torch.tensor([
            [1.0000,  0.0000,  1.5748],  # R
            [1.0000, -0.1873, -0.4681],  # G  
            [1.0000,  1.8556,  0.0000]   # B
        ], dtype=torch.float32))
        
    def nv12_to_rgb(self, y_plane, uv_plane):
        """
        将 NV12 格式转换为 RGB
        y_plane: (1, 1, H, W) - Y 分量
        uv_plane: (1, 1, H//2, W) - 交错存储的 UV 分量 (UVUV...)
        """
        # 分离 U 和 V 分量
        u_plane = uv_plane[:, :, :, 0::2]  # 取偶数位置 (U)
        v_plane = uv_plane[:, :, :, 1::2]  # 取奇数位置 (V)
        
        # 上采样 U 和 V 分量到和 Y 相同尺寸
        u_upsampled = F.interpolate(u_plane, size=(y_plane.shape[2], y_plane.shape[3]), 
                                   mode='bilinear', align_corners=False)
        v_upsampled = F.interpolate(v_plane, size=(y_plane.shape[2], y_plane.shape[3]), 
                                   mode='bilinear', align_corners=False)
        
        # 堆叠 YUV 通道: (1, 3, H, W)
        yuv = torch.cat([y_plane, u_upsampled, v_upsampled], dim=1)
        
        # 归一化: uint8 -> float32 (0-1)
        yuv = yuv / 255.0
        
        # U,V 分量中心化 (减去 0.5，相当于原来的 -128)
        yuv[:, 1:, :, :] = yuv[:, 1:, :, :] - 0.5
        
        # YUV to RGB 转换
        B, C, H, W = yuv.shape
        yuv_flat = yuv.permute(0, 2, 3, 1).reshape(-1, 3)  # (B*H*W, 3)
        rgb_flat = torch.matmul(yuv_flat, self.yuv_to_rgb_matrix.T)  # (B*H*W, 3)
        rgb = rgb_flat.reshape(B, H, W, 3).permute(0, 3, 1, 2)  # (B, 3, H, W)
        
        # 钳制到 [0, 1] 范围
        rgb = torch.clamp(rgb, 0.0, 1.0)
        
        return rgb
    
    def forward(self, y_plane, uv_plane, target_height, target_width):
        """
        前向传播
        
        Args:
            y_plane: (1, 1, H, W) - Y 分量，uint8 格式
            uv_plane: (1, 1, H//2, W) - UV 分量，uint8 格式，交错存储
            target_height: 目标输出高度 (tensor scalar)
            target_width: 目标输出宽度 (tensor scalar)
        
        Returns:
            RGB tensor: (1, 3, target_height, target_width), float32, 0-1
        """
        # 转换 NV12 到 RGB
        rgb = self.nv12_to_rgb(y_plane, uv_plane)
            
        # 动态调整到目标尺寸
        # 将 scalar tensor 转换为 int (ONNX 兼容)
        target_h = int(target_height.item()) if isinstance(target_height, torch.Tensor) else int(target_height)
        target_w = int(target_width.item()) if isinstance(target_width, torch.Tensor) else int(target_width)
        
        if rgb.shape[2] != target_h or rgb.shape[3] != target_w:
            rgb = F.interpolate(rgb, size=(target_h, target_w), 
                              mode='bilinear', align_corners=False)
        
        return rgb


def export_nv12_preprocess():
    """导出处理 NV12 格式的 ONNX 模型"""
    
    # 创建模型
    model = D3D11VAPreprocessModule()
    model.eval()
    
    # 创建示例输入 (模拟 NV12 格式)
    # 假设输入分辨率是 3840x1634 (来自你的测试视频)
    input_h, input_w = 1634, 3840
    y_plane = torch.randint(0, 256, (1, 1, input_h, input_w), dtype=torch.float32)
    uv_plane = torch.randint(0, 256, (1, 1, input_h//2, input_w), dtype=torch.float32)  # NV12: UV交错存储
    
    # 动态尺寸参数
    target_height = torch.tensor(392, dtype=torch.int64)
    target_width = torch.tensor(392, dtype=torch.int64)
    
    # 测试前向传播
    with torch.inference_mode():
        output = model(y_plane, uv_plane, target_height, target_width)
        logger.info(f"Output shape: {output.shape}, dtype: {output.dtype}")
        logger.info(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # 导出 ONNX (NV12 版本)
    output_path = "d3d11va_nv12_preprocess.onnx"
    logger.info(f"Exporting NV12 preprocessing model to {output_path}")
    
    torch.onnx.export(
        model,
        (y_plane, uv_plane, target_height, target_width),
        output_path,
        opset_version=18,
        input_names=["y_plane", "uv_plane", "target_height", "target_width"],
        output_names=["rgb_output"],
        dynamic_axes={
            "y_plane": {2: "height", 3: "width"},
            "uv_plane": {2: "height_half", 3: "width"},  # NV12: UV平面宽度和Y相同
            "rgb_output": {0: "batch_size", 2: "target_height", 3: "target_width"}
        }
    )
    
    return output_path


if __name__ == "__main__":
    logger.info("Exporting D3D11VA NV12 preprocessing model...")
    
    # 导出 NV12 预处理模型
    nv12_model_path = export_nv12_preprocess()
    logger.info(f"NV12 preprocessing model saved to: {nv12_model_path}")
    
    # 验证 ONNX 模型
    try:
        import onnxruntime as ort
        
        # 测试 NV12 模型
        session = ort.InferenceSession(nv12_model_path, providers=['CPUExecutionProvider'])
        
        # 模拟 NV12 输入 (3840x1634)
        test_y = np.random.randint(0, 256, (1, 1, 1634, 3840), dtype=np.float32)
        test_uv = np.random.randint(0, 256, (1, 1, 817, 3840), dtype=np.float32)  # height//2, width相同
        target_h = np.array(320, dtype=np.int64) 
        target_w = np.array(240, dtype=np.int64)
        
        outputs = session.run(None, {
            "y_plane": test_y,
            "uv_plane": test_uv, 
            "target_height": target_h, 
            "target_width": target_w
        })
        logger.info(f"ONNX NV12 model test - output shape: {outputs[0].shape}")
        
        logger.info("NV12 model exported and verified successfully!")
        
    except ImportError:
        logger.warning("onnxruntime not available, skipping verification")
    except Exception as e:
        logger.error(f"ONNX model verification failed: {e}")