# 导出一个 onnx 模型来处理输入格式转换
# 输入是 D3D11VA 的单帧图片 (直接从D3D11纹理映射的CUDA内存)
# 输出是 export_depth 计算所需要的格式，因为只有单张图片，所以 batch size总是 1
# 输入的图片尺寸和输出的图片尺寸不一样，因为 D3D11VA 可能会有一些 padding

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import numpy as np
from nunif.logger import logger


class D3D11VADirectPreprocessModule(nn.Module):
    """
    直接预处理模块：将 D3D11VA 映射的 NV12 纹理数据转换为深度估计模型的输入格式
    
    输入：完整的 NV12 纹理数据 (从D3D11直接映射到CUDA，未分离Y/UV) + 目标尺寸模板
    输出：RGB 格式的 BCHW tensor，float32，值范围 0-1，尺寸与目标模板一致
    """
    
    def __init__(self):
        super().__init__()
        
        # YUV to RGB 转换矩阵 (BT.709)
        self.register_buffer('yuv_to_rgb_matrix', torch.tensor([
            [1.0000,  0.0000,  1.5748],  # R
            [1.0000, -0.1873, -0.4681],  # G  
            [1.0000,  1.8556,  0.0000]   # B
        ], dtype=torch.float32))
    
    def extract_nv12_planes(self, nv12_texture):
        """
        从完整的 NV12 纹理中提取 Y 和 UV 平面
        
        Args:
            nv12_texture: (1, 1, H*1.5, W) - 完整的 NV12 纹理数据
                          前 H 行是 Y 数据，后 H/2 行是交错的 UV 数据
        
        Returns:
            y_plane: (1, 1, H, W)
            uv_plane: (1, 1, H//2, W) - 交错的 UV 数据
        """
        _, _, total_h, w = nv12_texture.shape
        
        # 计算实际的图像高度 (total_h = H + H/2 = H*1.5)
        # 使用 tensor 操作避免 TracerWarning
        h = torch.div(total_h * 2, 3, rounding_mode='floor')  # H = total_h / 1.5
        
        # 提取 Y 平面 (前 H 行)
        y_plane = nv12_texture[:, :, :h, :]
        
        # 提取 UV 平面 (后 H/2 行，交错存储)
        uv_plane = nv12_texture[:, :, h:, :]
        
        return y_plane, uv_plane
        
    def nv12_to_rgb(self, y_plane, uv_plane):
        """
        将 NV12 格式转换为 RGB
        y_plane: (1, 1, H, W) - Y 分量
        uv_plane: (1, 1, H//2, W) - 交错存储的 UV 分量 (UVUV...)
        """
        # 分离 U 和 V 分量 (交错存储：UVUV... -> UU.../VV...)
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
    
    def crop_to_target_size(self, rgb, target_template):
        """
        从左上角裁切到目标尺寸（去除 NVIDIA 解码的右/下 padding）
        
        Args:
            rgb: (1, 3, H, W) - 源 RGB tensor（带 padding）
            target_template: (1, 3, target_H, target_W) - 目标尺寸模板
        
        Returns:
            裁切后的 RGB tensor: (1, 3, target_H, target_W)
        """
        _, _, target_h, target_w = target_template.shape
        
        # 直接从左上角裁切到目标尺寸
        return rgb[:, :, :target_h, :target_w]
    
    def forward(self, nv12_texture, target_template):
        """
        前向传播 - 直接处理 D3D11 映射的 NV12 纹理数据
        
        Args:
            nv12_texture: (1, 1, H*1.5, W) - 完整的 NV12 纹理数据，uint8 格式
                          前 H 行是 Y 数据，后 H/2 行是交错的 UV 数据
            target_template: (1, 3, target_H, target_W) - 目标输出尺寸模板
        
        Returns:
            RGB tensor: (1, 3, target_H, target_W), float32, 0-1
        """
        # 从 NV12 纹理中提取 Y 和 UV 平面
        y_plane, uv_plane = self.extract_nv12_planes(nv12_texture)
        
        # 转换 NV12 到 RGB
        rgb = self.nv12_to_rgb(y_plane.float(), uv_plane.float())
        
        # 从左上角裁切到目标尺寸（去除右/下 padding）
        rgb = self.crop_to_target_size(rgb, target_template)
        
        return rgb


def export_d3d11va_direct_preprocess():
    """导出直接处理 D3D11VA NV12 纹理的 ONNX 模型"""
    
    # 创建模型
    model = D3D11VADirectPreprocessModule()
    model.eval()
    
    # 创建示例输入 (模拟完整的 NV12 纹理数据)
    # 假设输入分辨率是 3840x1634 (来自你的测试视频，可能带 padding)
    input_h, input_w = 1634, 3840
    
    # NV12 纹理总高度 = Y平面高度 + UV平面高度 = H + H/2 = H*1.5
    nv12_total_h = int(input_h * 1.5)  # 1634 * 1.5 = 2451
    nv12_texture = torch.randint(0, 256, (1, 1, nv12_total_h, input_w), dtype=torch.uint8)
    
    # 目标输出尺寸模板 (假设实际想要的尺寸是 3840x1608，去掉了 padding)
    target_h, target_w = 1608, 3840  # 根据你的需求调整
    target_template = torch.zeros((1, 3, target_h, target_w), dtype=torch.float32)
    
    # 测试前向传播
    with torch.inference_mode():
        output = model(nv12_texture, target_template)
        logger.info(f"Input NV12 texture shape: {nv12_texture.shape}")
        logger.info(f"Target template shape: {target_template.shape}")
        logger.info(f"Output shape: {output.shape}, dtype: {output.dtype}")
        logger.info(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # 导出 ONNX (直接 D3D11VA 版本)
    output_path = "d3d11va_direct_preprocess.onnx"
    logger.info(f"Exporting D3D11VA direct preprocessing model to {output_path}")
    
    torch.onnx.export(
        model,
        (nv12_texture, target_template),
        output_path,
        opset_version=18,
        input_names=["nv12_texture", "target_template"],
        output_names=["rgb_output"],
        dynamic_axes={
            "nv12_texture": {2: "nv12_height", 3: "input_width"},
            "target_template": {2: "target_height", 3: "target_width"}, 
            "rgb_output": {2: "target_height", 3: "target_width"}
        }
    )
    
    return output_path


if __name__ == "__main__":
    logger.info("Exporting D3D11VA direct preprocessing model...")
    
    # 导出直接 D3D11VA 预处理模型
    direct_model_path = export_d3d11va_direct_preprocess()
    logger.info(f"Direct D3D11VA preprocessing model saved to: {direct_model_path}")
    
    # 验证 ONNX 模型
    try:
        import onnxruntime as ort
        
        # 测试直接 D3D11VA 模型
        session = ort.InferenceSession(direct_model_path, providers=['CPUExecutionProvider'])
        
        # 模拟完整的 NV12 纹理输入 (3840x1634*1.5，包含Y和UV数据) + 目标尺寸模板
        nv12_h = int(1634 * 1.5)  # 2451
        test_nv12 = np.random.randint(0, 256, (1, 1, nv12_h, 3840), dtype=np.uint8)
        test_target = np.zeros((1, 3, 1608, 3840), dtype=np.float32)
        
        outputs = session.run(None, {
            "nv12_texture": test_nv12,
            "target_template": test_target
        })
        logger.info(f"ONNX direct D3D11VA model test - output shape: {outputs[0].shape}")
        
        logger.info("Direct D3D11VA model exported and verified successfully!")
        
    except ImportError:
        logger.warning("onnxruntime not available, skipping verification")
    except Exception as e:
        logger.error(f"ONNX model verification failed: {e}")