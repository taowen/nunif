#!/usr/bin/env python3
"""
导出 Any_V2_S 深度估计模型到 ONNX 格式的脚本

功能包括：
1. 导出 Any_V2_S 模型到 ONNX 格式
2. 验证 PyTorch 和 ONNX 推理的一致性
3. 生成可视化比较图像

使用方法：
    python export_any_v2_s_to_onnx.py

生成文件：
- any_v2_s_depth_model.onnx: 导出的 ONNX 模型
- depth_comparison_pytorch.png: PyTorch 推理结果
- depth_comparison_onnx.png: ONNX 推理结果  
- depth_comparison_diff.png: 差异图
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision.transforms import functional as TF
import cv2

# 添加项目路径到 sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from iw3.depth_anything_model import DepthAnythingModel, batch_preprocess, _forward


def export_model_to_onnx():
    """导出模型到 ONNX 格式"""
    print("=== 导出 Any_V2_S 模型到 ONNX ===")
    
    # 创建模型实例
    model = DepthAnythingModel("Any_V2_S")
    print("正在加载模型...")
    model.load(gpu=-1)  # 使用 CPU
    
    # 设置模型为评估模式
    model.model.eval()
    
    # 准备示例输入 (batch_size=1, channels=3, height=392, width=392)
    # 使用模型的预处理尺寸
    prep_size = model.model.prep_lower_bound
    print(f"模型预处理尺寸: {prep_size}")
    
    dummy_input = torch.randn(1, 3, prep_size, prep_size)
    
    # 应用预处理中的归一化
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    stdv = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    dummy_input = (dummy_input - mean) / stdv
    
    # 导出路径
    onnx_path = "any_v2_s_depth_model.onnx"
    
    print(f"正在导出模型到 {onnx_path}...")
    
    # 导出模型到 ONNX
    torch.onnx.export(
        model.model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,  # 使用 opset 14 支持 scaled_dot_product_attention
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )
    
    print(f"模型已成功导出到: {onnx_path}")
    return onnx_path, model


def load_test_image():
    """加载测试图片"""
    image_path = "waifu2x/docs/images/miku_128.png"
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"测试图片不存在: {image_path}")
    
    # 加载图片
    image = Image.open(image_path).convert("RGB")
    print(f"加载测试图片: {image_path}, 尺寸: {image.size}")
    return image


def pytorch_inference(model, image):
    """使用原始 PyTorch 模型进行推理"""
    print("\n=== PyTorch 推理 ===")
    
    # 转换为张量
    x = TF.to_tensor(image).unsqueeze(0)  # 添加 batch 维度
    
    # 应用预处理
    x_preprocessed = batch_preprocess(x, model.model.prep_lower_bound)
    
    # 推理
    with torch.no_grad():
        output = _forward(model.model, x_preprocessed, enable_amp=False)
    
    print(f"PyTorch 输入形状: {x_preprocessed.shape}")
    print(f"PyTorch 输出形状: {output.shape}")
    print(f"PyTorch 输出范围: [{output.min().item():.6f}, {output.max().item():.6f}]")
    
    return output.squeeze().numpy(), x_preprocessed


def onnx_inference(onnx_path, x_preprocessed):
    """使用 ONNX 模型进行推理"""
    print("\n=== ONNX 推理 ===")
    
    # 创建 ONNX Runtime 会话
    providers = ['CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)
    
    # 获取输入输出信息
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    print(f"ONNX 输入名称: {input_name}")
    print(f"ONNX 输出名称: {output_name}")
    
    # 准备输入数据
    input_data = x_preprocessed.numpy()
    
    # 推理
    output = session.run([output_name], {input_name: input_data})[0]
    
    print(f"ONNX 输入形状: {input_data.shape}")
    print(f"ONNX 输出形状: {output.shape}")
    print(f"ONNX 输出范围: [{output.min():.6f}, {output.max():.6f}]")
    
    return output.squeeze()


def compare_results(pytorch_output, onnx_output):
    """比较 PyTorch 和 ONNX 推理结果"""
    print("\n=== 结果比较 ===")
    
    # 计算像素差异
    diff = np.abs(pytorch_output - onnx_output)
    total_diff = np.sum(diff)
    
    print(f"像素差异总和: {total_diff:.6f}")
    print("✅ PyTorch 和 ONNX 推理完成")
    
    return total_diff


def save_depth_maps(pytorch_output, onnx_output, prefix="depth_comparison"):
    """保存深度图进行可视化比较"""
    print("\n=== 保存深度图 ===")
    
    def normalize_depth(depth):
        """归一化深度图到 0-255 范围"""
        depth_min = depth.min()
        depth_max = depth.max()
        if depth_max > depth_min:
            normalized = ((depth - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(depth, dtype=np.uint8)
        return normalized
    
    # 归一化深度图
    pytorch_norm = normalize_depth(pytorch_output)
    onnx_norm = normalize_depth(onnx_output)
    
    # 保存 PyTorch 结果
    pytorch_path = f"{prefix}_pytorch.png"
    cv2.imwrite(pytorch_path, pytorch_norm)
    print(f"PyTorch 深度图已保存: {pytorch_path}")
    
    # 保存 ONNX 结果
    onnx_path = f"{prefix}_onnx.png"
    cv2.imwrite(onnx_path, onnx_norm)
    print(f"ONNX 深度图已保存: {onnx_path}")
    
    # 保存差异图
    diff = np.abs(pytorch_norm.astype(np.float32) - onnx_norm.astype(np.float32))
    diff_norm = normalize_depth(diff)
    diff_path = f"{prefix}_diff.png"
    cv2.imwrite(diff_path, diff_norm)
    print(f"差异图已保存: {diff_path}")


def main():
    """主函数"""
    try:
        # 检查必要的依赖
        print("检查依赖...")
        import onnx
        print("✅ 所有依赖都已安装")
        
        # 导出模型
        onnx_path, model = export_model_to_onnx()
        
        # 加载测试图片
        test_image = load_test_image()
        test_image_path = "waifu2x/docs/images/miku_128.png"
        
        # PyTorch 推理
        pytorch_output, preprocessed_input = pytorch_inference(model, test_image)
        
        # ONNX 推理
        onnx_output = onnx_inference(onnx_path, preprocessed_input)
        
        # 比较结果
        total_diff = compare_results(pytorch_output, onnx_output)
        
        # 保存深度图
        save_depth_maps(pytorch_output, onnx_output)
        
        # 总结
        print("\n=== 导出和验证完成 ===")
        print(f"ONNX 模型路径: {onnx_path}")
        print(f"像素差异总和: {total_diff:.6f}")
        
        # 生成的文件列表
        print(f"\n生成的文件:")
        print(f"- ONNX 模型: {onnx_path}")
        print(f"- PyTorch 深度图: depth_comparison_pytorch.png")
        print(f"- ONNX 深度图: depth_comparison_onnx.png")
        print(f"- 差异图: depth_comparison_diff.png")
        
        print(f"\n🎉 导出完成!")
        
        return 0
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
