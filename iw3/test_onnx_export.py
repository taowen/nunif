#!/usr/bin/env python3
"""
测试ONNX导出功能的简单脚本
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import argparse

def test_basic_model():
    """测试基础端到端模型"""
    print("=== 测试基础端到端模型 ===")
    
    try:
        from . import end_to_end_onnx
        
        print("1. 测试模型兼容性...")
        if not end_to_end_onnx.test_model_compatibility():
            print("❌ 基础模型兼容性测试失败")
            return False
        
        print("✅ 基础模型兼容性测试通过")
        
        print("2. 导出小尺寸ONNX模型...")
        end_to_end_onnx.export_end_to_end_model(
            output_path="./test_basic_256.onnx",
            input_size=(3, 256, 256),
            dynamic_batch=False
        )
        
        if os.path.exists("./test_basic_256.onnx"):
            file_size = os.path.getsize("./test_basic_256.onnx") / (1024 * 1024)  # MB
            print(f"✅ 基础模型导出成功，文件大小: {file_size:.1f} MB")
            return True
        else:
            print("❌ 基础模型导出失败")
            return False
            
    except Exception as e:
        print(f"❌ 基础模型测试失败: {e}")
        return False


def test_parametric_model():
    """测试可参数化模型"""
    print("\n=== 测试可参数化模型 ===")
    
    try:
        from . import end_to_end_onnx_parametric
        
        print("1. 测试模型兼容性...")
        if not end_to_end_onnx_parametric.test_parametric_model():
            print("❌ 可参数化模型兼容性测试失败")
            return False
        
        print("✅ 可参数化模型兼容性测试通过")
        
        print("2. 导出优化版ONNX模型...")
        end_to_end_onnx_parametric.export_parametric_model(
            output_path="./test_parametric_256.onnx",
            input_size=(3, 256, 256),
            dynamic_batch=False,
            optimized=True
        )
        
        if os.path.exists("./test_parametric_256.onnx"):
            file_size = os.path.getsize("./test_parametric_256.onnx") / (1024 * 1024)  # MB
            print(f"✅ 可参数化模型导出成功，文件大小: {file_size:.1f} MB")
            return True
        else:
            print("❌ 可参数化模型导出失败")
            return False
            
    except Exception as e:
        print(f"❌ 可参数化模型测试失败: {e}")
        return False


def test_onnx_inference():
    """测试ONNX模型推理"""
    print("\n=== 测试ONNX模型推理 ===")
    
    try:
        import onnxruntime as ort
        
        # 测试基础模型
        if os.path.exists("./test_basic_256.onnx"):
            print("1. 测试基础模型推理...")
            
            session = ort.InferenceSession("./test_basic_256.onnx")
            
            # 创建测试输入
            test_input = np.random.randn(1, 3, 256, 256).astype(np.float32)
            
            # 运行推理
            outputs = session.run(None, {"input": test_input})
            
            print(f"   输入形状: {test_input.shape}")
            print(f"   左眼输出形状: {outputs[0].shape}")
            print(f"   右眼输出形状: {outputs[1].shape}")
            print(f"   深度输出形状: {outputs[2].shape}")
            print("✅ 基础模型推理测试通过")
        
        # 测试可参数化模型
        if os.path.exists("./test_parametric_256.onnx"):
            print("2. 测试可参数化模型推理...")
            
            session = ort.InferenceSession("./test_parametric_256.onnx")
            
            # 创建测试输入
            test_input = np.random.randn(1, 3, 256, 256).astype(np.float32)
            test_divergence = np.array([2.5], dtype=np.float32)
            test_convergence = np.array([0.6], dtype=np.float32)
            
            # 运行推理
            outputs = session.run(None, {
                "input": test_input,
                "divergence": test_divergence,
                "convergence": test_convergence
            })
            
            print(f"   输入形状: {test_input.shape}")
            print(f"   散度参数: {test_divergence[0]}")
            print(f"   收敛参数: {test_convergence[0]}")
            print(f"   左眼输出形状: {outputs[0].shape}")
            print(f"   右眼输出形状: {outputs[1].shape}")
            print(f"   深度输出形状: {outputs[2].shape}")
            print("✅ 可参数化模型推理测试通过")
        
        return True
        
    except ImportError:
        print("⚠️  ONNX Runtime未安装，跳过推理测试")
        print("   安装命令: pip install onnxruntime")
        return True
    except Exception as e:
        print(f"❌ ONNX推理测试失败: {e}")
        return False


def test_model_validation():
    """验证导出的ONNX模型"""
    print("\n=== 验证ONNX模型 ===")
    
    try:
        import onnx
        
        models_to_check = [
            ("基础模型", "./test_basic_256.onnx"),
            ("可参数化模型", "./test_parametric_256.onnx")
        ]
        
        for name, model_path in models_to_check:
            if os.path.exists(model_path):
                print(f"验证 {name}...")
                
                # 加载和验证模型
                model = onnx.load(model_path)
                onnx.checker.check_model(model)
                
                # 显示模型信息
                print(f"   输入: {[inp.name for inp in model.graph.input]}")
                print(f"   输出: {[out.name for out in model.graph.output]}")
                print(f"   节点数: {len(model.graph.node)}")
                
                print(f"✅ {name} 验证通过")
        
        return True
        
    except ImportError:
        print("⚠️  ONNX包未安装，跳过模型验证")
        print("   安装命令: pip install onnx")
        return True
    except Exception as e:
        print(f"❌ 模型验证失败: {e}")
        return False


def cleanup_test_files():
    """清理测试文件"""
    test_files = [
        "./test_basic_256.onnx",
        "./test_parametric_256.onnx"
    ]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"已删除测试文件: {file_path}")


def main():
    parser = argparse.ArgumentParser(description="测试IW3 ONNX导出功能")
    parser.add_argument("--cleanup", action="store_true", help="清理测试文件后退出")
    parser.add_argument("--skip-basic", action="store_true", help="跳过基础模型测试")
    parser.add_argument("--skip-parametric", action="store_true", help="跳过可参数化模型测试")
    parser.add_argument("--keep-files", action="store_true", help="保留测试文件")
    
    args = parser.parse_args()
    
    if args.cleanup:
        cleanup_test_files()
        return
    
    print("IW3 ONNX导出功能测试")
    print("=" * 50)
    
    success_count = 0
    total_tests = 0
    
    # 基础模型测试
    if not args.skip_basic:
        total_tests += 1
        if test_basic_model():
            success_count += 1
    
    # 可参数化模型测试  
    if not args.skip_parametric:
        total_tests += 1
        if test_parametric_model():
            success_count += 1
    
    # ONNX推理测试
    total_tests += 1
    if test_onnx_inference():
        success_count += 1
    
    # 模型验证测试
    total_tests += 1
    if test_model_validation():
        success_count += 1
    
    # 总结
    print(f"\n=== 测试总结 ===")
    print(f"通过: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("🎉 所有测试通过！ONNX导出功能正常工作")
    else:
        print("⚠️  部分测试失败，请检查错误信息")
    
    # 清理文件
    if not args.keep_files:
        print("\n清理测试文件...")
        cleanup_test_files()
    else:
        print(f"\n测试文件已保留:")
        for file_path in ["./test_basic_256.onnx", "./test_parametric_256.onnx"]:
            if os.path.exists(file_path):
                print(f"  {file_path}")


if __name__ == "__main__":
    main() 