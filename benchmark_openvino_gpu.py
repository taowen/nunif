#!/usr/bin/env python3
"""
使用 OpenVINO 原生 API 进行 Intel GPU 推理并估算 FPS

这个脚本直接使用 OpenVINO Python API，绕过 ONNX Runtime 的问题，
直接在 Intel GPU 上运行推理并测量性能。

使用方法：
    python benchmark_openvino_gpu.py

要求：
- any_v2_s_depth_model.onnx 文件存在
- 安装了 openvino 包
"""

import os
import sys
import time
import numpy as np
import openvino as ov
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
import cv2

# 添加项目路径到 sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_openvino_devices():
    """检查 OpenVINO 可用设备"""
    print("=== OpenVINO 设备检查 ===")
    
    core = ov.Core()
    devices = core.available_devices
    
    print(f"OpenVINO 版本: {ov.__version__}")
    print(f"可用设备: {devices}")
    
    # 获取设备详细信息
    for device in devices:
        try:
            device_name = core.get_property(device, "FULL_DEVICE_NAME")
            print(f"  {device}: {device_name}")
        except:
            print(f"  {device}: (无法获取设备名称)")
    
    # 选择 GPU 设备
    if 'GPU' in devices:
        print("✅ 找到 Intel GPU，将使用 GPU 进行推理")
        return 'GPU'
    elif 'NPU' in devices:
        print("✅ 找到 Intel NPU，将使用 NPU 进行推理")
        return 'NPU'
    else:
        print("⚠️ 未找到 GPU/NPU，将使用 CPU")
        return 'CPU'


def load_and_compile_model(onnx_path, device):
    """加载并编译 ONNX 模型到指定设备"""
    print(f"\n=== 加载模型到 {device} ===")
    print(f"模型路径: {onnx_path}")
    
    # 创建 OpenVINO 核心
    core = ov.Core()
    
    # 读取 ONNX 模型
    print("正在读取 ONNX 模型...")
    model = core.read_model(onnx_path)
    
    # 获取模型信息
    input_layer = model.input(0)
    output_layer = model.output(0)
    
    print(f"输入形状: {input_layer.partial_shape}")
    print(f"输入类型: {input_layer.element_type}")
    print(f"输出形状: {output_layer.partial_shape}")
    print(f"输出类型: {output_layer.element_type}")
    
    # 如果是动态形状，设置固定形状
    if input_layer.partial_shape.is_dynamic:
        print("检测到动态形状，设置为固定形状 [1,3,392,392]")
        model.reshape([1, 3, 392, 392])
    
    # 编译模型到目标设备
    print(f"正在编译模型到 {device}...")
    
    # 设置设备特定配置
    config = {}
    if device == 'GPU':
        # GPU 特定配置
        config = {
            "INFERENCE_PRECISION_HINT": "f16",  # 使用 FP16 精度
            "GPU_ENABLE_LOOP_UNROLLING": "YES",
            "CACHE_DIR": "./openvino_cache"
        }
    elif device == 'NPU':
        # NPU 特定配置
        config = {
            "NPU_USE_NPUW": "YES",
            "CACHE_DIR": "./openvino_cache"
        }
    
    compiled_model = core.compile_model(model, device, config)
    
    print(f"✅ 模型已成功编译到 {device}")
    
    return compiled_model, input_layer, output_layer


def create_test_data(batch_size=1, height=392, width=392):
    """创建标准化的测试数据"""
    # 创建随机数据
    data = torch.randn(batch_size, 3, height, width)
    
    # ImageNet 标准化
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    stdv = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    normalized = (data - mean) / stdv
    
    return normalized.numpy().astype(np.float32)


def warm_up(compiled_model, test_data, num_runs=5):
    """预热模型"""
    print(f"\n=== 预热 ({num_runs} 次) ===")
    
    infer_request = compiled_model.create_infer_request()
    
    for i in range(num_runs):
        infer_request.infer(test_data)
    
    print("✅ 预热完成")


def benchmark_sync(compiled_model, test_data, num_runs=50):
    """同步推理基准测试"""
    print(f"\n=== 同步推理基准测试 ({num_runs} 次) ===")
    
    batch_size = test_data.shape[0]
    infer_request = compiled_model.create_infer_request()
    
    # 记录时间
    times = []
    
    print("开始测试...")
    for i in range(num_runs):
        start = time.time()
        result = infer_request.infer(test_data)
        end = time.time()
        
        times.append(end - start)
        
        if (i + 1) % 10 == 0:
            print(f"进度: {i + 1}/{num_runs}")
    
    # 计算统计
    times = np.array(times)
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    # 计算 FPS
    batch_fps = 1.0 / mean_time
    image_fps = batch_fps * batch_size
    
    print(f"\n=== 同步推理性能结果 ===")
    print(f"批次大小: {batch_size}")
    print(f"平均推理时间: {mean_time * 1000:.2f} ± {std_time * 1000:.2f} ms")
    print(f"最快时间: {min_time * 1000:.2f} ms")
    print(f"最慢时间: {max_time * 1000:.2f} ms")
    print(f"批次 FPS: {batch_fps:.2f}")
    print(f"图像 FPS: {image_fps:.2f}")
    
    return {
        'type': 'sync',
        'batch_size': batch_size,
        'mean_time': mean_time,
        'std_time': std_time,
        'min_time': min_time,
        'max_time': max_time,
        'batch_fps': batch_fps,
        'image_fps': image_fps
    }


def benchmark_async(compiled_model, test_data, num_parallel=4, num_runs=100):
    """异步推理基准测试"""
    print(f"\n=== 异步推理基准测试 ({num_parallel} 并行, {num_runs} 次) ===")
    
    batch_size = test_data.shape[0]
    
    # 创建多个推理请求
    infer_requests = [compiled_model.create_infer_request() for _ in range(num_parallel)]
    
    # 异步推理
    start_time = time.time()
    
    completed = 0
    active_requests = []
    
    print("开始异步测试...")
    
    # 启动初始请求
    for i in range(min(num_parallel, num_runs)):
        req = infer_requests[i]
        req.start_async(test_data)
        active_requests.append((req, time.time()))
    
    times = []
    
    while completed < num_runs:
        for i, (req, start_t) in enumerate(active_requests):
            if req.wait_for(0):  # 非阻塞检查
                end_t = time.time()
                times.append(end_t - start_t)
                completed += 1
                
                # 启动新请求
                if completed + len(active_requests) - i - 1 < num_runs:
                    req.start_async(test_data)
                    active_requests[i] = (req, time.time())
                else:
                    active_requests[i] = None
        
        # 清理完成的请求
        active_requests = [item for item in active_requests if item is not None]
        
        if completed % 20 == 0 and completed > 0:
            print(f"进度: {completed}/{num_runs}")
        
        time.sleep(0.001)  # 避免过度占用 CPU
    
    total_time = time.time() - start_time
    
    # 计算统计
    times = np.array(times)
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    # 计算吞吐量
    throughput = completed / total_time
    image_throughput = throughput * batch_size
    
    print(f"\n=== 异步推理性能结果 ===")
    print(f"批次大小: {batch_size}")
    print(f"并行度: {num_parallel}")
    print(f"总时间: {total_time:.2f} 秒")
    print(f"平均推理时间: {mean_time * 1000:.2f} ± {std_time * 1000:.2f} ms")
    print(f"最快时间: {min_time * 1000:.2f} ms")
    print(f"最慢时间: {max_time * 1000:.2f} ms")
    print(f"批次吞吐量: {throughput:.2f} 批次/秒")
    print(f"图像吞吐量: {image_throughput:.2f} 图像/秒")
    
    return {
        'type': 'async',
        'batch_size': batch_size,
        'parallel': num_parallel,
        'total_time': total_time,
        'mean_time': mean_time,
        'std_time': std_time,
        'min_time': min_time,
        'max_time': max_time,
        'throughput': throughput,
        'image_throughput': image_throughput
    }


def test_different_batch_sizes(compiled_model, batch_sizes=[1, 2, 4]):
    """测试不同批次大小"""
    print(f"\n=== 测试不同批次大小 ===")
    
    results = []
    
    for batch_size in batch_sizes:
        print(f"\n--- 批次大小: {batch_size} ---")
        
        try:
            # 创建测试数据
            test_data = create_test_data(batch_size)
            
            # 预热
            warm_up(compiled_model, test_data, 3)
            
            # 同步基准测试
            sync_result = benchmark_sync(compiled_model, test_data, 30)
            results.append(sync_result)
            
            # 异步基准测试（仅批次大小为1时）
            if batch_size == 1:
                async_result = benchmark_async(compiled_model, test_data, 4, 100)
                results.append(async_result)
            
        except Exception as e:
            print(f"⚠️ 批次大小 {batch_size} 测试失败: {e}")
            continue
    
    return results


def test_real_image(compiled_model):
    """测试真实图像推理"""
    print(f"\n=== 真实图像测试 ===")
    
    # 寻找测试图像
    test_paths = ["test_input.png", "test_input.jpg", "waifu2x/docs/images/miku_128.png"]
    
    image = None
    image_path = None
    
    for path in test_paths:
        if os.path.exists(path):
            try:
                image = Image.open(path).convert("RGB")
                image_path = path
                break
            except:
                continue
    
    if image is None:
        print("⚠️ 未找到测试图像")
        return
    
    print(f"使用图像: {image_path}, 尺寸: {image.size}")
    
    # 预处理
    x = TF.to_tensor(image).unsqueeze(0)
    x = F.interpolate(x, size=(392, 392), mode='bilinear', align_corners=False)
    
    # 标准化
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    stdv = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    x = (x - mean) / stdv
    
    input_data = x.numpy().astype(np.float32)
    
    # 推理
    infer_request = compiled_model.create_infer_request()
    
    start = time.time()
    result = infer_request.infer(input_data)
    end = time.time()
    
    inference_time = end - start
    fps = 1.0 / inference_time
    
    print(f"推理时间: {inference_time * 1000:.2f} ms")
    print(f"FPS: {fps:.2f}")
    
    # 获取输出
    output = list(result.values())[0]
    
    # 保存结果
    depth = output.squeeze()
    depth_norm = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
    
    output_path = "openvino_gpu_result.png"
    cv2.imwrite(output_path, depth_norm)
    print(f"结果已保存: {output_path}")


def print_summary(results):
    """打印性能总结"""
    if not results:
        print("⚠️ 没有成功的测试结果")
        return
    
    print(f"\n=== 性能总结 ===")
    print("类型   | 批次大小 | 并行度 | 平均时间(ms) | 标准差(ms) | 吞吐量(FPS)")
    print("-" * 75)
    
    for result in results:
        if result['type'] == 'sync':
            print(f"同步   | {result['batch_size']:8d} | {'N/A':6s} | "
                  f"{result['mean_time']*1000:11.2f} | "
                  f"{result['std_time']*1000:10.2f} | "
                  f"{result['image_fps']:10.2f}")
        else:  # async
            print(f"异步   | {result['batch_size']:8d} | {result['parallel']:6d} | "
                  f"{result['mean_time']*1000:11.2f} | "
                  f"{result['std_time']*1000:10.2f} | "
                  f"{result['image_throughput']:10.2f}")
    
    # 最佳性能
    sync_results = [r for r in results if r['type'] == 'sync']
    async_results = [r for r in results if r['type'] == 'async']
    
    if sync_results:
        best_sync = max(sync_results, key=lambda x: x['image_fps'])
        print(f"\n🏆 最佳同步性能: 批次大小 {best_sync['batch_size']} - {best_sync['image_fps']:.2f} FPS")
    
    if async_results:
        best_async = max(async_results, key=lambda x: x['image_throughput'])
        print(f"🏆 最佳异步性能: 并行度 {best_async['parallel']} - {best_async['image_throughput']:.2f} FPS")


def main():
    """主函数"""
    try:
        # 检查 ONNX 模型
        onnx_path = "any_v2_s_depth_model.onnx"
        if not os.path.exists(onnx_path):
            print(f"❌ ONNX 模型不存在: {onnx_path}")
            print("请先运行 export_any_v2_s_to_onnx.py")
            return 1
        
        # 检查设备
        device = check_openvino_devices()
        
        # 加载并编译模型
        compiled_model, input_layer, output_layer = load_and_compile_model(onnx_path, device)
        
        # 真实图像测试
        test_real_image(compiled_model)
        
        # 不同批次大小测试
        results = test_different_batch_sizes(compiled_model, [1, 2, 4])
        
        # 打印总结
        print_summary(results)
        
        print(f"\n🎉 OpenVINO {device} 测试完成!")
        return 0
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
