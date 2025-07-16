#!/usr/bin/env python3
"""
使用 OpenVINO 原生 API 进行 Intel NPU 推理并估算 FPS

这个脚本专门针对 Intel NPU 优化，使用 OpenVINO Python API 在 NPU 上运行推理并测量性能。
NPU (Neural Processing Unit) 是专门为 AI 推理设计的硬件加速器。

使用方法：
    python benchmark_openvino_npu.py

要求：
- any_v2_s_depth_model.onnx 文件存在
- 安装了 openvino 包
- 支持 Intel NPU 的硬件（如 Intel Arc GPU 或 Intel Core Ultra 处理器）
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
    """检查 OpenVINO 可用设备，优先选择 NPU"""
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
    
    # 优先选择 NPU 设备
    if 'NPU' in devices:
        print("✅ 找到 Intel NPU，将使用 NPU 进行推理")
        return 'NPU'
    else:
        print("❌ 未找到 Intel NPU")
        print("   请确保您的硬件支持 NPU 并且已安装正确的驱动程序")
        print("   支持的硬件：Intel Arc GPU 或 Intel Core Ultra 处理器")
        return None


def load_and_compile_model(onnx_path, device='NPU'):
    """加载并编译 ONNX 模型到 NPU"""
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
    
    # NPU 需要固定形状，但我们可以在运行时处理动态输入
    if input_layer.partial_shape.is_dynamic:
        print("检测到动态形状，为 NPU 设置固定形状 [1,3,392,392]")
        # NPU 只支持批次大小为 1 的固定形状
        model.reshape([1, 3, 392, 392])
    
    # 编译模型到 NPU
    print(f"正在编译模型到 {device}...")
    
    # NPU 最简化配置 - 只使用确实支持的选项
    config = {
        "CACHE_DIR": "./openvino_cache",  # 缓存目录
        "INFERENCE_PRECISION_HINT": "f16",  # 使用 FP16 精度
    }
    
    compiled_model = core.compile_model(model, device, config)
    
    print(f"✅ 模型已成功编译到 {device}")
    print("注意：NPU 首次编译可能需要较长时间，后续会使用缓存")
    
    return compiled_model, input_layer, output_layer


def create_test_data(batch_size=1, height=392, width=392):
    """创建标准化的测试数据"""
    # NPU 通常只支持批次大小为 1
    if batch_size > 1:
        print(f"⚠️ NPU 通常只支持批次大小为 1，将使用批次大小 1 而不是 {batch_size}")
        batch_size = 1
    
    # 创建随机数据
    data = torch.randn(batch_size, 3, height, width)
    
    # ImageNet 标准化
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    stdv = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    normalized = (data - mean) / stdv
    
    return normalized.numpy().astype(np.float32)


def warm_up(compiled_model, test_data, num_runs=10):
    """预热模型 - NPU 需要更多预热次数"""
    print(f"\n=== 预热 ({num_runs} 次) ===")
    print("NPU 预热时间可能较长，请耐心等待...")
    
    infer_request = compiled_model.create_infer_request()
    
    for i in range(num_runs):
        start = time.time()
        infer_request.infer(test_data)
        end = time.time()
        
        if i == 0:
            print(f"首次推理时间: {(end - start) * 1000:.2f} ms")
        
        if (i + 1) % 5 == 0:
            print(f"预热进度: {i + 1}/{num_runs}")
    
    print("✅ 预热完成")


def benchmark_sync(compiled_model, test_data, num_runs=100):
    """同步推理基准测试 - 针对 NPU 优化"""
    print(f"\n=== NPU 同步推理基准测试 ({num_runs} 次) ===")
    
    batch_size = test_data.shape[0]
    infer_request = compiled_model.create_infer_request()
    
    # 记录时间
    times = []
    
    print("开始测试...")
    start_total = time.time()
    
    for i in range(num_runs):
        start = time.time()
        result = infer_request.infer(test_data)
        end = time.time()
        
        times.append(end - start)
        
        if (i + 1) % 20 == 0:
            print(f"进度: {i + 1}/{num_runs}")
    
    end_total = time.time()
    total_time = end_total - start_total
    
    # 计算统计
    times = np.array(times)
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    # 计算 FPS
    batch_fps = 1.0 / mean_time
    image_fps = batch_fps * batch_size
    total_throughput = num_runs / total_time
    
    print(f"\n=== NPU 同步推理性能结果 ===")
    print(f"批次大小: {batch_size}")
    print(f"总时间: {total_time:.2f} 秒")
    print(f"平均推理时间: {mean_time * 1000:.2f} ± {std_time * 1000:.2f} ms")
    print(f"最快时间: {min_time * 1000:.2f} ms")
    print(f"最慢时间: {max_time * 1000:.2f} ms")
    print(f"理论 FPS: {batch_fps:.2f}")
    print(f"实际总吞吐量: {total_throughput:.2f} 次/秒")
    print(f"图像 FPS: {image_fps:.2f}")
    
    return {
        'type': 'sync',
        'batch_size': batch_size,
        'total_time': total_time,
        'mean_time': mean_time,
        'std_time': std_time,
        'min_time': min_time,
        'max_time': max_time,
        'batch_fps': batch_fps,
        'image_fps': image_fps,
        'total_throughput': total_throughput
    }


def benchmark_async(compiled_model, test_data, num_parallel=2, num_runs=200):
    """异步推理基准测试 - NPU 优化版本"""
    print(f"\n=== NPU 异步推理基准测试 ({num_parallel} 并行, {num_runs} 次) ===")
    print("注意：NPU 的并行能力可能有限")
    
    batch_size = test_data.shape[0]
    
    # NPU 通常不支持高并行度，限制并行数
    if num_parallel > 4:
        num_parallel = 4
        print(f"⚠️ 限制 NPU 并行度为 {num_parallel}")
    
    # 创建多个推理请求
    infer_requests = [compiled_model.create_infer_request() for _ in range(num_parallel)]
    
    # 异步推理
    start_time = time.time()
    
    completed = 0
    active_requests = []
    times = []
    
    print("开始异步测试...")
    
    # 启动初始请求
    for i in range(min(num_parallel, num_runs)):
        req = infer_requests[i]
        req.start_async(test_data)
        active_requests.append((req, time.time()))
    
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
        
        if completed % 40 == 0 and completed > 0:
            print(f"进度: {completed}/{num_runs}")
        
        time.sleep(0.002)  # NPU 可能需要稍长的等待时间
    
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
    
    print(f"\n=== NPU 异步推理性能结果 ===")
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


def test_power_efficiency(compiled_model, test_data, duration=30):
    """测试 NPU 功耗效率 - 持续推理一段时间"""
    print(f"\n=== NPU 功耗效率测试 ({duration} 秒) ===")
    print("NPU 的主要优势是低功耗，此测试评估持续推理性能")
    
    infer_request = compiled_model.create_infer_request()
    
    start_time = time.time()
    inference_count = 0
    times = []
    
    print("开始持续推理...")
    
    while time.time() - start_time < duration:
        start = time.time()
        result = infer_request.infer(test_data)
        end = time.time()
        
        times.append(end - start)
        inference_count += 1
        
        if inference_count % 50 == 0:
            elapsed = time.time() - start_time
            print(f"已运行: {elapsed:.1f}/{duration} 秒, 完成: {inference_count} 次")
    
    total_time = time.time() - start_time
    
    # 计算统计
    times = np.array(times)
    mean_time = np.mean(times)
    std_time = np.std(times)
    sustained_fps = inference_count / total_time
    
    print(f"\n=== NPU 功耗效率结果 ===")
    print(f"持续运行时间: {total_time:.2f} 秒")
    print(f"总推理次数: {inference_count}")
    print(f"平均推理时间: {mean_time * 1000:.2f} ± {std_time * 1000:.2f} ms")
    print(f"持续 FPS: {sustained_fps:.2f}")
    print(f"时间稳定性: {(std_time/mean_time)*100:.2f}% 变异系数")
    
    return {
        'type': 'power_efficiency',
        'duration': total_time,
        'inference_count': inference_count,
        'mean_time': mean_time,
        'std_time': std_time,
        'sustained_fps': sustained_fps,
        'stability': (std_time/mean_time)*100
    }


def test_real_image(compiled_model):
    """测试真实图像推理"""
    print(f"\n=== NPU 真实图像测试 ===")
    
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
    
    # 推理多次以获得稳定结果
    infer_request = compiled_model.create_infer_request()
    
    times = []
    for i in range(10):
        start = time.time()
        result = infer_request.infer(input_data)
        end = time.time()
        times.append(end - start)
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    fps = 1.0 / mean_time
    
    print(f"平均推理时间: {mean_time * 1000:.2f} ± {std_time * 1000:.2f} ms")
    print(f"FPS: {fps:.2f}")
    
    # 获取输出
    output = list(result.values())[0]
    
    # 保存结果
    depth = output.squeeze()
    depth_norm = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
    
    output_path = "npu_stereo_output.png"
    cv2.imwrite(output_path, depth_norm)
    print(f"结果已保存: {output_path}")


def print_summary(results):
    """打印性能总结"""
    if not results:
        print("⚠️ 没有成功的测试结果")
        return
    
    print(f"\n=== NPU 性能总结 ===")
    print("类型          | 批次大小 | 并行度 | 平均时间(ms) | 标准差(ms) | 吞吐量(FPS) | 备注")
    print("-" * 90)
    
    for result in results:
        if result['type'] == 'sync':
            print(f"同步推理      | {result['batch_size']:8d} | {'N/A':6s} | "
                  f"{result['mean_time']*1000:11.2f} | "
                  f"{result['std_time']*1000:10.2f} | "
                  f"{result['image_fps']:10.2f} | 标准测试")
        elif result['type'] == 'async':
            print(f"异步推理      | {result['batch_size']:8d} | {result['parallel']:6d} | "
                  f"{result['mean_time']*1000:11.2f} | "
                  f"{result['std_time']*1000:10.2f} | "
                  f"{result['image_throughput']:10.2f} | 并行优化")
        elif result['type'] == 'power_efficiency':
            print(f"功耗效率测试  | {'1':8s} | {'N/A':6s} | "
                  f"{result['mean_time']*1000:11.2f} | "
                  f"{result['std_time']*1000:10.2f} | "
                  f"{result['sustained_fps']:10.2f} | 持续{result['duration']:.0f}秒")
    
    # NPU 特性分析
    sync_results = [r for r in results if r['type'] == 'sync']
    power_results = [r for r in results if r['type'] == 'power_efficiency']
    
    if sync_results:
        best_sync = max(sync_results, key=lambda x: x['image_fps'])
        print(f"\n🏆 最佳推理性能: {best_sync['image_fps']:.2f} FPS")
    
    if power_results:
        power_result = power_results[0]
        print(f"🔋 功耗效率特性: 持续 {power_result['sustained_fps']:.2f} FPS")
        print(f"⚡ 性能稳定性: {power_result['stability']:.2f}% 变异系数")
        
        if power_result['stability'] < 10:
            print("✅ NPU 性能稳定，适合长时间运行")
        else:
            print("⚠️ NPU 性能波动较大，可能受到热管理影响")


def main():
    """主函数"""
    try:
        # 检查 ONNX 模型
        onnx_path = "any_v2_s_depth_model.onnx"
        if not os.path.exists(onnx_path):
            print(f"❌ ONNX 模型不存在: {onnx_path}")
            print("请先运行 export_any_v2_s_to_onnx.py")
            return 1
        
        # 检查 NPU 设备
        device = check_openvino_devices()
        if device is None:
            return 1
        
        # 加载并编译模型
        compiled_model, input_layer, output_layer = load_and_compile_model(onnx_path, device)
        
        # 创建测试数据（NPU 只支持批次大小 1）
        test_data = create_test_data(1)
        
        # 预热 - NPU 需要更多预热
        warm_up(compiled_model, test_data, 10)
        
        results = []
        
        # 真实图像测试
        test_real_image(compiled_model)
        
        # 同步基准测试
        sync_result = benchmark_sync(compiled_model, test_data, 100)
        results.append(sync_result)
        
        # 异步基准测试（低并行度）
        async_result = benchmark_async(compiled_model, test_data, 2, 200)
        results.append(async_result)
        
        # 功耗效率测试
        power_result = test_power_efficiency(compiled_model, test_data, 30)
        results.append(power_result)
        
        # 打印总结
        print_summary(results)
        
        print(f"\n🎉 Intel NPU 测试完成!")
        print("\nNPU 特点总结:")
        print("✅ 低功耗：专为移动和边缘设备设计")
        print("✅ 持续性能：适合长时间运行的应用")
        print("⚠️ 批次限制：通常只支持批次大小为 1")
        print("⚠️ 并行限制：并行能力相对有限")
        
        return 0
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
