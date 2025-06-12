# IW3 ONNX 导出指南

本文档介绍如何将IW3的深度估计和立体视觉生成模型导出为ONNX格式，以便在TensorRT中进行算子融合和优化。

## 概述

我们提供了两种导出方案：

1. **基础端到端模型** (`end_to_end_onnx.py`) - 固定参数的单一模型
2. **可参数化模型** (`end_to_end_onnx_parametric.py`) - 运行时可调整参数的灵活模型

## 方案对比

| 特性 | 基础端到端模型 | 可参数化模型 |
|------|---------------|-------------|
| 参数调整 | 编译时固定 | 运行时动态 |
| TensorRT优化 | 更好 | 良好 |
| 灵活性 | 较低 | 很高 |
| 模型大小 | 较小 | 较大 |
| 推理速度 | 最快 | 快 |

## 安装依赖

```bash
pip install torch torchvision onnx
```

## 使用方法

### 1. 基础端到端模型

```bash
# 测试模型兼容性
python -m iw3.end_to_end_onnx --test-only

# 导出ONNX模型 (固定参数: divergence=2.0, convergence=0.5)
python -m iw3.end_to_end_onnx --output ./models/stereo_fixed.onnx

# 指定输入尺寸
python -m iw3.end_to_end_onnx \
    --output ./models/stereo_512.onnx \
    --input-height 512 \
    --input-width 512

# 启用动态batch size
python -m iw3.end_to_end_onnx \
    --output ./models/stereo_dynamic.onnx \
    --dynamic-batch
```

### 2. 可参数化模型 (推荐用于TensorRT)

```bash
# 测试模型兼容性
python -m iw3.end_to_end_onnx_parametric --test-only

# 导出优化版ONNX模型
python -m iw3.end_to_end_onnx_parametric \
    --output ./models/stereo_parametric.onnx \
    --optimized

# 导出标准版
python -m iw3.end_to_end_onnx_parametric \
    --output ./models/stereo_standard.onnx \
    --no-optimized
```

## 模型输入输出

### 基础端到端模型
- **输入**: 
  - `input`: RGB图像 `(batch_size, 3, height, width)`
- **输出**:
  - `left_eye`: 左眼图像 `(batch_size, 3, height, width)`
  - `right_eye`: 右眼图像 `(batch_size, 3, height, width)`
  - `depth`: 深度图 `(batch_size, 1, height, width)`

### 可参数化模型
- **输入**:
  - `input`: RGB图像 `(batch_size, 3, height, width)`
  - `divergence`: 散度参数 `(scalar)` 范围: 0.0-5.0
  - `convergence`: 收敛参数 `(scalar)` 范围: 0.0-1.0
- **输出**:
  - `left_eye`: 左眼图像 `(batch_size, 3, height, width)`
  - `right_eye`: 右眼图像 `(batch_size, 3, height, width)`
  - `depth`: 深度图 `(batch_size, 1, height, width)`

## TensorRT 集成

### 1. 转换为TensorRT引擎

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

def build_tensorrt_engine(onnx_path, engine_path, max_batch_size=1):
    """将ONNX模型转换为TensorRT引擎"""
    
    # 创建builder和network
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # 解析ONNX模型
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # 配置builder
    config = builder.create_builder_config()
    config.max_workspace_size = 2 << 30  # 2GB
    
    # 启用FP16优化 (如果支持)
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    
    # 构建引擎
    engine = builder.build_engine(network, config)
    
    # 保存引擎
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    
    return engine

# 使用示例
engine = build_tensorrt_engine(
    onnx_path="./models/stereo_parametric.onnx",
    engine_path="./models/stereo_parametric.trt",
    max_batch_size=4
)
```

### 2. TensorRT推理

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

class TensorRTStereoInference:
    def __init__(self, engine_path):
        # 加载引擎
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        # 创建执行上下文
        self.context = self.engine.create_execution_context()
        
        # 准备GPU内存
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # 分配GPU内存
            device_mem = cuda.mem_alloc(size * dtype().itemsize)
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({'device': device_mem, 'dtype': dtype, 'size': size})
            else:
                self.outputs.append({'device': device_mem, 'dtype': dtype, 'size': size})
    
    def infer(self, rgb_image, divergence=2.0, convergence=0.5):
        """执行推理"""
        # 准备输入数据
        inputs_data = [
            rgb_image.astype(np.float32),
            np.array([divergence], dtype=np.float32),
            np.array([convergence], dtype=np.float32)
        ]
        
        # 拷贝输入数据到GPU
        for i, input_data in enumerate(inputs_data):
            cuda.memcpy_htod(self.inputs[i]['device'], input_data)
        
        # 执行推理
        self.context.execute_v2(bindings=self.bindings)
        
        # 拷贝输出数据回CPU
        outputs = []
        for output in self.outputs:
            host_output = np.empty(output['size'], dtype=output['dtype'])
            cuda.memcpy_dtoh(host_output, output['device'])
            outputs.append(host_output)
        
        return outputs  # [left_eye, right_eye, depth]

# 使用示例
stereo_engine = TensorRTStereoInference("./models/stereo_parametric.trt")
left_eye, right_eye, depth = stereo_engine.infer(rgb_image, divergence=2.5, convergence=0.6)
```

## 性能优化建议

### 1. ONNX导出优化
- 使用 `--optimized` 标志获得TensorRT优化版本
- 合理设置输入尺寸，推荐512x512或768x768
- 启用动态batch size以获得更好的灵活性

### 2. TensorRT构建优化
- 启用FP16精度以获得更快速度
- 设置适当的workspace大小 (建议2GB+)
- 使用INT8量化进一步提升性能 (需要校准数据集)

### 3. 推理优化
- 批量处理多张图像
- 使用CUDA流进行异步处理
- 预分配GPU内存避免频繁分配

## 常见问题

### Q: 导出失败怎么办？
A: 首先运行 `--test-only` 检查模型兼容性，确保所有依赖都已正确安装。

### Q: ONNX模型太大怎么办？
A: 可以尝试使用更小的深度模型 (如v2_vits而非v2_vitl)，或者使用INT8量化。

### Q: TensorRT转换失败？
A: 检查ONNX opset版本是否与TensorRT版本兼容，可能需要调整opset_version参数。

### Q: 推理结果与PyTorch不一致？
A: 这通常是由于精度问题，可以尝试使用FP32而非FP16，或者检查归一化参数。

## 参数说明

### 散度 (Divergence)
- 范围: 0.0 - 5.0
- 默认: 2.0
- 说明: 控制立体效果强度，值越大立体感越强

### 收敛 (Convergence)  
- 范围: 0.0 - 1.0
- 默认: 0.5
- 说明: 控制图像收敛平面，0.5表示在屏幕平面

## 技术细节

### 模型架构
1. **深度估计**: 使用 Distill Any Depth 模型
2. **立体生成**: 使用 row_flow_v3_sym 模型
3. **图像变形**: 基于网格采样的后向变形

### 优化特性
- 预计算常量减少运行时计算
- 批量处理提高并行度
- 内存布局优化减少数据拷贝
- 算子融合机会最大化

## 示例代码

完整的Python使用示例可以在 `examples/tensorrt_inference.py` 中找到。

## 支持

如有问题，请提交Issue或联系开发团队。 