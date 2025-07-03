# 立体图像生成算法模块文档

## 模块概述

立体图像生成算法模块是iw3项目的核心组件，负责将深度图转换为立体视图对（左眼/右眼图像）。该模块实现了多种先进的视图合成算法，包括前向翘曲、后向翘曲和基于神经网络的方法。

## 技术原理

### 立体视觉基本原理

立体3D效果通过向左右眼呈现略有差异的图像来实现深度感知。关键参数：

1. **Divergence（视差）**：控制3D效果的强度
   - 值越大，3D效果越强，但可能产生更多伪影
   - 典型范围：0.5-5.0

2. **Convergence（会聚）**：控制会聚平面的位置
   - 0：无限远处会聚
   - 1：最近处会聚
   - 0.5：中间深度会聚（默认）

3. **视差计算公式**：
   ```
   parallax = depth * divergence * 0.01 * image_width
   shift = parallax * (1 - convergence)
   ```

## 算法实现

### 1. 前向翘曲算法 (Forward Warping)
`forward_warp.py`

#### 核心算法：深度排序双线性前向翘曲

**算法步骤**：
1. 计算每个像素的视差偏移
2. 按深度排序像素（远到近）
3. 将像素投影到新位置
4. 处理遮挡和空洞

**关键函数**：

```python
def depth_order_bilinear_forward_warp(c, depth, divergence, convergence, synthetic_view):
    """
    基于深度排序的双线性前向翘曲
    
    参数:
        c: 输入图像 [B, 3, H, W]
        depth: 深度图 [B, 1, H, W]
        divergence: 视差强度
        convergence: 会聚参数
        synthetic_view: "both"/"left"/"right"
    """
```

**空洞填充策略**：

1. **shift_fill**: 左右交替填充
   ```python
   def shift_fill(x, max_tries=100):
       # 交替从左右邻近像素填充空洞
   ```

2. **fix_layered_holes**: 层次化空洞修复
   ```python
   def fix_layered_holes(side_image, index_image, sign, max_tries=100):
       # 根据深度层次修复遮挡产生的空洞
   ```

3. **blur_blend**: 模糊混合
   ```python
   def blur_blend(x, mask):
       # 对空洞区域进行模糊处理
   ```

**优点**：
- 正确处理遮挡关系
- 适合复杂场景
- 保持物体边缘

**缺点**：
- 计算密集（使用Numba加速）
- 可能产生空洞
- 内存使用较高

### 2. 后向翘曲算法 (Backward Warping)
`backward_warp.py`

#### 核心算法：基于grid_sample的反向映射

**算法原理**：
不是将源像素推送到目标位置，而是从目标位置反向查找源像素。

**关键函数**：

```python
def apply_divergence_grid_sample(c, depth, divergence, convergence, synthetic_view):
    """
    使用grid_sample进行后向翘曲
    
    返回:
        左右眼图像对或单个合成视图
    """
```

```python
def backward_warp(c, grid, delta, delta_scale):
    """
    执行后向翘曲
    
    使用F.grid_sample进行双线性/双三次插值
    """
```

**特点**：
- 使用PyTorch的`grid_sample`函数
- 支持多种插值模式（双线性、双三次）
- 边界处理模式（border、reflection）

**优点**：
- 计算效率高
- 不产生空洞
- GPU友好

**缺点**：
- 难以正确处理遮挡
- 可能产生拉伸伪影

### 3. Row Flow神经网络系列
`models/row_flow_v3.py`

#### 最新一代：Row Flow V3

**架构特点**：
1. **窗口注意力机制**：
   ```python
   WABlock(C, window_size=(4, 4))
   WABlock(C, window_size=(3, 3))
   ```

2. **像素混洗优化**：
   ```python
   x = pixel_unshuffle(x, self.downscaling_factor)  # 下采样
   # ... 处理 ...
   x = pixel_shuffle(x, self.downscaling_factor)    # 上采样
   ```

3. **对称生成模式**：
   ```python
   if self.symmetric:
       left = self._warp(rgb, grid, delta, self.delta_scale)
       right = self._warp(rgb, grid, -delta, self.delta_scale)
   ```

**输入特征**：
- RGB图像
- 深度图
- 视差特征
- 会聚特征
- 网格坐标

**训练细节**：
- 使用合成数据训练
- 训练范围：0 ≤ divergence ≤ 5.0
- 损失函数：L1 + 感知损失

#### 前代模型

**Row Flow V2** (`models/row_flow_v2.py`):
- 增加了重叠/非重叠处理
- 训练范围：0 ≤ divergence ≤ 2.5

**Row Flow V1** (`models/row_flow.py`):
- 基础CNN架构
- 简单的视差预测

### 4. 深度映射函数
`mapper.py`

深度映射函数用于调整深度分布，优化立体效果。

#### 映射函数类型

1. **幂函数映射**：
   ```python
   "pow2": lambda x: x ** 2  # 增强前景深度差异
   ```

2. **Softplus映射**：
   ```python
   def softplus01(x, bias, scale):
       # 平滑的非线性映射
   ```

3. **距离到视差映射**：
   ```python
   def distance_to_disparity(x, c):
       # 将度量深度转换为视差
   ```

4. **组合映射**：
   ```python
   "mul_1": 1.5倍平滑增强
   "mul_2": 2倍平滑增强
   "mul_3": 3倍平滑增强
   ```

#### 自动映射选择

```python
def resolve_mapper_name(mapper, foreground_scale, metric_depth):
    """
    根据前景缩放和深度类型自动选择映射函数
    """
```

### 5. 辅助算法

#### 边缘膨胀处理
`dilation.py`

用于处理Depth Anything模型的边缘伪影：

```python
def apply_edge_dilation(depth, iterations=2):
    """
    膨胀前景物体边缘，减少边缘伪影
    """
```

#### 立体图像生成工具
`training/sbs/stereoimage_generation.py`

来自stable-diffusion-webui-depthmap-script的算法：

1. **apply_stereo_divergence_naive**: 简单像素移位
2. **apply_stereo_divergence_polylines**: 基于多段线的高质量插值

## 使用示例

### 基本立体生成

```python
from iw3.forward_warp import apply_divergence_forward_warp
from iw3.backward_warp import apply_divergence_grid_sample

# 前向翘曲
left, right = apply_divergence_forward_warp(
    image, depth, 
    divergence=2.0, 
    convergence=0.5,
    synthetic_view="both"
)

# 后向翘曲
left, right = apply_divergence_grid_sample(
    image, depth,
    divergence=2.0,
    convergence=0.5,
    synthetic_view="both"
)
```

### 使用Row Flow模型

```python
from iw3.models.row_flow_v3 import RowFlowV3
from iw3.backward_warp import make_input_tensor

# 加载模型
model = RowFlowV3()
model.load_state_dict(torch.load("path/to/checkpoint.pth"))
model.eval()

# 准备输入
x = make_input_tensor(
    image, depth, 
    divergence=2.0, 
    convergence=0.5,
    image_width=1920,
    mapper="pow2"
)

# 生成立体图像
with torch.no_grad():
    output = model(x)
    if model.symmetric:
        left, right = output[:, :3], output[:, 3:6]
    else:
        stereo = output
```

### 深度映射调整

```python
from iw3.mapper import get_mapper

# 获取映射函数
mapper = get_mapper("mul_2")  # 2倍平滑增强

# 应用映射
adjusted_depth = mapper(depth)

# 组合映射
complex_mapper = get_mapper("pow2:mul_1")  # 先平方再1.5倍增强
```

## 性能优化

### 1. 算法选择指南

| 场景 | 推荐算法 | 原因 |
|------|---------|------|
| 实时处理 | 后向翘曲 | 计算效率高 |
| 高质量离线 | 前向翘曲 | 正确的遮挡处理 |
| 视频处理 | Row Flow V3 | 时序一致性好 |
| 简单场景 | 后向翘曲 | 快速且效果可接受 |

### 2. 内存优化

- 使用`--low-vram`选项减少内存使用
- 批处理时注意批大小设置
- 前向翘曲可能需要更多内存

### 3. 速度优化

- 启用模型编译：`model.compile()`
- 使用合适的图像分辨率
- 多GPU并行处理

## 常见问题

**Q: 前向翘曲vs后向翘曲如何选择？**
A: 
- 质量优先：前向翘曲
- 速度优先：后向翘曲
- 平衡选择：Row Flow模型

**Q: 如何处理边缘伪影？**
A:
- 使用边缘膨胀（--edge-dilation）
- 调整映射函数
- 尝试不同的深度模型

**Q: 视差参数如何调整？**
A:
- 开始时使用默认值（divergence=2.0）
- VR观看通常需要较小值（1.0-2.0）
- 大屏幕可以使用较大值（3.0-5.0）

## 扩展指南

### 添加新的翘曲算法

1. 创建新的算法文件
2. 实现核心翘曲函数
3. 在主处理流程中注册

示例：
```python
def my_custom_warp(c, depth, divergence, convergence, synthetic_view):
    # 实现自定义翘曲算法
    pass
```

### 自定义映射函数

在`mapper.py`中添加新的映射函数：

```python
elif name == "my_mapper":
    return lambda x: my_custom_function(x)
```

### 训练新的Row Flow模型

参考`training/sbs/trainer.py`中的训练代码，准备数据集并训练新模型。

## 算法比较

| 特性 | 前向翘曲 | 后向翘曲 | Row Flow V3 |
|------|---------|---------|-------------|
| 遮挡处理 | 优秀 | 差 | 良好 |
| 计算速度 | 慢 | 快 | 中等 |
| 内存使用 | 高 | 低 | 中等 |
| 空洞问题 | 有 | 无 | 无 |
| GPU友好度 | 中 | 高 | 高 |
| 时序一致性 | 无 | 无 | 有 |

## 相关文档

- [深度估计模型模块](depth_estimation_models.md)
- [视频处理管线](video_processing_pipeline.md)
- [参数系统与配置](parameter_system.md)
- [训练模块](training_modules.md)