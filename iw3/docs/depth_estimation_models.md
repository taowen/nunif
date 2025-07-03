# 深度估计模型模块文档

## 模块概述

深度估计模型模块是iw3项目的核心组件之一，负责从2D图像或视频中估计深度信息。该模块采用了面向对象的设计，提供了统一的接口来支持多种最先进的深度估计模型。

## 技术架构

### 基础架构设计

#### BaseDepthModel 抽象基类
`base_depth_model.py` 定义了所有深度估计模型的抽象基类，提供了统一的接口和通用功能。

**核心设计特点：**
1. **抽象方法模式**：使用Python的ABCMeta确保子类实现必要的方法
2. **设备管理**：自动处理GPU/CPU设备切换和多GPU支持
3. **模型编译优化**：支持PyTorch模型编译以提升性能
4. **深度归一化**：内置EMA（指数移动平均）归一化器

**关键属性：**
```python
- device: 计算设备（CPU/GPU）
- model: 深度估计模型实例
- model_backup: 编译前的模型备份
- model_type: 模型类型标识符
- scaler: EMA深度归一化器
```

#### 工厂模式实现
`depth_model_factory.py` 实现了工厂模式，根据模型类型创建相应的深度模型实例。

```python
def create_depth_model(model_type):
    if ZoeDepthModel.supported(model_type):
        return ZoeDepthModel(model_type)
    elif DepthAnythingModel.supported(model_type):
        return DepthAnythingModel(model_type)
    # ... 其他模型
```

### 深度归一化系统

#### EMAMinMaxScaler
`depth_scaler.py` 实现了基于指数移动平均的深度归一化器，用于处理视频序列中的深度值归一化。

**三种模式：**
1. **SimpleMinMaxScaler** (decay=0, buffer_size=1)
   - 简单的最小-最大归一化
   - 适用于单张图像
   
2. **IncrementalEMAScaler** (decay=0.75, buffer_size=1)
   - 增量式EMA归一化
   - 适用于实时处理
   
3. **WindowEMAScaler** (decay=0.9, buffer_size=30)
   - 窗口式EMA归一化
   - 适用于视频处理，提供时序平滑

**归一化公式：**
```python
normalized = 1.0 - ((depth - min_value) / (max_value - min_value))
```

注意：深度值被反转（1.0 - value），使得近处物体值更大。

## 具体模型实现

### 1. ZoeDepth模型系列
`zoedepth_model.py`

**支持的模型：**
- **ZoeD_N**: NYUv2数据集训练，适合室内场景
- **ZoeD_K**: KITTI数据集训练，适合室外场景（车载视角）
- **ZoeD_NK**: 混合训练，通用模型
- **ZoeD_Any_N**: 结合Depth Anything的NYUv2模型
- **ZoeD_Any_K**: 结合Depth Anything的KITTI模型

**特点：**
- 度量深度估计（可输出真实距离）
- 自适应深度分箱技术
- 室内外场景自动分类

### 2. Depth Anything模型系列
`depth_anything_model.py`

**支持的模型：**
- **Any_S/B/L**: 相对深度估计，S(小)/B(基础)/L(大)三种规模
- **Any_V2_S/B/L**: 第二代模型，性能更优
- **Any_V2_N_S/B/L**: Hypersim数据集训练的度量深度模型（室内）
- **Any_V2_K_S/B/L**: VKITTI数据集训练的度量深度模型（室外）

**特点：**
- 基于大规模无标注数据训练
- 极强的泛化能力
- 边缘准确但可能需要膨胀处理

### 3. Depth Pro模型
`depth_pro_model.py`

**支持的模型：**
- **DepthPro**: 原始1536x1536分辨率
- **DepthPro_S**: 修改的1024x1024分辨率

**特点：**
- Apple开发的高质量度量深度模型
- 极其锐利的边缘
- 高分辨率输出
- 仅支持图像，不支持视频

### 4. Video Depth Anything模型
`video_depth_anything_model.py`

**支持的模型：**
- **VDA_S**: 小型模型
- **VDA_L**: 大型模型（需要许可）
- **VDA_Metric**: 度量深度模型（需要许可）

**特点：**
- 专门为视频设计
- 保持时序一致性
- 使用全局最小/最大值归一化
- 批量处理优化（批大小32）

### 5. Null深度模型
`null_depth_model.py`

用于测试和调试的空模型，直接使用输入的深度图而不进行估计。

## API参考

### BaseDepthModel类

#### 抽象方法（子类必须实现）

```python
@abstractmethod
def get_name() -> str:
    """返回模型名称"""

@abstractmethod
def supported(cls, model_type: str) -> bool:
    """检查是否支持指定的模型类型"""

@abstractmethod
def has_checkpoint_file(cls, model_type: str) -> bool:
    """检查模型文件是否存在"""

@abstractmethod
def get_model_path(cls, model_type: str) -> str:
    """获取模型文件路径"""

@abstractmethod
def is_metric(self) -> bool:
    """返回是否为度量深度模型"""

@abstractmethod
def load_model(self, model_type: str, resolution: int, device: torch.device):
    """加载模型实现"""

@abstractmethod
def infer(self, x: torch.Tensor, *kwargs) -> torch.Tensor:
    """执行深度推理"""
```

#### 公共方法

```python
def load(self, gpu: Union[int, List[int]], resolution: Optional[int] = None):
    """
    加载模型到指定设备
    
    参数:
        gpu: GPU设备ID或ID列表（多GPU）
        resolution: 深度图分辨率（某些模型支持）
    """

def compile(self):
    """编译模型以提升性能（PyTorch 2.0+）"""

def enable_ema(self, decay: float, buffer_size: Optional[int] = None):
    """
    启用EMA归一化
    
    参数:
        decay: EMA衰减系数（0-1）
        buffer_size: 缓冲区大小
    """

def minmax_normalize(self, depth: torch.Tensor, reset_ema: Optional[List[bool]] = None):
    """
    对深度图进行归一化
    
    参数:
        depth: 深度张量 [B, 1, H, W]
        reset_ema: 每帧是否重置EMA的标志列表
    """

@staticmethod
def save_normalized_depth(depth, file_path, png_info={}, min_depth_value=None, max_depth_value=None):
    """
    保存归一化深度图为16位PNG
    
    参数:
        depth: 归一化深度张量 [1, H, W]
        file_path: 输出文件路径
        png_info: PNG元数据
        min_depth_value: 最小深度值（用于恢复度量深度）
        max_depth_value: 最大深度值
    """

@staticmethod
def load_depth(file_path) -> Tuple[torch.Tensor, dict]:
    """
    从文件加载深度图
    
    返回:
        depth: 深度张量
        metadata: 元数据字典
    """
```

### 工厂函数

```python
def create_depth_model(model_type: str) -> BaseDepthModel:
    """
    创建深度模型实例
    
    参数:
        model_type: 模型类型字符串
        
    返回:
        BaseDepthModel的子类实例
        
    异常:
        ValueError: 不支持的模型类型
    """
```

## 使用示例

### 基本使用

```python
from iw3.depth_model_factory import create_depth_model

# 创建模型
model = create_depth_model("Any_B")

# 加载到GPU
model.load(gpu=0)

# 推理
import torch
from torchvision import transforms

# 准备输入
image = ... # PIL Image
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
x = transform(image).unsqueeze(0).cuda()

# 获取深度
depth = model.infer(x)

# 归一化并保存
normalized_depth = model.minmax_normalize(depth)
model.save_normalized_depth(normalized_depth[0], "depth.png")
```

### 视频处理

```python
# 启用EMA归一化
model.enable_ema(decay=0.9, buffer_size=30)

# 处理视频帧
for frame in video_frames:
    depth = model.infer(frame)
    normalized = model.minmax_normalize(depth)
    # 处理归一化深度...

# 处理剩余帧
remaining = model.flush_minmax_normalize()
```

### 多GPU推理

```python
# 使用多个GPU
model = create_depth_model("Any_L")
model.load(gpu=[0, 1, 2, 3])  # 使用4个GPU

# 推理自动分布到多个GPU
depth = model.infer(batch_input)
```

## 性能考虑

1. **模型选择**：
   - 实时应用选择S（小）模型
   - 质量优先选择L（大）模型
   - 视频处理推荐VDA系列

2. **内存使用**：
   - 大模型可能需要8GB+ VRAM
   - 使用`--low-vram`选项减少内存使用

3. **批处理**：
   - 尽可能使用批处理提升效率
   - VDA模型内部使用批大小32

4. **编译优化**：
   - 使用`model.compile()`提升推理速度
   - 首次编译需要时间，后续会更快

## 常见问题

**Q: 如何选择合适的深度模型？**
A: 
- 室内场景：ZoeD_N, Any_V2_N_*
- 室外场景：ZoeD_K, Any_V2_K_*
- 通用场景：Any_B, ZoeD_NK
- 视频处理：VDA系列
- 最高质量：DepthPro（仅图像）

**Q: 相对深度vs度量深度？**
A: 
- 相对深度：归一化到[0,1]，表示相对远近
- 度量深度：真实世界距离（米）
- 3D生成通常使用相对深度即可

**Q: EMA归一化参数如何设置？**
A:
- 单张图像：decay=0, buffer_size=1
- 短视频：decay=0.75, buffer_size=1
- 长视频：decay=0.9, buffer_size=30

## 扩展指南

### 添加新的深度模型

1. 创建新的模型类，继承BaseDepthModel
2. 实现所有抽象方法
3. 在depth_model_factory.py中注册
4. 添加模型文件到pretrained_models/hub/checkpoints/

示例：
```python
class MyDepthModel(BaseDepthModel):
    def __init__(self, model_type):
        super().__init__(model_type)
    
    @classmethod
    def get_name(cls):
        return "MyDepth"
    
    def supported(cls, model_type):
        return model_type in ["My_S", "My_L"]
    
    # 实现其他必要方法...
```

### 自定义归一化策略

可以通过继承或修改EMAMinMaxScaler类来实现自定义的归一化策略：

```python
class CustomScaler(EMAMinMaxScaler):
    def update(self, frame, return_minmax=False):
        # 自定义归一化逻辑
        pass
```

## 相关文档

- [立体图像生成算法模块](stereo_generation_algorithms.md)
- [视频处理管线](video_processing_pipeline.md)
- [参数系统与配置](parameter_system.md)