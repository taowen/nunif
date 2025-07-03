# iw3 项目系统性文档写作计划

## 概述
本计划旨在为iw3项目的各个模块和功能特性创建详细的技术文档，以便开发者理解、使用和扩展项目。

## 文档结构规划

### 1. 核心模块文档

#### 1.1 深度估计模型模块 (depth_estimation_models.md)
- **基础架构**
  - `base_depth_model.py` - 深度模型基类设计
  - `depth_model_factory.py` - 模型工厂模式实现
  - `null_depth_model.py` - 空模型实现

- **具体模型实现**
  - `zoedepth_model.py` - ZoeDepth系列模型
  - `depth_anything_model.py` - Depth Anything系列模型
  - `depth_pro_model.py` - Apple Depth Pro模型
  - `video_depth_anything_model.py` - 视频深度估计模型

- **深度后处理**
  - `depth_scaler.py` - 深度缩放与调整
  - `dilation.py` - 边缘膨胀处理

#### 1.2 立体图像生成算法模块 (stereo_generation_algorithms.md)
- **翘曲算法**
  - `forward_warp.py` - 前向翘曲实现
  - `backward_warp.py` - 后向翘曲实现
  
- **神经网络模型**
  - `models/row_flow.py` - Row Flow V1
  - `models/row_flow_v2.py` - Row Flow V2
  - `models/row_flow_v3.py` - Row Flow V3
  - `models/light_inpaint_v1.py` - 轻量级修复模型
  - `models/depth_aa.py` - 深度抗锯齿模型

- **立体效果处理**
  - `mapper.py` - 深度映射函数
  - `training/sbs/stereoimage_generation.py` - 立体图像生成算法

#### 1.3 视频处理管线 (video_processing_pipeline.md)
- **主处理流程**
  - `utils.py` - 核心处理函数 (iw3_main, process_image, process_video)
  - `cli.py` - 命令行参数处理

- **视频特定功能**
  - 场景检测与分割
  - 批处理优化
  - 多GPU并行处理
  - 时序一致性处理

### 2. 输出格式与特效文档

#### 2.1 输出格式模块 (output_formats.md)
- **立体格式**
  - SBS (Side-by-Side) - 全尺寸和半尺寸
  - TB (Top-Bottom) - 全尺寸和半尺寸
  - 交叉眼 (Cross-eyed)
  
- **特殊格式**
  - `anaglyph.py` - 红青立体图实现
  - `equirectangular.py` - VR180等角投影
  
- **视频编码**
  - 支持的编解码器
  - 质量与性能平衡
  - 格式兼容性

### 3. 用户接口文档

#### 3.1 GUI模块 (gui_interface.md)
- `gui.py` - 主GUI实现
- `desktop/gui.py` - 桌面版GUI
- 界面组件与交互设计
- 参数可视化

#### 3.2 CLI模块 (cli_interface.md)
- `cli.py` - 命令行接口
- 参数详解
- 使用示例
- 批处理脚本

### 4. 配置与参数系统文档

#### 4.1 参数系统 (parameter_system.md)
- **核心参数**
  - Divergence（视差强度）
  - Convergence（会聚点）
  - IPD Offset（瞳距偏移）
  - Foreground Scale（前景缩放）
  - Edge Dilation（边缘膨胀）

- **高级参数**
  - Synthetic View（合成视图模式）
  - Depth Resolution（深度分辨率）
  - Color Space（色彩空间）

#### 4.2 配置管理 (configuration.md)
- `export_config.py` - 配置导出
- 配置文件格式
- 预设管理

### 5. 训练与开发文档

#### 5.1 训练模块 (training_modules.md)
- **数据集准备**
  - `training/sbs/create_training_data.py` - 训练数据创建
  - `training/sbs/dataset.py` - 数据集加载
  - `training/extract_keyframes.py` - 关键帧提取

- **训练器实现**
  - `training/sbs/trainer.py` - SBS训练器
  - `training/depth_aa/trainer.py` - 深度AA训练器
  - `training/inpaint/trainer.py` - 修复模型训练器

- **优化工具**
  - `training/find_mapper.py` - 映射函数查找

### 6. 辅助功能文档

#### 6.1 桌面实时3D (desktop_realtime_3d.md)
- `desktop/` 目录下的所有模块
- 屏幕捕获与流式传输
- 实时处理管线

#### 6.2 工具集 (utility_tools.md)
- `download_models.py` - 模型下载管理
- `onnx/export_iw3.py` - ONNX导出
- `locales/` - 国际化支持

#### 6.3 实验性功能 (experimental_features.md)
- `poc/` - 概念验证代码
- `nunif_addon.py` - Nunif框架集成

### 7. API参考文档

#### 7.1 公共API (public_api.md)
- 主要类和函数
- 扩展点
- 回调机制

#### 7.2 内部API (internal_api.md)
- 内部实现细节
- 数据流
- 算法细节

## 文档编写标准

每个文档应包含：
1. **模块概述** - 功能和用途说明
2. **技术原理** - 算法或实现原理
3. **API参考** - 类、函数、参数详解
4. **使用示例** - 代码示例和最佳实践
5. **性能考虑** - 性能影响和优化建议
6. **常见问题** - FAQ和故障排除
7. **扩展指南** - 如何扩展或修改功能

## 执行顺序

1. 高优先级：核心模块文档（深度估计、立体生成、视频处理）
2. 中优先级：用户接口和参数系统文档
3. 低优先级：训练模块和辅助功能文档

## 预期成果

- 完整的技术文档体系
- 便于新开发者快速上手
- 支持项目的长期维护和扩展
- 提高代码的可理解性和可维护性