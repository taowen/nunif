# End-to-End 立体图像生成

这是一个简化的端到端脚本，可以从单张图片生成立体视觉的左眼和右眼图像。

## 功能特性

- 🖼️ 单张图像输入
- 🧠 使用 Distill Any Depth Small 模型进行深度估计
- 👁️ 生成左眼和右眼立体图像
- 📦 只依赖 PyTorch 等开源库
- 🚀 简化的代码结构，易于理解和修改

## 依赖要求

```bash
pip install torch torchvision pillow numpy
```

## 使用方法

### 基本用法

```bash
# 处理单张图片
python iw3/end_to_end.py --input path/to/your/image.jpg

# 指定输出目录
python iw3/end_to_end.py --input path/to/your/image.jpg --output ./my_output

# 调整散度和收敛参数
python iw3/end_to_end.py --input path/to/your/image.jpg --divergence 3.0 --convergence 0.3
```

### 测试模式

如果存在默认测试图像，可以运行：

```bash
python iw3/end_to_end.py --test
```

### 参数说明

- `--input`: 输入图像路径
- `--output`: 输出目录（默认: `./output`）
- `--divergence`: 散度强度，控制立体效果强度（0-5，默认: 2.0）
- `--convergence`: 收敛平面，控制焦点位置（0-1，默认: 0.5）
- `--test`: 使用默认测试图像进行测试

## 输出文件

脚本会在输出目录中生成以下文件：

- `left_eye.png`: 左眼图像
- `right_eye.png`: 右眼图像  
- `depth.png`: 深度图
- `side_by_side.png`: 左右并排的立体图像

## 算法说明

### 深度估计

使用 Distill Any Depth Small 模型：
- 模型大小较小，推理速度快
- 基于 ViT-S (Vision Transformer Small) 架构
- 通过 torch.hub 自动下载

### 立体视觉生成

- **首选方法**: row_flow_v3_sym 神经网络模型（如果可用）
- **备用方法**: 网格采样变形（Grid Sample Warping）

立体效果通过以下步骤实现：
1. 根据深度信息计算视差
2. 应用散度参数调整立体强度
3. 使用收敛参数确定焦点平面
4. 生成左右眼视角的图像

## 参数调整建议

### 散度 (Divergence)
- `1.0-2.0`: 轻微立体效果，适合近景
- `2.0-3.0`: 中等立体效果，通用设置
- `3.0-5.0`: 强烈立体效果，适合远景

### 收敛 (Convergence)  
- `0.0`: 焦点在前景
- `0.5`: 焦点在中景（默认）
- `1.0`: 焦点在背景

## 注意事项

1. **GPU 加速**: 如果有可用的 GPU，会自动使用 CUDA 加速
2. **内存使用**: 大图像可能需要更多内存
3. **模型下载**: 首次运行时会自动下载深度估计模型
4. **网格采样**: 如果神经网络立体模型不可用，会使用简化的网格采样方法

## 故障排除

### 常见问题

1. **模型下载失败**
   - 检查网络连接
   - 确保可以访问 GitHub 和 Hugging Face

2. **内存不足**
   - 减小输入图像尺寸
   - 如果使用 GPU，检查显存使用情况

3. **输出效果不佳**
   - 调整散度和收敛参数
   - 确保输入图像有明显的前后景深度变化

## 代码结构

主要函数：
- `batch_preprocess_depth()`: 图像预处理
- `infer_depth()`: 深度估计推理
- `apply_divergence_symmetric()`: 对称立体变形
- `backward_warp()`: 后向变形
- `process_single_image()`: 主处理函数

## 扩展建议

- 添加批处理支持
- 集成更多深度估计模型
- 支持视频处理  
- 添加更多输出格式（如红蓝眼镜格式）

## 许可证

请参考原项目的许可证要求。 