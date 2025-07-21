# Nunif Project Instructions

## Virtual Environment (venv) Usage

### Correct way to activate and use venv:

**Windows (Git Bash/WSL):**
```bash
source venv/Scripts/activate
python -m iw3.onnx.export_iw3
deactivate  # when done
```

## 子项目
- 3d-player: 参考 3d-player/CLAUDE.md 和 3d-player/README.md
- spike: C++ 实验项目，包含 TensorRT 推理实现 (spike/src/infer_iw3.cpp)
- mkv_player: 视频播放器相关实验
- iw3: Python 立体视觉模块，包含 ONNX 导出 (iw3/onnx/export_iw3.py)

## TensorRT 集成状态

### 相关文件
- **参考实现**：`spike/src/infer_iw3.cpp` - 完整的 TensorRT 推理实现
- **测试代码**：`3d-player/tests/test_tensorrt.cpp` - TensorRT 功能测试
- **ONNX 模型**：`stereo_module_half_sbs.onnx` - 立体视觉推理模型
- **缓存文件**：`stereo_module_half_sbs_fp16.trt` - TensorRT 引擎缓存

### 当前状态
- ✅ ONNX 模型导出和解析正常
- ✅ TensorRT 运行时初始化成功
- ✅ 动态形状检测功能正常 (-1, 4, -1, -1)
- ❌ 优化配置文件设置存在问题 (addOptimizationProfile 失败)

### 运行 TensorRT 测试
```bash
# 在 3d-player 目录下运行
cd 3d-player
build/Debug/integration-test.exe "TensorRT ONNX inference" -s
```
