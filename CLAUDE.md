# Nunif Project Instructions

please ultrathink

## Virtual Environment (venv) Usage

### Correct way to activate and use venv:

**Windows (Git Bash/WSL):**
```bash
source venv/Scripts/activate
python -m iw3.onnx.export_iw3
deactivate  # when done
```

## 子项目
- 3d-player: 参考 @3d-player/CLAUDE.md 运行测试 `cd "C:\games\nunif\3d-player" && powershell -ExecutionPolicy Bypass -File test.ps1 "[test_stereo_video_decoder.cpp]"`
- spike: C++ 实验项目，包含 TensorRT 推理实现 (spike/src/infer_iw3.cpp)
- iw3: Python 立体视觉模块，包含 ONNX 导出 (iw3/onnx/export_iw3.py)