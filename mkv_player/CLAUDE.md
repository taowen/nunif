# MKV Player - Claude Code Instructions

## Build Commands

在WSL环境下编译Windows DirectX11项目：
```bash
cmd.exe /c "build.bat"
```

这个命令会：
- 自动检测并链接FFmpeg、CUDA、TensorRT、DirectX11依赖
- 编译生成Debug版本可执行文件
- 复制所需DLL到输出目录
- 运行生成的程序验证

## 项目依赖
- DirectX11 (d3d11.lib, dxgi.lib, d3dcompiler.lib)
- FFmpeg 7.1.1 (位于 C:/games/ffmpeg-7.1.1)
- CUDA 12.9 (位于 C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9)
- TensorRT 10 (位于 C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT)
- Catch2 v3.4.0 (通过FetchContent自动获取)

## 注意事项
- 项目使用C++20标准
- 在WSL中可以直接调用Windows构建工具
- 无需SSH到Windows环境