# MKV Player - Claude Code Instructions

## Build Commands

### 完整构建和测试 (推荐)
在WSL环境下编译并运行所有测试：
```bash
cmd.exe /c "build.bat"
```

这个命令会：
- 自动检测并链接FFmpeg、CUDA、TensorRT、DirectX11依赖
- 编译生成Debug版本可执行文件
- 复制所需DLL到输出目录
- 自动运行所有测试并输出结果

### 只编译不运行测试
```bash
cmd.exe /c "cmake --build build --config Debug"
```

## Test Commands

### 运行所有测试
```bash
cmd.exe /c "build\\Debug\\spike.exe"
```

### 运行测试并显示详细信息
```bash
cmd.exe /c "build\\Debug\\spike.exe -v high"
```

### 运行特定标签的测试
```bash
# 运行frame_decoder相关测试
cmd.exe /c "build\\Debug\\spike.exe [frame_decoder]"

# 运行packet_demuxer相关测试  
cmd.exe /c "build\\Debug\\spike.exe [packet_demuxer]"

# 运行RGB相关测试
cmd.exe /c "build\\Debug\\spike.exe [rgb_verification]"
```

### 运行测试并显示成功的断言
```bash
cmd.exe /c "build\\Debug\\spike.exe -s"
```

### 列出所有测试
```bash
cmd.exe /c "build\\Debug\\spike.exe --list-tests"
```

### 查看测试帮助
```bash
cmd.exe /c "build\\Debug\\spike.exe --help"
```

## Player Applications

### CLI播放器 (命令行视频播放器)
```bash
cmd.exe /c "build\\Debug\\cli_player.exe"
```
功能：
- 将视频帧保存为BMP文件
- 可配置保存间隔和数量
- 播放速度控制
- 用于无头环境测试

### GUI播放器 (图形界面视频播放器)
```bash
cmd.exe /c "build\\Debug\\gui_player.exe"
```
功能：
- DirectX11实时视频渲染
- DirectSound音频播放
- 键盘控制：SPACE(暂停/恢复), ESC(退出)
- 实时音视频同步播放

## 项目依赖
- DirectX11 (d3d11.lib, dxgi.lib, d3dcompiler.lib)
- FFmpeg 7.1.1 (位于 C:/games/ffmpeg-7.1.1)
- CUDA 12.9 (位于 C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9)
- TensorRT 10 (位于 C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT)
- Catch2 v3.4.0 (通过FetchContent自动获取)