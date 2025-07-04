# MKV Player - Claude Code Instructions

## Build Commands

### 快速构建脚本 (推荐)
新的build.bat脚本支持多种构建目标：

```bash
# 构建并运行测试 (默认)
cmd.exe /c "build.bat"
cmd.exe /c "build.bat test"

# 只构建CLI播放器
cmd.exe /c "build.bat cli"

# 只构建GUI播放器  
cmd.exe /c "build.bat gui"

# 构建所有目标
cmd.exe /c "build.bat all"

# 测试所有构建目标
cmd.exe /c "test_builds.bat"
```

### 手动构建命令
```bash
# 配置项目
cmd.exe /c "cmake -S . -B build"

# 构建特定目标
cmd.exe /c "cmake --build build --config Debug --target spike"      # 测试
cmd.exe /c "cmake --build build --config Debug --target cli_player" # CLI播放器
cmd.exe /c "cmake --build build --config Debug --target gui_player" # GUI播放器

# 构建所有目标
cmd.exe /c "cmake --build build --config Debug"
```

构建过程会：
- 自动检测并链接FFmpeg、CUDA、TensorRT、DirectX11依赖
- 编译生成Debug版本可执行文件
- 复制所需DLL到输出目录

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

## CMake Targets

项目包含以下可执行文件目标：

### spike (测试套件)
- **目标名**: `spike`
- **可执行文件**: `build/Debug/spike.exe`
- **用途**: Catch2测试套件，包含所有单元测试
- **依赖**: Catch2, FFmpeg, CUDA, TensorRT, DirectX11

### cli_player (命令行播放器)  
- **目标名**: `cli_player`
- **可执行文件**: `build/Debug/cli_player.exe`
- **用途**: 命令行视频播放器，将帧保存为BMP文件
- **依赖**: FFmpeg, CUDA, TensorRT, DirectX11

### gui_player (图形界面播放器)
- **目标名**: `gui_player` 
- **可执行文件**: `build/Debug/gui_player.exe`
- **用途**: 实时音视频播放器，支持DirectX11渲染和DirectSound音频
- **依赖**: FFmpeg, CUDA, TensorRT, DirectX11, DirectSound

## 项目依赖
- DirectX11 (d3d11.lib, dxgi.lib, d3dcompiler.lib)
- DirectSound (dsound.lib, dxguid.lib) - 仅GUI播放器需要
- FFmpeg 7.1.1 (位于 C:/games/ffmpeg-7.1.1)
- CUDA 12.9 (位于 C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9)
- TensorRT 10 (位于 C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT)
- Catch2 v3.4.0 (通过FetchContent自动获取) - 仅测试目标需要