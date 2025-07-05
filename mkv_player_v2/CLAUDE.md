# MKV Player - Claude Code Instructions

## Build Commands

### 快速构建脚本 (推荐)
新的build.bat脚本支持多种构建目标：

```bash
# 构建并运行测试 (默认)
cmd.exe /c "build.bat test"

### 手动构建命令
```bash
# 配置项目
cmd.exe /c "cmake -S . -B build"

# 构建特定目标
cmd.exe /c "cmake --build build --config Debug --target spike"      # 测试

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