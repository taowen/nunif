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

### 代码组织和文件结构规则

#### 目录结构
```
src/           # 核心实现代码
├── *.h        # 头文件 - 类定义和接口声明
├── *.cpp      # 实现文件 - 具体功能实现
tests/         # 测试代码
├── test_*.cpp # 单元测试文件，以 test_ 开头
```

#### 文件命名规则
- **头文件：** `class_name.h` - 小写，下划线分隔
- **实现文件：** `class_name.cpp` - 与头文件同名
- **测试文件：** `test_class_name.cpp` - 以 test_ 开头

#### 依赖关系组织
- **底层组件：** `MKVStreamReader` - 负责底层文件读取和解析
- **中层组件：** `PacketDemuxer` - 组合 `MKVStreamReader`，提供包同步功能
- **上层组件：** `FrameDecoder` - 使用 `PacketDemuxer` 进行解码
- **应用层：** `RGBFrameDecoder` - 使用 `FrameDecoder` 进行特定格式转换

#### 访问模式
- **直接访问：** 上层通过 `getComponent()` 方法访问底层组件
- **避免代理：** 不在上层重复实现底层的所有方法
- **组合设计：** 通过组合而非继承来复用功能

#### 测试文件对应关系
- `src/packet_demuxer.h` ↔ `tests/test_packet_demuxer.cpp`
- `src/frame_decoder.h` ↔ `tests/test_frame_decoder.cpp`
- `src/mkv_stream_reader.h` ↔ `tests/test_mkv_stream_reader.cpp`

#### 修改影响范围
当修改某个组件的接口时，需要检查和更新：
1. 对应的实现文件（`.cpp`）
2. 使用该组件的上层代码
3. 对应的测试文件
4. 可能依赖该组件的其他组件

## 资源管理原则

### 双缓冲机制
所有解码器类都采用双缓冲设计，支持两个数据对象同时存在：
- `PacketDemuxer`: `borrowed_pairs_[2]` 存储PacketPair
- `HwFrameDecoder`: `borrowed_pairs_[2]` 存储HwFramePair  
- `RGBFrameDecoder`: `borrowed_pairs_[2]` 存储RGBFramePair

### 内部资源管理
**核心原则：使用方完全不需要管理资源**
- 所有FFmpeg资源（AVPacket、AVFrame）由类内部池化管理
- 所有DirectX资源使用ComPtr智能指针自动管理
- 资源在下次读取调用时自动复用，析构时自动释放

### 资源所有权规则
1. **PacketDemuxer**: 
   - 管理AVPacket生命周期
   - 使用方不要手动释放audio_packet和video_packet
   
2. **HwFrameDecoder**:
   - 管理AVFrame生命周期和硬件加速上下文
   - 使用方不要调用av_frame_free()或任何释放函数
   
3. **RGBFrameDecoder**:
   - 管理D3D11纹理和视频处理器资源
   - 使用ComPtr自动管理DirectX对象生命周期

### 接口设计规范
- 所有read方法返回引用，而非拷贝
- 标记所有返回的数据结构为"内部管理，使用方无需释放"
- 提供is_valid标志位判断数据有效性
- 提供状态查询方法（hasValidPair、getValidPairCount等）