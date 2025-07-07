# MKV Player V3 Project Documentation

## Project Overview
This is a C++-based MKV video player project with hardware-accelerated decoding and rendering. The project uses modern C++ standards (C++20), CMake build system, and DirectX 11 for GPU-based video processing.

## Architecture Design

### Core Design Philosophy
- **Separate Audio/Video Processing**: Independent audio and video decoding pipelines
- **Hardware-First Approach**: All decoding and color conversion on GPU using DirectX 11
- **GPU Memory Management**: Zero-copy processing with all data remaining in VRAM
- **Asynchronous Design**: Worker threads for non-blocking decode operations

### Video Processing Pipeline
```
mkv_stream_reader -> hw_video_decoder -> rgb_video_decoder -> async_rgb_video_decoder
```

### Audio Processing Pipeline  
```
mkv_stream_reader -> audio_decoder -> async_audio_decoder
```

### Component Responsibilities
- **mkv_stream_reader**: Parse MKV container, extract video/audio packets
- **hw_video_decoder**: DirectX 11 hardware video decoding (NVDEC/Intel QSV/AMD VCE)
- **rgb_video_decoder**: Color space conversion using VideoProcessorBlt
- **async_rgb_video_decoder**: Asynchronous wrapper with worker thread
- **audio_decoder**: Audio decoding to PCM
- **async_audio_decoder**: Asynchronous audio processing

## Memory Management Strategy

### GPU Memory Model
- **Who Allocates, Who Releases**: Each component manages its own VRAM allocations
- **Double Buffering**: Maintain 2 valid frames at all times for smooth playback
- **Frame Recycling**: 3rd decode operation reuses oldest frame's memory
- **Zero CPU-GPU Transfer**: All processing stays in VRAM

### Async Frame Management
- **Current Frame**: Available for rendering
- **Next Frame**: Being prepared by worker thread
- **Memory Lifecycle**: Only released on player shutdown

## DirectX 11 Integration

### Hardware Decoding
- Use DirectX Video Acceleration (DXVA) APIs
- ID3D11VideoDevice for decoder creation
- ID3D11VideoDecoder for actual decoding operations

### Color Space Conversion
- ID3D11VideoProcessor for format conversion
- VideoProcessorBlt for efficient GPU-based color space transforms
- Support for various input formats (NV12, YUV420, etc.) to RGB

### Resource Management
- ID3D11Texture2D for frame storage
- Shared texture handles for zero-copy operations
- DXGI integration for display output

## Directory Structure

```
mkv_player_v3/
├── CLAUDE.md              # This documentation
├── CMakeLists.txt         # CMake build configuration
├── build.bat              # Windows build script
├── build/                 # Build output directory
├── cmake/                 # CMake modules
│   ├── FindCUDA.cmake
│   ├── FindFFmpeg.cmake
│   └── FindTensorRT.cmake
├── src/                   # Source code directory
│   ├── mkv_stream_reader.cpp/.h     # MKV container parsing
│   ├── hw_video_decoder.cpp/.h      # DirectX 11 video decoding
│   ├── rgb_video_decoder.cpp/.h     # Color space conversion
│   ├── async_rgb_video_decoder.cpp/.h  # Async video wrapper
│   ├── audio_decoder.cpp/.h         # Audio decoding
│   └── async_audio_decoder.cpp/.h   # Async audio wrapper
├── test_data/             # Test media files
│   └── sample_hw.mkv
└── tests/                 # Unit tests
    └── test_*.cpp
```

## Build System

### Dependencies
- **FFmpeg**: Container parsing and audio decoding
- **DirectX 11**: Hardware video decoding and processing
- **DXVA**: Video acceleration APIs
- **Catch2**: Unit testing framework

### CMake Configuration
- **C++ Standard**: C++20
- **Target Platform**: Windows (DirectX 11 requirement)
- **Build Targets**: 
  - `spike`: Test executable
  - `gui_player`: Main player application

### Build Commands
```bash
# Build and run tests
build.bat test

# Build GUI player
build.bat gui

# Direct CMake build
cd build
cmake --build . --config Debug --target spike
```

## Development Workflow

### Implementation Order
1. **Synchronous Components First**: Implement and test sync decoders
2. **Add Async Wrappers**: Layer async functionality on top
3. **Integration Testing**: Test full pipeline with real media files
4. **Performance Optimization**: Profile and optimize GPU usage

### Testing Strategy
- **Unit Tests**: Each component tested in isolation
- **Integration Tests**: End-to-end pipeline testing
- **Performance Tests**: Frame rate and memory usage validation
- **Hardware Compatibility**: Test across different GPU vendors

### Audio/Video Sync Strategy
- **Independent Playback**: Audio and video run separately initially
- **Frame Dropping**: Video adds frame skipping for sync later
- **Timestamp Management**: Use presentation timestamps for coordination

## Key Technical Decisions

### Why DirectX 11 Over Other APIs
- **Hardware Decode Support**: Excellent DXVA integration
- **VideoProcessor**: Built-in color space conversion
- **Windows Ecosystem**: Native Windows media acceleration
- **Stable API**: Mature and well-documented

### Why Separate Audio/Video
- **Simplified Development**: Each pipeline can be optimized independently
- **Easier Debugging**: Isolate issues to specific domain
- **Flexible Sync**: Can implement various sync strategies later

### Why GPU-Only Processing
- **Performance**: Avoid expensive CPU-GPU memory transfers
- **Efficiency**: Leverage specialized video processing units
- **Scalability**: Better performance with high-resolution content

## Development Notes

### Current Status
- ✅ MKVStreamReader: Implemented and tested
- ⏳ Hardware decoders: To be implemented
- ⏳ Async wrappers: To be implemented
- ⏳ GUI integration: To be implemented

### Next Steps
1. Implement hw_video_decoder with DirectX 11 DXVA
2. Add rgb_video_decoder with VideoProcessorBlt
3. Create async wrappers with worker threads
4. Integrate audio decoding pipeline
5. Build GUI player application

### Performance Targets
- **4K Video**: Smooth playback at 60fps
- **Memory Usage**: Minimal VRAM footprint with double buffering
- **Latency**: Sub-frame decode latency for responsive playback
- **CPU Usage**: Minimal CPU involvement in decode pipeline