# MKV Player V3 Project Documentation

## Project Overview
This is a C++-based MKV video player project with hardware-accelerated decoding. The project uses modern C++ standards (C++20), CMake build system, and FFmpeg D3D11VA for GPU-based video processing.

## Architecture Design

### Core Design Philosophy
- **Separate Audio/Video Processing**: Independent audio and video decoding pipelines
- **Hardware-First Approach**: FFmpeg D3D11VA for zero-copy GPU decoding
- **Double Buffer Memory Management**: 2-frame rotation for optimal GPU memory usage
- **Simplified State Management**: Delegate state queries to underlying components

### Current Pipeline (Implemented)
```
mkv_stream_reader -> hw_video_decoder (with double buffering)
mkv_stream_reader -> audio_decoder
```

### Component Responsibilities
- **mkv_stream_reader**: Parse MKV container, extract video/audio packets
- **hw_video_decoder**: FFmpeg D3D11VA hardware video decoding with frame rotation
- **audio_decoder**: Audio decoding to PCM

## Memory Management Strategy

### Double Buffer Frame Rotation
- **Frame Pool**: 2 pre-allocated AVFrame objects (`hw_frames_[0]`, `hw_frames_[1]`)
- **Rotation Logic**: Frame index alternates 0→1→0→1... after each decode
- **Memory Reuse Pattern**: Frame3 reuses Frame1 memory, Frame4 reuses Frame2 memory
- **Zero Allocation**: No runtime memory allocation after initialization
- **Verified Behavior**: Tests confirm `frame3.frame == frame1.frame` pointer equality

### Hardware Context Management
- **FFmpeg Auto-Management**: Use `av_hwdevice_ctx_create()` for D3D11VA context
- **No Manual D3D11**: Avoid hand-crafted DirectX device management
- **Simplified Resource Cleanup**: FFmpeg handles all GPU resource lifecycle

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
│   ├── hw_video_decoder.cpp/.h      # FFmpeg D3D11VA video decoding
│   └── audio_decoder.cpp/.h         # Audio decoding
├── test_data/             # Test media files
│   └── sample_hw.mkv
└── tests/                 # Unit tests
    └── test_*.cpp
```

## Build System

### Dependencies
- **FFmpeg**: Container parsing, audio decoding, D3D11VA hardware context
- **Catch2**: Unit testing framework

### CMake Configuration
- **C++ Standard**: C++20
- **Target Platform**: Windows (D3D11VA requirement)
- **Build Target**: `spike` - Test executable with all components

### Build Commands

#### WSL Environment
```bash
# Run Windows batch script from WSL
cmd.exe /c "build.bat test"

# Alternative: Use PowerShell
powershell.exe -Command "& './build.bat' test"

# Note: WSL can access Windows executables and build tools
# The /mnt/c/ mount point provides access to Windows filesystem
```

## Key Implementation Lessons

### FFmpeg D3D11VA Integration Success
- **Let FFmpeg Manage Everything**: `av_hwdevice_ctx_create()` handles all D3D11 complexity
- **Avoid Manual DirectX**: Hand-crafted D3D11 device management causes conflicts
- **No HWFramesContext Needed**: Hardware decoding works without complex frame context setup

### Double Buffer Frame Management
- **Simple Array + Index**: `AVFrame* hw_frames_[2]` + `current_frame_index_`
- **Rotation After Use**: Increment index in `fillDecodedFrame()` after assigning frame
- **Memory Pattern Verified**: `frame[n] == frame[n+2]` rotation confirmed by tests

### State Management Anti-Pattern
- **Don't Copy State**: Redundant `is_open_`, `is_eof_` variables create sync issues  
- **Delegate to Source**: Use `stream_reader_->isOpen()` instead of local flags
- **Single Source of Truth**: Avoid duplicating information from underlying components

## Development Notes

### Current Status (✅ = Working, 🔧 = Ready for Enhancement)
- ✅ **MKVStreamReader**: MKV container parsing and packet extraction
- ✅ **AudioDecoder**: PCM audio decoding with format detection
- ✅ **HwVideoDecoder**: D3D11VA hardware video decoding with double buffering
- 🔧 **Color Space Conversion**: Can be added as next step for display
- 🔧 **Async Wrappers**: Worker thread patterns can be layered on top
- 🔧 **Audio/Video Sync**: Timestamp-based synchronization ready to implement

### Testing Architecture Success
- **Memory Reuse Tests**: Verify `frame[n] == frame[n+2]` pointer equality
- **Double Buffer Validation**: Confirm alternating frame allocation pattern
- **Integration Tests**: Real MKV file processing with hardware acceleration
- **WSL Cross-Platform**: Windows build tools accessible from Linux environment

### WSL Development Workflow
- ✅ `cmd.exe /c "build.bat test"` works perfectly from WSL
- ✅ Windows filesystem accessible via `/mnt/c/` mount point  
- ✅ All Windows build tools (MSVC, MSBuild) accessible from WSL
- ✅ Hardware acceleration works on Windows host
- 💡 WSL provides excellent cross-platform development workflow
