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
mkv_stream_reader -> async_audio_decoder -> audio_player (WASAPI output)
hw_video_decoder -> rgb_video_decoder (with RGB color space conversion)
external_d3d11_resources -> video_player (with dependency injection)
```

### Component Responsibilities
- **mkv_stream_reader**: Parse MKV container, extract video/audio packets
- **hw_video_decoder**: FFmpeg D3D11VA hardware video decoding with frame rotation
- **rgb_video_decoder**: Hardware NV12→RGB color space conversion using D3D11 Video Processor
- **video_player**: Timer-driven video playback with fake signal source and frame rate conversion
- **audio_decoder**: Audio decoding to PCM
- **async_audio_decoder**: Asynchronous audio decoding with worker thread
- **audio_player**: WASAPI audio output with external timer integration

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
│   ├── rgb_video_decoder.cpp/.h     # Hardware RGB color space conversion
│   ├── video_player.cpp/.h          # Timer-driven video playback with dependency injection
│   ├── gui_player.cpp               # GUI application with D3D11 resource management
│   ├── audio_decoder.cpp/.h         # Audio decoding
│   ├── async_audio_decoder.cpp/.h   # Asynchronous audio decoding wrapper
│   └── audio_player.cpp/.h          # WASAPI audio output player
├── test_data/             # Test media files
│   └── sample_hw.mkv
└── tests/                 # Unit tests
    └── test_*.cpp
```

## Build System

### Dependencies
- **FFmpeg**: Container parsing, audio decoding, D3D11VA hardware context, swscale for software color conversion
- **DirectX 11**: Video Processor for hardware color space conversion
- **WASAPI**: Windows audio output (ole32.lib, oleaut32.lib)
- **Catch2**: Unit testing framework

### CMake Configuration
- **C++ Standard**: C++20
- **Target Platform**: Windows (D3D11VA requirement)
- **Build Targets**: 
  - `spike` - Test executable with all components
  - `gui_player` - GUI application with video playback

### Build Commands

#### WSL Environment
```bash
# Run tests
cmd.exe /c "build.bat test"

# Run GUI application  
cmd.exe /c "build.bat gui"

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

### RGB Color Space Conversion
- **Hardware Acceleration**: Uses D3D11 Video Processor for NV12→RGB conversion
- **Zero-Copy Pipeline**: Direct GPU-to-GPU conversion without CPU involvement
- **Double Buffer RGB Textures**: `RgbFrame rgb_frames_[2]` for alternating RGB output
- **Resource Caching**: Output views cached to minimize D3D11 object creation overhead
- **Format**: DXGI_FORMAT_B8G8R8A8_UNORM (32-bit BGRA)

### State Management Anti-Pattern
- **Don't Copy State**: Redundant `is_open_`, `is_eof_` variables create sync issues  
- **Delegate to Source**: Use `stream_reader_->isOpen()` instead of local flags
- **Single Source of Truth**: Avoid duplicating information from underlying components

## Development Notes

### Current Status (✅ = Working, 🔧 = Ready for Enhancement)
- ✅ **MKVStreamReader**: MKV container parsing and packet extraction
- ✅ **AudioDecoder**: PCM audio decoding with format detection
- ✅ **AsyncAudioDecoder**: Asynchronous audio decoding wrapper
- ✅ **AudioPlayer**: WASAPI audio output with external timer integration
- ✅ **HwVideoDecoder**: D3D11VA hardware video decoding with double buffering
- ✅ **RgbVideoDecoder**: Hardware NV12→RGB color space conversion with double buffering
- ✅ **VideoPlayer**: Timer-driven video playback with dependency injection architecture
- ✅ **GUI Application**: Windows message loop with D3D11 resource management
- 🔧 **Audio/Video Sync**: Timestamp-based synchronization ready to implement
- 🔧 **Real Video Input**: Replace fake signal with actual video decoder integration

### Testing Architecture Success
- **Memory Reuse Tests**: Verify `frame[n] == frame[n+2]` pointer equality for both NV12 and RGB buffers
- **Double Buffer Validation**: Confirm alternating frame allocation pattern for hardware textures
- **Color Conversion Verification**: Hardware vs software conversion comparison tests
- **Offscreen Rendering Tests**: VideoPlayer works without window dependency using dependency injection
- **Integration Tests**: Real MKV file processing with full hardware acceleration pipeline
- **WSL Cross-Platform**: Windows build tools accessible from Linux environment

### WSL Development Workflow
- ✅ `cmd.exe /c "build.bat test"` works perfectly from WSL
- ✅ Windows filesystem accessible via `/mnt/c/` mount point  
- ✅ All Windows build tools (MSVC, MSBuild) accessible from WSL
- ✅ Hardware acceleration works on Windows host
- 💡 WSL provides excellent cross-platform development workflow

## RGB Video Decoder Implementation

### Architecture Overview
The `RgbVideoDecoder` provides a high-level interface for hardware-accelerated video decoding with automatic color space conversion to RGB for display purposes.

### Key Features
- **Seamless Integration**: Wraps `HwVideoDecoder` and inherits all D3D11 resources
- **Hardware Color Conversion**: Uses D3D11 Video Processor for NV12→BGRA conversion
- **Double Buffering**: Alternates between two RGB texture buffers for optimal performance
- **Resource Management**: Automatic texture creation, view caching, and cleanup
- **Format Consistency**: Always outputs DXGI_FORMAT_B8G8R8A8_UNORM textures

### Usage Pattern
```cpp
RgbVideoDecoder decoder;
decoder.open("video.mkv");

RgbVideoDecoder::DecodedFrame frame;
while (decoder.readNextFrame(frame)) {
    // frame.hw_frame: Original NV12 hardware frame
    // frame.rgb_frame.rgb_texture: Ready-to-render RGB texture
    // frame.rgb_frame.rgb_srv: Shader resource view for rendering
}
```

### Performance Characteristics
- **Decoding Speed**: ~3-4ms per frame for 320x240 video
- **Memory Efficiency**: Zero-copy GPU pipeline, no CPU involvement
- **Resource Reuse**: Texture objects recycled between frames
- **Validation**: Comprehensive tests verify correct double buffering and color accuracy

### Technical Implementation
- **Device Sharing**: Reuses D3D11 device/context from `HwVideoDecoder`
- **Video Processor**: Single instance handles all color space conversions
- **View Caching**: Output views cached per RGB frame to minimize creation overhead
- **Error Handling**: Comprehensive HRESULT checking and fallback mechanisms

## Audio Player Implementation

### Architecture Overview
The `AudioPlayer` provides a Windows-native audio output solution using WASAPI (Windows Audio Session API) with external timer integration for GUI applications.

### Key Features
- **WASAPI Integration**: Hardware-accelerated audio output with low latency
- **External Timer Control**: Time management delegated to GUI message loop
- **Thread-Safe Design**: Separate audio thread for buffer management
- **Format Flexibility**: Automatic audio format detection and conversion
- **State Management**: Play/pause/stop with proper resource cleanup

### Usage Pattern
```cpp
AudioPlayer player;
player.initialize();
player.open("audio.mkv");
player.play();

// In Windows message loop (WM_TIMER):
double current_time = GetCurrentPlaybackTime();
player.onTimer(current_time);  // Updates buffers and time
```

### Message Loop Integration
- **Timer-Driven**: External timer calls `onTimer(current_time)` every 5-10ms
- **Buffer Management**: WASAPI buffer monitoring and automatic refill
- **Decoupled Design**: Audio processing independent of video timing
- **GUI Friendly**: No blocking operations in main thread

### Technical Implementation
- **Buffer Strategy**: Continuous WASAPI buffer filling based on `GetCurrentPadding()`
- **Format Support**: Automatic PCM format conversion for WASAPI compatibility
- **Resource Management**: COM initialization, device enumeration, and cleanup
- **Error Handling**: Comprehensive HRESULT checking and fallback mechanisms

### Performance Characteristics
- **Latency**: Low-latency WASAPI shared mode for responsive audio
- **Memory Efficiency**: Streaming buffer management, no large audio caches
- **CPU Usage**: Efficient buffer checking, minimal CPU overhead
- **Integration**: Seamless Windows message loop integration for GUI apps

## Video Player Implementation

### Architecture Overview
The `VideoPlayer` provides a clean, dependency-injection based video playback solution with timer-driven frame updates and external D3D11 resource management.

### Key Features
- **Dependency Injection**: External D3D11 resources (device, context, render target) passed in
- **Timer-Driven Updates**: `onTimer()` callback driven by external timer (Windows `WM_TIMER`)
- **Frame Rate Conversion**: 24fps fake video signal converted to 60Hz display timing
- **Test-Friendly Design**: Supports both windowed and offscreen rendering modes
- **Clean Separation**: Video logic separated from GUI and resource management

### Usage Pattern
```cpp
// GUI application creates D3D11 resources
ID3D11Device* device = ...;
ID3D11DeviceContext* context = ...;
ID3D11RenderTargetView* rtv = ...;
IDXGISwapChain* swap_chain = ...;  // optional for offscreen

// Initialize VideoPlayer with external resources
VideoPlayer player;
bool success = player.initialize(device, context, rtv, swap_chain);

// Timer-driven updates (called every 16ms from WM_TIMER)
player.onTimer();  // Handles frame generation, rendering, and timing
```

### Design Philosophy
- **External Resource Management**: VideoPlayer does not create or destroy D3D11 resources
- **Single Responsibility**: Focuses only on video playback timing and rendering
- **Testable Architecture**: Supports offscreen rendering for automated testing
- **No Window Dependency**: Works with any D3D11 render target (window, texture, etc.)

### Fake Video Signal
- **Purpose**: Demonstrates frame rate conversion and timing without complex video decoding
- **Pattern**: Red → Green → Blue color rotation at 24fps
- **Conversion**: 24fps source converted to 60Hz display using frame repetition
- **Future**: Ready to be replaced with real video decoder integration

### Technical Implementation
- **Frame Rate Converter**: Handles 24fps → 60Hz timing with accumulated time calculations
- **Resource Safety**: Null pointer checks prevent crashes when uninitialized
- **Performance**: Minimal overhead when properly initialized
- **Testing**: Comprehensive offscreen rendering validation
