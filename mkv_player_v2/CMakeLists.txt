cmake_minimum_required(VERSION 3.20)

project(spike 
    VERSION 1.0.0
    DESCRIPTION "IW3 C++ Hello World"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Fetch Catch2 for testing
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.4.0  # 或者使用最新的稳定版本
    GIT_SHALLOW    TRUE
    UPDATE_DISCONNECTED TRUE
)
FetchContent_MakeAvailable(Catch2)

add_executable(spike 
    src/mkv_stream_reader.cpp
    tests/test_mkv_stream_reader.cpp
    src/packet_demuxer.cpp
    tests/test_packet_demuxer.cpp
    src/hw_frame_decoder.cpp
    tests/test_frame_decoder.cpp
    src/rgb_frame_decoder.cpp
    tests/test_rgb_frame_decoder.cpp
    src/async_rgb_frame_decoder.cpp
    tests/test_async_rgb_frame_decoder.cpp
    src/rgb_verification.cpp
    tests/test_rgb_verification.cpp
    tests/test_gui_player.cpp
)

# Link Catch2 to the executable
target_link_libraries(spike PRIVATE Catch2::Catch2WithMain)

# Apply Windows/MSVC compile options
target_compile_options(spike PRIVATE
    /W4 /WX- /permissive- /std:c++latest
)

# Set Visual Studio startup project
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT spike)

# Link DirectX libraries (Windows only)
target_link_libraries(spike PRIVATE
    d3d11.lib
    dxgi.lib
    d3dcompiler.lib
)

# Link FFmpeg to target
include(cmake/FindFFmpeg.cmake)
target_link_ffmpeg(spike)
copy_ffmpeg_dlls(spike)

# Link CUDA to target (runtime only, no nvcc compilation)
include(cmake/FindCUDA.cmake)
target_link_cuda(spike)
copy_cuda_dlls(spike)

# Link TensorRT to target
include(cmake/FindTensorRT.cmake)
target_link_tensorrt(spike)
copy_tensorrt_dlls(spike)

# GUI Player target
add_executable(gui_player
    src/gui_player.cpp
    src/mkv_stream_reader.cpp
    src/packet_demuxer.cpp
    src/hw_frame_decoder.cpp
    src/rgb_frame_decoder.cpp
    src/async_rgb_frame_decoder.cpp
    src/rgb_verification.cpp
)

# Apply Windows/MSVC compile options for gui_player
target_compile_options(gui_player PRIVATE
    /W4 /WX- /permissive- /std:c++latest
)

# Set subsystem to Console for gui_player to enable stdout
set_target_properties(gui_player PROPERTIES
    LINK_FLAGS "/SUBSYSTEM:CONSOLE"
)

# Link DirectX libraries for gui_player (Windows only)
target_link_libraries(gui_player PRIVATE
    d3d11.lib
    dxgi.lib
    d3dcompiler.lib
    user32.lib
    gdi32.lib
)

# Link FFmpeg to gui_player
target_link_ffmpeg(gui_player)
copy_ffmpeg_dlls(gui_player)

# Link CUDA to gui_player
target_link_cuda(gui_player)
copy_cuda_dlls(gui_player)

# Link TensorRT to gui_player
target_link_tensorrt(gui_player)
copy_tensorrt_dlls(gui_player)