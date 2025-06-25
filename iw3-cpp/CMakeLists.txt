cmake_minimum_required(VERSION 3.20)

project(iw3-cpp 
    VERSION 1.0.0
    DESCRIPTION "IW3 C++ Hello World"
    LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA settings
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)  # Modern GPU architectures
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Add cmake directory to module path
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

add_executable(iw3_cpp src/main.cu)

# Apply Windows/MSVC compile options only to C++ files, not CUDA
target_compile_options(iw3_cpp PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/W4 /WX- /permissive- /std:c++latest>
)

# Set CUDA properties for the target
set_target_properties(iw3_cpp PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Set Visual Studio startup project
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT iw3_cpp)

# Link DirectX libraries (Windows only)
target_link_libraries(iw3_cpp PRIVATE
    d3d11.lib
    dxgi.lib
    d3dcompiler.lib
)

# Link FFmpeg to target
include(FindFFmpeg)
target_link_ffmpeg(iw3_cpp)
copy_ffmpeg_dlls(iw3_cpp)

# Link CUDA to target
include(FindCUDA)
target_link_cuda(iw3_cpp)
copy_cuda_dlls(iw3_cpp)

# Link TensorRT to target
include(FindTensorRT)
target_link_tensorrt(iw3_cpp)
copy_tensorrt_dlls(iw3_cpp)
