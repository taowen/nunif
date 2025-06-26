cmake_minimum_required(VERSION 3.20)

project(iw3-cpp 
    VERSION 1.0.0
    DESCRIPTION "IW3 C++ Hello World"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add cmake directory to module path
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

add_executable(iw3_cpp src/main.cpp src/decode_thread.cpp src/convert_color_thread.cpp src/infer_sbs.cpp src/video_file_opener.cpp src/encode_thread.cpp src/diagnose_convert_color_thread.cpp)

# Apply Windows/MSVC compile options
target_compile_options(iw3_cpp PRIVATE
    /W4 /WX- /permissive- /std:c++latest
)

# Set Visual Studio startup project
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT iw3_cpp)

# Add DirectXTex library
set(DIRECTXTEX_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../directxtex_desktop_win10.2025.3.25.2)
target_include_directories(iw3_cpp PRIVATE ${DIRECTXTEX_ROOT}/include)

target_link_libraries(iw3_cpp PRIVATE
    ${DIRECTXTEX_ROOT}/native/lib/x64/Debug/DirectXTex.lib
)

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

# Link CUDA to target (runtime only, no nvcc compilation)
include(FindCUDA)
target_link_cuda(iw3_cpp)
copy_cuda_dlls(iw3_cpp)

# Link TensorRT to target
include(FindTensorRT)
target_link_tensorrt(iw3_cpp)
copy_tensorrt_dlls(iw3_cpp)

