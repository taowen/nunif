cmake_minimum_required(VERSION 3.20)

project(iw3-cpp 
    VERSION 1.0.0
    DESCRIPTION "IW3 C++ Hello World"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Windows/MSVC compile options
add_compile_options(/W4 /WX- /permissive-)
add_compile_options(/std:c++latest)

# Add cmake directory to module path
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

add_executable(iw3_cpp src/main.cpp)

# Set Visual Studio startup project
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT iw3_cpp)

# Link FFmpeg to target
include(FindFFmpeg)
target_link_ffmpeg(iw3_cpp)
copy_ffmpeg_dlls(iw3_cpp)
