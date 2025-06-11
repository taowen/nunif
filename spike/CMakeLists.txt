cmake_minimum_required(VERSION 3.18)

# Set the project name and enable CUDA
project(spike LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA package (modern way)
find_package(CUDAToolkit REQUIRED)

# Try to find cuDNN in common locations
set(CUDNN_ROOT_DIR "${CUDAToolkit_ROOT}" CACHE PATH "Folder containing cuDNN")
find_path(CUDNN_INCLUDE_DIR cudnn.h
    HINTS ${CUDNN_ROOT_DIR} ${CUDAToolkit_ROOT}
    PATH_SUFFIXES include
)
find_library(CUDNN_LIBRARY cudnn
    HINTS ${CUDNN_ROOT_DIR} ${CUDAToolkit_ROOT}
    PATH_SUFFIXES lib lib64 lib/x64
)

# Check if cuDNN was found
if(NOT CUDNN_INCLUDE_DIR OR NOT CUDNN_LIBRARY)
    message(WARNING "cuDNN not found. Please install cuDNN or set CUDNN_ROOT_DIR")
    message(STATUS "CUDNN_INCLUDE_DIR: ${CUDNN_INCLUDE_DIR}")
    message(STATUS "CUDNN_LIBRARY: ${CUDNN_LIBRARY}")
    message(STATUS "CUDAToolkit_ROOT: ${CUDAToolkit_ROOT}")
endif()

# Add executable
add_executable(spike src/main.cpp)

# Set target properties for CUDA
set_target_properties(spike PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Include directories
target_include_directories(spike PRIVATE 
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDNN_INCLUDE_DIR}
)

# Link libraries
target_link_libraries(spike 
    CUDA::cudart
    CUDA::cuda_driver
)

# Link cuDNN if found
if(CUDNN_LIBRARY)
    target_link_libraries(spike ${CUDNN_LIBRARY})
    target_compile_definitions(spike PRIVATE CUDNN_AVAILABLE)
endif()

# Set CUDA architectures (adjust based on your GPU)
set_property(TARGET spike PROPERTY CUDA_ARCHITECTURES 75 80 86)

# For Windows-specific settings
if(WIN32)
    # Set runtime library to match CUDA requirements
    set_property(TARGET spike PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()
