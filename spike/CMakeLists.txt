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
find_path(CUDNN_INCLUDE_DIR cudnn.h
    HINTS ${CUDAToolkit_INCLUDE_DIRS}
)
find_library(CUDNN_LIBRARY cudnn
    HINTS ${CUDAToolkit_LIBRARY_DIR}
)

# Check if cuDNN was found
if(NOT CUDNN_INCLUDE_DIR OR NOT CUDNN_LIBRARY)
    message(WARNING "cuDNN not found. Please install cuDNN or set CUDAToolkit_ROOT")
    message(STATUS "CUDNN_INCLUDE_DIR: ${CUDNN_INCLUDE_DIR}")
    message(STATUS "CUDNN_LIBRARY: ${CUDNN_LIBRARY}")
    message(STATUS "CUDAToolkit_ROOT: ${CUDAToolkit_ROOT}")
endif()

# Create a proper TensorRT find module
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Try to find TensorRT using a more comprehensive approach
set(TENSORRT_ROOT_DIR "" CACHE PATH "Folder containing TensorRT")

# Common TensorRT installation paths
set(TENSORRT_POSSIBLE_PATHS
    ${TENSORRT_ROOT_DIR}
    $ENV{TENSORRT_ROOT}
    $ENV{TensorRT_ROOT}
    "C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT"
    "C:/TensorRT"
    "/usr/local/TensorRT"
    "/opt/TensorRT"
    ${CUDAToolkit_ROOT}
)

# Find TensorRT include directory
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_POSSIBLE_PATHS}
    PATH_SUFFIXES include
    DOC "TensorRT include directory"
)

# Debug: Print what we found and where we're looking
message(STATUS "Debug: TENSORRT_INCLUDE_DIR = ${TENSORRT_INCLUDE_DIR}")
if(TENSORRT_INCLUDE_DIR)
    get_filename_component(TENSORRT_ROOT_FROM_INCLUDE ${TENSORRT_INCLUDE_DIR} DIRECTORY)
    message(STATUS "Debug: Derived TensorRT root from include: ${TENSORRT_ROOT_FROM_INCLUDE}")
    
    # Check what's actually in the lib directory
    file(GLOB LIB_FILES "${TENSORRT_ROOT_FROM_INCLUDE}/lib/*")
    message(STATUS "Debug: Files in lib directory: ${LIB_FILES}")
    
    file(GLOB LIB64_FILES "${TENSORRT_ROOT_FROM_INCLUDE}/lib64/*")
    message(STATUS "Debug: Files in lib64 directory: ${LIB64_FILES}")
    
    file(GLOB LIBX64_FILES "${TENSORRT_ROOT_FROM_INCLUDE}/lib/x64/*")
    message(STATUS "Debug: Files in lib/x64 directory: ${LIBX64_FILES}")
endif()

# Find TensorRT libraries
find_library(TENSORRT_LIBRARY
    NAMES nvinfer_10 nvinfer_10.lib nvinfer nvinfer.lib
    HINTS ${TENSORRT_POSSIBLE_PATHS} ${TENSORRT_ROOT_FROM_INCLUDE}
    PATH_SUFFIXES lib lib64 lib/x64 lib/Win32 lib/x86_64-linux-gnu
    DOC "TensorRT nvinfer library"
)

find_library(TENSORRT_PLUGIN_LIBRARY
    NAMES nvinfer_plugin_10 nvinfer_plugin_10.lib nvinfer_plugin nvinfer_plugin.lib
    HINTS ${TENSORRT_POSSIBLE_PATHS} ${TENSORRT_ROOT_FROM_INCLUDE}
    PATH_SUFFIXES lib lib64 lib/x64 lib/Win32 lib/x86_64-linux-gnu
    DOC "TensorRT nvinfer_plugin library"
)

find_library(TENSORRT_PARSERS_LIBRARY
    NAMES nvonnxparser_10 nvonnxparser_10.lib nvonnxparser nvonnxparser.lib nvparsers nvparsers.lib
    HINTS ${TENSORRT_POSSIBLE_PATHS} ${TENSORRT_ROOT_FROM_INCLUDE}
    PATH_SUFFIXES lib lib64 lib/x64 lib/Win32 lib/x86_64-linux-gnu
    DOC "TensorRT parsers library"
)

# Create TensorRT target if found
if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY)
    add_library(TensorRT::TensorRT UNKNOWN IMPORTED)
    set_target_properties(TensorRT::TensorRT PROPERTIES
        IMPORTED_LOCATION ${TENSORRT_LIBRARY}
        INTERFACE_INCLUDE_DIRECTORIES ${TENSORRT_INCLUDE_DIR}
    )
    
    # Add plugin library if found
    if(TENSORRT_PLUGIN_LIBRARY)
        add_library(TensorRT::Plugin UNKNOWN IMPORTED)
        set_target_properties(TensorRT::Plugin PROPERTIES
            IMPORTED_LOCATION ${TENSORRT_PLUGIN_LIBRARY}
            INTERFACE_INCLUDE_DIRECTORIES ${TENSORRT_INCLUDE_DIR}
        )
    endif()
    
    # Add parsers library if found
    if(TENSORRT_PARSERS_LIBRARY)
        add_library(TensorRT::Parsers UNKNOWN IMPORTED)
        set_target_properties(TensorRT::Parsers PROPERTIES
            IMPORTED_LOCATION ${TENSORRT_PARSERS_LIBRARY}
            INTERFACE_INCLUDE_DIRECTORIES ${TENSORRT_INCLUDE_DIR}
        )
    endif()
    
    set(TensorRT_FOUND TRUE)
    message(STATUS "Found TensorRT: ${TENSORRT_LIBRARY}")
    message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIR}")
else()
    set(TensorRT_FOUND FALSE)
    message(WARNING "TensorRT not found. Please install TensorRT or set TENSORRT_ROOT_DIR")
    message(STATUS "TENSORRT_INCLUDE_DIR: ${TENSORRT_INCLUDE_DIR}")
    message(STATUS "TENSORRT_LIBRARY: ${TENSORRT_LIBRARY}")
    message(STATUS "Searched paths: ${TENSORRT_POSSIBLE_PATHS}")
endif()

# Instead of using pkg-config for FFmpeg, manually find the libraries
if(WIN32)
    # Set FFMPEG_ROOT to point to your FFmpeg installation
    set(FFMPEG_ROOT "C:/games/ffmpeg-7.1.1/install" CACHE PATH "FFmpeg installation directory")
    
    # Find FFmpeg include directory
    find_path(FFMPEG_INCLUDE_DIR libavcodec/avcodec.h
        HINTS ${FFMPEG_ROOT}/include
        REQUIRED
    )
    
    # Find FFmpeg libraries
    find_library(AVCODEC_LIBRARY avcodec
        HINTS ${FFMPEG_ROOT}/bin
        REQUIRED
    )
    find_library(AVFORMAT_LIBRARY avformat
        HINTS ${FFMPEG_ROOT}/bin
        REQUIRED
    )
    find_library(AVUTIL_LIBRARY avutil
        HINTS ${FFMPEG_ROOT}/bin
        REQUIRED
    )
    find_library(SWSCALE_LIBRARY swscale
        HINTS ${FFMPEG_ROOT}/bin
        REQUIRED
    )
else()
    # Use pkg-config on non-Windows platforms
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(FFMPEG REQUIRED libavformat libavcodec libswscale libavutil)
endif()

# Add executable
add_executable(spike src/main.cpp)

# Add FFmpeg libraries
if(WIN32)
    target_link_libraries(spike
        ${AVCODEC_LIBRARY}
        ${AVFORMAT_LIBRARY}
        ${AVUTIL_LIBRARY}
        ${SWSCALE_LIBRARY}
    )
else()
    target_link_libraries(spike ${FFMPEG_LIBRARIES})
endif()

# Set target properties for CUDA
set_target_properties(spike PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Include directories
target_include_directories(spike PRIVATE 
    ${CUDAToolkit_INCLUDE_DIRS}
    ${FFMPEG_ROOT}/include
)

# Add cuDNN if available
if(CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
    target_include_directories(spike PRIVATE ${CUDNN_INCLUDE_DIR})
    target_link_libraries(spike ${CUDNN_LIBRARY})
    target_compile_definitions(spike PRIVATE CUDNN_AVAILABLE)
endif()

# Link libraries
target_link_libraries(spike 
    CUDA::cudart
    CUDA::cuda_driver
    ${FFMPEG_LIBRARIES}
)

# Add TensorRT if available
if(TensorRT_FOUND)
    target_link_libraries(spike TensorRT::TensorRT)
    target_compile_definitions(spike PRIVATE TENSORRT_AVAILABLE)
    
    # Link additional TensorRT libraries if available
    if(TARGET TensorRT::Plugin)
        target_link_libraries(spike TensorRT::Plugin)
    endif()
    
    if(TARGET TensorRT::Parsers)
        target_link_libraries(spike TensorRT::Parsers)
    endif()
endif()

# Set CUDA architectures (adjust based on your GPU)
set_property(TARGET spike PROPERTY CUDA_ARCHITECTURES 75 80 86)

# For Windows-specific settings
if(WIN32)
    # Set runtime library to match CUDA requirements
    set_property(TARGET spike PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()
