# FindTensorRT.cmake - Find TensorRT libraries and setup targets

# Try to find TensorRT from PATH first
find_program(TENSORRT_DUMMY_EXECUTABLE
    NAMES trtexec
    PATHS ENV PATH
    DOC "TensorRT executable for path detection"
)

if(TENSORRT_DUMMY_EXECUTABLE)
    get_filename_component(TENSORRT_BIN_DIR ${TENSORRT_DUMMY_EXECUTABLE} DIRECTORY)
    get_filename_component(TENSORRT_ROOT_DIR ${TENSORRT_BIN_DIR} DIRECTORY)
    
    set(TENSORRT_INCLUDE_DIR ${TENSORRT_ROOT_DIR}/include)
    set(TENSORRT_LIB_DIR ${TENSORRT_ROOT_DIR}/lib)
    
    message(STATUS "Found TensorRT root: ${TENSORRT_ROOT_DIR}")
    message(STATUS "TensorRT include dir: ${TENSORRT_INCLUDE_DIR}")
    message(STATUS "TensorRT lib dir: ${TENSORRT_LIB_DIR}")
endif()

# If not found via PATH, try default installation paths
if(NOT TENSORRT_ROOT_DIR)
    find_path(TENSORRT_ROOT_DIR
        NAMES include/NvInfer.h
        PATHS
            "C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT"
            "C:/Program Files/NVIDIA/TensorRT"
            "/usr/local/TensorRT"
            "/opt/TensorRT"
        DOC "TensorRT root directory"
    )
    
    if(TENSORRT_ROOT_DIR)
        set(TENSORRT_INCLUDE_DIR ${TENSORRT_ROOT_DIR}/include)
        set(TENSORRT_LIB_DIR ${TENSORRT_ROOT_DIR}/lib)
        set(TENSORRT_BIN_DIR ${TENSORRT_ROOT_DIR}/bin)
        
        message(STATUS "Found TensorRT root: ${TENSORRT_ROOT_DIR}")
        message(STATUS "TensorRT include dir: ${TENSORRT_INCLUDE_DIR}")
        message(STATUS "TensorRT lib dir: ${TENSORRT_LIB_DIR}")
    endif()
endif()

# Find TensorRT libraries (version 10 only)
find_library(TENSORRT_NVINFER_LIBRARY
    NAMES nvinfer_10
    PATHS ${TENSORRT_LIB_DIR}
    NO_DEFAULT_PATH
)

find_library(TENSORRT_NVONNXPARSER_LIBRARY
    NAMES nvonnxparser_10
    PATHS ${TENSORRT_LIB_DIR}
    NO_DEFAULT_PATH
)

find_library(TENSORRT_NVINFER_PLUGIN_LIBRARY
    NAMES nvinfer_plugin_10
    PATHS ${TENSORRT_LIB_DIR}
    NO_DEFAULT_PATH
)

find_library(TENSORRT_NVPARSERS_LIBRARY
    NAMES nvparsers_10
    PATHS ${TENSORRT_LIB_DIR}
    NO_DEFAULT_PATH
)

# Additional libraries that might be available
find_library(TENSORRT_NVINFER_LEAN_LIBRARY
    NAMES nvinfer_lean_10
    PATHS ${TENSORRT_LIB_DIR}
    NO_DEFAULT_PATH
)

find_library(TENSORRT_NVINFER_DISPATCH_LIBRARY
    NAMES nvinfer_dispatch_10
    PATHS ${TENSORRT_LIB_DIR}
    NO_DEFAULT_PATH
)

# Check if TensorRT was found
if(TENSORRT_INCLUDE_DIR AND TENSORRT_NVINFER_LIBRARY)
    set(TENSORRT_FOUND TRUE)
    message(STATUS "Found TensorRT 10:")
    message(STATUS "  Include dir: ${TENSORRT_INCLUDE_DIR}")
    message(STATUS "  nvinfer: ${TENSORRT_NVINFER_LIBRARY}")
    if(TENSORRT_NVONNXPARSER_LIBRARY)
        message(STATUS "  nvonnxparser: ${TENSORRT_NVONNXPARSER_LIBRARY}")
    endif()
    if(TENSORRT_NVINFER_PLUGIN_LIBRARY)
        message(STATUS "  nvinfer_plugin: ${TENSORRT_NVINFER_PLUGIN_LIBRARY}")
    endif()
    if(TENSORRT_NVPARSERS_LIBRARY)
        message(STATUS "  nvparsers: ${TENSORRT_NVPARSERS_LIBRARY}")
    endif()
    if(TENSORRT_NVINFER_LEAN_LIBRARY)
        message(STATUS "  nvinfer_lean: ${TENSORRT_NVINFER_LEAN_LIBRARY}")
    endif()
    if(TENSORRT_NVINFER_DISPATCH_LIBRARY)
        message(STATUS "  nvinfer_dispatch: ${TENSORRT_NVINFER_DISPATCH_LIBRARY}")
    endif()
else()
    message(FATAL_ERROR "TensorRT 10 not found. Make sure TensorRT 10 is installed and available in PATH or at standard locations. Required library: nvinfer_10")
endif()

# Function to link TensorRT to a target
function(target_link_tensorrt target_name)
    if(NOT TENSORRT_FOUND)
        message(FATAL_ERROR "TensorRT not found")
    endif()
    
    target_include_directories(${target_name} PRIVATE ${TENSORRT_INCLUDE_DIR})
    target_link_libraries(${target_name} PRIVATE ${TENSORRT_NVINFER_LIBRARY})
    
    if(TENSORRT_NVONNXPARSER_LIBRARY)
        target_link_libraries(${target_name} PRIVATE ${TENSORRT_NVONNXPARSER_LIBRARY})
    endif()
    
    if(TENSORRT_NVINFER_PLUGIN_LIBRARY)
        target_link_libraries(${target_name} PRIVATE ${TENSORRT_NVINFER_PLUGIN_LIBRARY})
    endif()
    
    if(TENSORRT_NVPARSERS_LIBRARY)
        target_link_libraries(${target_name} PRIVATE ${TENSORRT_NVPARSERS_LIBRARY})
    endif()
    
    if(TENSORRT_NVINFER_LEAN_LIBRARY)
        target_link_libraries(${target_name} PRIVATE ${TENSORRT_NVINFER_LEAN_LIBRARY})
    endif()
    
    if(TENSORRT_NVINFER_DISPATCH_LIBRARY)
        target_link_libraries(${target_name} PRIVATE ${TENSORRT_NVINFER_DISPATCH_LIBRARY})
    endif()
endfunction()

# Function to copy TensorRT DLLs to target directory
function(copy_tensorrt_dlls target_name)
    if(NOT TENSORRT_FOUND OR NOT TENSORRT_LIB_DIR)
        return()
    endif()
    
    # TensorRT DLLs are in the lib directory, not bin directory
    file(GLOB TENSORRT_DLLS "${TENSORRT_LIB_DIR}/*.dll")
    if(TENSORRT_DLLS)
        add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TENSORRT_DLLS}
            $<TARGET_FILE_DIR:${target_name}>
            COMMENT "Copying TensorRT DLLs to output directory"
        )
        message(STATUS "Will copy TensorRT DLLs from: ${TENSORRT_LIB_DIR}")
    endif()
endfunction() 