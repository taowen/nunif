# FindCUDA.cmake - Find CUDA libraries and setup targets

# Find CUDA installation directory from PATH
find_program(NVCC_EXECUTABLE
    NAMES nvcc
    PATHS ENV PATH
    DOC "NVIDIA CUDA Compiler"
)

if(NVCC_EXECUTABLE)
    get_filename_component(CUDA_BIN_DIR ${NVCC_EXECUTABLE} DIRECTORY)
    get_filename_component(CUDA_ROOT_DIR ${CUDA_BIN_DIR} DIRECTORY)
    
    set(CUDA_INCLUDE_DIR ${CUDA_ROOT_DIR}/include)
    set(CUDA_LIB_DIR ${CUDA_ROOT_DIR}/lib/x64)
    
    message(STATUS "Found CUDA root: ${CUDA_ROOT_DIR}")
    message(STATUS "CUDA include dir: ${CUDA_INCLUDE_DIR}")
    message(STATUS "CUDA lib dir: ${CUDA_LIB_DIR}")
endif()

# Find CUDA libraries
find_library(CUDA_CUDART_LIBRARY
    NAMES cudart
    PATHS ${CUDA_LIB_DIR}
    NO_DEFAULT_PATH
)

find_library(CUDA_CUDA_LIBRARY
    NAMES cuda
    PATHS ${CUDA_LIB_DIR}
    NO_DEFAULT_PATH
)

find_library(CUDA_CUBLAS_LIBRARY
    NAMES cublas
    PATHS ${CUDA_LIB_DIR}
    NO_DEFAULT_PATH
)

find_library(CUDA_CUFFT_LIBRARY
    NAMES cufft
    PATHS ${CUDA_LIB_DIR}
    NO_DEFAULT_PATH
)

find_library(CUDA_CURAND_LIBRARY
    NAMES curand
    PATHS ${CUDA_LIB_DIR}
    NO_DEFAULT_PATH
)

# Check if CUDA was found
if(CUDA_INCLUDE_DIR AND CUDA_CUDART_LIBRARY)
    set(CUDA_FOUND TRUE)
    message(STATUS "Found CUDA:")
    message(STATUS "  Include dir: ${CUDA_INCLUDE_DIR}")
    message(STATUS "  cudart: ${CUDA_CUDART_LIBRARY}")
    if(CUDA_CUDA_LIBRARY)
        message(STATUS "  cuda: ${CUDA_CUDA_LIBRARY}")
    endif()
    if(CUDA_CUBLAS_LIBRARY)
        message(STATUS "  cublas: ${CUDA_CUBLAS_LIBRARY}")
    endif()
    if(CUDA_CUFFT_LIBRARY)
        message(STATUS "  cufft: ${CUDA_CUFFT_LIBRARY}")
    endif()
    if(CUDA_CURAND_LIBRARY)
        message(STATUS "  curand: ${CUDA_CURAND_LIBRARY}")
    endif()
else()
    message(FATAL_ERROR "CUDA not found. Make sure CUDA is installed and available in PATH. Required library: cudart")
endif()

# Function to link CUDA to a target
function(target_link_cuda target_name)
    if(NOT CUDA_FOUND)
        message(FATAL_ERROR "CUDA not found")
    endif()
    
    target_include_directories(${target_name} PRIVATE ${CUDA_INCLUDE_DIR})
    target_link_libraries(${target_name} PRIVATE ${CUDA_CUDART_LIBRARY})
    
    if(CUDA_CUDA_LIBRARY)
        target_link_libraries(${target_name} PRIVATE ${CUDA_CUDA_LIBRARY})
    endif()
    
    if(CUDA_CUBLAS_LIBRARY)
        target_link_libraries(${target_name} PRIVATE ${CUDA_CUBLAS_LIBRARY})
    endif()
    
    if(CUDA_CUFFT_LIBRARY)
        target_link_libraries(${target_name} PRIVATE ${CUDA_CUFFT_LIBRARY})
    endif()
    
    if(CUDA_CURAND_LIBRARY)
        target_link_libraries(${target_name} PRIVATE ${CUDA_CURAND_LIBRARY})
    endif()
endfunction()

# Function to copy CUDA DLLs to target directory
function(copy_cuda_dlls target_name)
    if(NOT CUDA_FOUND OR NOT CUDA_BIN_DIR)
        return()
    endif()
    
    # Common CUDA DLLs
    set(CUDA_DLL_NAMES
        cudart64_*.dll
        cuda64_*.dll
        cublas64_*.dll
        cublasLt64_*.dll
        cufft64_*.dll
        curand64_*.dll
    )
    
    foreach(dll_pattern ${CUDA_DLL_NAMES})
        file(GLOB CUDA_DLLS "${CUDA_BIN_DIR}/${dll_pattern}")
        if(CUDA_DLLS)
            add_custom_command(TARGET ${target_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${CUDA_DLLS}
                $<TARGET_FILE_DIR:${target_name}>
                COMMENT "Copying CUDA DLLs to output directory"
            )
        endif()
    endforeach()
    
    message(STATUS "Will copy CUDA DLLs from: ${CUDA_BIN_DIR}")
endfunction() 