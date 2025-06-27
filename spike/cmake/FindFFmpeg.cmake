# FindFFmpeg.cmake - Find FFmpeg libraries and setup targets

# Find FFmpeg installation directory from PATH
find_program(FFMPEG_EXECUTABLE
    NAMES ffmpeg
    PATHS ENV PATH
    DOC "FFmpeg executable"
)

if(FFMPEG_EXECUTABLE)
    get_filename_component(FFMPEG_BIN_DIR ${FFMPEG_EXECUTABLE} DIRECTORY)
    get_filename_component(FFMPEG_ROOT_DIR ${FFMPEG_BIN_DIR} DIRECTORY)
    
    set(FFMPEG_INCLUDE_DIR ${FFMPEG_ROOT_DIR}/include)
    set(FFMPEG_LIB_DIR ${FFMPEG_ROOT_DIR}/lib)
    
    message(STATUS "Found FFmpeg root: ${FFMPEG_ROOT_DIR}")
    message(STATUS "FFmpeg include dir: ${FFMPEG_INCLUDE_DIR}")
    message(STATUS "FFmpeg lib dir: ${FFMPEG_LIB_DIR}")
endif()

# Find FFmpeg libraries
find_library(FFMPEG_AVCODEC_LIBRARY
    NAMES avcodec
    PATHS ${FFMPEG_LIB_DIR}
    NO_DEFAULT_PATH
)

find_library(FFMPEG_AVFORMAT_LIBRARY
    NAMES avformat
    PATHS ${FFMPEG_LIB_DIR}
    NO_DEFAULT_PATH
)

find_library(FFMPEG_AVUTIL_LIBRARY
    NAMES avutil
    PATHS ${FFMPEG_LIB_DIR}
    NO_DEFAULT_PATH
)

find_library(FFMPEG_SWSCALE_LIBRARY
    NAMES swscale
    PATHS ${FFMPEG_LIB_DIR}
    NO_DEFAULT_PATH
)

find_library(FFMPEG_SWRESAMPLE_LIBRARY
    NAMES swresample
    PATHS ${FFMPEG_LIB_DIR}
    NO_DEFAULT_PATH
)

# Check if FFmpeg was found
if(FFMPEG_INCLUDE_DIR AND FFMPEG_AVCODEC_LIBRARY AND FFMPEG_AVFORMAT_LIBRARY AND FFMPEG_AVUTIL_LIBRARY)
    set(FFMPEG_FOUND TRUE)
    message(STATUS "Found FFmpeg:")
    message(STATUS "  Include dir: ${FFMPEG_INCLUDE_DIR}")
    message(STATUS "  avcodec: ${FFMPEG_AVCODEC_LIBRARY}")
    message(STATUS "  avformat: ${FFMPEG_AVFORMAT_LIBRARY}")
    message(STATUS "  avutil: ${FFMPEG_AVUTIL_LIBRARY}")
    if(FFMPEG_SWSCALE_LIBRARY)
        message(STATUS "  swscale: ${FFMPEG_SWSCALE_LIBRARY}")
    endif()
    if(FFMPEG_SWRESAMPLE_LIBRARY)
        message(STATUS "  swresample: ${FFMPEG_SWRESAMPLE_LIBRARY}")
    endif()
else()
    message(FATAL_ERROR "FFmpeg not found. Make sure FFmpeg is installed and available in PATH. Required libraries: avcodec, avformat, avutil")
endif()

# Function to link FFmpeg to a target
function(target_link_ffmpeg target_name)
    if(NOT FFMPEG_FOUND)
        message(FATAL_ERROR "FFmpeg not found")
    endif()
    
    target_include_directories(${target_name} PRIVATE ${FFMPEG_INCLUDE_DIR})
    target_link_libraries(${target_name} PRIVATE 
        ${FFMPEG_AVCODEC_LIBRARY}
        ${FFMPEG_AVFORMAT_LIBRARY}
        ${FFMPEG_AVUTIL_LIBRARY}
    )
    
    if(FFMPEG_SWSCALE_LIBRARY)
        target_link_libraries(${target_name} PRIVATE ${FFMPEG_SWSCALE_LIBRARY})
    endif()
    
    if(FFMPEG_SWRESAMPLE_LIBRARY)
        target_link_libraries(${target_name} PRIVATE ${FFMPEG_SWRESAMPLE_LIBRARY})
    endif()
endfunction()

# Function to copy FFmpeg DLLs to target directory
function(copy_ffmpeg_dlls target_name)
    if(NOT FFMPEG_FOUND OR NOT FFMPEG_BIN_DIR)
        return()
    endif()
    
    file(GLOB FFMPEG_DLLS "${FFMPEG_BIN_DIR}/*.dll")
    if(FFMPEG_DLLS)
        add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${FFMPEG_DLLS}
            $<TARGET_FILE_DIR:${target_name}>
            COMMENT "Copying FFmpeg DLLs to output directory"
        )
        message(STATUS "Will copy FFmpeg DLLs from: ${FFMPEG_BIN_DIR}")
    endif()
endfunction() 