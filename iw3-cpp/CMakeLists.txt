cmake_minimum_required(VERSION 3.20)

project(iw3-cpp 
    VERSION 1.0.0
    DESCRIPTION "IW3 C++ Hello World"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/W4 /WX- /permissive-)
    add_compile_options(/std:c++latest)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_executable(iw3_cpp src/main.cpp)

set_target_properties(iw3_cpp PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

if(MSVC)
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT hello_world)
endif()
