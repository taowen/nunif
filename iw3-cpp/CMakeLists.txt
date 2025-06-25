cmake_minimum_required(VERSION 3.20)

project(iw3-cpp 
    VERSION 1.0.0
    DESCRIPTION "IW3 C++ Hello World"
    LANGUAGES CXX)

# 设置 C++ 标准为 C++23 (如果编译器不支持则回退到 C++20)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 设置编译器特定的警告和优化选项
if(MSVC)
    # Visual Studio 编译器选项
    add_compile_options(/W4 /WX- /permissive-)
    # 启用最新 C++ 特性
    add_compile_options(/std:c++latest)
else()
    # GCC/Clang 编译器选项
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 创建可执行文件
add_executable(iw3_cpp src/main.cpp)

# 设置输出目录
set_target_properties(iw3_cpp PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 如果使用 Visual Studio，设置启动项目
if(MSVC)
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT hello_world)
endif()
