@echo off
setlocal EnableDelayedExpansion

:: 检查命令行参数
set TARGET=%1
if "%TARGET%"=="" set TARGET=test

echo ========================================
echo MKV Player Build Script
echo ========================================

:: 创建build目录
if not exist build mkdir build
pushd build

:: 配置项目
echo Configuring with CMake...
cmake -S .. -B .
if %ERRORLEVEL% neq 0 (
    echo CMake configuration failed! Exiting...
    popd
    exit /b %ERRORLEVEL%
)

:: 根据参数设置构建目标和描述
set BUILD_TARGET=
set BUILD_DESC=
set RUN_TESTS=0

if "%TARGET%"=="test" (
    set BUILD_TARGET=spike
    set BUILD_DESC=test target
    set RUN_TESTS=1
    goto :build
) else if "%TARGET%"=="gui" (
    set BUILD_TARGET=gui_player
    set BUILD_DESC=gui target
    goto :build
)

echo Unknown target: %TARGET%
echo.
echo Usage: build.bat [target]
echo Available targets:
echo   test  - Build and run tests (default)
popd
exit /b 1

:build
:: 执行构建
echo Building !BUILD_DESC!...
if "!BUILD_TARGET!"=="" (
    cmake --build . --config Debug
) else (
    cmake --build . --config Debug --target !BUILD_TARGET!
)

if %ERRORLEVEL% neq 0 (
    echo Build failed! Exiting...
    popd
    exit /b %ERRORLEVEL%
)

popd
echo.

:: 根据目标执行后续操作
if "!RUN_TESTS!"=="1" (
    echo Running tests...
    build\Debug\spike.exe
) else if "%TARGET%"=="gui" (
    build\Debug\gui_player.exe
)