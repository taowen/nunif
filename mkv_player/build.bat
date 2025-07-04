@echo off
setlocal

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

:: 根据参数决定构建目标
if "%TARGET%"=="test" (
    echo Building test target...
    cmake --build . --config Debug --target spike
    if %ERRORLEVEL% neq 0 (
        echo Build failed! Exiting...
        popd
        exit /b %ERRORLEVEL%
    )
    echo.
    echo Running tests...
    popd
    build\Debug\spike.exe
) else if "%TARGET%"=="cli" (
    echo Building CLI player...
    cmake --build . --config Debug --target cli_player
    if %ERRORLEVEL% neq 0 (
        echo Build failed! Exiting...
        popd
        exit /b %ERRORLEVEL%
    )
    popd
    echo.
    echo CLI player built successfully!
    echo Run: build\Debug\cli_player.exe
) else if "%TARGET%"=="gui" (
    echo Building GUI player...
    cmake --build . --config Debug --target gui_player
    if %ERRORLEVEL% neq 0 (
        echo Build failed! Exiting...
        popd
        exit /b %ERRORLEVEL%
    )
    popd
    echo.
    echo GUI player built successfully!
    echo Run: build\Debug\gui_player.exe
) else if "%TARGET%"=="all" (
    echo Building all targets...
    cmake --build . --config Debug
    if %ERRORLEVEL% neq 0 (
        echo Build failed! Exiting...
        popd
        exit /b %ERRORLEVEL%
    )
    popd
    echo.
    echo All targets built successfully!
    echo Running tests...
    build\Debug\spike.exe
) else (
    echo Unknown target: %TARGET%
    echo.
    echo Usage: build.bat [target]
    echo Available targets:
    echo   test  - Build and run tests (default)
    echo   cli   - Build CLI player only
    echo   gui   - Build GUI player only
    echo   all   - Build all targets
    popd
    exit /b 1
)