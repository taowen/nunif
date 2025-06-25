mkdir iw3-cpp\build
pushd iw3-cpp\build
cmake -S .. -B .
cmake --build . --config Debug
if %ERRORLEVEL% neq 0 (
    echo Build failed! Exiting...
    popd
    exit /b %ERRORLEVEL%
)
popd
iw3-cpp\build\Debug\iw3_cpp.exe "06 4k.mp4"