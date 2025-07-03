mkdir build
pushd build
cmake -S .. -B .
cmake --build . --config Debug
if %ERRORLEVEL% neq 0 (
    echo Build failed! Exiting...
    popd
    exit /b %ERRORLEVEL%
)
popd
:: Run with verbose output to see stdout
build\Debug\spike.exe