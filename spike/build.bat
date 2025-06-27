mkdir spike\build
pushd spike\build
cmake -S .. -B .
cmake --build . --config Debug
if %ERRORLEVEL% neq 0 (
    echo Build failed! Exiting...
    popd
    exit /b %ERRORLEVEL%
)
popd
spike\build\Debug\spike.exe