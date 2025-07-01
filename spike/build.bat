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
:: Run with verbose output to see stdout
spike\build\Debug\spike.exe "C:\Users\taowen\Downloads\iw3\test.mkv"