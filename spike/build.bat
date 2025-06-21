pushd spike\build
cmake -S .. -B .
cmake --build . --config Debug
if %ERRORLEVEL% neq 0 (
    echo Build failed! Exiting...
    popd
    exit /b %ERRORLEVEL%
)
popd
spike\build\Debug\spike.exe stereo_module_half_sbs.trt "C:\Users\taowen\Downloads\06 4k.mp4" "C:\Users\taowen\Downloads\test.mkv" 2