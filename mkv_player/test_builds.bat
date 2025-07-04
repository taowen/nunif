@echo off
echo Testing all build targets...
echo.

echo Testing CLI build...
call build.bat cli
if %ERRORLEVEL% neq 0 (
    echo CLI build failed!
    exit /b 1
)
echo CLI build: OK
echo.

echo Testing GUI build...
call build.bat gui  
if %ERRORLEVEL% neq 0 (
    echo GUI build failed!
    exit /b 1
)
echo GUI build: OK
echo.

echo Testing test build...
call build.bat test | findstr /C:"passed" /C:"failed" /C:"PASSED" /C:"FAILED"
if %ERRORLEVEL% neq 0 (
    echo Test build failed!
    exit /b 1
)
echo Test build: OK
echo.

echo All builds completed successfully!