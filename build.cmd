@echo off
setlocal EnableDelayedExpansion

rem Color codes
set COLOR_RED=[31m
set COLOR_GREEN=[32m
set COLOR_CYAN=[36m
set COLOR_BOLD=[1m
set COLOR_RESET=[0m

rem Parse debug argument
if not [%1]==[debug] ^
if not [%1]==[release] ^
goto :help

set DEBUG=0
if [%1]==[debug] ^
set DEBUG=1

rem Parse force argument
if not [%2]==[] ^
if not [%2]==[-f] ^
goto :help

set FORCE=0
if [%2]==[-f] ^
set FORCE=1

rem Determine game version
set GAME_VERSION=debug
if %DEBUG%==0 (
for /f "tokens=3 delims= " %%a in ('findstr /i /c:"#define GAME_VERSION" src\headers\defines.hpp') do set GAME_VERSION_STRING=%%a
set GAME_VERSION=!GAME_VERSION_STRING:~2,-1!

if exist dist\!GAME_VERSION!\ ^
if %FORCE%==0 ^
echo %COLOR_BOLD%%COLOR_RED%Output directory ^"!GAME_VERSION!^" already exists, use '-f' flag to force build, aborting build.%COLOR_RESET% && exit /b 0
)
echo %COLOR_BOLD%%COLOR_CYAN%Building to output directory "%GAME_VERSION%"%COLOR_RESET%

rem Clear output directory
if exist dist\%GAME_VERSION%\ rmdir /s /q dist\%GAME_VERSION%\
mkdir dist\%GAME_VERSION%\

rem Clear build directory
if exist out\ rmdir /s /q out\
mkdir out\

rem Build shaders

rem Copy assets
xcopy /s /q /v assets\ dist\%GAME_VERSION%\

rem Build source
set DEBUG_ARGS=
if %DEBUG%==1 set DEBUG_ARGS=/DDEBUG /D_DEBUG /MDd
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if ERRORLEVEL 1 echo %COLOR_BOLD%%COLOR_RED%Variable initialization failed, aborting build.%COLOR_RESET% && exit /b 0
nvcc -c -std=c++20 -O2 -Xcompiler "/W4 /EHsc /utf-8 /DUNICODE /D_UNICODE %DEBUG_ARGS%" -o out\ src\renderer.cu
if ERRORLEVEL 1 echo %COLOR_BOLD%%COLOR_RED%NVCC compilation failed, aborting build.%COLOR_RESET% && exit /b 0
cl /nologo /c /std:c++20 /O2 /W4 /EHsc /utf-8 /DUNICODE /D_UNICODE %DEBUG_ARGS% /Fo:out\ src\*.cpp
if ERRORLEVEL 1 echo %COLOR_BOLD%%COLOR_RED%CL compilation failed, aborting build.%COLOR_RESET% && exit /b 0
link /nologo /subsystem:windows /out:dist\%GAME_VERSION%\main.exe out\*.obj user32.lib gdi32.lib cudart.lib "/libpath:C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\lib\x64"
if ERRORLEVEL 1 echo %COLOR_BOLD%%COLOR_RED%Linking failed, aborting build.%COLOR_RESET% && exit /b 0
echo %COLOR_BOLD%%COLOR_GREEN%Build successful.%COLOR_RESET%

rem Start game
if %DEBUG%==1 ^
echo %COLOR_BOLD%%COLOR_CYAN%Starting game...%COLOR_RESET% && start dist\%GAME_VERSION%\main.exe

exit /b 0

:help
echo %COLOR_BOLD%%COLOR_RED%Usage: build.cmd debug^|release [-f]%COLOR_RESET%
exit /b 1
