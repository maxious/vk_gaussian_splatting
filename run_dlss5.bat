@echo off
setlocal

set "VIEWER=%~dp0_bin\Debug\vk_viewer.exe"
if not exist "%VIEWER%" (
  echo Viewer not found: "%VIEWER%"
  echo Build the Debug configuration first.
  exit /b 1
)

if not exist "%~dp0_bin\Debug\dlss5-bridge.addon64" if not exist "%~dp0dlss5-bridge.addon64" (
  echo WARNING: dlss5-bridge.addon64 was not found.
  echo Place it in the ReShade AddonPath or beside the viewer executable.
)
if not exist "%~dp0_bin\Debug\renodx-dlss5.addon64" if not exist "%~dp0renodx-dlss5.addon64" (
  echo WARNING: renodx-dlss5.addon64 was not found.
)
if not exist "%~dp0_bin\Debug\nvngx_dlssnr.dll" if not exist "%~dp0nvngx_dlssnr.dll" (
  echo WARNING: nvngx_dlssnr.dll was not found.
)

"%VIEWER%" --pipeline 2 --dlss 1 %*
exit /b %errorlevel%
