@echo off
setlocal

REM Uvicorn startup wrapper for Windows Task Scheduler deployments.
REM This script assumes it lives in the application root directory.

set "APP_ROOT=%~dp0"
if "%APP_ROOT:~-1%"=="\" set "APP_ROOT=%APP_ROOT:~0,-1%"

set "VENV_PY=%APP_ROOT%\.venv\Scripts\python.exe"
set "LOG_DIR=%APP_ROOT%\logs"
set "LOG_FILE=%LOG_DIR%\uvicorn-startup.log"

REM Optional temp isolation for scheduled-task deployments.
REM Uncomment these lines if the deployment account needs app-local temp space.
REM set "TEMP=%APP_ROOT%\TEMP"
REM set "TMP=%APP_ROOT%\TEMP"
REM if not exist "%TEMP%" mkdir "%TEMP%"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo ==== %date% %time% starting uvicorn ====>> "%LOG_FILE%"
echo APP_ROOT=%APP_ROOT%>> "%LOG_FILE%"
echo VENV_PY=%VENV_PY%>> "%LOG_FILE%"

if not exist "%VENV_PY%" (
    echo ERROR: Python interpreter not found at "%VENV_PY%" >> "%LOG_FILE%"
    endlocal & exit /b 1
)

cd /d "%APP_ROOT%"

"%VENV_PY%" -m uvicorn main:app --host 127.0.0.1 --port 8101 --app-dir src\webapp >> "%LOG_FILE%" 2>&1

set "EXIT_CODE=%ERRORLEVEL%"
echo ==== %date% %time% uvicorn exited with errorlevel %EXIT_CODE% ====>> "%LOG_FILE%"

endlocal & exit /b %EXIT_CODE%
