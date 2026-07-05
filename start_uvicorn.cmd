@echo off
setlocal

set "APP_ROOT=C:\webapps\StructureRelations"
set "VENV_PY=%APP_ROOT%\.venv\Scripts\python.exe"
set "LOG_DIR=%APP_ROOT%\logs"
set "LOG_FILE=%LOG_DIR%\uvicorn-startup.log"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

cd /d "%APP_ROOT%"

echo ==== %date% %time% starting uvicorn ====>> "%LOG_FILE%"
echo APP_ROOT=%APP_ROOT%>> "%LOG_FILE%"
echo VENV_PY=%VENV_PY%>> "%LOG_FILE%"

"%VENV_PY%" -m uvicorn main:app --host 127.0.0.1 --port 8101 --app-dir src\webapp >> "%LOG_FILE%" 2>&1

echo ==== %date% %time% uvicorn exited with errorlevel %errorlevel% ====>> "%LOG_FILE%"

endlocal
