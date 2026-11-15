@echo off
setlocal
cd /d "%~dp0" || exit /b 1
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m pipetwin.web --port 0 --open
) else (
    python -m pipetwin.web --port 0 --open
)
if errorlevel 1 pause
