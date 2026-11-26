@echo off
cd /d "%~dp0"
if not exist "..\.venv\Scripts\pythonw.exe" (
    echo [ERROR] Python environment ..\.venv not found. Run the install script in the parent folder first.
    pause
    exit /b 1
)
start "" "..\.venv\Scripts\pythonw.exe" "%~dp0main.py"
