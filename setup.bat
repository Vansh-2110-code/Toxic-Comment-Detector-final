@echo off
echo =====================================
echo Toxic Comments Classifier - Setup
echo =====================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://www.python.org/
    pause
    exit /b 1
)

echo [1/3] Installing Python dependencies...
echo.
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo =====================================
echo Installation Complete!
echo =====================================
echo.
echo Next steps:
echo   1. Run start_backend.bat to start the server
echo   2. Run start_frontend.bat to open the web app
echo.
pause
