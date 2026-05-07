@echo off
echo ================================================
echo   Put Option Anomaly Scanner
echo ================================================

cd /d "%~dp0"

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

:: Install / upgrade dependencies
echo.
echo [1/2] Installing dependencies...
pip install -r requirements.txt --quiet --upgrade

:: Launch app
echo.
echo [2/2] Starting server at http://localhost:5000
echo       Press Ctrl+C to stop.
echo.
start "" http://localhost:5000
python app.py

pause
