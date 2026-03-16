@echo off
SETLOCAL EnableDelayedExpansion
title Manga Tagger API Server [Universal Access]
color 0A

echo.
echo ============================================================
echo      Manga Tagger API Server - Startup Script
echo ============================================================
echo.

:: Detect IPs for user convenience
echo [INFO] Current detected IP addresses on this machine:
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr "IPv4"') do (
    set ip=%%a
    set ip=!ip: =!
    echo    - http://!ip!:8000
)
echo.

:: Check for virtual environment
if exist "venv\Scripts\activate.bat" (
    echo [1/3] Activating virtual environment...
    call venv\Scripts\activate.bat
    echo [OK] Virtual environment activated.
) else (
    echo [!] Warning: venv\Scripts\activate.bat not found.
)

:: Check if app directory exists
if not exist "app\main.py" (
    echo [ERROR] app\main.py not found! 
    echo Please make sure you are running this from the project root.
    pause
    exit /b 1
)

echo [2/3] Starting Manga Tagger API Server on ALL interfaces...
echo.
echo ------------------------------------------------------------
echo API is now starting. You can try accessing it via:
echo 1. Local:   http://localhost:8000
echo 2. Hamachi: http://25.8.89.95:8000 (Based on your ipconfig)
echo ------------------------------------------------------------
echo.

:: Bind to 0.0.0.0 to allow any IP (including Hamachi) to connect
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --log-level info

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Server failed to start.
    pause
)

echo.
echo Server stopped. Press any key to exit...
pause > nul