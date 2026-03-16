@echo off
title MANGA TAGGER SERVER
color 0A
cls

echo.
echo ===============================================
echo       MANGA TAGGER LOCAL SERVER
echo ===============================================
echo.

REM Check if we're in the right directory
if not exist "app\main.py" (
    echo ERROR: Please run this from the C:\tagger directory
    echo.
    pause
    exit /b 1
)

echo [1/4] Checking Python environment...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    pause
    exit /b 1
)
echo      Python found

echo [2/4] Activating virtual environment...
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo      Virtual environment activated
) else (
    echo ERROR: Virtual environment not found!
    pause
    exit /b 1
)

echo [3/4] Installing dependencies...
python -m pip install --quiet fastapi uvicorn python-multipart >nul 2>&1
echo      Dependencies ready

echo [4/4] Starting server...
echo.
echo ===============================================
echo  SERVER IS RUNNING!
echo ===============================================
echo.
echo  Access URLs:
echo    - Main:        http://localhost:8000
echo    - Alternative:  http://127.0.0.1:8000
echo    - API Docs:    http://localhost:8000/docs
echo    - Health:      http://localhost:8000/health
echo.
echo  Press CTRL+C to stop the server
echo ===============================================
echo.

python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info --reload

echo.
echo Server has been stopped.
pause