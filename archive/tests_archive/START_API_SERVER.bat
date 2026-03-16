@echo off
SETLOCAL EnableDelayedExpansion
title Manga Tagger API Server
color 0B

echo.
echo ============================================================
echo      Manga Tagger API Server - Startup Script
echo ============================================================
echo.

:: Check for .env file
if not exist ".env" (
    echo [!] .env file not found. Creating default from .env.example...
    copy .env.example .env
    echo [OK] .env created. Please edit it if you need specific settings.
    echo.
)

:: Check for virtual environment
if exist "venv\Scripts\activate.bat" (
    echo [1/3] Activating virtual environment...
    call venv\Scripts\activate.bat
    echo [OK] Virtual environment activated.
) else (
    echo [!] Warning: venv\Scripts\activate.bat not found. 
    echo     Attempting to run with system python...
    echo.
)

:: Check if app directory exists
if not exist "app\main.py" (
    echo [ERROR] app\main.py not found! 
    echo Please make sure you are running this from the project root.
    pause
    exit /b 1
)

echo [2/3] Checking dependencies...
pip show uvicorn >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [!] uvicorn is not installed in the current environment.
    echo     Attempting to install requirements...
    pip install -r requirements.txt
)

echo [3/3] Starting Manga Tagger API Server...
echo.
echo ------------------------------------------------------------
echo API will be available at: http://localhost:8000
echo Documentation (Swagger): http://localhost:8000/docs
echo ------------------------------------------------------------
echo.

:: Start the server using uvicorn
:: We use 0.0.0.0 to allow access from other devices in the same network
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --log-level info

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Server failed to start or crashed.
    pause
)

echo.
echo Server stopped. Press any key to exit...
pause > nul