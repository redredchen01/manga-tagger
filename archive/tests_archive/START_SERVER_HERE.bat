@echo off
color 0B
title Manga Tagger Server
echo.
echo =========================================
echo    MANGA TAGGER SERVER LAUNCHER
echo =========================================
echo.

REM Check if directory exists
if not exist "C:\tagger" (
    echo ERROR: C:\tagger directory not found!
    pause
    exit /b 1
)

cd /d C:\tagger

echo [Step 1] Activating Python virtual environment...
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo [SUCCESS] Virtual environment activated
) else (
    echo ERROR: Virtual environment not found!
    echo Please run: python -m venv venv
    pause
    exit /b 1
)

echo.
echo [Step 2] Installing required packages...
python -m pip install fastapi uvicorn > nul 2>&1

echo.
echo [Step 3] Starting server...
echo.
echo ======================================
echo  SERVER URLS:
echo  - http://127.0.0.1:8000
echo  - http://localhost:8000  
echo  - http://0.0.0.0:8000
echo  - API Docs: http://127.0.0.1:8000/docs
echo ======================================
echo.
echo Press Ctrl+C to stop the server
echo.

python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info

echo.
echo Server stopped. Press any key to exit...
pause > nul