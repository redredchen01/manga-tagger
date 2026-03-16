@echo off
SETLOCAL EnableDelayedExpansion
title Manga Tagger API Server [Port 80]
color 0D

echo.
echo ============================================================
echo      Manga Tagger API Server - Port 80 (Standard HTTP)
echo ============================================================
echo [IMPORTANT] This script requires Administrator privileges.
echo.

:: Check for virtual environment
if exist "venv\Scripts\activate.bat" (
    echo [1/3] Activating virtual environment...
    call venv\Scripts\activate.bat
    echo [OK] Virtual environment activated.
)

echo [2/3] Starting server on Port 80...
echo.
echo ------------------------------------------------------------
echo API Documentation (Swagger): http://25.4.11.198/docs
echo ReDoc Documentation:         http://25.4.11.198/redoc
echo ------------------------------------------------------------
echo.

:: Run on port 80 and host 0.0.0.0 (allows access from 25.4.11.198)
python -m uvicorn app.main:app --host 0.0.0.0 --port 80 --reload --log-level info

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Failed to start on Port 80. 
    echo Common reasons:
    echo 1. You are not running this as Administrator.
    echo 2. Another program (like IIS, Skype, or Apache) is using Port 80.
    pause
)

echo.
echo Server stopped. Press any key to exit...
pause > nul