@echo off
title Manga Tagger - Full System
color 0E
cls

echo.
echo ===============================================
echo     MANGA TAGGER - FULL SYSTEM
echo     (API + Frontend)
echo ===============================================
echo.

cd /d C:\tagger

echo [Step 1] Starting API Backend Server...
echo      Location: http://localhost:8000
start "API Server" cmd /k "call venv\Scripts\activate.bat && python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --log-level info"

echo.
echo [Step 2] Waiting for API to start...
timeout /t 5 >nul

echo [Step 3] Starting Streamlit Frontend...
echo      Location: http://localhost:8501
start "Frontend" cmd /k "call venv\Scripts\activate.bat && streamlit run streamlit_app.py --server.port 8501"

echo.
echo ===============================================
echo      SYSTEMS ARE STARTING!
echo ===============================================
echo.
echo      API Backend:  http://localhost:8000
echo      API Docs:     http://localhost:8000/docs
echo      Frontend UI:   http://localhost:8501
echo.
echo      Please wait a moment for full startup...
echo      Both services are opening in new windows
echo ===============================================
echo.

echo Opening frontend in browser...
timeout /t 3 >nul
start http://localhost:8501

echo.
echo Press any key to exit this launcher...
pause > nul