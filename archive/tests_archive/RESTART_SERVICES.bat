@echo off
title Manga Tagger - Automated Restart
color 0B
cls

echo.
echo ===============================================
echo     MANGA TAGGER - AUTOMATED RESTART
echo ===============================================
echo.

cd /d %~dp0

echo [Step 1] Killing existing processes...
".\venv\Scripts\python.exe" kill_ports.py 8000 8501

echo.
echo [Step 2] Starting Backend API...
start "Manga Tagger Backend" cmd /k ".\venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

echo.
echo [Step 3] Waiting for Backend to initialize...
timeout /t 5 > nul

echo.
echo [Step 4] Starting Frontend UI...
start "Manga Tagger Frontend" cmd /k ".\venv\Scripts\python.exe" -m streamlit run streamlit_app.py --server.port 8501

echo.
echo ===============================================
echo     RESTART COMPLETE
echo ===============================================
echo.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:8501
echo.
pause
