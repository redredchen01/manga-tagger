@echo off
title Manga Tagger Launch
color 0B

echo ========================================
echo    MANGA TAGGER - WEB INTERFACE
echo ========================================
echo.

cd /d C:\tagger
call venv\Scripts\activate.bat

echo.
echo Starting API server on port 8000...
start "API Server" cmd /k "python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --log-level info"

echo Waiting 5 seconds for API to start...
timeout /t 5 >nul 2>&1

echo.
echo Starting Streamlit frontend on port 8501...
echo.

echo ========================================
echo      SYSTEM READY!
echo ========================================
echo      API:   http://localhost:8000
echo      Frontend: http://localhost:8501
echo ========================================
echo.

streamlit run streamlit_app.py --server.port 8501

pause