@echo off
title Manga Tagger Server
color 0A
echo.
echo ========================================
echo    Manga Tagger Server Starting...
echo ========================================
echo.

cd /d C:\tagger
call venv\Scripts\activate.bat

echo.
echo [1/3] Virtual environment activated
echo [2/3] Starting server...
echo.

python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --log-level info

echo.
echo Server stopped. Press any key to exit...
pause > nul