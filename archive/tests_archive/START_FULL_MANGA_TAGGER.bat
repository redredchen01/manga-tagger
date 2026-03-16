@echo off
title Full Manga Tagger Server
color 0C
cls

echo.
echo ==============================================
echo    MANGA TAGGER - FULL SERVER
echo ==============================================
echo.

cd /d C:\tagger

echo [Step 1] Activating virtual environment...
call venv\Scripts\activate.bat
echo      ✓ Virtual environment ready

echo [Step 2] Starting full Manga Tagger server...
echo      This includes:
echo        - 611 tags loaded
echo        - Image tagging API
echo        - Mock AI services
echo        - Full documentation
echo.

echo ==============================================
echo      STARTING FULL SERVER...
echo ==============================================
echo.

python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info

echo.
echo ==============================================
echo      Full server stopped
echo ==============================================
pause