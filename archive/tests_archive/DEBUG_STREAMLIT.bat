@echo off
title Manga Tagger - Streamlit DEBUG
color 0E
cls

echo.
echo ===============================================
echo     MANGA TAGGER - STREAMLIT DEBUG MODE
echo ===============================================
echo.

cd /d %~dp0

echo [Step 1] Checking API status...
".\venv\Scripts\python.exe" -c "import requests; print('API is OK' if requests.get('http://127.0.0.1:8000/health').status_code==200 else 'API ERROR')"

echo.
echo [Step 2] Starting Streamlit with DEBUG logging...
echo.
".\venv\Scripts\python.exe" -m streamlit run streamlit_app.py --server.port 8501 --logger.level=debug

pause
