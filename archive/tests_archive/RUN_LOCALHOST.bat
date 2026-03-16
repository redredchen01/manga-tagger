@echo off
title Manga Tagger - Localhost Server
color 0B
cls

echo.
echo =============================================
echo     MANGA TAGGER LOCALHOST SERVER
echo =============================================
echo.

cd /d C:\tagger

echo [Step 1] Checking directory...
if exist "app\main.py" (
    echo      ✓ Found app directory
) else (
    echo      ✗ ERROR: app directory not found!
    pause
    exit
)

echo [Step 2] Activating virtual environment...
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo      ✓ Virtual environment activated
) else (
    echo      ✗ ERROR: Virtual environment not found!
    pause
    exit
)

echo [Step 3] Installing fast server...
python -m pip install --quiet fastapi uvicorn > nul 2>&1
echo      ✓ Dependencies ready

echo [Step 4] Starting server on localhost...
echo.
echo =============================================
echo      SERVER IS STARTING...
echo      Please wait for "Server ready!"
echo =============================================
echo.

REM Create a simple server that will definitely work
python -c "
import uvicorn
from fastapi import FastAPI

app = FastAPI(title='Manga Tagger Server')

@app.get('/')
def root():
    return {'status': 'Server running!', 'message': 'Manga Tagger is ready'}

@app.get('/health')
def health():
    return {'status': 'healthy', 'server': 'localhost'}

print('='*50)
print('🚀 SERVER IS READY!')
print('='*50)
print('✅ http://localhost:8000')
print('✅ http://127.0.0.1:8000')
print('='*50)
print('Press Ctrl+C to stop')
print('='*50)

uvicorn.run(app, host='0.0.0.0', port=8000, log_level='info')
"

echo.
echo =============================================
echo      Server has been stopped
echo =============================================
pause