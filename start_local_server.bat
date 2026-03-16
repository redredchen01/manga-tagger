@echo off
echo Starting Manga Tagger Server...
echo =====================================

cd /d "%~dp0"

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Starting server...
echo.
echo Server will be available at: http://127.0.0.1:8000
echo API Documentation: http://127.0.0.1:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo =====================================

python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

pause