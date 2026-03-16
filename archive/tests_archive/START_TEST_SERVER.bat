@echo off
title Localhost Test Server
color 0A
cls

echo.
echo =====================================
echo     LOCALHOST TEST SERVER
echo =====================================
echo.

cd /d C:\tagger

echo Starting simple test server...
echo.

python simple_server.py

echo.
echo Server stopped.
pause