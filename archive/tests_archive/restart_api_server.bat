@echo off
chcp 65001 >nul
echo ================================================
echo    Manga Tagger API Server 重啟腳本
echo ================================================
echo.

echo [1/3] 正在終止舊的 API Server 进程...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *uvicorn*" 2>nul
timeout /t 2 /nobreak >nul

echo [2/3] 檢查 port 8000 是否已釋放...
netstat -ano | findstr :8000 >nul
if %errorlevel% equ 0 (
    echo 警告: Port 8000 仍被佔用
    echo 請手動關閉佔用該 port 的程式
    echo.
    echo 佔用 port 8000 的進程:
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000') do (
        echo   PID: %%a
    )
    echo.
    choice /C YN /M "是否嘗試強制終止這些進程？需要管理員權限"
    if %errorlevel% equ 1 (
        for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000') do (
            taskkill /F /PID %%a 2>nul
        )
        timeout /t 2 /nobreak >nul
    )
)

echo [3/3] 啟動新的 API Server...
start "Manga Tagger API" python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

echo.
echo ================================================
echo API Server 已啟動在 http://127.0.0.1:8000
echo ================================================
echo.
echo 按任意鍵關閉此視窗...
pause >nul
