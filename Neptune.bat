@echo off
setlocal
title NEPTUNE INTELLIGENCE SYSTEM
cd /d %~dp0

echo.
echo    _   _   U _____ u  ____  _____  _   _   _   _  U _____ u 
echo   ^| \ ^| ^|  \^| ___" ^|/  _ ^"\^|_ " _^| ^| \ ^| ^| ^| ^| ^| ^|  \^| ___" ^| 
echo  ^<^|  \^| ^|   ^|  _^|" ^| / " ^|  ^| ^|   ^<^|  \^| ^|}^| ^| ^| ^|   ^|  _^|"  
echo   ^| ^|\  ^|   ^| ^|___ ^| {_.  ^|/^| ^|    ^| ^|\  ^| ^| ^|_^| ^|   ^| ^|___  
echo   ^|_^| \_^|   ^|_____^| \____/  ^|_^|    ^|_^| \_^|  \____/    ^|_____^| 
echo   ^|^|^|_^|^|^|_  ^|^|^|^|^|^|  ^|^|^|_   _^|^|^|_  ^|^|^|_^|^|^|_   ^|^|^|_     ^|^|^|^|^|^|  
echo   (_" ) (") ^|^|^|^|^|^| (") (") (") ("") (_" ) (") (") (")   ^|^|^|^|^|^|  
echo.
echo ================================================================
echo   NEPTUNE DASHBOARD LOADER // GLOBAL INTELLIGENCE NETWORK
echo ================================================================
echo.

:: Check if setup has been run
if not exist backend\venv (
    echo [WARNING] Environment not detected. Running initial setup first...
    call setup.bat
)

:: Clean up any existing hung processes on port 8000
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000') do taskkill /f /pid %%a > nul 2>&1

echo [1/2] Initiating Geopolitical Risk Inference Engine...
cd backend
start /b "" venv\Scripts\python.exe -m uvicorn main:app --port 8000 --no-access-log > nul 2>&1

:: Wait for server port to bind
:wait_loop
timeout /t 2 /nobreak > nul
netstat -ano | findstr :8000 > nul
if %errorlevel% neq 0 (
    echo ... establishing secure handshake ...
    goto wait_loop
)

:: Extra wait for models to fully load (LSTM + CSV take a few seconds after port binds)
echo ... loading ML models and heatmap data ...
timeout /t 6 /nobreak > nul

echo [2/2] Launching Intelligence Dashboard...
start http://localhost:8000

echo.
echo ================================================================
echo   SYSTEM STATUS: ONLINE
echo   DASHBOARD ACCESS: http://localhost:8000
echo.
echo   [!] KEEP THIS WINDOW OPEN WHILE USING NEPTUNE
echo   [!] PRESS ANY KEY TO TERMINATE THE ENTIRE NETWORK
echo ================================================================
echo.

pause > nul

echo Shutting down Geopolitical Risk Hubs...
for /f "tokens=5" %%a in ('netstat -aon | findstr :8000') do taskkill /f /pid %%a > nul 2>&1
echo System Offline.
timeout /t 2 > nul
exit
