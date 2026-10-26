@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Local environment was not found.
    echo Please run install.bat first.
    pause
    exit /b 1
)

echo Starting Guji OCR...
echo.
echo Open this address in your browser:
echo http://127.0.0.1:8501
echo.
echo Keep this window open while using the application.
echo Press Ctrl+C to stop the server.
echo.

".venv\Scripts\python.exe" -m streamlit run "%~dp0app.py" ^
    --server.address 127.0.0.1 ^
    --server.port 8501 ^
    --browser.gatherUsageStats false

if errorlevel 1 (
    echo.
    echo Application startup failed. See the error above.
    pause
)

endlocal