@echo off
setlocal
cd /d "%~dp0"

echo Checking Python 3.11...

py -3.11 -c "import sys; assert sys.maxsize > 2**32, '64-bit Python is required'"
if errorlevel 1 (
    echo.
    echo Python 3.11 64-bit or Python Launcher was not found.
    echo Please install Python 3.11 64-bit, then run this file again.
    pause
    exit /b 1
)

if not exist "requirements-lock-py311.txt" (
    echo Missing requirements-lock-py311.txt
    pause
    exit /b 1
)

if not exist "wheels\" (
    echo Missing wheels folder.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating local virtual environment...
    py -3.11 -m venv ".venv"
    if errorlevel 1 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
)

echo Installing dependencies from local wheels...

".venv\Scripts\python.exe" -m pip install --no-index --find-links="%~dp0wheels" -r "%~dp0requirements-lock-py311.txt"
if errorlevel 1 (
    echo Dependency installation failed.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m pip check
if errorlevel 1 (
    echo Dependency consistency check failed.
    pause
    exit /b 1
)

echo Checking Python syntax...

".venv\Scripts\python.exe" -m py_compile app.py ocr_engine.py local_recognizer.py check_install.py
if errorlevel 1 (
    echo Source code syntax check failed.
    pause
    exit /b 1
)

echo Checking the OCR model...

".venv\Scripts\python.exe" check_install.py
if errorlevel 1 (
    echo Model check failed.
    pause
    exit /b 1
)

echo.
echo Installation completed successfully.
echo Double-click start.bat to launch the application.
pause
endlocal