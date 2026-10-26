@echo off
setlocal
cd /d "%~dp0"

echo ==========================================
echo Guji OCR Uninstaller
echo ==========================================
echo.
echo Target folder:
echo "%CD%"
echo.
echo Stop the application before continuing.
echo Closing the browser alone does NOT stop it.
echo Use Ctrl+C in the start.bat window.
echo.
echo This script does NOT uninstall system Python.
echo.
echo 1. Remove application, KEEP the data folder
echo 2. Remove EVERYTHING, including all user data
echo 3. Cancel
echo.

choice /c 123 /n /m "Select [1/2/3]: "
if errorlevel 3 exit /b 0
if errorlevel 2 goto full
if errorlevel 1 goto keep

:keep
echo.
echo User data will remain in:
echo "%CD%\data"
echo.
choice /c YN /n /m "Continue uninstalling? [Y/N]: "
if errorlevel 2 exit /b 0
set "REMOVE_DATA=0"
goto remove

:full
echo.
echo WARNING: This deletes documents, OCR results,
echo manual transcriptions, and the database.
echo Back up the entire data folder outside this folder first.
echo.
set "CONFIRM="
set /p "CONFIRM=Type DELETE to confirm: "
if not "%CONFIRM%"=="DELETE" (
    echo Cancelled.
    pause
    exit /b 0
)
set "REMOVE_DATA=1"

:remove
echo.
echo Removing application files...

rem Remove directories, preserving data if requested.
for /d %%D in (*) do (
    if /i "%%D"=="data" (
        if "%REMOVE_DATA%"=="1" rmdir /s /q "%%D"
    ) else (
        rmdir /s /q "%%D"
    )
)

rem Keep this script until the cleanup is finished.
for %%F in (*) do (
    if /i not "%%F"=="uninstall.bat" del /f /q "%%F"
)

echo.
echo Cleanup attempted.
echo If access-denied or file-in-use errors appeared,
echo stop the application and run this script again.
echo.
if "%REMOVE_DATA%"=="0" (
    echo The data folder has been preserved.
)
echo You may delete this uninstall.bat manually.
echo If no data is needed, delete the remaining project folder.
pause
endlocal