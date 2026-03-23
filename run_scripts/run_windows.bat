@echo off
REM SadTalker Run Script for Windows
REM This script starts the SadTalker WebUI

setlocal enabledelayedexpansion

REM Get the project root directory (one level up from this script)
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%.."
set "PROJECT_ROOT=%CD%"

echo ================================================================
echo Starting SadTalker WebUI on Windows
echo ================================================================
echo.
echo Project root: %PROJECT_ROOT%
echo.

REM Check if virtual environment exists
set "VENV_DIR=%PROJECT_ROOT%\venv"

if not exist "%VENV_DIR%" (
    echo ERROR: Virtual environment not found at %VENV_DIR%
    echo Please run the installation script first:
    echo   installation\windows\install.bat
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment activated
echo.

REM Check if models are downloaded
if not exist "%PROJECT_ROOT%\checkpoints" (
    echo WARNING: Checkpoints directory not found
    echo Please download models first. See installation\windows\install.bat for details.
    echo.
    set /p REPLY="Continue anyway? (y/N): "
    if /i not "!REPLY!"=="y" (
        exit /b 1
    )
)

REM Disable analytics
set GRADIO_ANALYTICS_ENABLED=False

echo Launching SadTalker WebUI...
echo The WebUI will be available at: http://localhost:7860
echo.
echo Press Ctrl+C to stop the server
echo.

REM Run the application
python app_sadtalker.py %*

if errorlevel 1 (
    echo.
    echo ERROR: Failed to start SadTalker
    pause
    exit /b 1
)
