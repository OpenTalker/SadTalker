@echo off
REM SadTalker Installation Script for Windows
REM This script automates the installation process for SadTalker on Windows systems

setlocal enabledelayedexpansion

echo ================================================================
echo SadTalker Installation Script for Windows
echo ================================================================
echo.

REM Get the project root directory (two levels up from this script)
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%..\.."
set "PROJECT_ROOT=%CD%"

echo Project root: %PROJECT_ROOT%
echo.

REM Step 1: Check prerequisites
echo ================================================================
echo Step 1: Checking prerequisites
echo ================================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.8 or higher from https://www.python.org/downloads/windows/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Python %PYTHON_VERSION% found

REM Check git
git --version >nul 2>&1
if errorlevel 1 (
    echo WARNING: Git is not installed or not in PATH.
    echo You can install it from https://git-scm.com/download/win
    echo Or using scoop: scoop install git
) else (
    echo [OK] Git found
)

REM Check ffmpeg
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo WARNING: FFmpeg is not installed or not in PATH.
    echo Please install FFmpeg:
    echo   - Download from: https://www.ffmpeg.org/download.html
    echo   - Or use scoop: scoop install ffmpeg
    echo   - Or use chocolatey: choco install ffmpeg
    echo.
    echo FFmpeg is required for video processing. Installation will continue,
    echo but you need to install FFmpeg before running SadTalker.
    echo.
) else (
    echo [OK] FFmpeg found
)

echo.
pause
echo.

REM Step 2: Create virtual environment
echo ================================================================
echo Step 2: Creating virtual environment
echo ================================================================
echo.

set "VENV_DIR=%PROJECT_ROOT%\venv"

if exist "%VENV_DIR%" (
    echo WARNING: Virtual environment already exists at %VENV_DIR%
    set /p REPLY="Do you want to remove it and create a new one? (y/N): "
    if /i "!REPLY!"=="y" (
        echo Removing existing virtual environment...
        rmdir /s /q "%VENV_DIR%"
        echo Removed existing virtual environment
    ) else (
        echo Skipping virtual environment creation
        goto :activate_venv
    )
)

echo Creating virtual environment...
python -m venv "%VENV_DIR%"
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment created at %VENV_DIR%

:activate_venv
echo Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment activated
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Step 3: Install PyTorch
echo ================================================================
echo Step 3: Installing PyTorch
echo ================================================================
echo.

REM Detect NVIDIA GPU
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo No NVIDIA GPU detected. Installing CPU version of PyTorch...
    set "TORCH_COMMAND=pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1"
) else (
    echo NVIDIA GPU detected. Installing CUDA version of PyTorch...
    set "TORCH_COMMAND=pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113"
)

echo Installing PyTorch...
%TORCH_COMMAND%
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch
    pause
    exit /b 1
)
echo [OK] PyTorch installed
echo.

REM Step 4: Install dependencies
echo ================================================================
echo Step 4: Installing dependencies
echo ================================================================
echo.

if not exist "%PROJECT_ROOT%\requirements.txt" (
    echo ERROR: requirements.txt not found in %PROJECT_ROOT%
    pause
    exit /b 1
)

echo Installing dependencies from requirements.txt...
pip install -r "%PROJECT_ROOT%\requirements.txt"
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo [OK] Dependencies installed
echo.

REM TTS is optional and may not work on all Windows systems
echo Installing TTS (optional, for gradio demo with text-to-speech)...
pip install TTS
if errorlevel 1 (
    echo WARNING: TTS installation failed. This is optional.
    echo You can still use SadTalker without TTS functionality.
)
echo.

REM Step 5: Download models
echo ================================================================
echo Step 5: Downloading models
echo ================================================================
echo.

echo Models need to be downloaded manually on Windows.
echo Please download the models using one of these methods:
echo.
echo Method 1: Using Git Bash (if installed):
echo   Open Git Bash in the project directory and run:
echo   bash scripts/download_models.sh
echo.
echo Method 2: Manual download from:
echo   - Google Drive: https://drive.google.com/file/d/1gwWh45pF7aelNP_P78uDJL8Sycep-K7j/view?usp=sharing
echo   - GitHub Releases: https://github.com/OpenTalker/SadTalker/releases
echo   - Baidu (password: sadt): https://pan.baidu.com/s/1kb1BCPaLOWX1JJb9Czbn6w?pwd=sadt
echo.
echo Extract the models to: %PROJECT_ROOT%\checkpoints\
echo.
set /p SKIP_MODELS="Have you already downloaded the models? (y/N): "
if /i "!SKIP_MODELS!"=="y" (
    echo Skipping model download step
) else (
    echo Please download the models before running SadTalker.
)
echo.

REM Step 6: Installation complete
echo ================================================================
echo Installation Complete!
echo ================================================================
echo.
echo SadTalker has been successfully installed.
echo.
echo To activate the environment in the future, run:
echo   %VENV_DIR%\Scripts\activate.bat
echo.
echo To run SadTalker:
echo   1. WebUI mode: run_scripts\run_windows.bat
echo   2. CLI mode: python inference.py --driven_audio audio.wav --source_image image.png
echo.
echo For more information, see the README.md file.
echo.
echo ================================================================
echo.
pause
