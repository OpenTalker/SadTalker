@echo off

IF NOT EXIST venv (
python -m venv venv
) ELSE (
echo venv folder already exists, skipping creation...
)
call .\venv\Scripts\activate.bat

set PYTHON="venv\Scripts\Python.exe"
echo venv %PYTHON%

%PYTHON% Launcher.py

echo.
echo Launch unsuccessful. Exiting.
pause