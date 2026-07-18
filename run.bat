@echo off
REM Launch TagGUI, creating a virtual environment and installing dependencies
REM on first run. Pass "update" (or "-u") to reinstall requirements into an
REM existing environment, e.g. after requirements.txt changes:  run.bat update
setlocal
cd /d "%~dp0"

set "VENV_DIR=venv"
set "PYTHON=python"
set "FRESH=0"

if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Creating virtual environment in "%VENV_DIR%"...
    %PYTHON% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo.
        echo Failed to create the virtual environment.
        echo Make sure Python 3.11 or 3.12 is installed and on your PATH.
        pause
        exit /b 1
    )
    set "FRESH=1"
)

call "%VENV_DIR%\Scripts\activate.bat"

set "DO_INSTALL=0"
if "%FRESH%"=="1" set "DO_INSTALL=1"
if /i "%~1"=="update" set "DO_INSTALL=1"
if /i "%~1"=="-u" set "DO_INSTALL=1"

if "%FRESH%"=="1" (
    echo Upgrading pip...
    python -m pip install --upgrade pip
)

if "%DO_INSTALL%"=="1" (
    echo Installing dependencies from requirements.txt ^(this may take a while^)...
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo Failed to install dependencies.
        pause
        exit /b 1
    )
)

echo Starting TagGUI...
python taggui\run_gui.py

endlocal
