@echo off
REM Launch TagGUI, creating a virtual environment and installing dependencies
REM on first run. Pass "update" (or "-u") to force a reinstall of the
REM requirements into an existing environment:  run.bat update
REM
REM The console window is kept open on any failure so the error is readable
REM when the script is started by double-clicking it in Explorer.
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"

set "VENV_DIR=venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
REM Copy of requirements.txt written only after a successful install. Its
REM absence (or difference) means the environment is incomplete, so an install
REM that failed or was interrupted is retried instead of silently launching a
REM broken interpreter.
set "STAMP=%VENV_DIR%\.installed-requirements.txt"

REM Keep the Hugging Face model cache on the C: drive (an SSD). This is the
REM standard location under the user profile; models download here once and
REM are reused. Scoped to this script by setlocal, so the global environment
REM is left unchanged.
set "HF_HOME=%USERPROFILE%\.cache\huggingface"

set "DO_INSTALL=0"
if /i "%~1"=="update" set "DO_INSTALL=1"
if /i "%~1"=="-u" set "DO_INSTALL=1"

if not exist "%VENV_PY%" goto :create_venv
goto :check_install


:create_venv
REM Find a supported interpreter. requirements.txt only ships Windows wheels
REM for torch and flash-attn on CPython 3.11 and 3.12; on any other version
REM pip silently skips them (the environment markers do not match) and the app
REM then fails at "import torch". So pin the version here rather than trusting
REM whatever "python" happens to be first on PATH.
set "PY_CMD="
py -3.12 -V >nul 2>&1
if not errorlevel 1 set "PY_CMD=py -3.12"
if defined PY_CMD goto :have_python

py -3.11 -V >nul 2>&1
if not errorlevel 1 set "PY_CMD=py -3.11"
if defined PY_CMD goto :have_python

REM No py launcher, or no supported version registered with it. Fall back to
REM whatever "python" is on PATH, but only if it reports a supported version.
python -c "import sys; sys.exit(0 if sys.version_info[:2] in ((3,11),(3,12)) else 1)" >nul 2>&1
if not errorlevel 1 set "PY_CMD=python"
if not defined PY_CMD goto :no_python

:have_python

echo Creating virtual environment in "%VENV_DIR%" using %PY_CMD%...
%PY_CMD% -m venv "%VENV_DIR%"
if errorlevel 1 goto :venv_failed
if not exist "%VENV_PY%" goto :venv_failed

echo Upgrading pip...
"%VENV_PY%" -m pip install --upgrade pip
goto :install


:check_install
REM The venv interpreter is a stub that resolves its standard library through
REM the base installation recorded in pyvenv.cfg. If that base Python was
REM uninstalled or upgraded in place, this exe no longer starts at all and no
REM amount of reinstalling will help, so check it before anything else.
"%VENV_PY%" -V >nul 2>&1
if errorlevel 1 goto :venv_broken

REM Reinstall when requirements.txt changed since the last successful install,
REM or when no successful install was ever recorded.
if "%DO_INSTALL%"=="1" goto :install
if not exist "%STAMP%" goto :install
fc /b "%STAMP%" requirements.txt >nul 2>&1
if errorlevel 1 goto :install

REM The stamp says the install completed, but the environment can still be
REM broken (a deleted package, a half-finished upgrade). Verify the two imports
REM the app cannot start without, and repair rather than crash on launch.
"%VENV_PY%" -c "import PySide6, torch" >nul 2>&1
if not errorlevel 1 goto :launch
echo Environment looks incomplete ^(PySide6 or torch failed to import^).
echo Reinstalling dependencies...
goto :install


:install
echo Installing dependencies from requirements.txt ^(this may take a while^)...
echo The PyTorch CUDA wheel is about 2.4 GB, so the first run is slow.
if exist "%STAMP%" del /q "%STAMP%" >nul 2>&1
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 goto :install_failed

REM Confirm the install actually produced a usable environment before recording
REM it. On an unsupported Python version pip exits 0 while skipping torch.
"%VENV_PY%" -c "import PySide6, torch" >nul 2>&1
if errorlevel 1 goto :verify_failed
copy /y requirements.txt "%STAMP%" >nul
goto :launch


:launch
echo Starting TagGUI...
"%VENV_PY%" taggui\run_gui.py
if errorlevel 1 goto :app_failed
endlocal
exit /b 0


:no_python
echo.
echo Could not find a supported Python interpreter.
echo TagGUI needs CPython 3.11 or 3.12 - 3.13 and newer have no PyTorch
echo wheel in requirements.txt and the app will not start.
echo.
echo Install Python 3.12 from https://www.python.org/downloads/ and be sure
echo to tick "Add python.exe to PATH" in the installer.
echo.
"%SystemRoot%\System32\where.exe" python >nul 2>&1
if errorlevel 1 goto :no_python_done
echo For reference, the "python" currently on your PATH reports:
python -V 2>&1
:no_python_done
echo.
pause
exit /b 1


:venv_broken
echo.
echo The virtual environment in "%VENV_DIR%" is unusable - its Python
echo interpreter will not start. This normally means the Python installation
echo it was built from was removed, upgraded, or moved.
echo.
echo Delete the "%VENV_DIR%" folder and run this script again to rebuild it.
echo.
pause
exit /b 1


:venv_failed
echo.
echo Failed to create the virtual environment in "%VENV_DIR%".
echo Delete that folder if it exists and try again.
echo.
pause
exit /b 1


:install_failed
echo.
echo Failed to install dependencies. The most common causes are a dropped
echo network connection during the large PyTorch download, or not enough
echo free disk space ^(the environment needs roughly 8 GB^).
echo.
echo Fix the cause and run "run.bat update" to retry the install.
echo.
pause
exit /b 1


:verify_failed
echo.
echo Dependencies installed but PySide6 or torch still cannot be imported.
echo This usually means the virtual environment is on an unsupported Python
echo version, so pip skipped the platform-specific wheels. This environment is:
"%VENV_PY%" -V
echo.
echo Delete the "%VENV_DIR%" folder and run this script again to rebuild it
echo against Python 3.11 or 3.12.
echo.
pause
exit /b 1


:app_failed
echo.
echo TagGUI exited with an error. The traceback is above.
echo.
pause
exit /b 1
