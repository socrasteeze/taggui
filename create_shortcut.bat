@echo off
REM Create a Desktop shortcut (.lnk) for TagGUI that launches run.bat
REM with the app icon from images\icon.ico.
setlocal EnableExtensions
cd /d "%~dp0"

set "VENV_PY=venv\Scripts\python.exe"
if exist "%VENV_PY%" (
  "%VENV_PY%" -c "import sys; sys.path.insert(0, 'taggui'); from utils.create_shortcut import create_taggui_shortcuts; paths = create_taggui_shortcuts(desktop=True); print('Created:'); [print(' ', p) for p in paths]"
) else (
  python -c "import sys; sys.path.insert(0, 'taggui'); from utils.create_shortcut import create_taggui_shortcuts; paths = create_taggui_shortcuts(desktop=True); print('Created:'); [print(' ', p) for p in paths]"
)
if errorlevel 1 (
  echo.
  echo Failed to create the shortcut.
  pause
  exit /b 1
)
echo.
echo Done. Look for "TagGUI" on your Desktop.
pause
exit /b 0
