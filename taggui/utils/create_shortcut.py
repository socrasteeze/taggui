"""Create a Windows .lnk shortcut for TagGUI (run.bat + app icon)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from utils.utils import get_resource_path

ICON_PATH = Path('images/icon.ico')


def get_repo_root() -> Path:
    return Path(get_resource_path(Path('.'))).resolve()


def get_run_bat_path() -> Path:
    return get_repo_root() / 'run.bat'


def get_icon_path() -> Path:
    return get_resource_path(ICON_PATH)


def get_desktop_path() -> Path:
    return Path(os.path.expanduser('~')) / 'Desktop'


def get_start_menu_path() -> Path:
    programs = Path(os.environ.get(
        'APPDATA', Path.home() / 'AppData' / 'Roaming')) / (
            'Microsoft' / 'Windows' / 'Start Menu' / 'Programs')
    return programs


def create_windows_shortcut(
        destination: Path,
        target: Path | None = None,
        working_directory: Path | None = None,
        icon_path: Path | None = None,
        description: str = 'TagGUI — image tagging / captioning',
        name: str = 'TagGUI.lnk') -> Path:
    """
    Create a Windows .lnk via WScript.Shell (no extra Python packages).
    Returns the path of the created shortcut.
    """
    if sys.platform != 'win32':
        raise OSError('Shortcuts (.lnk) can only be created on Windows.')

    target = (target or get_run_bat_path()).resolve()
    working_directory = (working_directory or get_repo_root()).resolve()
    icon_path = (icon_path or get_icon_path()).resolve()
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    shortcut_path = destination / name

    if not target.is_file():
        raise FileNotFoundError(f'Launch target not found: {target}')

    icon_location = str(icon_path) if icon_path.is_file() else str(target)

    # Escape single quotes for the PowerShell single-quoted string literals.
    def ps_quote(value: str) -> str:
        return value.replace("'", "''")

    script = f"""
$ErrorActionPreference = 'Stop'
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut('{ps_quote(str(shortcut_path))}')
$shortcut.TargetPath = '{ps_quote(str(target))}'
$shortcut.WorkingDirectory = '{ps_quote(str(working_directory))}'
$shortcut.IconLocation = '{ps_quote(icon_location)},0'
$shortcut.Description = '{ps_quote(description)}'
$shortcut.Save()
"""
    completed = subprocess.run(
        ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass',
         '-Command', script],
        capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or '').strip()
        raise RuntimeError(detail or 'PowerShell failed to create the shortcut.')
    if not shortcut_path.is_file():
        raise RuntimeError(f'Shortcut was not created at {shortcut_path}.')
    return shortcut_path


def create_taggui_shortcuts(*, desktop: bool = True,
                            start_menu: bool = False) -> list[Path]:
    created: list[Path] = []
    if desktop:
        created.append(create_windows_shortcut(get_desktop_path()))
    if start_menu:
        created.append(create_windows_shortcut(get_start_menu_path()))
    return created
