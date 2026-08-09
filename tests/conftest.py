"""
Shared test fixtures.

The Qt-backed models need a `QApplication`, but no display: everything runs on
the `offscreen` platform plugin so the suite works in CI containers. Qt's test
mode redirects `QStandardPaths` and `QSettings` into a scratch directory, so
tests never read or write the developer's real TagGUI settings, dimension
cache or downloaded tag lists.
"""
import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest
from PySide6.QtCore import QCoreApplication, QSettings, QStandardPaths
from PySide6.QtWidgets import QApplication

from utils.image import Image


@pytest.fixture(scope='session', autouse=True)
def qt_application():
    QStandardPaths.setTestModeEnabled(True)
    QCoreApplication.setOrganizationName('taggui-tests')
    QCoreApplication.setApplicationName('taggui-tests')
    application = QApplication.instance() or QApplication([])
    yield application


@pytest.fixture(autouse=True)
def clean_settings(qt_application):
    """Start every test from default settings."""
    settings = QSettings('taggui', 'taggui')
    settings.clear()
    settings.sync()
    yield settings
    settings.clear()
    settings.sync()


@pytest.fixture
def make_image(tmp_path):
    """Create an `Image` backed by a real (empty) file under `tmp_path`."""
    def factory(name: str, tags: list[str] | None = None,
                dimensions: tuple[int, int] | None = (1024, 1024)) -> Image:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return Image(path, dimensions, list(tags or []))
    return factory
