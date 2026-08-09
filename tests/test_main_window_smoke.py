"""
Smoke test: the main window constructs and its wiring holds together.

Most of this fork's changes are signal and slot wiring across the window, the
models and the dialogs, which unit tests exercise only in pieces. Building the
real window catches a mistyped slot, a missing import or a signal connected to
the wrong signature - the class of breakage a merge introduces most easily.

`bitsandbytes` is stubbed because the window only probes whether it is
importable, and it needs torch, which CI deliberately does not install.
"""
import importlib.machinery
import sys
import types

import pytest
from PySide6.QtCore import QEventLoop, QTimer

if 'bitsandbytes' not in sys.modules:
    _stub = types.ModuleType('bitsandbytes')
    # transformers probes installed packages with `importlib.util.find_spec`,
    # which rejects a module whose `__spec__` is None.
    _stub.__spec__ = importlib.machinery.ModuleSpec('bitsandbytes', None)
    sys.modules['bitsandbytes'] = _stub

transformers = pytest.importorskip(
    'transformers', reason='the window loads a tokenizer in the background')

from widgets.main_window import MainWindow  # noqa: E402


@pytest.fixture
def window(qt_application):
    window = MainWindow(qt_application)
    yield window
    worker = window._tokenizer_worker
    if worker is not None:
        worker.wait(5000)
    window.image_list_model.tag_writer.shutdown()
    window.close()


def get_menu_titles(window) -> list[str]:
    return [action.text() for action in window.menuBar().actions()]


def get_menu(window, title: str):
    """
    The owning action and its menu. Both are returned because dropping the
    action's Python wrapper takes the submenu's wrapper with it.
    """
    menu_actions = window.menuBar().actions()
    for action in menu_actions:
        if action.text() == title:
            return action, action.menu()
    raise AssertionError(f'no {title} menu: '
                         f'{[action.text() for action in menu_actions]}')


def get_menu_action_titles(window, title: str) -> list[str]:
    menu_action, menu = get_menu(window, title)
    titles = [action.text() for action in menu.actions()]
    assert menu_action is not None
    return titles


def find_action(menu, title: str):
    actions = menu.actions()
    for action in actions:
        if action.text() == title:
            return action
    raise AssertionError(f'no {title} action: '
                         f'{[action.text() for action in actions]}')


def test_the_window_builds(window):
    assert window.image_list_model is not None
    assert window.proxy_image_list_model.sourceModel() is window.image_list_model


def test_the_menu_bar_has_its_top_level_menus(window):
    assert get_menu_titles(window) == ['File', 'Edit', 'Tools', 'View', 'Help']


def test_the_tools_menu_exposes_the_fork_features(window):
    titles = get_menu_action_titles(window, 'Tools')
    for expected in ('Aspect Ratio Bucket Calculator...', 'Caption Stats...',
                     'Update Tag Lists...', 'Download Token Counter...',
                     'Create Desktop Shortcut...'):
        assert expected in titles, expected


def test_the_caption_profile_submenu_lists_every_profile(window):
    from utils.caption_profiles import CaptionProfile

    _, tools_menu = get_menu(window, 'Tools')
    profile_action = find_action(tools_menu, 'Caption Profile')
    titles = [action.text() for action in profile_action.menu().actions()]
    for profile in CaptionProfile:
        assert profile.value in titles


def test_the_edit_menu_exposes_the_tag_tools(window):
    titles = get_menu_action_titles(window, 'Edit')
    for expected in ('Insert Trigger Token...', 'Reorder Tags (Illustrious)',
                     'Batch Reorder Tags...'):
        assert expected in titles, expected


def test_the_illustrious_reorder_action_runs_without_a_tag_list(window,
                                                                monkeypatch):
    """
    The menu action passes `triggered`'s checked flag, which must not be
    mistaken for the do-not-reorder-first-tag argument.
    """
    calls = []
    monkeypatch.setattr(
        window.image_list_model, 'reorder_illustrious_tags',
        lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(
        'widgets.main_window.QMessageBox.question',
        staticmethod(lambda *args, **kwargs: __import__(
            'PySide6.QtWidgets', fromlist=['QMessageBox']
        ).QMessageBox.StandardButton.Yes))

    _, edit_menu = get_menu(window, 'Edit')
    find_action(edit_menu, 'Reorder Tags (Illustrious)').trigger()

    assert calls == [{'character_tags': set(), 'series_tags': set(),
                      'do_not_reorder_first_tag': True}]


def test_write_errors_reach_the_window(window, monkeypatch, tmp_path):
    """The model reports failed writes; the window is what shows them."""
    shown = []
    monkeypatch.setattr('widgets.main_window.QMessageBox.critical',
                        staticmethod(lambda *args, **kwargs: shown.append(args)))
    window.image_list_model.write_errors_occurred.emit(
        [tmp_path / 'a.txt', tmp_path / 'b.txt'])
    assert len(shown) == 1
    assert 'a.txt' in shown[0][-1]


def test_the_background_tokenizer_load_finishes(window):
    loop = QEventLoop()
    QTimer.singleShot(3000, loop.quit)
    window._tokenizer_worker.loaded.connect(lambda *args: loop.quit())
    loop.exec()
    assert window.tokenizer is not None
    assert window.image_tags_editor.tokenizer is window.tokenizer
