"""
Tests for `utils.tag_writer` and how write failures reach the user.

Sidecar writes happen on a background thread so batch operations do not block
the UI. That makes error reporting a separate problem: a failure is only known
some time after the write was requested, so it has to be reported against the
file that actually failed, and a batch of failures has to arrive as one report
rather than one dialog per image.
"""
import pytest
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QMessageBox

import models.image_list_model as image_list_model_module
from models.image_list_model import ImageListModel
from utils.tag_writer import TagWriter


def wait_for(predicate, timeout_ms: int = 2000):
    """Spin the Qt event loop until `predicate` holds, so queued signals run."""
    loop = QEventLoop()
    elapsed = 0
    while not predicate() and elapsed < timeout_ms:
        QTimer.singleShot(10, loop.quit)
        loop.exec()
        elapsed += 10
    return predicate()


@pytest.fixture
def writer():
    writer = TagWriter()
    yield writer
    writer.shutdown()


class TestTagWriter:
    def test_it_writes_the_text_to_the_path(self, writer, tmp_path):
        path = tmp_path / 'image.txt'
        writer.enqueue(path, 'a, b')
        writer.flush()
        assert path.read_text(encoding='utf-8') == 'a, b'

    def test_the_last_write_for_a_path_wins(self, writer, tmp_path):
        path = tmp_path / 'image.txt'
        writer.enqueue(path, 'first')
        writer.enqueue(path, 'second')
        writer.flush()
        assert path.read_text(encoding='utf-8') == 'second'

    def test_a_failing_write_is_reported_against_its_own_path(self, writer,
                                                              tmp_path):
        # A directory cannot be overwritten by a file write.
        failing = tmp_path / 'unwritable'
        failing.mkdir()
        good = tmp_path / 'good.txt'

        reported = []
        writer.errors_occurred.connect(reported.extend)
        writer.enqueue(failing, 'nope')
        writer.enqueue(good, 'fine')
        writer.flush()

        assert wait_for(lambda: reported), 'no error was reported'
        assert reported == [failing]
        assert good.read_text(encoding='utf-8') == 'fine'

    def test_a_batch_of_failures_arrives_as_one_report(self, writer, tmp_path):
        failing = []
        for index in range(5):
            path = tmp_path / f'unwritable_{index}'
            path.mkdir()
            failing.append(path)

        reports = []
        writer.errors_occurred.connect(reports.append)
        for path in failing:
            writer.enqueue(path, 'nope')
        writer.flush()

        assert wait_for(lambda: reports)
        assert len(reports) == 1, 'one report per failure instead of one batch'
        assert sorted(reports[0]) == sorted(failing)

    def test_a_successful_run_reports_nothing(self, writer, tmp_path):
        reports = []
        writer.errors_occurred.connect(reports.append)
        writer.enqueue(tmp_path / 'image.txt', 'a')
        writer.flush()
        wait_for(lambda: False, timeout_ms=100)
        assert reports == []

    def test_errors_are_not_reported_twice(self, writer, tmp_path):
        failing = tmp_path / 'unwritable'
        failing.mkdir()
        reports = []
        writer.errors_occurred.connect(reports.append)

        writer.enqueue(failing, 'nope')
        writer.flush()
        assert wait_for(lambda: reports)

        writer.enqueue(tmp_path / 'good.txt', 'fine')
        writer.flush()
        wait_for(lambda: False, timeout_ms=100)
        assert len(reports) == 1


class TestModelErrorReporting:
    def test_the_model_forwards_write_errors_instead_of_blocking(
            self, tmp_path, monkeypatch, make_image):
        """
        The model must not open a modal dialog from inside a batch loop; it
        reports the failures and lets the window decide how to show them.
        """
        def fail_the_test(*args, **kwargs):
            raise AssertionError('a modal dialog was opened during a batch')

        monkeypatch.setattr(QMessageBox, 'exec', fail_the_test)
        monkeypatch.setattr(
            image_list_model_module, 'get_confirmation_dialog_reply',
            lambda *args, **kwargs: QMessageBox.StandardButton.Yes)

        model = ImageListModel(image_list_image_width=200, tag_separator=', ')
        reported = []
        model.write_errors_occurred.connect(reported.extend)

        unwritable = tmp_path / 'image_0.txt'
        unwritable.mkdir()
        model.beginResetModel()
        model.images = [make_image('image_0.png', ['solo'])]
        model.endResetModel()

        model.insert_trigger_token('sks', mode='first_tag')
        model.tag_writer.flush()

        assert wait_for(lambda: reported)
        assert reported == [unwritable]
        model.tag_writer.shutdown()
