"""Background queue for writing image tag sidecar files."""
from __future__ import annotations

import queue
import threading
from pathlib import Path

from PySide6.QtCore import QObject, Signal


class TagWriter(QObject):
    """
    Writes `.txt` sidecars off the UI thread.

    Failures are reported through `errors_occurred` once the queue drains, so
    a batch operation produces a single report naming the files that actually
    failed. The signal crosses from the worker thread to whichever thread owns
    this object, so receivers can safely open dialogs.
    """

    errors_occurred = Signal(list)

    def __init__(self):
        super().__init__()
        self._queue: queue.Queue[tuple[Path, str] | None] = queue.Queue()
        self._errors: list[Path] = []
        self._errors_lock = threading.Lock()
        # Started last: the worker touches the state above.
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name='TagWriter')
        self._thread.start()

    def enqueue(self, path: Path, text: str):
        self._queue.put((path, text))

    def flush(self):
        """Block until every queued write has been attempted."""
        self._queue.join()

    def pop_errors(self) -> list[Path]:
        """Take the pending failures without waiting for the next report."""
        with self._errors_lock:
            errors = self._errors
            self._errors = []
            return errors

    def shutdown(self):
        self._queue.put(None)
        self._thread.join(timeout=5)

    def _report_errors(self):
        errors = self.pop_errors()
        if errors:
            self.errors_occurred.emit(errors)

    def _run(self):
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                # The queue has drained, so every failure from the batch that
                # just finished is known. Report them together, then wait.
                self._report_errors()
                item = self._queue.get()
            try:
                if item is None:
                    self._report_errors()
                    return
                path, text = item
                try:
                    path.write_text(text, encoding='utf-8', errors='replace')
                except OSError:
                    with self._errors_lock:
                        self._errors.append(path)
            finally:
                self._queue.task_done()
