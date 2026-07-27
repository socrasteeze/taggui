"""Background queue for writing image tag sidecar files."""
from __future__ import annotations

import queue
import threading
from pathlib import Path


class TagWriter:
    def __init__(self):
        self._queue: queue.Queue[tuple[Path, str] | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name='TagWriter')
        self._thread.start()
        self._errors: list[Path] = []
        self._errors_lock = threading.Lock()

    def enqueue(self, path: Path, text: str):
        self._queue.put((path, text))

    def flush(self):
        self._queue.join()

    def pop_errors(self) -> list[Path]:
        with self._errors_lock:
            errors = self._errors
            self._errors = []
            return errors

    def shutdown(self):
        self._queue.put(None)
        self._thread.join(timeout=5)

    def _run(self):
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                path, text = item
                try:
                    path.write_text(text, encoding='utf-8', errors='replace')
                except OSError:
                    with self._errors_lock:
                        self._errors.append(path)
            finally:
                self._queue.task_done()
