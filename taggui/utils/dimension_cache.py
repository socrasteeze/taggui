"""Disk cache of image dimensions keyed by path, mtime, and size."""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QStandardPaths


def get_cache_path() -> Path:
    base = Path(QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.AppDataLocation))
    base.mkdir(parents=True, exist_ok=True)
    return base / 'dimension_cache.json'


# Entries are keyed by path, mtime and size, so every edit to an image leaves
# its old entry behind for good. Above this many, saving keeps only what the
# current session touched, which bounds both the file and the startup parse.
MAXIMUM_ENTRY_COUNT = 100_000


class DimensionCache:
    def __init__(self, cache_path: Path | None = None,
                 maximum_entry_count: int = MAXIMUM_ENTRY_COUNT):
        self.cache_path = cache_path or get_cache_path()
        self.maximum_entry_count = maximum_entry_count
        self._data: dict[str, dict] = {}
        # Keys read or written this session; the ones worth keeping if the
        # cache has to be pruned.
        self._used_keys: set[str] = set()
        self._dirty = False
        self.load()

    def load(self):
        if not self.cache_path.is_file():
            self._data = {}
            return
        try:
            self._data = json.loads(
                self.cache_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            self._data = {}

    def prune(self):
        """Drop entries this session never used, once the cache grows large."""
        if len(self._data) <= self.maximum_entry_count:
            return
        self._data = {key: value for key, value in self._data.items()
                      if key in self._used_keys}
        self._dirty = True

    def save(self):
        self.prune()
        if not self._dirty:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(self._data), encoding='utf-8')
            self._dirty = False
        except OSError:
            pass

    @staticmethod
    def _key(path: Path) -> str | None:
        try:
            stat = path.stat()
        except OSError:
            return None
        return f'{path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}'

    def get(self, path: Path) -> tuple[int, int] | None:
        key = self._key(path)
        if key is None:
            return None
        entry = self._data.get(key)
        if not entry:
            return None
        self._used_keys.add(key)
        width = entry.get('w')
        height = entry.get('h')
        if width is None or height is None:
            return None
        return int(width), int(height)

    def set(self, path: Path, dimensions: tuple[int, int] | None):
        if dimensions is None:
            return
        key = self._key(path)
        if key is None:
            return
        self._data[key] = {'w': dimensions[0], 'h': dimensions[1]}
        self._used_keys.add(key)
        self._dirty = True
