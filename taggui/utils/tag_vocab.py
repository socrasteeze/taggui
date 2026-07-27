"""a1111-tagcomplete CSV vocab loader with prefix index and aliases."""
from __future__ import annotations

import csv
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QAbstractListModel, QModelIndex, Qt, QStandardPaths

# DominikDoom a1111-sd-webui-tagcomplete default lists on GitHub.
TAG_LIST_URLS = {
    'danbooru.csv': (
        'https://raw.githubusercontent.com/DominikDoom/a1111-sd-webui-tagcomplete'
        '/main/tags/danbooru.csv'),
    'e621.csv': (
        'https://raw.githubusercontent.com/DominikDoom/a1111-sd-webui-tagcomplete'
        '/main/tags/e621.csv'),
}

# Small built-in SDXL quality/style list used when no download is available.
SDXL_QUALITY_TAGS = [
    ('masterpiece', 0, 1000000, ''),
    ('best quality', 0, 900000, ''),
    ('high quality', 0, 800000, ''),
    ('amazing quality', 0, 700000, ''),
    ('very aesthetic', 0, 600000, ''),
    ('absurdres', 0, 500000, ''),
    ('highres', 0, 400000, ''),
    ('detailed', 0, 300000, ''),
    ('intricate details', 0, 200000, ''),
    ('sharp focus', 0, 100000, ''),
    ('cinematic lighting', 0, 90000, ''),
    ('depth of field', 0, 80000, ''),
    ('bokeh', 0, 70000, ''),
    ('realistic', 0, 60000, ''),
    ('photorealistic', 0, 50000, ''),
]


@dataclass(frozen=True)
class VocabTag:
    name: str
    category: int
    post_count: int
    aliases: tuple[str, ...]


def get_tags_directory() -> Path:
    base = Path(QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.AppDataLocation))
    path = base / 'tags'
    path.mkdir(parents=True, exist_ok=True)
    return path


class TagVocab:
    def __init__(self):
        self.tags: list[VocabTag] = []
        self.alias_to_canonical: dict[str, str] = {}
        self._prefix_index: dict[str, list[int]] = defaultdict(list)
        self.source_name: str | None = None

    def clear(self):
        self.tags.clear()
        self.alias_to_canonical.clear()
        self._prefix_index.clear()
        self.source_name = None

    def load_csv(self, path: Path) -> int:
        self.clear()
        self.source_name = path.name
        with path.open('r', encoding='utf-8', errors='replace',
                       newline='') as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row:
                    continue
                name = row[0].strip()
                if not name or name.lower() == 'name':
                    continue
                category = int(row[1]) if len(row) > 1 and row[1].isdigit() else 0
                post_count = int(row[2]) if len(row) > 2 and row[2].lstrip('-').isdigit() else 0
                aliases_raw = row[3] if len(row) > 3 else ''
                aliases = tuple(
                    alias.strip() for alias in aliases_raw.split(',')
                    if alias.strip())
                index = len(self.tags)
                self.tags.append(VocabTag(name, category, post_count, aliases))
                self._index_term(name, index)
                self.alias_to_canonical[name.casefold()] = name
                for alias in aliases:
                    self.alias_to_canonical[alias.casefold()] = name
                    self._index_term(alias, index)
        return len(self.tags)

    def load_builtin_sdxl(self) -> int:
        self.clear()
        self.source_name = 'sdxl_quality.csv'
        for name, category, post_count, aliases_raw in SDXL_QUALITY_TAGS:
            aliases = tuple(
                alias.strip() for alias in aliases_raw.split(',')
                if alias.strip()) if aliases_raw else ()
            index = len(self.tags)
            self.tags.append(VocabTag(name, category, post_count, aliases))
            self._index_term(name, index)
            self.alias_to_canonical[name.casefold()] = name
        return len(self.tags)

    def _index_term(self, term: str, index: int):
        key = term.casefold()[:2] if len(term) >= 2 else term.casefold()
        self._prefix_index[key].append(index)

    def resolve(self, text: str) -> str:
        return self.alias_to_canonical.get(text.casefold(), text)

    def suggest(self, prefix: str, limit: int = 50) -> list[VocabTag]:
        if not prefix:
            return []
        key = prefix.casefold()
        bucket_key = key[:2] if len(key) >= 2 else key
        candidates = self._prefix_index.get(bucket_key, [])
        # Also check single-char bucket when prefix grows past 1 char.
        if len(key) >= 2:
            candidates = list(dict.fromkeys(
                candidates + self._prefix_index.get(key[:1], [])))
        matches: list[VocabTag] = []
        seen: set[str] = set()
        for index in candidates:
            tag = self.tags[index]
            name_cf = tag.name.casefold()
            if name_cf.startswith(key) or any(
                    alias.casefold().startswith(key) for alias in tag.aliases):
                if tag.name in seen:
                    continue
                seen.add(tag.name)
                matches.append(tag)
        matches.sort(key=lambda tag: (-tag.post_count, tag.name))
        return matches[:limit]


def download_tag_list(filename: str, destination: Path | None = None) -> Path:
    url = TAG_LIST_URLS.get(filename)
    if url is None:
        raise ValueError(f'Unknown tag list: {filename}')
    destination = destination or (get_tags_directory() / filename)
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destination)
    return destination


def ensure_vocab_file(filename: str, download_if_missing: bool = False
                      ) -> Path | None:
    """Return a local CSV path. Downloads only when explicitly requested."""
    if filename == 'sdxl_quality.csv':
        return None  # handled in-memory
    path = get_tags_directory() / filename
    if path.is_file():
        return path
    if download_if_missing and filename in TAG_LIST_URLS:
        try:
            return download_tag_list(filename, path)
        except Exception:
            return None
    return None


class MergedTagCompleterModel(QAbstractListModel):
    """Dataset frequency tags first, then vocab CSV matches."""

    def __init__(self, tag_counter_model, vocab: TagVocab):
        super().__init__()
        self.tag_counter_model = tag_counter_model
        self.vocab = vocab
        self._prefix = ''
        self._rows: list[str] = []

    def set_prefix(self, prefix: str):
        self.beginResetModel()
        self._prefix = prefix
        self._rows = self._build_rows(prefix)
        self.endResetModel()

    def _build_rows(self, prefix: str) -> list[str]:
        rows: list[str] = []
        seen: set[str] = set()
        prefix_cf = prefix.casefold()
        # Dataset tags from TagCounterModel (list of (tag, count) via tags).
        tags = getattr(self.tag_counter_model, 'tags', None)
        if tags is None and hasattr(self.tag_counter_model, 'tag_counter'):
            tags = list(self.tag_counter_model.tag_counter.most_common())
        if tags:
            for tag, _count in tags:
                if not prefix_cf or tag.casefold().startswith(prefix_cf):
                    if tag not in seen:
                        rows.append(tag)
                        seen.add(tag)
        for vocab_tag in self.vocab.suggest(prefix):
            canonical = vocab_tag.name
            if canonical not in seen:
                rows.append(canonical)
                seen.add(canonical)
        return rows[:80]

    def rowCount(self, parent=QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self._rows)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < len(self._rows)):
            return None
        tag = self._rows[index.row()]
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            return tag
        return None

    def filter_accepts(self, text: str) -> bool:
        self.set_prefix(text)
        return True
