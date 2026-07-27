import os
import random
import re
import sys
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import exifread
import imagesize
from PySide6.QtCore import (QAbstractListModel, QModelIndex, QRunnable, QSize,
                            Qt, QThread, QThreadPool, Signal, Slot)
from PySide6.QtGui import QIcon, QImage, QImageReader, QPixmap
from PySide6.QtWidgets import QMessageBox

from utils.dimension_cache import DimensionCache
from utils.image import Image
from utils.settings import DEFAULT_SETTINGS, get_settings
from utils.tag_writer import TagWriter
from utils.utils import get_confirmation_dialog_reply, pluralize

UNDO_STACK_SIZE = 32

BACKUP_DIRECTORY_NAME = 'original_images'

# Formats that commonly carry EXIF orientation.
_EXIF_SUFFIXES = {'.jpg', '.jpeg', '.tif', '.tiff', '.webp'}


def get_file_paths(directory_path: Path) -> set[Path]:
    """
    Recursively get all file paths in a directory, including those in
    subdirectories. The `original_images` backup directory created by the
    bucket processor is skipped so that backed-up originals are not reloaded.
    """
    file_paths = set()
    for path in directory_path.iterdir():
        if path.is_file():
            file_paths.add(path)
        elif path.is_dir() and path.name != BACKUP_DIRECTORY_NAME:
            file_paths.update(get_file_paths(path))
    return file_paths


@dataclass
class HistoryItem:
    action_name: str
    # Sparse map of image index -> tags before the action.
    previous_tags: dict[int, list[str]]
    should_ask_for_confirmation: bool


class Scope(str, Enum):
    ALL_IMAGES = 'All images'
    FILTERED_IMAGES = 'Filtered images'
    SELECTED_IMAGES = 'Selected images'


class DirectoryLoadWorker(QThread):
    progress = Signal(int, int)
    finished_loading = Signal(list)
    failed = Signal(str)

    def __init__(self, directory_path: Path, tag_separator: str,
                 image_suffixes: list[str], dimension_cache: DimensionCache):
        super().__init__()
        self.directory_path = directory_path
        self.tag_separator = tag_separator
        self.image_suffixes = image_suffixes
        self.dimension_cache = dimension_cache

    def run(self):
        try:
            file_paths = get_file_paths(self.directory_path)
            image_paths = sorted(
                path for path in file_paths
                if path.suffix.lower() in self.image_suffixes)
            text_file_path_strings = {str(path) for path in file_paths
                                      if path.suffix == '.txt'}
            total = len(image_paths)
            images: list[Image] = []
            max_workers = min(32, (os.cpu_count() or 4) * 4)
            completed = 0
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._load_image, image_path, text_file_path_strings
                    ): image_path
                    for image_path in image_paths
                }
                for future in as_completed(futures):
                    images.append(future.result())
                    completed += 1
                    if completed == 1 or completed == total or completed % 25 == 0:
                        self.progress.emit(completed, total)
            images.sort(key=lambda image_: image_.path)
            self.dimension_cache.save()
            self.finished_loading.emit(images)
        except Exception as exception:
            self.failed.emit(str(exception))

    def _load_image(self, image_path: Path,
                    text_file_path_strings: set[str]) -> Image:
        dimensions = self.dimension_cache.get(image_path)
        if dimensions is None:
            try:
                dimensions = imagesize.get(image_path)
                if image_path.suffix.lower() in _EXIF_SUFFIXES:
                    with open(image_path, 'rb') as image_file:
                        try:
                            exif_tags = exifread.process_file(
                                image_file, details=False,
                                stop_tag='Image Orientation')
                            if 'Image Orientation' in exif_tags:
                                orientations = (
                                    exif_tags['Image Orientation'].values)
                                if any(value in orientations
                                       for value in (5, 6, 7, 8)):
                                    dimensions = (dimensions[1],
                                                  dimensions[0])
                        except Exception as exception:
                            print(f'Failed to get Exif tags for {image_path}: '
                                  f'{exception}', file=sys.stderr)
                self.dimension_cache.set(image_path, dimensions)
            except (ValueError, OSError) as exception:
                print(f'Failed to get dimensions for {image_path}: '
                      f'{exception}', file=sys.stderr)
                dimensions = None
        tags = []
        text_file_path = image_path.with_suffix('.txt')
        if str(text_file_path) in text_file_path_strings:
            caption = text_file_path.read_text(encoding='utf-8',
                                               errors='replace')
            if caption:
                tags = caption.split(self.tag_separator)
                tags = [tag.strip() for tag in tags]
                tags = [tag for tag in tags if tag]
        return Image(image_path, dimensions, tags)


class _ThumbnailTask(QRunnable):
    def __init__(self, model: 'ImageListModel', row: int, path: Path,
                 width: int):
        super().__init__()
        self.model = model
        self.row = row
        self.path = path
        self.width = width

    def run(self):
        try:
            image_reader = QImageReader(str(self.path))
            image_reader.setAutoTransform(True)
            source_size = image_reader.size()
            if source_size.isValid() and source_size.width() > 0:
                scaled_height = round(self.width * source_size.height()
                                      / source_size.width())
                image_reader.setScaledSize(
                    QSize(self.width, max(scaled_height, 1)))
            image = image_reader.read()
            if (not image.isNull() and image.width() != self.width
                    and image.width() > 0):
                image = image.scaledToWidth(
                    self.width, Qt.TransformationMode.SmoothTransformation)
            self.model.thumbnail_ready.emit(self.row, str(self.path), image)
        except Exception:
            self.model.thumbnail_ready.emit(self.row, str(self.path),
                                            QImage())


class ImageListModel(QAbstractListModel):
    update_undo_and_redo_actions_requested = Signal()
    load_progress = Signal(int, int)
    load_finished = Signal()
    load_failed = Signal(str)
    thumbnail_ready = Signal(int, str, QImage)

    def __init__(self, image_list_image_width: int, tag_separator: str):
        super().__init__()
        self.image_list_image_width = image_list_image_width
        self.tag_separator = tag_separator
        self.images: list[Image] = []
        self.undo_stack = deque(maxlen=UNDO_STACK_SIZE)
        self.redo_stack = []
        self.proxy_image_list_model = None
        self.image_list_selection_model = None
        self.dimension_cache = DimensionCache()
        self.tag_writer = TagWriter()
        self._load_worker: DirectoryLoadWorker | None = None
        self._thumbnail_pool = QThreadPool.globalInstance()
        self._pending_thumbnails: set[int] = set()
        self.thumbnail_ready.connect(self._on_thumbnail_ready)

    def rowCount(self, parent=None) -> int:
        return len(self.images)

    def data(self, index, role=None) -> Image | str | QIcon | QSize:
        image = self.images[index.row()]
        if role == Qt.ItemDataRole.UserRole:
            return image
        if role == Qt.ItemDataRole.DisplayRole:
            text = image.path.name
            if image.tags:
                caption = self.tag_separator.join(image.tags)
                text += f'\n{caption}'
            return text
        if role == Qt.ItemDataRole.DecorationRole:
            if image.thumbnail:
                return image.thumbnail
            row = index.row()
            if row not in self._pending_thumbnails:
                self._pending_thumbnails.add(row)
                task = _ThumbnailTask(self, row, image.path,
                                      self.image_list_image_width)
                self._thumbnail_pool.start(task)
            return QIcon()
        if role == Qt.ItemDataRole.SizeHintRole:
            if image.thumbnail:
                sizes = image.thumbnail.availableSizes()
                if sizes:
                    return sizes[0]
            dimensions = image.dimensions
            if not dimensions:
                return QSize(self.image_list_image_width,
                             self.image_list_image_width)
            width, height = dimensions
            return QSize(self.image_list_image_width,
                         int(self.image_list_image_width * height / width))

    @Slot(int, str, QImage)
    def _on_thumbnail_ready(self, row: int, path_str: str, qimage: QImage):
        self._pending_thumbnails.discard(row)
        if row < 0 or row >= len(self.images):
            return
        image = self.images[row]
        if str(image.path) != path_str:
            return
        if qimage.isNull():
            return
        image.thumbnail = QIcon(QPixmap.fromImage(qimage))
        model_index = self.index(row)
        self.dataChanged.emit(model_index, model_index,
                              [Qt.ItemDataRole.DecorationRole,
                               Qt.ItemDataRole.SizeHintRole])

    def load_directory(self, directory_path: Path):
        """Start an asynchronous directory load (non-blocking)."""
        if self._load_worker and self._load_worker.isRunning():
            self._load_worker.wait(1000)
        self.beginResetModel()
        self.images.clear()
        self.endResetModel()
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._pending_thumbnails.clear()
        self.update_undo_and_redo_actions_requested.emit()
        settings = get_settings()
        image_suffixes_string = settings.value(
            'image_list_file_formats',
            defaultValue=DEFAULT_SETTINGS['image_list_file_formats'], type=str)
        image_suffixes = []
        for suffix in image_suffixes_string.split(','):
            suffix = suffix.strip().lower()
            if not suffix.startswith('.'):
                suffix = '.' + suffix
            image_suffixes.append(suffix)
        self._load_worker = DirectoryLoadWorker(
            directory_path, self.tag_separator, image_suffixes,
            self.dimension_cache)
        self._load_worker.progress.connect(self.load_progress)
        self._load_worker.finished_loading.connect(self._on_load_finished)
        self._load_worker.failed.connect(self.load_failed)
        self._load_worker.start()

    @Slot(list)
    def _on_load_finished(self, images: list):
        self.beginResetModel()
        self.images = images
        self.endResetModel()
        self.load_finished.emit()

    def add_to_undo_stack(self, action_name: str,
                          should_ask_for_confirmation: bool,
                          image_indices: list[int] | None = None):
        """Add a sparse snapshot of image tags to the undo stack."""
        if image_indices is None:
            previous_tags = {i: image.tags.copy()
                             for i, image in enumerate(self.images)}
        else:
            previous_tags = {i: self.images[i].tags.copy()
                             for i in image_indices}
        self.undo_stack.append(HistoryItem(action_name, previous_tags,
                                           should_ask_for_confirmation))
        self.redo_stack.clear()
        self.update_undo_and_redo_actions_requested.emit()

    def _commit_changes(self, action_name: str,
                        should_ask_for_confirmation: bool,
                        changes: dict[int, list[str]]):
        """
        Apply tag changes. `changes` maps image index -> tags *before* the
        change; the images already hold the new tags.
        """
        if not changes:
            return
        self.undo_stack.append(HistoryItem(
            action_name, {i: tags.copy() for i, tags in changes.items()},
            should_ask_for_confirmation))
        self.redo_stack.clear()
        self.update_undo_and_redo_actions_requested.emit()
        changed_indices = sorted(changes)
        for image_index in changed_indices:
            image = self.images[image_index]
            image.token_count = None
            self.write_image_tags_to_disk(image)
        self.dataChanged.emit(self.index(changed_indices[0]),
                              self.index(changed_indices[-1]))

    def write_image_tags_to_disk(self, image: Image):
        text = self.tag_separator.join(image.tags)
        self.tag_writer.enqueue(image.path.with_suffix('.txt'), text)
        errors = self.tag_writer.pop_errors()
        if errors:
            error_message_box = QMessageBox()
            error_message_box.setWindowTitle('Error')
            error_message_box.setIcon(QMessageBox.Icon.Critical)
            error_message_box.setText(
                f'Failed to save tags for {errors[0]}.')
            error_message_box.exec()

    def restore_history_tags(self, is_undo: bool):
        if is_undo:
            source_stack = self.undo_stack
            destination_stack = self.redo_stack
        else:
            source_stack = self.redo_stack
            destination_stack = self.undo_stack
        if not source_stack:
            return
        history_item = source_stack[-1]
        if history_item.should_ask_for_confirmation:
            undo_or_redo_string = 'Undo' if is_undo else 'Redo'
            reply = get_confirmation_dialog_reply(
                title=undo_or_redo_string,
                question=f'{undo_or_redo_string} '
                         f'"{history_item.action_name}"?')
            if reply != QMessageBox.StandardButton.Yes:
                return
        source_stack.pop()
        redo_previous: dict[int, list[str]] = {}
        changed_image_indices = []
        for image_index, history_image_tags in history_item.previous_tags.items():
            if image_index < 0 or image_index >= len(self.images):
                continue
            image = self.images[image_index]
            if image.tags == history_image_tags:
                continue
            redo_previous[image_index] = image.tags.copy()
            changed_image_indices.append(image_index)
            image.tags = history_image_tags.copy()
            image.token_count = None
            self.write_image_tags_to_disk(image)
        destination_stack.append(HistoryItem(
            history_item.action_name, redo_previous,
            history_item.should_ask_for_confirmation))
        if changed_image_indices:
            changed_image_indices.sort()
            self.dataChanged.emit(self.index(changed_image_indices[0]),
                                  self.index(changed_image_indices[-1]))
        self.update_undo_and_redo_actions_requested.emit()

    @Slot()
    def undo(self):
        self.restore_history_tags(is_undo=True)

    @Slot()
    def redo(self):
        self.restore_history_tags(is_undo=False)

    def is_image_in_scope(self, scope: Scope | str, image_index: int,
                          image: Image) -> bool:
        if scope == Scope.ALL_IMAGES:
            return True
        if scope == Scope.FILTERED_IMAGES:
            return self.proxy_image_list_model.is_image_in_filtered_images(
                image)
        if scope == Scope.SELECTED_IMAGES:
            proxy_index = self.proxy_image_list_model.mapFromSource(
                self.index(image_index))
            return self.image_list_selection_model.isSelected(proxy_index)

    def get_text_match_count(self, text: str, scope: Scope | str,
                             whole_tags_only: bool, use_regex: bool) -> int:
        match_count = 0
        for image_index, image in enumerate(self.images):
            if not self.is_image_in_scope(scope, image_index, image):
                continue
            if whole_tags_only:
                if use_regex:
                    match_count += len([
                        tag for tag in image.tags
                        if re.fullmatch(pattern=text, string=tag)
                    ])
                else:
                    match_count += image.tags.count(text)
            else:
                caption = self.tag_separator.join(image.tags)
                if use_regex:
                    match_count += len(re.findall(pattern=text,
                                                  string=caption))
                else:
                    match_count += caption.count(text)
        return match_count

    def find_and_replace(self, find_text: str, replace_text: str,
                         scope: Scope | str, use_regex: bool):
        if not find_text:
            return
        changes: dict[int, list[str]] = {}
        for image_index, image in enumerate(self.images):
            if not self.is_image_in_scope(scope, image_index, image):
                continue
            caption = self.tag_separator.join(image.tags)
            if use_regex:
                if not re.search(pattern=find_text, string=caption):
                    continue
                new_caption = re.sub(pattern=find_text, repl=replace_text,
                                     string=caption)
            else:
                if find_text not in caption:
                    continue
                new_caption = caption.replace(find_text, replace_text)
            old_tags = image.tags.copy()
            image.tags = new_caption.split(self.tag_separator)
            changes[image_index] = old_tags
        self._commit_changes('Find and Replace', True, changes)

    def sort_tags_alphabetically(self, do_not_reorder_first_tag: bool):
        changes: dict[int, list[str]] = {}
        for image_index, image in enumerate(self.images):
            if len(image.tags) < 2:
                continue
            old_tags = image.tags.copy()
            if do_not_reorder_first_tag:
                first_tag = image.tags[0]
                image.tags = [first_tag] + sorted(image.tags[1:])
            else:
                image.tags = sorted(image.tags)
            if image.tags != old_tags:
                changes[image_index] = old_tags
        self._commit_changes('Sort Tags', True, changes)

    def sort_tags_by_frequency(self, tag_counter: Counter,
                               do_not_reorder_first_tag: bool):
        changes: dict[int, list[str]] = {}
        for image_index, image in enumerate(self.images):
            if len(image.tags) < 2:
                continue
            old_tags = image.tags.copy()
            if do_not_reorder_first_tag:
                first_tag = image.tags[0]
                image.tags = [first_tag] + sorted(
                    image.tags[1:], key=lambda tag: tag_counter[tag],
                    reverse=True)
            else:
                image.tags = sorted(
                    image.tags, key=lambda tag: tag_counter[tag], reverse=True)
            if image.tags != old_tags:
                changes[image_index] = old_tags
        self._commit_changes('Sort Tags', True, changes)

    def reverse_tags_order(self, do_not_reorder_first_tag: bool):
        changes: dict[int, list[str]] = {}
        for image_index, image in enumerate(self.images):
            if len(image.tags) < 2:
                continue
            old_tags = image.tags.copy()
            if do_not_reorder_first_tag:
                image.tags = [image.tags[0]] + list(reversed(image.tags[1:]))
            else:
                image.tags = list(reversed(image.tags))
            changes[image_index] = old_tags
        self._commit_changes('Reverse Order of Tags', True, changes)

    def shuffle_tags(self, do_not_reorder_first_tag: bool):
        changes: dict[int, list[str]] = {}
        for image_index, image in enumerate(self.images):
            if len(image.tags) < 2:
                continue
            old_tags = image.tags.copy()
            if do_not_reorder_first_tag:
                first_tag, *remaining_tags = image.tags
                random.shuffle(remaining_tags)
                image.tags = [first_tag] + remaining_tags
            else:
                shuffled = image.tags[:]
                random.shuffle(shuffled)
                image.tags = shuffled
            changes[image_index] = old_tags
        self._commit_changes('Shuffle Tags', True, changes)

    def move_tags_to_front(self, tags_to_move: list[str]):
        changes: dict[int, list[str]] = {}
        for image_index, image in enumerate(self.images):
            if not any(tag in image.tags for tag in tags_to_move):
                continue
            old_tags = image.tags.copy()
            moved_tags = []
            for tag in tags_to_move:
                tag_count = image.tags.count(tag)
                moved_tags.extend([tag] * tag_count)
            unmoved_tags = [tag for tag in image.tags if tag not in moved_tags]
            image.tags = moved_tags + unmoved_tags
            if image.tags != old_tags:
                changes[image_index] = old_tags
        self._commit_changes('Move Tags to Front', True, changes)

    def move_tags_to_back(self, tags_to_move: list[str]):
        changes: dict[int, list[str]] = {}
        for image_index, image in enumerate(self.images):
            if not any(tag in image.tags for tag in tags_to_move):
                continue
            old_tags = image.tags.copy()
            moved_tags = []
            for tag in tags_to_move:
                tag_count = image.tags.count(tag)
                moved_tags.extend([tag] * tag_count)
            unmoved_tags = [tag for tag in image.tags if tag not in moved_tags]
            image.tags = unmoved_tags + moved_tags
            if image.tags != old_tags:
                changes[image_index] = old_tags
        self._commit_changes('Move Tags to Back', True, changes)

    def insert_trigger_token(self, trigger: str, mode: str = 'first_tag',
                             scope: Scope | str = Scope.ALL_IMAGES):
        """Insert a trigger token as first tag or embed into first sentence."""
        if not trigger.strip():
            return
        trigger = trigger.strip()
        changes: dict[int, list[str]] = {}
        for image_index, image in enumerate(self.images):
            if not self.is_image_in_scope(scope, image_index, image):
                continue
            if trigger in image.tags or any(trigger in tag for tag in image.tags):
                continue
            old_tags = image.tags.copy()
            if mode == 'embedded':
                if image.tags:
                    image.tags[0] = f'{trigger} {image.tags[0]}'
                else:
                    image.tags = [trigger]
            else:
                image.tags = [trigger] + image.tags
            changes[image_index] = old_tags
        self._commit_changes('Insert Trigger Token', True, changes)

    def reorder_illustrious_tags(self, character_tags: set[str] | None = None,
                                 series_tags: set[str] | None = None,
                                 do_not_reorder_first_tag: bool = True):
        """
        Reorder tags: count → character → series → general.
        Count tags match patterns like 1girl / 2boys. Character/series sets
        come from WD category metadata when available.
        """
        character_tags = character_tags or set()
        series_tags = series_tags or set()
        count_pattern = re.compile(
            r'^\d+(girl|girls|boy|boys|other|others)$', re.IGNORECASE)

        def sort_key(tag: str) -> tuple[int, str]:
            if count_pattern.match(tag.replace(' ', '')):
                return (0, tag)
            folded = tag.casefold()
            if folded in character_tags or tag in character_tags:
                return (1, tag)
            if folded in series_tags or tag in series_tags:
                return (2, tag)
            return (3, tag)

        changes: dict[int, list[str]] = {}
        for image_index, image in enumerate(self.images):
            if len(image.tags) < 2:
                continue
            old_tags = image.tags.copy()
            if do_not_reorder_first_tag:
                first = image.tags[0]
                rest = sorted(image.tags[1:], key=sort_key)
                image.tags = [first] + rest
            else:
                image.tags = sorted(image.tags, key=sort_key)
            if image.tags != old_tags:
                changes[image_index] = old_tags
        self._commit_changes('Illustrious Tag Order', True, changes)

    def remove_duplicate_tags(self) -> int:
        changes: dict[int, list[str]] = {}
        removed_tag_count = 0
        for image_index, image in enumerate(self.images):
            tag_count = len(image.tags)
            unique_tag_count = len(set(image.tags))
            if tag_count == unique_tag_count:
                continue
            old_tags = image.tags.copy()
            removed_tag_count += tag_count - unique_tag_count
            image.tags = list(dict.fromkeys(image.tags))
            changes[image_index] = old_tags
        self._commit_changes('Remove Duplicate Tags', True, changes)
        return removed_tag_count

    def remove_empty_tags(self) -> int:
        changes: dict[int, list[str]] = {}
        removed_tag_count = 0
        for image_index, image in enumerate(self.images):
            old_tags = image.tags.copy()
            image.tags = [tag for tag in image.tags if tag.strip()]
            removed = len(old_tags) - len(image.tags)
            if removed:
                removed_tag_count += removed
                changes[image_index] = old_tags
        self._commit_changes('Remove Empty Tags', True, changes)
        return removed_tag_count

    def update_image_tags(self, image_index: QModelIndex, tags: list[str]):
        image: Image = self.data(image_index, Qt.ItemDataRole.UserRole)
        if image.tags == tags:
            return
        image.tags = tags
        image.token_count = None
        self.dataChanged.emit(image_index, image_index)
        self.write_image_tags_to_disk(image)

    @Slot(list, list)
    def add_tags(self, tags: list[str], image_indices: list[QModelIndex]):
        if not image_indices:
            return
        action_name = f'Add {pluralize("Tag", len(tags))}'
        should_ask_for_confirmation = len(image_indices) > 1
        rows = [index.row() for index in image_indices]
        self.add_to_undo_stack(action_name, should_ask_for_confirmation, rows)
        for image_index in image_indices:
            image: Image = self.data(image_index, Qt.ItemDataRole.UserRole)
            image.tags.extend(tags)
            image.token_count = None
            self.write_image_tags_to_disk(image)
        min_image_index = min(image_indices, key=lambda index: index.row())
        max_image_index = max(image_indices, key=lambda index: index.row())
        self.dataChanged.emit(min_image_index, max_image_index)

    @Slot(list, str)
    def rename_tags(self, old_tags: list[str], new_tag: str,
                    scope: Scope | str = Scope.ALL_IMAGES,
                    use_regex: bool = False):
        changes: dict[int, list[str]] = {}
        for image_index, image in enumerate(self.images):
            if not self.is_image_in_scope(scope, image_index, image):
                continue
            old_image_tags = image.tags.copy()
            if use_regex:
                pattern = old_tags[0]
                if not any(re.fullmatch(pattern=pattern, string=image_tag)
                           for image_tag in image.tags):
                    continue
                image.tags = [new_tag if re.fullmatch(pattern=pattern,
                                                      string=image_tag)
                              else image_tag for image_tag in image.tags]
            else:
                if not any(old_tag in image.tags for old_tag in old_tags):
                    continue
                image.tags = [new_tag if image_tag in old_tags else image_tag
                              for image_tag in image.tags]
            if image.tags != old_image_tags:
                changes[image_index] = old_image_tags
        self._commit_changes(
            f'Rename {pluralize("Tag", len(old_tags))}', True, changes)

    @Slot(list)
    def delete_tags(self, tags: list[str],
                    scope: Scope | str = Scope.ALL_IMAGES,
                    use_regex: bool = False):
        changes: dict[int, list[str]] = {}
        for image_index, image in enumerate(self.images):
            if not self.is_image_in_scope(scope, image_index, image):
                continue
            old_image_tags = image.tags.copy()
            if use_regex:
                pattern = tags[0]
                if not any(re.fullmatch(pattern=pattern, string=image_tag)
                           for image_tag in image.tags):
                    continue
                image.tags = [image_tag for image_tag in image.tags
                              if not re.fullmatch(pattern=pattern,
                                                  string=image_tag)]
            else:
                if not any(tag in image.tags for tag in tags):
                    continue
                image.tags = [image_tag for image_tag in image.tags
                              if image_tag not in tags]
            if image.tags != old_image_tags:
                changes[image_index] = old_image_tags
        self._commit_changes(
            f'Delete {pluralize("Tag", len(tags))}', True, changes)

    def export_jsonl(self, destination: Path) -> int:
        import json
        lines = []
        for image in self.images:
            lines.append(json.dumps({
                'file_name': image.path.name,
                'file_path': str(image.path),
                'text': self.tag_separator.join(image.tags),
            }, ensure_ascii=False))
        destination.write_text('\n'.join(lines) + ('\n' if lines else ''),
                               encoding='utf-8')
        return len(lines)

    def export_kohya_metadata(self, destination: Path) -> int:
        import json
        metadata = {}
        for image in self.images:
            key = str(image.path)
            metadata[key] = {
                'caption': self.tag_separator.join(image.tags),
                'tags': image.tags,
            }
        destination.write_text(json.dumps(metadata, indent=2,
                                          ensure_ascii=False),
                               encoding='utf-8')
        return len(metadata)
