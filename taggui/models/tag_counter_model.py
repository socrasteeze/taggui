from collections import Counter

from PySide6.QtCore import QAbstractListModel, QTimer, Qt, Signal, Slot
from PySide6.QtWidgets import QMessageBox

from utils.image import Image
from utils.utils import get_confirmation_dialog_reply, list_with_and, pluralize


class TagCounterModel(QAbstractListModel):
    tags_renaming_requested = Signal(list, str)

    def __init__(self):
        super().__init__()
        self.tag_counter = Counter()
        self.most_common_tags = []
        self.all_tags_list = None
        # A snapshot of the tags last counted for each image, keyed by path, so
        # that edits can be applied incrementally instead of recounting every
        # image on every change.
        self.counted_tags = {}
        self._is_publish_pending = False
        self._publish_timer = QTimer(self)
        self._publish_timer.setSingleShot(True)
        self._publish_timer.setInterval(0)
        self._publish_timer.timeout.connect(self.publish_pending_counts)

    @property
    def tags(self):
        """Alias used by the merged vocab completer."""
        return self.most_common_tags

    def rowCount(self, parent=None) -> int:
        return len(self.most_common_tags)

    def data(self, index, role=None) -> tuple[str, int] | str:
        tag, count = self.most_common_tags[index.row()]
        if role == Qt.ItemDataRole.UserRole:
            return tag, count
        if role == Qt.ItemDataRole.DisplayRole:
            return f'{tag} ({count})'
        if role == Qt.ItemDataRole.EditRole:
            return tag

    def flags(self, index) -> Qt.ItemFlag:
        """Make the tags editable."""
        return (Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEditable
                | Qt.ItemFlag.ItemIsEnabled)

    def setData(self, index, value: str,
                role=Qt.ItemDataRole.EditRole) -> bool:
        new_tag = value
        if not new_tag or role != Qt.ItemDataRole.EditRole:
            return False
        old_tag = self.data(index, Qt.ItemDataRole.EditRole)
        if new_tag == old_tag:
            return False
        selected_indices = self.all_tags_list.selectedIndexes()
        old_tags = []
        old_tags_count = 0
        for selected_index in selected_indices:
            old_tag, old_tag_count = selected_index.data(
                Qt.ItemDataRole.UserRole)
            old_tags.append(old_tag)
            old_tags_count += old_tag_count
        question = (f'Rename {old_tags_count} '
                    f'{pluralize("instance", old_tags_count)} of ')
        if len(old_tags) < 10:
            quoted_tags = [f'"{tag}"' for tag in old_tags]
            question += (f'{pluralize("tag", len(old_tags))} '
                         f'{list_with_and(quoted_tags)} ')
        else:
            question += f'{len(old_tags)} tags '
        question += f'to "{new_tag}"?'
        reply = get_confirmation_dialog_reply(
            title=f'Rename {pluralize("Tag", len(old_tags))}',
            question=question)
        if reply == QMessageBox.StandardButton.Yes:
            self.tags_renaming_requested.emit(old_tags, new_tag)
            return True
        return False

    @Slot()
    def count_tags(self, images: list[Image]):
        """Recount every tag from scratch (used on directory load / reset)."""
        # `beginResetModel` has to come before the data changes: between the
        # two calls a view may still read the old contents, and it must never
        # see new rows under the old row count.
        self._is_publish_pending = False
        self._publish_timer.stop()
        self.beginResetModel()
        self.tag_counter = Counter()
        self.counted_tags = {}
        for image in images:
            self.tag_counter.update(image.tags)
            self.counted_tags[image.path] = list(image.tags)
        self.most_common_tags = self.tag_counter.most_common()
        self.endResetModel()

    def update_tag_counts(self, images: list[Image], first_row: int,
                          last_row: int):
        """
        Incrementally update the tag counts for the images in the given row
        range, applying only the difference between each image's previous and
        current tags. This avoids an O(dataset) recount on every edit.
        """
        # Find the changed rows before touching anything, so nothing is
        # published when there is really nothing to apply.
        changed_rows = [
            row for row in range(first_row, min(last_row, len(images) - 1) + 1)
            if self.counted_tags.get(images[row].path, []) != images[row].tags
        ]
        if not changed_rows:
            return
        for row in changed_rows:
            image = images[row]
            old_tags = self.counted_tags.get(image.path, [])
            if old_tags:
                self.tag_counter.subtract(old_tags)
                # Drop tags that fell to zero. Rebuilding the whole Counter
                # with `+counter` would cost a full copy on every edit.
                for tag in set(old_tags):
                    if self.tag_counter[tag] <= 0:
                        del self.tag_counter[tag]
            self.tag_counter.update(image.tags)
            self.counted_tags[image.path] = list(image.tags)
        self._schedule_publish()

    def _schedule_publish(self):
        """
        Ranking every tag and resetting the view is the expensive part, and a
        batch captioning run reports one changed row at a time. Coalescing on
        a zero-delay timer turns a run of M edits into one ranking pass once
        the event loop comes back round, instead of M.
        """
        self._is_publish_pending = True
        if not self._publish_timer.isActive():
            self._publish_timer.start()

    @Slot()
    def publish_pending_counts(self):
        """Apply any coalesced updates now."""
        if not self._is_publish_pending:
            return
        self._is_publish_pending = False
        self._publish_timer.stop()
        self.beginResetModel()
        self.most_common_tags = self.tag_counter.most_common()
        self.endResetModel()