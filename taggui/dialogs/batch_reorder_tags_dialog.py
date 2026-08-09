import re

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import QDialog, QHBoxLayout, QPushButton, QVBoxLayout

from models.image_list_model import ImageListModel
from models.tag_counter_model import TagCounterModel
from utils.settings_widgets import SettingsBigCheckBox, SettingsLineEdit
from widgets.auto_captioner import HorizontalLine


class BatchReorderTagsDialog(QDialog):
    reorder_illustrious_requested = Signal(bool)

    def __init__(self, parent, image_list_model: ImageListModel,
                 tag_counter_model: TagCounterModel):
        super().__init__(parent)
        self.image_list_model = image_list_model
        self.setWindowTitle('Batch Reorder Tags')
        layout = QVBoxLayout(self)
        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(20, 20, 20, 20)
        top_layout.setSpacing(20)
        do_not_reorder_first_tag_check_box = SettingsBigCheckBox(
            key='do_not_reorder_first_tag', default=True)
        do_not_reorder_first_tag_check_box.setText('Do not reorder first tag')
        top_layout.addWidget(do_not_reorder_first_tag_check_box)
        top_buttons_layout = QVBoxLayout()
        top_buttons_layout.setSpacing(20)
        sort_alphabetically_button = QPushButton('Sort Tags Alphabetically')
        sort_alphabetically_button.clicked.connect(
            lambda: self.image_list_model.sort_tags_alphabetically(
                do_not_reorder_first_tag_check_box.isChecked()))
        top_buttons_layout.addWidget(sort_alphabetically_button)
        sort_by_frequency_button = QPushButton('Sort Tags by Frequency')
        sort_by_frequency_button.clicked.connect(
            lambda: self.image_list_model.sort_tags_by_frequency(
                tag_counter_model.tag_counter,
                do_not_reorder_first_tag_check_box.isChecked()))
        top_buttons_layout.addWidget(sort_by_frequency_button)
        reverse_button = QPushButton('Reverse Order of Tags')
        reverse_button.clicked.connect(
            lambda: self.image_list_model.reverse_tags_order(
                do_not_reorder_first_tag_check_box.isChecked()))
        top_buttons_layout.addWidget(reverse_button)
        shuffle_button = QPushButton('Shuffle Tags Randomly')
        shuffle_button.clicked.connect(
            lambda: self.image_list_model.shuffle_tags(
                do_not_reorder_first_tag_check_box.isChecked()))
        top_buttons_layout.addWidget(shuffle_button)
        top_layout.addLayout(top_buttons_layout)
        horizontal_line = HorizontalLine()
        bottom_layout = QVBoxLayout()
        bottom_layout.setContentsMargins(20, 20, 20, 20)
        bottom_layout.setSpacing(20)
        self.move_tags_line_edit = SettingsLineEdit(key='move_to_front_tags')
        self.move_tags_line_edit.setPlaceholderText('Tags to move '
                                                    '(comma-separated)')
        self.move_tags_line_edit.setClearButtonEnabled(True)
        self.move_tags_line_edit.textChanged.connect(
            self._update_move_buttons)
        self.move_tags_button = QPushButton('Move Tags to Front')
        self.move_tags_button.setEnabled(False)
        self.move_tags_button.clicked.connect(self.move_tags_to_front)
        self.move_tags_back_button = QPushButton('Move Tags to Back')
        self.move_tags_back_button.setEnabled(False)
        self.move_tags_back_button.clicked.connect(self.move_tags_to_back)
        illustrious_button = QPushButton('Illustrious Order '
                                         '(count→char→series→general)')
        # Routed through the window so the reorder gets the character and
        # series tags from the loaded vocabulary; without them it can only
        # promote count tags.
        illustrious_button.clicked.connect(
            lambda: self.reorder_illustrious_requested.emit(
                do_not_reorder_first_tag_check_box.isChecked()))
        bottom_layout.addWidget(self.move_tags_line_edit)
        bottom_layout.addWidget(self.move_tags_button)
        bottom_layout.addWidget(self.move_tags_back_button)
        bottom_layout.addWidget(illustrious_button)
        layout.addLayout(top_layout)
        layout.addWidget(horizontal_line)
        layout.addLayout(bottom_layout)

        self.move_tags_line_edit.textChanged.emit(
            self.move_tags_line_edit.text())

    def _update_move_buttons(self):
        enabled = bool(self.move_tags_line_edit.text())
        self.move_tags_button.setEnabled(enabled)
        self.move_tags_back_button.setEnabled(enabled)

    def _parsed_move_tags(self) -> list[str]:
        tags = re.split(r'(?<!\\),', self.move_tags_line_edit.text())
        return [tag.strip().replace(r'\,', ',') for tag in tags]

    @Slot()
    def move_tags_to_front(self):
        self.image_list_model.move_tags_to_front(self._parsed_move_tags())

    @Slot()
    def move_tags_to_back(self):
        self.image_list_model.move_tags_to_back(self._parsed_move_tags())
