from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (QDialog, QFormLayout, QLabel, QPushButton,
                               QVBoxLayout)

from models.image_list_model import ImageListModel, Scope
from utils.caption_profiles import get_profile_config
from utils.settings import DEFAULT_SETTINGS, get_settings
from utils.settings_widgets import SettingsComboBox, SettingsLineEdit


class TriggerTokenDialog(QDialog):
    def __init__(self, parent, image_list_model: ImageListModel):
        super().__init__(parent)
        self.image_list_model = image_list_model
        self.settings = get_settings()
        self.setWindowTitle('Insert Trigger Token')
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        form = QFormLayout()
        self.trigger_line_edit = SettingsLineEdit(
            key='trigger_token',
            default=DEFAULT_SETTINGS['trigger_token'])
        self.trigger_line_edit.setClearButtonEnabled(True)
        form.addRow('Trigger token', self.trigger_line_edit)
        self.mode_combo_box = SettingsComboBox(key='trigger_insert_mode')
        self.mode_combo_box.addItems(['first_tag', 'embedded'])
        profile = get_profile_config(self.settings.value(
            'caption_profile',
            defaultValue=DEFAULT_SETTINGS['caption_profile'], type=str))
        if not self.settings.contains('trigger_insert_mode'):
            self.mode_combo_box.setCurrentText(profile.trigger_mode
                                               if profile.trigger_mode != 'none'
                                               else 'first_tag')
        form.addRow('Placement mode', self.mode_combo_box)
        self.scope_combo_box = SettingsComboBox(key='trigger_scope')
        self.scope_combo_box.addItems(list(Scope))
        form.addRow('Scope', self.scope_combo_box)
        layout.addLayout(form)
        hint = QLabel(
            'first_tag: prepend as its own tag (SDXL / Illustrious).\n'
            'embedded: prefix the first caption sentence (FLUX-family).')
        hint.setWordWrap(True)
        layout.addWidget(hint)
        insert_button = QPushButton('Insert Trigger')
        insert_button.clicked.connect(self.insert_trigger)
        layout.addWidget(insert_button)

    @Slot()
    def insert_trigger(self):
        trigger = self.trigger_line_edit.text().strip()
        if not trigger:
            return
        self.image_list_model.insert_trigger_token(
            trigger,
            mode=self.mode_combo_box.currentText(),
            scope=self.scope_combo_box.currentText())
        self.accept()
