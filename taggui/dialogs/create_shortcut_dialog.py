from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (QCheckBox, QDialog, QLabel, QMessageBox,
                               QPushButton, QVBoxLayout)

from utils.create_shortcut import create_taggui_shortcuts


class CreateShortcutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Create Shortcut')
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        description = QLabel(
            'Create a Windows shortcut (.lnk) that launches TagGUI via '
            'run.bat, using the app icon.')
        description.setWordWrap(True)
        layout.addWidget(description)
        self.desktop_check_box = QCheckBox('Desktop')
        self.desktop_check_box.setChecked(True)
        self.start_menu_check_box = QCheckBox('Start Menu')
        layout.addWidget(self.desktop_check_box)
        layout.addWidget(self.start_menu_check_box)
        create_button = QPushButton('Create Shortcut')
        create_button.clicked.connect(self.create_shortcut)
        layout.addWidget(create_button, alignment=Qt.AlignmentFlag.AlignRight)

    @Slot()
    def create_shortcut(self):
        desktop = self.desktop_check_box.isChecked()
        start_menu = self.start_menu_check_box.isChecked()
        if not desktop and not start_menu:
            QMessageBox.warning(self, 'Create Shortcut',
                                'Select at least one location.')
            return
        try:
            paths = create_taggui_shortcuts(desktop=desktop,
                                            start_menu=start_menu)
        except Exception as exception:
            QMessageBox.critical(self, 'Create Shortcut',
                                 f'Failed to create shortcut:\n{exception}')
            return
        listing = '\n'.join(str(path) for path in paths)
        QMessageBox.information(self, 'Create Shortcut',
                                f'Created:\n{listing}')
        self.accept()
