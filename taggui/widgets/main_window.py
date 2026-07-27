from pathlib import Path

from PySide6.QtCore import (QKeyCombination, QModelIndex, QTimer, QUrl, Qt,
                            QThread, Signal, Slot)
from PySide6.QtGui import (QAction, QActionGroup, QCloseEvent, QDesktopServices,
                           QIcon, QKeySequence, QPixmap, QShortcut)
from PySide6.QtWidgets import (QApplication, QFileDialog, QMainWindow,
                               QMessageBox, QProgressDialog, QStackedWidget,
                               QVBoxLayout, QWidget)

from dialogs.batch_reorder_tags_dialog import BatchReorderTagsDialog
from dialogs.bucket_calculator_dialog import BucketCalculatorDialog
from dialogs.caption_stats_dialog import CaptionStatsDialog
from dialogs.create_shortcut_dialog import CreateShortcutDialog
from dialogs.find_and_replace_dialog import FindAndReplaceDialog
from dialogs.settings_dialog import SettingsDialog
from dialogs.trigger_token_dialog import TriggerTokenDialog
from models.image_list_model import ImageListModel
from models.image_tag_list_model import ImageTagListModel
from models.proxy_image_list_model import ProxyImageListModel
from models.tag_counter_model import TagCounterModel
from utils.big_widgets import BigPushButton
from utils.caption_profiles import CaptionProfile, get_profile_config
from utils.image import Image
from utils.key_press_forwarder import KeyPressForwarder
from utils.settings import DEFAULT_SETTINGS, get_settings, get_tag_separator
from utils.shortcut_remover import ShortcutRemover
from utils.tag_vocab import (TagVocab, download_tag_list, ensure_vocab_file,
                             get_tags_directory)
from utils.utils import get_resource_path, pluralize
from widgets.all_tags_editor import AllTagsEditor
from widgets.auto_captioner import AutoCaptioner
from widgets.image_list import ImageList
from widgets.image_tags_editor import ImageTagsEditor
from widgets.image_viewer import ImageViewer

ICON_PATH = Path('images/icon.ico')
GITHUB_REPOSITORY_URL = 'https://github.com/jhc13/taggui'
TOKENIZER_DIRECTORY_PATH = Path('clip-vit-base-patch32')


class TokenizerLoadWorker(QThread):
    loaded = Signal(object)

    def __init__(self, profile_name: str):
        super().__init__()
        self.profile_name = profile_name

    def run(self):
        from transformers import AutoTokenizer
        profile = get_profile_config(self.profile_name)
        try:
            if profile.encoder.value == 'clip':
                tokenizer = AutoTokenizer.from_pretrained(
                    get_resource_path(TOKENIZER_DIRECTORY_PATH))
            elif profile.encoder.value == 't5':
                tokenizer = AutoTokenizer.from_pretrained(
                    'google/t5-v1_1-base')
            else:
                # Approximate Qwen token counts with a fast GPT-2 tokenizer
                # until a dedicated Qwen tokenizer is cached locally.
                tokenizer = AutoTokenizer.from_pretrained('gpt2')
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(
                get_resource_path(TOKENIZER_DIRECTORY_PATH))
        self.loaded.emit(tokenizer)


class MainWindow(QMainWindow):
    def __init__(self, app: QApplication):
        super().__init__()
        self.app = app
        self.settings = get_settings()
        self.directory_path = None
        self._pending_select_index = 0
        self._load_progress_dialog: QProgressDialog | None = None
        self.tokenizer = None
        self._tokenizer_worker: TokenizerLoadWorker | None = None
        self.vocab = TagVocab()
        image_list_image_width = self.settings.value(
            'image_list_image_width',
            defaultValue=DEFAULT_SETTINGS['image_list_image_width'], type=int)
        tag_separator = get_tag_separator()
        self.image_list_model = ImageListModel(image_list_image_width,
                                               tag_separator)
        self.proxy_image_list_model = ProxyImageListModel(
            self.image_list_model, None, tag_separator)
        self.image_list_model.proxy_image_list_model = (
            self.proxy_image_list_model)
        self.tag_counter_model = TagCounterModel()
        self.image_tag_list_model = ImageTagListModel()

        self.setWindowIcon(QIcon(QPixmap(get_resource_path(ICON_PATH))))
        self.setPalette(self.app.style().standardPalette())
        self.set_font_size()
        self.image_viewer = ImageViewer(self.proxy_image_list_model)
        self.create_central_widget()
        self.image_list = ImageList(self.proxy_image_list_model,
                                    tag_separator, image_list_image_width)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea,
                           self.image_list)
        self.image_tags_editor = ImageTagsEditor(
            self.proxy_image_list_model, self.tag_counter_model,
            self.image_tag_list_model, self.image_list, None,
            tag_separator, self.vocab)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea,
                           self.image_tags_editor)
        self.all_tags_editor = AllTagsEditor(self.tag_counter_model)
        self.tag_counter_model.all_tags_list = (self.all_tags_editor
                                                .all_tags_list)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea,
                           self.all_tags_editor)
        self.auto_captioner = AutoCaptioner(self.image_list_model,
                                            self.image_list)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea,
                           self.auto_captioner)
        self.tabifyDockWidget(self.all_tags_editor, self.auto_captioner)
        self.all_tags_editor.raise_()
        self.resize(image_list_image_width * 8,
                    int(image_list_image_width * 4.5))
        self.resizeDocks([self.image_list, self.image_tags_editor,
                          self.all_tags_editor],
                         [int(image_list_image_width * 2.5)] * 3,
                         Qt.Orientation.Horizontal)
        self.image_tags_editor.tag_input_box.setDisabled(True)
        self.auto_captioner.start_cancel_button.setDisabled(True)
        self.reload_directory_action = QAction('Reload Directory', parent=self)
        self.reload_directory_action.setDisabled(True)
        self.undo_action = QAction('Undo', parent=self)
        self.redo_action = QAction('Redo', parent=self)
        self.toggle_image_list_action = QAction('Images', parent=self)
        self.toggle_image_tags_editor_action = QAction('Image Tags',
                                                       parent=self)
        self.toggle_all_tags_editor_action = QAction('All Tags', parent=self)
        self.toggle_auto_captioner_action = QAction('Auto-Captioner',
                                                    parent=self)
        self.create_menus()

        self.image_list_selection_model = (self.image_list.list_view
                                           .selectionModel())
        self.image_list_model.image_list_selection_model = (
            self.image_list_selection_model)
        self._filter_timer = QTimer(self)
        self._filter_timer.setSingleShot(True)
        self._filter_timer.setInterval(200)
        self._filter_timer.timeout.connect(self.set_image_list_filter)
        self.connect_image_list_signals()
        self.connect_image_tags_editor_signals()
        self.connect_all_tags_editor_signals()
        self.connect_auto_captioner_signals()
        self.image_list_model.load_progress.connect(self._on_load_progress)
        self.image_list_model.load_finished.connect(self._on_load_finished)
        self.image_list_model.load_failed.connect(self._on_load_failed)

        key_press_forwarder = KeyPressForwarder(
            parent=self, target=self.image_list.list_view,
            keys_to_forward=(Qt.Key.Key_Up, Qt.Key.Key_Down, Qt.Key.Key_PageUp,
                             Qt.Key.Key_PageDown, Qt.Key.Key_Home,
                             Qt.Key.Key_End))
        self.installEventFilter(key_press_forwarder)
        ctrl_z = QKeyCombination(Qt.KeyboardModifier.ControlModifier,
                                 key=Qt.Key.Key_Z)
        ctrl_y = QKeyCombination(Qt.KeyboardModifier.ControlModifier,
                                 key=Qt.Key.Key_Y)
        shortcut_remover = ShortcutRemover(parent=self,
                                           shortcuts=(ctrl_z, ctrl_y))
        self.image_list.filter_line_edit.installEventFilter(shortcut_remover)
        self.image_tags_editor.tag_input_box.installEventFilter(
            shortcut_remover)
        self.all_tags_editor.filter_line_edit.installEventFilter(
            shortcut_remover)
        focus_filter_images_box_shortcut = QShortcut(
            QKeySequence('Alt+F'), self)
        focus_filter_images_box_shortcut.activated.connect(
            self.image_list.raise_)
        focus_filter_images_box_shortcut.activated.connect(
            self.image_list.filter_line_edit.setFocus)
        focus_add_tag_box_shortcut = QShortcut(QKeySequence('Alt+A'), self)
        focus_add_tag_box_shortcut.activated.connect(
            self.image_tags_editor.raise_)
        focus_add_tag_box_shortcut.activated.connect(
            self.image_tags_editor.tag_input_box.setFocus)
        focus_image_tags_list_shortcut = QShortcut(QKeySequence('Alt+I'), self)
        focus_image_tags_list_shortcut.activated.connect(
            self.image_tags_editor.raise_)
        focus_image_tags_list_shortcut.activated.connect(
            self.image_tags_editor.image_tags_list.setFocus)
        focus_image_tags_list_shortcut.activated.connect(
            self.image_tags_editor.select_first_tag)
        focus_search_tags_box_shortcut = QShortcut(QKeySequence('Alt+S'), self)
        focus_search_tags_box_shortcut.activated.connect(
            self.all_tags_editor.raise_)
        focus_search_tags_box_shortcut.activated.connect(
            self.all_tags_editor.filter_line_edit.setFocus)
        focus_caption_button_shortcut = QShortcut(QKeySequence('Alt+C'), self)
        focus_caption_button_shortcut.activated.connect(
            self.auto_captioner.raise_)
        focus_caption_button_shortcut.activated.connect(
            self.auto_captioner.start_cancel_button.setFocus)
        go_to_previous_image_shortcut = QShortcut(QKeySequence('Ctrl+Up'),
                                                  self)
        go_to_previous_image_shortcut.activated.connect(
            self.image_list.go_to_previous_image)
        go_to_next_image_shortcut = QShortcut(QKeySequence('Ctrl+Down'), self)
        go_to_next_image_shortcut.activated.connect(
            self.image_list.go_to_next_image)
        jump_to_first_untagged_image_shortcut = QShortcut(
            QKeySequence('Ctrl+J'), self)
        jump_to_first_untagged_image_shortcut.activated.connect(
            self.image_list.jump_to_first_untagged_image)
        self.apply_image_list_view_mode()
        self.reload_vocab_for_profile()
        self.start_tokenizer_load()
        self.restore()
        self.image_tags_editor.tag_input_box.setFocus()

    def closeEvent(self, event: QCloseEvent):
        self.settings.setValue('geometry', self.saveGeometry())
        self.settings.setValue('window_state', self.saveState())
        self.image_list_model.tag_writer.flush()
        super().closeEvent(event)

    def set_font_size(self):
        font = self.app.font()
        font_size = self.settings.value(
            'font_size', defaultValue=DEFAULT_SETTINGS['font_size'], type=int)
        font.setPointSize(font_size)
        self.app.setFont(font)

    def create_central_widget(self):
        central_widget = QStackedWidget()
        load_directory_widget = QWidget()
        load_directory_button = BigPushButton('Load Directory...')
        load_directory_button.clicked.connect(self.select_and_load_directory)
        QVBoxLayout(load_directory_widget).addWidget(
            load_directory_button, alignment=Qt.AlignmentFlag.AlignCenter)
        central_widget.addWidget(load_directory_widget)
        central_widget.addWidget(self.image_viewer)
        self.setCentralWidget(central_widget)

    def start_tokenizer_load(self):
        profile_name = self.settings.value(
            'caption_profile',
            defaultValue=DEFAULT_SETTINGS['caption_profile'], type=str)
        self._tokenizer_worker = TokenizerLoadWorker(profile_name)
        self._tokenizer_worker.loaded.connect(self._on_tokenizer_loaded)
        self._tokenizer_worker.start()

    @Slot(object)
    def _on_tokenizer_loaded(self, tokenizer):
        self.tokenizer = tokenizer
        self.proxy_image_list_model.set_tokenizer(tokenizer)
        self.image_tags_editor.set_tokenizer(tokenizer)

    def reload_vocab_for_profile(self):
        profile = get_profile_config(self.settings.value(
            'caption_profile',
            defaultValue=DEFAULT_SETTINGS['caption_profile'], type=str))
        self.vocab.clear()
        if not profile.vocab_csv:
            return
        if profile.vocab_csv == 'sdxl_quality.csv':
            self.vocab.load_builtin_sdxl()
            return
        path = ensure_vocab_file(profile.vocab_csv)
        if path and path.is_file():
            try:
                self.vocab.load_csv(path)
            except Exception as exception:
                print(f'Failed to load tag vocab {path}: {exception}')

    def load_directory(self, path: Path, select_index: int = 0,
                       save_path_to_settings: bool = False):
        self.directory_path = path.resolve()
        if save_path_to_settings:
            self.settings.setValue('directory_path', str(self.directory_path))
        self.setWindowTitle(path.name)
        self._pending_select_index = select_index
        self._load_progress_dialog = QProgressDialog(
            'Loading images...', 'Cancel', 0, 0, self)
        self._load_progress_dialog.setWindowTitle('Load Directory')
        self._load_progress_dialog.setWindowModality(
            Qt.WindowModality.WindowModal)
        self._load_progress_dialog.setMinimumDuration(0)
        self._load_progress_dialog.setValue(0)
        self.image_list_model.load_directory(path)
        self.image_list.filter_line_edit.clear()
        self.all_tags_editor.filter_line_edit.clear()

    @Slot(int, int)
    def _on_load_progress(self, completed: int, total: int):
        # `QProgressDialog.setValue()` calls `QCoreApplication.processEvents()`
        # for modal dialogs, which can deliver the queued `load_finished`
        # signal and close the dialog in the middle of this method. Hold a
        # local reference and update the value last so that the dialog is
        # never touched through `self` after the event loop has been pumped.
        progress_dialog = self._load_progress_dialog
        if progress_dialog is None:
            return
        if progress_dialog.maximum() != total:
            progress_dialog.setMaximum(max(total, 1))
        progress_dialog.setLabelText(f'Loading images... {completed}/{total}')
        progress_dialog.setValue(completed)

    @Slot()
    def _on_load_finished(self):
        if self._load_progress_dialog is not None:
            self._load_progress_dialog.close()
            self._load_progress_dialog = None
        select_index = self._pending_select_index
        if select_index >= self.proxy_image_list_model.rowCount():
            select_index = max(self.proxy_image_list_model.rowCount() - 1, 0)
        self.image_list_selection_model.clearCurrentIndex()
        self.image_list.list_view.setCurrentIndex(
            self.proxy_image_list_model.index(select_index, 0))
        self.centralWidget().setCurrentWidget(self.image_viewer)
        self.reload_directory_action.setDisabled(False)
        self.image_tags_editor.tag_input_box.setDisabled(False)
        self.auto_captioner.start_cancel_button.setDisabled(False)

    @Slot(str)
    def _on_load_failed(self, message: str):
        if self._load_progress_dialog is not None:
            self._load_progress_dialog.close()
            self._load_progress_dialog = None
        QMessageBox.critical(self, 'Load Directory',
                             f'Failed to load directory:\n{message}')

    @Slot()
    def select_and_load_directory(self):
        initial_directory = (str(self.directory_path)
                             if self.directory_path else '')
        load_directory_path = QFileDialog.getExistingDirectory(
            parent=self, caption='Select directory to load images from',
            dir=initial_directory)
        if not load_directory_path:
            return
        self.load_directory(Path(load_directory_path),
                            save_path_to_settings=True)

    @Slot()
    def reload_directory(self):
        filter_text = self.image_list.filter_line_edit.text()
        select_index_key = ('image_index'
                            if self.proxy_image_list_model.filter is None
                            else 'filtered_image_index')
        select_index = self.settings.value(select_index_key, type=int) or 0
        self.load_directory(self.directory_path)
        self.image_list.filter_line_edit.setText(filter_text)
        self._pending_select_index = select_index

    @Slot()
    def show_settings_dialog(self):
        settings_dialog = SettingsDialog(parent=self)
        settings_dialog.exec()
        self.image_tags_editor.apply_caption_profile()
        self.reload_vocab_for_profile()
        self.start_tokenizer_load()
        self.apply_image_list_view_mode()

    @Slot()
    def show_find_and_replace_dialog(self):
        find_and_replace_dialog = FindAndReplaceDialog(
            parent=self, image_list_model=self.image_list_model)
        find_and_replace_dialog.exec()

    @Slot()
    def show_batch_reorder_tags_dialog(self):
        batch_reorder_tags_dialog = BatchReorderTagsDialog(
            parent=self, image_list_model=self.image_list_model,
            tag_counter_model=self.tag_counter_model)
        batch_reorder_tags_dialog.exec()

    @Slot()
    def show_bucket_calculator_dialog(self):
        bucket_calculator_dialog = BucketCalculatorDialog(
            parent=self, image_list_model=self.image_list_model,
            directory_path=self.directory_path)
        bucket_calculator_dialog.exec()

    @Slot()
    def show_create_shortcut_dialog(self):
        dialog = CreateShortcutDialog(parent=self)
        dialog.exec()

    @Slot()
    def show_caption_stats_dialog(self):
        dialog = CaptionStatsDialog(
            parent=self, image_list_model=self.image_list_model,
            tokenizer=self.tokenizer,
            tag_separator=get_tag_separator())
        dialog.exec()

    @Slot()
    def show_trigger_token_dialog(self):
        dialog = TriggerTokenDialog(parent=self,
                                    image_list_model=self.image_list_model)
        dialog.exec()

    @Slot()
    def reorder_illustrious_tags(self):
        self.image_list_model.reorder_illustrious_tags()

    @Slot()
    def export_jsonl(self):
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export JSONL',
            str((self.directory_path or Path('.')) / 'captions.jsonl'),
            'JSONL (*.jsonl)')
        if not path:
            return
        count = self.image_list_model.export_jsonl(Path(path))
        QMessageBox.information(self, 'Export JSONL',
                                f'Exported {count} captions.')

    @Slot()
    def export_kohya_metadata(self):
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export Kohya Metadata JSON',
            str((self.directory_path or Path('.')) / 'meta_cap.json'),
            'JSON (*.json)')
        if not path:
            return
        count = self.image_list_model.export_kohya_metadata(Path(path))
        QMessageBox.information(self, 'Export Kohya Metadata',
                                f'Exported metadata for {count} images.')

    @Slot()
    def update_tag_lists(self):
        try:
            downloaded = []
            for filename in ('danbooru.csv', 'e621.csv'):
                path = download_tag_list(filename)
                downloaded.append(str(path))
            self.reload_vocab_for_profile()
            QMessageBox.information(
                self, 'Update Tag Lists',
                'Downloaded:\n' + '\n'.join(downloaded) +
                f'\n\nCache: {get_tags_directory()}')
        except Exception as exception:
            QMessageBox.critical(self, 'Update Tag Lists',
                                 f'Failed to update tag lists:\n{exception}')

    @Slot(str)
    def set_caption_profile(self, profile_name: str):
        self.settings.setValue('caption_profile', profile_name)
        self.image_tags_editor.apply_caption_profile()
        self.reload_vocab_for_profile()
        self.start_tokenizer_load()

    def apply_image_list_view_mode(self):
        mode = self.settings.value(
            'image_list_view_mode',
            defaultValue=DEFAULT_SETTINGS['image_list_view_mode'], type=str)
        self.image_list.set_view_mode(mode)

    @Slot()
    def set_list_view_mode(self):
        self.settings.setValue('image_list_view_mode', 'list')
        self.apply_image_list_view_mode()

    @Slot()
    def set_grid_view_mode(self):
        self.settings.setValue('image_list_view_mode', 'grid')
        self.apply_image_list_view_mode()

    @Slot()
    def remove_duplicate_tags(self):
        removed_tag_count = self.image_list_model.remove_duplicate_tags()
        message_box = QMessageBox()
        message_box.setWindowTitle('Remove Duplicate Tags')
        message_box.setIcon(QMessageBox.Icon.Information)
        if not removed_tag_count:
            text = 'No duplicate tags were found.'
        else:
            text = (f'Removed {removed_tag_count} duplicate '
                    f'{pluralize("tag", removed_tag_count)}.')
        message_box.setText(text)
        message_box.exec()

    @Slot()
    def remove_empty_tags(self):
        removed_tag_count = self.image_list_model.remove_empty_tags()
        message_box = QMessageBox()
        message_box.setWindowTitle('Remove Empty Tags')
        message_box.setIcon(QMessageBox.Icon.Information)
        if not removed_tag_count:
            text = 'No empty tags were found.'
        else:
            text = (f'Removed {removed_tag_count} empty '
                    f'{pluralize("tag", removed_tag_count)}.')
        message_box.setText(text)
        message_box.exec()

    def create_menus(self):
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu('File')
        load_directory_action = QAction('Load Directory...', parent=self)
        load_directory_action.setShortcut(QKeySequence('Ctrl+L'))
        load_directory_action.triggered.connect(self.select_and_load_directory)
        file_menu.addAction(load_directory_action)
        self.reload_directory_action.setShortcuts(
            [QKeySequence('Ctrl+Shift+L'), QKeySequence('F5')])
        self.reload_directory_action.triggered.connect(self.reload_directory)
        file_menu.addAction(self.reload_directory_action)
        export_jsonl_action = QAction('Export JSONL...', parent=self)
        export_jsonl_action.triggered.connect(self.export_jsonl)
        file_menu.addAction(export_jsonl_action)
        export_kohya_action = QAction('Export Kohya Metadata JSON...',
                                      parent=self)
        export_kohya_action.triggered.connect(self.export_kohya_metadata)
        file_menu.addAction(export_kohya_action)
        settings_action = QAction('Settings...', parent=self)
        settings_action.setShortcut(QKeySequence('Ctrl+Alt+S'))
        settings_action.triggered.connect(self.show_settings_dialog)
        file_menu.addAction(settings_action)
        exit_action = QAction('Exit', parent=self)
        exit_action.setShortcut(QKeySequence('Ctrl+W'))
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        edit_menu = menu_bar.addMenu('Edit')
        self.undo_action.setShortcut(QKeySequence('Ctrl+Z'))
        self.undo_action.triggered.connect(self.image_list_model.undo)
        self.undo_action.setDisabled(True)
        edit_menu.addAction(self.undo_action)
        self.redo_action.setShortcut(QKeySequence('Ctrl+Y'))
        self.redo_action.triggered.connect(self.image_list_model.redo)
        self.redo_action.setDisabled(True)
        edit_menu.addAction(self.redo_action)
        find_and_replace_action = QAction('Find and Replace...', parent=self)
        find_and_replace_action.setShortcut(QKeySequence('Ctrl+R'))
        find_and_replace_action.triggered.connect(
            self.show_find_and_replace_dialog)
        edit_menu.addAction(find_and_replace_action)
        batch_reorder_tags_action = QAction('Batch Reorder Tags...',
                                            parent=self)
        batch_reorder_tags_action.setShortcut(QKeySequence('Ctrl+B'))
        batch_reorder_tags_action.triggered.connect(
            self.show_batch_reorder_tags_dialog)
        edit_menu.addAction(batch_reorder_tags_action)
        remove_duplicate_tags_action = QAction('Remove Duplicate Tags',
                                               parent=self)
        remove_duplicate_tags_action.setShortcut(QKeySequence('Ctrl+D'))
        remove_duplicate_tags_action.triggered.connect(
            self.remove_duplicate_tags)
        edit_menu.addAction(remove_duplicate_tags_action)
        remove_empty_tags_action = QAction('Remove Empty Tags', parent=self)
        remove_empty_tags_action.setShortcut(QKeySequence('Ctrl+E'))
        remove_empty_tags_action.triggered.connect(
            self.remove_empty_tags)
        edit_menu.addAction(remove_empty_tags_action)
        trigger_action = QAction('Insert Trigger Token...', parent=self)
        trigger_action.triggered.connect(self.show_trigger_token_dialog)
        edit_menu.addAction(trigger_action)
        illustrious_reorder_action = QAction(
            'Reorder Tags (Illustrious)', parent=self)
        illustrious_reorder_action.triggered.connect(
            self.reorder_illustrious_tags)
        edit_menu.addAction(illustrious_reorder_action)

        tools_menu = menu_bar.addMenu('Tools')
        bucket_calculator_action = QAction('Aspect Ratio Bucket Calculator...',
                                           parent=self)
        bucket_calculator_action.triggered.connect(
            self.show_bucket_calculator_dialog)
        tools_menu.addAction(bucket_calculator_action)
        stats_action = QAction('Caption Stats...', parent=self)
        stats_action.triggered.connect(self.show_caption_stats_dialog)
        tools_menu.addAction(stats_action)
        update_tags_action = QAction('Update Tag Lists...', parent=self)
        update_tags_action.triggered.connect(self.update_tag_lists)
        tools_menu.addAction(update_tags_action)
        create_shortcut_action = QAction('Create Desktop Shortcut...',
                                         parent=self)
        create_shortcut_action.triggered.connect(
            self.show_create_shortcut_dialog)
        tools_menu.addAction(create_shortcut_action)

        profile_menu = tools_menu.addMenu('Caption Profile')
        profile_group = QActionGroup(self)
        profile_group.setExclusive(True)
        current_profile = self.settings.value(
            'caption_profile',
            defaultValue=DEFAULT_SETTINGS['caption_profile'], type=str)
        for profile in CaptionProfile:
            action = QAction(profile.value, parent=self)
            action.setCheckable(True)
            action.setChecked(profile.value == current_profile)
            action.triggered.connect(
                lambda checked=False, name=profile.value:
                self.set_caption_profile(name))
            profile_group.addAction(action)
            profile_menu.addAction(action)

        view_menu = menu_bar.addMenu('View')
        self.toggle_image_list_action.setCheckable(True)
        self.toggle_image_tags_editor_action.setCheckable(True)
        self.toggle_all_tags_editor_action.setCheckable(True)
        self.toggle_auto_captioner_action.setCheckable(True)
        self.toggle_image_list_action.triggered.connect(
            lambda is_checked: self.image_list.setVisible(is_checked))
        self.toggle_image_tags_editor_action.triggered.connect(
            lambda is_checked: self.image_tags_editor.setVisible(is_checked))
        self.toggle_all_tags_editor_action.triggered.connect(
            lambda is_checked: self.all_tags_editor.setVisible(is_checked))
        self.toggle_auto_captioner_action.triggered.connect(
            lambda is_checked: self.auto_captioner.setVisible(is_checked))
        view_menu.addAction(self.toggle_image_list_action)
        view_menu.addAction(self.toggle_image_tags_editor_action)
        view_menu.addAction(self.toggle_all_tags_editor_action)
        view_menu.addAction(self.toggle_auto_captioner_action)
        view_menu.addSeparator()
        list_view_action = QAction('Image List View', parent=self)
        list_view_action.triggered.connect(self.set_list_view_mode)
        grid_view_action = QAction('Image Grid View', parent=self)
        grid_view_action.triggered.connect(self.set_grid_view_mode)
        view_menu.addAction(list_view_action)
        view_menu.addAction(grid_view_action)

        help_menu = menu_bar.addMenu('Help')
        open_github_repository_action = QAction('GitHub', parent=self)
        open_github_repository_action.triggered.connect(
            lambda: QDesktopServices.openUrl(QUrl(GITHUB_REPOSITORY_URL)))
        help_menu.addAction(open_github_repository_action)

    @Slot()
    def update_undo_and_redo_actions(self):
        if self.image_list_model.undo_stack:
            undo_action_name = self.image_list_model.undo_stack[-1].action_name
            self.undo_action.setText(f'Undo "{undo_action_name}"')
            self.undo_action.setDisabled(False)
        else:
            self.undo_action.setText('Undo')
            self.undo_action.setDisabled(True)
        if self.image_list_model.redo_stack:
            redo_action_name = self.image_list_model.redo_stack[-1].action_name
            self.redo_action.setText(f'Redo "{redo_action_name}"')
            self.redo_action.setDisabled(False)
        else:
            self.redo_action.setText('Redo')
            self.redo_action.setDisabled(True)

    @Slot()
    def set_image_list_filter(self):
        filter_ = self.image_list.filter_line_edit.parse_filter_text()
        self.proxy_image_list_model.filter = filter_
        self.proxy_image_list_model.invalidateFilter()
        if filter_ is None:
            all_tags_list_selection_model = (self.all_tags_editor
                                             .all_tags_list.selectionModel())
            all_tags_list_selection_model.clearSelection()
            self.all_tags_editor.all_tags_list.setCurrentIndex(QModelIndex())
            select_index = self.settings.value('image_index', type=int) or 0
            self.image_list.list_view.setCurrentIndex(
                self.proxy_image_list_model.index(select_index, 0))
        else:
            self.image_list.list_view.setCurrentIndex(
                self.proxy_image_list_model.index(0, 0))

    @Slot()
    def save_image_index(self, proxy_image_index: QModelIndex):
        settings_key = ('image_index'
                        if self.proxy_image_list_model.filter is None
                        else 'filtered_image_index')
        self.settings.setValue(settings_key, proxy_image_index.row())

    def connect_image_list_signals(self):
        self.image_list.filter_line_edit.textChanged.connect(
            self._filter_timer.start)
        self.image_list_selection_model.currentChanged.connect(
            self.save_image_index)
        self.image_list_selection_model.currentChanged.connect(
            self.image_list.update_image_index_label)
        self.image_list_selection_model.currentChanged.connect(
            self.image_viewer.load_image)
        self.image_list_selection_model.currentChanged.connect(
            self.image_tags_editor.load_image_tags)
        self.image_list_model.modelReset.connect(
            lambda: self.tag_counter_model.count_tags(
                self.image_list_model.images))
        self.image_list_model.dataChanged.connect(
            lambda top_left, bottom_right, _roles=None:
            self.tag_counter_model.update_tag_counts(
                self.image_list_model.images, top_left.row(),
                bottom_right.row()))
        self.image_list_model.dataChanged.connect(
            self.image_tags_editor.reload_image_tags_if_changed)
        self.image_list_model.update_undo_and_redo_actions_requested.connect(
            self.update_undo_and_redo_actions)
        self.proxy_image_list_model.rowsInserted.connect(
            lambda: self.image_list.update_image_index_label(
                self.image_list.list_view.currentIndex()))
        self.proxy_image_list_model.rowsRemoved.connect(
            lambda: self.image_list.update_image_index_label(
                self.image_list.list_view.currentIndex()))
        self.image_list.list_view.directory_reload_requested.connect(
            self.reload_directory)
        self.image_list.list_view.tags_paste_requested.connect(
            self.image_list_model.add_tags)
        self.image_list.visibilityChanged.connect(
            lambda: self.toggle_image_list_action.setChecked(
                self.image_list.isVisible()))

    @Slot()
    def update_image_tags(self):
        image_index = self.image_tags_editor.image_index
        if image_index is None:
            return
        image: Image = self.image_list_model.data(image_index,
                                                  Qt.ItemDataRole.UserRole)
        old_tags = image.tags
        new_tags = self.image_tag_list_model.stringList()
        if old_tags == new_tags:
            return
        old_tags_count = len(old_tags)
        new_tags_count = len(new_tags)
        row = image_index.row()
        if new_tags_count > old_tags_count:
            self.image_list_model.add_to_undo_stack(
                action_name='Add Tag', should_ask_for_confirmation=False,
                image_indices=[row])
        elif new_tags_count == old_tags_count:
            if set(new_tags) == set(old_tags):
                self.image_list_model.add_to_undo_stack(
                    action_name='Reorder Tags',
                    should_ask_for_confirmation=False,
                    image_indices=[row])
            else:
                self.image_list_model.add_to_undo_stack(
                    action_name='Rename Tag',
                    should_ask_for_confirmation=False,
                    image_indices=[row])
        elif old_tags_count - new_tags_count == 1:
            self.image_list_model.add_to_undo_stack(
                action_name='Delete Tag', should_ask_for_confirmation=False,
                image_indices=[row])
        else:
            self.image_list_model.add_to_undo_stack(
                action_name='Delete Tags', should_ask_for_confirmation=False,
                image_indices=[row])
        self.image_list_model.update_image_tags(image_index, new_tags)

    def connect_image_tags_editor_signals(self):
        self.image_tag_list_model.modelReset.connect(self.update_image_tags)
        self.image_tag_list_model.dataChanged.connect(self.update_image_tags)
        self.image_tag_list_model.rowsMoved.connect(self.update_image_tags)
        self.image_tags_editor.visibilityChanged.connect(
            lambda: self.toggle_image_tags_editor_action.setChecked(
                self.image_tags_editor.isVisible()))
        self.image_tags_editor.tag_input_box.tags_addition_requested.connect(
            self.image_list_model.add_tags)

    @Slot()
    def set_image_list_filter_text(self, selected_tag: str):
        escaped_selected_tag = (selected_tag.replace('\\', '\\\\')
                                .replace('"', r'\"').replace("'", r"\'"))
        self.image_list.filter_line_edit.setText(
            f'tag:"{escaped_selected_tag}"')

    @Slot(str)
    def add_tag_to_selected_images(self, tag: str):
        selected_image_indices = self.image_list.get_selected_image_indices()
        self.image_list_model.add_tags([tag], selected_image_indices)
        self.image_tags_editor.select_last_tag()

    def connect_all_tags_editor_signals(self):
        self.all_tags_editor.clear_filter_button.clicked.connect(
            self.image_list.filter_line_edit.clear)
        self.tag_counter_model.tags_renaming_requested.connect(
            self.image_list_model.rename_tags)
        self.tag_counter_model.tags_renaming_requested.connect(
            self.image_list.filter_line_edit.clear)
        self.all_tags_editor.all_tags_list.image_list_filter_requested.connect(
            self.set_image_list_filter_text)
        self.all_tags_editor.all_tags_list.tag_addition_requested.connect(
            self.add_tag_to_selected_images)
        self.all_tags_editor.all_tags_list.tags_deletion_requested.connect(
            self.image_list_model.delete_tags)
        self.all_tags_editor.all_tags_list.tags_deletion_requested.connect(
            self.image_list.filter_line_edit.clear)
        self.all_tags_editor.visibilityChanged.connect(
            lambda: self.toggle_all_tags_editor_action.setChecked(
                self.all_tags_editor.isVisible()))

    def connect_auto_captioner_signals(self):
        self.auto_captioner.caption_generated.connect(
            lambda image_index, _, tags:
            self.image_list_model.update_image_tags(image_index, tags))
        self.auto_captioner.caption_generated.connect(
            lambda image_index, *_:
            self.image_tags_editor.reload_image_tags_if_changed(image_index,
                                                                image_index))
        self.auto_captioner.visibilityChanged.connect(
            lambda: self.toggle_auto_captioner_action.setChecked(
                self.auto_captioner.isVisible()))

    def restore(self):
        if self.settings.contains('geometry'):
            self.restoreGeometry(self.settings.value('geometry', type=bytes))
        else:
            self.showMaximized()
        self.restoreState(self.settings.value('window_state', type=bytes))
        if self.settings.contains('image_index'):
            image_index = self.settings.value('image_index', type=int)
        else:
            image_index = 0
        if self.settings.contains('directory_path'):
            directory_path = Path(self.settings.value('directory_path',
                                                      type=str))
            if directory_path.is_dir():
                self.load_directory(directory_path, select_index=image_index)
