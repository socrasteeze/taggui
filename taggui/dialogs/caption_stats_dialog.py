from collections import Counter

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QLabel, QPushButton, QTextEdit,
                               QVBoxLayout)

from models.image_list_model import ImageListModel
from utils.caption_profiles import get_profile_config
from utils.settings import DEFAULT_SETTINGS, get_settings


class CaptionStatsDialog(QDialog):
    def __init__(self, parent, image_list_model: ImageListModel,
                 tokenizer=None, tag_separator: str = ', '):
        super().__init__(parent)
        self.setWindowTitle('Caption Stats')
        self.resize(520, 480)
        layout = QVBoxLayout(self)
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        layout.addWidget(self.text_edit)
        close_button = QPushButton('Close')
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignRight)

        settings = get_settings()
        profile_name = settings.value(
            'caption_profile',
            defaultValue=DEFAULT_SETTINGS['caption_profile'], type=str)
        profile = get_profile_config(profile_name)
        trigger = settings.value(
            'trigger_token',
            defaultValue=DEFAULT_SETTINGS['trigger_token'], type=str).strip()

        images = image_list_model.images
        total = len(images)
        tagged = sum(1 for image in images if image.tags)
        untagged = total - tagged
        tag_counts = [len(image.tags) for image in images]
        token_counts = []
        for image in images:
            caption = tag_separator.join(image.tags)
            if tokenizer is not None and caption:
                count = len(tokenizer(caption).input_ids)
                if count >= 2:
                    count -= 2
            else:
                count = len(caption.split()) if caption else 0
            token_counts.append(count)

        trigger_hits = 0
        if trigger:
            for image in images:
                if any(trigger in tag for tag in image.tags):
                    trigger_hits += 1

        over_limit = sum(1 for count in token_counts
                         if count > profile.token_limit)
        avg_tags = (sum(tag_counts) / total) if total else 0
        avg_tokens = (sum(token_counts) / total) if total else 0
        top_tags = Counter(
            tag for image in images for tag in image.tags).most_common(20)

        lines = [
            f'Caption profile: {profile.profile.value}',
            f'Token encoder: {profile.encoder.value} '
            f'(limit {profile.token_limit})',
            '',
            f'Images: {total}',
            f'Tagged: {tagged} ({(tagged / total * 100) if total else 0:.1f}%)',
            f'Untagged: {untagged}',
            f'Average tags/image: {avg_tags:.1f}',
            f'Average tokens/image: {avg_tokens:.1f}',
            f'Over token limit: {over_limit}',
        ]
        if trigger:
            lines.append(
                f'Trigger "{trigger}" present: {trigger_hits}/'
                f'{total} ({(trigger_hits / total * 100) if total else 0:.1f}%)')
        lines.append('')
        lines.append('Top tags:')
        for tag, count in top_tags:
            lines.append(f'  {tag}: {count}')
        self.text_edit.setPlainText('\n'.join(lines))
