import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from time import perf_counter

from PIL import UnidentifiedImageError
from PySide6.QtCore import QModelIndex, QThread, Qt, Signal

from models.image_list_model import ImageListModel
from utils.enums import CaptionPosition
from utils.image import Image
from utils.settings import get_tag_separator

WD_DEFAULT_BATCH_SIZE = 8


def get_wd_batch_size(model, caption_settings: dict) -> int:
    """
    Images per inference call, clamped to what the model accepts. ONNX exports
    with a fixed batch dimension of 1 reject batched input, so honour that
    rather than failing at run time.
    """
    batch_size = caption_settings.get('wd_tagger_settings', {}).get(
        'batch_size', WD_DEFAULT_BATCH_SIZE)
    batch_size = max(1, int(batch_size))
    try:
        model_batch_dimension = (model.model.inference_session.get_inputs()[0]
                                 .shape[0])
    except Exception:
        return batch_size
    if isinstance(model_batch_dimension, int) and model_batch_dimension > 0:
        return min(batch_size, model_batch_dimension)
    return batch_size


def add_caption_to_tags(tags: list[str], caption: str,
                        caption_position: CaptionPosition) -> list[str]:
    if caption_position == CaptionPosition.DO_NOT_ADD or not caption:
        return tags
    tag_separator = get_tag_separator()
    new_tags = caption.split(tag_separator)
    tags = tags.copy()
    if caption_position == CaptionPosition.BEFORE_FIRST_TAG:
        tags[:0] = new_tags
    elif caption_position == CaptionPosition.AFTER_LAST_TAG:
        tags.extend(new_tags)
    elif caption_position == CaptionPosition.OVERWRITE_FIRST_TAG:
        if tags:
            tags[:1] = new_tags
        else:
            tags = new_tags
    elif caption_position == CaptionPosition.OVERWRITE_ALL_TAGS:
        tags = new_tags
    return tags


def format_duration(seconds: float) -> str:
    seconds_per_minute = 60
    seconds_per_hour = 60 * seconds_per_minute
    seconds_per_day = 24 * seconds_per_hour
    if seconds < seconds_per_minute:
        return f'{seconds:.1f} seconds'
    if seconds < seconds_per_hour:
        minutes = seconds / seconds_per_minute
        return f'{minutes:.1f} minutes'
    if seconds < seconds_per_day:
        hours = seconds / seconds_per_hour
        return f'{hours:.1f} hours'
    days = seconds / seconds_per_day
    return f'{days:.1f} days'


class CaptioningThread(QThread):
    text_outputted = Signal(str)
    clear_console_text_edit_requested = Signal()
    caption_generated = Signal(QModelIndex, str, list)
    progress_bar_update_requested = Signal(int)

    def __init__(self, parent, image_list_model: ImageListModel,
                 selected_image_indices: list[QModelIndex],
                 caption_settings: dict, tag_separator: str,
                 models_directory_path: Path | None):
        super().__init__(parent)
        self.image_list_model = image_list_model
        self.selected_image_indices = selected_image_indices
        self.caption_settings = caption_settings
        self.tag_separator = tag_separator
        self.models_directory_path = models_directory_path
        self.is_error = False
        self.is_canceled = False

    def _prepare_inputs(self, model, image_index):
        image: Image = self.image_list_model.data(image_index,
                                                  Qt.ItemDataRole.UserRole)
        image_prompt = model.get_image_prompt(image)
        model_inputs = model.get_model_inputs(image_prompt, image)
        return image, image_prompt, model_inputs

    def run_captioning(self):
        # Lazy imports avoid circular import with auto_captioning_model /
        # models_list during module initialization.
        from auto_captioning.models_list import (get_model_class,
                                                 is_group_separator,
                                                 is_tagger_model_id)

        model_id = self.caption_settings['model_id']
        if not model_id or is_group_separator(model_id):
            self.is_error = True
            self.clear_console_text_edit_requested.emit()
            print('Select a model to caption with.')
            return
        model_class = get_model_class(model_id)
        model = model_class(
            captioning_thread_=self, caption_settings=self.caption_settings)
        error_message = model.get_error_message()
        if error_message:
            self.is_error = True
            self.clear_console_text_edit_requested.emit()
            print(error_message)
            return
        model.load_processor_and_model()
        model.monkey_patch_after_loading()
        if self.is_canceled:
            print('Canceled captioning.')
            return
        self.clear_console_text_edit_requested.emit()
        selected_image_count = len(self.selected_image_indices)
        are_multiple_images_selected = selected_image_count > 1
        captioning_start_datetime = datetime.now()
        captioning_message = model.get_captioning_message(
            are_multiple_images_selected, captioning_start_datetime)
        print(captioning_message)
        caption_position = self.caption_settings['caption_position']

        if is_tagger_model_id(model_id) and selected_image_count > 1:
            self._run_wd_batched(model, caption_position,
                                 are_multiple_images_selected,
                                 captioning_start_datetime)
            return

        # Prefetch the next image's inputs on a worker while the GPU generates.
        with ThreadPoolExecutor(max_workers=1) as prefetch_pool:
            next_future = None
            for i, image_index in enumerate(self.selected_image_indices):
                start_time = perf_counter()
                if self.is_canceled:
                    print('Canceled captioning.')
                    return
                try:
                    if next_future is not None:
                        image, image_prompt, model_inputs = next_future.result()
                    else:
                        image, image_prompt, model_inputs = self._prepare_inputs(
                            model, image_index)
                except UnidentifiedImageError:
                    skipped = self.image_list_model.data(
                        image_index, Qt.ItemDataRole.UserRole)
                    name = skipped.path.name if skipped else 'image'
                    print(f'Skipping {name} because its file format is not '
                          'supported or it is a corrupted image.')
                    next_future = None
                    if i + 1 < selected_image_count:
                        next_future = prefetch_pool.submit(
                            self._prepare_inputs, model,
                            self.selected_image_indices[i + 1])
                    continue
                if i + 1 < selected_image_count:
                    next_future = prefetch_pool.submit(
                        self._prepare_inputs, model,
                        self.selected_image_indices[i + 1])
                else:
                    next_future = None
                caption, console_output_caption = model.generate_caption(
                    model_inputs, image_prompt)
                tags = add_caption_to_tags(image.tags, caption,
                                           caption_position)
                self.caption_generated.emit(image_index, caption, tags)
                if are_multiple_images_selected:
                    self.progress_bar_update_requested.emit(i + 1)
                if i == 0 and not are_multiple_images_selected:
                    self.clear_console_text_edit_requested.emit()
                if console_output_caption is None:
                    console_output_caption = caption
                print(f'{image.path.name} ({perf_counter() - start_time:.1f} s):\n'
                      f'{console_output_caption}')
        if are_multiple_images_selected:
            captioning_end_datetime = datetime.now()
            total_captioning_duration = ((captioning_end_datetime
                                          - captioning_start_datetime)
                                         .total_seconds())
            average_captioning_duration = (total_captioning_duration /
                                           selected_image_count)
            print(f'Finished captioning {selected_image_count} images in '
                  f'{format_duration(total_captioning_duration)} '
                  f'({average_captioning_duration:.1f} s/image) at '
                  f'{captioning_end_datetime.strftime("%Y-%m-%d %H:%M:%S")}.')

    def _run_wd_batched(self, model, caption_position,
                        are_multiple_images_selected,
                        captioning_start_datetime):
        selected_image_count = len(self.selected_image_indices)
        processed = 0
        batch_size = get_wd_batch_size(model, self.caption_settings)
        for batch_start in range(0, selected_image_count, batch_size):
            if self.is_canceled:
                print('Canceled captioning.')
                return
            batch_indices = self.selected_image_indices[
                batch_start:batch_start + batch_size]
            prepared = []
            for image_index in batch_indices:
                image: Image = self.image_list_model.data(
                    image_index, Qt.ItemDataRole.UserRole)
                try:
                    model_inputs = model.get_model_inputs('', image)
                except UnidentifiedImageError:
                    print(f'Skipping {image.path.name} because its file '
                          'format is not supported or it is a corrupted '
                          'image.')
                    continue
                prepared.append((image_index, image, model_inputs))
            if not prepared:
                continue
            captions = model.generate_captions_batch(
                [item[2] for item in prepared])
            for (image_index, image, _), (caption, console_output_caption) in (
                    zip(prepared, captions)):
                tags = add_caption_to_tags(image.tags, caption,
                                           caption_position)
                self.caption_generated.emit(image_index, caption, tags)
                processed += 1
                if are_multiple_images_selected:
                    self.progress_bar_update_requested.emit(processed)
                print(f'{image.path.name}:\n{console_output_caption}')
        captioning_end_datetime = datetime.now()
        total_captioning_duration = ((captioning_end_datetime
                                      - captioning_start_datetime)
                                     .total_seconds())
        if selected_image_count:
            average_captioning_duration = (total_captioning_duration /
                                           selected_image_count)
            print(f'Finished captioning {selected_image_count} images in '
                  f'{format_duration(total_captioning_duration)} '
                  f'({average_captioning_duration:.1f} s/image) at '
                  f'{captioning_end_datetime.strftime("%Y-%m-%d %H:%M:%S")}.')

    def run(self):
        try:
            self.run_captioning()
        except Exception as exception:
            self.is_error = True
            raise exception

    def write(self, text: str):
        self.text_outputted.emit(text)
        # Also mirror the output to the real terminal (not the redirected
        # `sys.stdout`) so progress is visible from the CLI, not just the
        # in-app console panel.
        if sys.__stdout__ is not None:
            sys.__stdout__.write(text)
            sys.__stdout__.flush()
