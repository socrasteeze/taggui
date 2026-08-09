# Based on
# https://huggingface.co/spaces/SmilingWolf/wd-tagger/blob/main/app.py.
import csv
from datetime import datetime
from pathlib import Path

import huggingface_hub
import numpy as np
from PIL import Image as PilImage
from onnxruntime import InferenceSession

import auto_captioning.captioning_thread as captioning_thread
from auto_captioning.auto_captioning_model import AutoCaptioningModel
from auto_captioning.tag_utils import (KAOMOJIS, get_onnx_providers,
                                       get_tags_to_exclude)
from utils.image import Image

class WdTaggerModel:
    def __init__(self, model_id: str, providers: list | None = None):
        model_path = Path(model_id) / 'model.onnx'
        if not model_path.is_file():
            model_path = huggingface_hub.hf_hub_download(model_id,
                                                         filename='model.onnx')
        tags_path = Path(model_id) / 'selected_tags.csv'
        if not tags_path.is_file():
            tags_path = huggingface_hub.hf_hub_download(
                model_id, filename='selected_tags.csv')
        if providers is None:
            providers = ['CPUExecutionProvider']
        self.inference_session = InferenceSession(str(model_path),
                                                  providers=providers)
        self.tags = []
        self.rating_tags_indices = []
        self.general_tags_indices = []
        self.character_tags_indices = []
        with open(tags_path, 'r') as tags_file:
            reader = csv.DictReader(tags_file)
            for index, line in enumerate(reader):
                tag = line['name']
                if tag not in KAOMOJIS:
                    tag = tag.replace('_', ' ')
                self.tags.append(tag)
                category = line['category']
                if category == '9':
                    self.rating_tags_indices.append(index)
                elif category == '0':
                    self.general_tags_indices.append(index)
                elif category == '4':
                    self.character_tags_indices.append(index)

    def _tags_from_probabilities(
            self, probabilities: np.ndarray,
            wd_tagger_settings: dict) -> tuple[tuple, tuple]:
        tags = [tag for index, tag in enumerate(self.tags)
                if index not in self.rating_tags_indices]
        probabilities = np.array([
            probability for index, probability in enumerate(probabilities)
            if index not in self.rating_tags_indices
        ])
        tags_to_exclude = get_tags_to_exclude(
            wd_tagger_settings['tags_to_exclude'])
        tags_and_probabilities = []
        for tag, probability in zip(tags, probabilities):
            if (probability < wd_tagger_settings['min_probability']
                    or tag in tags_to_exclude):
                continue
            tags_and_probabilities.append((tag, probability))
        tags_and_probabilities.sort(key=lambda x: x[1], reverse=True)
        tags_and_probabilities = tags_and_probabilities[
                                 :wd_tagger_settings['max_tags']]
        if tags_and_probabilities:
            tags, probabilities = zip(*tags_and_probabilities)
        else:
            tags, probabilities = (), ()
        return tags, probabilities

    def generate_tags(self, image_array: np.ndarray,
                      wd_tagger_settings: dict) -> tuple[tuple, tuple]:
        input_name = self.inference_session.get_inputs()[0].name
        output_name = self.inference_session.get_outputs()[0].name
        probabilities = self.inference_session.run(
            [output_name], {input_name: image_array})[0][0].astype(np.float32)
        return self._tags_from_probabilities(probabilities, wd_tagger_settings)

    def generate_tags_batch(
            self, image_arrays: list[np.ndarray],
            wd_tagger_settings: dict) -> list[tuple[tuple, tuple]]:
        """Run WD inference on a true batch of preprocessed images."""
        if not image_arrays:
            return []
        if len(image_arrays) == 1:
            return [self.generate_tags(image_arrays[0], wd_tagger_settings)]
        batch = np.concatenate(image_arrays, axis=0)
        input_name = self.inference_session.get_inputs()[0].name
        output_name = self.inference_session.get_outputs()[0].name
        probabilities_batch = self.inference_session.run(
            [output_name], {input_name: batch})[0].astype(np.float32)
        return [
            self._tags_from_probabilities(probabilities, wd_tagger_settings)
            for probabilities in probabilities_batch
        ]


class WdTagger(AutoCaptioningModel):
    image_mode = 'RGBA'

    def __init__(self,
                 captioning_thread_: 'captioning_thread.CaptioningThread',
                 caption_settings: dict):
        super().__init__(captioning_thread_, caption_settings)
        self.wd_tagger_settings = self.caption_settings['wd_tagger_settings']
        self.show_probabilities = self.wd_tagger_settings['show_probabilities']

    def get_error_message(self) -> str | None:
        return None

    def get_processor(self):
        return None

    def get_model(self):
        use_gpu = self.device.type == 'cuda'
        gpu_index = self.device.index or 0
        providers = get_onnx_providers(use_gpu=use_gpu, gpu_index=gpu_index)
        return WdTaggerModel(self.model_id, providers=providers)

    def get_captioning_message(self, are_multiple_images_selected: bool,
                               captioning_start_datetime: datetime) -> str:
        if are_multiple_images_selected:
            captioning_start_datetime_string = (
                self.get_captioning_start_datetime_string(
                    captioning_start_datetime))
            return (f'Generating tags... (start time: '
                    f'{captioning_start_datetime_string})')
        return 'Generating tags...'

    def get_model_inputs(self, image_prompt: str, image: Image) -> np.ndarray:
        pil_image = self.load_image(image)
        # Add a white background to the image in case it has transparent areas.
        canvas = PilImage.new('RGBA', pil_image.size, (255, 255, 255))
        canvas.alpha_composite(pil_image)
        pil_image = canvas.convert('RGB')
        # Pad the image to make it square.
        max_dimension = max(pil_image.size)
        canvas = PilImage.new('RGB', (max_dimension, max_dimension),
                              (255, 255, 255))
        horizontal_padding = (max_dimension - pil_image.width) // 2
        vertical_padding = (max_dimension - pil_image.height) // 2
        canvas.paste(pil_image, (horizontal_padding, vertical_padding))
        # Resize the image to the model's input dimensions.
        _, input_dimension, *_ = (self.model.inference_session.get_inputs()[0]
                                  .shape)
        if max_dimension != input_dimension:
            input_dimensions = (input_dimension, input_dimension)
            canvas = canvas.resize(input_dimensions,
                                   resample=PilImage.Resampling.BICUBIC)
        # Convert the image to a numpy array.
        image_array = np.array(canvas, dtype=np.float32)
        # Reverse the order of the color channels.
        image_array = image_array[:, :, ::-1]
        # Add a batch dimension.
        image_array = np.expand_dims(image_array, axis=0)
        return image_array

    def generate_caption(self, model_inputs: np.ndarray,
                         image_prompt: str) -> tuple[str, str]:
        tags, probabilities = self.model.generate_tags(model_inputs,
                                                       self.wd_tagger_settings)
        return self._format_caption(tags, probabilities)

    def generate_captions_batch(
            self, model_inputs_list: list[np.ndarray]
    ) -> list[tuple[str, str]]:
        results = self.model.generate_tags_batch(
            model_inputs_list, self.wd_tagger_settings)
        return [self._format_caption(tags, probabilities)
                for tags, probabilities in results]

    def _format_caption(self, tags: tuple, probabilities: tuple
                        ) -> tuple[str, str]:
        caption = self.thread.tag_separator.join(tags)
        if self.show_probabilities:
            console_output_caption = self.thread.tag_separator.join(
                f'{tag} ({probability:.2f})'
                for tag, probability in zip(tags, probabilities)
            )
        else:
            console_output_caption = caption
        return caption, console_output_caption

