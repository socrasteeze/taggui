"""
pixai-tagger support.

pixai-tagger is a newer Danbooru-trained tagger than WD v3 - roughly 13.5k tags
against WD's 10.8k, with better recall and newer character coverage - so it
complements `wd-eva02-large-tagger-v3` rather than replacing it.

The model id points at deepghs's ONNX export rather than `pixai-labs`, which
publishes PyTorch weights with no `model.onnx`. Despite the shared filenames,
this is *not* a WD tagger: it takes channel-first RGB with ImageNet
normalisation, where WD takes channel-last BGR at raw 0-255. Reusing the WD
image path would produce tags rather than an error, and they would be wrong.
Preprocessing therefore comes from the export's own `preprocess.json`.

The inference itself lives in `pixai_tagger_model.py`, which needs no torch.
"""
from datetime import datetime

import numpy as np

from auto_captioning.auto_captioning_model import AutoCaptioningModel
from auto_captioning.models.pixai_tagger_model import PixaiTaggerModel
from auto_captioning.tag_utils import get_onnx_providers
from utils.image import Image


class PixaiTagger(AutoCaptioningModel):
    # Keep the alpha channel; the `convert_rgb` stage flattens it onto the
    # background the export asks for, rather than a hardcoded white.
    image_mode = 'RGBA'
    model_display_name = 'pixai-tagger'

    def __init__(self, captioning_thread_, caption_settings: dict):
        super().__init__(captioning_thread_, caption_settings)
        self.wd_tagger_settings = self.caption_settings['wd_tagger_settings']
        self.show_probabilities = self.wd_tagger_settings[
            'show_probabilities']

    def get_error_message(self) -> str | None:
        return None

    def get_processor(self):
        return None

    def get_model(self):
        providers = get_onnx_providers(use_gpu=self.device.type == 'cuda',
                                       gpu_index=self.device.index or 0)
        return PixaiTaggerModel(self.model_id, providers=providers)

    def get_captioning_message(self, are_multiple_images_selected: bool,
                               captioning_start_datetime: datetime) -> str:
        if are_multiple_images_selected:
            start = self.get_captioning_start_datetime_string(
                captioning_start_datetime)
            return f'Generating tags with pixai-tagger... (start time: {start})'
        return 'Generating tags with pixai-tagger...'

    def get_model_inputs(self, image_prompt: str, image: Image) -> np.ndarray:
        array = self.model.transform(self.load_image(image))
        return np.asarray(array, dtype=np.float32)[None, ...]

    def generate_caption(self, model_inputs: np.ndarray,
                         image_prompt: str) -> tuple[str, str]:
        tags, probabilities = self.model.generate_tags(
            model_inputs, self.wd_tagger_settings)
        return self._format_caption(tags, probabilities)

    def generate_captions_batch(self, model_inputs_list: list[np.ndarray]
                                ) -> list[tuple[str, str]]:
        results = self.model.generate_tags_batch(model_inputs_list,
                                                 self.wd_tagger_settings)
        return [self._format_caption(tags, probabilities)
                for tags, probabilities in results]

    def _format_caption(self, tags: tuple, probabilities: tuple
                        ) -> tuple[str, str]:
        caption = self.thread.tag_separator.join(tags)
        if self.show_probabilities:
            console_output_caption = self.thread.tag_separator.join(
                f'{tag} ({probability:.2f})'
                for tag, probability in zip(tags, probabilities))
        else:
            console_output_caption = caption
        return caption, console_output_caption
