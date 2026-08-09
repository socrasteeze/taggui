"""
Gemma 4 captioning support.

Gemma 4 is a dense open-weights VLM. All of its parameters are active, unlike
Qwen3-VL-30B-A3B, so the larger variant wants 4-bit on consumer GPUs; the
existing quantization path covers that. Its architecture is only registered in
transformers 5.5 and later, which is newer than the pinned version, so the
version gate reports what to update rather than letting the load fail with an
unrecognised `model_type`.
"""
from datetime import datetime

import torch
from transformers import AutoProcessor

from auto_captioning.auto_captioning_model import AutoCaptioningModel
from auto_captioning.transformers_compat import \
    GEMMA_4_MINIMUM_TRANSFORMERS_VERSION
from utils.image import Image


class Gemma4(AutoCaptioningModel):
    dtype = torch.bfloat16
    use_safetensors = True
    minimum_transformers_version = GEMMA_4_MINIMUM_TRANSFORMERS_VERSION
    model_display_name = 'Gemma 4'

    def get_processor(self):
        return AutoProcessor.from_pretrained(self.model_id,
                                             trust_remote_code=True)

    @staticmethod
    def get_default_prompt() -> str:
        return 'Describe this image in detail.'

    def format_prompt(self, prompt: str) -> str:
        if (self.processor is not None
                and hasattr(self.processor, 'apply_chat_template')):
            messages = [{
                'role': 'user',
                'content': [
                    {'type': 'image'},
                    {'type': 'text', 'text': prompt},
                ],
            }]
            try:
                return self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                return prompt
        return prompt

    def get_model_inputs(self, image_prompt: str, image: Image):
        text = self.get_input_text(image_prompt)
        pil_image = self.load_image(image)
        model_inputs = self.processor(
            text=[text], images=[pil_image], return_tensors='pt',
            padding=True)
        return model_inputs.to(self.device, **self.dtype_argument)

    def get_captioning_message(self, are_multiple_images_selected: bool,
                               captioning_start_datetime: datetime) -> str:
        if are_multiple_images_selected:
            start = self.get_captioning_start_datetime_string(
                captioning_start_datetime)
            return (f'Captioning with Gemma 4... (device: {self.device}, '
                    f'start time: {start})')
        return f'Captioning with Gemma 4... (device: {self.device})'
