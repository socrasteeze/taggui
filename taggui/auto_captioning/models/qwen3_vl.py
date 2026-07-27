"""Qwen3-VL captioning support."""
from datetime import datetime

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

from auto_captioning.auto_captioning_model import AutoCaptioningModel
from utils.image import Image


class Qwen3Vl(AutoCaptioningModel):
    dtype = torch.bfloat16
    transformers_model_class = AutoModelForVision2Seq
    use_safetensors = True

    def get_processor(self):
        return AutoProcessor.from_pretrained(self.model_id,
                                             trust_remote_code=True)

    @staticmethod
    def get_default_prompt() -> str:
        return 'Describe this image in detail.'

    def format_prompt(self, prompt: str) -> str:
        # Prefer chat template when available.
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
            return (f'Captioning with Qwen3-VL... (device: {self.device}, '
                    f'start time: {start})')
        return f'Captioning with Qwen3-VL... (device: {self.device})'
