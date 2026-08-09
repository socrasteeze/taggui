"""
Helpers shared by the ONNX taggers.

Kept out of `models/wd_tagger.py` so the tagger implementations can import them
without dragging in `AutoCaptioningModel`, and therefore torch. None of this
needs it.
"""
import re

import onnxruntime

# Tags that are punctuation faces rather than words, so their underscores are
# meaningful and must not be turned into spaces.
KAOMOJIS = ['0_0', '(o)_(o)', '+_+', '+_-', '._.', '<o>_<o>', '<|>_<|>', '=_=',
            '>_<', '3_3', '6_9', '>_o', '@_@', '^_^', 'o_o', 'u_u', 'x_x',
            '|_|', '||_||']


def get_onnx_providers(use_gpu: bool, gpu_index: int = 0) -> list:
    """
    Build the ONNX Runtime execution provider list, preferring GPU providers
    when the user selected the GPU and the corresponding provider is actually
    installed (e.g. onnxruntime-gpu or onnxruntime-directml). Falls back to CPU
    otherwise, so this is safe even with the CPU-only onnxruntime package.
    """
    available = onnxruntime.get_available_providers()
    providers = []
    if use_gpu:
        if 'CUDAExecutionProvider' in available:
            providers.append(('CUDAExecutionProvider',
                              {'device_id': gpu_index}))
        elif 'DmlExecutionProvider' in available:
            providers.append(('DmlExecutionProvider',
                              {'device_id': gpu_index}))
    providers.append('CPUExecutionProvider')
    return providers


def get_tags_to_exclude(tags_to_exclude_string: str) -> list[str]:
    if not tags_to_exclude_string.strip():
        return []
    tags = re.split(r'(?<!\\),', tags_to_exclude_string)
    tags = [tag.strip().replace(r'\,', ',') for tag in tags]
    return tags


def normalize_tag(tag: str) -> str:
    """Underscores become spaces, except in kaomoji tags."""
    if tag in KAOMOJIS:
        return tag
    return tag.replace('_', ' ')
