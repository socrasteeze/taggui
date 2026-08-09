"""
The auto-captioning model roster and the routing from a model id to the class
that implements it.

Model classes are imported on demand rather than at module import: pulling all
of them in loads torch and transformers, which the rest of the application does
not need until captioning actually starts.
"""
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from auto_captioning.auto_captioning_model import AutoCaptioningModel

# The roster is grouped so the model list reflects what is worth reaching for
# today. Legacy models remain fully runnable; they are 2023-2024-era quality
# and no longer competitive, so they sit below a separator rather than being
# removed.
RECOMMENDED_MODELS = [
    'fancyfeast/llama-joycaption-beta-one-hf-llava',
    'Qwen/Qwen3-VL-2B-Instruct',
    'Qwen/Qwen3-VL-4B-Instruct',
    'Qwen/Qwen3-VL-8B-Instruct',
    'Qwen/Qwen3-VL-30B-A3B-Instruct',
    'google/gemma-4-31b-it',
    'google/gemma-4-e4b-it',
    'microsoft/Florence-2-large-ft',
    'microsoft/Florence-2-large',
    'microsoft/Florence-2-base-ft',
    'microsoft/Florence-2-base',
    'MiaoshouAI/Florence-2-large-PromptGen-v2.0',
    'MiaoshouAI/Florence-2-base-PromptGen-v2.0',
    'deepghs/pixai-tagger-v0.9-onnx',
    'SmilingWolf/wd-eva02-large-tagger-v3',
    'SmilingWolf/wd-vit-large-tagger-v3',
    'SmilingWolf/wd-swinv2-tagger-v3',
    'SmilingWolf/wd-convnext-tagger-v3',
    'SmilingWolf/wd-vit-tagger-v3',
    'microsoft/Phi-3-vision-128k-instruct',
]

LEGACY_MODELS = [
    'llava-hf/llava-v1.6-mistral-7b-hf',
    'llava-hf/llava-v1.6-vicuna-7b-hf',
    'llava-hf/llava-v1.6-vicuna-13b-hf',
    'llava-hf/llava-v1.6-34b-hf',
    'xtuner/llava-llama-3-8b-v1_1-transformers',
    'vikhyatk/moondream2',
    'vikhyatk/moondream1',
    'SmilingWolf/wd-v1-4-moat-tagger-v2',
    'SmilingWolf/wd-v1-4-swinv2-tagger-v2',
    'SmilingWolf/wd-v1-4-convnext-tagger-v2',
    'SmilingWolf/wd-v1-4-convnextv2-tagger-v2',
    'SmilingWolf/wd-v1-4-vit-tagger-v2',
    'llava-hf/llava-1.5-7b-hf',
    'llava-hf/llava-1.5-13b-hf',
    'llava-hf/bakLlava-v1-hf',
    'Salesforce/instructblip-vicuna-7b',
    'Salesforce/instructblip-vicuna-13b',
    'Salesforce/instructblip-flan-t5-xl',
    'Salesforce/instructblip-flan-t5-xxl',
    'Salesforce/blip2-opt-2.7b',
    'Salesforce/blip2-opt-6.7b',
    'Salesforce/blip2-opt-6.7b-coco',
    'Salesforce/blip2-flan-t5-xl',
    'Salesforce/blip2-flan-t5-xxl',
    'microsoft/kosmos-2-patch14-224',
]

# Label inserted between the two groups in the model combo box. It is not a
# selectable model; `is_group_separator` identifies it.
LEGACY_GROUP_SEPARATOR = '--- Legacy (still runnable) ---'

MODELS = RECOMMENDED_MODELS + [LEGACY_GROUP_SEPARATOR] + LEGACY_MODELS


def is_group_separator(model_id: str) -> bool:
    return model_id == LEGACY_GROUP_SEPARATOR


def get_model_class_location(model_id: str) -> tuple[str, str]:
    """
    The module and class name implementing a model id, as data so the routing
    can be checked without importing torch. Order matters: the first matching
    pattern wins, so narrower ids are tested before broader ones.
    """
    lowercase_model_id = model_id.lower()
    if 'qwen3-vl' in lowercase_model_id or 'qwen3_vl' in lowercase_model_id:
        return 'auto_captioning.models.qwen3_vl', 'Qwen3Vl'
    if 'gemma-4' in lowercase_model_id or 'gemma4' in lowercase_model_id:
        return 'auto_captioning.models.gemma_4', 'Gemma4'
    if 'florence' in lowercase_model_id:
        if 'promptgen' in lowercase_model_id:
            return 'auto_captioning.models.florence_2', 'Florence2Promptgen'
        return 'auto_captioning.models.florence_2', 'Florence2'
    if 'joycaption' in lowercase_model_id:
        return 'auto_captioning.models.joycaption', 'Joycaption'
    if 'kosmos' in lowercase_model_id:
        return 'auto_captioning.models.kosmos_2', 'Kosmos2'
    if 'llava-v1.6-34b' in lowercase_model_id:
        return 'auto_captioning.models.llava_next', 'LlavaNext34b'
    if 'llava-v1.6-mistral' in lowercase_model_id:
        return 'auto_captioning.models.llava_next', 'LlavaNextMistral'
    if 'llava-v1.6-vicuna' in lowercase_model_id:
        return 'auto_captioning.models.llava_next', 'LlavaNextVicuna'
    if 'llava-llama-3' in lowercase_model_id:
        return 'auto_captioning.models.llava_llama_3', 'LlavaLlama3'
    if 'llava' in lowercase_model_id:
        return 'auto_captioning.models.llava_1_point_5', 'Llava1Point5'
    if 'moondream1' in lowercase_model_id:
        return 'auto_captioning.models.moondream', 'Moondream1'
    if 'moondream2' in lowercase_model_id:
        return 'auto_captioning.models.moondream', 'Moondream2'
    if 'phi-3' in lowercase_model_id:
        return 'auto_captioning.models.phi_3_vision', 'Phi3Vision'
    if 'pixai' in lowercase_model_id and 'tagger' in lowercase_model_id:
        return 'auto_captioning.models.pixai_tagger', 'PixaiTagger'
    if 'wd' in lowercase_model_id and 'tagger' in lowercase_model_id:
        return 'auto_captioning.models.wd_tagger', 'WdTagger'
    return 'auto_captioning.auto_captioning_model', 'AutoCaptioningModel'


def get_model_class(model_id: str) -> type['AutoCaptioningModel']:
    module_name, class_name = get_model_class_location(model_id)
    return getattr(import_module(module_name), class_name)


# Taggers that emit booru-style tags and share the tagger settings panel.
TAGGER_CLASS_NAMES = frozenset({'WdTagger', 'PixaiTagger'})


def is_tagger_model_id(model_id: str) -> bool:
    """
    Whether a model id is a tagger, without importing the class - the UI asks
    this while building itself, long before torch is needed.
    """
    return get_model_class_location(model_id)[1] in TAGGER_CLASS_NAMES
