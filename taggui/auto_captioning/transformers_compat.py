"""
Compatibility helpers for the installed `transformers` release.

The captioner roster spans models added over several years, and the library's
API has moved underneath them: `AutoModelForVision2Seq` gave way to
`AutoModelForImageTextToText`, and the `torch_dtype` load argument was renamed
`dtype`. Newer models also simply do not exist in older releases - Qwen3-VL
needs 4.57 or later - which is worth saying plainly instead of surfacing a
"model type not recognized" traceback.

Keeping that knowledge here means the pin in `requirements.txt` is a choice
about which version is tested, not a version the code is welded to.
"""
from __future__ import annotations

import inspect
import re

# Minimum releases that first shipped support for a given architecture,
# checked against the installed library rather than assumed.
#   qwen3_vl / qwen3_vl_moe  first present in 4.57.0
#   gemma4                   absent in 5.4.0, present in 5.5.0
QWEN3_VL_MINIMUM_TRANSFORMERS_VERSION = '4.57'
GEMMA_4_MINIMUM_TRANSFORMERS_VERSION = '5.5'

# `torch_dtype` became `dtype` in 4.56.0.
DTYPE_RENAME_TRANSFORMERS_VERSION = '4.56'


def get_transformers_version() -> str:
    import transformers
    return getattr(transformers, '__version__', '0')


def parse_version(version: str) -> tuple[int, ...]:
    """
    Turn a version string into comparable integers, ignoring any suffix such
    as `.dev0` or `rc1`.
    """
    numbers = re.findall(r'\d+', version.split('+')[0])
    return tuple(int(number) for number in numbers[:3]) or (0,)


def is_version_at_least(minimum: str, version: str | None = None) -> bool:
    if version is None:
        version = get_transformers_version()
    installed = parse_version(version)
    required = parse_version(minimum)
    # Compare on the shorter length so "4.57" matches "4.57.6".
    length = min(len(installed), len(required))
    return installed[:length] >= required[:length]


def get_version_error_message(model_name: str, minimum: str,
                              version: str | None = None) -> str:
    if version is None:
        version = get_transformers_version()
    return (f'{model_name} requires transformers {minimum} or newer, but '
            f'{version} is installed.\nUpdate the version in '
            f'`requirements.txt`, then re-run `run.bat update` (Windows) or '
            f'`pip install -r requirements.txt`.')


def get_image_text_model_class():
    """
    The auto class for image-text-to-text models.

    `AutoModelForImageTextToText` is the current name; `AutoModelForVision2Seq`
    is the older one, kept as a deprecated alias for part of the 4.x line and
    dropped afterwards.
    """
    import transformers
    for class_name in ('AutoModelForImageTextToText',
                       'AutoModelForVision2Seq'):
        model_class = getattr(transformers, class_name, None)
        if model_class is not None:
            return model_class
    raise ImportError(
        'The installed transformers release provides neither '
        '`AutoModelForImageTextToText` nor `AutoModelForVision2Seq`.')


def get_dtype_argument_name() -> str:
    """
    Whether `from_pretrained` takes `dtype` (current) or `torch_dtype` (older).

    `from_pretrained` absorbs both into `**kwargs` rather than naming either,
    so the signature usually cannot answer this and the version decides. The
    rename landed in 4.56.0: 4.55.4 accepts only `torch_dtype`, 4.56.2 warns
    that it is deprecated and wants `dtype`. Passing the wrong one is silently
    ignored, which loads the model at full precision instead of failing.
    """
    from transformers import PreTrainedModel
    try:
        parameters = inspect.signature(
            PreTrainedModel.from_pretrained).parameters
    except (TypeError, ValueError):
        parameters = {}
    if 'torch_dtype' in parameters:
        return 'torch_dtype'
    if 'dtype' in parameters:
        return 'dtype'
    return ('dtype' if is_version_at_least(DTYPE_RENAME_TRANSFORMERS_VERSION)
            else 'torch_dtype')


def get_dtype_arguments(dtype) -> dict:
    """The load argument that pins a model's compute dtype."""
    return {get_dtype_argument_name(): dtype}
