"""
Interpreter for the `preprocess.json` files that ship with deepghs ONNX model
exports.

Those exports describe their preprocessing as data - an ordered list of typed
stages - instead of leaving each application to reimplement it. Getting any
stage wrong produces plausible-looking but wrong predictions rather than an
error, so the stages are implemented here to match the reference
(`imgutils.preprocess.pillow`) exactly, and an unrecognised stage raises
instead of being skipped.

The reference is not self-consistent about the order of the two numbers in a
`size` pair, so neither is this: `resize` and `center_crop` take (height,
width), while `pad_to_size` takes (width, height). Square sizes hide the
difference; non-square ones do not.

This was checked against dghs-imgutils 0.19.0 by running both implementations
over the same images and pipelines and comparing the arrays exactly - see
`tools/differential_preprocess_check.py`. Re-run it after changing this file.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from PIL import Image as PilImage
from PIL import ImageColor

# Stage names accepted in `preprocess.json`, mapped to Pillow resampling
# filters where relevant.
RESAMPLING_FILTERS = {
    'nearest': PilImage.Resampling.NEAREST,
    'bilinear': PilImage.Resampling.BILINEAR,
    'bicubic': PilImage.Resampling.BICUBIC,
    'box': PilImage.Resampling.BOX,
    'hamming': PilImage.Resampling.HAMMING,
    'lanczos': PilImage.Resampling.LANCZOS,
}


class UnknownPreprocessStageError(ValueError):
    """Raised for a stage this interpreter does not implement."""


def get_resampling_filter(name: str | int):
    if isinstance(name, int):
        return name
    try:
        return RESAMPLING_FILTERS[name]
    except KeyError:
        raise UnknownPreprocessStageError(
            f'Unknown interpolation "{name}". Expected one of: '
            f'{", ".join(sorted(RESAMPLING_FILTERS))}.') from None


def parse_size(size, height_first: bool) -> tuple[int, int]:
    """
    Normalise a stage's `size` field to (width, height).

    `height_first` selects how a two-element pair is read, because the stages
    disagree - see the module docstring.
    """
    if isinstance(size, int):
        return size, size
    values = list(size)
    if len(values) == 1:
        return values[0], values[0]
    if len(values) != 2:
        raise UnknownPreprocessStageError(
            f'Expected a size of one or two numbers, got {size!r}.')
    if height_first:
        return values[1], values[0]
    return values[0], values[1]


def parse_color(color, mode: str):
    """Turn a stage's colour field into something `Image.new` accepts."""
    if isinstance(color, str):
        color = ImageColor.getrgb(color)
    elif isinstance(color, (list, tuple)):
        color = tuple(color)
    elif isinstance(color, int):
        color = (color, color, color)
    if not isinstance(color, tuple):
        return color
    channel_count = len(PilImage.new(mode, (1, 1)).getbands())
    if channel_count == 1:
        return int(sum(color[:3]) / 3)
    if len(color) < channel_count:
        # Pad an RGB colour out to RGBA with full opacity.
        color = color + (255,) * (channel_count - len(color))
    return color[:channel_count]


def convert_rgb(image: PilImage.Image, force_background: str | None = 'white'
                ) -> PilImage.Image:
    if image.mode == 'RGB':
        return image
    if force_background is not None and image.mode in ('RGBA', 'LA', 'P'):
        prepared = image.convert('RGBA')
        canvas = PilImage.new('RGBA', prepared.size,
                              parse_color(force_background, 'RGBA'))
        canvas.alpha_composite(prepared)
        return canvas.convert('RGB')
    return image.convert('RGB')


def resize(image: PilImage.Image, size, interpolation='bilinear',
           max_size: int | None = None, antialias: bool = True
           ) -> PilImage.Image:
    resample = get_resampling_filter(interpolation)
    width, height = image.size
    if isinstance(size, int) or len(list(size)) == 1:
        # A single number scales the shorter edge and keeps the aspect ratio.
        edge = size if isinstance(size, int) else list(size)[0]
        if width < height:
            new_width = edge
            new_height = int(edge * height / width)
        else:
            new_height = edge
            new_width = int(edge * width / height)
        if max_size is not None and max(new_height, new_width) > max_size:
            if new_height > new_width:
                new_width = int(max_size * new_width / new_height)
                new_height = max_size
            else:
                new_height = int(max_size * new_height / new_width)
                new_width = max_size
    else:
        new_width, new_height = parse_size(size, height_first=True)
    if (new_width, new_height) == image.size:
        return image
    if resample in (PilImage.Resampling.BILINEAR, PilImage.Resampling.BICUBIC):
        return image.resize((new_width, new_height), resample,
                            reducing_gap=None if antialias else 1.0)
    return image.resize((new_width, new_height), resample)


def center_crop(image: PilImage.Image, size) -> PilImage.Image:
    crop_width, crop_height = parse_size(size, height_first=True)
    width, height = image.size
    if width < crop_width or height < crop_height:
        # The reference pads short edges with black before it crops.
        padded_width = max(width, crop_width)
        padded_height = max(height, crop_height)
        canvas = PilImage.new(image.mode, (padded_width, padded_height),
                              parse_color(0, image.mode))
        canvas.paste(image, (max(crop_width - width, 0) // 2,
                             max(crop_height - height, 0) // 2))
        image = canvas
        width, height = image.size
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    return image.crop((left, top, left + crop_width, top + crop_height))


def pad_to_size(image: PilImage.Image, size, background_color='white',
                interpolation='bilinear') -> PilImage.Image:
    target_width, target_height = parse_size(size, height_first=False)
    width, height = image.size
    ratio = min(target_width / width, target_height / height)
    new_width, new_height = round(width * ratio), round(height * ratio)
    resized = image.resize((new_width, new_height),
                           get_resampling_filter(interpolation))
    canvas = PilImage.new(image.mode, (target_width, target_height),
                          parse_color(background_color, image.mode))
    canvas.paste(resized, ((target_width - new_width) // 2,
                           (target_height - new_height) // 2))
    return canvas


def to_tensor(image) -> np.ndarray:
    """Convert an image to a channel-first float array scaled to [0, 1]."""
    if isinstance(image, np.ndarray):
        return image
    array = np.array(image, copy=True)
    if array.ndim == 2:
        # Greyscale gains an explicit channel.
        array = array[None, ...]
    else:
        array = array.transpose((2, 0, 1))
    if np.issubdtype(array.dtype, np.floating):
        return array.astype(np.float32)
    return array.astype(np.float32) / 255


def normalize(array: np.ndarray, mean, std) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32).copy()
    mean = np.asarray(mean, dtype=np.float32).reshape(-1, 1, 1)
    std = np.asarray(std, dtype=np.float32).reshape(-1, 1, 1)
    array -= mean
    array /= std
    return array


def rescale(array: np.ndarray, rescale_factor: float = 1 / 255) -> np.ndarray:
    return np.asarray(array, dtype=np.float32) * np.float32(rescale_factor)


# Each entry takes the running value plus the stage's remaining fields.
STAGE_HANDLERS: dict[str, Callable] = {
    'convert_rgb': convert_rgb,
    'resize': resize,
    'center_crop': center_crop,
    'pad_to_size': pad_to_size,
    'to_tensor': lambda value: to_tensor(value),
    'maybe_to_tensor': lambda value: to_tensor(value),
    'normalize': normalize,
    'rescale': rescale,
}


def apply_stage(value, stage: dict):
    fields = {key: field for key, field in stage.items() if key != 'type'}
    stage_type = stage.get('type')
    handler = STAGE_HANDLERS.get(stage_type)
    if handler is None:
        raise UnknownPreprocessStageError(
            f'Unsupported preprocessing stage "{stage_type}". Supported '
            f'stages: {", ".join(sorted(STAGE_HANDLERS))}.')
    try:
        return handler(value, **fields)
    except TypeError as exception:
        raise UnknownPreprocessStageError(
            f'Preprocessing stage "{stage_type}" was given fields it does not '
            f'accept ({", ".join(sorted(fields))}): {exception}') from exception


def build_transform(stages: list[dict]) -> Callable:
    """
    Compose `preprocess.json` stages into one callable.

    The callable takes a PIL image and returns whatever the final stage
    produces - for a tagger, a channel-first float array ready for the model.
    """
    if isinstance(stages, dict):
        stages = [stages]
    stage_list = list(stages)

    def transform(image):
        value = image
        for stage in stage_list:
            value = apply_stage(value, stage)
        return value

    return transform
