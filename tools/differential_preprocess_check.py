"""
Differential test: taggui's preprocess.json interpreter vs deepghs's own.

Runs both over the same images and stage pipelines and compares the results
exactly. Any divergence means taggui would feed the model something different
from what the export expects.

NOT part of the pytest suite, and deliberately so: it needs `dghs-imgutils`,
whose dependency tree (opencv-contrib, scikit-learn, pandas, scipy, shapely,
pilmoji, `numpy<2`) is larger than taggui's own. Install it into a throwaway
environment when `utils/onnx_preprocess.py` changes, run this once, and throw
the environment away.

    python -m venv /tmp/diffcheck
    /tmp/diffcheck/bin/pip install dghs-imgutils
    /tmp/diffcheck/bin/python tools/differential_preprocess_check.py

Last run clean against dghs-imgutils 0.19.0: 53 comparisons, 0 failures.

Cases the reference itself rejects (a three-channel `normalize` against RGBA or
greyscale input, a greyscale canvas colour) are reported and skipped rather
than counted as divergences.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image as PilImage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'taggui'))

from imgutils.preprocess import create_pillow_transforms  # reference
from utils.onnx_preprocess import build_transform          # ours

PIPELINES = {
    'pixai-style 448 bicubic': [
        {'type': 'convert_rgb', 'force_background': 'white'},
        {'type': 'resize', 'size': [448, 448], 'interpolation': 'bicubic'},
        {'type': 'center_crop', 'size': [448, 448]},
        {'type': 'to_tensor'},
        {'type': 'normalize', 'mean': [0.485, 0.456, 0.406],
         'std': [0.229, 0.224, 0.225]},
    ],
    'shorter-edge resize': [
        {'type': 'resize', 'size': 224, 'interpolation': 'bilinear'},
        {'type': 'center_crop', 'size': [224, 224]},
        {'type': 'to_tensor'},
    ],
    'pad_to_size white': [
        {'type': 'pad_to_size', 'size': [384, 384],
         'background_color': 'white', 'interpolation': 'bilinear'},
        {'type': 'to_tensor'},
    ],
    'pad_to_size black non-square': [
        {'type': 'pad_to_size', 'size': [320, 240],
         'background_color': 'black', 'interpolation': 'lanczos'},
        {'type': 'to_tensor'},
    ],
    'non-square resize + crop (rgb only)': [
        {'type': 'resize', 'size': [200, 300], 'interpolation': 'lanczos'},
        {'type': 'center_crop', 'size': [128, 256]},
        {'type': 'to_tensor'},
        {'type': 'normalize', 'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]},
    ],
    'maybe_to_tensor + rescale': [
        {'type': 'resize', 'size': [64, 64], 'interpolation': 'nearest'},
        {'type': 'maybe_to_tensor'},
        {'type': 'rescale', 'rescale_factor': 0.5},
    ],
    'upscale via center_crop padding': [
        {'type': 'center_crop', 'size': [512, 512]},
        {'type': 'to_tensor'},
    ],
    'max_size clamp': [
        {'type': 'resize', 'size': 256, 'interpolation': 'bicubic',
         'max_size': 300},
        {'type': 'to_tensor'},
    ],
}

IMAGES = [
    ('landscape rgb', (640, 360), 'RGB'),
    ('portrait rgb', (360, 640), 'RGB'),
    ('square rgb', (500, 500), 'RGB'),
    ('tiny rgb', (37, 91), 'RGB'),
    ('transparent rgba', (400, 300), 'RGBA'),
    ('greyscale', (300, 200), 'L'),
    ('wide strip', (1200, 80), 'RGB'),
]


def make_image(size, mode, seed):
    generator = np.random.default_rng(seed)
    channels = {'RGB': 3, 'RGBA': 4, 'L': 1}[mode]
    shape = (size[1], size[0]) if channels == 1 else (size[1], size[0],
                                                      channels)
    array = generator.integers(0, 256, shape, dtype=np.uint8)
    return PilImage.fromarray(array.squeeze() if channels == 1 else array,
                              mode=mode)


def main():
    failures = 0
    comparisons = 0
    for pipeline_name, stages in PIPELINES.items():
        for image_index, (image_name, size, mode) in enumerate(IMAGES):
            image = make_image(size, mode, seed=image_index)
            try:
                expected = create_pillow_transforms(stages)(image.copy())
            except Exception as exception:
                print(f'  reference refused {pipeline_name} / {image_name}: '
                      f'{type(exception).__name__}: {exception}')
                continue
            try:
                actual = build_transform(stages)(image.copy())
            except Exception as exception:
                print(f'FAIL {pipeline_name} / {image_name}: ours raised '
                      f'{type(exception).__name__}: {exception}')
                failures += 1
                continue

            comparisons += 1
            expected_array = np.asarray(expected, dtype=np.float64)
            actual_array = np.asarray(actual, dtype=np.float64)
            if expected_array.shape != actual_array.shape:
                print(f'FAIL {pipeline_name} / {image_name}: shape '
                      f'{actual_array.shape} != {expected_array.shape}')
                failures += 1
                continue
            difference = np.abs(expected_array - actual_array).max()
            if difference > 1e-6:
                print(f'FAIL {pipeline_name} / {image_name}: max abs '
                      f'difference {difference}')
                failures += 1

    print(f'\n{comparisons} comparisons, {failures} failures')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
