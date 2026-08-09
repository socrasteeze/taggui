"""
pixai-tagger ONNX inference.

Kept apart from the captioning-model wrapper in `pixai_tagger.py` because none
of this needs torch: it is ONNX Runtime, numpy and PIL, so it stays testable
without the captioning dependencies installed.

pixai-tagger is a newer Danbooru-trained tagger than WD v3 - roughly 13.5k tags
against WD's 10.8k, with better recall and newer character coverage - so it
complements `wd-eva02-large-tagger-v3` rather than replacing it.

The model id points at deepghs's ONNX export rather than `pixai-labs`, which
publishes PyTorch weights with no `model.onnx`. Despite the shared filenames,
this is *not* a WD tagger: it takes channel-first RGB with ImageNet
normalisation, where WD takes channel-last BGR at raw 0-255. Reusing the WD
image path would produce tags rather than an error, and they would be wrong.
Preprocessing therefore comes from the export's own `preprocess.json`.
"""
import csv
import json
from pathlib import Path

import huggingface_hub
import numpy as np
from onnxruntime import InferenceSession

from auto_captioning.tag_utils import KAOMOJIS, get_tags_to_exclude
from utils.onnx_preprocess import build_transform

MODEL_FILE_NAME = 'model.onnx'
TAGS_FILE_NAME = 'selected_tags.csv'
PREPROCESS_FILE_NAME = 'preprocess.json'
THRESHOLDS_FILE_NAME = 'thresholds.csv'

# Used when the export ships no `thresholds.csv`, matching the reference
# implementation's fallback.
DEFAULT_CATEGORY_THRESHOLD = 0.4

# Danbooru tag categories. The export has no rating category, unlike WD, whose
# rating tags are category 9.
RATING_CATEGORY = 2


def get_model_file(model_id: str, file_name: str, required: bool = True
                   ) -> Path | None:
    """Find a model file locally, else download it. `None` when optional."""
    local_path = Path(model_id) / file_name
    if local_path.is_file():
        return local_path
    try:
        return Path(huggingface_hub.hf_hub_download(model_id,
                                                    filename=file_name))
    except Exception:
        if required:
            raise
        return None


class PixaiTaggerModel:
    def __init__(self, model_id: str, providers: list | None = None):
        model_path = get_model_file(model_id, MODEL_FILE_NAME)
        tags_path = get_model_file(model_id, TAGS_FILE_NAME)
        preprocess_path = get_model_file(model_id, PREPROCESS_FILE_NAME)
        thresholds_path = get_model_file(model_id, THRESHOLDS_FILE_NAME,
                                         required=False)

        self.inference_session = InferenceSession(
            str(model_path), providers=providers or ['CPUExecutionProvider'])
        self.transform = build_transform(
            json.loads(preprocess_path.read_text(encoding='utf-8'))['stages'])
        self.tags, self.categories = self._read_tags(tags_path)
        self.category_thresholds = self._read_thresholds(thresholds_path)

    @staticmethod
    def _read_tags(tags_path: Path) -> tuple[list[str], list[int]]:
        tags, categories = [], []
        with open(tags_path, 'r', encoding='utf-8') as tags_file:
            for line in csv.DictReader(tags_file):
                tag = line['name']
                if tag not in KAOMOJIS:
                    tag = tag.replace('_', ' ')
                tags.append(tag)
                try:
                    categories.append(int(line['category']))
                except (KeyError, TypeError, ValueError):
                    categories.append(0)
        return tags, categories

    @staticmethod
    def _read_thresholds(thresholds_path: Path | None) -> dict[int, float]:
        """
        Per-category thresholds shipped with the export. They differ sharply -
        characters want a much higher bar than general tags - so a single
        value across every category is a poor default.
        """
        if thresholds_path is None:
            return {}
        thresholds = {}
        with open(thresholds_path, 'r', encoding='utf-8') as thresholds_file:
            for line in csv.DictReader(thresholds_file):
                try:
                    category = int(line['category'])
                    threshold = float(line['threshold'])
                except (KeyError, TypeError, ValueError):
                    continue
                thresholds.setdefault(category, threshold)
        return thresholds

    def get_prediction_output_name(self) -> str:
        """
        The export has several outputs. `prediction` already has the sigmoid
        applied; `logits` does not. Picking the first output blindly could
        return the embedding.
        """
        output_names = [output.name for output in
                        self.inference_session.get_outputs()]
        for preferred in ('prediction', 'logits'):
            if preferred in output_names:
                return preferred
        return output_names[0]

    def _predictions(self, batch: np.ndarray) -> np.ndarray:
        output_name = self.get_prediction_output_name()
        input_name = self.inference_session.get_inputs()[0].name
        predictions = self.inference_session.run(
            [output_name], {input_name: batch})[0].astype(np.float32)
        if output_name == 'logits':
            predictions = 1 / (1 + np.exp(-predictions))
        return predictions

    def _tags_from_predictions(self, predictions: np.ndarray,
                               tagger_settings: dict) -> tuple[tuple, tuple]:
        use_model_thresholds = tagger_settings.get(
            'use_model_thresholds', True) and bool(self.category_thresholds)
        minimum_probability = tagger_settings['min_probability']
        tags_to_exclude = get_tags_to_exclude(
            tagger_settings['tags_to_exclude'])

        tags_and_probabilities = []
        for index, probability in enumerate(predictions[:len(self.tags)]):
            category = self.categories[index]
            if category == RATING_CATEGORY:
                continue
            if use_model_thresholds:
                threshold = self.category_thresholds.get(
                    category, DEFAULT_CATEGORY_THRESHOLD)
            else:
                threshold = minimum_probability
            tag = self.tags[index]
            if probability < threshold or tag in tags_to_exclude:
                continue
            tags_and_probabilities.append((tag, float(probability)))

        tags_and_probabilities.sort(key=lambda pair: -pair[1])
        tags_and_probabilities = tags_and_probabilities[
                                 :tagger_settings['max_tags']]
        if not tags_and_probabilities:
            return (), ()
        tags, probabilities = zip(*tags_and_probabilities)
        return tags, probabilities

    def generate_tags(self, image_array: np.ndarray,
                      tagger_settings: dict) -> tuple[tuple, tuple]:
        predictions = self._predictions(image_array)[0]
        return self._tags_from_predictions(predictions, tagger_settings)

    def generate_tags_batch(self, image_arrays: list[np.ndarray],
                            tagger_settings: dict
                            ) -> list[tuple[tuple, tuple]]:
        if not image_arrays:
            return []
        if len(image_arrays) == 1:
            return [self.generate_tags(image_arrays[0], tagger_settings)]
        predictions = self._predictions(np.concatenate(image_arrays, axis=0))
        return [self._tags_from_predictions(prediction, tagger_settings)
                for prediction in predictions]
