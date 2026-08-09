"""
Tests for `auto_captioning.models.pixai_tagger`.

The model weights are ~1 GB and the export cannot be reached from CI, so the
ONNX session is faked. What is worth pinning is everything around it: which
output is read, how per-category thresholds are applied, and the tag file
parsing - the places where a mistake yields wrong tags rather than an error.
"""
import csv
import json

import numpy as np
import pytest

from auto_captioning.models import pixai_tagger_model as pixai_tagger
from auto_captioning.models.pixai_tagger_model import (
    DEFAULT_CATEGORY_THRESHOLD, RATING_CATEGORY, PixaiTaggerModel)

# name, category. Danbooru categories: 0 general, 2 rating, 4 character.
TAG_ROWS = [
    ('1girl', 0),
    ('long_hair', 0),
    ('smile', 0),
    ('hatsune_miku', 4),
    ('kagamine_rin', 4),
    ('safe', RATING_CATEGORY),
]

PREPROCESS_STAGES = {
    'stages': [
        {'type': 'convert_rgb', 'force_background': 'white'},
        {'type': 'resize', 'size': [8, 8], 'interpolation': 'bicubic'},
        {'type': 'to_tensor'},
    ]
}


class FakeOutput:
    def __init__(self, name):
        self.name = name


class FakeInput:
    def __init__(self, name='input', shape=None):
        self.name = name
        self.shape = shape or ['batch', 3, 8, 8]


class FakeSession:
    """Returns a fixed score vector per image, for the requested output."""

    def __init__(self, model_path, providers=None, output_names=('prediction',),
                 scores=None):
        self.model_path = model_path
        self.providers = providers
        self.output_names = list(output_names)
        self.scores = scores if scores is not None else [0.9, 0.8, 0.2, 0.9,
                                                         0.1, 0.99]
        self.requested_outputs = []

    def get_inputs(self):
        return [FakeInput()]

    def get_outputs(self):
        return [FakeOutput(name) for name in self.output_names]

    def run(self, output_names, feeds):
        self.requested_outputs.append(list(output_names))
        batch = list(feeds.values())[0]
        batch_size = len(batch)
        return [np.array([self.scores] * batch_size, dtype=np.float32)]


@pytest.fixture
def model_directory(tmp_path):
    """A local model directory laid out like the deepghs export."""
    (tmp_path / 'model.onnx').write_bytes(b'')
    with open(tmp_path / 'selected_tags.csv', 'w', newline='',
              encoding='utf-8') as tags_file:
        writer = csv.writer(tags_file)
        writer.writerow(['name', 'category'])
        writer.writerows(TAG_ROWS)
    (tmp_path / 'preprocess.json').write_text(json.dumps(PREPROCESS_STAGES),
                                              encoding='utf-8')
    return tmp_path


@pytest.fixture
def write_thresholds(model_directory):
    def factory(rows):
        with open(model_directory / 'thresholds.csv', 'w', newline='',
                  encoding='utf-8') as thresholds_file:
            writer = csv.writer(thresholds_file)
            writer.writerow(['category', 'threshold', 'name'])
            writer.writerows(rows)
    return factory


@pytest.fixture
def build_model(model_directory, monkeypatch):
    def factory(**session_kwargs):
        sessions = []

        def make_session(model_path, providers=None):
            session = FakeSession(model_path, providers, **session_kwargs)
            sessions.append(session)
            return session

        monkeypatch.setattr(pixai_tagger, 'InferenceSession', make_session)
        model = PixaiTaggerModel(str(model_directory))
        model.session_spy = sessions[0]
        return model
    return factory


def settings(**overrides) -> dict:
    base = {
        'min_probability': 0.4,
        'max_tags': 30,
        'tags_to_exclude': '',
        'use_model_thresholds': True,
    }
    base.update(overrides)
    return base


class TestLoading:
    def test_tags_and_categories_are_read(self, build_model):
        model = build_model()
        assert model.tags[:3] == ['1girl', 'long hair', 'smile']
        assert model.categories == [0, 0, 0, 4, 4, RATING_CATEGORY]

    def test_underscores_become_spaces_like_the_wd_taggers(self, build_model):
        assert 'hatsune miku' in build_model().tags

    def test_a_missing_thresholds_file_is_not_fatal(self, build_model):
        assert build_model().category_thresholds == {}

    def test_thresholds_are_read_per_category(self, build_model,
                                              write_thresholds):
        write_thresholds([(0, 0.3, 'general'), (4, 0.75, 'character')])
        assert build_model().category_thresholds == {0: 0.3, 4: 0.75}

    def test_the_first_threshold_for_a_category_wins(self, build_model,
                                                     write_thresholds):
        write_thresholds([(0, 0.3, 'general'), (0, 0.9, 'general again')])
        assert build_model().category_thresholds == {0: 0.3}

    def test_the_preprocess_pipeline_is_built(self, build_model):
        from PIL import Image as PilImage

        model = build_model()
        array = model.transform(PilImage.new('RGBA', (32, 16)))
        assert array.shape == (3, 8, 8)


class TestOutputSelection:
    def test_prediction_is_preferred(self, build_model):
        model = build_model(output_names=('embedding', 'logits',
                                          'prediction'))
        assert model.get_prediction_output_name() == 'prediction'

    def test_logits_are_used_when_there_is_no_prediction(self, build_model):
        model = build_model(output_names=('embedding', 'logits'))
        assert model.get_prediction_output_name() == 'logits'

    def test_the_embedding_output_is_never_picked_by_accident(self,
                                                              build_model):
        """It is listed first; taking output zero blindly would return it."""
        model = build_model(output_names=('embedding', 'prediction'))
        model.generate_tags(np.zeros((1, 3, 8, 8), np.float32), settings())
        assert model.session_spy.requested_outputs == [['prediction']]

    def test_logits_get_a_sigmoid_applied(self, build_model):
        # A logit of 0 is a probability of 0.5, which clears a 0.4 threshold.
        model = build_model(output_names=('logits',),
                            scores=[0.0] * len(TAG_ROWS))
        tags, probabilities = model.generate_tags(
            np.zeros((1, 3, 8, 8), np.float32),
            settings(use_model_thresholds=False, min_probability=0.4))
        assert tags
        assert probabilities[0] == pytest.approx(0.5)

    def test_prediction_is_used_as_is(self, build_model):
        """`prediction` already has the sigmoid; applying another would halve it."""
        model = build_model(scores=[0.9] * len(TAG_ROWS))
        _, probabilities = model.generate_tags(
            np.zeros((1, 3, 8, 8), np.float32), settings())
        assert probabilities[0] == pytest.approx(0.9)


class TestThresholds:
    def test_model_thresholds_are_applied_per_category(self, build_model,
                                                       write_thresholds):
        write_thresholds([(0, 0.3, 'general'), (4, 0.75, 'character')])
        model = build_model()
        tags, _ = model.generate_tags(np.zeros((1, 3, 8, 8), np.float32),
                                      settings())
        # kagamine rin scores 0.1, below the 0.75 character bar; smile scores
        # 0.2, below the 0.3 general bar.
        assert set(tags) == {'1girl', 'long hair', 'hatsune miku'}

    def test_a_single_threshold_is_used_when_the_setting_is_off(
            self, build_model, write_thresholds):
        write_thresholds([(0, 0.3, 'general'), (4, 0.75, 'character')])
        model = build_model()
        tags, _ = model.generate_tags(
            np.zeros((1, 3, 8, 8), np.float32),
            settings(use_model_thresholds=False, min_probability=0.85))
        assert set(tags) == {'1girl', 'hatsune miku'}

    def test_the_single_threshold_is_used_when_the_export_has_none(
            self, build_model):
        model = build_model()
        tags, _ = model.generate_tags(
            np.zeros((1, 3, 8, 8), np.float32),
            settings(min_probability=0.85))
        assert set(tags) == {'1girl', 'hatsune miku'}

    def test_a_category_without_a_threshold_falls_back_to_the_default(
            self, build_model, write_thresholds):
        write_thresholds([(0, 0.05, 'general')])
        model = build_model()
        tags, _ = model.generate_tags(np.zeros((1, 3, 8, 8), np.float32),
                                      settings())
        assert 'smile' in tags                     # general, 0.2 >= 0.05
        assert 'hatsune miku' in tags              # character, 0.9 >= default
        assert DEFAULT_CATEGORY_THRESHOLD == 0.4
        assert 'kagamine rin' not in tags          # character, 0.1 < default


class TestTagSelection:
    def test_rating_tags_are_dropped(self, build_model):
        model = build_model()
        tags, _ = model.generate_tags(np.zeros((1, 3, 8, 8), np.float32),
                                      settings(min_probability=0.0,
                                               use_model_thresholds=False))
        assert 'safe' not in tags

    def test_tags_come_back_in_descending_confidence(self, build_model):
        model = build_model()
        _, probabilities = model.generate_tags(
            np.zeros((1, 3, 8, 8), np.float32),
            settings(min_probability=0.0, use_model_thresholds=False))
        assert list(probabilities) == sorted(probabilities, reverse=True)

    def test_max_tags_truncates_the_lowest_scoring(self, build_model):
        model = build_model()
        tags, _ = model.generate_tags(
            np.zeros((1, 3, 8, 8), np.float32),
            settings(min_probability=0.0, use_model_thresholds=False,
                     max_tags=2))
        assert set(tags) == {'1girl', 'hatsune miku'}

    def test_excluded_tags_are_removed(self, build_model):
        model = build_model()
        tags, _ = model.generate_tags(
            np.zeros((1, 3, 8, 8), np.float32),
            settings(min_probability=0.0, use_model_thresholds=False,
                     tags_to_exclude='1girl, long hair'))
        assert '1girl' not in tags
        assert 'long hair' not in tags

    def test_nothing_above_the_threshold_yields_no_tags(self, build_model):
        model = build_model()
        assert model.generate_tags(
            np.zeros((1, 3, 8, 8), np.float32),
            settings(min_probability=1.1,
                     use_model_thresholds=False)) == ((), ())


class TestBatching:
    def test_a_batch_returns_one_result_per_image(self, build_model):
        model = build_model()
        arrays = [np.zeros((1, 3, 8, 8), np.float32) for _ in range(4)]
        results = model.generate_tags_batch(arrays, settings())
        assert len(results) == 4

    def test_a_batch_runs_one_inference_call(self, build_model):
        model = build_model()
        arrays = [np.zeros((1, 3, 8, 8), np.float32) for _ in range(4)]
        model.generate_tags_batch(arrays, settings())
        assert len(model.session_spy.requested_outputs) == 1

    def test_a_batch_agrees_with_tagging_one_at_a_time(self, build_model):
        model = build_model()
        arrays = [np.zeros((1, 3, 8, 8), np.float32) for _ in range(3)]
        batched = model.generate_tags_batch(arrays, settings())
        single = model.generate_tags(arrays[0], settings())
        assert all(result == single for result in batched)

    def test_an_empty_batch_is_handled(self, build_model):
        assert build_model().generate_tags_batch([], settings()) == []

    def test_a_single_item_batch_works(self, build_model):
        model = build_model()
        results = model.generate_tags_batch(
            [np.zeros((1, 3, 8, 8), np.float32)], settings())
        assert len(results) == 1


def test_the_roster_entry_points_at_the_onnx_export():
    """`pixai-labs` publishes PyTorch weights with no `model.onnx`."""
    from auto_captioning.models_list import RECOMMENDED_MODELS

    pixai_entries = [model_id for model_id in RECOMMENDED_MODELS
                     if 'pixai' in model_id]
    assert pixai_entries == ['deepghs/pixai-tagger-v0.9-onnx']


def test_the_tagger_shares_the_tagger_settings_panel():
    from auto_captioning.models_list import is_tagger_model_id

    assert is_tagger_model_id('deepghs/pixai-tagger-v0.9-onnx')
