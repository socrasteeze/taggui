"""
Tests for the captioner roster in `auto_captioning.models_list`.

The roster is grouped rather than flat so the model list reflects what is
worth reaching for today. `get_model_class` routes an id to its implementation
by pattern, which is easy to break by adding an entry that an earlier pattern
already matches.
"""
import pytest

from auto_captioning.models_list import (LEGACY_GROUP_SEPARATOR,
                                         LEGACY_MODELS, MODELS,
                                         RECOMMENDED_MODELS,
                                         get_model_class_location,
                                         is_group_separator)


def test_the_roster_is_the_two_groups_with_a_separator_between_them():
    assert MODELS == RECOMMENDED_MODELS + [LEGACY_GROUP_SEPARATOR] \
        + LEGACY_MODELS


def test_the_separator_is_recognised_and_normal_ids_are_not():
    assert is_group_separator(LEGACY_GROUP_SEPARATOR)
    assert not is_group_separator('Qwen/Qwen3-VL-8B-Instruct')
    assert not is_group_separator('')


def test_no_model_appears_in_both_groups():
    assert not set(RECOMMENDED_MODELS) & set(LEGACY_MODELS)


def test_no_model_is_listed_twice():
    assert len(MODELS) == len(set(MODELS))


def test_the_recommended_group_leads_with_a_current_captioner():
    assert 'joycaption' in RECOMMENDED_MODELS[0].lower()


def test_the_superseded_captioners_are_all_in_the_legacy_group():
    for marker in ('llava-1.5', 'bakLlava', 'instructblip', 'blip2',
                   'kosmos-2', 'moondream1', 'wd-v1-4'):
        assert any(marker.lower() in model.lower()
                   for model in LEGACY_MODELS), marker
        assert not any(marker.lower() in model.lower()
                       for model in RECOMMENDED_MODELS), marker


def test_the_current_taggers_and_captioners_stay_promoted():
    for marker in ('wd-eva02-large-tagger-v3', 'Florence-2-large',
                   'Qwen3-VL', 'pixai-tagger'):
        assert any(marker.lower() in model.lower()
                   for model in RECOMMENDED_MODELS), marker


class TestModelClassRouting:
    @pytest.mark.parametrize('model_id,expected_class_name', [
        ('Qwen/Qwen3-VL-8B-Instruct', 'Qwen3Vl'),
        ('Qwen/Qwen3-VL-30B-A3B-Instruct', 'Qwen3Vl'),
        ('google/gemma-4-31b-it', 'Gemma4'),
        ('google/gemma-4-e4b-it', 'Gemma4'),
        ('pixai-labs/pixai-tagger-v0.9', 'PixaiTagger'),
        ('SmilingWolf/wd-eva02-large-tagger-v3', 'WdTagger'),
        ('SmilingWolf/wd-v1-4-moat-tagger-v2', 'WdTagger'),
        ('microsoft/Florence-2-large-ft', 'Florence2'),
        ('MiaoshouAI/Florence-2-large-PromptGen-v2.0', 'Florence2Promptgen'),
        ('fancyfeast/llama-joycaption-beta-one-hf-llava', 'Joycaption'),
        ('microsoft/kosmos-2-patch14-224', 'Kosmos2'),
        ('llava-hf/llava-v1.6-34b-hf', 'LlavaNext34b'),
        ('llava-hf/llava-v1.6-mistral-7b-hf', 'LlavaNextMistral'),
        ('llava-hf/llava-v1.6-vicuna-7b-hf', 'LlavaNextVicuna'),
        ('xtuner/llava-llama-3-8b-v1_1-transformers', 'LlavaLlama3'),
        ('llava-hf/llava-1.5-7b-hf', 'Llava1Point5'),
        ('vikhyatk/moondream1', 'Moondream1'),
        ('vikhyatk/moondream2', 'Moondream2'),
        ('microsoft/Phi-3-vision-128k-instruct', 'Phi3Vision'),
    ])
    def test_known_ids_route_to_their_implementation(self, model_id,
                                                     expected_class_name):
        assert get_model_class_location(model_id)[1] == expected_class_name

    def test_only_the_blip_family_uses_the_generic_base_class(self):
        """
        InstructBLIP and BLIP-2 need no specialisation and run through the
        generic path. Anything else landing there is an entry whose
        implementation is missing.
        """
        generic = [model_id for model_id in RECOMMENDED_MODELS + LEGACY_MODELS
                   if get_model_class_location(model_id)[1]
                   == 'AutoCaptioningModel']
        assert all('blip' in model_id.lower() for model_id in generic), generic

    def test_the_recommended_group_has_no_generic_entries(self):
        for model_id in RECOMMENDED_MODELS:
            assert (get_model_class_location(model_id)[1]
                    != 'AutoCaptioningModel'), model_id

    def test_routing_ignores_case(self):
        assert (get_model_class_location('QWEN/QWEN3-VL-8B-INSTRUCT')[1]
                == 'Qwen3Vl')

    def test_a_local_path_still_routes_by_name(self):
        assert (get_model_class_location(
            '/models/wd-eva02-large-tagger-v3')[1] == 'WdTagger')

    def test_a_pixai_tagger_is_not_captured_by_the_wd_pattern(self):
        assert (get_model_class_location('pixai-labs/pixai-tagger-v0.9')[1]
                == 'PixaiTagger')

    def test_an_unknown_id_falls_back_to_the_base_class(self):
        assert (get_model_class_location('some/unknown-model')[1]
                == 'AutoCaptioningModel')

    def test_every_routed_module_exists(self):
        """The lazy import means a typo'd module path fails only at runtime."""
        import importlib.util

        for model_id in RECOMMENDED_MODELS + LEGACY_MODELS:
            module_name, _ = get_model_class_location(model_id)
            assert importlib.util.find_spec(module_name) is not None, \
                module_name
