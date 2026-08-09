"""
Tests for `auto_captioning.transformers_compat`.

These run without transformers installed: the module is exercised against
stub modules standing in for each API shape the library has had.
"""
import sys
import types

import pytest

from auto_captioning import transformers_compat


@pytest.fixture
def fake_transformers(monkeypatch):
    """Install a stub `transformers` module with a chosen API surface."""
    def factory(version='4.57.6', auto_classes=('AutoModelForImageTextToText',),
                from_pretrained_argument='dtype'):
        module = types.ModuleType('transformers')
        module.__version__ = version
        for class_name in auto_classes:
            setattr(module, class_name, type(class_name, (), {}))

        if from_pretrained_argument == 'torch_dtype':
            def from_pretrained(cls, name, torch_dtype=None, **kwargs):
                ...
        else:
            def from_pretrained(cls, name, dtype=None, **kwargs):
                ...
        module.PreTrainedModel = type(
            'PreTrainedModel', (), {'from_pretrained': from_pretrained})

        monkeypatch.setitem(sys.modules, 'transformers', module)
        return module
    return factory


class TestVersionComparison:
    @pytest.mark.parametrize('installed,minimum,expected', [
        ('4.57.6', '4.57', True),
        ('4.57.0', '4.57', True),
        ('4.48.3', '4.57', False),
        ('5.14.1', '4.57', True),
        ('4.9.0', '4.57', False),
        ('5.0.0', '5.0', True),
        ('4.57.0.dev0', '4.57', True),
        ('4.56.2', '4.57', False),
    ])
    def test_comparisons(self, installed, minimum, expected):
        assert transformers_compat.is_version_at_least(
            minimum, installed) is expected

    def test_a_double_digit_minor_beats_a_single_digit_one(self):
        """String comparison would rank 4.9 above 4.57."""
        assert transformers_compat.is_version_at_least('4.57', '4.100.0')
        assert not transformers_compat.is_version_at_least('4.57', '4.9')

    def test_local_version_suffixes_are_ignored(self):
        assert transformers_compat.is_version_at_least('4.57', '4.57.1+local')

    def test_an_unparseable_version_is_treated_as_too_old(self):
        assert not transformers_compat.is_version_at_least('4.57', 'unknown')

    def test_the_installed_version_is_read_from_the_module(self,
                                                          fake_transformers):
        fake_transformers(version='4.57.6')
        assert transformers_compat.get_transformers_version() == '4.57.6'
        assert transformers_compat.is_version_at_least('4.57')


def test_the_error_message_names_the_model_and_both_versions():
    message = transformers_compat.get_version_error_message(
        'Qwen3-VL', '4.57', version='4.48.3')
    assert 'Qwen3-VL' in message
    assert '4.57' in message
    assert '4.48.3' in message
    assert 'requirements.txt' in message


class TestModelClassResolution:
    def test_the_current_auto_class_is_preferred(self, fake_transformers):
        module = fake_transformers(
            auto_classes=('AutoModelForImageTextToText',
                          'AutoModelForVision2Seq'))
        assert (transformers_compat.get_image_text_model_class()
                is module.AutoModelForImageTextToText)

    def test_the_older_auto_class_is_used_when_it_is_all_there_is(
            self, fake_transformers):
        module = fake_transformers(auto_classes=('AutoModelForVision2Seq',))
        assert (transformers_compat.get_image_text_model_class()
                is module.AutoModelForVision2Seq)

    def test_neither_class_raises_a_clear_error(self, fake_transformers):
        fake_transformers(auto_classes=())
        with pytest.raises(ImportError, match='AutoModelForImageTextToText'):
            transformers_compat.get_image_text_model_class()


class TestDtypeArgument:
    def test_the_legacy_name_is_used_when_the_signature_takes_it(
            self, fake_transformers):
        fake_transformers(from_pretrained_argument='torch_dtype')
        assert transformers_compat.get_dtype_arguments('bf16') == {
            'torch_dtype': 'bf16'}

    def test_the_current_name_is_used_otherwise(self, fake_transformers):
        fake_transformers(from_pretrained_argument='dtype')
        assert transformers_compat.get_dtype_arguments('bf16') == {
            'dtype': 'bf16'}

    def test_a_signature_that_cannot_be_inspected_falls_back_to_the_version(
            self, fake_transformers, monkeypatch):
        fake_transformers(version='4.57.6')
        monkeypatch.setattr(
            transformers_compat.inspect, 'signature',
            lambda *args, **kwargs: (_ for _ in ()).throw(ValueError()))
        assert transformers_compat.get_dtype_argument_name() == 'dtype'

    @pytest.mark.parametrize('version,expected', [
        ('4.48.3', 'torch_dtype'),
        ('4.55.4', 'torch_dtype'),
        ('4.56.0', 'dtype'),
        ('4.56.2', 'dtype'),
        ('4.57.6', 'dtype'),
        ('5.14.1', 'dtype'),
    ])
    def test_the_version_decides_when_the_signature_names_neither(
            self, fake_transformers, monkeypatch, version, expected):
        """
        `from_pretrained` absorbs both names into **kwargs, so the signature
        cannot answer. Verified against the real libraries: 4.55.4 accepts
        only `torch_dtype`, 4.56.2 deprecates it in favour of `dtype`.
        """
        module = fake_transformers(version=version)

        def from_pretrained(cls, name, **kwargs):
            ...

        module.PreTrainedModel.from_pretrained = from_pretrained
        assert transformers_compat.get_dtype_argument_name() == expected
