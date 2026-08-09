"""
Tests for `utils.tokenizers`.

The point of this module is honesty about which encoder produced a count: a
FLUX profile must not present a CLIP count as though it came from T5 or Qwen3.
"""
import sys
import types

import pytest

from utils import tokenizers
from utils.caption_profiles import CaptionProfile, TokenEncoder, \
    get_profile_config


class StubTokenizer:
    def __init__(self, source):
        self.source = source


@pytest.fixture
def stub_transformers(monkeypatch):
    """A stand-in `transformers.AutoTokenizer` that records what it loaded."""
    module = types.ModuleType('transformers')

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(source, *args, **kwargs):
            return StubTokenizer(str(source))

    module.AutoTokenizer = AutoTokenizer
    monkeypatch.setitem(sys.modules, 'transformers', module)
    return module


@pytest.fixture
def cache_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(tokenizers, 'get_tokenizers_directory',
                        lambda: tmp_path)
    return tmp_path


def write_cached_tokenizer(cache_directory, encoder):
    path = (cache_directory
            / tokenizers.ENCODER_TOKENIZER_IDS[encoder].replace('/', '--'))
    path.mkdir(parents=True, exist_ok=True)
    (path / 'tokenizer_config.json').write_text('{}', encoding='utf-8')
    return path


class TestCacheLookup:
    def test_an_uncached_encoder_reports_nothing(self, cache_directory):
        assert tokenizers.get_cached_tokenizer_path(TokenEncoder.T5) is None

    def test_a_cached_encoder_is_found(self, cache_directory):
        expected = write_cached_tokenizer(cache_directory, TokenEncoder.T5)
        assert (tokenizers.get_cached_tokenizer_path(TokenEncoder.T5)
                == expected)

    def test_a_directory_without_a_config_does_not_count_as_cached(
            self, cache_directory):
        path = (cache_directory
                / tokenizers.ENCODER_TOKENIZER_IDS[
                    TokenEncoder.QWEN].replace('/', '--'))
        path.mkdir(parents=True)
        assert tokenizers.get_cached_tokenizer_path(TokenEncoder.QWEN) is None

    def test_clip_has_no_download_entry_because_it_is_bundled(self):
        assert TokenEncoder.CLIP not in tokenizers.ENCODER_TOKENIZER_IDS


class TestLoadingForAProfile:
    def test_clip_profiles_load_the_bundled_tokenizer_exactly(
            self, stub_transformers, cache_directory):
        for profile in (CaptionProfile.SDXL, CaptionProfile.ILLUSTRIOUS):
            result = tokenizers.load_tokenizer_for_profile(
                get_profile_config(profile))
            assert result.is_exact
            assert result.encoder is TokenEncoder.CLIP
            assert 'clip-vit-base-patch32' in result.tokenizer.source

    def test_a_cached_encoder_is_used_and_marked_exact(self, stub_transformers,
                                                       cache_directory):
        write_cached_tokenizer(cache_directory, TokenEncoder.QWEN)
        result = tokenizers.load_tokenizer_for_profile(
            get_profile_config(CaptionProfile.FLUX2_KLEIN))
        assert result.is_exact
        assert result.encoder is TokenEncoder.QWEN
        assert 'Qwen' in result.tokenizer.source

    def test_a_missing_encoder_falls_back_to_clip_and_says_so(
            self, stub_transformers, cache_directory):
        result = tokenizers.load_tokenizer_for_profile(
            get_profile_config(CaptionProfile.FLUX1_KREA))
        assert not result.is_exact
        assert result.encoder is TokenEncoder.CLIP
        assert result.tokenizer is not None
        assert 't5' in result.message.lower()

    def test_the_fallback_message_names_the_repository(self, stub_transformers,
                                                       cache_directory):
        result = tokenizers.load_tokenizer_for_profile(
            get_profile_config(CaptionProfile.FLUX2_KLEIN))
        assert tokenizers.ENCODER_TOKENIZER_IDS[TokenEncoder.QWEN] \
            in result.message

    def test_a_corrupt_cache_falls_back_instead_of_raising(
            self, stub_transformers, cache_directory, monkeypatch):
        write_cached_tokenizer(cache_directory, TokenEncoder.T5)

        def explode(source, *args, **kwargs):
            if 't5' in str(source):
                raise OSError('corrupt')
            return StubTokenizer(str(source))

        monkeypatch.setattr(stub_transformers.AutoTokenizer, 'from_pretrained',
                            explode)
        result = tokenizers.load_tokenizer_for_profile(
            get_profile_config(CaptionProfile.FLUX1_KREA))
        assert not result.is_exact
        assert result.tokenizer is not None

    def test_a_failed_download_falls_back_rather_than_raising(
            self, stub_transformers, cache_directory, monkeypatch):
        monkeypatch.setattr(
            tokenizers, 'download_tokenizer',
            lambda encoder: (_ for _ in ()).throw(OSError('offline')))
        result = tokenizers.load_tokenizer_for_profile(
            get_profile_config(CaptionProfile.FLUX1_KREA),
            allow_download=True)
        assert not result.is_exact
        assert result.tokenizer is not None

    def test_nothing_is_downloaded_unless_asked(self, stub_transformers,
                                                cache_directory, monkeypatch):
        calls = []
        monkeypatch.setattr(tokenizers, 'download_tokenizer', calls.append)
        tokenizers.load_tokenizer_for_profile(
            get_profile_config(CaptionProfile.FLUX1_KREA))
        assert calls == []


def test_an_unconfigured_encoder_cannot_be_downloaded():
    with pytest.raises(ValueError):
        tokenizers.download_tokenizer(TokenEncoder.CLIP)
