"""Tests for `utils.caption_profiles`."""
from utils.caption_profiles import (PROFILE_CONFIGS, CaptionProfile,
                                    TokenEncoder, get_profile_config)


def test_every_profile_has_a_config():
    for profile in CaptionProfile:
        assert profile in PROFILE_CONFIGS
        assert PROFILE_CONFIGS[profile].profile is profile


def test_lookup_accepts_the_enum_member():
    config = get_profile_config(CaptionProfile.ILLUSTRIOUS)
    assert config.encoder is TokenEncoder.CLIP
    assert config.token_limit == 75


def test_lookup_accepts_the_display_string():
    config = get_profile_config('FLUX.2 Klein 9B')
    assert config.profile is CaptionProfile.FLUX2_KLEIN
    assert config.encoder is TokenEncoder.QWEN


def test_an_unknown_name_falls_back_to_sdxl():
    assert get_profile_config('nonsense').profile is CaptionProfile.SDXL
    assert get_profile_config('').profile is CaptionProfile.SDXL


def test_clip_profiles_use_the_75_token_chunk():
    for profile in (CaptionProfile.SDXL, CaptionProfile.ILLUSTRIOUS):
        config = get_profile_config(profile)
        assert config.encoder is TokenEncoder.CLIP
        assert config.token_limit == 75


def test_flux_profiles_use_a_512_token_window():
    for profile in (CaptionProfile.FLUX2_KLEIN, CaptionProfile.FLUX1_KREA):
        config = get_profile_config(profile)
        assert config.token_limit == 512
        assert config.encoder is not TokenEncoder.CLIP


def test_tag_profiles_carry_a_vocab_and_flux_profiles_do_not():
    assert get_profile_config(CaptionProfile.ILLUSTRIOUS).vocab_csv
    assert get_profile_config(CaptionProfile.FLUX2_KLEIN).vocab_csv is None


def test_trigger_modes_match_the_target_model_convention():
    # Tag-based models put the trigger first; FLUX embeds it in a sentence.
    assert get_profile_config(CaptionProfile.SDXL).trigger_mode == 'first_tag'
    assert (get_profile_config(CaptionProfile.ILLUSTRIOUS).trigger_mode
            == 'first_tag')
    assert (get_profile_config(CaptionProfile.FLUX2_KLEIN).trigger_mode
            == 'embedded')
    assert (get_profile_config(CaptionProfile.FLUX1_KREA).trigger_mode
            == 'embedded')


def test_configs_are_immutable():
    import dataclasses
    import pytest

    config = get_profile_config(CaptionProfile.SDXL)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.token_limit = 1
