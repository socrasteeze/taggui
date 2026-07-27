"""Caption profiles for SDXL / Illustrious / FLUX workflows."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CaptionProfile(str, Enum):
    SDXL = 'SDXL (general)'
    ILLUSTRIOUS = 'Illustrious XL'
    FLUX2_KLEIN = 'FLUX.2 Klein 9B'
    FLUX1_KREA = 'FLUX.1 Krea'


class TokenEncoder(str, Enum):
    CLIP = 'clip'
    T5 = 't5'
    QWEN = 'qwen'


@dataclass(frozen=True)
class ProfileConfig:
    profile: CaptionProfile
    encoder: TokenEncoder
    token_limit: int
    vocab_csv: str | None
    default_prompt_hint: str
    trigger_mode: str  # 'first_tag' | 'embedded' | 'none'


PROFILE_CONFIGS: dict[CaptionProfile, ProfileConfig] = {
    CaptionProfile.SDXL: ProfileConfig(
        profile=CaptionProfile.SDXL,
        encoder=TokenEncoder.CLIP,
        token_limit=75,
        vocab_csv='sdxl_quality.csv',
        default_prompt_hint='Short natural language or hybrid tags+NL',
        trigger_mode='first_tag',
    ),
    CaptionProfile.ILLUSTRIOUS: ProfileConfig(
        profile=CaptionProfile.ILLUSTRIOUS,
        encoder=TokenEncoder.CLIP,
        token_limit=75,
        vocab_csv='danbooru.csv',
        default_prompt_hint='Danbooru-style tags',
        trigger_mode='first_tag',
    ),
    CaptionProfile.FLUX2_KLEIN: ProfileConfig(
        profile=CaptionProfile.FLUX2_KLEIN,
        encoder=TokenEncoder.QWEN,
        token_limit=512,
        vocab_csv=None,
        default_prompt_hint='Rich natural-language sentences',
        trigger_mode='embedded',
    ),
    CaptionProfile.FLUX1_KREA: ProfileConfig(
        profile=CaptionProfile.FLUX1_KREA,
        encoder=TokenEncoder.T5,
        token_limit=512,
        vocab_csv=None,
        default_prompt_hint='1–3 descriptive natural-language sentences',
        trigger_mode='embedded',
    ),
}


def get_profile_config(profile_name: str | CaptionProfile) -> ProfileConfig:
    if isinstance(profile_name, CaptionProfile):
        return PROFILE_CONFIGS[profile_name]
    try:
        return PROFILE_CONFIGS[CaptionProfile(profile_name)]
    except (ValueError, KeyError):
        return PROFILE_CONFIGS[CaptionProfile.SDXL]
