"""
Tokenizers for the caption profiles.

Each profile counts tokens against the text encoder its target model actually
uses: CLIP for SDXL and Illustrious, T5 for FLUX.1, Qwen3 for FLUX.2 Klein.
Only the CLIP tokenizer ships with the app; the others are fetched once and
cached under the user's app data directory, the same way tag vocabularies are.

When a tokenizer is not available - no network, nothing cached yet - the count
falls back to CLIP, and `TokenizerLoadResult.is_exact` is False so the UI can
say the number is an estimate rather than quietly presenting a CLIP count as
if it came from T5.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QStandardPaths

from utils.caption_profiles import ProfileConfig, TokenEncoder
from utils.utils import get_resource_path

# Bundled with the application.
CLIP_TOKENIZER_DIRECTORY_PATH = Path('clip-vit-base-patch32')

# Hugging Face repositories holding the tokenizer for each encoder. These are
# the text encoders the target models use, so the counts match what the
# trainer will see.
ENCODER_TOKENIZER_IDS = {
    TokenEncoder.T5: 'google/t5-v1_1-xxl',
    TokenEncoder.QWEN: 'Qwen/Qwen3-8B',
}

# Only the tokenizer files are needed, never the weights.
TOKENIZER_FILE_NAMES = ('tokenizer_config.json', 'tokenizer.json',
                        'special_tokens_map.json', 'spiece.model',
                        'vocab.json', 'merges.txt')


@dataclass(frozen=True)
class TokenizerLoadResult:
    tokenizer: object | None
    encoder: TokenEncoder
    # False when a fallback stood in for the profile's real encoder, so counts
    # are approximate.
    is_exact: bool
    message: str = ''


def get_tokenizers_directory() -> Path:
    base = Path(QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.AppDataLocation))
    path = base / 'tokenizers'
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_cached_tokenizer_path(encoder: TokenEncoder) -> Path | None:
    """The cached directory for an encoder, if it has been downloaded."""
    tokenizer_id = ENCODER_TOKENIZER_IDS.get(encoder)
    if tokenizer_id is None:
        return None
    path = get_tokenizers_directory() / tokenizer_id.replace('/', '--')
    return path if (path / 'tokenizer_config.json').is_file() else None


def download_tokenizer(encoder: TokenEncoder) -> Path:
    """
    Fetch an encoder's tokenizer files into the cache. Raises if the encoder
    has no configured repository or the download fails.
    """
    from huggingface_hub import hf_hub_download

    tokenizer_id = ENCODER_TOKENIZER_IDS.get(encoder)
    if tokenizer_id is None:
        raise ValueError(f'No tokenizer is configured for {encoder}.')
    destination = get_tokenizers_directory() / tokenizer_id.replace('/', '--')
    destination.mkdir(parents=True, exist_ok=True)
    downloaded_any = False
    for file_name in TOKENIZER_FILE_NAMES:
        try:
            downloaded_path = hf_hub_download(tokenizer_id, filename=file_name)
        except Exception:
            # Repositories carry different subsets; a missing optional file is
            # not a failure as long as something arrives.
            continue
        (destination / file_name).write_bytes(
            Path(downloaded_path).read_bytes())
        downloaded_any = True
    if not downloaded_any:
        raise OSError(f'No tokenizer files could be downloaded for '
                      f'{tokenizer_id}.')
    return destination


def load_clip_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(
        get_resource_path(CLIP_TOKENIZER_DIRECTORY_PATH))


def load_tokenizer_for_profile(profile: ProfileConfig,
                               allow_download: bool = False
                               ) -> TokenizerLoadResult:
    """
    Load the tokenizer for a profile's encoder, falling back to the bundled
    CLIP tokenizer and saying so rather than silently substituting it.
    """
    encoder = profile.encoder
    if encoder == TokenEncoder.CLIP:
        return TokenizerLoadResult(load_clip_tokenizer(), TokenEncoder.CLIP,
                                   is_exact=True)

    from transformers import AutoTokenizer

    path = get_cached_tokenizer_path(encoder)
    if path is None and allow_download:
        try:
            path = download_tokenizer(encoder)
        except Exception as exception:
            print(f'Failed to download the {encoder.value} tokenizer: '
                  f'{exception}')
            path = None
    if path is not None:
        try:
            return TokenizerLoadResult(
                AutoTokenizer.from_pretrained(str(path)), encoder,
                is_exact=True)
        except Exception as exception:
            print(f'Failed to load the cached {encoder.value} tokenizer: '
                  f'{exception}')

    tokenizer_id = ENCODER_TOKENIZER_IDS.get(encoder, encoder.value)
    return TokenizerLoadResult(
        load_clip_tokenizer(), TokenEncoder.CLIP, is_exact=False,
        message=(f'Counting with the CLIP tokenizer: the {encoder.value} '
                 f'tokenizer ({tokenizer_id}) is not downloaded yet. Use '
                 f'Tools > Download Token Counter to fetch it.'))
