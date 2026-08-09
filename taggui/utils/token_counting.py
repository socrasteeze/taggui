"""
Caption token counting.

What the user cares about is how much of an encoder's prompt window a caption
uses, which excludes the wrapper tokens the tokenizer adds around it. Those
differ per encoder - CLIP adds two (BOS/EOS), T5 adds one (EOS), and byte-level
tokenizers such as GPT-2 add none - so the overhead is measured from the
tokenizer itself instead of being assumed.
"""
from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

# Tokenizers are reused across every image, so the overhead is worth caching.
# Keyed on the tokenizer itself rather than its `id()`: ids are reused once an
# object is collected, which would hand one tokenizer's overhead to its
# replacement and silently shift every count.
_special_token_counts: 'weakref.WeakKeyDictionary' = \
    weakref.WeakKeyDictionary()


def _measure_special_token_count(tokenizer) -> int:
    try:
        return len(tokenizer('').input_ids)
    except Exception:
        # A tokenizer that cannot encode an empty string tells us nothing
        # about its overhead; counting the raw tokens is the safer error.
        return 0


def get_special_token_count(tokenizer: 'PreTrainedTokenizerBase') -> int:
    """Number of tokens the tokenizer adds to an empty string."""
    try:
        count = _special_token_counts.get(tokenizer)
    except TypeError:
        # Not weak-referenceable or not hashable; measure every time.
        return _measure_special_token_count(tokenizer)
    if count is None:
        count = _measure_special_token_count(tokenizer)
        try:
            _special_token_counts[tokenizer] = count
        except TypeError:
            pass
    return count


def count_caption_tokens(caption: str,
                         tokenizer: 'PreTrainedTokenizerBase | None') -> int:
    """
    Count the tokens a caption contributes to the prompt window.

    Without a tokenizer - during startup, before the background load finishes -
    fall back to a whitespace word count so the UI still shows something in the
    right ballpark.
    """
    if tokenizer is None:
        return len(caption.split()) if caption else 0
    token_count = len(tokenizer(caption).input_ids)
    return max(token_count - get_special_token_count(tokenizer), 0)
