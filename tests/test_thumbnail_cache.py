"""
Tests for `utils.thumbnail_cache`.

Thumbnails were previously kept on every `Image` for the lifetime of the
directory, so scrolling a large dataset retained a decoded pixmap per image.
At the default 200 px width that is on the order of gigabytes for a 50k-image
set, which is exactly the size of dataset this fork targets. Memory should
track the viewport, not the dataset.
"""
import pytest

from utils.thumbnail_cache import ThumbnailCache


def test_a_stored_thumbnail_can_be_read_back():
    cache = ThumbnailCache(maximum_size=4)
    cache.set('a', 'thumb-a')
    assert cache.get('a') == 'thumb-a'


def test_a_missing_key_returns_none():
    assert ThumbnailCache(maximum_size=4).get('nothing') is None


def test_the_cache_never_exceeds_its_maximum():
    cache = ThumbnailCache(maximum_size=3)
    for index in range(10):
        cache.set(f'key-{index}', index)
    assert len(cache) == 3


def test_the_least_recently_used_entry_is_evicted_first():
    cache = ThumbnailCache(maximum_size=2)
    cache.set('a', 1)
    cache.set('b', 2)
    cache.set('c', 3)
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.get('c') == 3


def test_reading_an_entry_makes_it_recently_used():
    cache = ThumbnailCache(maximum_size=2)
    cache.set('a', 1)
    cache.set('b', 2)
    cache.get('a')      # 'b' is now the least recently used.
    cache.set('c', 3)
    assert cache.get('a') == 1
    assert cache.get('b') is None


def test_restoring_an_existing_key_does_not_grow_the_cache():
    cache = ThumbnailCache(maximum_size=2)
    cache.set('a', 1)
    cache.set('a', 2)
    assert len(cache) == 1
    assert cache.get('a') == 2


def test_clearing_empties_the_cache():
    cache = ThumbnailCache(maximum_size=4)
    cache.set('a', 1)
    cache.clear()
    assert len(cache) == 0
    assert cache.get('a') is None


def test_a_single_entry_cache_holds_only_the_newest():
    cache = ThumbnailCache(maximum_size=1)
    cache.set('a', 1)
    cache.set('b', 2)
    assert cache.get('a') is None
    assert cache.get('b') == 2


def test_the_maximum_is_at_least_one():
    """A zero or negative size would evict every entry as it was stored."""
    cache = ThumbnailCache(maximum_size=0)
    cache.set('a', 1)
    assert cache.get('a') == 1


def test_containment_does_not_change_the_usage_order():
    cache = ThumbnailCache(maximum_size=2)
    cache.set('a', 1)
    cache.set('b', 2)
    assert 'a' in cache
    cache.set('c', 3)
    # `in` is a lookup, not a use, so 'a' was still the least recent.
    assert cache.get('a') is None


@pytest.mark.parametrize('maximum_size', [1, 2, 8, 64])
def test_the_cache_stays_within_bounds_under_churn(maximum_size):
    cache = ThumbnailCache(maximum_size=maximum_size)
    for index in range(500):
        cache.set(f'key-{index % 100}', index)
        assert len(cache) <= maximum_size
