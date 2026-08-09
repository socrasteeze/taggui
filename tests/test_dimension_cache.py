"""Tests for `utils.dimension_cache`."""
import json

from utils.dimension_cache import DimensionCache


def make_file(tmp_path, name='image.jpg', content=b'x'):
    path = tmp_path / name
    path.write_bytes(content)
    return path


def test_a_miss_returns_none(tmp_path):
    cache = DimensionCache(tmp_path / 'cache.json')
    assert cache.get(make_file(tmp_path)) is None


def test_a_stored_value_round_trips(tmp_path):
    cache = DimensionCache(tmp_path / 'cache.json')
    path = make_file(tmp_path)
    cache.set(path, (1920, 1080))
    assert cache.get(path) == (1920, 1080)


def test_values_survive_being_saved_and_reloaded(tmp_path):
    cache_path = tmp_path / 'cache.json'
    path = make_file(tmp_path)
    cache = DimensionCache(cache_path)
    cache.set(path, (800, 600))
    cache.save()

    assert DimensionCache(cache_path).get(path) == (800, 600)


def test_editing_the_file_invalidates_its_entry(tmp_path):
    """The key includes mtime and size, so a changed file misses the cache."""
    cache = DimensionCache(tmp_path / 'cache.json')
    path = make_file(tmp_path)
    cache.set(path, (800, 600))
    path.write_bytes(b'much longer content than before')
    assert cache.get(path) is None


def test_a_missing_file_is_not_cached(tmp_path):
    cache = DimensionCache(tmp_path / 'cache.json')
    missing = tmp_path / 'gone.jpg'
    cache.set(missing, (100, 100))
    assert cache.get(missing) is None


def test_storing_none_is_ignored(tmp_path):
    cache = DimensionCache(tmp_path / 'cache.json')
    path = make_file(tmp_path)
    cache.set(path, None)
    assert cache.get(path) is None


def test_saving_a_clean_cache_writes_nothing(tmp_path):
    cache_path = tmp_path / 'cache.json'
    DimensionCache(cache_path).save()
    assert not cache_path.exists()


def test_a_corrupt_cache_file_is_ignored_rather_than_raising(tmp_path):
    cache_path = tmp_path / 'cache.json'
    cache_path.write_text('{not json', encoding='utf-8')
    cache = DimensionCache(cache_path)
    assert cache.get(make_file(tmp_path)) is None


class TestPruning:
    """
    Entries are keyed by path, mtime and size, so editing an image leaves its
    old entry behind. Without pruning the file grows for the life of the
    install and is parsed in full at every startup.
    """

    def test_a_small_cache_is_left_alone(self, tmp_path):
        cache = DimensionCache(tmp_path / 'cache.json', maximum_entry_count=10)
        for index in range(5):
            cache.set(make_file(tmp_path, f'image_{index}.jpg',
                                b'x' * index), (10, 10))
        cache.prune()
        assert len(cache._data) == 5

    def test_an_oversized_cache_keeps_only_what_this_session_used(self,
                                                                  tmp_path):
        cache_path = tmp_path / 'cache.json'
        stale = {f'/old/path/{index}|0|0': {'w': 1, 'h': 1}
                 for index in range(20)}
        cache_path.write_text(json.dumps(stale), encoding='utf-8')

        cache = DimensionCache(cache_path, maximum_entry_count=10)
        current = make_file(tmp_path, 'current.jpg')
        cache.set(current, (800, 600))
        cache.save()

        reloaded = DimensionCache(cache_path, maximum_entry_count=10)
        assert reloaded.get(current) == (800, 600)
        assert len(reloaded._data) == 1

    def test_entries_read_this_session_survive_pruning(self, tmp_path):
        cache_path = tmp_path / 'cache.json'
        kept = make_file(tmp_path, 'kept.jpg')
        seed = DimensionCache(cache_path, maximum_entry_count=1000)
        seed.set(kept, (100, 100))
        seed.save()

        cache = DimensionCache(cache_path, maximum_entry_count=0)
        assert cache.get(kept) == (100, 100)     # marks it as used
        cache.prune()
        assert cache.get(kept) == (100, 100)


def test_two_files_with_identical_content_get_separate_entries(tmp_path):
    cache = DimensionCache(tmp_path / 'cache.json')
    first = make_file(tmp_path, 'a.jpg', b'same')
    second = make_file(tmp_path, 'b.jpg', b'same')
    cache.set(first, (10, 10))
    cache.set(second, (20, 20))
    assert cache.get(first) == (10, 10)
    assert cache.get(second) == (20, 20)
