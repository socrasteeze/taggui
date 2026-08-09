"""
Tests for `models.proxy_image_list_model` — the image list filter.

Filters arrive as the nested lists produced by the filter-text parser, e.g.
`['tag', 'solo']` or `[['tag', 'a'], 'AND', ['tag', 'b']]`.
"""
import pytest

from models.image_list_model import ImageListModel
from models.proxy_image_list_model import ProxyImageListModel


class FakeTokenizer:
    """
    Stand-in for a Hugging Face tokenizer. `special_token_count` mimics the
    wrapping tokens real encoders add: CLIP adds two, T5 adds one, GPT-2 adds
    none.
    """

    def __init__(self, special_token_count: int = 2):
        self.special_token_count = special_token_count
        self.call_count = 0

    def __call__(self, text: str):
        self.call_count += 1
        words = text.split() if text else []
        return type('Encoding', (),
                    {'input_ids': [0] * (len(words)
                                         + self.special_token_count)})()


@pytest.fixture
def proxy(tmp_path, make_image):
    def factory(tag_lists: list[list[str]], tokenizer=None):
        source = ImageListModel(image_list_image_width=200,
                                tag_separator=', ')
        source.beginResetModel()
        source.images = [make_image(f'image_{index}.png', tags)
                         for index, tags in enumerate(tag_lists)]
        source.endResetModel()
        return ProxyImageListModel(source, tokenizer, ', ')
    return factory


def visible_rows(proxy_model) -> int:
    return proxy_model.rowCount()


class TestFiltering:
    def test_no_filter_shows_everything(self, proxy):
        model = proxy([['a'], ['b']])
        assert visible_rows(model) == 2

    def test_a_bare_string_matches_the_caption(self, proxy):
        model = proxy([['red car'], ['blue sky']])
        model.filter = 'red'
        model.invalidate()
        assert visible_rows(model) == 1

    def test_a_bare_string_also_matches_the_path(self, proxy):
        model = proxy([['a'], ['b']])
        model.filter = 'image_1'
        model.invalidate()
        assert visible_rows(model) == 1

    def test_the_tag_term_matches_whole_tags(self, proxy):
        model = proxy([['cat'], ['catalog']])
        model.filter = ['tag', 'cat']
        model.invalidate()
        assert visible_rows(model) == 1

    def test_the_caption_term_matches_substrings(self, proxy):
        model = proxy([['cat'], ['catalog']])
        model.filter = ['caption', 'cat']
        model.invalidate()
        assert visible_rows(model) == 2

    def test_the_name_term_matches_the_file_name(self, proxy):
        model = proxy([['a'], ['b']])
        model.filter = ['name', 'image_0']
        model.invalidate()
        assert visible_rows(model) == 1

    def test_not_inverts_a_term(self, proxy):
        model = proxy([['cat'], ['dog']])
        model.filter = ['NOT', ['tag', 'cat']]
        model.invalidate()
        assert visible_rows(model) == 1

    def test_and_requires_both_terms(self, proxy):
        model = proxy([['cat', 'dog'], ['cat']])
        model.filter = [['tag', 'cat'], 'AND', ['tag', 'dog']]
        model.invalidate()
        assert visible_rows(model) == 1

    def test_or_accepts_either_term(self, proxy):
        model = proxy([['cat'], ['dog'], ['bird']])
        model.filter = [['tag', 'cat'], 'OR', ['tag', 'dog']]
        model.invalidate()
        assert visible_rows(model) == 2

    def test_the_tags_term_compares_tag_counts(self, proxy):
        model = proxy([['a'], ['a', 'b'], ['a', 'b', 'c']])
        model.filter = ['tags', '>', '1']
        model.invalidate()
        assert visible_rows(model) == 2

    def test_the_chars_term_compares_caption_length(self, proxy):
        model = proxy([['ab'], ['abcdefghij']])
        model.filter = ['chars', '>', '5']
        model.invalidate()
        assert visible_rows(model) == 1


class TestTokenCounting:
    def test_the_tokens_term_filters_on_token_count(self, proxy):
        model = proxy([['one'], ['one two three four']],
                      tokenizer=FakeTokenizer())
        model.filter = ['tokens', '>', '2']
        model.invalidate()
        assert visible_rows(model) == 1

    def test_special_tokens_are_excluded_from_the_count(self, proxy):
        """A four-word caption is four tokens, whatever the encoder wraps it in."""
        for special_token_count in (0, 1, 2):
            model = proxy([['one two three four']],
                          tokenizer=FakeTokenizer(special_token_count))
            image = model.sourceModel().images[0]
            assert model._get_token_count(image) == 4

    def test_counts_are_cached_between_lookups(self, proxy):
        tokenizer = FakeTokenizer()
        model = proxy([['one two']], tokenizer=tokenizer)
        image = model.sourceModel().images[0]
        model._get_token_count(image)
        calls_after_first = tokenizer.call_count
        model._get_token_count(image)
        assert tokenizer.call_count == calls_after_first

    def test_changing_the_tokenizer_invalidates_cached_counts(self, proxy):
        model = proxy([['one two three']], tokenizer=FakeTokenizer())
        image = model.sourceModel().images[0]
        assert model._get_token_count(image) == 3
        model.set_tokenizer(FakeTokenizer(special_token_count=0))
        assert image.token_count is None

    def test_changing_the_tokenizer_refreshes_an_active_filter(self, proxy):
        """
        The tokenizer loads in the background, after the window is already
        usable. A `tokens:` filter applied before it arrives has to be
        re-evaluated once the real counts are available.
        """
        model = proxy([['one two three four five']], tokenizer=None)
        model.filter = ['tokens', '>', '3']
        model.invalidate()
        assert visible_rows(model) == 1

        model.set_tokenizer(FakeTokenizer())
        model.filter = ['tokens', '>', '10']
        assert visible_rows(model) == 0

    def test_without_a_tokenizer_the_count_falls_back_to_words(self, proxy):
        model = proxy([['one two three']], tokenizer=None)
        image = model.sourceModel().images[0]
        assert model._get_token_count(image) == 3


def test_is_image_in_filtered_images_matches_the_row_filter(proxy):
    model = proxy([['cat'], ['dog']])
    model.filter = ['tag', 'cat']
    model.invalidate()
    images = model.sourceModel().images
    assert model.is_image_in_filtered_images(images[0])
    assert not model.is_image_in_filtered_images(images[1])
