"""
Tests for `utils.tag_vocab` — the a1111-tagcomplete CSV loader.

The CSV columns are: name, category, post count, comma-separated aliases.
Danbooru categories are 0 general, 1 artist, 3 copyright/series, 4 character.
"""
import pytest

from utils.tag_vocab import TagVocab


def write_csv(tmp_path, rows, name='danbooru.csv'):
    path = tmp_path / name
    path.write_text('\n'.join(rows), encoding='utf-8')
    return path


def test_loading_returns_the_tag_count(tmp_path):
    path = write_csv(tmp_path, ['1girl,0,5000000,', 'solo,0,4000000,'])
    vocab = TagVocab()
    assert vocab.load_csv(path) == 2
    assert vocab.source_name == 'danbooru.csv'


def test_categories_and_post_counts_are_parsed(tmp_path):
    path = write_csv(tmp_path, ['hatsune miku,4,900000,',
                                'vocaloid,3,800000,'])
    vocab = TagVocab()
    vocab.load_csv(path)
    by_name = {tag.name: tag for tag in vocab.tags}
    assert by_name['hatsune miku'].category == 4
    assert by_name['hatsune miku'].post_count == 900000
    assert by_name['vocaloid'].category == 3


def test_aliases_are_split_and_resolve_to_the_canonical_name(tmp_path):
    path = write_csv(tmp_path, ['hatsune miku,4,900000,"miku,hatsune_miku"'])
    vocab = TagVocab()
    vocab.load_csv(path)
    assert vocab.resolve('miku') == 'hatsune miku'
    assert vocab.resolve('hatsune_miku') == 'hatsune miku'


def test_resolving_an_unknown_term_returns_it_unchanged(tmp_path):
    vocab = TagVocab()
    vocab.load_csv(write_csv(tmp_path, ['1girl,0,5000000,']))
    assert vocab.resolve('not a real tag') == 'not a real tag'


def test_resolution_is_case_insensitive(tmp_path):
    vocab = TagVocab()
    vocab.load_csv(write_csv(tmp_path, ['Hatsune Miku,4,900000,']))
    assert vocab.resolve('hatsune miku') == 'Hatsune Miku'


def test_a_header_row_and_blank_rows_are_skipped(tmp_path):
    path = write_csv(tmp_path, ['name,category,count,aliases', '',
                                '1girl,0,5000000,'])
    vocab = TagVocab()
    assert vocab.load_csv(path) == 1


def test_malformed_numeric_columns_default_to_zero(tmp_path):
    path = write_csv(tmp_path, ['weird,notanumber,alsonot,'])
    vocab = TagVocab()
    vocab.load_csv(path)
    assert vocab.tags[0].category == 0
    assert vocab.tags[0].post_count == 0


def test_a_row_with_only_a_name_still_loads(tmp_path):
    vocab = TagVocab()
    vocab.load_csv(write_csv(tmp_path, ['lonely']))
    assert vocab.tags[0].name == 'lonely'


class TestSuggest:
    def test_an_empty_prefix_suggests_nothing(self, tmp_path):
        vocab = TagVocab()
        vocab.load_csv(write_csv(tmp_path, ['1girl,0,5000000,']))
        assert vocab.suggest('') == []

    def test_matches_are_ordered_by_post_count(self, tmp_path):
        path = write_csv(tmp_path, ['sword,0,100,', 'swordsman,0,900,',
                                    'sword art online,3,500,'])
        vocab = TagVocab()
        vocab.load_csv(path)
        assert [tag.name for tag in vocab.suggest('sword')] == [
            'swordsman', 'sword art online', 'sword']

    def test_a_single_character_prefix_matches(self, tmp_path):
        vocab = TagVocab()
        vocab.load_csv(write_csv(tmp_path, ['solo,0,100,', 'smile,0,200,']))
        assert {tag.name for tag in vocab.suggest('s')} == {'solo', 'smile'}

    def test_an_alias_prefix_finds_the_canonical_tag(self, tmp_path):
        path = write_csv(tmp_path, ['hatsune miku,4,900000,"miku"'])
        vocab = TagVocab()
        vocab.load_csv(path)
        assert [tag.name for tag in vocab.suggest('mik')] == ['hatsune miku']

    def test_a_tag_matching_by_both_name_and_alias_appears_once(self,
                                                               tmp_path):
        path = write_csv(tmp_path, ['sword,0,100,"swords,swordy"'])
        vocab = TagVocab()
        vocab.load_csv(path)
        assert len(vocab.suggest('sword')) == 1

    def test_the_limit_is_respected(self, tmp_path):
        rows = [f'tag{index},0,{index},' for index in range(30)]
        vocab = TagVocab()
        vocab.load_csv(write_csv(tmp_path, rows))
        assert len(vocab.suggest('tag', limit=5)) == 5

    def test_suggestions_are_case_insensitive(self, tmp_path):
        vocab = TagVocab()
        vocab.load_csv(write_csv(tmp_path, ['Smile,0,200,']))
        assert [tag.name for tag in vocab.suggest('smi')] == ['Smile']


def test_loading_a_second_csv_replaces_the_first(tmp_path):
    vocab = TagVocab()
    vocab.load_csv(write_csv(tmp_path, ['1girl,0,5,'], 'a.csv'))
    vocab.load_csv(write_csv(tmp_path, ['solo,0,5,'], 'b.csv'))
    assert [tag.name for tag in vocab.tags] == ['solo']
    assert vocab.source_name == 'b.csv'


def test_clearing_empties_everything(tmp_path):
    vocab = TagVocab()
    vocab.load_csv(write_csv(tmp_path, ['1girl,0,5,"girl"']))
    vocab.clear()
    assert vocab.tags == []
    assert vocab.suggest('1g') == []
    assert vocab.resolve('girl') == 'girl'
    assert vocab.source_name is None


class TestCategoryLookup:
    """
    The Illustrious tag order needs to know which tags are characters and
    which are series. The a1111 CSV already carries that in its category
    column, so it is the source rather than the WD model, which has no series
    category at all.
    """

    @pytest.fixture
    def vocab(self, tmp_path):
        vocab = TagVocab()
        vocab.load_csv(write_csv(tmp_path, [
            '1girl,0,5000000,',
            'smile,0,4000000,',
            'hatsune miku,4,900000,"miku"',
            'kagamine rin,4,500000,',
            'vocaloid,3,800000,',
            'some artist,1,1000,',
        ]))
        return vocab

    def test_character_tags_are_returned(self, vocab):
        assert vocab.get_names_in_categories({4}) == {
            'hatsune miku', 'miku', 'kagamine rin'}

    def test_series_tags_are_returned(self, vocab):
        assert vocab.get_names_in_categories({3}) == {'vocaloid'}

    def test_several_categories_can_be_requested_at_once(self, vocab):
        assert vocab.get_names_in_categories({3, 4}) == {
            'hatsune miku', 'miku', 'kagamine rin', 'vocaloid'}

    def test_general_tags_are_excluded(self, vocab):
        characters = vocab.get_names_in_categories({4})
        assert '1girl' not in characters
        assert 'smile' not in characters

    def test_names_are_casefolded(self, tmp_path):
        vocab = TagVocab()
        vocab.load_csv(write_csv(tmp_path, ['Hatsune Miku,4,900000,']))
        assert vocab.get_names_in_categories({4}) == {'hatsune miku'}

    def test_an_unused_category_returns_nothing(self, vocab):
        assert vocab.get_names_in_categories({99}) == set()

    def test_an_empty_vocabulary_returns_nothing(self):
        assert TagVocab().get_names_in_categories({4}) == set()


def test_the_builtin_sdxl_list_loads_without_a_file():
    vocab = TagVocab()
    assert vocab.load_builtin_sdxl() > 0
    assert vocab.source_name == 'sdxl_quality.csv'
    assert [tag.name for tag in vocab.suggest('master')] == ['masterpiece']
