"""
Tests for `models.image_list_model`.

Several of these are regression tests for defects found while reconciling
`Plan.md` against the code: the Illustrious reorder discarded the tagger's
confidence ordering, and the trigger-token insert skipped images on a
substring match.
"""
import json

import pytest
from PySide6.QtWidgets import QMessageBox

import models.image_list_model as image_list_model_module
from models.image_list_model import ImageListModel, Scope


@pytest.fixture
def model(tmp_path, monkeypatch):
    """An `ImageListModel` whose undo prompts are auto-confirmed."""
    monkeypatch.setattr(
        image_list_model_module, 'get_confirmation_dialog_reply',
        lambda *args, **kwargs: QMessageBox.StandardButton.Yes)
    model = ImageListModel(image_list_image_width=200, tag_separator=', ')
    yield model
    model.tag_writer.flush()


@pytest.fixture
def populated(model, make_image):
    def factory(tag_lists: list[list[str]]) -> ImageListModel:
        model.beginResetModel()
        model.images = [make_image(f'image_{index}.png', tags)
                        for index, tags in enumerate(tag_lists)]
        model.endResetModel()
        return model
    return factory


def tags_of(model) -> list[list[str]]:
    return [image.tags for image in model.images]


class TestIllustriousReorder:
    def test_count_tags_are_promoted_to_the_front(self, populated):
        model = populated([['solo', 'smile', '1girl']])
        model.reorder_illustrious_tags(do_not_reorder_first_tag=False)
        assert model.images[0].tags[0] == '1girl'

    def test_categories_order_as_count_character_series_general(self,
                                                               populated):
        model = populated([['smile', 'vocaloid', 'hatsune miku', '1girl']])
        model.reorder_illustrious_tags(
            character_tags={'hatsune miku'}, series_tags={'vocaloid'},
            do_not_reorder_first_tag=False)
        assert model.images[0].tags == ['1girl', 'hatsune miku', 'vocaloid',
                                        'smile']

    def test_general_tags_keep_their_original_order(self, populated):
        """
        The WD tagger emits general tags in descending confidence. Sorting
        them alphabetically would throw that ordering away.
        """
        model = populated([['1girl', 'zebra', 'apple', 'mango']])
        model.reorder_illustrious_tags(do_not_reorder_first_tag=False)
        assert model.images[0].tags == ['1girl', 'zebra', 'apple', 'mango']

    def test_category_membership_is_case_insensitive(self, populated):
        model = populated([['smile', 'Hatsune Miku']])
        model.reorder_illustrious_tags(character_tags={'hatsune miku'},
                                       do_not_reorder_first_tag=False)
        assert model.images[0].tags == ['Hatsune Miku', 'smile']

    def test_the_first_tag_can_be_pinned(self, populated):
        model = populated([['my trigger', 'smile', '1girl']])
        model.reorder_illustrious_tags(do_not_reorder_first_tag=True)
        assert model.images[0].tags[0] == 'my trigger'
        assert model.images[0].tags[1] == '1girl'

    def test_plural_count_tags_are_recognised(self, populated):
        model = populated([['smile', '2boys'], ['smile', '3girls']])
        model.reorder_illustrious_tags(do_not_reorder_first_tag=False)
        assert model.images[0].tags[0] == '2boys'
        assert model.images[1].tags[0] == '3girls'

    def test_images_with_fewer_than_two_tags_are_skipped(self, populated):
        model = populated([[], ['solo']])
        model.reorder_illustrious_tags(do_not_reorder_first_tag=False)
        assert tags_of(model) == [[], ['solo']]

    def test_an_already_ordered_dataset_records_no_undo_step(self, populated):
        model = populated([['1girl', 'smile']])
        model.reorder_illustrious_tags(do_not_reorder_first_tag=False)
        assert not model.undo_stack


class TestInsertTriggerToken:
    def test_first_tag_mode_prepends_the_trigger(self, populated):
        model = populated([['solo'], ['smile']])
        model.insert_trigger_token('sks', mode='first_tag')
        assert tags_of(model) == [['sks', 'solo'], ['sks', 'smile']]

    def test_embedded_mode_joins_the_trigger_to_the_first_tag(self, populated):
        model = populated([['a photo of a dog']])
        model.insert_trigger_token('sks', mode='embedded')
        assert model.images[0].tags == ['sks a photo of a dog']

    def test_an_image_that_already_has_the_trigger_is_left_alone(self,
                                                                populated):
        model = populated([['sks', 'solo']])
        model.insert_trigger_token('sks', mode='first_tag')
        assert model.images[0].tags == ['sks', 'solo']

    def test_a_tag_merely_containing_the_trigger_does_not_count(self,
                                                               populated):
        """`sks` appears inside `masks`, but the image has no trigger."""
        model = populated([['masks', 'solo']])
        model.insert_trigger_token('sks', mode='first_tag')
        assert model.images[0].tags == ['sks', 'masks', 'solo']

    def test_an_embedded_trigger_is_detected_as_a_whole_word(self, populated):
        model = populated([['sks a photo of a dog']])
        model.insert_trigger_token('sks', mode='embedded')
        assert model.images[0].tags == ['sks a photo of a dog']

    def test_an_empty_trigger_does_nothing(self, populated):
        model = populated([['solo']])
        model.insert_trigger_token('   ', mode='first_tag')
        assert model.images[0].tags == ['solo']
        assert not model.undo_stack

    def test_an_untagged_image_gets_the_trigger_alone(self, populated):
        model = populated([[]])
        model.insert_trigger_token('sks', mode='embedded')
        assert model.images[0].tags == ['sks']

    def test_the_trigger_is_written_to_the_sidecar(self, populated):
        model = populated([['solo']])
        model.insert_trigger_token('sks', mode='first_tag')
        model.tag_writer.flush()
        sidecar = model.images[0].path.with_suffix('.txt')
        assert sidecar.read_text(encoding='utf-8') == 'sks, solo'


class TestUndoRedo:
    def test_undo_restores_the_previous_tags(self, populated):
        model = populated([['solo'], ['smile']])
        model.insert_trigger_token('sks', mode='first_tag')
        model.undo()
        assert tags_of(model) == [['solo'], ['smile']]

    def test_redo_reapplies_them(self, populated):
        model = populated([['solo']])
        model.insert_trigger_token('sks', mode='first_tag')
        model.undo()
        model.redo()
        assert model.images[0].tags == ['sks', 'solo']

    def test_history_stores_only_the_images_that_changed(self, populated):
        model = populated([['sks', 'solo'], ['smile']])
        model.insert_trigger_token('sks', mode='first_tag')
        assert list(model.undo_stack[-1].previous_tags) == [1]

    def test_a_new_edit_clears_the_redo_stack(self, populated):
        model = populated([['solo']])
        model.insert_trigger_token('sks', mode='first_tag')
        model.undo()
        assert model.redo_stack
        model.insert_trigger_token('xyz', mode='first_tag')
        assert not model.redo_stack

    def test_undo_with_no_history_is_a_no_op(self, populated):
        model = populated([['solo']])
        model.undo()
        assert model.images[0].tags == ['solo']


class TestFindAndReplace:
    def test_plain_text_replacement(self, populated):
        model = populated([['red car', 'blue sky']])
        model.find_and_replace('red', 'green', Scope.ALL_IMAGES,
                               use_regex=False)
        assert model.images[0].tags == ['green car', 'blue sky']

    def test_regex_replacement(self, populated):
        model = populated([['cat1', 'cat2']])
        model.find_and_replace(r'cat(\d)', r'dog\1', Scope.ALL_IMAGES,
                               use_regex=True)
        assert model.images[0].tags == ['dog1', 'dog2']

    def test_images_without_a_match_are_untouched(self, populated):
        model = populated([['red'], ['blue']])
        model.find_and_replace('red', 'green', Scope.ALL_IMAGES,
                               use_regex=False)
        assert list(model.undo_stack[-1].previous_tags) == [0]

    def test_an_empty_search_does_nothing(self, populated):
        model = populated([['red']])
        model.find_and_replace('', 'green', Scope.ALL_IMAGES, use_regex=False)
        assert model.images[0].tags == ['red']

    def test_match_counting_covers_whole_tags_and_substrings(self, populated):
        model = populated([['cat', 'cat'], ['catalog']])
        assert model.get_text_match_count(
            'cat', Scope.ALL_IMAGES, whole_tags_only=True,
            use_regex=False) == 2
        assert model.get_text_match_count(
            'cat', Scope.ALL_IMAGES, whole_tags_only=False,
            use_regex=False) == 3


class TestTagCleanup:
    def test_duplicates_are_removed_keeping_first_occurrence(self, populated):
        model = populated([['a', 'b', 'a', 'c', 'b']])
        assert model.remove_duplicate_tags() == 2
        assert model.images[0].tags == ['a', 'b', 'c']

    def test_empty_tags_are_removed(self, populated):
        model = populated([['a', '', '  ', 'b']])
        assert model.remove_empty_tags() == 2
        assert model.images[0].tags == ['a', 'b']

    def test_a_clean_dataset_reports_nothing_removed(self, populated):
        model = populated([['a', 'b']])
        assert model.remove_duplicate_tags() == 0
        assert model.remove_empty_tags() == 0


class TestTagOrdering:
    def test_alphabetical_sort_can_pin_the_first_tag(self, populated):
        model = populated([['trigger', 'c', 'a', 'b']])
        model.sort_tags_alphabetically(do_not_reorder_first_tag=True)
        assert model.images[0].tags == ['trigger', 'a', 'b', 'c']

    def test_reversing_restores_the_original_order(self, populated):
        model = populated([['a', 'b', 'c']])
        model.reverse_tags_order(do_not_reorder_first_tag=False)
        assert model.images[0].tags == ['c', 'b', 'a']

    def test_tags_can_be_moved_to_the_front(self, populated):
        model = populated([['a', 'b', 'c']])
        model.move_tags_to_front(['c'])
        assert model.images[0].tags == ['c', 'a', 'b']

    def test_tags_can_be_moved_to_the_back(self, populated):
        model = populated([['a', 'b', 'c']])
        model.move_tags_to_back(['a'])
        assert model.images[0].tags == ['b', 'c', 'a']


class TestExport:
    def test_jsonl_export_writes_one_object_per_image(self, populated,
                                                      tmp_path):
        model = populated([['a', 'b'], ['c']])
        destination = tmp_path / 'out.jsonl'
        assert model.export_jsonl(destination) == 2
        records = [json.loads(line)
                   for line in destination.read_text().splitlines()]
        assert [record['text'] for record in records] == ['a, b', 'c']
        assert records[0]['file_name'] == 'image_0.png'

    def test_kohya_metadata_export_is_keyed_by_path(self, populated,
                                                    tmp_path):
        model = populated([['a', 'b']])
        destination = tmp_path / 'meta.json'
        assert model.export_kohya_metadata(destination) == 1
        metadata = json.loads(destination.read_text())
        entry = metadata[str(model.images[0].path)]
        assert entry == {'caption': 'a, b', 'tags': ['a', 'b']}

    def test_exporting_an_empty_dataset_produces_an_empty_file(self,
                                                               populated,
                                                               tmp_path):
        model = populated([])
        destination = tmp_path / 'out.jsonl'
        assert model.export_jsonl(destination) == 0
        assert destination.read_text() == ''
