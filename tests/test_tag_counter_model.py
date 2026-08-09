"""
Tests for `models.tag_counter_model`.

The counter is updated incrementally on every edit rather than recounting the
whole dataset, so the property that matters is that it always agrees with a
full recount.
"""
from collections import Counter

import pytest

from models.tag_counter_model import TagCounterModel
from utils.image import Image


@pytest.fixture
def counter():
    return TagCounterModel()


def make_images(tag_lists: list[list[str]]) -> list[Image]:
    return [Image(f'/images/image_{index}.png', (1024, 1024), list(tags))
            for index, tags in enumerate(tag_lists)]


def full_recount(images: list[Image]) -> Counter:
    tally = Counter()
    for image in images:
        tally.update(image.tags)
    return tally


def test_counting_tallies_every_tag(counter):
    images = make_images([['a', 'b'], ['b', 'c'], ['b']])
    counter.count_tags(images)
    assert counter.tag_counter == Counter({'b': 3, 'a': 1, 'c': 1})


def test_most_common_is_ordered_by_count(counter):
    counter.count_tags(make_images([['a', 'b'], ['b'], ['b', 'c']]))
    assert counter.most_common_tags[0] == ('b', 3)


def test_the_tags_alias_exposes_the_ranked_list(counter):
    counter.count_tags(make_images([['a']]))
    assert counter.tags == counter.most_common_tags


def test_the_row_count_matches_the_number_of_distinct_tags(counter):
    counter.count_tags(make_images([['a', 'b'], ['b']]))
    assert counter.rowCount() == 2


class TestIncrementalUpdates:
    def test_an_edit_matches_a_full_recount(self, counter):
        images = make_images([['a', 'b'], ['b', 'c']])
        counter.count_tags(images)
        images[0].tags = ['a', 'z']
        counter.update_tag_counts(images, 0, 0)
        assert counter.tag_counter == full_recount(images)

    def test_clearing_an_images_tags_matches_a_full_recount(self, counter):
        images = make_images([['a', 'b'], ['b']])
        counter.count_tags(images)
        images[0].tags = []
        counter.update_tag_counts(images, 0, 0)
        assert counter.tag_counter == full_recount(images)

    def test_a_batch_edit_matches_a_full_recount(self, counter):
        images = make_images([['a'], ['b'], ['c']])
        counter.count_tags(images)
        for image in images:
            image.tags = ['shared', 'x']
        counter.update_tag_counts(images, 0, 2)
        assert counter.tag_counter == full_recount(images)

    def test_tags_that_fall_to_zero_disappear(self, counter):
        images = make_images([['gone']])
        counter.count_tags(images)
        images[0].tags = ['kept']
        counter.update_tag_counts(images, 0, 0)
        counter.publish_pending_counts()
        assert 'gone' not in counter.tag_counter
        assert [tag for tag, _ in counter.most_common_tags] == ['kept']

    def test_duplicate_tags_within_one_image_are_all_counted(self, counter):
        images = make_images([['a', 'a', 'b']])
        counter.count_tags(images)
        assert counter.tag_counter['a'] == 2
        images[0].tags = ['a', 'b']
        counter.update_tag_counts(images, 0, 0)
        assert counter.tag_counter == full_recount(images)

    def test_a_no_op_update_leaves_the_counts_alone(self, counter):
        images = make_images([['a', 'b']])
        counter.count_tags(images)
        before = counter.most_common_tags
        counter.update_tag_counts(images, 0, 0)
        counter.publish_pending_counts()
        assert counter.most_common_tags == before

    def test_a_row_range_past_the_end_is_clamped(self, counter):
        images = make_images([['a']])
        counter.count_tags(images)
        images[0].tags = ['b']
        counter.update_tag_counts(images, 0, 99)
        assert counter.tag_counter == full_recount(images)

    def test_rows_outside_the_range_are_not_rescanned(self, counter):
        """Only the rows the view reported as changed should be diffed."""
        images = make_images([['a'], ['b']])
        counter.count_tags(images)
        images[1].tags = ['changed']
        counter.update_tag_counts(images, 0, 0)
        assert counter.tag_counter == Counter({'a': 1, 'b': 1})

    def test_a_long_sequence_of_edits_stays_consistent(self, counter):
        images = make_images([['a', 'b'], ['b', 'c'], ['c', 'd']])
        counter.count_tags(images)
        edits = [(0, ['x']), (1, ['x', 'y']), (2, []), (0, ['a', 'b', 'x']),
                 (1, ['b']), (2, ['z', 'z'])]
        for row, tags in edits:
            images[row].tags = tags
            counter.update_tag_counts(images, row, row)
            assert counter.tag_counter == full_recount(images)


class TestCoalescedPublishing:
    """
    Ranking every tag and resetting the view is the expensive part of an edit,
    and a batch captioning run reports one row at a time. The counts stay
    correct after each edit; only the published ranking is deferred.
    """

    def test_a_run_of_edits_produces_one_reset(self, counter):
        images = make_images([['a'], ['b'], ['c']])
        counter.count_tags(images)
        resets = []
        counter.modelReset.connect(lambda: resets.append(True))

        for row in range(3):
            images[row].tags = [f'new-{row}']
            counter.update_tag_counts(images, row, row)
        assert resets == [], 'reset before the batch finished'

        counter.publish_pending_counts()
        assert len(resets) == 1

    def test_the_counts_are_correct_before_publishing(self, counter):
        images = make_images([['a']])
        counter.count_tags(images)
        images[0].tags = ['b']
        counter.update_tag_counts(images, 0, 0)
        assert counter.tag_counter == full_recount(images)

    def test_publishing_makes_the_ranking_current(self, counter):
        images = make_images([['a']])
        counter.count_tags(images)
        images[0].tags = ['a', 'b']
        counter.update_tag_counts(images, 0, 0)
        counter.publish_pending_counts()
        assert dict(counter.most_common_tags) == {'a': 1, 'b': 1}

    def test_publishing_with_nothing_pending_does_nothing(self, counter):
        counter.count_tags(make_images([['a']]))
        resets = []
        counter.modelReset.connect(lambda: resets.append(True))
        counter.publish_pending_counts()
        assert resets == []

    def test_a_full_recount_supersedes_a_pending_update(self, counter):
        images = make_images([['a']])
        counter.count_tags(images)
        images[0].tags = ['b']
        counter.update_tag_counts(images, 0, 0)

        counter.count_tags(make_images([['c']]))
        resets = []
        counter.modelReset.connect(lambda: resets.append(True))
        counter.publish_pending_counts()
        assert resets == [], 'a stale pending update was published'
        assert dict(counter.most_common_tags) == {'c': 1}

    def test_the_pending_update_is_published_by_the_event_loop(self, counter):
        """The app never calls the flush directly; the timer does."""
        from PySide6.QtCore import QEventLoop, QTimer

        images = make_images([['a']])
        counter.count_tags(images)
        images[0].tags = ['a', 'b']
        counter.update_tag_counts(images, 0, 0)

        loop = QEventLoop()
        QTimer.singleShot(50, loop.quit)
        loop.exec()
        assert dict(counter.most_common_tags) == {'a': 1, 'b': 1}


class TestModelReset:
    def test_the_view_is_told_before_the_data_changes(self, counter):
        """
        Qt requires `beginResetModel()` before the model's contents change and
        `endResetModel()` after; a view that reads the model in between must
        not see new data under an old row count.
        """
        events = []
        counter.modelAboutToBeReset.connect(
            lambda: events.append(('about_to_reset',
                                   len(counter.most_common_tags))))
        counter.modelReset.connect(
            lambda: events.append(('reset', len(counter.most_common_tags))))

        images = make_images([['a', 'b']])
        counter.count_tags(images)

        assert [name for name, _ in events] == ['about_to_reset', 'reset']
        assert events[0][1] == 0, 'row count changed before the reset began'
        assert events[1][1] == 2

    def test_an_incremental_update_also_brackets_its_changes(self, counter):
        images = make_images([['a']])
        counter.count_tags(images)
        events = []
        counter.modelAboutToBeReset.connect(
            lambda: events.append(('about_to_reset',
                                   list(counter.most_common_tags))))
        counter.modelReset.connect(lambda: events.append(('reset', None)))

        images[0].tags = ['a', 'b']
        counter.update_tag_counts(images, 0, 0)
        counter.publish_pending_counts()

        assert [name for name, _ in events] == ['about_to_reset', 'reset']
        assert events[0][1] == [('a', 1)], 'data changed before the reset began'
