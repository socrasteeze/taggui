"""
The roster's group heading must not be selectable.

The auto-captioner panel itself pulls in the captioning stack, so this
exercises the same combo-box behaviour against a plain `QComboBox` populated
with the real roster.
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox

from auto_captioning.models_list import (LEGACY_GROUP_SEPARATOR, MODELS,
                                         is_group_separator)


def disable_group_separators(combo_box: QComboBox):
    """Mirrors `CaptionSettingsForm._disable_group_separators`."""
    model = combo_box.model()
    if not hasattr(model, 'item'):
        return
    for row in range(combo_box.count()):
        if not is_group_separator(combo_box.itemText(row)):
            continue
        item = model.item(row)
        if item is not None:
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable
                          & ~Qt.ItemFlag.ItemIsEnabled)


def make_combo_box() -> QComboBox:
    combo_box = QComboBox()
    combo_box.setEditable(True)
    combo_box.addItems(MODELS)
    disable_group_separators(combo_box)
    return combo_box


def test_the_heading_is_present_in_the_list():
    combo_box = make_combo_box()
    texts = [combo_box.itemText(row) for row in range(combo_box.count())]
    assert LEGACY_GROUP_SEPARATOR in texts


def test_the_heading_is_neither_selectable_nor_enabled():
    combo_box = make_combo_box()
    row = combo_box.findText(LEGACY_GROUP_SEPARATOR)
    flags = combo_box.model().item(row).flags()
    assert not flags & Qt.ItemFlag.ItemIsSelectable
    assert not flags & Qt.ItemFlag.ItemIsEnabled


def test_real_models_stay_selectable():
    combo_box = make_combo_box()
    for model_id in ('Qwen/Qwen3-VL-8B-Instruct', 'llava-hf/llava-1.5-7b-hf'):
        row = combo_box.findText(model_id)
        assert row >= 0, model_id
        flags = combo_box.model().item(row).flags()
        assert flags & Qt.ItemFlag.ItemIsSelectable, model_id
        assert flags & Qt.ItemFlag.ItemIsEnabled, model_id


def test_a_combo_box_without_a_standard_item_model_is_left_alone():
    """The guard keeps a custom model from raising during construction."""
    from PySide6.QtCore import QStringListModel

    combo_box = QComboBox()
    combo_box.setModel(QStringListModel(MODELS))
    disable_group_separators(combo_box)      # must not raise
    assert combo_box.count() == len(MODELS)
