import operator
from fnmatch import fnmatchcase

from PySide6.QtCore import QModelIndex, QSortFilterProxyModel, Qt
from transformers import PreTrainedTokenizerBase

from models.image_list_model import ImageListModel
from utils.image import Image


class ProxyImageListModel(QSortFilterProxyModel):
    def __init__(self, image_list_model: ImageListModel,
                 tokenizer: PreTrainedTokenizerBase | None, tag_separator: str):
        super().__init__()
        self.setSourceModel(image_list_model)
        self.tokenizer = tokenizer
        self.tag_separator = tag_separator
        self.filter: list | None = None

    def set_tokenizer(self, tokenizer: PreTrainedTokenizerBase | None):
        self.tokenizer = tokenizer
        # Invalidate cached token counts when the encoder changes.
        source = self.sourceModel()
        if source is not None:
            for image in source.images:
                image.token_count = None

    def _get_token_count(self, image: Image) -> int:
        if image.token_count is not None:
            return image.token_count
        if self.tokenizer is None:
            # Approximate with whitespace-ish split until tokenizer is ready.
            caption = self.tag_separator.join(image.tags)
            image.token_count = max(len(caption.split()), 0)
            return image.token_count
        caption = self.tag_separator.join(image.tags)
        # Subtract 2 for BOS/EOS-style special tokens when present.
        token_ids = self.tokenizer(caption).input_ids
        count = len(token_ids)
        if count >= 2:
            count -= 2
        image.token_count = count
        return count

    def does_image_match_filter(self, image: Image,
                                filter_: list | str) -> bool:
        if isinstance(filter_, str):
            caption = self.tag_separator.join(image.tags)
            return (fnmatchcase(caption, f'*{filter_}*')
                    or fnmatchcase(str(image.path), f'*{filter_}*'))
        if len(filter_) == 1:
            return self.does_image_match_filter(image, filter_[0])
        if len(filter_) == 2:
            if filter_[0] == 'NOT':
                return not self.does_image_match_filter(image, filter_[1])
            if filter_[0] == 'tag':
                return any(fnmatchcase(tag, filter_[1]) for tag in image.tags)
            if filter_[0] == 'caption':
                caption = self.tag_separator.join(image.tags)
                return fnmatchcase(caption, f'*{filter_[1]}*')
            if filter_[0] == 'name':
                return fnmatchcase(image.path.name, f'*{filter_[1]}*')
            if filter_[0] == 'path':
                return fnmatchcase(str(image.path), f'*{filter_[1]}*')
        if filter_[1] == 'AND':
            return (self.does_image_match_filter(image, filter_[0])
                    and self.does_image_match_filter(image, filter_[2:]))
        if filter_[1] == 'OR':
            return (self.does_image_match_filter(image, filter_[0])
                    or self.does_image_match_filter(image, filter_[2:]))
        comparison_operators = {
            '=': operator.eq,
            '==': operator.eq,
            '!=': operator.ne,
            '<': operator.lt,
            '>': operator.gt,
            '<=': operator.le,
            '>=': operator.ge
        }
        comparison_operator = comparison_operators[filter_[1]]
        number_to_compare = None
        if filter_[0] == 'tags':
            number_to_compare = len(image.tags)
        elif filter_[0] == 'chars':
            caption = self.tag_separator.join(image.tags)
            number_to_compare = len(caption)
        elif filter_[0] == 'tokens':
            number_to_compare = self._get_token_count(image)
        return comparison_operator(number_to_compare, int(filter_[2]))

    def filterAcceptsRow(self, source_row: int,
                         source_parent: QModelIndex) -> bool:
        if self.filter is None:
            return True
        image_index = self.sourceModel().index(source_row, 0)
        image: Image = self.sourceModel().data(image_index,
                                               Qt.ItemDataRole.UserRole)
        return self.does_image_match_filter(image, self.filter)

    def is_image_in_filtered_images(self, image: Image) -> bool:
        return (self.filter is None
                or self.does_image_match_filter(image, self.filter))
