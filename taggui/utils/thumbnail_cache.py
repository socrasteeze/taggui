"""
A bounded cache of decoded thumbnails.

Holding a decoded thumbnail per image keeps memory proportional to the dataset
rather than to what is on screen, which does not scale to the tens of
thousands of images this fork is meant to handle. Keeping a bounded number of
the most recently shown thumbnails costs a re-decode when the user scrolls
back - the decode already happens on a background thread - in exchange for a
flat memory ceiling.
"""
from collections import OrderedDict

# Roughly a few hundred megabytes of decoded pixmaps at the default thumbnail
# width, and far more than fits on screen at once.
DEFAULT_MAXIMUM_SIZE = 1000


class ThumbnailCache:
    """Least-recently-used cache keyed by image path."""

    def __init__(self, maximum_size: int = DEFAULT_MAXIMUM_SIZE):
        self.maximum_size = max(1, maximum_size)
        self._entries: OrderedDict = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key) -> bool:
        return key in self._entries

    def get(self, key):
        entry = self._entries.get(key)
        if entry is None:
            return None
        self._entries.move_to_end(key)
        return entry

    def set(self, key, value):
        self._entries[key] = value
        self._entries.move_to_end(key)
        while len(self._entries) > self.maximum_size:
            self._entries.popitem(last=False)

    def clear(self):
        self._entries.clear()
