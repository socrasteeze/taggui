from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Image:
    path: Path
    dimensions: tuple[int, int] | None
    tags: list[str] = field(default_factory=list)
    # Cached token count for the active encoder.
    token_count: int | None = None
    # Cached `tag_separator`-joined caption. Filter evaluation would otherwise
    # rebuild it once per term, for every image, on every keystroke.
    caption: str | None = None

    def invalidate_caches(self):
        """Call after changing `tags`; both cached values derive from them."""
        self.token_count = None
        self.caption = None
