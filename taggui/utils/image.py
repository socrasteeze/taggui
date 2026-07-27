from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtGui import QIcon


@dataclass
class Image:
    path: Path
    dimensions: tuple[int, int] | None
    tags: list[str] = field(default_factory=list)
    thumbnail: QIcon | None = None
    # Cached token count for the active encoder; invalidated on tag change.
    token_count: int | None = None
