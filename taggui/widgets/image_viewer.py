from pathlib import Path

from PySide6.QtCore import QModelIndex, QSize, Qt, Slot
from PySide6.QtGui import QImage, QImageReader, QPixmap, QResizeEvent
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from models.proxy_image_list_model import ProxyImageListModel
from utils.image import Image


class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.image_path = None
        self._cached_image: QImage | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self.setMinimumSize(QSize(1, 1))

    def resizeEvent(self, event: QResizeEvent):
        """Rescale the cached image whenever the label is resized."""
        if self._cached_image is not None and not self._cached_image.isNull():
            self._apply_scaled_pixmap()
        elif self.image_path:
            self.load_image(self.image_path)

    def load_image(self, image_path: Path):
        if self.image_path == image_path and self._cached_image is not None:
            self._apply_scaled_pixmap()
            return
        self.image_path = image_path
        image_reader = QImageReader(str(image_path))
        image_reader.setAutoTransform(True)
        self._cached_image = image_reader.read()
        self._apply_scaled_pixmap()

    def _apply_scaled_pixmap(self):
        if self._cached_image is None or self._cached_image.isNull():
            return
        pixmap = QPixmap.fromImage(self._cached_image)
        pixmap.setDevicePixelRatio(self.devicePixelRatio())
        pixmap = pixmap.scaled(
            self.size() * pixmap.devicePixelRatio(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        self.setPixmap(pixmap)


class ImageViewer(QWidget):
    def __init__(self, proxy_image_list_model: ProxyImageListModel):
        super().__init__()
        self.proxy_image_list_model = proxy_image_list_model
        self.image_label = ImageLabel()
        QVBoxLayout(self).addWidget(self.image_label)

    @Slot()
    def load_image(self, proxy_image_index: QModelIndex):
        if not proxy_image_index.isValid():
            return
        image: Image = self.proxy_image_list_model.data(
            proxy_image_index, Qt.ItemDataRole.UserRole)
        if image is None:
            return
        self.image_label.load_image(image.path)
