from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFormLayout,
                               QHeaderView, QLabel, QSpinBox, QTableWidget,
                               QTableWidgetItem, QVBoxLayout)

from models.image_list_model import ImageListModel
from utils.bucketing import (BucketConfig, assign_bucket,
                             make_bucket_resolutions)

# Images cropped more than this fraction of their area are flagged.
HEAVY_CROP_FRACTION = 0.2
# Buckets with this many or fewer images are flagged as sparse (a bucket with
# fewer images than the batch size trains inefficiently).
SPARSE_BUCKET_COUNT = 1


class BucketCalculatorDialog(QDialog):
    """
    Show how the loaded dataset would be split into aspect-ratio buckets by
    kohya_ss / OneTrainer at train time, without modifying any images. Helps
    spot images that would be upscaled or heavily cropped, and buckets that are
    too sparse to train efficiently.
    """

    def __init__(self, parent, image_list_model: ImageListModel):
        super().__init__(parent)
        self.image_list_model = image_list_model
        self.setWindowTitle('Aspect Ratio Bucket Calculator')
        self.setMinimumSize(560, 560)
        layout = QVBoxLayout(self)

        # Controls.
        form_layout = QFormLayout()
        self.target_area_combo_box = QComboBox()
        for resolution in (512, 768, 1024, 1280, 1536):
            self.target_area_combo_box.addItem(f'{resolution} x {resolution}',
                                               resolution)
        self.target_area_combo_box.setCurrentText('1024 x 1024')
        form_layout.addRow('Target resolution', self.target_area_combo_box)

        self.steps_spin_box = QSpinBox()
        self.steps_spin_box.setRange(8, 256)
        self.steps_spin_box.setSingleStep(8)
        self.steps_spin_box.setValue(64)
        form_layout.addRow('Bucket step (px)', self.steps_spin_box)

        self.min_resolution_spin_box = QSpinBox()
        self.min_resolution_spin_box.setRange(64, 4096)
        self.min_resolution_spin_box.setSingleStep(64)
        self.min_resolution_spin_box.setValue(256)
        form_layout.addRow('Min bucket resolution', self.min_resolution_spin_box)

        self.max_resolution_spin_box = QSpinBox()
        self.max_resolution_spin_box.setRange(64, 8192)
        self.max_resolution_spin_box.setSingleStep(64)
        self.max_resolution_spin_box.setValue(2048)
        form_layout.addRow('Max bucket resolution', self.max_resolution_spin_box)

        self.allow_upscaling_check_box = QCheckBox('Allow upscaling')
        self.allow_upscaling_check_box.setChecked(True)
        form_layout.addRow('', self.allow_upscaling_check_box)
        layout.addLayout(form_layout)

        # Summary line.
        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        # Distribution table.
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(['Bucket (W x H)', 'Images',
                                              'Aspect ratio'])
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        layout.addWidget(self.table)

        # Recalculate whenever any control changes.
        self.target_area_combo_box.currentIndexChanged.connect(self.recalculate)
        self.steps_spin_box.valueChanged.connect(self.recalculate)
        self.min_resolution_spin_box.valueChanged.connect(self.recalculate)
        self.max_resolution_spin_box.valueChanged.connect(self.recalculate)
        self.allow_upscaling_check_box.toggled.connect(self.recalculate)

        self.recalculate()

    def get_config(self) -> BucketConfig:
        return BucketConfig(
            target_area_resolution=self.target_area_combo_box.currentData(),
            steps=self.steps_spin_box.value(),
            min_resolution=self.min_resolution_spin_box.value(),
            max_resolution=self.max_resolution_spin_box.value(),
            allow_upscaling=self.allow_upscaling_check_box.isChecked())

    def recalculate(self):
        config = self.get_config()
        dimensions_list = [image.dimensions
                           for image in self.image_list_model.images
                           if image.dimensions]
        total_images = len(dimensions_list)
        if not total_images:
            self.summary_label.setText('No images with known dimensions are '
                                       'loaded.')
            self.table.setRowCount(0)
            return

        bucket_resolutions = make_bucket_resolutions(config)
        distribution: dict[tuple[int, int], int] = {}
        upscaled_count = 0
        heavy_crop_count = 0
        for dimensions in dimensions_list:
            assignment = assign_bucket(dimensions, config, bucket_resolutions)
            distribution[assignment.bucket] = (
                distribution.get(assignment.bucket, 0) + 1)
            if assignment.is_upscaled:
                upscaled_count += 1
            if assignment.crop_fraction > HEAVY_CROP_FRACTION:
                heavy_crop_count += 1

        sparse_buckets = sum(1 for count in distribution.values()
                             if count <= SPARSE_BUCKET_COUNT)
        self.summary_label.setText(
            f'{total_images} images across {len(distribution)} buckets. '
            f'Upscaled: {upscaled_count}. '
            f'Cropped over {int(HEAVY_CROP_FRACTION * 100)}%: '
            f'{heavy_crop_count}. '
            f'Sparse buckets (<= {SPARSE_BUCKET_COUNT} image): '
            f'{sparse_buckets}.')

        # Fill the table, most-populated bucket first.
        sorted_buckets = sorted(distribution.items(), key=lambda item: item[1],
                                reverse=True)
        self.table.setRowCount(len(sorted_buckets))
        for row, (bucket, count) in enumerate(sorted_buckets):
            width, height = bucket
            bucket_item = QTableWidgetItem(f'{width} x {height}')
            count_item = QTableWidgetItem(str(count))
            count_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            aspect_item = QTableWidgetItem(f'{width / height:.3f}')
            aspect_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if count <= SPARSE_BUCKET_COUNT:
                for item in (bucket_item, count_item, aspect_item):
                    item.setForeground(Qt.GlobalColor.red)
            self.table.setItem(row, 0, bucket_item)
            self.table.setItem(row, 1, count_item)
            self.table.setItem(row, 2, aspect_item)
