import shutil
from pathlib import Path

from PIL import Image as PilImage
from PIL.ImageOps import exif_transpose
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFormLayout,
                               QHeaderView, QLabel, QMessageBox, QProgressDialog,
                               QPushButton, QSpinBox, QTableWidget,
                               QTableWidgetItem, QVBoxLayout)

from models.image_list_model import BACKUP_DIRECTORY_NAME, ImageListModel
from utils.bucketing import (BucketConfig, assign_bucket,
                             make_bucket_resolutions, plan_resize_crop)

# Images cropped more than this fraction of their area are flagged.
HEAVY_CROP_FRACTION = 0.2
# Buckets with this many or fewer images are flagged as sparse (a bucket with
# fewer images than the batch size trains inefficiently).
SPARSE_BUCKET_COUNT = 1


class BucketProcessingWorker(QThread):
    """
    Move each original image into an `original_images` backup folder (preserving
    the relative directory structure) and write a resized + center-cropped PNG
    at each image's original location, snapped to its assigned bucket. Runs off
    the UI thread because decoding and resizing many images is slow.
    """
    progress = Signal(int, int)  # (completed, total)
    finished_processing = Signal(dict)  # summary

    def __init__(self, parent, directory_path: Path, images: list,
                 config: BucketConfig):
        super().__init__(parent)
        self.directory_path = directory_path
        self.images = images
        self.config = config
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def _unique_path(self, path: Path) -> Path:
        """Return `path`, or a numbered variant if it already exists on disk."""
        if not path.exists():
            return path
        counter = 1
        while True:
            candidate = path.with_name(f'{path.stem}_{counter}{path.suffix}')
            if not candidate.exists():
                return candidate
            counter += 1

    def run(self):
        summary = {'processed': 0, 'skipped': 0, 'failed': 0, 'errors': [],
                   'cancelled': False, 'backup_dir': None}
        base_directory = self.directory_path.resolve()
        backup_root = base_directory / BACKUP_DIRECTORY_NAME
        summary['backup_dir'] = str(backup_root)
        bucket_resolutions = make_bucket_resolutions(self.config)

        # Build the work list, skipping anything already inside the backup
        # folder or outside the loaded directory. Disambiguate output PNG names
        # so that e.g. foo.jpg and foo.png do not both map to foo.png.
        work = []
        used_outputs = set()
        for image in self.images:
            source_path = image.path
            try:
                relative_path = source_path.resolve().relative_to(
                    base_directory)
            except ValueError:
                continue
            if BACKUP_DIRECTORY_NAME in relative_path.parts:
                summary['skipped'] += 1
                continue
            output_path = (base_directory / relative_path).with_suffix('.png')
            counter = 1
            while output_path in used_outputs:
                output_path = (base_directory / relative_path).with_name(
                    f'{relative_path.stem}_{counter}.png')
                counter += 1
            used_outputs.add(output_path)
            backup_path = backup_root / relative_path
            caption_path = source_path.with_suffix('.txt')
            work.append((source_path, backup_path, output_path, caption_path))

        total = len(work)

        # Phase 1: move every original into the backup folder first, so that
        # generating a PNG can never overwrite an original that has not been
        # backed up yet.
        moved = []
        for source_path, backup_path, output_path, caption_path in work:
            if self._is_cancelled:
                summary['cancelled'] = True
                break
            try:
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                backup_path = self._unique_path(backup_path)
                shutil.move(str(source_path), str(backup_path))
                moved.append((backup_path, output_path, caption_path,
                              source_path))
            except Exception as exception:
                summary['failed'] += 1
                summary['errors'].append(f'{source_path.name}: {exception}')

        # Phase 2: resize/crop each backed-up original into its bucket PNG.
        completed = 0
        for backup_path, output_path, caption_path, source_path in moved:
            if self._is_cancelled:
                summary['cancelled'] = True
                break
            try:
                with PilImage.open(backup_path) as pil_image:
                    pil_image = exif_transpose(pil_image)
                    bucket = assign_bucket(pil_image.size, self.config,
                                           bucket_resolutions).bucket
                    scaled_size, crop_box = plan_resize_crop(pil_image.size,
                                                             bucket)
                    pil_image = pil_image.resize(
                        scaled_size, PilImage.Resampling.LANCZOS)
                    pil_image = pil_image.crop(crop_box)
                    # Flatten any transparency onto white, then save as RGB PNG.
                    if pil_image.mode in ('RGBA', 'LA', 'P'):
                        pil_image = pil_image.convert('RGBA')
                        canvas = PilImage.new('RGB', pil_image.size,
                                              (255, 255, 255))
                        canvas.paste(pil_image, mask=pil_image.split()[-1])
                        pil_image = canvas
                    else:
                        pil_image = pil_image.convert('RGB')
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    pil_image.save(output_path, format='PNG')
                # If the output was renamed to avoid a collision, copy the
                # caption so the new PNG keeps a matching .txt file.
                if output_path.stem != caption_path.stem and caption_path.is_file():
                    shutil.copyfile(caption_path,
                                    output_path.with_suffix('.txt'))
                summary['processed'] += 1
            except Exception as exception:
                summary['failed'] += 1
                summary['errors'].append(f'{backup_path.name}: {exception}')
            completed += 1
            self.progress.emit(completed, total)

        self.finished_processing.emit(summary)


class BucketCalculatorDialog(QDialog):
    """
    Show how the loaded dataset would be split into aspect-ratio buckets by
    kohya_ss / OneTrainer, and optionally process the images into those buckets:
    back up the originals and write resized + center-cropped PNGs in place.
    """

    def __init__(self, parent, image_list_model: ImageListModel,
                 directory_path: Path | None):
        super().__init__(parent)
        self.image_list_model = image_list_model
        self.directory_path = directory_path
        self.worker = None
        self.progress_dialog = None
        self.setWindowTitle('Aspect Ratio Bucket Calculator')
        self.setMinimumSize(560, 600)
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

        # Process button.
        self.process_button = QPushButton('Process Images into Buckets...')
        self.process_button.clicked.connect(self.process_images)
        layout.addWidget(self.process_button)

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

    def get_eligible_images(self) -> list:
        return [image for image in self.image_list_model.images
                if image.dimensions
                and BACKUP_DIRECTORY_NAME not in image.path.parts]

    def recalculate(self):
        config = self.get_config()
        eligible_images = self.get_eligible_images()
        dimensions_list = [image.dimensions for image in eligible_images]
        total_images = len(dimensions_list)
        self.process_button.setEnabled(bool(total_images)
                                       and self.directory_path is not None)
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

    def process_images(self):
        if self.directory_path is None:
            return
        eligible_images = self.get_eligible_images()
        if not eligible_images:
            return
        backup_root = self.directory_path / BACKUP_DIRECTORY_NAME
        reply = QMessageBox.warning(
            self, 'Process Images into Buckets',
            f'This will move {len(eligible_images)} original '
            f'{"image" if len(eligible_images) == 1 else "images"} into\n'
            f'{backup_root}\n'
            f'and write resized, center-cropped PNGs in their place.\n\n'
            f'The originals are preserved in the backup folder. Continue?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.process_button.setEnabled(False)
        self.progress_dialog = QProgressDialog(
            'Processing images into buckets...', 'Cancel', 0,
            len(eligible_images), self)
        self.progress_dialog.setWindowTitle('Processing')
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setValue(0)

        self.worker = BucketProcessingWorker(self, self.directory_path,
                                             eligible_images, self.get_config())
        self.worker.progress.connect(self.on_progress)
        self.worker.finished_processing.connect(self.on_finished)
        self.progress_dialog.canceled.connect(self.worker.cancel)
        self.worker.start()

    def on_progress(self, completed: int, total: int):
        if self.progress_dialog is not None:
            self.progress_dialog.setValue(completed)

    def on_finished(self, summary: dict):
        if self.progress_dialog is not None:
            self.progress_dialog.close()
            self.progress_dialog = None
        # Reload the directory so the list shows the new PNGs (the backup folder
        # is skipped during loading).
        parent = self.parent()
        if hasattr(parent, 'reload_directory'):
            parent.reload_directory()
        self.recalculate()
        self.process_button.setEnabled(True)

        lines = [f'Processed: {summary["processed"]}']
        if summary['skipped']:
            lines.append(f'Skipped (already backed up): {summary["skipped"]}')
        if summary['failed']:
            lines.append(f'Failed: {summary["failed"]}')
        if summary['cancelled']:
            lines.append('Cancelled before finishing.')
        lines.append(f'\nOriginals backed up in:\n{summary["backup_dir"]}')
        message_box = QMessageBox(self)
        message_box.setWindowTitle('Bucket Processing Complete')
        message_box.setIcon(QMessageBox.Icon.Information)
        message_box.setText('\n'.join(lines))
        if summary['errors']:
            message_box.setDetailedText('\n'.join(summary['errors']))
        message_box.exec()
