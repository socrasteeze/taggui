"""
Aspect-ratio bucketing, matching the algorithm used by kohya_ss / sd-scripts
(and mirrored by OneTrainer). Given a target resolution area and a set of
constraints, every candidate bucket resolution is a multiple of a step size
(default 64 px) whose area is at or below the target area. Each image is then
assigned to the bucket whose aspect ratio is closest to the image's own aspect
ratio, exactly as the trainers do at train time.

This module is pure logic (no Qt / no image decoding) so it can be unit-tested
and reused. It only needs each image's pixel dimensions, which TagGUI already
reads when a directory is loaded.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class BucketConfig:
    target_area_resolution: int = 1024  # The square-equivalent target edge.
    steps: int = 64                     # kohya --bucket_reso_steps.
    min_resolution: int = 256           # kohya --min_bucket_reso.
    max_resolution: int = 2048          # kohya --max_bucket_reso.
    allow_upscaling: bool = True        # Inverse of --bucket_no_upscale.

    @property
    def target_area(self) -> int:
        return self.target_area_resolution * self.target_area_resolution


@dataclass(frozen=True)
class BucketAssignment:
    bucket: tuple[int, int]        # (width, height) the image is resized into.
    scale: float                   # Resize factor applied to the source image.
    crop: tuple[int, int]          # (width, height) pixels cropped after scale.
    is_upscaled: bool              # Whether the source was smaller than needed.

    @property
    def crop_fraction(self) -> float:
        """Fraction of the scaled image's area removed by cropping."""
        scaled_area = ((self.bucket[0] + self.crop[0])
                       * (self.bucket[1] + self.crop[1]))
        if scaled_area == 0:
            return 0.0
        cropped_area = scaled_area - (self.bucket[0] * self.bucket[1])
        return cropped_area / scaled_area


def make_bucket_resolutions(config: BucketConfig) -> list[tuple[int, int]]:
    """
    Enumerate all valid bucket resolutions, matching kohya's
    `make_bucket_resolutions`. Widths are stepped from min to max; for each
    width the largest step-aligned height whose area is <= target_area is used.
    Returns a de-duplicated, sorted list of (width, height) tuples.
    """
    target_area = config.target_area
    resolutions = set()
    width = config.min_resolution
    while width <= config.max_resolution:
        # Largest height (aligned to the step) that keeps area <= target_area.
        height = min(
            config.max_resolution,
            (target_area // width) // config.steps * config.steps)
        if height >= config.min_resolution:
            resolutions.add((width, height))
            resolutions.add((height, width))
        width += config.steps
    return sorted(resolutions)


def assign_bucket(dimensions: tuple[int, int],
                  config: BucketConfig,
                  bucket_resolutions: list[tuple[int, int]] | None = None
                  ) -> BucketAssignment:
    """
    Assign a single image (given its (width, height)) to the closest-aspect
    bucket and compute the resulting resize scale and crop, matching kohya's
    behavior. When `allow_upscaling` is False and the image is smaller than the
    target area, the image keeps its own step-aligned resolution instead.
    """
    if bucket_resolutions is None:
        bucket_resolutions = make_bucket_resolutions(config)
    width, height = dimensions
    aspect_ratio = width / height

    if not config.allow_upscaling and width * height < config.target_area:
        # No-upscale: snap the image's own size down to the step grid.
        bucket_width = max(config.steps,
                           (width // config.steps) * config.steps)
        bucket_height = max(config.steps,
                            (height // config.steps) * config.steps)
        bucket = (bucket_width, bucket_height)
    else:
        # Pick the bucket whose aspect ratio is closest to the image's.
        bucket = min(
            bucket_resolutions,
            key=lambda resolution: abs(
                (resolution[0] / resolution[1]) - aspect_ratio))

    bucket_width, bucket_height = bucket
    # kohya scales so the image covers the bucket (max of the two ratios),
    # then center-crops the overflow.
    scale = max(bucket_width / width, bucket_height / height)
    scaled_width = round(width * scale)
    scaled_height = round(height * scale)
    crop = (max(0, scaled_width - bucket_width),
            max(0, scaled_height - bucket_height))
    is_upscaled = scale > 1.0
    return BucketAssignment(bucket=bucket, scale=scale, crop=crop,
                            is_upscaled=is_upscaled)


def plan_resize_crop(source_size: tuple[int, int], bucket: tuple[int, int]
                     ) -> tuple[tuple[int, int], tuple[int, int, int, int]]:
    """
    Given a source image size and a target bucket, return the intermediate
    resize size and the center-crop box (left, top, right, bottom) that turns
    the source into exactly the bucket resolution, matching kohya's cover-then-
    center-crop behavior.
    """
    width, height = source_size
    bucket_width, bucket_height = bucket
    scale = max(bucket_width / width, bucket_height / height)
    scaled_width = round(width * scale)
    scaled_height = round(height * scale)
    left = max(0, (scaled_width - bucket_width) // 2)
    top = max(0, (scaled_height - bucket_height) // 2)
    crop_box = (left, top, left + bucket_width, top + bucket_height)
    return (scaled_width, scaled_height), crop_box


def bucket_distribution(image_dimensions: list[tuple[int, int]],
                        config: BucketConfig
                        ) -> dict[tuple[int, int], int]:
    """
    Return a mapping of bucket resolution -> number of images assigned to it,
    for a list of image dimensions.
    """
    bucket_resolutions = make_bucket_resolutions(config)
    distribution: dict[tuple[int, int], int] = {}
    for dimensions in image_dimensions:
        assignment = assign_bucket(dimensions, config, bucket_resolutions)
        distribution[assignment.bucket] = (
            distribution.get(assignment.bucket, 0) + 1)
    return distribution
