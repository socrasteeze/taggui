"""
Tests for `utils.bucketing`, which mirrors kohya_ss / sd-scripts bucketing.

These lock in the behaviour `Plan.md` describes, including the worked example
it cites (1920x1080 at a 1024 target area).
"""
import pytest

from utils.bucketing import (BucketAssignment, BucketConfig, assign_bucket,
                             bucket_distribution, make_bucket_resolutions,
                             plan_resize_crop)


def test_default_config_target_area_is_the_square_of_the_edge():
    assert BucketConfig().target_area == 1024 * 1024


class TestMakeBucketResolutions:
    def test_every_bucket_is_step_aligned_and_within_bounds(self):
        config = BucketConfig()
        resolutions = make_bucket_resolutions(config)
        assert resolutions
        for width, height in resolutions:
            assert width % config.steps == 0
            assert height % config.steps == 0
            assert config.min_resolution <= width <= config.max_resolution
            assert config.min_resolution <= height <= config.max_resolution

    def test_no_bucket_exceeds_the_target_area(self):
        config = BucketConfig()
        for width, height in make_bucket_resolutions(config):
            assert width * height <= config.target_area

    def test_the_square_bucket_is_present_at_the_target_edge(self):
        assert (1024, 1024) in make_bucket_resolutions(BucketConfig())

    def test_the_set_is_symmetric_so_portrait_matches_landscape(self):
        resolutions = set(make_bucket_resolutions(BucketConfig()))
        for width, height in resolutions:
            assert (height, width) in resolutions

    def test_results_are_sorted_and_free_of_duplicates(self):
        resolutions = make_bucket_resolutions(BucketConfig())
        assert resolutions == sorted(set(resolutions))

    def test_a_smaller_target_area_yields_smaller_buckets(self):
        buckets_512 = make_bucket_resolutions(
            BucketConfig(target_area_resolution=512))
        assert (512, 512) in buckets_512
        assert max(w * h for w, h in buckets_512) <= 512 * 512


class TestAssignBucket:
    def test_the_worked_example_from_the_roadmap(self):
        """1920x1080 at a 1024 target area buckets to 1344x768."""
        assignment = assign_bucket((1920, 1080), BucketConfig())
        assert assignment.bucket == (1344, 768)

    def test_a_square_image_lands_in_the_square_bucket_untouched(self):
        assignment = assign_bucket((1024, 1024), BucketConfig())
        assert assignment.bucket == (1024, 1024)
        assert assignment.scale == 1.0
        assert assignment.crop == (0, 0)
        assert not assignment.is_upscaled

    def test_a_portrait_image_gets_a_portrait_bucket(self):
        assignment = assign_bucket((1080, 1920), BucketConfig())
        assert assignment.bucket == (768, 1344)

    def test_a_small_image_is_flagged_as_upscaled(self):
        assignment = assign_bucket((256, 256), BucketConfig())
        assert assignment.is_upscaled
        assert assignment.scale > 1.0

    def test_no_upscale_keeps_the_images_own_step_aligned_size(self):
        config = BucketConfig(allow_upscaling=False)
        assignment = assign_bucket((500, 300), config)
        # 500 -> 448 and 300 -> 256, both snapped down to the 64 px grid.
        assert assignment.bucket == (448, 256)
        assert not assignment.is_upscaled

    def test_no_upscale_never_snaps_below_one_step(self):
        config = BucketConfig(allow_upscaling=False)
        assignment = assign_bucket((40, 30), config)
        assert assignment.bucket == (config.steps, config.steps)

    def test_no_upscale_still_buckets_images_at_or_above_the_target_area(self):
        config = BucketConfig(allow_upscaling=False)
        assignment = assign_bucket((1920, 1080), config)
        assert assignment.bucket == (1344, 768)

    def test_cropping_only_ever_removes_pixels(self):
        for dimensions in [(1920, 1080), (1000, 1000), (3000, 500), (7, 4000)]:
            assignment = assign_bucket(dimensions, BucketConfig())
            assert assignment.crop[0] >= 0
            assert assignment.crop[1] >= 0

    def test_an_extreme_aspect_ratio_picks_the_most_extreme_bucket(self):
        config = BucketConfig()
        assignment = assign_bucket((4000, 200), config)
        widest = max(make_bucket_resolutions(config),
                     key=lambda resolution: resolution[0] / resolution[1])
        assert assignment.bucket == widest

    def test_passing_precomputed_resolutions_matches_computing_them(self):
        config = BucketConfig()
        resolutions = make_bucket_resolutions(config)
        assert (assign_bucket((1920, 1080), config, resolutions)
                == assign_bucket((1920, 1080), config))


class TestCropFraction:
    def test_an_exact_fit_crops_nothing(self):
        assignment = assign_bucket((1024, 1024), BucketConfig())
        assert assignment.crop_fraction == 0.0

    def test_a_mismatched_aspect_ratio_reports_a_positive_fraction(self):
        assignment = BucketAssignment(bucket=(1024, 1024), scale=1.0,
                                      crop=(1024, 0), is_upscaled=False)
        # A 2048x1024 scaled image cropped to 1024x1024 loses half its area.
        assert assignment.crop_fraction == pytest.approx(0.5)

    def test_a_degenerate_bucket_does_not_divide_by_zero(self):
        assignment = BucketAssignment(bucket=(0, 0), scale=1.0, crop=(0, 0),
                                      is_upscaled=False)
        assert assignment.crop_fraction == 0.0


class TestPlanResizeCrop:
    def test_the_crop_box_produces_exactly_the_bucket_size(self):
        (scaled_width, scaled_height), crop_box = plan_resize_crop(
            (1920, 1080), (1344, 768))
        left, top, right, bottom = crop_box
        assert right - left == 1344
        assert bottom - top == 768
        assert scaled_width >= 1344
        assert scaled_height >= 768

    def test_the_crop_is_centred(self):
        _, (left, top, right, bottom) = plan_resize_crop((2048, 1024),
                                                         (1024, 1024))
        scaled_width = 2048
        assert left == (scaled_width - 1024) // 2
        assert top == 0

    def test_an_exact_fit_needs_no_crop(self):
        scaled, crop_box = plan_resize_crop((1024, 1024), (1024, 1024))
        assert scaled == (1024, 1024)
        assert crop_box == (0, 0, 1024, 1024)

    def test_it_agrees_with_assign_bucket(self):
        config = BucketConfig()
        assignment = assign_bucket((1920, 1080), config)
        (scaled_width, scaled_height), _ = plan_resize_crop(
            (1920, 1080), assignment.bucket)
        assert (scaled_width - assignment.bucket[0],
                scaled_height - assignment.bucket[1]) == assignment.crop


class TestBucketDistribution:
    def test_counts_sum_to_the_number_of_images(self):
        dimensions = [(1920, 1080), (1080, 1920), (1024, 1024), (1920, 1080)]
        distribution = bucket_distribution(dimensions, BucketConfig())
        assert sum(distribution.values()) == len(dimensions)

    def test_identical_images_share_one_bucket(self):
        distribution = bucket_distribution([(1920, 1080)] * 3, BucketConfig())
        assert distribution == {(1344, 768): 3}

    def test_an_empty_dataset_yields_an_empty_distribution(self):
        assert bucket_distribution([], BucketConfig()) == {}
