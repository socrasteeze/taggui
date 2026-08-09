"""
Tests for `utils.onnx_preprocess`.

A wrong preprocessing stage does not raise - it yields a normal-looking array
that produces wrong tags. So each stage is pinned against the behaviour of the
reference implementation (`imgutils.preprocess.pillow`), including the parts
that are surprising.
"""
import numpy as np
import pytest
from PIL import Image as PilImage

from utils.onnx_preprocess import (UnknownPreprocessStageError, apply_stage,
                                   build_transform, center_crop, convert_rgb,
                                   normalize, pad_to_size, parse_color,
                                   parse_size, rescale, resize, to_tensor)


def make_image(width: int, height: int, mode: str = 'RGB',
               color=(255, 0, 0)) -> PilImage.Image:
    if mode == 'L':
        color = 128
    elif mode == 'RGBA' and len(color) == 3:
        color = color + (255,)
    return PilImage.new(mode, (width, height), color)


class TestParseSize:
    def test_an_integer_is_used_for_both_edges(self):
        assert parse_size(448, height_first=True) == (448, 448)

    def test_a_one_element_pair_is_used_for_both_edges(self):
        assert parse_size([448], height_first=False) == (448, 448)

    def test_a_height_first_pair_is_flipped_to_width_height(self):
        assert parse_size([100, 200], height_first=True) == (200, 100)

    def test_a_width_first_pair_is_left_alone(self):
        assert parse_size([100, 200], height_first=False) == (100, 200)

    def test_a_longer_pair_is_rejected(self):
        with pytest.raises(UnknownPreprocessStageError):
            parse_size([1, 2, 3], height_first=True)


class TestResize:
    def test_a_pair_gives_exactly_that_size(self):
        # The pair is (height, width), so this is a 200 wide, 100 tall result.
        resized = resize(make_image(50, 50), [100, 200])
        assert resized.size == (200, 100)

    def test_a_single_number_scales_the_shorter_edge(self):
        resized = resize(make_image(200, 100), 50)
        assert resized.size == (100, 50)

    def test_a_single_number_keeps_the_aspect_ratio_when_taller(self):
        resized = resize(make_image(100, 200), 50)
        assert resized.size == (50, 100)

    def test_max_size_clamps_the_longer_edge(self):
        resized = resize(make_image(400, 100), 100, max_size=200)
        assert max(resized.size) <= 200

    def test_an_image_already_at_the_size_is_returned_unchanged(self):
        image = make_image(64, 64)
        assert resize(image, [64, 64]) is image

    @pytest.mark.parametrize('interpolation',
                             ['nearest', 'bilinear', 'bicubic', 'box',
                              'hamming', 'lanczos'])
    def test_every_named_filter_is_accepted(self, interpolation):
        assert resize(make_image(32, 32), [16, 16],
                      interpolation=interpolation).size == (16, 16)

    def test_an_unknown_filter_is_rejected_by_name(self):
        with pytest.raises(UnknownPreprocessStageError, match='sinc'):
            resize(make_image(32, 32), [16, 16], interpolation='sinc')


class TestCenterCrop:
    def test_it_crops_to_the_requested_size(self):
        assert center_crop(make_image(100, 100), [40, 60]).size == (60, 40)

    def test_it_takes_the_middle(self):
        image = PilImage.new('RGB', (3, 1), (0, 0, 0))
        image.putpixel((1, 0), (255, 255, 255))
        assert center_crop(image, [1, 1]).getpixel((0, 0)) == (255, 255, 255)

    def test_an_undersized_image_is_padded_with_black_first(self):
        cropped = center_crop(make_image(10, 10, color=(255, 255, 255)),
                              [20, 20])
        assert cropped.size == (20, 20)
        assert cropped.getpixel((0, 0)) == (0, 0, 0)
        assert cropped.getpixel((10, 10)) == (255, 255, 255)


class TestPadToSize:
    def test_the_result_is_exactly_the_target_size(self):
        assert pad_to_size(make_image(100, 50), [200, 200]).size == (200, 200)

    def test_the_pair_is_width_first_unlike_the_other_stages(self):
        assert pad_to_size(make_image(10, 10), [40, 20]).size == (40, 20)

    def test_the_aspect_ratio_is_preserved(self):
        padded = pad_to_size(make_image(100, 50, color=(255, 0, 0)),
                             [200, 200], background_color='black')
        # A 2:1 image fitted into a square keeps 2:1, so the middle row is red
        # and the top row is background.
        assert padded.getpixel((100, 100)) == (255, 0, 0)
        assert padded.getpixel((100, 5)) == (0, 0, 0)

    def test_the_background_colour_is_used(self):
        padded = pad_to_size(make_image(100, 50), [200, 200],
                             background_color='blue')
        assert padded.getpixel((0, 0)) == (0, 0, 255)

    def test_an_rgb_tuple_background_works(self):
        padded = pad_to_size(make_image(100, 50), [200, 200],
                             background_color=[0, 255, 0])
        assert padded.getpixel((0, 0)) == (0, 255, 0)


class TestConvertRgb:
    def test_an_rgb_image_passes_straight_through(self):
        image = make_image(8, 8)
        assert convert_rgb(image) is image

    def test_transparency_is_flattened_onto_the_background(self):
        image = PilImage.new('RGBA', (4, 4), (255, 0, 0, 0))
        assert convert_rgb(image, force_background='white').getpixel((0, 0)) \
            == (255, 255, 255)

    def test_a_different_background_is_honoured(self):
        image = PilImage.new('RGBA', (4, 4), (255, 0, 0, 0))
        assert convert_rgb(image, force_background='black').getpixel((0, 0)) \
            == (0, 0, 0)

    def test_an_opaque_pixel_survives_flattening(self):
        image = PilImage.new('RGBA', (4, 4), (255, 0, 0, 255))
        assert convert_rgb(image, force_background='white').getpixel((0, 0)) \
            == (255, 0, 0)

    def test_greyscale_is_converted(self):
        assert convert_rgb(make_image(4, 4, mode='L')).mode == 'RGB'


class TestToTensor:
    def test_the_result_is_channel_first_float_in_zero_to_one(self):
        array = to_tensor(make_image(6, 4, color=(255, 0, 0)))
        assert array.shape == (3, 4, 6)
        assert array.dtype == np.float32
        assert array[0].max() == pytest.approx(1.0)
        assert array[1].max() == pytest.approx(0.0)

    def test_greyscale_gains_a_channel(self):
        array = to_tensor(make_image(6, 4, mode='L'))
        assert array.shape == (1, 4, 6)

    def test_an_alpha_channel_is_kept(self):
        array = to_tensor(make_image(6, 4, mode='RGBA'))
        assert array.shape == (4, 4, 6)

    def test_an_array_passes_through_unchanged(self):
        array = np.zeros((3, 4, 4), dtype=np.float32)
        assert to_tensor(array) is array

    def test_the_channel_order_is_not_reversed(self):
        """The WD taggers use BGR; these exports do not."""
        array = to_tensor(make_image(2, 2, color=(255, 0, 0)))
        assert array[0].mean() == pytest.approx(1.0)
        assert array[2].mean() == pytest.approx(0.0)


class TestNormalize:
    def test_it_subtracts_the_mean_and_divides_by_the_std(self):
        array = np.ones((3, 2, 2), dtype=np.float32)
        result = normalize(array, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        assert result == pytest.approx(np.ones((3, 2, 2)))

    def test_each_channel_uses_its_own_values(self):
        array = np.ones((3, 1, 1), dtype=np.float32)
        result = normalize(array, mean=[0.0, 1.0, 2.0], std=[1.0, 1.0, 1.0])
        assert result[:, 0, 0] == pytest.approx([1.0, 0.0, -1.0])

    def test_the_input_array_is_not_modified(self):
        array = np.ones((3, 2, 2), dtype=np.float32)
        normalize(array, mean=[1.0, 1.0, 1.0], std=[1.0, 1.0, 1.0])
        assert array == pytest.approx(np.ones((3, 2, 2)))

    def test_the_imagenet_values_land_in_the_expected_range(self):
        array = to_tensor(make_image(4, 4, color=(128, 128, 128)))
        result = normalize(array, mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
        assert -1.0 < float(result.min()) < 1.0
        assert -1.0 < float(result.max()) < 1.0


class TestRescale:
    def test_it_multiplies_by_the_factor(self):
        assert rescale(np.ones((2, 2), dtype=np.float32),
                       rescale_factor=0.5) == pytest.approx(
            np.full((2, 2), 0.5))

    def test_the_field_is_named_rescale_factor(self):
        """The reference calls it `rescale_factor`, not `scale`."""
        array = apply_stage(np.full((2, 2), 255.0, dtype=np.float32),
                            {'type': 'rescale', 'rescale_factor': 1 / 255})
        assert array == pytest.approx(np.ones((2, 2)))

    def test_it_defaults_to_one_over_255(self):
        assert rescale(np.full((2, 2), 255.0, dtype=np.float32)) \
            == pytest.approx(np.ones((2, 2)))


class TestStageDispatch:
    def test_an_unknown_stage_is_named_in_the_error(self):
        with pytest.raises(UnknownPreprocessStageError, match='rotate'):
            apply_stage(make_image(4, 4), {'type': 'rotate', 'angle': 90})

    def test_the_error_lists_the_stages_that_are_supported(self):
        with pytest.raises(UnknownPreprocessStageError, match='normalize'):
            apply_stage(make_image(4, 4), {'type': 'rotate'})

    def test_a_stage_with_no_type_is_rejected(self):
        with pytest.raises(UnknownPreprocessStageError):
            apply_stage(make_image(4, 4), {'size': 4})

    def test_an_unexpected_field_is_reported_rather_than_ignored(self):
        with pytest.raises(UnknownPreprocessStageError, match='center_crop'):
            apply_stage(make_image(8, 8),
                        {'type': 'center_crop', 'size': 4, 'nonsense': 1})

    def test_maybe_to_tensor_behaves_like_to_tensor(self):
        array = apply_stage(make_image(4, 4), {'type': 'maybe_to_tensor'})
        assert array.shape == (3, 4, 4)


class TestBuildTransform:
    def test_the_pixai_style_pipeline_produces_a_model_input(self):
        """
        The shape the tagger needs: 3x448x448 float32, normalized. This is the
        pipeline the deepghs exports describe.
        """
        transform = build_transform([
            {'type': 'convert_rgb', 'force_background': 'white'},
            {'type': 'resize', 'size': [448, 448], 'interpolation': 'bicubic'},
            {'type': 'center_crop', 'size': [448, 448]},
            {'type': 'to_tensor'},
            {'type': 'normalize', 'mean': [0.485, 0.456, 0.406],
             'std': [0.229, 0.224, 0.225]},
        ])
        array = transform(make_image(1024, 768, mode='RGBA'))
        assert array.shape == (3, 448, 448)
        assert array.dtype == np.float32

    def test_stages_run_in_order(self):
        """Crop then resize differs from resize then crop."""
        crop_first = build_transform([
            {'type': 'center_crop', 'size': [50, 50]},
            {'type': 'resize', 'size': [10, 20]},
        ])
        assert crop_first(make_image(100, 100)).size == (20, 10)

    def test_a_single_stage_object_is_accepted(self):
        transform = build_transform({'type': 'resize', 'size': [8, 8]})
        assert transform(make_image(32, 32)).size == (8, 8)

    def test_an_empty_pipeline_returns_the_image_untouched(self):
        image = make_image(8, 8)
        assert build_transform([])(image) is image

    def test_an_unknown_stage_fails_when_the_pipeline_runs(self):
        transform = build_transform([{'type': 'solarize'}])
        with pytest.raises(UnknownPreprocessStageError, match='solarize'):
            transform(make_image(8, 8))


class TestParseColor:
    def test_a_colour_name_becomes_rgb(self):
        assert parse_color('white', 'RGB') == (255, 255, 255)

    def test_rgb_is_padded_to_rgba(self):
        assert parse_color('red', 'RGBA') == (255, 0, 0, 255)

    def test_a_colour_collapses_to_a_level_for_greyscale(self):
        assert parse_color((30, 30, 30), 'L') == 30

    def test_an_integer_becomes_a_grey(self):
        assert parse_color(0, 'RGB') == (0, 0, 0)
