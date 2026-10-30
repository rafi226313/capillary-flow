"""Tests for src/tools/load_bayer_stack.py."""

import json

import numpy as np
import pytest
import tifffile

from src.tools.load_bayer_stack import (
    TONE_KNEE_OUT,
    TONE_SLOPE_HIGH,
    TONE_SLOPE_LOW,
    BayerStackInfo,
    bayer_planes,
    green_channel,
    linearize,
    load_bayer_stack,
    normalize_frame_levels,
    stack_info,
)


def make_info(tone_curve_on: bool, black_level: float = 0.0) -> BayerStackInfo:
    return BayerStackInfo(path='x.tif', frame_count=1, pixel_format='BayerRG12', exposure_us=2000.0,
                          gain_db=0.0, black_level=black_level, fps=200.0, tone_curve_on=tone_curve_on,
                          um_per_px=None, um_per_px_source=None)


def rggb_mosaic(green: np.ndarray, red: float = 3000.0, blue: float = 4000.0) -> np.ndarray:
    """Mosaic whose green sites hold `green` and whose red/blue sites hold constants."""
    mosaic = green.astype(np.float32).copy()
    mosaic[0::2, 0::2] = red
    mosaic[1::2, 1::2] = blue
    return mosaic


def test_half_mode_averages_the_two_greens():
    rng = np.random.default_rng(0)
    green = rng.uniform(100, 200, (8, 12))
    half = green_channel(rggb_mosaic(green), 'RG', 'half')
    expected = 0.5 * (green[0::2, 1::2] + green[1::2, 0::2])
    assert half.shape == (4, 6)
    assert half.dtype == np.float32
    np.testing.assert_allclose(half, expected, rtol=1e-6)


def test_full_mode_is_exact_on_a_linear_ramp():
    yy, xx = np.mgrid[0:16, 0:20]
    ramp = 3.0 * yy + 2.0 * xx + 10.0
    full = green_channel(rggb_mosaic(ramp), 'RG', 'full')
    assert full.shape == ramp.shape
    # bilinear interpolation is exact for a plane away from the borders
    np.testing.assert_allclose(full[1:-1, 1:-1], ramp[1:-1, 1:-1], rtol=1e-6)


def test_full_mode_handles_stacks_and_borders():
    green = np.full((3, 6, 8), 150.0)
    mosaic = np.array([rggb_mosaic(g) for g in green])
    full = green_channel(mosaic, 'RG', 'full')
    np.testing.assert_allclose(full, 150.0, rtol=1e-6)


def test_bayer_patterns_place_green_correctly():
    mosaic = np.arange(16, dtype=np.float32).reshape(4, 4)
    planes = bayer_planes(mosaic, 'GR')
    np.testing.assert_array_equal(planes['G1'], mosaic[0::2, 0::2])
    np.testing.assert_array_equal(planes['G2'], mosaic[1::2, 1::2])
    np.testing.assert_array_equal(planes['R'], mosaic[0::2, 1::2])


def test_odd_frames_are_rejected():
    with pytest.raises(ValueError):
        green_channel(np.zeros((5, 6)), 'RG', 'half')


def test_linearize_undoes_the_tone_curve():
    raw = np.array([90.0, 200.0])
    out = linearize(raw, make_info(tone_curve_on=True))
    low = 90.0 / TONE_SLOPE_LOW + 0.6
    high = TONE_KNEE_OUT / TONE_SLOPE_LOW + (200.0 - TONE_KNEE_OUT) / TONE_SLOPE_HIGH + 0.6
    np.testing.assert_allclose(out, [low, high], rtol=1e-6)


def test_linearize_subtracts_dark_level_when_tone_curve_is_off():
    out = linearize(np.array([100.0]), make_info(tone_curve_on=False, black_level=8.0))
    np.testing.assert_allclose(out, [100.0 - (8.0 - 1.1)], rtol=1e-6)


def write_stack(tmp_path, camera_extra: dict, n_frames: int = 3):
    rng = np.random.default_rng(1)
    green = rng.integers(100, 300, (n_frames, 8, 10))
    mosaic = np.array([rggb_mosaic(g) for g in green]).astype(np.uint16)
    tif = tmp_path / 'burst.tif'
    tifffile.imwrite(tif, mosaic, photometric='minisblack')  # as camera_viewer/capture.py writes them
    camera = {'pixel_format': 'BayerRG12', 'exposure_us': 2000.0, 'gain_db': 0.0, 'black_level': 8.0,
              'resulting_frame_rate': 150.0, **camera_extra}
    sidecar = {'frame_count': n_frames, 'pixel_format': 'BayerRG12', 'um_per_px': None,
               'um_per_px_source': 'not measured', 'camera': camera,
               'camera_timestamps': [i * 5_000_000 for i in range(n_frames)]}
    (tmp_path / 'burst.json').write_text(json.dumps(sidecar))
    return str(tif), green


def test_load_bayer_stack_round_trip(tmp_path):
    tif, green = write_stack(tmp_path, {'bsl_contrast': 0.0, 'bsl_brightness': 0.0})
    frames, info = load_bayer_stack(tif, mode='half', linear=False)
    assert frames.shape == (3, 4, 5)
    assert frames.dtype == np.float32
    np.testing.assert_allclose(frames, 0.5 * (green[:, 0::2, 1::2] + green[:, 1::2, 0::2]))
    assert info.fps == pytest.approx(200.0)
    assert info.tone_curve_on is False

    linear, _ = load_bayer_stack(tif, mode='half', linear=True, frames=slice(1, 3))
    assert linear.shape == (2, 4, 5)
    np.testing.assert_allclose(linear, frames[1:3] - (8.0 - 1.1), rtol=1e-6)


def test_normalize_frame_levels_removes_global_flicker():
    rng = np.random.default_rng(2)
    scene = rng.uniform(100, 200, (16, 16)).astype(np.float32)
    gains = np.array([1.0, 1.02, 0.97, 1.005], dtype=np.float32)
    stack = scene[np.newaxis] * gains[:, np.newaxis, np.newaxis]
    out = normalize_frame_levels(stack)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, np.broadcast_to(scene * gains.mean(), out.shape), rtol=1e-5)


def test_sidecars_without_tone_fields_count_as_tone_curve_on(tmp_path):
    tif, _ = write_stack(tmp_path, {})
    assert stack_info(tif).tone_curve_on is True
    assert stack_info(tif, tone_curve_on=False).tone_curve_on is False
