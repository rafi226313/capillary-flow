"""Tests for the kymograph helpers in src/make_kymograph.py."""

import numpy as np
from scipy.ndimage import convolve

from src.make_kymograph import (
    build_centerline_vs_time_kernal,
    compute_average_surrounding_pixels,
    create_circular_kernel,
)


def original_average(image_stack, radius=4):
    """The pre-float version of compute_average_surrounding_pixels, for 8-bit input."""
    kernel = create_circular_kernel(radius)
    return np.uint8(convolve(np.float32(image_stack), kernel[np.newaxis, :, :]))


def test_uint8_input_matches_the_original_behaviour():
    stack = np.random.default_rng(0).integers(0, 256, (5, 30, 40)).astype(np.uint8)
    out = compute_average_surrounding_pixels(stack)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, original_average(stack))


def test_float_input_keeps_sub_dn_precision():
    rng = np.random.default_rng(1)
    stack = (150.0 + 0.3 * rng.standard_normal((4, 30, 40))).astype(np.float32)
    out = compute_average_surrounding_pixels(stack)
    assert out.dtype == np.float32
    assert np.abs(out - np.round(out)).max() > 0.01  # not rounded to whole DN
    np.testing.assert_allclose(out.mean(), stack.mean(), rtol=1e-3)


def test_kymograph_tracks_a_moving_dark_spot():
    n_frames, length = 12, 40
    stack = np.full((n_frames, 20, length + 20), 200.0, dtype=np.float32)
    for t in range(n_frames):
        stack[t, 8:13, 10 + 2 * t:15 + 2 * t] = 100.0  # moves 2 px per frame along row 10
    centerline = np.array([[10, c] for c in range(10, 10 + length)])
    kymo = build_centerline_vs_time_kernal(stack, centerline, long=True)
    assert kymo.shape == (length, n_frames)
    darkest = kymo.argmin(axis=0)
    np.testing.assert_allclose(np.diff(darkest), 2, atol=1)
