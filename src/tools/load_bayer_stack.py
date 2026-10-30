"""
Filename: load_bayer_stack.py
------------------------------------------------------
Loads raw Bayer TIFF stacks from a colour Basler camera (e.g. the
acA1920-155ucMED in BayerRG12) as float32 green-channel frames.

Each stack is a multi-page uint16 TIFF holding 12-bit Bayer mosaics, with a
JSON sidecar of the same name written by the capture software (exposure,
black level, tone-curve state, camera timestamps). The rest of the pipeline
was written for 8-bit mono frames; reading these stacks with cv2.imread
would crush their contrast, so this module keeps float precision.

The tone-curve and dark-level constants were measured on our camera
(acA1920-155ucMED, firmware V1.6-0) from histogram combs and dark frames.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import tifffile
from scipy.ndimage import convolve

# Basler BslContrast S-curve as found on power-up: output = 5/9 x input up to
# 141 output DN, then a slope of about 0.615
TONE_KNEE_OUT = 141.0
TONE_SLOPE_LOW = 5.0 / 9.0
TONE_SLOPE_HIGH = 0.615
# True dark level in input DN, relative to the BlackLevel setting
DARK_BELOW_BLACK_LEVEL_TONE_ON = 0.6
DARK_BELOW_BLACK_LEVEL_TONE_OFF = 1.1

# Green sites (row, col) inside each 2x2 Bayer cell
GREEN_SITES = {
    'RG': ((0, 1), (1, 0)),
    'BG': ((0, 1), (1, 0)),
    'GR': ((0, 0), (1, 1)),
    'GB': ((0, 0), (1, 1)),
}
# Bilinear green interpolation: at a red or blue site the four direct
# neighbours are all green
GREEN_KERNEL = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], dtype=np.float32) / 4


@dataclass(frozen=True)
class BayerStackInfo:
    """Capture settings for one stack, read from its JSON sidecar.

    Attributes:
        path (str): path to the TIFF stack
        frame_count (int): number of frames in the stack
        pixel_format (str): camera pixel format, e.g. 'BayerRG12'
        exposure_us (float): exposure time in microseconds
        gain_db (float): analogue gain in dB
        black_level (float): BlackLevel setting in DN
        fps (float): frame rate from the camera timestamps
        tone_curve_on (bool): whether the camera's S-curve was applied
        um_per_px (float | None): native pixel size as recorded, if any
        um_per_px_source (str | None): where um_per_px came from
    """
    path: str
    frame_count: int
    pixel_format: str
    exposure_us: float
    gain_db: float
    black_level: float
    fps: float
    tone_curve_on: bool
    um_per_px: float | None
    um_per_px_source: str | None

    @property
    def bayer_pattern(self) -> str:
        """Two-letter Bayer layout, e.g. 'RG' for 'BayerRG12'."""
        if not self.pixel_format.startswith('Bayer'):
            raise ValueError(f'Not a Bayer pixel format: {self.pixel_format}')
        return self.pixel_format[5:7]


def read_sidecar(tiff_path: str) -> dict:
    """Read the JSON sidecar saved next to a TIFF stack.

    Args:
        tiff_path (str): path to the TIFF stack

    Returns:
        dict: the sidecar contents
    """
    sidecar_path = os.path.splitext(tiff_path)[0] + '.json'
    with open(sidecar_path) as f:
        return json.load(f)


def stack_info(tiff_path: str, tone_curve_on: bool | None = None) -> BayerStackInfo:
    """Collect the capture settings needed to load a stack.

    Args:
        tiff_path (str): path to the TIFF stack
        tone_curve_on (bool | None): override the tone-curve state. By default
            it is read from the sidecar; sidecars without the bsl_* fields
            predate 2026-10-01, when the S-curve was always on.

    Returns:
        BayerStackInfo: the capture settings
    """
    meta = read_sidecar(tiff_path)
    cam = meta['camera']

    if tone_curve_on is None:
        if cam.get('bsl_contrast') is None:
            tone_curve_on = True
        else:
            tone_curve_on = (abs(float(cam['bsl_contrast'])) > 1e-6
                             or abs(float(cam.get('bsl_brightness') or 0.0)) > 1e-6)

    timestamps = meta.get('camera_timestamps') or []
    if len(timestamps) > 1:
        fps = 1e9 / float(np.median(np.diff(timestamps)))  # timestamps are in ns
    else:
        fps = float(cam.get('resulting_frame_rate') or np.nan)

    return BayerStackInfo(
        path=tiff_path,
        frame_count=int(meta.get('frame_count') or len(timestamps)),
        pixel_format=meta.get('pixel_format') or cam['pixel_format'],
        exposure_us=float(cam['exposure_us']),
        gain_db=float(cam.get('gain_db') or 0.0),
        black_level=float(cam.get('black_level') or 0.0),
        fps=fps,
        tone_curve_on=bool(tone_curve_on),
        um_per_px=meta.get('um_per_px'),
        um_per_px_source=meta.get('um_per_px_source'),
    )


def linearize(raw: np.ndarray, info: BayerStackInfo) -> np.ndarray:
    """Convert raw DN to linear signal above the true dark level.

    Undoes the camera's S-curve when it was on, then subtracts the dark
    level. Works on whole mosaics, so it can run before green extraction.

    Args:
        raw (np.ndarray): raw frame(s) in DN, any shape
        info (BayerStackInfo): capture settings for the stack

    Returns:
        np.ndarray: float32 signal in linear input DN
    """
    y = raw.astype(np.float32)
    if info.tone_curve_on:
        x = np.where(y <= TONE_KNEE_OUT,
                     y / TONE_SLOPE_LOW,
                     TONE_KNEE_OUT / TONE_SLOPE_LOW + (y - TONE_KNEE_OUT) / TONE_SLOPE_HIGH)
        dark = info.black_level - DARK_BELOW_BLACK_LEVEL_TONE_ON
    else:
        x = y
        dark = info.black_level - DARK_BELOW_BLACK_LEVEL_TONE_OFF
    return (x - dark).astype(np.float32)


def bayer_planes(raw: np.ndarray, pattern: str = 'RG') -> dict[str, np.ndarray]:
    """Split Bayer mosaic(s) of shape (..., H, W) into four half-res planes.

    Args:
        raw (np.ndarray): Bayer mosaic(s) with even H and W
        pattern (str): two-letter layout of the top-left 2x2 cell, e.g. 'RG'

    Returns:
        dict: views keyed 'R', 'G1', 'G2', 'B' (G1 is the green on the
            top row of the cell, G2 the one on the bottom row)
    """
    cell = {
        'RG': {'R': (0, 0), 'G1': (0, 1), 'G2': (1, 0), 'B': (1, 1)},
        'BG': {'B': (0, 0), 'G1': (0, 1), 'G2': (1, 0), 'R': (1, 1)},
        'GR': {'G1': (0, 0), 'R': (0, 1), 'B': (1, 0), 'G2': (1, 1)},
        'GB': {'G1': (0, 0), 'B': (0, 1), 'R': (1, 0), 'G2': (1, 1)},
    }[pattern]
    return {name: raw[..., r::2, c::2] for name, (r, c) in cell.items()}


def green_channel(raw: np.ndarray, pattern: str = 'RG', mode: str = 'half') -> np.ndarray:
    """Extract the green channel from Bayer mosaic(s) as float32.

    Args:
        raw (np.ndarray): Bayer mosaic(s) of shape (H, W) or (N, H, W)
        pattern (str): two-letter Bayer layout, e.g. 'RG'
        mode (str): 'half' averages the two greens of each 2x2 cell, giving
            exact data at half resolution (pixel size doubles); 'full'
            interpolates green bilinearly at the red and blue sites

    Returns:
        np.ndarray: float32 green image(s)
    """
    raw = np.asarray(raw, dtype=np.float32)
    if raw.shape[-2] % 2 or raw.shape[-1] % 2:
        raise ValueError(f'Bayer frames need even height and width, got {raw.shape[-2:]}')

    if mode == 'half':
        planes = bayer_planes(raw, pattern)
        return 0.5 * (planes['G1'] + planes['G2'])

    if mode == 'full':
        mask = np.zeros(raw.shape[-2:], dtype=np.float32)
        for r, c in GREEN_SITES[pattern]:
            mask[r::2, c::2] = 1
        kernel = GREEN_KERNEL if raw.ndim == 2 else GREEN_KERNEL[np.newaxis]
        # 'mirror' keeps the Bayer phase at the borders
        return convolve(raw * mask, kernel, mode='mirror')

    raise ValueError(f"mode must be 'half' or 'full', got {mode!r}")


def normalize_frame_levels(stack: np.ndarray) -> np.ndarray:
    """Divide out frame-to-frame changes in overall brightness (LED jitter).

    Without this, a few tenths of a percent of light jitter shows up as
    vertical stripes in kymographs, which find_slopes_hough reads as very
    fast flow even when nothing is moving.

    Args:
        stack (np.ndarray): frames of shape (N, H, W)

    Returns:
        np.ndarray: float32 frames, each scaled to the stack's mean level
    """
    stack = np.asarray(stack, dtype=np.float32)
    levels = stack.reshape(len(stack), -1).mean(axis=1)
    return stack * (levels.mean() / levels)[:, np.newaxis, np.newaxis]


def load_bayer_stack(tiff_path: str, mode: str = 'half', linear: bool = True,
                     frames: slice | None = None,
                     tone_curve_on: bool | None = None) -> tuple[np.ndarray, BayerStackInfo]:
    """Load a raw Bayer TIFF stack as float32 green frames.

    Frames are converted one page at a time, so long recordings never hold
    a second full-size copy in memory.

    Args:
        tiff_path (str): path to the multi-page TIFF (sidecar alongside)
        mode (str): 'half' or 'full', see green_channel()
        linear (bool): undo the tone curve and subtract the dark level
        frames (slice | None): subset of frames to load, default all
        tone_curve_on (bool | None): override the sidecar's tone-curve state

    Returns:
        tuple: (frames, info) where frames has shape (N, H', W') in float32
            and info is the stack's BayerStackInfo. In 'half' mode the
            pixel size is twice info.um_per_px.
    """
    info = stack_info(tiff_path, tone_curve_on=tone_curve_on)
    pattern = info.bayer_pattern

    with tifffile.TiffFile(tiff_path) as tif:
        pages = tif.pages
        indices = range(len(pages))[frames] if frames is not None else range(len(pages))
        first = pages[indices[0]].asarray()
        if first.ndim != 2:
            raise ValueError(f'Expected one 2D Bayer frame per TIFF page, got shape {first.shape}')
        height, width = first.shape
        out_shape = (height // 2, width // 2) if mode == 'half' else (height, width)
        stack = np.empty((len(indices),) + out_shape, dtype=np.float32)
        for i, index in enumerate(indices):
            raw = first if i == 0 else pages[index].asarray()
            if linear:
                raw = linearize(raw, info)
            stack[i] = green_channel(raw, pattern, mode)

    return stack, info
