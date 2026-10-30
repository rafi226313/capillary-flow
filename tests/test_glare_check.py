"""Tests for src/analysis/glare_check.py on synthetic captures with known glare."""

import json

import numpy as np
import pytest
import tifffile

from src.analysis.glare_check import analyze_set

TISSUE_DN = 300.0   # skin light at the camera
GLARE_DN = 1200.0   # light that never touched tissue: glare share 0.8
SCOPE_PART = 0.5    # half of the glare comes from inside the scope
DARK_DN = 7.0
DIP = 0.30          # capillaries are 30% darker than the skin in the tissue light alone


def tissue_image(shape=(200, 300), sigma_px=4.0):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(float)
    profile = np.zeros(shape)
    for y0 in (60, 100, 140):
        profile = np.maximum(profile, np.exp(-0.5 * ((yy - y0) / sigma_px) ** 2) * ((xx > 30) & (xx < 270)))
    return TISSUE_DN * (1 - DIP * profile)


def write_capture(folder, role, half_res, exposure_us=2000.0, n_frames=5, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for _ in range(n_frames):
        noisy = half_res + rng.normal(0, 1.0, half_res.shape)
        frames.append(np.kron(noisy, np.ones((2, 2))))  # every Bayer site sees the same light
    stem = f'glare_epi_{role}_20261103-120000_BayerRG12_{exposure_us:g}us_0.0dB_x{n_frames}'
    tifffile.imwrite(folder / f'{stem}.tif', np.clip(np.round(frames), 0, 4095).astype(np.uint16),
                     photometric='minisblack')
    camera = {'pixel_format': 'BayerRG12', 'exposure_us': exposure_us, 'gain_db': 0.0, 'black_level': 8.0,
              'bsl_contrast': 0.0, 'bsl_brightness': 0.0}
    sidecar = {'frame_count': n_frames, 'pixel_format': 'BayerRG12', 'camera': camera,
               'camera_timestamps': [i * 5_000_000 for i in range(n_frames)], 'bench': {'note': 'LED knob 3'}}
    (folder / f'{stem}.json').write_text(json.dumps(sidecar))


def write_set(folder, black_exposure_us=2000.0):
    shape = (200, 300)
    glare = GLARE_DN * np.ones(shape)
    write_capture(folder, 'dark', np.full(shape, DARK_DN), seed=1)
    write_capture(folder, 'finger', DARK_DN + glare + tissue_image(shape), seed=2)
    write_capture(folder, 'empty', DARK_DN + SCOPE_PART * glare, seed=3)
    write_capture(folder, 'black', DARK_DN + glare, exposure_us=black_exposure_us, seed=4)


def test_recovers_glare_share_and_dip_depth(tmp_path):
    write_set(tmp_path)
    res = analyze_set(str(tmp_path), 'glare', 'epi', um_px=1.4)
    share = GLARE_DN / (GLARE_DN + TISSUE_DN)
    assert res['glare_share'] == pytest.approx(share, abs=0.02)
    assert res['scope_glare_share'] == pytest.approx(SCOPE_PART * share, abs=0.02)
    assert res['dip_glare_free_median'] == pytest.approx(DIP, abs=0.06)
    assert res['dip_measured_median'] == pytest.approx(DIP * (1 - share), abs=0.015)
    assert res['centreline_px'] > 300


def test_missing_set_returns_none(tmp_path):
    assert analyze_set(str(tmp_path), 'glare', 'side', um_px=1.4) is None


def test_mismatched_exposure_stops(tmp_path):
    write_set(tmp_path, black_exposure_us=4000.0)
    with pytest.raises(SystemExit, match='Settings differ'):
        analyze_set(str(tmp_path), 'glare', 'epi', um_px=1.4)
