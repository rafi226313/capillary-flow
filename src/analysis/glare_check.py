"""
Filename: glare_check.py
------------------------------------------------------
Glare share and capillary dip depth from the capture protocol's glare check.

For each lighting set ('epi' through the objective, 'side' from the side) it
reads the camctl bursts labelled <prefix>_<set>_<role> (roles: dark, finger,
empty, black), all taken at one LED setting and exposure, and reports:

- glare share: the fraction of the finger image's light that is there without
  any tissue, (black - dark) / (finger - dark); 'empty' gives the part that
  comes from inside the scope alone
- dip depth: how much darker each capillary centreline is than the skin
  around it, as measured and with the glare taken out

Usage:
    python -m src.analysis.glare_check "Outputs/Captures/2026-11-03"
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, label
from skimage.filters import sato
from skimage.morphology import skeletonize

from src.tools.load_bayer_stack import BayerStackInfo, load_bayer_stack

ROLES = {'epi': ('dark', 'finger', 'empty', 'black'), 'side': ('dark', 'finger', 'empty', 'black')}
GLARE_SIGMA_UM = 20.0        # glare is smooth; average over ~20 um before dividing
BACKGROUND_SIGMA_UM = 35.0   # local skin level around each capillary
RIDGE_SIGMAS_UM = (3.0, 4.5, 6.0, 8.0)
MIN_VESSEL_UM2 = 400.0       # drop specks smaller than this from the vessel map


def find_capture(folder: str, label_: str) -> str | None:
    """Newest TIFF in folder whose name starts with '<label_>_'."""
    matches = sorted(glob.glob(os.path.join(folder, f'{label_}_*.tif')))
    return matches[-1] if matches else None


def mean_frame(path: str) -> tuple[np.ndarray, BayerStackInfo]:
    """Frame-averaged half-res green in raw DN; the camera's tone curve must be off."""
    frames, info = load_bayer_stack(path, mode='half', linear=False)
    if info.tone_curve_on:
        raise SystemExit(f'{os.path.basename(path)}: tone curve was on; run camctl raw-mode and recapture')
    return frames.mean(axis=0), info


def check_settings(infos: dict[str, BayerStackInfo]) -> None:
    """Stop unless every capture in a set used the same exposure, gain and black level."""
    keys = {role: (i.exposure_us, i.gain_db, i.black_level) for role, i in infos.items()}
    if len(set(keys.values())) > 1:
        lines = [f'  {role}: exposure {e:g} us, gain {g:g} dB, black level {b:g}' for role, (e, g, b) in keys.items()]
        raise SystemExit('Settings differ within the set, so the glare share would be wrong:\n' + '\n'.join(lines))


def glare_share(finger: np.ndarray, dark: np.ndarray, other: np.ndarray, um_px: float) -> np.ndarray:
    """Smoothed (other - dark) / (finger - dark): the share of the finger light also present in 'other'."""
    sigma = GLARE_SIGMA_UM / um_px
    return gaussian_filter(other - dark, sigma) / gaussian_filter(finger - dark, sigma)


def capillary_centrelines(image: np.ndarray, um_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Centreline pixels and vessel mask of the dark capillaries in a dark-subtracted image."""
    high_pass = gaussian_filter(image / gaussian_filter(image, BACKGROUND_SIGMA_UM / um_px) - 1, 1.0)
    ridge = sato(high_pass, sigmas=[s / um_px for s in RIDGE_SIGMAS_UM], black_ridges=True)
    border = int(BACKGROUND_SIGMA_UM / um_px)
    inner = np.zeros(image.shape, bool)
    inner[border:-border, border:-border] = True
    vessel = (ridge > np.percentile(ridge[inner], 93)) & (high_pass < 0) & inner
    labels, _ = label(vessel)
    sizes = np.bincount(labels.ravel())
    keep = np.nonzero(sizes >= MIN_VESSEL_UM2 / um_px ** 2)[0]
    vessel = np.isin(labels, keep[keep != 0])
    return skeletonize(vessel), vessel


def local_background(image: np.ndarray, exclude: np.ndarray, sigma_px: float) -> np.ndarray:
    """Gaussian-weighted mean of the image around each pixel, ignoring excluded pixels."""
    keep = (~exclude).astype(np.float32)
    return gaussian_filter(image * keep, sigma_px) / np.maximum(gaussian_filter(keep, sigma_px), 1e-6)


def dip_depths(finger: np.ndarray, dark: np.ndarray, black: np.ndarray | None,
               um_px: float) -> dict[str, np.ndarray]:
    """Dip depth at each capillary centreline pixel.

    Args:
        finger, dark, black: frame-averaged images in DN (black may be None)
        um_px: pixel size of the images

    Returns:
        dict with 'centreline' (bool mask), 'measured' = (B - V) / (B - dark) and,
        with a black target, 'glare_free' = (B - V) / (B - black), where V is the
        centreline level and B the local skin level around it
    """
    signal = finger - dark
    centreline, vessel = capillary_centrelines(signal, um_px)
    background = local_background(signal, binary_dilation(vessel, iterations=3), BACKGROUND_SIGMA_UM / um_px)
    valley = gaussian_filter(signal, 1.0)
    out = {'centreline': centreline, 'measured': ((background - valley) / background)[centreline]}
    if black is not None:
        tissue_background = background - gaussian_filter(black - dark, GLARE_SIGMA_UM / um_px)
        ok = tissue_background > 0.02 * background  # glare can't exceed the light it sits in
        out['glare_free'] = ((background - valley) / tissue_background)[centreline & ok]
    return out


def analyze_set(folder: str, prefix: str, set_name: str, um_px: float) -> dict | None:
    """Glare share and dip depths for one lighting set; None if it wasn't captured."""
    paths = {role: find_capture(folder, f'{prefix}_{set_name}_{role}') for role in ROLES[set_name]}
    if paths['finger'] is None or paths['dark'] is None:
        return None
    images, infos = {}, {}
    for role, path in paths.items():
        if path is not None:
            images[role], infos[role] = mean_frame(path)
    check_settings(infos)

    finger, dark = images['finger'], images['dark']
    centre = tuple(slice(n // 4, 3 * n // 4) for n in finger.shape)
    result = {'files': {r: os.path.basename(p) for r, p in paths.items() if p},
              'exposure_us': infos['finger'].exposure_us,
              'finger_dn_per_ms': float((finger - dark)[centre].mean() / (infos['finger'].exposure_us / 1000))}
    for role, key in (('black', 'glare_share'), ('empty', 'scope_glare_share')):
        if role in images:
            share = glare_share(finger, dark, images[role], um_px)
            result[key] = float(np.median(share[centre]))
            result[key + '_p10_p90'] = [float(np.percentile(share[centre], q)) for q in (10, 90)]
            if role == 'black':
                result['_share_map'] = share
    dips = dip_depths(finger, dark, images.get('black'), um_px)
    for key in ('measured', 'glare_free'):
        if key in dips and dips[key].size:
            result[f'dip_{key}_median'] = float(np.median(dips[key]))
            result[f'dip_{key}_p90'] = float(np.percentile(dips[key], 90))
            result[f'_dips_{key}'] = dips[key]
    result['centreline_px'] = int(dips['centreline'].sum())
    result['_finger'] = finger - dark
    result['_centreline'] = dips['centreline']
    return result


def plot_results(results: dict[str, dict], out_path: str) -> None:
    """Finger image with centrelines, glare-share map and dip histograms."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    epi = results.get('epi') or next(iter(results.values()))
    image = epi['_finger']
    axes[0].imshow(image / gaussian_filter(image, 25), cmap='gray', vmin=0.95, vmax=1.05)
    ys, xs = np.nonzero(epi['_centreline'])
    axes[0].plot(xs, ys, 'r.', ms=0.5)
    axes[0].set_title('Finger (first set), detected capillary centrelines', fontsize=9)
    if '_share_map' in epi:
        im = axes[1].imshow(100 * epi['_share_map'], cmap='magma', vmin=0, vmax=100)
        fig.colorbar(im, ax=axes[1], label='glare share (%)')
        axes[1].set_title('Glare share, (black - dark) / (finger - dark)', fontsize=9)
    else:
        axes[1].set_axis_off()
    bins = np.linspace(-0.05, 0.6, 66)
    for name, res in results.items():
        for key, style in (('measured', '-'), ('glare_free', '--')):
            if f'_dips_{key}' in res:
                axes[2].hist(res[f'_dips_{key}'], bins=bins, histtype='step', ls=style,
                             label=f'{name}, {key.replace("_", "-")}', density=True)
    axes[2].set_xlabel('dip depth (fraction of local skin level)')
    axes[2].set_ylabel('centreline pixels (density)')
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='Glare share and capillary dip depth from the glare check.')
    parser.add_argument('folder', help='capture folder, e.g. Outputs/Captures/2026-11-03')
    parser.add_argument('--prefix', default='glare', help="label prefix used in camctl (default 'glare')")
    parser.add_argument('--um-per-px', type=float, default=0.70, help='native pixel size (default 0.70, uncalibrated)')
    parser.add_argument('--out', help='output folder (default <folder>/glare_check)')
    args = parser.parse_args()

    um_px = 2 * args.um_per_px  # half-res green
    results = {name: res for name in ROLES if (res := analyze_set(args.folder, args.prefix, name, um_px))}
    if not results:
        raise SystemExit(f'No {args.prefix}_epi_* or {args.prefix}_side_* finger and dark captures in {args.folder}')

    out = args.out or os.path.join(args.folder, 'glare_check')
    os.makedirs(out, exist_ok=True)
    plot_results(results, os.path.join(out, 'glare_check.png'))
    summary = {name: {k: v for k, v in res.items() if not k.startswith('_')} for name, res in results.items()}
    with open(os.path.join(out, 'glare_check.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    for name, res in summary.items():
        print(f'{name}: finger signal {res["finger_dn_per_ms"]:.1f} DN/ms at {res["exposure_us"] / 1000:g} ms; '
              f'{res["centreline_px"]} centreline px')
        if 'glare_share' in res:
            print(f'  glare share {100 * res["glare_share"]:.0f}% (10-90%: '
                  f'{100 * res["glare_share_p10_p90"][0]:.0f}-{100 * res["glare_share_p10_p90"][1]:.0f}%)')
        if 'scope_glare_share' in res:
            print(f'  of which inside the scope {100 * res["scope_glare_share"]:.0f}%')
        print(f'  dip depth as measured: median {100 * res["dip_measured_median"]:.1f}%, '
              f'90th percentile {100 * res["dip_measured_p90"]:.1f}%')
        if 'dip_glare_free_median' in res:
            print(f'  dip depth without the glare: median {100 * res["dip_glare_free_median"]:.1f}%, '
                  f'90th percentile {100 * res["dip_glare_free_p90"]:.1f}%')
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
