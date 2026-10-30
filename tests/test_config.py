"""Tests for src/config.py on machines outside the hostname table."""

import os

from src.config import PATHS, load_source_sans


def test_paths_resolve_to_existing_folders():
    assert os.path.isdir(PATHS['cap_flow'])
    assert os.path.isdir(PATHS['frog_analysis'])


def test_font_loader_never_raises():
    font = load_source_sans()
    assert font is None or os.path.isfile(font.get_file())
