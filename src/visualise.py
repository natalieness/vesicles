"""Saving predictions as inspectable PNGs."""

from __future__ import annotations

import numpy as np
from PIL import Image

PRED_COLOR = (0, 255, 136)     # green: predicted objects
TRUTH_COLOR = (255, 45, 85)    # red: ground truth
OVERLAP_ALPHA = 0.35


def _boundaries(mask):
    """Outline of a binary mask, without pulling in scipy/skimage."""
    mask = mask.astype(bool)
    inner = np.ones_like(mask)
    inner[1:, :] &= mask[:-1, :]
    inner[:-1, :] &= mask[1:, :]
    inner[:, 1:] &= mask[:, :-1]
    inner[:, :-1] &= mask[:, 1:]
    return mask & ~inner


def save_prediction_overlay(path, image, prediction, truth=None):
    """Grayscale image with the prediction filled in and the truth outlined."""
    base = np.clip(image, 0, 1) if image.dtype != np.uint8 else image / 255.0
    rgb = np.repeat((base * 255).astype(np.uint8)[..., None], 3, axis=2).astype(np.float32)

    fill = prediction.astype(bool)
    rgb[fill] = (1 - OVERLAP_ALPHA) * rgb[fill] + OVERLAP_ALPHA * np.array(PRED_COLOR)

    if truth is not None:
        rgb[_boundaries(truth)] = np.array(TRUTH_COLOR, dtype=np.float32)

    Image.fromarray(rgb.round().clip(0, 255).astype(np.uint8)).save(path)


def save_mask(path, mask):
    """Binary mask as an 8-bit PNG (0 / 255)."""
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)
