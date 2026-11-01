"""Fixed controlled alterations, restricted to the source part mask."""
from __future__ import annotations

import cv2
import numpy as np

from app.settings import ImageMatchSettings


def corrupt(source: np.ndarray, mask: np.ndarray, recipe: str, cfg: ImageMatchSettings) -> np.ndarray:
    if mask.shape != source.shape[:2] or not np.any(mask):
        raise ValueError("invalid part mask")
    part = mask != 0
    if recipe == "blur":
        changed = cv2.GaussianBlur(source, (cfg.corruption_blur, cfg.corruption_blur), 0)
    elif recipe == "recolour":
        changed = 255 - source
    elif recipe == "shift":
        changed = np.roll(source, cfg.corruption_shift, axis=1)
    else:
        raise ValueError(f"unknown recipe: {recipe}")
    output = source.copy()
    output[part] = changed[part]
    return output
