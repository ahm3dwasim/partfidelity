"""Local, offline comparison of an edited image with its photographed source."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.sam import SamClient
from app.settings import ImageMatchSettings


@dataclass(frozen=True)
class Match:
    iou: float | None
    ssim: float | None
    change: float
    iou_pass: bool
    ssim_pass: bool
    edit_pass: bool
    reason: str

    @property
    def verdict(self) -> str:
        return "ACCEPTED" if self.iou_pass and self.ssim_pass and self.edit_pass else "REJECTED"


def _read(path: Path, flags: int) -> np.ndarray:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise ValueError(f"cannot decode image: {path}")
    return image


def _ssim(source: np.ndarray, candidate: np.ndarray, mask: np.ndarray, cfg: ImageMatchSettings) -> float:
    # Gaussian SSIM map, averaged over part pixels and all colour channels.
    a, b = source.astype(np.float32) / 255, candidate.astype(np.float32) / 255
    b[~mask] = a[~mask]
    blur = lambda x: cv2.GaussianBlur(x, (cfg.ssim_window, cfg.ssim_window), cfg.ssim_sigma)
    ma, mb = blur(a), blur(b)
    va, vb, cov = blur(a * a) - ma * ma, blur(b * b) - mb * mb, blur(a * b) - ma * mb
    score = ((2 * ma * mb + 0.01**2) * (2 * cov + 0.03**2)) / (
        (ma * ma + mb * mb + 0.01**2) * (va + vb + 0.03**2)
    )
    return float(np.mean(score[mask]))


def compare(source_path: Path, mask_path: Path, candidate_path: Path, phrase: str,
            sam: SamClient, cfg: ImageMatchSettings) -> Match:
    try:
        source = _read(source_path, cv2.IMREAD_COLOR)
        mask = _read(mask_path, cv2.IMREAD_GRAYSCALE) != 0
        candidate = _read(candidate_path, cv2.IMREAD_COLOR)
    except (ValueError, OSError, cv2.error) as exc:
        return Match(None, None, 0.0, False, False, False, str(exc))
    height, width = source.shape[:2]
    if mask.shape != (height, width) or not mask.any() or mask.all():
        return Match(None, None, 0.0, False, False, False, "source mask must match the image and leave background")
    original_size = candidate.shape[:2]
    candidate = cv2.resize(candidate, (width, height))
    change = float(np.mean(cv2.absdiff(source, candidate))) / 255
    edit_pass = change > cfg.noop_floor
    try:
        result = sam.segment_still(candidate_path, phrase, include_confidence=True)
        if result.status != "ok" or len(result.masks) != 1:
            return Match(None, None, change, False, False, edit_pass, "no unambiguous SAM mask")
        refound = cv2.imdecode(np.frombuffer(result.masks[0].mask_png, np.uint8), cv2.IMREAD_GRAYSCALE)
        if refound is None or refound.shape != original_size or not refound.any():
            return Match(None, None, change, False, False, edit_pass, "invalid SAM mask")
        refound = cv2.resize(refound, (width, height), interpolation=cv2.INTER_NEAREST)
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, cfg.ecc_iterations, cfg.ecc_epsilon)
        cv2.findTransformECC(cv2.cvtColor(source, cv2.COLOR_BGR2GRAY),
                             cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY), warp,
                             cv2.MOTION_EUCLIDEAN, criteria, (~mask).astype(np.uint8) * 255)
        aligned = cv2.warpAffine(candidate, warp, (width, height), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        aligned_mask = cv2.warpAffine(refound, warp, (width, height), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) != 0
        coverage = cv2.warpAffine(np.full((height, width), 255, np.uint8), warp,
                                  (width, height), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
        if not np.all(coverage[mask]):
            return Match(None, None, change, False, False, edit_pass, "registration loses part pixels")
    except Exception:
        return Match(None, None, change, False, False, edit_pass, "registration or SAM failed")
    union = np.count_nonzero(mask | aligned_mask)
    iou = float(np.count_nonzero(mask & aligned_mask) / union)
    score = _ssim(source, aligned, mask, cfg)
    iou_pass = iou >= cfg.iou_cutoff
    ssim_pass = cfg.ssim_cutoff is not None and score >= cfg.ssim_cutoff
    reasons = []
    if not iou_pass:
        reasons.append("SAM IoU below cutoff")
    if cfg.ssim_cutoff is None:
        reasons.append("SSIM cutoff not calibrated")
    elif not ssim_pass:
        reasons.append("masked SSIM below cutoff")
    if not edit_pass:
        reasons.append("no-op floor")
    return Match(iou, score, change, iou_pass, ssim_pass, edit_pass, "; ".join(reasons) or "faithful edit")
