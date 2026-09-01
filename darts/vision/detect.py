"""Dart detection by background differencing, plus tip localisation.

Tip localisation is the whole ballgame. A dart sticks 30-40 mm out of the
board, so from an off-axis camera the *barrel* can appear a centimetre or more
from where the point actually went in. Taking the blob's centroid, or its
extreme point along the axis, both give you the wrong answer often enough to
ruin the score.

The heuristic here: a dart silhouette tapers at the point and flares at the
flight. Fit the principal axis, look at both ends, and take the narrower one.
That is geometry-driven rather than position-driven, so it holds up regardless
of which side of the board the dart landed on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class DetectorConfig:
    min_area: int = 250  # px; below this it's noise or a shadow edge
    max_area: int = 40_000  # above this it's a hand or an arm
    diff_threshold: int = 28
    tip_fraction: float = 0.15  # portion of the blob length treated as "an end"
    min_elongation: float = 2.0  # length/width; darts are long and thin


@dataclass
class Blob:
    tip: tuple[float, float]  # px, in camera image space
    centroid: tuple[float, float]
    area: float
    elongation: float
    angle_deg: float  # principal axis, for debugging overlays


class BackgroundModel:
    """Median-of-N background with explicit re-baselining.

    Deliberately not an adaptive MOG subtractor: a dart that stays in the board
    for 30 seconds must *not* fade into the background, or the removal
    detection that drives auto-advance stops working.
    """

    def __init__(self, frames: int = 9):
        self.frames = frames
        self._buf: list[np.ndarray] = []
        self.background: np.ndarray | None = None

    def add(self, gray: np.ndarray) -> None:
        self._buf.append(gray)
        del self._buf[: -self.frames]

    def commit(self) -> bool:
        """Freeze the buffered frames as the new background."""
        if len(self._buf) < self.frames:
            return False
        self.background = np.median(np.stack(self._buf), axis=0).astype(np.uint8)
        return True

    def reset(self) -> None:
        self._buf.clear()
        self.background = None

    @property
    def ready(self) -> bool:
        return self.background is not None


def preprocess(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def foreground_mask(gray: np.ndarray, background: np.ndarray, cfg: DetectorConfig) -> np.ndarray:
    diff = cv2.absdiff(gray, background)
    _, mask = cv2.threshold(diff, cfg.diff_threshold, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask


def change_mass(gray: np.ndarray, background: np.ndarray, cfg: DetectorConfig) -> int:
    """Number of changed pixels -- the cheap trigger signal."""
    return int(cv2.countNonZero(foreground_mask(gray, background, cfg)))


def _tip_from_points(pts: np.ndarray, cfg: DetectorConfig) -> tuple[tuple[float, float], float, float]:
    """Locate the dart point within a blob's pixel set.

    Returns (tip_xy, elongation, axis_angle_deg).
    """
    pts = pts.astype(np.float32)
    mean = pts.mean(axis=0)
    centred = pts - mean

    # Principal axis via SVD -- more stable than cv2.PCACompute on thin blobs.
    vt = np.linalg.svd(centred, full_matrices=False)[2]
    axis = vt[0]
    perp = vt[1]

    t = centred @ axis  # position along the dart
    p = centred @ perp  # offset across it

    length = float(t.max() - t.min())
    width = float(np.percentile(p, 95) - np.percentile(p, 5)) or 1e-6
    elongation = length / width

    span = max(length * cfg.tip_fraction, 1.0)
    lo_end = p[t <= t.min() + span]
    hi_end = p[t >= t.max() - span]

    lo_spread = float(lo_end.std()) if lo_end.size else 1e9
    hi_spread = float(hi_end.std()) if hi_end.size else 1e9

    # The narrow end is the point; the flared end is the flight.
    if lo_spread <= hi_spread:
        t_tip = float(np.percentile(t, 1.0))
    else:
        t_tip = float(np.percentile(t, 99.0))

    # Project back, keeping the perpendicular offset of the pixels near that end.
    near = np.abs(t - t_tip) <= span
    p_tip = float(np.median(p[near])) if near.any() else 0.0
    tip = mean + axis * t_tip + perp * p_tip

    angle = float(np.degrees(np.arctan2(axis[1], axis[0])))
    return (float(tip[0]), float(tip[1])), elongation, angle


def find_darts(
    gray: np.ndarray,
    background: np.ndarray,
    cfg: DetectorConfig | None = None,
    known: int = 0,
) -> list[Blob]:
    """Find dart-shaped blobs that are new since `background`.

    `known` is how many darts were already in the board; it's only used for
    logging, since each detection pass re-baselines the background after
    scoring a dart.
    """
    cfg = cfg or DetectorConfig()
    mask = foreground_mask(gray, background, cfg)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    blobs: list[Blob] = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (cfg.min_area <= area <= cfg.max_area):
            continue
        pts = c.reshape(-1, 2)
        if len(pts) < 5:
            continue
        tip, elongation, angle = _tip_from_points(pts, cfg)
        if elongation < cfg.min_elongation:
            log.debug("rejected blob: elongation %.1f below threshold", elongation)
            continue
        m = cv2.moments(c)
        cx = m["m10"] / m["m00"] if m["m00"] else tip[0]
        cy = m["m01"] / m["m00"] if m["m00"] else tip[1]
        blobs.append(Blob(tip, (cx, cy), area, elongation, angle))

    blobs.sort(key=lambda b: b.area, reverse=True)
    if blobs:
        log.debug("found %d dart blob(s) with %d already in board", len(blobs), known)
    return blobs


def fuse(points_mm: list[tuple[float, float]], disagree_mm: float = 8.0) -> tuple[tuple[float, float], float]:
    """Combine per-camera board-plane estimates into one point plus a confidence.

    With two cameras there is no majority to take, so this reports the midpoint
    and lets the spread drive confidence. A wide spread nearly always means one
    camera mistook the barrel for the point -- exactly the case where the UI
    should be inviting a correction rather than asserting a score.
    """
    if not points_mm:
        raise ValueError("no points to fuse")
    arr = np.array(points_mm, np.float64)
    if len(arr) == 1:
        return (float(arr[0][0]), float(arr[0][1])), 0.6

    centre = arr.mean(axis=0)
    spread = float(np.max(np.linalg.norm(arr - centre, axis=1)))
    confidence = 0.95 if spread <= disagree_mm else max(0.2, 0.95 - spread / 60.0)
    return (float(centre[0]), float(centre[1])), confidence
