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
    # Above this it is a hand or an arm, not a dart. Measured on this setup:
    # real darts came in at 255-1100 px, the one arm that got scored was 16627.
    max_area: int = 6_000
    diff_threshold: int = 28
    tip_fraction: float = 0.15  # portion of the blob length treated as "an end"
    min_elongation: float = 2.0  # length/width; darts are long and thin
    # A dark dart over a black sector barely differs from it, so one dart
    # arrives as several disconnected pieces. These control putting it back
    # together: keep pieces well below dart size, then merge what is collinear.
    fragment_min_area: int = 60
    merge_tolerance_px: float = 16.0  # perpendicular slack when joining pieces
    max_dart_span_px: float = 260.0  # longest a single dart can plausibly be


@dataclass
class Blob:
    tip: tuple[float, float]  # px, in camera image space -- the taper cue's pick
    other_end: tuple[float, float]  # the far end of the same axis
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

    def commit(self, cfg: "DetectorConfig | None" = None, quiet_px: int = 0) -> bool:
        """Freeze the buffered frames as the new background.

        `quiet_px` refuses to commit while the scene is still moving: if the
        oldest and newest buffered frames differ by more than that many pixels,
        something is walking about in shot and this is the worst possible
        moment to decide what the empty board looks like.

        That is not a nicety. Baselining a person into the background is
        unrecoverable on its own terms: every later frame then differs from it
        by roughly a whole person, the mass never falls back under the "board is
        quiet" threshold, and the pipeline sits in the hand state ignoring every
        dart thrown at it. Measured in play as "it stops counting after I walk
        up to the board".
        """
        if len(self._buf) < self.frames:
            return False
        if quiet_px:
            moving = int(
                cv2.countNonZero(
                    cv2.threshold(
                        cv2.absdiff(self._buf[0], self._buf[-1]),
                        (cfg or DetectorConfig()).diff_threshold,
                        255,
                        cv2.THRESH_BINARY,
                    )[1]
                )
            )
            if moving > quiet_px:
                # Drop the oldest frame so the window slides rather than
                # deadlocking on a buffer that will never be quiet.
                del self._buf[0]
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


def _inside(roi: np.ndarray, point: np.ndarray) -> bool:
    x, y = int(round(float(point[0]))), int(round(float(point[1])))
    h, w = roi.shape[:2]
    return 0 <= x < w and 0 <= y < h and bool(roi[y, x])


def change_mass(gray: np.ndarray, background: np.ndarray, cfg: DetectorConfig) -> int:
    """Number of changed pixels -- the cheap trigger signal."""
    return int(cv2.countNonZero(foreground_mask(gray, background, cfg)))


def _tip_from_points(pts: np.ndarray, cfg: DetectorConfig):
    """Locate the dart point within a blob's pixel set.

    Returns (tip_xy, other_end_xy, elongation, axis_angle_deg).

    Both ends come back because the taper cue below is weak when the dart points
    towards the camera -- the silhouette shortens, and the point and the flight
    stop looking different. Measured on this board it chose the flight end in 4
    of 13 throws. The caller has calibration and can apply the constraint this
    function cannot see: the point is *in* the board, so of two ends it is the
    one that lands on it.
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

    t_lo = float(np.percentile(t, 1.0))
    t_hi = float(np.percentile(t, 99.0))

    def at(t_end: float) -> tuple[float, float]:
        # Project back, keeping the perpendicular offset of the pixels near
        # that end, so the point sits on the silhouette rather than the axis.
        near = np.abs(t - t_end) <= span
        p_end = float(np.median(p[near])) if near.any() else 0.0
        pt = mean + axis * t_end + perp * p_end
        return float(pt[0]), float(pt[1])

    # The narrow end is the point; the flared end is the flight.
    if lo_spread <= hi_spread:
        tip, other = at(t_lo), at(t_hi)
    else:
        tip, other = at(t_hi), at(t_lo)

    angle = float(np.degrees(np.arctan2(axis[1], axis[0])))
    return tip, other, elongation, angle


def _merge_collinear(pieces: list[np.ndarray], cfg: DetectorConfig) -> list[np.ndarray]:
    """Join fragments that lie along one straight line.

    A dart is dark, and this board alternates black and yellow sectors, so the
    silhouette only shows up where it crosses a light one: differencing returns
    a *broken chain* of pieces rather than one dart. Measured on a real throw,
    one dart came back as six fragments spread over 230px, and taking the
    largest of them put the "tip" in the middle of the dart and scored a treble
    13 for a dart in the 3.

    A dart is rigid and straight, so its fragments are collinear -- which is
    what lets them be put back together.

    The line comes from a *pair* of fragments rather than from one fragment's
    own principal axis. A short fragment is a poor witness to the dart's
    direction: a 30x10 chunk reads as horizontal whichever way the dart
    actually runs, and seeding from it rejects the very pieces that would have
    corrected it. Every pair defines a candidate line, the one gathering the
    most fragments wins, and the process repeats on what is left.
    """
    if len(pieces) < 2:
        return list(pieces)

    centroids = [p.mean(axis=0) for p in pieces]
    remaining = list(range(len(pieces)))
    merged: list[np.ndarray] = []

    while len(remaining) > 1:
        best: tuple[tuple[int, int], list[int]] | None = None
        for idx, a in enumerate(remaining):
            for b in remaining[idx + 1:]:
                delta = centroids[b] - centroids[a]
                length = float(np.hypot(*delta))
                if length < 1e-6 or length > cfg.max_dart_span_px:
                    continue
                axis = delta / length
                normal = np.array([-axis[1], axis[0]])

                near = []
                for c in remaining:
                    offset = centroids[c] - centroids[a]
                    if abs(offset @ normal) <= cfg.merge_tolerance_px:
                        near.append((float(offset @ axis), c))
                # Keep only what fits inside one dart's length, measured as a
                # window that still contains the seed.
                near.sort()
                members = _longest_window(near, cfg.max_dart_span_px)
                score = (len(members), sum(len(pieces[m]) for m in members))
                if best is None or score > best[0]:
                    best = (score, members)

        if best is None or len(best[1]) < 2:
            break
        members = best[1]
        merged.append(np.vstack([pieces[m] for m in members]))
        remaining = [c for c in remaining if c not in members]

    merged.extend(pieces[c] for c in remaining)
    return merged


def _longest_window(along: list[tuple[float, int]], span: float) -> list[int]:
    """Most fragments fitting inside one dart's length, and containing the seed.

    `along` is (distance from the seed, index), sorted. The seed is at 0.
    """
    best: list[int] = []
    for i, (start, _) in enumerate(along):
        if start > 0:
            break  # a window starting past the seed cannot contain it
        window = [idx for pos, idx in along[i:] if pos - start <= span]
        if len(window) > len(best):
            best = window
    return best


def find_darts(
    gray: np.ndarray,
    background: np.ndarray,
    cfg: DetectorConfig | None = None,
    known: int = 0,
    roi: np.ndarray | None = None,
) -> list[Blob]:
    """Find dart-shaped blobs that are new since `background`.

    `known` is how many darts were already in the board; it's only used for
    logging, since each detection pass re-baselines the background after
    scoring a dart.

    `roi` is a mask of the board, and it matters once there is a second camera.
    The one looking down from above also takes in a doorway, a fridge and the
    dart holders on the cabinet doors, and to a differencing detector a person
    walking through that doorway is a large, dark, elongated blob -- the same
    description as a dart. Calibration already knows exactly where the board is.

    It is applied per contour rather than to the pixels, which is not a detail.
    Masking the pixels *cuts* a blob that straddles the edge, and the offcut is
    a new piece whose centroid has moved inwards -- far enough, measured here,
    to come within merging distance of a real dart. The two then merge into one
    shape too big to be a dart, and a mask meant to remove one false blob
    removes the true one along with it. Dropping whole contours cannot do that,
    and it also leaves each silhouette intact for the taper cue.
    """
    cfg = cfg or DetectorConfig()
    mask = foreground_mask(gray, background, cfg)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Keep pieces far smaller than a dart: they are dart *fragments*, and the
    # size test belongs after they have been reassembled.
    pieces = [
        c.reshape(-1, 2).astype(np.float32)
        for c in contours
        if len(c) >= 5 and cv2.contourArea(c) >= cfg.fragment_min_area
    ]
    if roi is not None:
        pieces = [p for p in pieces if _inside(roi, p.mean(axis=0))]
    if not pieces:
        return []

    blobs: list[Blob] = []
    for pts in _merge_collinear(pieces, cfg):
        hull = cv2.convexHull(pts.astype(np.float32))
        area = float(cv2.contourArea(hull))
        if not (cfg.min_area <= area <= cfg.max_area):
            continue
        tip, other_end, elongation, angle = _tip_from_points(pts, cfg)
        if elongation < cfg.min_elongation:
            log.debug("rejected blob: elongation %.1f below threshold", elongation)
            continue
        cx, cy = pts.mean(axis=0)
        blobs.append(Blob(tip, other_end, (float(cx), float(cy)), area, elongation, angle))

    blobs.sort(key=lambda b: b.area, reverse=True)
    if blobs:
        log.debug("found %d dart blob(s) with %d already in board", len(blobs), known)
    return blobs


def fuse(
    points_mm: list[tuple[float, float]],
    disagree_mm: float = 8.0,
    trust_one_mm: float = 25.0,
) -> tuple[tuple[float, float], float]:
    """Combine per-camera board-plane estimates into one point plus a confidence.

    `points_mm` is in camera-preference order: the first entry is the estimate
    to fall back on when the cameras cannot be reconciled.

    Averaging is right when the two roughly agree -- a dart stands out of the
    board and each camera misjudges that offset in a different direction, so the
    midpoint genuinely beats either alone. It is wrong when they do not. Two
    estimates 45mm apart are not a measurement with noise on it; one of them has
    mistaken the barrel for the point, and their midpoint is a place *neither*
    camera saw, in a third sector, arrived at with more confidence than either
    original deserved. Measured: a dart in the 20 fused to a point in the 4.

    Past `trust_one_mm` this returns the preferred camera's own estimate. That
    is not a claim about which camera is right -- it is a claim that a sector
    one camera actually reported beats a sector invented by splitting the
    difference. The low confidence rides along either way, and the UI puts a
    one-tap correction under a low-confidence dart.
    """
    if not points_mm:
        raise ValueError("no points to fuse")
    arr = np.array(points_mm, np.float64)
    if len(arr) == 1:
        return (float(arr[0][0]), float(arr[0][1])), 0.6

    centre = arr.mean(axis=0)
    spread = float(np.max(np.linalg.norm(arr - centre, axis=1)))
    confidence = 0.95 if spread <= disagree_mm else max(0.2, 0.95 - spread / 60.0)
    if spread > trust_one_mm:
        log.info(
            "cameras disagree by %.0fmm; taking the preferred one rather than "
            "a midpoint in neither", spread,
        )
        return (float(arr[0][0]), float(arr[0][1])), confidence
    return (float(centre[0]), float(centre[1])), confidence
