"""Automatic dartboard calibration -- no clicking, no stored reference photo.

The pipeline exploits the one thing that makes this board *easier* than a
regulation one: the numbers and the double ring are printed straight onto a
big, high-contrast yellow/black face. The yellow paint is far more saturated
than the wood cabinet around it, so a colour mask isolates the board cleanly.

    1. yellow mask                -> isolate the painted board face
    2. convex hull + fitEllipse   -> the double ring's outer edge, as an ellipse
    3. affine rectify             -> ellipse to circle (rotation still unknown)
    4. rotational template match  -> 20 sector steps x mirror, pick best NCC
    5. ECC homography refinement  -> upgrade affine to full perspective

Step 5 matters: at ~60 cm from a 340 mm board the weak-perspective assumption
behind step 3 is off by several mm at the far rim, which is the difference
between a treble and a single.

Everything is matched against a *synthetically rendered* reference board, so
there is no one-time capture step to get wrong.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import cv2
import numpy as np

from ..board import SECTORS, BoardGeometry, REGULATION

log = logging.getLogger(__name__)

RECT_SIZE = 800  # side length of the rectified board image, px
MARGIN = 1.12  # rectified view extends this far past the double ring


def px_per_mm(geom: BoardGeometry = REGULATION) -> float:
    return (RECT_SIZE / 2.0) / (geom.double_outer * MARGIN)


# --------------------------------------------------------------------------
# synthetic reference
# --------------------------------------------------------------------------


def _stamp_number(mask: np.ndarray, value: int, angle_deg: float, radius_mm: float,
                  ppm: float, ink: int) -> None:
    """Draw one sector numeral into the double ring, rotated to follow the rim.

    `ink` is 255 to add yellow (a light numeral on a dark band) or 0 to punch a
    hole (a dark numeral on a yellow band), matching how the numbers are printed
    straight onto this board's face.
    """
    text = str(value)
    font, scale, thick = cv2.FONT_HERSHEY_DUPLEX, 0.9, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    side = int(max(tw, th) * 2.2)

    patch = np.zeros((side, side), np.uint8)
    cv2.putText(patch, text, ((side - tw) // 2, (side + th) // 2), font, scale, 255, thick, cv2.LINE_AA)
    # Negative angle == clockwise, so the numeral's "up" points outward.
    rot = cv2.getRotationMatrix2D((side / 2.0, side / 2.0), -angle_deg, 1.0)
    patch = cv2.warpAffine(patch, rot, (side, side))

    th_rad = math.radians(angle_deg)
    cx = int(round(RECT_SIZE / 2.0 + radius_mm * math.sin(th_rad) * ppm))
    cy = int(round(RECT_SIZE / 2.0 - radius_mm * math.cos(th_rad) * ppm))
    x0, y0 = cx - side // 2, cy - side // 2
    x1, y1 = x0 + side, y0 + side
    if x0 < 0 or y0 < 0 or x1 > RECT_SIZE or y1 > RECT_SIZE:
        return
    region = mask[y0:y1, x0:x1]
    region[patch > 127] = ink


def render_reference(geom: BoardGeometry = REGULATION) -> np.ndarray:
    """Render a canonical rectified board as a binary 'is yellow' mask.

    The numerals matter more than they look. Without them the pattern repeats
    every *two* sectors -- alternating singles with the ring colours inverted is
    36-degree symmetric -- so ten of the twenty candidate rotations score
    identically and the search picks one at random. That failure is silent and
    scores T20 as T18.

    The numbers are the only thing on the face that breaks the symmetry, and on
    this board they are printed large and high-contrast directly into the double
    ring, which is exactly what makes automatic calibration viable here.
    """
    size = RECT_SIZE
    ppm = px_per_mm(geom)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    x_mm = (xx - size / 2.0) / ppm
    y_mm = (size / 2.0 - yy) / ppm

    r = np.hypot(x_mm, y_mm)
    ang = (np.degrees(np.arctan2(x_mm, y_mm))) % 360.0
    idx = ((ang + 9.0) % 360.0 // 18.0).astype(np.int32)  # 0 == the 20

    # Alternating colour by sector index; the rings invert the parity.
    sector_yellow = (idx % 2) == 1
    in_double = (r > geom.double_inner) & (r <= geom.double_outer)
    in_triple = (r > geom.triple_inner) & (r <= geom.triple_outer)
    in_single = ((r > geom.outer_bull) & (r <= geom.triple_inner)) | (
        (r > geom.triple_outer) & (r <= geom.double_inner)
    )

    mask = np.zeros((size, size), np.uint8)
    mask[in_single & sector_yellow] = 255
    mask[(in_double | in_triple) & ~sector_yellow] = 255

    number_r = (geom.double_inner + geom.double_outer) / 2.0
    for i, value in enumerate(SECTORS):
        double_is_yellow = (i % 2) == 0  # rings invert the single parity
        _stamp_number(mask, value, i * 18.0, number_r, ppm, ink=0 if double_is_yellow else 255)
    return mask


# --------------------------------------------------------------------------
# board detection
# --------------------------------------------------------------------------


@dataclass
class YellowRange:
    """HSV bounds for the board's yellow. Tune once via tools/tune_mask.py."""

    h_lo: int = 15
    h_hi: int = 42
    s_lo: int = 70
    v_lo: int = 70


def yellow_mask(bgr: np.ndarray, rng: YellowRange | None = None) -> np.ndarray:
    rng = rng or YellowRange()
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([rng.h_lo, rng.s_lo, rng.v_lo], np.uint8),
        np.array([rng.h_hi, 255, 255], np.uint8),
    )
    # Close over the wire spider and the black wedges without swallowing the rim.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask


def fit_board_ellipse(mask: np.ndarray) -> tuple | None:
    """Fit the double ring's outer edge. Returns an OpenCV rotated-rect ellipse."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # The board's yellow fragments together dominate the frame; anything much
    # smaller than the largest blob is glare or a stray yellow object.
    areas = [cv2.contourArea(c) for c in contours]
    biggest = max(areas)
    if biggest < 500:
        return None
    keep = [c for c, a in zip(contours, areas) if a > biggest * 0.02]

    pts = np.vstack(keep)
    hull = cv2.convexHull(pts)
    if len(hull) < 5:
        return None

    ellipse = cv2.fitEllipse(hull)
    (_, _), (w, h), _ = ellipse
    if min(w, h) < 40 or min(w, h) / max(w, h) < 0.15:
        # Degenerate -- almost certainly not the board.
        return None
    return ellipse


def affine_from_ellipse(ellipse, geom: BoardGeometry = REGULATION) -> np.ndarray:
    """3x3 homography mapping camera pixels -> rectified board pixels.

    Maps the detected ellipse onto a circle of the correct radius. Rotation
    about the board centre is still unresolved at this point.
    """
    (cx, cy), (w, h), angle = ellipse
    a, b = w / 2.0, h / 2.0
    target_r = geom.double_outer * px_per_mm(geom)

    th = math.radians(angle)
    cos_t, sin_t = math.cos(th), math.sin(th)

    translate = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], np.float64)
    # fitEllipse's angle rotates the *width* axis; undo it.
    rotate = np.array([[cos_t, sin_t, 0], [-sin_t, cos_t, 0], [0, 0, 1]], np.float64)
    scale = np.array(
        [[target_r / a, 0, 0], [0, target_r / b, 0], [0, 0, 1]], np.float64
    )
    recentre = np.array(
        [[1, 0, RECT_SIZE / 2.0], [0, 1, RECT_SIZE / 2.0], [0, 0, 1]], np.float64
    )
    return recentre @ scale @ rotate @ translate


# --------------------------------------------------------------------------
# rotation + refinement
# --------------------------------------------------------------------------


def _rotation_matrix(deg: float) -> np.ndarray:
    m = cv2.getRotationMatrix2D((RECT_SIZE / 2.0, RECT_SIZE / 2.0), deg, 1.0)
    return np.vstack([m, [0, 0, 1]])


def _mirror_matrix() -> np.ndarray:
    return np.array([[-1, 0, RECT_SIZE], [0, 1, 0], [0, 0, 1]], np.float64)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float32).ravel()
    bf = b.astype(np.float32).ravel()
    af -= af.mean()
    bf -= bf.mean()
    denom = np.linalg.norm(af) * np.linalg.norm(bf)
    return float(af @ bf / denom) if denom else -1.0


def resolve_rotation(
    mask: np.ndarray, base_h: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """Search the 20 sector rotations (x mirror) for the best alignment.

    Returns (homography, best NCC, margin over the runner-up). The margin is the
    number to watch: the ring pattern alone is 36-degree symmetric, so if the
    numerals aren't registering, ten candidates tie and the margin collapses
    toward zero. A confident lock separates cleanly.

    The board is warped to the rectified frame *once* and then rotated at low
    resolution rather than re-warped per candidate: rotation and mirroring are
    both centred on the board, so they commute with the downscale. Forty 800x800
    perspective warps would take seconds on a Pi 4; forty 200x200 rotations take
    a few milliseconds.
    """
    small = 200
    ref_small = cv2.resize(reference, (small, small), interpolation=cv2.INTER_AREA)
    warped = cv2.warpPerspective(mask, base_h, (RECT_SIZE, RECT_SIZE))
    base_small = cv2.resize(warped, (small, small), interpolation=cv2.INTER_AREA)

    centre = (small / 2.0, small / 2.0)
    scored: list[tuple[float, np.ndarray]] = []
    for mirror in (False, True):
        src = cv2.flip(base_small, 1) if mirror else base_small
        pre = _mirror_matrix() @ base_h if mirror else base_h
        for k in range(len(SECTORS)):
            deg = k * 18.0
            rot = cv2.getRotationMatrix2D(centre, deg, 1.0)
            cand = cv2.warpAffine(src, rot, (small, small))
            scored.append((_ncc(cand, ref_small), _rotation_matrix(deg) @ pre))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    best_score, best_h = scored[0]
    margin = best_score - scored[1][0] if len(scored) > 1 else 0.0
    return best_h, best_score, margin


def refine_homography(
    mask: np.ndarray, h: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    """Sub-pixel perspective refinement via ECC against the reference.

    Runs on blurred masks rather than raw pixels: the synthetic reference has no
    wood grain, wire spider or specular highlights, so matching intensities
    directly would fight the very thing we want it to ignore.

    The result is accepted only if it actually improves the match. ECC's warp
    convention is easy to get backwards (it composes with ``WARP_INVERSE_MAP``,
    so the matrix maps template coordinates to input coordinates), and it can
    also converge to a worse local optimum. Rather than trusting either, the
    refinement is scored and discarded if it didn't help -- a bad calibration
    that silently survives is far more expensive than a slightly coarse one.
    """
    warped = cv2.warpPerspective(mask, h, (RECT_SIZE, RECT_SIZE))
    tmpl = cv2.GaussianBlur(reference, (21, 21), 0).astype(np.float32) / 255.0
    inp = cv2.GaussianBlur(warped, (21, 21), 0).astype(np.float32) / 255.0

    warp = np.eye(3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
    try:
        _, warp = cv2.findTransformECC(
            tmpl, inp, warp, cv2.MOTION_HOMOGRAPHY, criteria, None, 5
        )
    except cv2.error as exc:
        log.warning("ECC refinement did not converge, keeping affine estimate: %s", exc)
        return h

    before = _ncc(warped, reference)
    candidate = np.linalg.inv(warp.astype(np.float64)) @ h
    after = _ncc(cv2.warpPerspective(mask, candidate, (RECT_SIZE, RECT_SIZE)), reference)
    if after <= before:
        log.debug("ECC refinement rejected (%.4f -> %.4f)", before, after)
        return h
    log.debug("ECC refinement accepted (%.4f -> %.4f)", before, after)
    return candidate


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------


@dataclass
class Calibration:
    """Maps camera pixels to board millimetres."""

    h_img2rect: np.ndarray
    geom: BoardGeometry
    score: float  # rotation-match NCC, a rough confidence
    image_size: tuple[int, int]
    margin: float = 0.0  # gap to the runner-up rotation; low means "check the overlay"

    @property
    def rotation_is_confident(self) -> bool:
        return self.margin >= 0.04

    def rotated(self, sectors: int) -> "Calibration":
        """Nudge the orientation by whole sectors.

        The escape hatch for a mis-locked rotation: the overlay draws the sector
        numbers onto the live view, so a wrong lock is visible at a glance and
        fixable with a tap instead of a re-shoot.
        """
        h = _rotation_matrix(sectors * 18.0) @ self.h_img2rect
        return Calibration(h, self.geom, self.score, self.image_size, self.margin)

    def image_to_board(self, x: float, y: float) -> tuple[float, float]:
        pt = np.array([[[float(x), float(y)]]], np.float64)
        u, v = cv2.perspectiveTransform(pt, self.h_img2rect)[0][0]
        ppm = px_per_mm(self.geom)
        return ((u - RECT_SIZE / 2.0) / ppm, (RECT_SIZE / 2.0 - v) / ppm)

    def board_to_image(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        ppm = px_per_mm(self.geom)
        u = RECT_SIZE / 2.0 + x_mm * ppm
        v = RECT_SIZE / 2.0 - y_mm * ppm
        pt = np.array([[[u, v]]], np.float64)
        inv = np.linalg.inv(self.h_img2rect)
        px, py = cv2.perspectiveTransform(pt, inv)[0][0]
        return float(px), float(py)

    def rectify(self, bgr: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(bgr, self.h_img2rect, (RECT_SIZE, RECT_SIZE))

    def to_dict(self) -> dict:
        return {
            "h": self.h_img2rect.tolist(),
            "score": self.score,
            "margin": self.margin,
            "confident": self.rotation_is_confident,
            "image_size": list(self.image_size),
            "geom": self.geom.__dict__,
        }


def auto_calibrate(
    bgr: np.ndarray,
    geom: BoardGeometry = REGULATION,
    yellow: YellowRange | None = None,
    min_score: float = 0.35,
    passes: int = 2,
) -> Calibration | None:
    """Full auto-calibration from a single frame of the empty board.

    Rotation-search and ECC are *interleaved*, not run once each. The affine
    seed from the ellipse cannot represent perspective at all, and a camera at
    55 cm and 45 degrees foreshortens the board by about 1.6:1 near-to-far --
    so the first rectification is visibly wrong and matches the reference
    poorly, even when the board was found perfectly.

    That matters in both directions:

      * scoring the affine estimate and rejecting on it throws away boards that
        would have locked on fine one step later;
      * picking the rotation from a badly-warped image can select the wrong
        sector bin, and ECC will then happily polish a wrong answer.

    So: pick a rotation, fix the geometry, then re-pick the rotation now that
    the geometry is good. The second pass almost always confirms the first; when
    it doesn't, the first was wrong and this is what catches it.

    Returns None if the board could not be located confidently -- callers should
    treat that as "keep using manual entry", never as a silent zero.
    """
    mask = yellow_mask(bgr, yellow)
    ellipse = fit_board_ellipse(mask)
    if ellipse is None:
        log.warning("calibration: could not fit a board ellipse from the yellow mask")
        return None

    reference = render_reference(geom)
    h_est = affine_from_ellipse(ellipse, geom)

    margin = 0.0
    for attempt in range(max(passes, 1)):
        h_est, rot_score, margin = resolve_rotation(mask, h_est, reference)
        h_est = refine_homography(mask, h_est, reference)
        log.debug("calibration pass %d: rotation match %.3f", attempt + 1, rot_score)

    # Gate on the *refined* alignment, which is the thing that actually gets used.
    score = _ncc(cv2.warpPerspective(mask, h_est, (RECT_SIZE, RECT_SIZE)), reference)
    if score < min_score:
        log.warning(
            "calibration: board found but alignment only scored %.2f (need %.2f). "
            "Usually the board is partly out of frame, a dart is still in it, or "
            "the yellow window needs tuning -- see tools/check_calib.py --tune.",
            score, min_score,
        )
        return None

    h, w = bgr.shape[:2]
    calib = Calibration(h_est, geom, score, (w, h), margin)

    if calib.rotation_is_confident:
        log.info("calibration: locked on (match %.2f, margin %.3f)", score, margin)
    else:
        log.warning(
            "calibration: orientation is ambiguous (match %.2f, margin %.3f). The ring "
            "pattern repeats every 2 sectors, so the numerals are what pin it down -- "
            "check the overlay and use Rotate if the numbers are in the wrong place.",
            score, margin,
        )
    return calib


def debug_overlay(bgr: np.ndarray, calib: Calibration) -> np.ndarray:
    """Draw the calibrated ring/sector grid back onto the camera image.

    This is the fastest way to eyeball whether a calibration is actually good,
    and it is what the web UI's camera preview shows.
    """
    out = bgr.copy()
    g = calib.geom
    for radius in (g.outer_bull, g.triple_inner, g.triple_outer, g.double_inner, g.double_outer):
        pts = []
        for deg in range(0, 360, 3):
            th = math.radians(deg)
            pts.append(calib.board_to_image(radius * math.sin(th), radius * math.cos(th)))
        cv2.polylines(out, [np.array(pts, np.int32)], True, (0, 255, 0), 1, cv2.LINE_AA)

    for i, value in enumerate(SECTORS):
        # Sector *boundaries* sit half a sector off the centres.
        bth = math.radians(i * 18.0 + 9.0)
        b_in = calib.board_to_image(g.outer_bull * math.sin(bth), g.outer_bull * math.cos(bth))
        b_out = calib.board_to_image(g.double_outer * math.sin(bth), g.double_outer * math.cos(bth))
        cv2.line(out, tuple(map(int, b_in)), tuple(map(int, b_out)), (0, 200, 0), 1, cv2.LINE_AA)

        th = math.radians(i * 18.0)
        label_pt = calib.board_to_image(
            (g.double_outer + 12) * math.sin(th), (g.double_outer + 12) * math.cos(th)
        )
        cv2.putText(
            out, str(value), tuple(map(int, label_pt)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA,
        )
    return out
