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
from functools import lru_cache

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


def render_reference(geom: BoardGeometry = REGULATION, numerals: bool = True) -> np.ndarray:
    """Render a canonical rectified board as a binary 'is yellow' mask.

    The numerals matter more than they look. Without them the pattern repeats
    every *two* sectors -- alternating singles with the ring colours inverted is
    36-degree symmetric -- so ten of the twenty candidate rotations score
    identically and the search picks one at random. That failure is silent and
    scores T20 as T18.

    The numbers are the only thing on the face that breaks the symmetry, which
    makes ``geom.number_radius`` load-bearing: stamp them at the wrong radius
    and the comparison sees nothing but symmetric ring pattern. Measured on the
    real board, the 36-degree self-similarity of the number band sits at +0.41
    where the numerals actually are and at +0.95 twenty millimetres further out.
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

    if numerals:
        number_r = geom.number_radius
        # Read the band colour out of the rendered board rather than deriving it
        # from sector parity. The parity flips between the single bands and the
        # rings, so a hard-coded rule is only correct for one choice of
        # number_radius and silently inverts every numeral if that radius moves
        # across a ring boundary. Sample first, stamp after, so a numeral can't
        # be read as its own background.
        inks = []
        for i in range(len(SECTORS)):
            th = math.radians(i * 18.0)
            px = int(round(size / 2.0 + number_r * math.sin(th) * ppm))
            py = int(round(size / 2.0 - number_r * math.cos(th) * ppm))
            band_is_yellow = 0 <= py < size and 0 <= px < size and mask[py, px] > 127
            inks.append(0 if band_is_yellow else 255)
        for i, value in enumerate(SECTORS):
            _stamp_number(mask, value, i * 18.0, number_r, ppm, ink=inks[i])
    return mask


@lru_cache(maxsize=8)
def numeral_region(
    geom: BoardGeometry = REGULATION, size: int = 400, dilate: int = 3
) -> np.ndarray:
    """Boolean mask of just the numeral pixels -- the pure asymmetric signal.

    Restricting the comparison to the number-band *annulus* is not enough. That
    annulus is mostly double ring, whose alternating colours are identical under
    a 36-degree turn, so the correlation there is dominated by a term that
    carries no orientation information at all. Measured on a perfect
    self-comparison the resulting margin was under 0.04 -- signal buried in a
    much larger pile of noise.

    Differencing the board against a numeral-free render isolates exactly the
    pixels that distinguish one orientation from another, and drops every pixel
    that cannot. A little dilation keeps it tolerant of small misalignment and
    of the real board's font differing from this rendered one.
    """
    marked = render_reference(geom, numerals=True)
    plain = render_reference(geom, numerals=False)
    diff = cv2.absdiff(marked, plain)
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate * 2 + 1,) * 2)
        diff = cv2.dilate(diff, k)
    small = cv2.resize(diff, (size, size), interpolation=cv2.INTER_AREA)
    return small > 0


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


def yellow_mask(
    bgr: np.ndarray, rng: YellowRange | None = None, clean: bool = True
) -> np.ndarray:
    """Threshold the board's yellow paint.

    Two callers want opposite things from this, so it has two modes.

    ``clean=True`` closes hard over the wire spider and speckle. That is right
    for fitting the outer ellipse, where only the rim silhouette matters and
    noise is pure cost.

    ``clean=False`` skips the closing. That is required for the rotation search,
    because the closing *destroys the printed numerals* -- a 5x5 close over two
    iterations bridges roughly 9 px, and the numeral strokes are about 3 px wide
    in a 720p frame. Erasing them erases the only thing that distinguishes this
    board from itself rotated 36 degrees.
    """
    rng = rng or YellowRange()
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([rng.h_lo, rng.s_lo, rng.v_lo], np.uint8),
        np.array([rng.h_hi, 255, 255], np.uint8),
    )
    if not clean:
        # Just knock out single-pixel speckle; leave the numerals intact.
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

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


def _ncc(a: np.ndarray, b: np.ndarray, sel: np.ndarray | None = None) -> float:
    """Normalised cross-correlation, optionally over a subset of pixels."""
    af = a.astype(np.float32)
    bf = b.astype(np.float32)
    if sel is not None:
        af, bf = af[sel], bf[sel]
    af = af.ravel() - af.mean()
    bf = bf.ravel() - bf.mean()
    denom = np.linalg.norm(af) * np.linalg.norm(bf)
    return float(af @ bf / denom) if denom else -1.0


SEARCH_SIZE = 400  # resolution of the rotation search


@dataclass(frozen=True)
class RotationCandidate:
    """One of the forty orientations (20 sectors x mirrored) with its scores.

    Named rather than a bare tuple because the diagnostics report these: when a
    lock is wrong you want to see *which* orientation won and by how little, and
    "sectors=4, mirror=False" is the difference between a two-sector symmetry
    confusion and something being properly broken.
    """

    whole: float  # NCC over the whole board -- symmetric candidates tie here
    numerals: float  # NCC over numeral pixels only -- the tie-breaker
    h: np.ndarray
    sectors: int  # rotation in whole sectors, 0..19
    mirror: bool


def _rotation_candidates(
    mask: np.ndarray,
    base_h: np.ndarray,
    reference: np.ndarray,
    geom: BoardGeometry,
) -> list[RotationCandidate]:
    """Score all 20 sector rotations x mirror as (whole NCC, numeral NCC, H).

    The board is warped to the rectified frame once and then rotated, rather
    than re-warped per candidate: rotation and mirroring are both centred on the
    board, so they commute with the resize. Forty 800x800 perspective warps
    would take seconds on a Pi 4; forty 400x400 rotations take a fraction of one.
    """
    size = SEARCH_SIZE
    ref_small = cv2.resize(reference, (size, size), interpolation=cv2.INTER_AREA)
    warped = cv2.warpPerspective(mask, base_h, (RECT_SIZE, RECT_SIZE))
    base_small = cv2.resize(warped, (size, size), interpolation=cv2.INTER_AREA)
    sel = numeral_region(geom, size)

    # The numeral term is compared *blurred*, making it an ink-density
    # comparison rather than exact glyph matching. That matters on the real
    # board, whose printed font will not match this rendered one -- but ten of
    # the twenty numbers having two digits still reads clearly as density.
    ref_blur = cv2.GaussianBlur(ref_small, (9, 9), 0)

    centre = (size / 2.0, size / 2.0)
    out: list[RotationCandidate] = []
    for mirror in (False, True):
        src = cv2.flip(base_small, 1) if mirror else base_small
        pre = _mirror_matrix() @ base_h if mirror else base_h
        for k in range(len(SECTORS)):
            deg = k * 18.0
            rot = cv2.getRotationMatrix2D(centre, deg, 1.0)
            cand = cv2.warpAffine(src, rot, (size, size))
            out.append(RotationCandidate(
                whole=_ncc(cand, ref_small),
                numerals=_ncc(cv2.GaussianBlur(cand, (9, 9), 0), ref_blur, sel),
                h=_rotation_matrix(deg) @ pre,
                sectors=k,
                mirror=mirror,
            ))
    return out


def coarse_rotation(
    mask: np.ndarray,
    base_h: np.ndarray,
    reference: np.ndarray,
    geom: BoardGeometry = REGULATION,
) -> tuple[np.ndarray, float]:
    """Pick an orientation using the ring pattern alone, ignoring the numerals.

    Run *before* ECC, where the affine seed's geometry is still wrong. The
    numerals cannot be read from a badly-rectified board -- they land in the
    wrong places, so correlating against them over a small pixel set produces
    confident nonsense. The ring pattern is robust to that distortion.

    This deliberately cannot tell 0 from 36 degrees, and does not try. It only
    has to land in one of the ten symmetry-equivalent bins so that ECC has
    something sane to refine; resolve_rotation() sorts out which one afterwards.

    Each candidate gets a cheap 1/8-scale ECC *before* being scored. Ranking the
    raw candidates does not work: off the affine seed all forty score within
    0.003 of each other, because a rectification that wrong correlates equally
    badly with every orientation. Only after each one has been allowed to settle
    do the correct bins separate.
    """
    scored: list[tuple[float, np.ndarray]] = []
    for cand in _rotation_candidates(mask, base_h, reference, geom):
        settled = _ecc_at_scale(mask, cand.h, reference, scale=0.125, iterations=40)
        scored.append((alignment_score(mask, settled, reference), settled))

    score, best = max(scored, key=lambda t: t[0])
    return best, score


def resolve_rotation(
    mask: np.ndarray,
    base_h: np.ndarray,
    reference: np.ndarray,
    geom: BoardGeometry = REGULATION,
    ring_tolerance: float = 0.05,
) -> tuple[np.ndarray, float, float, list["RotationCandidate"]]:
    """Decide the orientation the ring pattern cannot: shortlist, then numerals.

    Returns (homography, whole-board NCC, numeral margin over the runner-up, and
    the shortlist of symmetric alternatives in numeral-score order).

    Run *after* ECC, once the geometry is trustworthy. Two stages:

      1. keep every candidate whose whole-board fit is within `ring_tolerance`
         of the best -- that is the set of genuinely symmetric alternatives,
         and it throws out anything grossly misaligned;
      2. among those, decide on the numeral pixels alone.

    Stage 2 must *not* include the whole-board term. Once ECC has fitted a
    particular orientation, its 8 degrees of freedom have absorbed a little
    skew that flatters that specific hypothesis -- enough to outvote the numeral
    evidence and lock in whichever bin the coarse pass happened to pick. Among
    genuinely symmetric candidates the ring term carries no information anyway,
    only bias.
    """
    cands = _rotation_candidates(mask, base_h, reference, geom)
    best_whole = max(c.whole for c in cands)
    eligible = sorted(
        (c for c in cands if c.whole >= best_whole - ring_tolerance),
        key=lambda c: c.numerals,
        reverse=True,
    )
    best = eligible[0]
    margin = best.numerals - eligible[1].numerals if len(eligible) > 1 else best.numerals
    return best.h, best.whole, margin, eligible


def orient_to_template(
    mask: np.ndarray,
    base_h: np.ndarray,
    template: np.ndarray,
    geom: BoardGeometry = REGULATION,
) -> tuple[np.ndarray, float, float]:
    """Pick the orientation that matches a board image the user already confirmed.

    This exists because matching a real board against a *synthetic* reference is
    what makes the orientation a coin flip. The rendered numerals differ from the
    printed ones in font, weight, stroke width and exact radius, so the only
    signal that breaks the 36-degree symmetry is weak enough to lose to noise --
    measured margins of 0.002 to 0.026 on the real board, against a 0.04
    threshold, with the winner changing between consecutive frames.

    A template is this board, in this light, at the orientation its owner
    confirmed. Comparing against it is like-for-like, so the whole-board score
    discriminates on its own and the numerals stop being load-bearing.

    Returns (homography, match, margin over the runner-up).
    """
    cands = _rotation_candidates(mask, base_h, template, geom)
    ranked = sorted(cands, key=lambda c: c.whole, reverse=True)
    best = ranked[0]
    margin = best.whole - ranked[1].whole if len(ranked) > 1 else best.whole
    return best.h, best.whole, margin


def alignment_score(mask: np.ndarray, h: np.ndarray, reference: np.ndarray) -> float:
    """How well `mask` rectified by `h` matches the reference board."""
    return _ncc(cv2.warpPerspective(mask, h, (RECT_SIZE, RECT_SIZE)), reference)


def _ecc_at_scale(
    mask: np.ndarray,
    h: np.ndarray,
    reference: np.ndarray,
    scale: float,
    iterations: int = 100,
    blur: int = 9,
) -> np.ndarray:
    """One ECC pass at a fraction of full resolution. Returns h unchanged on failure.

    Working at reduced scale with a fixed blur kernel is what gives the
    coarse-to-fine behaviour: a 9x9 blur on a 100px board is enormous relative
    to the image and tolerates gross misalignment, while the same kernel at full
    resolution is small and preserves precision.
    """
    size = max(int(RECT_SIZE * scale), 32)
    s = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], np.float64)

    ref_s = cv2.resize(reference, (size, size), interpolation=cv2.INTER_AREA)
    warped = cv2.warpPerspective(mask, s @ h, (size, size))
    tmpl = cv2.GaussianBlur(ref_s, (blur, blur), 0).astype(np.float32) / 255.0
    inp = cv2.GaussianBlur(warped, (blur, blur), 0).astype(np.float32) / 255.0

    warp = np.eye(3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, 1e-6)
    try:
        _, warp = cv2.findTransformECC(
            tmpl, inp, warp, cv2.MOTION_HOMOGRAPHY, criteria, None, 5
        )
    except cv2.error:
        return h
    # ECC composes with WARP_INVERSE_MAP, so its matrix maps template
    # coordinates to input coordinates -- invert before composing. Conjugating
    # by `s` lifts the correction back to full rectified resolution.
    return np.linalg.inv(s) @ np.linalg.inv(warp.astype(np.float64)) @ s @ h


def refine_homography(
    mask: np.ndarray,
    h: np.ndarray,
    reference: np.ndarray,
    scales: tuple[float, ...] = (0.125, 0.25, 0.5, 1.0),
) -> np.ndarray:
    """Coarse-to-fine ECC refinement.

    A single full-resolution pass is not enough, and measuring that was what
    finally explained the bad calibrations. The affine seed from the ellipse
    cannot represent perspective, so under a realistic 1.6:1 foreshortening the
    ellipse centre sits tens of pixels from the true board centre. A 21x21
    Gaussian gives ECC a convergence basin of about 3.5 px. It could not
    possibly reach, and it didn't -- alignment crawled from 0.30 to 0.37 against
    a ground truth of 0.98, leaving every rotation candidate equally wrong and
    the orientation search with nothing to work from.

    Starting at 1/8 scale shrinks that same displacement to a few pixels, well
    inside the basin, and each finer level tightens what the last one found.

    Every level is accepted only if it improves the score. ECC can converge to a
    worse local optimum, and a bad calibration that survives silently is far
    more expensive than a coarse one that is honest about it.
    """
    best = h
    best_score = alignment_score(mask, best, reference)
    for scale in scales:
        candidate = _ecc_at_scale(mask, best, reference, scale)
        score = alignment_score(mask, candidate, reference)
        if score > best_score:
            log.debug("ECC @ %.3fx: %.4f -> %.4f", scale, best_score, score)
            best, best_score = candidate, score
        else:
            log.debug("ECC @ %.3fx rejected (%.4f -> %.4f)", scale, best_score, score)
    return best


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
    # The symmetric alternatives that were in the running, best first, as
    # (sectors, mirror, numeral score). Kept for diagnostics: when a lock looks
    # wrong, this says whether it was a close race between two symmetry-related
    # orientations or whether nothing fitted at all.
    shortlist: tuple[tuple[int, bool, float], ...] = ()

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
        return Calibration(h, self.geom, self.score, self.image_size, self.margin, self.shortlist)

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

    def _board_outline(self, reach: float) -> np.ndarray:
        """The board's rim projected into camera pixels, as an Nx2 array.

        A circle in the board plane becomes an ellipse in the image, so this
        projects points rather than approximating with a circle in image space.
        """
        r = self.geom.double_outer * reach
        ang = np.linspace(0.0, 2.0 * np.pi, 180, endpoint=False)
        ppm = px_per_mm(self.geom)
        rect = np.stack(
            [
                RECT_SIZE / 2.0 + r * np.cos(ang) * ppm,
                RECT_SIZE / 2.0 - r * np.sin(ang) * ppm,
            ],
            axis=1,
        ).astype(np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(
            rect, np.linalg.inv(self.h_img2rect)
        ).reshape(-1, 2)

    def board_bounds(
        self, shape: tuple[int, ...], reach: float = 1.25
    ) -> tuple[int, int, int, int]:
        """Bounding box of the board in camera pixels, clamped to the frame.

        For cropping the preview. The overhead camera devotes about three
        quarters of its frame to a doorway, a fridge and a bin; on a phone tile
        that leaves the board too small to tell whether the overlay is on the
        rings, which is the only thing the preview is for.
        """
        h, w = int(shape[0]), int(shape[1])
        pts = self._board_outline(reach)
        x0 = int(max(np.floor(pts[:, 0].min()), 0))
        y0 = int(max(np.floor(pts[:, 1].min()), 0))
        x1 = int(min(np.ceil(pts[:, 0].max()), w))
        y1 = int(min(np.ceil(pts[:, 1].max()), h))
        if x1 - x0 < 16 or y1 - y0 < 16:  # degenerate fit; show the whole frame
            return 0, 0, w, h
        return x0, y0, x1, y1

    def board_mask(self, shape: tuple[int, ...], reach: float = 1.6) -> np.ndarray:
        """Filled mask of the board's face, in camera pixels, out to `reach`
        times the double-ring radius.

        Built by projecting a circle of board-plane points through the inverse
        homography, so it is an ellipse in the image and follows the real
        perspective rather than approximating it with a circle.

        `reach` is deliberately wider than the 1.35 at which _measure rejects a
        tip: the taper cue needs the *whole* dart silhouette to tell the point
        from the flight, so a mask that hugged the scoring area would clip the
        flight off darts in the doubles and make them ambiguous.
        """
        h, w = int(shape[0]), int(shape[1])
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [np.rint(self._board_outline(reach)).astype(np.int32)], 255)
        return mask

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
) -> Calibration | None:
    """Full auto-calibration from a single frame of the empty board.

    Four stages, each using the signal that is actually trustworthy at that
    point:

      1. **ellipse** -- locate the board from its painted rim.
      2. **coarse rotation**, ring pattern only. The affine seed cannot
         represent perspective, and a camera at 55 cm and 45 degrees
         foreshortens the board about 1.6:1 near-to-far, so the rectification
         here is visibly wrong. Numerals are unreadable off a board in that
         state. This stage only has to land in one of the ten
         symmetry-equivalent bins.
      3. **ECC** -- fix the geometry. It can do this from any of those ten bins,
         because the ring pattern fits equally well in all of them.
      4. **symmetry resolution**, numerals only, now that the board is properly
         rectified. Rotation in the rectified frame is metric and exact, so
         correcting the orientation here costs nothing geometrically.

    Ordering these wrong is what made the lock a coin flip: judging orientation
    before the geometry was fixed, then letting ECC cement whichever bin got
    picked.

    Returns None if the board could not be located confidently -- callers should
    treat that as "keep using manual entry", never as a silent zero.
    """
    # Two masks, because the ellipse fit and the rotation search want opposite
    # things -- see yellow_mask(). Using the cleaned mask for both is what made
    # the rotation lock a coin flip: the closing erased the numerals.
    mask_clean = yellow_mask(bgr, yellow, clean=True)
    mask_fine = yellow_mask(bgr, yellow, clean=False)

    ellipse = fit_board_ellipse(mask_clean)
    if ellipse is None:
        log.warning("calibration: could not fit a board ellipse from the yellow mask")
        return None

    reference = render_reference(geom)
    h_est = affine_from_ellipse(ellipse, geom)

    # Ring pattern first, geometry second, numerals last. The order matters:
    # numerals are unreadable off a badly-rectified board, and ECC needs an
    # approximately-right orientation before it can fix the geometry. Because
    # the rectified frame is metric, the final rotation correction is exact --
    # no further warping is needed to apply it.
    h_est, coarse_score = coarse_rotation(mask_fine, h_est, reference, geom)
    h_est = refine_homography(mask_fine, h_est, reference)
    h_est, rot_score, margin, shortlist = resolve_rotation(mask_fine, h_est, reference, geom)
    h_est = refine_homography(mask_fine, h_est, reference)
    log.debug(
        "calibration: coarse %.3f -> refined %.3f, numeral margin %.4f",
        coarse_score, rot_score, margin,
    )

    # Gate on the *refined* alignment, which is the thing that actually gets used.
    score = _ncc(cv2.warpPerspective(mask_fine, h_est, (RECT_SIZE, RECT_SIZE)), reference)
    if score < min_score:
        log.warning(
            "calibration: board found but alignment only scored %.2f (need %.2f). "
            "Usually the board is partly out of frame, a dart is still in it, or "
            "the yellow window needs tuning -- see tools/check_calib.py --tune.",
            score, min_score,
        )
        return None

    h, w = bgr.shape[:2]
    calib = Calibration(h_est, geom, score, (w, h), margin,
                        tuple((c.sectors, c.mirror, round(c.numerals, 4)) for c in shortlist[:6]))

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
