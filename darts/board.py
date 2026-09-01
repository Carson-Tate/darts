"""Dartboard geometry and scoring.

Coordinate system: millimetres, origin at the bull centre, +x right, +y UP.
Angles are measured clockwise from +y (straight up, the centre of the 20).

Radii default to regulation. This board is regulation-proportioned but has the
numbers printed inside the double ring instead of on a separate number ring,
so the scoring radii are still standard -- but calibration is allowed to
refine them from the measured radial profile (see vision/calibrate.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

# Sector values, clockwise starting at the 20 (straight up).
SECTORS = (20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5)

SECTOR_DEGREES = 360.0 / len(SECTORS)  # 18.0


@dataclass(frozen=True)
class BoardGeometry:
    """Ring radii in mm, measured from the bull centre."""

    inner_bull: float = 6.35
    outer_bull: float = 15.9
    triple_inner: float = 99.0
    triple_outer: float = 107.0
    double_inner: float = 162.0
    double_outer: float = 170.0

    # Radius at which the sector numbers are printed. Only calibration cares:
    # the numerals are the sole thing breaking the board's 36-degree rotational
    # symmetry, so looking for them at the wrong radius means looking at pure
    # symmetric ring pattern and locking the orientation on a coin flip.
    #
    # A regulation board carries its numbers on a separate ring *outside* the
    # double, which is why the obvious default is the middle of the double ring.
    # A board without that ring prints them inside the outer single band
    # instead. Measured on this one at 128-149mm, centred on 140.
    number_radius: float = 140.0

    def scaled_to(self, double_outer_mm: float) -> "BoardGeometry":
        """Uniformly rescale so the double ring's outer edge lands on the given radius."""
        k = double_outer_mm / self.double_outer
        return replace(
            self,
            inner_bull=self.inner_bull * k,
            outer_bull=self.outer_bull * k,
            triple_inner=self.triple_inner * k,
            triple_outer=self.triple_outer * k,
            double_inner=self.double_inner * k,
            double_outer=double_outer_mm,
            number_radius=self.number_radius * k,
        )


REGULATION = BoardGeometry()


@dataclass(frozen=True)
class Hit:
    """A resolved dart location."""

    sector: int  # 1..20, or 25 for the bull. 0 for a miss.
    ring: str  # "miss" | "single" | "double" | "triple" | "outer_bull" | "inner_bull"
    points: int
    radius_mm: float
    angle_deg: float  # clockwise from top

    @property
    def is_double(self) -> bool:
        """True for the double ring and for the inner bull (counts as D25 for check-outs)."""
        return self.ring in ("double", "inner_bull")

    @property
    def label(self) -> str:
        if self.ring == "miss":
            return "MISS"
        if self.ring == "inner_bull":
            return "BULL"
        if self.ring == "outer_bull":
            return "25"
        prefix = {"single": "S", "double": "D", "triple": "T"}[self.ring]
        return f"{prefix}{self.sector}"

    def spoken(self) -> str:
        """Phrase key for the audio callouts. Must match tools/render_audio.py."""
        if self.ring == "miss":
            return "miss"
        if self.ring == "inner_bull":
            return "bullseye"
        if self.ring == "outer_bull":
            return "outer_bull"
        return f"{self.ring}_{self.sector}"


def sector_at_angle(angle_deg: float) -> int:
    """Sector number for an angle measured clockwise from straight up."""
    # Each sector is centred on its nominal angle, so shift by half a sector
    # before bucketing.
    shifted = (angle_deg + SECTOR_DEGREES / 2.0) % 360.0
    return SECTORS[int(shifted // SECTOR_DEGREES)]


def angle_of_sector(sector: int) -> float:
    """Centre angle (clockwise from top) of a sector number. Inverse of sector_at_angle."""
    return SECTORS.index(sector) * SECTOR_DEGREES


def score_at(x_mm: float, y_mm: float, geom: BoardGeometry = REGULATION) -> Hit:
    """Resolve a board-plane coordinate to a scoring hit."""
    r = math.hypot(x_mm, y_mm)
    # atan2(x, y) gives the angle from +y toward +x, i.e. clockwise from the top.
    angle = math.degrees(math.atan2(x_mm, y_mm)) % 360.0

    if r <= geom.inner_bull:
        return Hit(25, "inner_bull", 50, r, angle)
    if r <= geom.outer_bull:
        return Hit(25, "outer_bull", 25, r, angle)
    if r > geom.double_outer:
        return Hit(0, "miss", 0, r, angle)

    sector = sector_at_angle(angle)
    if geom.triple_inner < r <= geom.triple_outer:
        return Hit(sector, "triple", sector * 3, r, angle)
    if r > geom.double_inner:
        return Hit(sector, "double", sector * 2, r, angle)
    return Hit(sector, "single", sector, r, angle)


def hit_from_label(label: str, geom: BoardGeometry = REGULATION) -> Hit:
    """Build a Hit from a label like "T20", "D16", "S5", "BULL", "25", "MISS".

    Used by the manual-entry UI and by the correction flow, so a corrected dart
    carries the same shape as a detected one. Radius/angle are set to the centre
    of the named region.
    """
    label = label.strip().upper()
    if not label:
        raise ValueError("empty dart label")
    if label in ("MISS", "0"):
        return Hit(0, "miss", 0, geom.double_outer + 10.0, 0.0)
    if label in ("BULL", "DB", "D25", "50"):
        return Hit(25, "inner_bull", 50, 0.0, 0.0)
    if label in ("25", "SB", "S25", "OUTER_BULL"):
        return Hit(25, "outer_bull", 25, (geom.inner_bull + geom.outer_bull) / 2, 0.0)

    ring_code, rest = label[0], label[1:]
    ring = {"S": "single", "D": "double", "T": "triple"}.get(ring_code)
    if ring is None or not rest.isdigit():
        raise ValueError(f"unrecognised dart label: {label!r}")
    sector = int(rest)
    if sector not in SECTORS:
        raise ValueError(f"not a valid sector: {sector}")

    if ring == "triple":
        r = (geom.triple_inner + geom.triple_outer) / 2
        pts = sector * 3
    elif ring == "double":
        r = (geom.double_inner + geom.double_outer) / 2
        pts = sector * 2
    else:
        # Outer single band -- the bigger of the two single regions.
        r = (geom.triple_outer + geom.double_inner) / 2
        pts = sector
    return Hit(sector, ring, pts, r, angle_of_sector(sector))
