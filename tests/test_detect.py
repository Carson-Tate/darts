"""Dart detection, and specifically which end of the blob is the point.

Tip localisation is the whole ballgame: a dart stands 30-40mm out of the board,
so picking the flight instead of the point puts the score a whole sector out --
and on real throws it put it 280mm out on a 170mm board, reported as a miss.
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from darts.board import REGULATION  # noqa: E402
from darts.vision.detect import (  # noqa: E402
    _merge_collinear,
    DetectorConfig,
    _tip_from_points,
    find_darts,
    foreground_mask,
)
from darts.vision.pipeline import PipelineConfig, VisionPipeline  # noqa: E402


def dart_polygon(x0, y0, x1, y1, point_w=1.5, flight_w=9.0):
    """A tapered quad: narrow at (x0,y0), flared at (x1,y1)."""
    axis = np.array([x1 - x0, y1 - y0], float)
    axis /= np.linalg.norm(axis)
    perp = np.array([-axis[1], axis[0]])
    return np.array([
        [x0, y0] + perp * point_w,
        [x1, y1] + perp * flight_w,
        [x1, y1] - perp * flight_w,
        [x0, y0] - perp * point_w,
    ], np.int32)


class TestTipFromPoints:
    def test_returns_the_narrow_end_first(self):
        poly = dart_polygon(100, 100, 200, 140)
        tip, other, elong, _ = _tip_from_points(poly.reshape(-1, 2), DetectorConfig())
        assert np.hypot(tip[0] - 100, tip[1] - 100) < 20, "narrow end should be the tip"
        assert np.hypot(other[0] - 200, other[1] - 140) < 20
        assert elong > 2

    def test_both_ends_are_distinct_and_on_the_axis(self):
        poly = dart_polygon(300, 200, 380, 260)
        tip, other, _, _ = _tip_from_points(poly.reshape(-1, 2), DetectorConfig())
        assert np.hypot(tip[0] - other[0], tip[1] - other[1]) > 50


class TestBlobFiltering:
    def _mask_with(self, poly):
        bg = np.zeros((480, 640), np.uint8)
        img = bg.copy()
        cv2.fillPoly(img, [poly], 255)
        return img, bg

    def test_finds_a_dart_shaped_blob(self):
        img, bg = self._mask_with(dart_polygon(200, 200, 300, 250))
        blobs = find_darts(img, bg, DetectorConfig(min_area=100))
        assert len(blobs) == 1
        assert blobs[0].other_end != blobs[0].tip

    def test_rejects_an_arm_sized_blob(self):
        """The one arm that got scored as a dart had an area of 16627 px."""
        img, bg = self._mask_with(
            np.array([[100, 100], [500, 120], [500, 400], [100, 380]], np.int32)
        )
        assert find_darts(img, bg, DetectorConfig()) == []

    def test_rejects_a_round_blob(self):
        img, bg = self._mask_with(
            cv2.ellipse2Poly((300, 240), (30, 28), 0, 0, 360, 10)
        )
        assert find_darts(img, bg, DetectorConfig(min_area=100)) == []


class FakeCalib:
    """Maps pixels to board mm at 1mm per px, centred on (0,0)."""

    def image_to_board(self, x, y):
        return float(x), float(y)


class TestTipChoice:
    """The rule that fixed the false misses: the point is in the board."""

    def _pipeline(self):
        p = VisionPipeline.__new__(VisionPipeline)
        p.cfg = PipelineConfig(geom=REGULATION)
        return p

    def _blob(self, tip, other):
        from darts.vision.detect import Blob
        return Blob(tip, other, (0, 0), 500.0, 5.0, 0.0)

    def test_takes_the_on_board_end_when_the_taper_cue_picked_the_other(self):
        # Exactly the measured failure: 284mm off the board vs 67mm on it.
        blob = self._blob(tip=(284.0, 0.0), other=(67.0, 0.0))
        assert self._pipeline()._pick_tip(blob, FakeCalib()) == (67.0, 0.0)

    def test_keeps_the_taper_pick_when_it_is_already_on_the_board(self):
        blob = self._blob(tip=(67.0, 0.0), other=(284.0, 0.0))
        assert self._pipeline()._pick_tip(blob, FakeCalib()) == (67.0, 0.0)

    def test_keeps_the_taper_pick_when_both_ends_are_on_the_board(self):
        """A dart lying nearly flat to the face: the taper cue is all there is."""
        blob = self._blob(tip=(147.0, 0.0), other=(35.0, 0.0))
        assert self._pipeline()._pick_tip(blob, FakeCalib()) == (147.0, 0.0)

    def test_keeps_the_taper_pick_when_neither_end_is_on_the_board(self):
        """Both off means it is not a dart; leave it for the off-board reject."""
        blob = self._blob(tip=(276.0, 0.0), other=(186.0, 0.0))
        assert self._pipeline()._pick_tip(blob, FakeCalib()) == (276.0, 0.0)


class TestFragmentMerging:
    """A dark dart over a black sector barely differs from it, so one dart
    arrives as several disconnected pieces. Measured on a real throw: six
    fragments over 230px, and taking the largest scored a treble 13 for a dart
    in the 3."""

    def _pieces(self, *boxes):
        return [
            np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
            for x0, y0, x1, y1 in boxes
        ]

    def test_joins_pieces_along_one_line(self):
        # Three chunks strung along a shallow diagonal, as a dart's fragments are.
        pieces = self._pieces((100, 200, 130, 210), (200, 180, 230, 190), (300, 160, 330, 170))
        merged = _merge_collinear(pieces, DetectorConfig())
        assert len(merged) == 1, "collinear fragments should become one dart"
        assert len(merged[0]) == 12

    def test_keeps_pieces_off_the_line_separate(self):
        """Wire lines and shadows sit near a dart without being part of it."""
        pieces = self._pieces((100, 200, 130, 210), (200, 180, 230, 190), (200, 400, 230, 410))
        merged = _merge_collinear(pieces, DetectorConfig())
        assert len(merged) == 2

    def test_does_not_join_across_more_than_a_dart_length(self):
        far = DetectorConfig().max_dart_span_px * 3
        pieces = self._pieces((100, 200, 130, 210), (100 + far, 200, 130 + far, 210))
        assert len(_merge_collinear(pieces, DetectorConfig())) == 2

    def test_a_single_piece_survives_intact(self):
        pieces = self._pieces((100, 200, 130, 210))
        merged = _merge_collinear(pieces, DetectorConfig())
        assert len(merged) == 1 and len(merged[0]) == 4

    def test_merged_dart_is_more_elongated_than_its_fragments(self):
        """The point of merging: elongation is what identifies a dart, and a
        single fragment of one is not elongated enough to look like a dart."""
        cfg = DetectorConfig()
        pieces = self._pieces((100, 200, 130, 212), (200, 180, 230, 192), (300, 160, 330, 172))
        one = _tip_from_points(pieces[0], cfg)[2]
        whole = _tip_from_points(np.vstack(_merge_collinear(pieces, cfg)), cfg)[2]
        assert whole > one * 2
