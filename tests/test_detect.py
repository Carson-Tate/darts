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
    change_mass,
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
        tip, other, elong, _, _ = _tip_from_points(poly.reshape(-1, 2), DetectorConfig())
        assert np.hypot(tip[0] - 100, tip[1] - 100) < 20, "narrow end should be the tip"
        assert np.hypot(other[0] - 200, other[1] - 140) < 20
        assert elong > 2

    def test_both_ends_are_distinct_and_on_the_axis(self):
        poly = dart_polygon(300, 200, 380, 260)
        tip, other, _, _, _ = _tip_from_points(poly.reshape(-1, 2), DetectorConfig())
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

    def test_does_not_flip_for_a_tip_just_past_the_wire(self):
        """The measured misread: 172mm against a 170mm board became S18.

        A dart in the double reads a whisker outside a board whose calibration
        is a whisker small. Flipping on that does not nudge the score to the
        neighbouring sector -- it moves it the length of the dart, to the far
        side of the board, and this one landed next to the bull for a dart in
        the 12.
        """
        blob = self._blob(tip=(172.0, 0.0), other=(17.0, 0.0))
        assert self._pipeline()._pick_tip(blob, FakeCalib()) == (172.0, 0.0)

    def test_still_flips_for_a_tip_a_whole_dart_off_the_board(self):
        """The genuine flights in the same session read 261mm and 284mm."""
        blob = self._blob(tip=(261.0, 0.0), other=(63.0, 0.0))
        assert self._pipeline()._pick_tip(blob, FakeCalib()) == (63.0, 0.0)

    def test_the_slack_applies_only_to_the_end_being_rejected(self):
        """Both ends off the board must stay a no-flip.

        Slackening both sides of the test at once turns "one end is on the
        board" into "one end is less far off", which would score a dart that
        missed entirely.
        """
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


class TestBackgroundQuiet:
    """Refusing to baseline while something is moving in shot.

    Baselining a person into the background is unrecoverable on its own terms:
    every later frame then differs from it by roughly a whole person, the mass
    never falls back under the quiet threshold, and the pipeline sits in the
    hand state ignoring every dart. Reported in play as "it stops counting
    after I walk up to the board".
    """

    def _model(self, frames):
        from darts.vision.detect import BackgroundModel

        bg = BackgroundModel(frames=5)
        for f in frames_of(frames):
            bg.add(f)
        return bg

    def test_a_still_scene_commits(self):
        bg = self._model("still")
        assert bg.commit(DetectorConfig(), quiet_px=500) is True
        assert bg.ready

    def test_a_moving_scene_does_not(self):
        bg = self._model("moving")
        assert bg.commit(DetectorConfig(), quiet_px=500) is False
        assert not bg.ready

    def test_a_moving_scene_still_commits_when_quiet_is_waived(self):
        """A busy room must not mean no scoreboard at all."""
        bg = self._model("moving")
        assert bg.commit(DetectorConfig(), quiet_px=0) is True

    def test_a_rejected_commit_slides_the_window(self):
        """Otherwise it deadlocks on a buffer that can never go quiet."""
        bg = self._model("moving")
        before = len(bg._buf)
        bg.commit(DetectorConfig(), quiet_px=500)
        assert len(bg._buf) == before - 1


def frames_of(kind):
    """Five 200x200 frames, either identical or with a big moving block."""
    out = []
    for i in range(5):
        f = np.zeros((200, 200), np.uint8)
        if kind == "moving":
            f[20:120, 10 + i * 12: 110 + i * 12] = 255
        out.append(f)
    return out


class TestWireArtifacts:
    """The board's own wires are not darts.

    A phantom dart was scored off an empty board while nobody was throwing.
    Left-low is on auto-exposure; the room dimmed, the exposure shifted, and the
    high-contrast radial wires flipped threshold and appeared as foreground. The
    detector took one and scored an S13.

    Measured, phantom against real darts thrown the same evening:
        phantom      area  294   elongation 40.4   about  2.7px wide
        real darts   area 1014-1197  elongation 5.9-7.9  about 12.4px wide

    Elongation cannot separate them, and not because the margin is thin -- a
    wire is *more* elongated than a dart, so a minimum-only test waves it
    through as the most dart-like thing in frame.
    """

    def _line(self, x0, y0, x1, y1, thickness):
        img = np.zeros((480, 640), np.uint8)
        cv2.line(img, (x0, y0), (x1, y1), 255, thickness)
        return img, np.zeros((480, 640), np.uint8)

    def test_a_hairline_along_a_wire_is_rejected(self):
        img, bg = self._line(560, 358, 760, 358, 3)
        assert find_darts(img, bg, DetectorConfig()) == []

    def test_a_dart_of_real_thickness_is_kept(self):
        img, bg = self._line(560, 358, 700, 358, 13)
        assert len(find_darts(img, bg, DetectorConfig())) == 1

    def test_width_is_what_separates_them(self):
        """Both pass the elongation floor; only width tells them apart."""
        thin = _tip_from_points(
            np.argwhere(self._line(560, 358, 760, 358, 3)[0] > 0)[:, ::-1].astype(np.float32),
            DetectorConfig(),
        )
        fat = _tip_from_points(
            np.argwhere(self._line(560, 358, 700, 358, 13)[0] > 0)[:, ::-1].astype(np.float32),
            DetectorConfig(),
        )
        assert thin[2] > DetectorConfig().min_elongation
        assert fat[2] > DetectorConfig().min_elongation
        assert thin[4] < DetectorConfig().min_width_px < fat[4]

    def test_the_threshold_sits_between_the_measured_values(self):
        """2.7px phantom, 12.4px darts -- the gap is wide, so keep it wide."""
        w = DetectorConfig().min_width_px
        assert 2.7 < w < 12.4


class TestChangeMassIsBoardOnly:
    """A player standing at the oche must not read as a hand on the board.

    This is the "it stops counting when someone is near the board" bug. The
    scoring path already restricted itself to the board ROI; the trigger that
    gates it did not, so a person anywhere in frame raised the change mass past
    the hand threshold and the pipeline parked, refusing to score.
    """

    SHAPE = (720, 1280)

    def _scene(self):
        """An empty background, plus a frame holding a dart and a person."""
        bg = np.zeros(self.SHAPE, np.uint8)
        frame = bg.copy()
        roi = np.zeros(self.SHAPE, np.uint8)
        cv2.circle(roi, (900, 360), 150, 255, -1)          # the board, off to one side
        cv2.line(frame, (880, 330), (930, 390), 255, 11)   # a dart, inside it
        cv2.rectangle(frame, (60, 120), (330, 700), 255, -1)  # a person, well outside
        return frame, bg, roi

    def test_the_person_dominates_a_whole_frame_count(self):
        frame, bg, roi = self._scene()
        cfg = DetectorConfig()
        whole = change_mass(frame, bg, cfg)
        board = change_mass(frame, bg, cfg, roi)
        assert whole > 100_000        # the person, by area alone
        assert board < whole / 20     # the dart is a tiny fraction of it

    def test_only_the_dart_is_counted_inside_the_board(self):
        frame, bg, roi = self._scene()
        board = change_mass(frame, bg, DetectorConfig(), roi)
        area = int(cv2.countNonZero(roi))
        # Comfortably above "something landed", far below "a hand is at the board".
        assert PipelineConfig().dart_min_mass < board
        assert board < PipelineConfig().hand_fraction * area

    def test_a_hand_over_the_board_still_reads_as_a_hand(self):
        """The fix must not cost us the detection it was guarding."""
        bg = np.zeros(self.SHAPE, np.uint8)
        roi = np.zeros(self.SHAPE, np.uint8)
        cv2.circle(roi, (900, 360), 150, 255, -1)
        frame = bg.copy()
        cv2.rectangle(frame, (820, 260), (1000, 460), 255, -1)   # an arm across it
        board = change_mass(frame, bg, DetectorConfig(), roi)
        assert board > PipelineConfig().hand_fraction * int(cv2.countNonZero(roi))

    def test_no_roi_falls_back_to_the_whole_frame(self):
        frame, bg, _ = self._scene()
        cfg = DetectorConfig()
        assert change_mass(frame, bg, cfg, None) == change_mass(frame, bg, cfg)
