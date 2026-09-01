"""Board geometry and scoring. Pure Python -- no OpenCV, no cameras."""

import math

import pytest

from darts.board import (
    REGULATION,
    SECTORS,
    angle_of_sector,
    hit_from_label,
    score_at,
    sector_at_angle,
)


def polar(radius_mm, angle_deg):
    """Board coordinate at a radius and a clockwise-from-top angle."""
    th = math.radians(angle_deg)
    return radius_mm * math.sin(th), radius_mm * math.cos(th)


class TestSectorLayout:
    def test_twenty_is_straight_up(self):
        assert sector_at_angle(0.0) == 20

    def test_three_is_straight_down(self):
        assert sector_at_angle(180.0) == 3

    def test_six_is_due_right_and_eleven_due_left(self):
        assert sector_at_angle(90.0) == 6
        assert sector_at_angle(270.0) == 11

    def test_sector_spans_eighteen_degrees_centred_on_its_number(self):
        assert sector_at_angle(8.9) == 20
        assert sector_at_angle(9.1) == 1
        assert sector_at_angle(351.1) == 20
        assert sector_at_angle(350.9) == 5

    def test_every_sector_round_trips(self):
        for value in SECTORS:
            assert sector_at_angle(angle_of_sector(value)) == value

    def test_wraps_past_full_circle(self):
        assert sector_at_angle(360.0) == 20
        assert sector_at_angle(-18.0) == 5


class TestRings:
    def test_bulls(self):
        assert score_at(0, 0).points == 50
        assert score_at(0, 0).ring == "inner_bull"
        assert score_at(*polar(10, 0)).points == 25
        assert score_at(*polar(10, 0)).ring == "outer_bull"

    def test_treble_twenty(self):
        hit = score_at(*polar(103, 0))
        assert (hit.ring, hit.points, hit.label) == ("triple", 60, "T20")

    def test_double_twenty(self):
        hit = score_at(*polar(166, 0))
        assert (hit.ring, hit.points, hit.label) == ("double", 40, "D20")

    def test_both_single_bands(self):
        for radius in (50, 130):
            hit = score_at(*polar(radius, 0))
            assert (hit.ring, hit.points) == ("single", 20)

    def test_outside_the_doubles_scores_nothing(self):
        hit = score_at(*polar(175, 0))
        assert (hit.ring, hit.points, hit.label) == ("miss", 0, "MISS")

    @pytest.mark.parametrize("radius,expected", [
        (6.3, "inner_bull"), (6.4, "outer_bull"), (15.8, "outer_bull"),
        (16.0, "single"), (98.9, "single"), (99.1, "triple"),
        (107.1, "single"), (162.1, "double"), (170.1, "miss"),
    ])
    def test_ring_boundaries(self, radius, expected):
        assert score_at(*polar(radius, 0)).ring == expected


class TestDoubleFlag:
    def test_double_ring_counts_as_a_double(self):
        assert score_at(*polar(166, 0)).is_double

    def test_inner_bull_counts_as_a_double(self):
        # Standard rule: the bull is D25, so it finishes a double-out leg.
        assert score_at(0, 0).is_double

    def test_outer_bull_does_not(self):
        assert not score_at(*polar(10, 0)).is_double

    def test_trebles_and_singles_do_not(self):
        assert not score_at(*polar(103, 0)).is_double
        assert not score_at(*polar(50, 0)).is_double


class TestLabels:
    @pytest.mark.parametrize("label,points", [
        ("T20", 60), ("D16", 32), ("S5", 5), ("BULL", 50), ("25", 25), ("MISS", 0),
    ])
    def test_points(self, label, points):
        assert hit_from_label(label).points == points

    def test_label_round_trips_through_geometry(self):
        for label in ("T20", "D16", "S7", "D1", "T19"):
            assert hit_from_label(label).label == label

    def test_labels_are_case_insensitive(self):
        assert hit_from_label("t20").points == 60
        assert hit_from_label("bull").points == 50

    @pytest.mark.parametrize("bad", ["X20", "T21", "T0", "", "hello", "D"])
    def test_rejects_nonsense(self, bad):
        with pytest.raises(ValueError):
            hit_from_label(bad)


class TestNonRegulationScaling:
    def test_rings_scale_together(self):
        small = REGULATION.scaled_to(150.0)
        assert small.double_outer == 150.0
        ratio = 150.0 / 170.0
        assert small.triple_outer == pytest.approx(107.0 * ratio)
        assert small.inner_bull == pytest.approx(6.35 * ratio)

    def test_scaled_board_still_scores_consistently(self):
        small = REGULATION.scaled_to(150.0)
        # 103mm is the treble band on a regulation board; scaled down it is 90.9.
        hit = score_at(*polar(103.0 * 150.0 / 170.0, 0), geom=small)
        assert (hit.ring, hit.points) == ("triple", 60)
