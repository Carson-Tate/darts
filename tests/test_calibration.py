"""Auto-calibration, tested against a synthetic board under a known homography.

Building the test board from render_reference() means these tests validate the
*machinery* -- ellipse fit, rotation lock, ECC refinement -- rather than the
colour tuning. Whether the yellow HSV window matches the real board can only be
checked against a real photo; drop one in samples/ and use tools/check_calib.py.

The rotation test is the important one. The ring pattern alone repeats every two
sectors, so before the numerals were added to the reference this search had ten
equally-good answers and picked one at random.
"""

import math

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from darts.board import REGULATION, score_at  # noqa: E402
from darts.vision.calibrate import (  # noqa: E402
    RECT_SIZE,
    _ncc,
    auto_calibrate,
    fit_board_ellipse,
    px_per_mm,
    render_reference,
    resolve_rotation,
    yellow_mask,
)

YELLOW = (40, 190, 230)  # BGR -- lands mid-window in the default YellowRange
DARK = (18, 18, 18)
WOOD = (90, 95, 105)  # desaturated, so it must not survive the yellow mask

CAM_W, CAM_H = 1280, 720

# A camera to the left of and below the board: the left rim is nearer, so it
# projects taller than the right. Roughly the geometry described for this setup.
CAM_QUAD = np.float32([[200, 100], [1060, 260], [1020, 600], [150, 640]])


def rect_from_board(x_mm, y_mm, geom=REGULATION):
    ppm = px_per_mm(geom)
    return RECT_SIZE / 2.0 + x_mm * ppm, RECT_SIZE / 2.0 - y_mm * ppm


def polar(radius_mm, angle_deg):
    th = math.radians(angle_deg)
    return radius_mm * math.sin(th), radius_mm * math.cos(th)


def synthetic_rect(geom=REGULATION):
    """A rectified BGR board image consistent with render_reference()."""
    mask = render_reference(geom)
    ppm = px_per_mm(geom)
    yy, xx = np.mgrid[0:RECT_SIZE, 0:RECT_SIZE].astype(np.float32)
    r = np.hypot(xx - RECT_SIZE / 2.0, yy - RECT_SIZE / 2.0) / ppm

    img = np.zeros((RECT_SIZE, RECT_SIZE, 3), np.uint8)
    img[:] = WOOD
    img[r <= geom.double_outer * 1.04] = DARK
    img[mask > 0] = YELLOW
    return img


def ground_truth_homography():
    src = np.float32([[0, 0], [RECT_SIZE, 0], [RECT_SIZE, RECT_SIZE], [0, RECT_SIZE]])
    return cv2.getPerspectiveTransform(src, CAM_QUAD)


def synthetic_camera_view(geom=REGULATION):
    h_gt = ground_truth_homography()
    view = cv2.warpPerspective(
        synthetic_rect(geom), h_gt, (CAM_W, CAM_H), borderValue=WOOD
    )
    return view, h_gt


def project(h, x, y):
    pt = np.array([[[float(x), float(y)]]], np.float64)
    out = cv2.perspectiveTransform(pt, h)[0][0]
    return float(out[0]), float(out[1])


# ---------------------------------------------------------------- reference


class TestReference:
    def test_is_not_thirty_six_degree_symmetric(self):
        """The regression test for the silent T20-scored-as-T18 bug.

        Alternating sector colours with the ring colours inverted repeat every
        two sectors. Only the numerals distinguish a board from itself rotated
        by 36 degrees.
        """
        ref = render_reference()
        centre = (RECT_SIZE / 2.0, RECT_SIZE / 2.0)
        rot = cv2.warpAffine(
            ref, cv2.getRotationMatrix2D(centre, 36.0, 1.0), (RECT_SIZE, RECT_SIZE)
        )
        assert _ncc(ref, ref) == pytest.approx(1.0)
        assert _ncc(rot, ref) < 0.995, "reference is symmetric; rotation cannot be pinned"

    def test_has_yellow_out_to_the_double_ring(self):
        ref = render_reference()
        # Probe 7 degrees off each sector's centre. The numeral sits dead centre
        # and spans roughly +/-3 degrees, and it is drawn in the *opposite*
        # colour to its band -- so sampling at the centre reads the glyph, not
        # the band. The sector is 18 degrees wide, so +7 is clear of the numeral
        # and still comfortably inside the sector.
        radius = REGULATION.double_outer - 3

        # Rings invert the single parity, so sector index 0 -- the 20 -- has a
        # yellow double. This outer arc is what the ellipse fit latches onto.
        u, v = rect_from_board(*polar(radius, 7.0))
        assert ref[int(v), int(u)] > 0

        # ...and its neighbour the 1 has a black double.
        u, v = rect_from_board(*polar(radius, 25.0))
        assert ref[int(v), int(u)] == 0

    def test_numerals_are_drawn_into_the_double_ring(self):
        """The numerals are load-bearing -- they are the only thing breaking the
        36-degree symmetry. Check they actually landed on the board."""
        ref = render_reference()
        # Centre of the 1's double band: black band, so the numeral is yellow.
        u, v = rect_from_board(*polar((REGULATION.double_inner + REGULATION.double_outer) / 2, 18.0))
        assert ref[int(v), int(u)] > 0, "numeral missing from the 1's double"

    def test_bull_area_is_not_yellow(self):
        ref = render_reference()
        assert ref[RECT_SIZE // 2, RECT_SIZE // 2] == 0


class TestRotationSearch:
    def test_locks_onto_the_identity_rotation(self):
        ref = render_reference()
        h, score, margin = resolve_rotation(ref, np.eye(3), ref)
        assert score > 0.99
        assert margin > 0.0
        # Identity up to resampling noise.
        assert np.allclose(h / h[2, 2], np.eye(3), atol=1e-6)

    def test_reports_a_usable_margin(self):
        ref = render_reference()
        _, _, margin = resolve_rotation(ref, np.eye(3), ref)
        # If this regresses toward 0 the numerals have stopped registering and
        # the orientation is being chosen at random from ten candidates. It has
        # happened once already, at margin 0.0003.
        assert margin > 0.04, f"rotation margin collapsed to {margin:.5f}"

    def test_recovers_a_seventy_two_degree_offset(self):
        """The exact failure seen on the first real run: T20 read as T16.

        72 degrees is a multiple of 36, so the ring pattern alone genuinely
        cannot distinguish it from zero. Only the numerals can.
        """
        ref = render_reference()
        centre = (RECT_SIZE / 2.0, RECT_SIZE / 2.0)
        turned = cv2.warpAffine(
            ref, cv2.getRotationMatrix2D(centre, 72.0, 1.0), (RECT_SIZE, RECT_SIZE)
        )
        h, _, margin = resolve_rotation(turned, np.eye(3), ref)
        recovered = cv2.warpPerspective(turned, h, (RECT_SIZE, RECT_SIZE))
        assert _ncc(recovered, ref) > 0.97, "did not undo the 72-degree offset"
        assert margin > 0.02

    @pytest.mark.parametrize("offset", [36.0, 72.0, 108.0, 144.0, 180.0])
    def test_recovers_every_symmetric_offset(self, offset):
        """All five multiples of 36 degrees are invisible to the ring pattern."""
        ref = render_reference()
        centre = (RECT_SIZE / 2.0, RECT_SIZE / 2.0)
        turned = cv2.warpAffine(
            ref, cv2.getRotationMatrix2D(centre, offset, 1.0), (RECT_SIZE, RECT_SIZE)
        )
        h, _, _ = resolve_rotation(turned, np.eye(3), ref)
        recovered = cv2.warpPerspective(turned, h, (RECT_SIZE, RECT_SIZE))
        assert _ncc(recovered, ref) > 0.97, f"{offset} degree offset not recovered"


# ------------------------------------------------------------ full pipeline


@pytest.fixture(scope="module")
def calibrated():
    view, h_gt = synthetic_camera_view()
    calib = auto_calibrate(view)
    if calib is None:
        # Report which stage gave up, so a failure here is actionable rather
        # than just "it didn't work".
        mask = yellow_mask(view, clean=False)
        ellipse = fit_board_ellipse(yellow_mask(view, clean=True))
        coverage = cv2.countNonZero(mask) / mask.size
        detail = (
            f"yellow mask covered {coverage * 100:.1f}% of the frame; "
            f"ellipse fit {'succeeded' if ellipse else 'FAILED'}"
        )
        if ellipse is not None:
            from darts.vision.calibrate import (
                RECT_SIZE as RS,
                affine_from_ellipse,
                refine_homography,
                render_reference as rr,
                resolve_rotation as rres,
            )

            ref = rr()
            h = affine_from_ellipse(ellipse)
            affine_score = _ncc(cv2.warpPerspective(mask, h, (RS, RS)), ref)
            h, rot, _ = rres(mask, h, ref)
            h = refine_homography(mask, h, ref)
            refined = _ncc(cv2.warpPerspective(mask, h, (RS, RS)), ref)
            detail += (
                f"; affine NCC {affine_score:.3f}, best rotation {rot:.3f}, "
                f"after one ECC pass {refined:.3f}"
            )
        pytest.fail(f"auto-calibration failed on a clean synthetic board -- {detail}")
    return calib, h_gt


class TestAutoCalibrate:
    def test_masks_the_board_and_not_the_wood(self):
        view, _ = synthetic_camera_view()
        mask = yellow_mask(view)
        assert cv2.countNonZero(mask) > 5000, "yellow board face was not detected"
        # Corners are wood; none of them should register as board.
        for (y, x) in [(5, 5), (5, CAM_W - 5), (CAM_H - 5, 5), (CAM_H - 5, CAM_W - 5)]:
            assert mask[y, x] == 0, "wood background leaked into the yellow mask"

    def test_finds_an_ellipse(self):
        view, _ = synthetic_camera_view()
        assert fit_board_ellipse(yellow_mask(view)) is not None

    def test_fine_mask_keeps_detail_the_clean_mask_destroys(self):
        """Regression: the rotation search ran on the cleaned mask, whose
        MORPH_CLOSE bridges roughly 9 px and so erases the ~3 px numeral
        strokes. That left the search with only the 36-degree-symmetric ring
        pattern to go on, and the lock became a coin flip."""
        view, _ = synthetic_camera_view()
        clean = yellow_mask(view, clean=True)
        fine = yellow_mask(view, clean=False)
        differing = cv2.countNonZero(cv2.absdiff(clean, fine))
        assert differing > 1000, "the two mask modes are identical; fix not in effect"

    def test_recovers_board_coordinates(self, calibrated):
        """Project known board points out through the ground-truth camera and
        check calibration brings them back."""
        calib, h_gt = calibrated
        errors = []
        for radius in (30, 80, 103, 140, 166):
            for angle in range(0, 360, 30):
                x_mm, y_mm = polar(radius, angle)
                u, v = rect_from_board(x_mm, y_mm)
                cam = project(h_gt, u, v)
                got = calib.image_to_board(*cam)
                errors.append(math.hypot(got[0] - x_mm, got[1] - y_mm))

        worst = max(errors)
        mean = sum(errors) / len(errors)
        print(f"\ncalibration error: mean {mean:.2f} mm, worst {worst:.2f} mm")
        # For reference: the treble band is 8 mm wide, so anything above ~3 mm
        # starts costing you trebles on a real board.
        assert worst < 6.0, f"worst-case error {worst:.2f} mm is too high"

    def test_scores_the_right_sector_and_ring(self, calibrated):
        calib, h_gt = calibrated
        cases = [
            (polar(103, 0.0), "T20"),
            (polar(166, 18.0), "D1"),
            (polar(103, 90.0), "T6"),
            (polar(50, 180.0), "S3"),
            (polar(166, 270.0), "D11"),
            ((0.0, 0.0), "BULL"),
        ]
        for (x_mm, y_mm), expected in cases:
            u, v = rect_from_board(x_mm, y_mm)
            got = score_at(*calib.image_to_board(*project(h_gt, u, v)))
            assert got.label == expected, f"({x_mm:.0f},{y_mm:.0f}) read as {got.label}"

    def test_rotation_lock_is_confident(self, calibrated):
        calib, _ = calibrated
        assert calib.rotation_is_confident, (
            f"margin only {calib.margin:.4f}; orientation was effectively a guess"
        )

    def test_board_to_image_inverts_image_to_board(self, calibrated):
        calib, _ = calibrated
        for x_mm, y_mm in [(0, 0), (100, 40), (-60, -120), (150, 0)]:
            px, py = calib.board_to_image(x_mm, y_mm)
            back = calib.image_to_board(px, py)
            assert back[0] == pytest.approx(x_mm, abs=0.1)
            assert back[1] == pytest.approx(y_mm, abs=0.1)

    def test_rotate_nudge_moves_by_exactly_one_sector(self, calibrated):
        calib, h_gt = calibrated
        u, v = rect_from_board(*polar(103, 0.0))  # T20 before the nudge
        cam = project(h_gt, u, v)
        assert score_at(*calib.image_to_board(*cam)).label == "T20"

        # One nudge should read the same physical point as one of the 20's
        # neighbours. Which direction depends on the sign convention; what the
        # nudge button has to guarantee is a single-sector step.
        nudged = score_at(*calib.rotated(1).image_to_board(*cam)).label
        assert nudged in ("T5", "T1"), f"one nudge jumped to {nudged}"

        # Twenty nudges is a full turn, back where it started.
        assert score_at(*calib.rotated(20).image_to_board(*cam)).label == "T20"


class TestFailureModes:
    def test_returns_none_on_a_blank_frame(self):
        blank = np.full((CAM_H, CAM_W, 3), WOOD, np.uint8)
        assert auto_calibrate(blank) is None

    def test_returns_none_when_there_is_no_board(self):
        noise = np.random.default_rng(0).integers(0, 60, (CAM_H, CAM_W, 3), dtype=np.uint8)
        assert auto_calibrate(noise) is None
