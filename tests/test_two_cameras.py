"""Two cameras: keeping them apart, and keeping the room out of the detector.

The second camera is mounted about three feet above the first and angled down.
That buys a genuinely different view of a dart's protrusion -- the reason for
having two at all -- but it also brings three problems that one camera never
had, and this file is about those three:

  * it sees a doorway, a fridge and the dart holders as well as the board, and
    to a differencing detector a person walking past is a large dark elongated
    blob, which is also the description of a dart;
  * the kernel numbers video nodes in enumeration order, so which webcam is
    "low" and which is "high" can swap on a reboot -- silently, since both
    still calibrate and both still score;
  * each camera resolves the board's 36-degree symmetry against its own view,
    so they can disagree about the orientation, and one shared Rotate button
    cannot fix one without breaking the other.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from darts.board import REGULATION  # noqa: E402
from darts.vision import camera as camera_mod  # noqa: E402
from darts.vision import detect  # noqa: E402
from darts.vision.calibrate import Calibration, RECT_SIZE, px_per_mm  # noqa: E402
from darts.vision.detect import BackgroundModel, DetectorConfig, find_darts  # noqa: E402
from darts.vision import pipeline as pipeline_mod  # noqa: E402
from darts.vision.pipeline import PipelineConfig, VisionPipeline  # noqa: E402

IMG_W, IMG_H = 640, 480
BOARD_PX = (320, 240)  # where the board centre sits in these synthetic frames


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


def flat_calibration() -> Calibration:
    """A calibration for a camera square-on to the board at 1 px per mm."""
    ppm = px_per_mm(REGULATION)
    h = np.array(
        [
            [ppm, 0.0, RECT_SIZE / 2.0 - BOARD_PX[0] * ppm],
            [0.0, ppm, RECT_SIZE / 2.0 - BOARD_PX[1] * ppm],
            [0.0, 0.0, 1.0],
        ],
        np.float64,
    )
    return Calibration(h, REGULATION, score=0.9, image_size=(IMG_W, IMG_H), margin=0.2)


class TestFusingDisagreement:
    """Averaging is right when they agree and wrong when they don't."""

    def test_close_estimates_are_averaged(self):
        """The whole reason for two cameras: each misjudges the dart's
        protrusion in a different direction, so the midpoint beats either."""
        point, conf = detect.fuse([(100.0, 0.0), (106.0, 0.0)])
        assert point == pytest.approx((103.0, 0.0))
        assert conf > 0.9

    def test_a_wide_split_takes_the_preferred_camera_not_the_middle(self):
        """Measured: a dart in the 20 fused to a point in the 4.

        Two estimates 45mm apart are not one measurement with noise on it --
        one has mistaken the barrel for the point. Their midpoint is a place
        neither camera saw, and on a board divided into 18-degree wedges that
        lands in a third sector.
        """
        first = (0.0, 120.0)
        point, conf = detect.fuse([first, (95.0, 60.0)], trust_one_mm=25.0)
        assert point == pytest.approx(first)
        assert conf < 0.5, "still flagged, so the UI invites a correction"

    def test_the_split_point_is_reported_as_no_better_than_before(self):
        """Picking a side is not a claim to have resolved the disagreement."""
        _, conf = detect.fuse([(0.0, 120.0), (95.0, 60.0)], trust_one_mm=25.0)
        _, same = detect.fuse([(0.0, 120.0), (95.0, 60.0)], trust_one_mm=10_000.0)
        assert conf == pytest.approx(same)

    def test_one_camera_is_unchanged(self):
        point, conf = detect.fuse([(42.0, -13.0)])
        assert point == pytest.approx((42.0, -13.0))
        assert conf == pytest.approx(0.6)


class TestMeasureOrdersCamerasByPreference:
    def test_the_primary_leads_so_the_fallback_is_defined(self, tmp_path):
        """fuse falls back to the first entry, so it must not be dict order."""
        pipe = VisionPipeline([], PipelineConfig(geom=REGULATION, template_dir=tmp_path))

        class Named:
            def __init__(self, n):
                self.cfg = camera_mod.CameraConfig(name=n)

        pipe.cameras = [Named("low"), Named("high")]
        pipe.calibrations = {"low": flat_calibration(), "high": flat_calibration()}
        for n in ("low", "high"):
            bg = BackgroundModel()
            bg.background = detect.preprocess(np.zeros((IMG_H, IMG_W, 3), np.uint8))
            pipe.backgrounds[n] = bg

        frame = np.zeros((IMG_H, IMG_W, 3), np.uint8)
        cv2.fillPoly(frame, [dart_polygon(300, 225, 355, 248)], (255, 255, 255))
        seen = []
        pipe.on_dart = seen.append

        # Deliberately the wrong way round, as a stalled primary would leave it.
        pipe._measure({"high": frame, "low": frame.copy()})

        assert len(seen) == 1
        assert list(seen[0].per_camera) == ["low", "high"]


class TestBoardMask:
    """The ROI that keeps the rest of the room out of dart detection."""

    def test_covers_the_board_and_nothing_far_from_it(self):
        mask = flat_calibration().board_mask((IMG_H, IMG_W), reach=1.0)
        cx, cy = BOARD_PX
        assert mask[cy, cx] == 255, "the bull is on the board"
        assert mask[cy, cx + 150] == 255, "150mm out is still on the board"
        assert mask[cy, cx + 200] == 0, "200mm out is past a 170mm double ring"
        assert mask[5, 5] == 0, "the corner of the frame is not the board"

    def test_reach_extends_past_the_scoring_area(self):
        """Wide enough to keep a whole dart, not just its scoring end.

        The taper cue needs both ends of the silhouette to tell the point from
        the flight. A mask that stopped at the double ring would clip the
        flight off darts in the doubles and make them ambiguous.
        """
        calib = flat_calibration()
        tight = calib.board_mask((IMG_H, IMG_W), reach=1.0)
        wide = calib.board_mask((IMG_H, IMG_W), reach=1.6)
        assert cv2.countNonZero(wide) > cv2.countNonZero(tight)
        cx, cy = BOARD_PX
        assert wide[cy, cx + 200] == 255

    def test_matches_the_frame_it_was_asked_for(self):
        mask = flat_calibration().board_mask((IMG_H, IMG_W))
        assert mask.shape == (IMG_H, IMG_W)
        assert mask.dtype == np.uint8


class TestBoardBounds:
    """Cropping the preview to the board. A display choice, not a detector one."""

    def test_brackets_the_board_and_stays_inside_the_frame(self):
        x0, y0, x1, y1 = flat_calibration().board_bounds((IMG_H, IMG_W), reach=1.0)
        cx, cy = BOARD_PX
        assert x0 <= cx - 170 and x1 >= cx + 170
        assert 0 <= x0 < x1 <= IMG_W
        assert 0 <= y0 < y1 <= IMG_H

    def test_it_is_a_real_crop_of_a_cluttered_frame(self):
        """The overhead camera gives about three quarters of its frame to the
        room; on a phone tile that leaves the board too small to check."""
        x0, y0, x1, y1 = flat_calibration().board_bounds((IMG_H, IMG_W))
        assert (x1 - x0) * (y1 - y0) < IMG_W * IMG_H * 0.75

    def test_a_degenerate_fit_falls_back_to_the_whole_frame(self):
        """Better a wrong-looking picture than an empty one."""
        broken = replace(flat_calibration(), h_img2rect=np.diag([1e6, 1e6, 1.0]))
        assert broken.board_bounds((IMG_H, IMG_W)) == (0, 0, IMG_W, IMG_H)


class TestPreviewCropping:
    def _pipeline(self, tmp_path):
        pipe = VisionPipeline([], PipelineConfig(geom=REGULATION, template_dir=tmp_path))

        class Named:
            cfg = camera_mod.CameraConfig(name="low")

        pipe.cameras = [Named()]
        pipe.latest = {"low": np.random.randint(0, 255, (IMG_H, IMG_W, 3), dtype=np.uint8)}
        pipe.calibrations = {"low": flat_calibration()}
        return pipe

    def test_cropping_changes_the_aspect_it_returns(self, tmp_path):
        pipe = self._pipeline(tmp_path)
        full = pipe.preview_jpeg("low", overlay=False, crop=False)
        cropped = pipe.preview_jpeg("low", overlay=False, crop=True)
        a = cv2.imdecode(np.frombuffer(full, np.uint8), cv2.IMREAD_COLOR)
        b = cv2.imdecode(np.frombuffer(cropped, np.uint8), cv2.IMREAD_COLOR)
        assert a.shape[1] == IMG_W
        assert b.shape[1] < a.shape[1]

    def test_an_uncalibrated_camera_is_not_cropped_to_nothing(self, tmp_path):
        """No calibration means no idea where the board is -- show everything."""
        pipe = self._pipeline(tmp_path)
        pipe.calibrations = {}
        jpeg = pipe.preview_jpeg("low", overlay=False, crop=True)
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert img.shape[1] == IMG_W


class TestRoiFiltering:
    def _frames(self):
        """A dart on the board, and a bigger dart-shaped thing off in the room."""
        frame = np.zeros((IMG_H, IMG_W, 3), np.uint8)
        cv2.fillPoly(frame, [dart_polygon(300, 225, 355, 248)], (255, 255, 255))
        cv2.fillPoly(frame, [dart_polygon(30, 40, 150, 85)], (255, 255, 255))
        background = detect.preprocess(np.zeros((IMG_H, IMG_W, 3), np.uint8))
        return frame, background

    def test_without_a_roi_the_room_is_the_biggest_blob(self):
        frame, background = self._frames()
        blobs = find_darts(detect.preprocess(frame), background, DetectorConfig())
        assert len(blobs) == 2
        # Sorted by area: the thing across the room wins, which is exactly the
        # failure mode -- it is not a dart and it is not on the board.
        assert blobs[0].centroid[0] < 200

    def test_the_roi_removes_it_entirely(self):
        frame, background = self._frames()
        roi = flat_calibration().board_mask((IMG_H, IMG_W), reach=1.6)
        blobs = find_darts(
            detect.preprocess(frame), background, DetectorConfig(), roi=roi
        )
        assert len(blobs) == 1
        assert blobs[0].centroid[0] > 250

    def test_a_blob_straddling_the_edge_does_not_take_the_dart_with_it(self):
        """Why the ROI drops whole contours instead of masking pixels.

        The off-board blob here crosses the ROI boundary. Masking the pixels
        cut it, and the offcut's centroid moved inwards far enough to come
        within merging distance of the real dart; the two merged into one shape
        too big to be a dart, and both were rejected. A mask meant to remove
        one false blob removed the true one as well.
        """
        frame, background = self._frames()
        roi = flat_calibration().board_mask((IMG_H, IMG_W), reach=1.6)
        cut = cv2.bitwise_and(
            detect.foreground_mask(detect.preprocess(frame), background, DetectorConfig()),
            roi,
        )
        contours, _ = cv2.findContours(cut, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        assert len(contours) == 2, "the off-board blob really does straddle the edge"

        blobs = find_darts(
            detect.preprocess(frame), background, DetectorConfig(), roi=roi
        )
        assert len(blobs) == 1, "the dart must survive its neighbour being dropped"
        # Still the whole dart, not a merged pair spanning half the frame.
        assert np.hypot(*np.subtract(blobs[0].tip, blobs[0].other_end)) < 100


class TestMeasureBlobChoice:
    """Falling through to the real dart instead of stopping at the first blob."""

    def _pipeline(self, tmp_path):
        pipe = VisionPipeline([], PipelineConfig(geom=REGULATION, template_dir=tmp_path))
        pipe.calibrations = {"low": flat_calibration()}
        bg = BackgroundModel()
        bg.background = detect.preprocess(np.zeros((IMG_H, IMG_W, 3), np.uint8))
        pipe.backgrounds = {"low": bg}
        return pipe

    def _frame(self):
        frame = np.zeros((IMG_H, IMG_W, 3), np.uint8)
        cv2.fillPoly(frame, [dart_polygon(300, 225, 355, 248)], (255, 255, 255))
        cv2.fillPoly(frame, [dart_polygon(30, 40, 150, 85)], (255, 255, 255))
        return frame

    def test_scores_the_dart_despite_a_bigger_blob_off_the_board(self, tmp_path):
        """Stopping at the largest blob lost the throw on that camera.

        The largest blob is off the board, so it is rejected -- and with a
        single-blob look that rejection ended the camera's contribution, even
        though the actual dart was sitting in the next contour along.
        """
        pipe = self._pipeline(tmp_path)
        seen = []
        pipe.on_dart = seen.append
        pipe._measure({"low": self._frame()})

        assert len(seen) == 1, "the dart on the board should still have scored"
        x_mm, y_mm = seen[0].per_camera["low"]
        assert np.hypot(x_mm, y_mm) <= REGULATION.double_outer

    def test_a_blob_far_from_the_board_is_still_ignored(self, tmp_path):
        """354mm out on a 170mm board is something else in the room.

        The miss fallback must bound itself rather than lean on the detection
        ROI having excluded it: the ROI only exists once a camera has
        calibrated, and a phantom dart costs a real one out of the turn.
        """
        pipe = self._pipeline(tmp_path)
        seen = []
        pipe.on_dart = seen.append
        frame = np.zeros((IMG_H, IMG_W, 3), np.uint8)
        cv2.fillPoly(frame, [dart_polygon(30, 40, 150, 85)], (255, 255, 255))
        pipe._measure({"low": frame})
        assert seen == []

    def test_a_dart_just_off_the_board_is_scored_as_a_miss(self, tmp_path):
        """A miss is still a dart thrown; dropping it kept it off the scoreboard."""
        pipe = self._pipeline(tmp_path)
        seen = []
        pipe.on_dart = seen.append
        cx, cy = BOARD_PX
        frame = np.zeros((IMG_H, IMG_W, 3), np.uint8)
        cv2.fillPoly(
            frame, [dart_polygon(cx + 200, cy + 40, cx + 255, cy + 63)], (255, 255, 255)
        )
        pipe._measure({"low": frame})

        assert len(seen) == 1
        assert seen[0].hit.label == "MISS"
        assert seen[0].hit.points == 0


class TestPerCameraRotation:
    def _pipeline(self, tmp_path):
        pipe = VisionPipeline([], PipelineConfig(geom=REGULATION, template_dir=tmp_path))
        pipe.calibrations = {"low": flat_calibration(), "high": flat_calibration()}
        return pipe

    def test_rotating_one_camera_leaves_the_other_alone(self, tmp_path):
        """The two lock onto the symmetry independently, so they can disagree.

        Rotating both together can only ever fix one of them.
        """
        pipe = self._pipeline(tmp_path)
        before = {n: c.h_img2rect.copy() for n, c in pipe.calibrations.items()}

        pipe.nudge_rotation(1, camera="high")

        assert np.allclose(pipe.calibrations["low"].h_img2rect, before["low"])
        assert not np.allclose(pipe.calibrations["high"].h_img2rect, before["high"])

    def test_no_camera_named_still_rotates_everything(self, tmp_path):
        pipe = self._pipeline(tmp_path)
        before = {n: c.h_img2rect.copy() for n, c in pipe.calibrations.items()}
        pipe.nudge_rotation(1)
        for name in before:
            assert not np.allclose(pipe.calibrations[name].h_img2rect, before[name])

    def test_an_unknown_camera_changes_nothing(self, tmp_path):
        pipe = self._pipeline(tmp_path)
        before = {n: c.h_img2rect.copy() for n, c in pipe.calibrations.items()}
        pipe.nudge_rotation(1, camera="nonesuch")
        for name in before:
            assert np.allclose(pipe.calibrations[name].h_img2rect, before[name])


class TestConfirmingAnOrientation:
    """Saying "this one is right" without having to rotate all the way round.

    The overhead camera locked onto the correct orientation with a margin of
    0.008 -- right, but not confidently so. Templates were only ever saved as a
    side effect of rotating, so the only way to record that was twenty taps
    back to where it started.
    """

    def _pipeline(self, tmp_path, margin=0.001):
        pipe = VisionPipeline([], PipelineConfig(geom=REGULATION, template_dir=tmp_path))

        class Named:
            cfg = camera_mod.CameraConfig(name="high")

        pipe.cameras = [Named()]
        pipe.calibrations = {"high": replace(flat_calibration(), margin=margin)}
        pipe.latest = {"high": np.zeros((IMG_H, IMG_W, 3), np.uint8)}
        return pipe

    def test_confirming_saves_the_template_without_moving_anything(self, tmp_path):
        pipe = self._pipeline(tmp_path)
        before = pipe.calibrations["high"].h_img2rect.copy()

        pipe.nudge_rotation(0, camera="high")

        assert np.allclose(pipe.calibrations["high"].h_img2rect, before)
        assert "high" in pipe.templates
        assert (tmp_path / "high.png").is_file(), "must survive a restart"

    def test_a_remembered_orientation_reads_as_confident(self, tmp_path):
        """Someone looking at the overlay is better evidence than the numerals.

        The margin only says how well the printed numbers picked one of ten
        symmetric candidates. A saved template means a person checked.
        """
        pipe = self._pipeline(tmp_path)
        assert pipe.status()["per_camera"]["high"]["rotation_confident"] is False

        pipe.nudge_rotation(0, camera="high")

        after = pipe.status()["per_camera"]["high"]
        assert after["remembered"] is True
        assert after["rotation_confident"] is True

    def test_forgetting_makes_it_unsure_again(self, tmp_path):
        pipe = self._pipeline(tmp_path)
        pipe.nudge_rotation(0, camera="high")
        pipe.forget_orientation()
        after = pipe.status()["per_camera"]["high"]
        assert after["remembered"] is False
        assert after["rotation_confident"] is False

    def test_confirming_one_camera_does_not_speak_for_the_other(self, tmp_path):
        """Confirming is the user asserting something about one view.

        Saving every camera's template meant confirming the overhead camera also
        recorded the other's orientation as truth -- so a wrong lock on the one
        you weren't looking at got written down and faithfully restored on every
        later calibration.
        """
        pipe = VisionPipeline([], PipelineConfig(geom=REGULATION, template_dir=tmp_path))
        pipe.calibrations = {"low": flat_calibration(), "high": flat_calibration()}
        blank = np.zeros((IMG_H, IMG_W, 3), np.uint8)
        pipe.latest = {"low": blank, "high": blank.copy()}

        pipe.nudge_rotation(0, camera="high")

        assert set(pipe.templates) == {"high"}
        assert not (tmp_path / "low.png").exists()

    def test_rotating_every_camera_still_saves_every_template(self, tmp_path):
        pipe = VisionPipeline([], PipelineConfig(geom=REGULATION, template_dir=tmp_path))
        pipe.calibrations = {"low": flat_calibration(), "high": flat_calibration()}
        blank = np.zeros((IMG_H, IMG_W, 3), np.uint8)
        pipe.latest = {"low": blank, "high": blank.copy()}

        pipe.nudge_rotation(1)

        assert set(pipe.templates) == {"low", "high"}

    def test_an_already_confident_camera_is_not_reported_as_remembered(self, tmp_path):
        pipe = self._pipeline(tmp_path, margin=0.5)
        info = pipe.status()["per_camera"]["high"]
        assert info["rotation_confident"] is True
        assert info["remembered"] is False, "confident is not the same as confirmed"


class TestLateCamera:
    """A camera that misses the first attempt must get another chance.

    The overhead camera runs its own auto-exposure against a scene with a
    bright doorway in it, and is still settling when calibration first runs: it
    scored 0.27 against a 0.35 gate at startup and 0.74 on a frame taken a
    minute later. Once the primary calibrates there is nothing left to trigger
    a retry, so without this it stayed dropped for the whole session.
    """

    def _pipeline(self, tmp_path):
        pipe = VisionPipeline([], PipelineConfig(geom=REGULATION, template_dir=tmp_path))
        low = flat_calibration()
        pipe.calibrations = {"low": low}
        pipe.rois = {"low": low.board_mask((IMG_H, IMG_W))}
        return pipe, low

    def _frames(self):
        blank = np.zeros((IMG_H, IMG_W, 3), np.uint8)
        return {"low": blank, "high": blank.copy()}

    def test_a_straggler_joins_without_disturbing_the_primary(self, tmp_path, monkeypatch):
        pipe, low = self._pipeline(tmp_path)
        monkeypatch.setattr(
            pipeline_mod, "auto_calibrate", lambda *a, **k: flat_calibration()
        )
        assert pipe._calibrate(self._frames(), only={"high"}) is True
        assert set(pipe.calibrations) == {"low", "high"}
        assert pipe.calibrations["low"] is low, "the working camera was rebuilt"
        assert set(pipe.rois) == {"low", "high"}

    def test_a_straggler_that_fails_again_changes_nothing(self, tmp_path, monkeypatch):
        pipe, low = self._pipeline(tmp_path)
        monkeypatch.setattr(pipeline_mod, "auto_calibrate", lambda *a, **k: None)
        assert pipe._calibrate(self._frames(), only={"high"}) is False
        assert set(pipe.calibrations) == {"low"}
        assert pipe.calibrations["low"] is low

    def test_a_retry_never_touches_cameras_it_was_not_asked_about(self, tmp_path, monkeypatch):
        """`only` must gate which cameras are calibrated, not just which are kept.

        Recalibrating the primary here would be worse than pointless: it would
        re-roll the orientation it had already settled, mid-session.
        """
        pipe, _ = self._pipeline(tmp_path)
        seen = []

        def spy(frame, *a, **k):
            seen.append(frame.shape)
            return flat_calibration()

        monkeypatch.setattr(pipeline_mod, "auto_calibrate", spy)
        pipe._calibrate(self._frames(), only={"high"})
        assert len(seen) == 1, "only the straggler should have been calibrated"


@pytest.fixture
def by_id(tmp_path, monkeypatch):
    """A stand-in for /dev/v4l/by-id, laid out as the Pi's really is.

    Real symlinks rather than patched os calls, so this exercises the same
    readlink path the Pi takes. The index1 entries are the metadata nodes: they
    open happily and never yield a picture, so picking one looks like a camera
    that is present and permanently blank.
    """
    root = tmp_path / "by-id"
    root.mkdir()
    entries = {
        "usb-Sonix_Technology_Co.__Ltd._USB_2.0_Camera_SN0001-video-index0": "/dev/video2",
        "usb-Sonix_Technology_Co.__Ltd._USB_2.0_Camera_SN0001-video-index1": "/dev/video3",
        "usb-Sonix_Technology_Co.__Ltd._onn_4K_Webcam_SN0001-video-index0": "/dev/video0",
        "usb-Sonix_Technology_Co.__Ltd._onn_4K_Webcam_SN0001-video-index1": "/dev/video1",
    }
    for name, target in entries.items():
        try:
            (root / name).symlink_to(target)
        except (OSError, NotImplementedError):  # Windows without privileges
            pytest.skip("symlinks are not available here")
    monkeypatch.setattr(camera_mod, "BY_ID", str(root))
    return root


def resolved(**kw):
    cam = camera_mod.Camera(camera_mod.CameraConfig(**kw))
    cam._resolve()
    return cam


class TestDeviceResolution:
    """Which webcam is which must not depend on enumeration order.

    The kernel hands out video0/video2 in the order devices come up, so a
    reboot or a replug can swap the low and high cameras. Nothing about that
    looks broken: both still calibrate and both still score, each using the
    other's view of the board.
    """

    def test_name_hint_wins_over_the_index(self, by_id):
        cam = resolved(name="high", name_hint="USB_2.0_Camera", index=7)
        assert cam.source == 2
        assert cam.node == "/dev/video2"

    def test_picks_the_capture_node_not_the_metadata_node(self, by_id):
        cam = resolved(name="low", name_hint="onn_4K_Webcam", index=9)
        assert cam.source == 0, "index1 is the metadata node, not the camera"

    def test_the_two_cameras_resolve_to_different_nodes(self, by_id):
        low = resolved(name="low", name_hint="onn_4K_Webcam", index=0)
        high = resolved(name="high", name_hint="USB_2.0_Camera", index=2)
        assert low.source != high.source

    def test_falls_back_to_the_index_when_the_hint_matches_nothing(self, by_id):
        cam = resolved(name="low", name_hint="no_such_camera", index=3)
        assert cam.source == 3
        assert cam.node == "/dev/video3"

    def test_a_bare_index_still_works(self, by_id):
        cam = resolved(name="low", index=1)
        assert cam.source == 1
        assert cam.node == "/dev/video1"


class TestPreviewSizing:
    """The page shows every camera at once, so preview bytes are doubled."""

    def _pipeline(self, tmp_path):
        pipe = VisionPipeline([], PipelineConfig(geom=REGULATION, template_dir=tmp_path))

        class Named:
            cfg = camera_mod.CameraConfig(name="low")

        pipe.cameras = [Named()]
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        pipe.latest = {"low": frame}
        return pipe

    def test_width_shrinks_the_frame(self, tmp_path):
        pipe = self._pipeline(tmp_path)
        small = pipe.preview_jpeg("low", overlay=False, width=360, quality=55)
        decoded = cv2.imdecode(np.frombuffer(small, np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape[1] == 360
        assert decoded.shape[0] == 202  # 16:9 preserved

    def test_a_tile_costs_far_less_than_a_full_frame(self, tmp_path):
        pipe = self._pipeline(tmp_path)
        full = pipe.preview_jpeg("low", overlay=False)
        tile = pipe.preview_jpeg("low", overlay=False, width=360, quality=55)
        assert len(tile) * 4 < len(full)

    def test_width_never_upscales(self, tmp_path):
        pipe = self._pipeline(tmp_path)
        big = pipe.preview_jpeg("low", overlay=False, width=4000)
        decoded = cv2.imdecode(np.frombuffer(big, np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape[1] == 1280


class TestBoardMetering:
    """Steering a manual-exposure camera by how bright the *board* is.

    The camera's own auto-exposure meters the whole frame, and the overhead
    camera's frame is three-quarters doorway and white door -- it stopped down
    to 312us and left the board at half the brightness and half the contrast of
    the other camera. Pinning the exposure by hand fixed that and introduced a
    worse problem: it is only right for the light that was in the room when it
    was pinned. Measured over ninety minutes of an evening the board fell from
    median grey 117 to 88, and the camera stopped being able to calibrate.
    """

    class FakeCam:
        def __init__(self, exposure=1100, autoexposure=False):
            self.cfg = camera_mod.CameraConfig(
                name="high", exposure=exposure, autoexposure=autoexposure
            )
            self.set_calls = []

        def set_exposure(self, v):
            self.set_calls.append(v)
            self.cfg.exposure = v
            return True

    def _pipeline(self, tmp_path, cam):
        pipe = VisionPipeline([], PipelineConfig(geom=REGULATION, template_dir=tmp_path))
        pipe.cameras = [cam]
        pipe.rois = {"high": np.full((IMG_H, IMG_W), 255, np.uint8)}
        return pipe

    def _frame(self, grey):
        return np.full((IMG_H, IMG_W, 3), grey, np.uint8)

    def test_a_dark_board_raises_the_exposure(self, tmp_path):
        cam = self.FakeCam(exposure=1100)
        pipe = self._pipeline(tmp_path, cam)
        assert pipe._tune_exposure({"high": self._frame(88)}) is True
        assert cam.set_calls and cam.set_calls[0] > 1100

    def test_a_bright_board_lowers_it(self, tmp_path):
        cam = self.FakeCam(exposure=1100)
        pipe = self._pipeline(tmp_path, cam)
        assert pipe._tune_exposure({"high": self._frame(200)}) is True
        assert cam.set_calls[0] < 1100

    def test_a_board_already_on_target_is_left_alone(self, tmp_path):
        """Exposure changes invalidate the background; don't chase noise."""
        cam = self.FakeCam(exposure=1100)
        pipe = self._pipeline(tmp_path, cam)
        assert pipe._tune_exposure({"high": self._frame(120)}) is False
        assert cam.set_calls == []

    def test_one_correction_is_bounded(self, tmp_path):
        """A single bad frame must not swing the exposure wildly."""
        cam = self.FakeCam(exposure=1100)
        pipe = self._pipeline(tmp_path, cam)
        pipe._tune_exposure({"high": self._frame(4)})
        assert cam.set_calls[0] <= 1100 * PipelineConfig().exposure_step + 1

    def test_it_converges(self, tmp_path):
        """Repeated passes should reach the target, not oscillate."""
        cam = self.FakeCam(exposure=1100)
        pipe = self._pipeline(tmp_path, cam)
        grey = 88.0
        for _ in range(8):
            if not pipe._tune_exposure({"high": self._frame(int(grey))}):
                break
            grey *= cam.set_calls[-1] / max(cam.set_calls[-2] if len(cam.set_calls) > 1 else 1100, 1)
        assert abs(grey - PipelineConfig().board_grey_target) <= PipelineConfig().board_grey_tolerance

    def test_an_autoexposure_camera_is_never_touched(self, tmp_path):
        cam = self.FakeCam(autoexposure=True)
        pipe = self._pipeline(tmp_path, cam)
        assert pipe._tune_exposure({"high": self._frame(20)}) is False
        assert cam.set_calls == []


class TestRejectingANonBoard:
    """A template match is evidence about *what* was found, not just how it sits.

    Seen for real: an exposure change pushed the wooden cabinet into the yellow
    range, the ellipse fitted the doorway, it passed the alignment gate, matched
    the saved template at 0.32 against the other camera's 0.78, and was reported
    as calibrated and confident while drawing the scoring grid over a fridge.
    Confidently wrong is worse than refusing.
    """

    def _pipeline(self, tmp_path, match):
        pipe = VisionPipeline([], PipelineConfig(geom=REGULATION, template_dir=tmp_path))

        class Named:
            cfg = camera_mod.CameraConfig(name="high")

        pipe.cameras = [Named()]
        pipe.templates = {"high": np.zeros((RECT_SIZE, RECT_SIZE), np.uint8)}
        return pipe, match

    def _run(self, tmp_path, monkeypatch, match):
        pipe, _ = self._pipeline(tmp_path, match)
        monkeypatch.setattr(pipeline_mod, "auto_calibrate", lambda *a, **k: flat_calibration())
        monkeypatch.setattr(pipeline_mod, "yellow_mask", lambda *a, **k: np.zeros((IMG_H, IMG_W), np.uint8))
        monkeypatch.setattr(
            pipeline_mod, "orient_to_template",
            lambda *a, **k: (flat_calibration().h_img2rect, match, 0.2),
        )
        frames = {"high": np.zeros((IMG_H, IMG_W, 3), np.uint8)}
        return pipe, pipe._calibrate(frames)

    def test_a_poor_template_match_is_rejected(self, tmp_path, monkeypatch):
        pipe, ok = self._run(tmp_path, monkeypatch, match=0.32)
        assert ok is False
        assert pipe.calibrations == {}

    def test_a_good_template_match_is_accepted(self, tmp_path, monkeypatch):
        pipe, ok = self._run(tmp_path, monkeypatch, match=0.78)
        assert ok is True
        assert "high" in pipe.calibrations


class TestExposureBounds:
    """Exposure and the yellow threshold are not independent."""

    def _cam(self, **kw):
        return camera_mod.Camera(camera_mod.CameraConfig(
            name="high", exposure=1100, autoexposure=False, **kw
        ))

    def test_it_will_not_climb_past_the_ceiling(self, monkeypatch):
        cam = self._cam(exposure_min=700, exposure_max=1400)
        monkeypatch.setattr(cam, "_v4l2_set", lambda *a: True)
        cam.set_exposure(1610)
        assert cam.cfg.exposure == 1400

    def test_it_will_not_fall_below_the_floor(self, monkeypatch):
        cam = self._cam(exposure_min=700, exposure_max=1400)
        monkeypatch.setattr(cam, "_v4l2_set", lambda *a: True)
        cam.set_exposure(100)
        assert cam.cfg.exposure == 700

    def test_sitting_at_a_limit_reports_no_change(self, monkeypatch):
        """Otherwise the pipeline resets the background every pass forever."""
        cam = self._cam(exposure_min=700, exposure_max=1400)
        monkeypatch.setattr(cam, "_v4l2_set", lambda *a: True)
        cam.set_exposure(5000)
        assert cam.set_exposure(5000) is False

    def test_unbounded_by_default(self, monkeypatch):
        cam = self._cam()
        monkeypatch.setattr(cam, "_v4l2_set", lambda *a: True)
        cam.set_exposure(3000)
        assert cam.cfg.exposure == 3000


def shadow_on_board(camera_xyz, point_xyz):
    """Where a point above the board projects to, seen from a camera.

    This is exactly what a homography does to an off-plane point: casts it onto
    the board from the camera's viewpoint. Building the test this way means the
    inputs are the real geometry rather than a restatement of the algorithm.
    """
    cx, cy, cz = camera_xyz
    px, py, pz = point_xyz
    t = cz / (cz - pz)          # travel from camera until z hits 0
    return (cx + t * (px - cx), cy + t * (py - cy))


class TestTipByCrossingViews:
    """The tip is where the two cameras' dart-lines cross.

    A dart stands out of the board, so only its tip is *in* the board plane. A
    homography maps a point at z=0 to where it truly is and everything above the
    plane somewhere else, so each camera's view of the dart lands on the board
    as a line -- the dart's shadow from that viewpoint. Both shadows pass
    through the tip, because the tip is its own shadow. So the tip is the
    crossing point, and no decision about which end of the blob is the point is
    needed anywhere.
    """

    LOW = (-700.0, -450.0, 900.0)    # roughly where the low camera sits, in mm
    HIGH = (-700.0, 500.0, 1400.0)   # and the overhead one

    def _views(self, tip_mm, flight_mm, cams=None):
        """Both cameras' board-plane lines for one dart."""
        tip = (tip_mm[0], tip_mm[1], 0.0)
        lines = []
        for cam in (cams or (self.LOW, self.HIGH)):
            lines.append((shadow_on_board(cam, tip), shadow_on_board(cam, flight_mm)))
        return lines

    def test_it_finds_the_tip_of_a_dart_standing_out_of_the_board(self):
        tip = (40.0, 95.0)                     # a treble 20, near enough
        flight = (55.0, 130.0, 38.0)           # 38mm out of the board
        found, sin_angle = detect.tip_from_lines(self._views(tip, flight))
        assert found == pytest.approx(tip, abs=0.5)
        assert sin_angle > 0

    def test_it_does_not_care_which_end_the_detector_called_the_tip(self):
        """The failure that produced 90-200mm errors becomes irrelevant."""
        tip, flight = (-60.0, -30.0), (-95.0, -70.0, 35.0)
        lines = self._views(tip, flight)
        swapped = [(b, a) for a, b in lines]   # both cameras got it backwards
        found, _ = detect.tip_from_lines(swapped)
        assert found == pytest.approx(tip, abs=0.5)

    def test_it_survives_a_ragged_silhouette(self):
        """Only the line matters, not where along it the endpoints landed.

        Half of each dart vanishes against same-brightness sectors, so the
        endpoints are unreliable -- but a line fitted through them is not.
        """
        tip, flight = (100.0, -40.0), (135.0, -75.0, 40.0)
        lines = []
        for cam in (self.LOW, self.HIGH):
            a = shadow_on_board(cam, (tip[0], tip[1], 0.0))
            b = shadow_on_board(cam, flight)
            # Keep only the middle of the dart: both ends chopped off.
            mid1 = (a[0] + 0.3 * (b[0] - a[0]), a[1] + 0.3 * (b[1] - a[1]))
            mid2 = (a[0] + 0.8 * (b[0] - a[0]), a[1] + 0.8 * (b[1] - a[1]))
            lines.append((mid1, mid2))
        found, _ = detect.tip_from_lines(lines)
        assert found == pytest.approx(tip, abs=1.0)

    def test_a_dart_flat_against_the_board_still_works(self):
        """Barely any protrusion means the lines nearly coincide."""
        tip, flight = (0.0, 60.0), (20.0, 95.0, 3.0)
        out = detect.tip_from_lines(self._views(tip, flight))
        if out is not None:
            assert out[0] == pytest.approx(tip, abs=3.0)

    def test_two_cameras_in_the_same_place_cannot_solve_it(self):
        """Nearly the same viewpoint means nearly the same shadow line.

        Both cameras here are on the left of the board, so this guard is not
        hypothetical -- it is the case that has to fail honestly rather than
        return a confident number pulled out of a shallow crossing.
        """
        near = (-700.0, -450.0, 905.0)
        lines = self._views((40.0, 95.0), (55.0, 130.0, 38.0), cams=(self.LOW, near))
        assert detect.tip_from_lines(lines) is None

    def test_one_camera_is_not_enough(self):
        assert detect.tip_from_lines([((0.0, 0.0), (10.0, 10.0))]) is None

    def test_a_blob_with_no_extent_defines_no_line(self):
        lines = [((5.0, 5.0), (5.0, 5.0)), ((0.0, 0.0), (10.0, 3.0))]
        assert detect.tip_from_lines(lines) is None

    def test_the_reported_angle_reflects_how_squarely_they_cross(self):
        square = detect.tip_from_lines(
            [((0.0, -50.0), (0.0, 50.0)), ((-50.0, 0.0), (50.0, 0.0))]
        )
        assert square[1] == pytest.approx(1.0, abs=1e-6)
