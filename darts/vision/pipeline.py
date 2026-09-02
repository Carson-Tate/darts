"""The vision loop: watch the board, emit darts and turn boundaries.

State machine
-------------
    CALIBRATING -> IDLE -> SETTLING -> (score dart) -> IDLE
                     |
                     +--> HAND -> (board goes empty) -> darts-removed -> IDLE

The one non-obvious trick: after each scored dart the background is
re-baselined to *include* that dart. Every detection pass is then a
single-new-blob problem, so there's no bookkeeping about which of three
overlapping shapes is new, and a dart that occludes an earlier one degrades to
"missed this dart" rather than "rescored the previous one".

Nothing here knows about HTTP or audio; it calls the callbacks it was given.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..board import BoardGeometry, Hit, REGULATION, score_at
from . import detect
from .calibrate import (
    RECT_SIZE,
    Calibration,
    auto_calibrate,
    debug_overlay,
    orient_to_template,
    yellow_mask,
)
from .camera import Camera
from .detect import BackgroundModel, DetectorConfig

log = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    dart_min_mass: int = 300  # changed px that counts as "something landed"
    hand_mass: int = 25_000  # changed px that means a person is in shot
    quiet_mass: int = 200  # changed px that counts as "board is back to empty"
    stable_frames: int = 3  # consecutive steady frames before measuring
    stable_tolerance: float = 0.18  # allowed frame-to-frame mass wobble
    settle_timeout_s: float = 2.5
    recalibrate_every_s: float = 0.0  # 0 disables periodic recalibration
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    geom: BoardGeometry = REGULATION
    yellow: object | None = None  # YellowRange override for the board colour
    # Where confirmed-orientation board snapshots live. Persisting them is what
    # makes a manual rotation survive recalibration and restarts.
    template_dir: Path = Path("calibration")
    # Set to a directory to save what the detector saw for each dart. Off in
    # normal play; invaluable for arguing about which end is the point.
    debug_dir: Path | None = None
    debug_max_dumps: int = 40  # /tmp is a tmpfs on a Pi; an uncapped dump eats RAM


@dataclass
class DartEvent:
    hit: Hit
    confidence: float
    per_camera: dict[str, tuple[float, float]]  # camera name -> board mm


class VisionPipeline:
    def __init__(
        self,
        cameras: list[Camera],
        config: PipelineConfig | None = None,
        on_dart: Callable[[DartEvent], None] | None = None,
        on_darts_removed: Callable[[], None] | None = None,
        on_status: Callable[[dict], None] | None = None,
    ):
        self.cameras = cameras
        self.cfg = config or PipelineConfig()
        self.on_dart = on_dart or (lambda e: None)
        self.on_darts_removed = on_darts_removed or (lambda: None)
        self.on_status = on_status or (lambda s: None)

        self.calibrations: dict[str, Calibration] = {}
        # Board snapshots at a user-confirmed orientation, per camera. These are
        # what stop every recalibration from re-rolling the rotation.
        self.templates: dict[str, np.ndarray] = {}
        self.backgrounds: dict[str, BackgroundModel] = {
            c.cfg.name: BackgroundModel() for c in cameras
        }
        self.latest: dict[str, np.ndarray] = {}
        self.state = "starting"
        self.darts_in_board = 0

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._recalibrate_requested = threading.Event()

    # ---- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if not self.cameras:
            log.warning("vision pipeline not started: no cameras")
            self._set_state("disabled")
            return
        self._load_templates()
        self._thread = threading.Thread(target=self._run, name="vision", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        for cam in self.cameras:
            cam.release()

    def request_recalibration(self) -> None:
        """Ask for a fresh calibration. The board must be empty."""
        self._recalibrate_requested.set()

    def reset_background(self) -> None:
        """Forget the current darts and re-learn the empty board."""
        with self._lock:
            for bg in self.backgrounds.values():
                bg.reset()
            self.darts_in_board = 0

    def nudge_rotation(self, sectors: int = 1) -> None:
        """Rotate every calibration by whole sectors, and remember the result.

        Needed because the ring pattern is 36-degree symmetric and the numerals
        that break that tie are only a moderately strong signal. If the overlay
        shows the numbers in the wrong place, this is the fix.

        Rotating also *saves a template*. Without that, every recalibration
        re-rolls the orientation from ten equally-plausible candidates and
        discards the correction -- so tapping Recalibrate after moving the
        camera silently threw away the rotation the user had just dialled in,
        and they had to redo it every time. A rotation is the user stating the
        truth, so it is worth keeping.
        """
        with self._lock:
            self.calibrations = {
                name: calib.rotated(sectors) for name, calib in self.calibrations.items()
            }
        log.info("calibration rotated by %+d sector(s)", sectors)
        self._save_templates()
        self.on_status(self.status())

    # ---- learned orientation ----------------------------------------------

    def _template_path(self, name: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        return self.cfg.template_dir / f"{safe}.png"

    def _load_templates(self) -> None:
        for cam in self.cameras:
            path = self._template_path(cam.cfg.name)
            if not path.is_file():
                continue
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                self.templates[cam.cfg.name] = img
                log.info("loaded orientation template for %s", cam.cfg.name)

    def _save_templates(self) -> None:
        """Snapshot each camera's board at the orientation now in force."""
        with self._lock:
            calibs = dict(self.calibrations)
            frames = dict(self.latest)
        for name, calib in calibs.items():
            frame = frames.get(name)
            if frame is None:
                continue
            mask = yellow_mask(frame, self.cfg.yellow, clean=False)
            rect = cv2.warpPerspective(mask, calib.h_img2rect, (RECT_SIZE, RECT_SIZE))
            self.templates[name] = rect
            try:
                self.cfg.template_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(self._template_path(name)), rect)
            except OSError as exc:
                log.warning("could not save orientation template for %s: %s", name, exc)

    def forget_orientation(self) -> None:
        """Drop the learned templates and go back to guessing from the numerals."""
        self.templates.clear()
        for cam in self.cameras:
            self._template_path(cam.cfg.name).unlink(missing_ok=True)
        log.info("orientation templates cleared")
        self.on_status(self.status())

    # ---- main loop ---------------------------------------------------------

    def _run(self) -> None:
        """Keep the loop alive.

        An unhandled exception in a thread kills only that thread, silently: the
        server stays up, the UI stays connected, and the camera preview just
        stops updating forever. That is a much worse failure than dropping a
        frame, so anything unexpected is logged and the loop is restarted.
        """
        while not self._stop.is_set():
            try:
                self._loop()
            except Exception:
                log.exception("vision loop crashed; restarting it")
                time.sleep(1.0)

    def _loop(self) -> None:
        self._set_state("calibrating")
        last_calibration = 0.0
        stable_run = 0
        prev_mass = 0
        settle_started = 0.0
        saw_hand = False

        while not self._stop.is_set():
            frames = self._grab()
            if not frames:
                time.sleep(0.05)
                continue

            now = time.monotonic()

            # -- calibration ------------------------------------------------
            needs_calib = (
                not self.calibrations
                or self._recalibrate_requested.is_set()
                or (
                    self.cfg.recalibrate_every_s
                    and now - last_calibration > self.cfg.recalibrate_every_s
                )
            )
            if needs_calib:
                self._recalibrate_requested.clear()
                if self._calibrate(frames):
                    last_calibration = now
                    self.reset_background()
                else:
                    time.sleep(0.5)
                    continue

            # -- background -------------------------------------------------
            primary = self.cameras[0].cfg.name
            for name, frame in frames.items():
                self.backgrounds[name].add(detect.preprocess(frame))

            # Bind the array once. Checking .ready and then reading .background
            # is a race: reset_background() runs on the web thread (Next Player
            # calls it) and nulls the array in between, which crashed the whole
            # vision thread and froze the camera for the rest of the session.
            background = self.backgrounds[primary].background
            if background is None:
                if all(bg.commit() for bg in self.backgrounds.values()):
                    self._set_state("idle")
                continue

            gray = detect.preprocess(frames[primary])
            mass = detect.change_mass(gray, background, self.cfg.detector)

            # -- hand / removal ---------------------------------------------
            if mass > self.cfg.hand_mass:
                saw_hand = True
                stable_run = 0
                self._set_state("hand")
                continue

            if saw_hand and mass < self.cfg.quiet_mass:
                # Person stepped away and the board is bare again: turn is over.
                saw_hand = False
                log.info("darts removed (mass fell to %d)", mass)
                self.reset_background()
                self.darts_in_board = 0
                self._set_state("idle")
                self.on_darts_removed()
                continue

            if saw_hand:
                # Person still near the board; don't try to score anything.
                continue

            # -- dart landing -----------------------------------------------
            if mass < self.cfg.dart_min_mass:
                stable_run = 0
                if self.state != "idle":
                    self._set_state("idle")
                continue

            if self.state != "settling":
                self._set_state("settling")
                settle_started = now
                stable_run = 0

            tolerance = max(self.cfg.stable_tolerance * max(prev_mass, 1), 40)
            stable_run = stable_run + 1 if abs(mass - prev_mass) <= tolerance else 0
            prev_mass = mass

            if stable_run >= self.cfg.stable_frames:
                self._measure(frames)
                stable_run = 0
                self._set_state("idle")
            elif now - settle_started > self.cfg.settle_timeout_s:
                log.debug("settle timed out at mass %d; ignoring", mass)
                self._set_state("idle")
                stable_run = 0

    # ---- steps -------------------------------------------------------------

    def _grab(self) -> dict[str, np.ndarray]:
        frames: dict[str, np.ndarray] = {}
        for cam in self.cameras:
            frame = cam.read()
            if frame is not None:
                frames[cam.cfg.name] = frame
        with self._lock:
            self.latest = frames
        return frames

    def _calibrate(self, frames: dict[str, np.ndarray]) -> bool:
        found: dict[str, Calibration] = {}
        for name, frame in frames.items():
            calib = auto_calibrate(frame, self.cfg.geom, self.cfg.yellow)
            if calib is None:
                log.warning("camera %s: calibration failed this attempt", name)
                continue
            # If this board's orientation has been confirmed before, take it from
            # the template rather than from the numerals. The geometry from
            # auto_calibrate is good; it is only the choice among the ten
            # symmetry-equivalent rotations that is unreliable.
            template = self.templates.get(name)
            if template is not None:
                mask = yellow_mask(frame, self.cfg.yellow, clean=False)
                h, score, margin = orient_to_template(
                    mask, calib.h_img2rect, template, self.cfg.geom
                )
                calib = replace(calib, h_img2rect=h, score=score, margin=margin)
                log.info(
                    "camera %s: orientation from template (match %.3f, margin %.3f)",
                    name, score, margin,
                )
            found[name] = calib

        # The primary camera must calibrate; a secondary that fails is dropped
        # to single-camera operation rather than blocking the game.
        primary = self.cameras[0].cfg.name
        if primary not in found:
            self._set_state("calibration-failed")
            return False

        self.calibrations = found
        self._set_state("calibrated")
        self.on_status(self.status())
        return True

    def _pick_tip(self, blob, calib) -> tuple[float, float]:
        """Choose which end of the blob is the dart's point.

        The detector guesses from the silhouette's taper, which is a weak cue
        when the dart points towards the camera: measured on this board it took
        the flight end on 4 of 13 throws, putting the score 280mm out on a
        170mm board and reporting a miss for a dart that was in the 17.

        The point is embedded in the board and the flight sticks out of it, so
        when exactly one end lands on the board, that end is the point. This is
        a constraint rather than a heuristic, and it needs the calibration the
        detector does not have. When both ends land on the board -- a dart lying
        nearly flat to the face -- the taper cue is all there is, so keep it.
        """
        on_board = self.cfg.geom.double_outer

        def radius(pt):
            x_mm, y_mm = calib.image_to_board(*pt)
            return float(np.hypot(x_mm, y_mm))

        r_tip = radius(blob.tip)
        r_other = radius(blob.other_end)
        if r_tip > on_board >= r_other:
            log.info(
                "tip: taking the other end (%.0fmm on the board, not %.0fmm off it)",
                r_other, r_tip,
            )
            return blob.other_end
        return blob.tip

    def _dump_blob(self, name, frame, gray, background, blob, tip=None) -> None:
        """Save what the detector saw and where it put the tip.

        Which end of a dart is the point cannot be argued about from a score
        that came out wrong -- it needs the silhouette. Off by default; set
        vision.debug_dir to turn it on.
        """
        try:
            # Hard cap. /tmp on a Pi is a tmpfs, so an uncapped dump is not
            # writing to disk, it is eating RAM: this ran to 2280 files and
            # 623MB in twenty-five minutes, because the detector triggers far
            # more often than it scores.
            self._dump_seq = getattr(self, "_dump_seq", 0) + 1
            if self._dump_seq > self.cfg.debug_max_dumps:
                return
            d = self.cfg.debug_dir
            d.mkdir(parents=True, exist_ok=True)
            stem = d / f"{name}_{self._dump_seq:03d}"

            diff = cv2.absdiff(gray, background)
            _, mask = cv2.threshold(
                diff, self.cfg.detector.diff_threshold, 255, cv2.THRESH_BINARY
            )
            vis = frame.copy()
            vis[mask > 0] = (0, 0, 255)
            tx, ty = (int(tip[0]), int(tip[1])) if tip else (int(blob.tip[0]), int(blob.tip[1]))
            cx, cy = int(blob.centroid[0]), int(blob.centroid[1])
            cv2.line(vis, (cx, cy), (tx, ty), (255, 255, 0), 1)
            cv2.circle(vis, (cx, cy), 5, (255, 255, 0), 1)   # centroid, cyan
            cv2.circle(vis, (tx, ty), 6, (0, 255, 0), 2)     # chosen tip, green
            # Crop around the blob so the detail survives being shrunk for transfer.
            x0, x1 = max(tx - 170, 0), min(tx + 170, vis.shape[1])
            y0, y1 = max(ty - 130, 0), min(ty + 130, vis.shape[0])
            cv2.imwrite(str(stem) + "_crop.png", vis[y0:y1, x0:x1])
            # The frame and the background it was differenced against, so the
            # diff can be recomputed offline -- including with the two aligned
            # first, which is the question when the mask looks like edges rather
            # than like a dart.
            cv2.imwrite(str(stem) + "_frame.png", gray)
            cv2.imwrite(str(stem) + "_bg.png", background)
            shift, response = cv2.phaseCorrelate(
                background.astype(np.float64), gray.astype(np.float64)
            )
            log.info(
                "frame vs background: shift=(%+.2f,%+.2f)px response=%.3f",
                shift[0], shift[1], response,
            )
            log.info(
                "blob dump %s: tip=(%d,%d) centroid=(%d,%d) area=%.0f elong=%.2f angle=%.0f",
                stem.name, tx, ty, cx, cy, blob.area, blob.elongation, blob.angle_deg,
            )
        except Exception as exc:  # debugging must never break a game
            log.warning("blob dump failed: %s", exc)

    def _measure(self, frames: dict[str, np.ndarray]) -> None:
        points: list[tuple[float, float]] = []
        per_camera: dict[str, tuple[float, float]] = {}

        for name, frame in frames.items():
            calib = self.calibrations.get(name)
            bg = self.backgrounds.get(name)
            if calib is None or bg is None or not bg.ready:
                continue
            gray = detect.preprocess(frame)
            blobs = detect.find_darts(
                gray, bg.background, self.cfg.detector, self.darts_in_board
            )
            if not blobs:
                continue
            tip = self._pick_tip(blobs[0], calib)
            if self.cfg.debug_dir is not None:
                self._dump_blob(name, frame, gray, bg.background, blobs[0], tip)
            x_mm, y_mm = calib.image_to_board(*tip)
            # Reject anything that maps well outside the board -- usually a
            # shadow on the cabinet frame or a dart that bounced onto the floor.
            if np.hypot(x_mm, y_mm) > self.cfg.geom.double_outer * 1.35:
                log.debug("camera %s: blob maps off-board, ignoring", name)
                continue
            points.append((x_mm, y_mm))
            per_camera[name] = (x_mm, y_mm)

        if not points:
            log.debug("settled change but no dart-shaped blob found")
            return

        (x_mm, y_mm), confidence = detect.fuse(points)
        hit = score_at(x_mm, y_mm, self.cfg.geom)
        self.darts_in_board += 1

        # Fold the new dart into the background so the next one is the only
        # new thing in frame.
        for name, frame in frames.items():
            bg = self.backgrounds[name]
            bg.reset()
            for _ in range(bg.frames):
                bg.add(detect.preprocess(frame))
            bg.commit()

        log.info(
            "dart: %s (%.1f, %.1f) mm from %d camera(s), confidence %.2f",
            hit.label, x_mm, y_mm, len(points), confidence,
        )
        self.on_dart(DartEvent(hit, confidence, per_camera))

    # ---- introspection -----------------------------------------------------

    def _set_state(self, state: str) -> None:
        if state != self.state:
            self.state = state
            log.debug("vision state -> %s", state)
            self.on_status(self.status())

    def status(self) -> dict:
        return {
            "state": self.state,
            "cameras": [c.cfg.name for c in self.cameras],
            "calibrated": sorted(self.calibrations),
            "calibration_scores": {
                n: round(c.score, 3) for n, c in self.calibrations.items()
            },
            "rotation_confident": all(
                c.rotation_is_confident for c in self.calibrations.values()
            ) if self.calibrations else False,
            "darts_in_board": self.darts_in_board,
        }

    def preview_jpeg(self, camera: str | None = None, overlay: bool = True) -> bytes | None:
        """A single JPEG for the web UI's camera preview."""
        import cv2

        with self._lock:
            frames = dict(self.latest)
        if not frames:
            return None
        name = camera or self.cameras[0].cfg.name
        frame = frames.get(name)
        if frame is None:
            return None
        calib = self.calibrations.get(name)
        if overlay and calib is not None:
            frame = debug_overlay(frame, calib)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return buf.tobytes() if ok else None
