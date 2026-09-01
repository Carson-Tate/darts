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
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..board import BoardGeometry, Hit, REGULATION, score_at
from . import detect
from .calibrate import Calibration, auto_calibrate, debug_overlay
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
        """Rotate every calibration by whole sectors.

        Needed because the ring pattern is 36-degree symmetric and the numerals
        that break that tie are only a moderately strong signal. If the overlay
        shows the numbers in the wrong place, this is the fix.
        """
        with self._lock:
            self.calibrations = {
                name: calib.rotated(sectors) for name, calib in self.calibrations.items()
            }
        log.info("calibration rotated by %+d sector(s)", sectors)
        self.on_status(self.status())

    # ---- main loop ---------------------------------------------------------

    def _run(self) -> None:
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
            if not self.backgrounds[primary].ready:
                if all(bg.commit() for bg in self.backgrounds.values()):
                    self._set_state("idle")
                continue

            gray = detect.preprocess(frames[primary])
            mass = detect.change_mass(
                gray, self.backgrounds[primary].background, self.cfg.detector
            )

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
            if calib is not None:
                found[name] = calib
            else:
                log.warning("camera %s: calibration failed this attempt", name)

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
            tip = blobs[0].tip
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
