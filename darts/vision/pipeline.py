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

import itertools
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
    # How often to re-offer calibration to a camera that hasn't managed it yet.
    # Only ever runs with an empty board, and never disturbs a camera that has
    # already calibrated.
    straggler_retry_s: float = 20.0
    # How far past the double ring an end has to sit before it counts as "off
    # the board" and the other end is taken as the point. 1.0 would mean the
    # wire itself, which calibration error alone can straddle. See _pick_tip.
    off_board_slack: float = 1.15
    # Spread between cameras, in mm, past which averaging them is worse than
    # picking one. See detect.fuse.
    trust_one_camera_mm: float = 25.0
    # How closely two cameras must land on the same end of a dart before that
    # agreement is taken as having found the point. See _agree_on_an_end.
    agree_mm: float = 14.0
    # Give up waiting for a still scene to baseline against after this long and
    # take whatever is in front of the camera. A busy room must not mean no
    # scoreboard at all.
    baseline_wait_s: float = 6.0
    # Force a re-baseline after this long stuck at "hand". See _loop.
    hand_timeout_s: float = 12.0
    # How far off the board a dart-shaped blob may land and still be scored as
    # a miss rather than ignored. Matches the detection ROI; beyond it, a blob
    # is something else in the room, and a phantom dart costs a real one.
    miss_reach: float = 1.6
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
        # Where the board is in each camera's frame. Rebuilt on calibration and
        # used to keep the rest of the room out of dart detection.
        self.rois: dict[str, np.ndarray] = {}
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

    def nudge_rotation(self, sectors: int = 1, camera: str | None = None) -> None:
        """Rotate a calibration by whole sectors, and remember the result.

        Needed because the ring pattern is 36-degree symmetric and the numerals
        that break that tie are only a moderately strong signal. If the overlay
        shows the numbers in the wrong place, this is the fix.

        `camera` names one to rotate; None rotates all of them. Per-camera
        matters as soon as there are two, because each resolves the symmetry
        independently against its own view -- one can lock onto the right
        orientation while the other is two sectors out, and rotating both
        together can only ever fix one of them.

        `sectors=0` changes nothing and saves the template anyway: that is how
        you confirm an orientation the system got right but is not confident
        about. The overhead camera locked correctly here with a margin of 0.008,
        and without this the only way to say so was to tap Rotate twenty times
        back to where it started.

        Rotating also *saves a template*. Without that, every recalibration
        re-rolls the orientation from ten equally-plausible candidates and
        discards the correction -- so tapping Recalibrate after moving the
        camera silently threw away the rotation the user had just dialled in,
        and they had to redo it every time. A rotation is the user stating the
        truth, so it is worth keeping.
        """
        with self._lock:
            if camera is not None and camera not in self.calibrations:
                log.warning("cannot rotate unknown/uncalibrated camera %r", camera)
                return
            self.calibrations = {
                name: calib.rotated(sectors) if camera in (None, name) else calib
                for name, calib in self.calibrations.items()
            }
        who = camera or "every camera"
        if sectors:
            log.info("rotated %s by %+d sector(s)", who, sectors)
        else:
            log.info("orientation confirmed for %s", who)
        self._save_templates(None if camera is None else {camera})
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

    def _save_templates(self, only: set[str] | None = None) -> None:
        """Snapshot a camera's board at the orientation now in force.

        `only` limits it to the cameras the user actually spoke about. Saving
        all of them was wrong once there were two: confirming the overhead
        camera's orientation also recorded the other camera's as truth, so a
        wrong lock on the one you *weren't* looking at got written down and
        faithfully restored on every later calibration.
        """
        with self._lock:
            calibs = dict(self.calibrations)
            frames = dict(self.latest)
        for name, calib in calibs.items():
            if only is not None and name not in only:
                continue
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
        calib_failures = 0
        last_straggler = 0.0
        baseline_started = 0.0
        hand_since = 0.0
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
                    calib_failures = 0
                    self.reset_background()
                else:
                    # Back off. When calibration cannot succeed at all -- the
                    # room lights are off, say -- retrying twice a second just
                    # burns CPU and writes a warning per attempt all night. It
                    # logged 205 of them in six minutes.
                    calib_failures += 1
                    time.sleep(min(0.5 * calib_failures, 15.0))
                    continue

            # -- a camera that missed the first attempt ----------------------
            # Keep offering the stragglers another go. The overhead camera runs
            # its own auto-exposure against a scene with a bright doorway in it
            # and is still settling when the first attempt happens: it scored
            # 0.27 against a 0.35 gate at startup and 0.74 on a frame taken a
            # minute later. Without this it stayed dropped for the whole
            # session, because once the primary calibrates `needs_calib` is
            # false and nothing ever asks again.
            missing = [
                c.cfg.name for c in self.cameras if c.cfg.name not in self.calibrations
            ]
            if (
                self.calibrations
                and missing
                and self.darts_in_board == 0
                and now - last_straggler > self.cfg.straggler_retry_s
            ):
                last_straggler = now
                self._calibrate(frames, only=set(missing))

            # -- background -------------------------------------------------
            primary = self.cameras[0].cfg.name
            if primary not in frames:
                # The secondary camera delivered and the primary didn't. Every
                # step below is keyed off the primary, so carry on to the next
                # grab rather than indexing a frame that isn't there.
                continue
            for name, frame in frames.items():
                self.backgrounds[name].add(detect.preprocess(frame))

            # Bind the array once. Checking .ready and then reading .background
            # is a race: reset_background() runs on the web thread (Next Player
            # calls it) and nulls the array in between, which crashed the whole
            # vision thread and froze the camera for the rest of the session.
            background = self.backgrounds[primary].background
            if background is None:
                # A list, not a generator: all() short-circuits, so with two
                # cameras a not-yet-full primary buffer would stop the secondary
                # from ever committing its own background, and it would sit
                # un-ready forever while the primary looked fine.
                #
                # quiet_px refuses to baseline while someone is still moving in
                # shot. Tapping Next Player and walking straight up to the board
                # used to bake the player into the background, after which
                # nothing was ever scored again until Next Player was tapped a
                # second time.
                if baseline_started == 0.0:
                    baseline_started = now
                quiet = 0 if now - baseline_started > self.cfg.baseline_wait_s else self.cfg.quiet_mass
                committed = [
                    bg.commit(self.cfg.detector, quiet) for bg in self.backgrounds.values()
                ]
                if all(committed):
                    baseline_started = 0.0
                    self._set_state("idle")
                continue
            baseline_started = 0.0

            gray = detect.preprocess(frames[primary])
            mass = detect.change_mass(gray, background, self.cfg.detector)

            # -- hand / removal ---------------------------------------------
            if mass > self.cfg.hand_mass:
                if not saw_hand:
                    hand_since = now
                saw_hand = True
                stable_run = 0
                self._set_state("hand")
                # Belt and braces for the lock-up above. Leaving the hand state
                # needs the frame to go quiet against the background, so any
                # background that disagrees with an empty board forever -- for
                # whatever reason, not just a person baked into it -- parks the
                # pipeline here permanently and silently. Nobody can debug
                # "it stopped counting"; it just has to fix itself.
                if now - hand_since > self.cfg.hand_timeout_s:
                    log.warning(
                        "stuck at 'hand' for %.0fs with mass %d; the background "
                        "must be wrong, re-learning it",
                        now - hand_since, mass,
                    )
                    self.reset_background()
                    saw_hand = False
                    hand_since = now
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
            # Merge rather than replace: a camera that drops a single frame
            # should not blank its tile on the website, which now shows both
            # continuously. Detection still only ever sees `frames`, so a stale
            # picture can be looked at but never scored from.
            self.latest.update(frames)
        return frames

    def _calibrate(
        self, frames: dict[str, np.ndarray], only: set[str] | None = None
    ) -> bool:
        """Calibrate the cameras in `frames`, or just those named in `only`.

        `only` is the retry path for a camera that missed the first attempt. It
        merges into the existing calibrations instead of replacing them, so a
        working primary is never torn down to have another go at a secondary.
        """
        found: dict[str, Calibration] = {}
        for name, frame in frames.items():
            if only is not None and name not in only:
                continue
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

        rois = {
            name: calib.board_mask(frames[name].shape)
            for name, calib in found.items()
            if name in frames
        }

        if only is not None:
            if not found:
                return False
            self.calibrations = {**self.calibrations, **found}
            self.rois = {**self.rois, **rois}
            log.info("camera(s) %s joined late", ", ".join(sorted(found)))
            self.on_status(self.status())
            return True

        # The primary camera must calibrate; a secondary that fails is dropped
        # to single-camera operation rather than blocking the game.
        primary = self.cameras[0].cfg.name
        if primary not in found:
            self._set_state("calibration-failed")
            return False

        self.calibrations = found
        # A rotation about the board centre leaves this circle where it is, so
        # nudge_rotation does not invalidate it -- only a fresh calibration does.
        self.rois = rois
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

        The end being rejected has to be *clearly* off the board, not merely
        past the wire. A dart in the double reads a whisker outside a board
        whose calibration is a whisker small, and flipping on that reading is
        catastrophic rather than marginal: it does not move the score to the
        neighbouring sector, it moves it the full length of the dart, to the
        other side of the board. Measured here, a tip at 172mm against a 170mm
        board flipped to the far end at 17mm and scored S18 for a dart in the
        12. A real flight end is nothing like that close -- the genuine flips
        in the same session read 261mm and 172mm-against-63mm -- so requiring
        clear daylight separates the two cases without giving up the rule.
        """
        on_board = self.cfg.geom.double_outer
        clearly_off = on_board * self.cfg.off_board_slack

        def radius(pt):
            x_mm, y_mm = calib.image_to_board(*pt)
            return float(np.hypot(x_mm, y_mm))

        r_tip = radius(blob.tip)
        r_other = radius(blob.other_end)
        # Not a chained comparison: only the *rejected* end gets the slack. The
        # end being kept still has to be genuinely on the board, or a blob with
        # both ends outside gets flipped to whichever is less far out.
        if r_tip > clearly_off and r_other <= on_board:
            log.info(
                "tip: taking the other end (%.0fmm on the board, not %.0fmm off it)",
                r_other, r_tip,
            )
            return blob.other_end
        if r_tip > on_board >= r_other:
            log.info(
                "tip: keeping the taper's end at %.0fmm -- only %.0fmm past the "
                "board, too close to call it a flight (other end %.0fmm)",
                r_tip, r_tip - on_board, r_other,
            )
        return blob.tip

    def _agree_on_an_end(self, ends) -> dict[str, tuple[float, float]] | None:
        """Pick one end per camera so that the cameras land on the same spot.

        This is the thing two cameras can do that one cannot, and it replaces a
        guess with a measurement.

        Each camera sees a dart as a line with two ends and has to say which is
        the point. The taper cue that decides it is weak, and it fails hardest
        on the overhead camera, which looks down the length of a dart pointing
        up at it. Measured over eleven throws, the low camera and the overhead
        camera placed the same dart 91, 140, 141, 131, 146, 131, 198, 146 and
        93mm apart -- around a dart length, every time. They were tracking
        opposite ends of it.

        The way out is that the two ends are not geometrically alike. Every
        homography here maps the image to the *board plane*. The point is in
        that plane, so both cameras project it to the same millimetres. The
        flight stands 30-40mm out of it, so the two cameras project it to
        different millimetres -- that is parallax, and it is exactly the error
        this whole file works around elsewhere. So of the four ways to pair up
        two cameras' ends, the pairing that agrees is the pairing that found
        the point. Nothing about taper or silhouette is needed.

        Returns None when there is nothing to arbitrate (one camera) or when no
        pairing agrees well enough to be believed, leaving the taper cue's
        answer alone rather than replacing it with a worse one.
        """
        if len(ends) < 2:
            return None

        names = list(ends)
        limit = self.cfg.geom.double_outer * 1.35
        best = None
        for combo in itertools.product(*(ends[n] for n in names)):
            arr = np.array(combo, np.float64)
            spread = float(np.max(np.linalg.norm(arr - arr.mean(axis=0), axis=1)))
            # A pairing that agrees off the edge of the board is agreement
            # about something that is not a dart in the board; rank those last
            # rather than excluding them, so there is always an answer.
            off = sum(1 for p in combo if float(np.hypot(*p)) > limit)
            if best is None or (off, spread) < best[0]:
                best = ((off, spread), combo)

        (off, spread), combo = best
        if off or spread > self.cfg.agree_mm:
            # Logged at info, and with the numbers, because agree_mm is a
            # guess until there is data behind it: this line is how the
            # threshold gets set from what the cameras actually achieve rather
            # than from what seemed reasonable.
            log.info(
                "no pairing of ends agreed (best %.0fmm, %d off-board, "
                "tolerance %.0fmm); keeping the taper's pick | %s",
                spread, off, self.cfg.agree_mm,
                {n: [(round(p[0]), round(p[1])) for p in ends[n]] for n in names},
            )
            return None
        chosen = dict(zip(names, (tuple(map(float, p)) for p in combo)))
        log.info("ends agreed to %.0fmm: %s", spread, chosen)
        return chosen

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
        # Primary first, so that when the cameras cannot be reconciled fuse()
        # falls back to a defined one rather than to dict ordering. Reorders
        # without dropping: a frame from a camera not in self.cameras is not
        # something to discard silently, it just has no claim to be preferred.
        order = [c.cfg.name for c in self.cameras]
        ordered = {n: frames[n] for n in order if n in frames}
        ordered.update(frames)
        frames = ordered

        points: list[tuple[float, float]] = []
        per_camera: dict[str, tuple[float, float]] = {}
        ends: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {}

        for name, frame in frames.items():
            calib = self.calibrations.get(name)
            bg = self.backgrounds.get(name)
            if calib is None or bg is None or not bg.ready:
                continue
            gray = detect.preprocess(frame)
            blobs = detect.find_darts(
                gray, bg.background, self.cfg.detector, self.darts_in_board,
                roi=self.rois.get(name),
            )
            if not blobs:
                continue

            # Take the largest blob that actually lands on the board, rather
            # than the largest blob outright. They used to be the same thing
            # with one camera pointed at nothing else; the overhead camera also
            # sees a doorway and the dart holders on the cabinet doors, and an
            # arm reaching for a dart is bigger than a dart. Stopping at the
            # first blob meant an off-board reject cost us the throw entirely
            # on that camera, instead of falling through to the real dart.
            limit = self.cfg.geom.double_outer * 1.35
            chosen = None
            off_board = None
            for blob in blobs:
                tip = self._pick_tip(blob, calib)
                x_mm, y_mm = calib.image_to_board(*tip)
                r_mm = float(np.hypot(x_mm, y_mm))
                if r_mm <= limit:
                    chosen = (blob, tip, x_mm, y_mm)
                    break
                # Bound the miss fallback here rather than leaning on the ROI
                # to have excluded it. The ROI is only present once a camera has
                # calibrated, so relying on it alone means an uncalibrated or
                # freshly-restarted camera would call a blob anywhere in frame a
                # miss -- and a phantom dart costs a real one out of the turn.
                if off_board is None and r_mm <= self.cfg.geom.double_outer * self.cfg.miss_reach:
                    off_board = (blob, tip, x_mm, y_mm)
            if chosen is None:
                # A dart that missed the board is still a dart thrown, and
                # dropping it silently meant a miss simply never appeared on
                # the scoreboard -- reported as "a couple of misses didn't
                # count". The blob has already passed the dart-shape filters
                # and landed inside the board ROI, so calling it a miss is a
                # smaller claim than calling it a score.
                if off_board is None:
                    log.debug("camera %s: %d blob(s), none usable", name, len(blobs))
                    continue
                chosen = off_board
                log.info(
                    "camera %s: dart-shaped blob %.0fmm out; scoring it a miss",
                    name, float(np.hypot(off_board[2], off_board[3])),
                )

            blob, tip, x_mm, y_mm = chosen
            if self.cfg.debug_dir is not None:
                self._dump_blob(name, frame, gray, bg.background, blob, tip)
            points.append((x_mm, y_mm))
            per_camera[name] = (x_mm, y_mm)
            ends[name] = (
                calib.image_to_board(*blob.tip),
                calib.image_to_board(*blob.other_end),
            )

        if not points:
            log.debug("settled change but no dart-shaped blob found")
            return

        agreed = self._agree_on_an_end(ends)
        if agreed is not None:
            per_camera = agreed
            points = [agreed[n] for n in frames if n in agreed]

        (x_mm, y_mm), confidence = detect.fuse(
            points, trust_one_mm=self.cfg.trust_one_camera_mm
        )
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

        # Log what each camera said on its own, not just the answer they were
        # boiled down to. Without this a wrong score is uninvestigable: you can
        # see that two cameras produced a dart in the 4 and not whether either
        # of them ever thought so.
        per_cam_text = "  ".join(
            f"{n}={score_at(x, y, self.cfg.geom).label}({x:.0f},{y:.0f})"
            for n, (x, y) in per_camera.items()
        )
        log.info(
            "dart: %s (%.1f, %.1f) mm from %d camera(s), confidence %.2f | %s",
            hit.label, x_mm, y_mm, len(points), confidence, per_cam_text,
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
            # Templates count here too, or this contradicts per_camera below:
            # it read false while every individual camera read true.
            "rotation_confident": all(
                c.rotation_is_confident or n in self.templates
                for n, c in self.calibrations.items()
            ) if self.calibrations else False,
            # Per camera, because with two of them "the rotation is unsure" is
            # not actionable on its own -- the fix is to rotate the one that is
            # wrong, so the UI has to be able to say which.
            #
            # A remembered orientation counts as confident. The margin measures
            # how well the numerals picked one of ten symmetric candidates; a
            # saved template means someone looked at the overlay and said it was
            # right, which is better evidence than the numerals ever were.
            "per_camera": {
                c.cfg.name: {
                    "calibrated": c.cfg.name in self.calibrations,
                    "score": round(self.calibrations[c.cfg.name].score, 3)
                    if c.cfg.name in self.calibrations else None,
                    "remembered": c.cfg.name in self.templates,
                    "rotation_confident": (
                        self.calibrations[c.cfg.name].rotation_is_confident
                        or c.cfg.name in self.templates
                        if c.cfg.name in self.calibrations else False
                    ),
                }
                for c in self.cameras
            },
            "darts_in_board": self.darts_in_board,
        }

    def preview_jpeg(
        self,
        camera: str | None = None,
        overlay: bool = True,
        width: int = 0,
        quality: int = 70,
        crop: bool = True,
    ) -> bytes | None:
        """A single JPEG for the web UI's camera preview.

        `width` shrinks the frame before encoding. The site shows both cameras
        continuously, so this is now two streams rather than one on a link that
        has been the bottleneck all along: a full 1280x720 frame encodes to
        ~150KB, and at any useful frame rate two of those swamp it. A 480px-wide
        tile is about 15KB and still shows plainly whether the overlay lines up
        with the rings, which is all the preview is for.
        """
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
        # Crop after the overlay, which is drawn in full-frame coordinates.
        # Note this is a *display* crop: it does not change what the detector
        # sees (the board ROI already handles that) and it cannot change
        # exposure, which the sensor meters across the whole frame regardless.
        if crop and calib is not None:
            x0, y0, x1, y1 = calib.board_bounds(frame.shape)
            frame = frame[y0:y1, x0:x1]
        if width and 0 < width < frame.shape[1]:
            scale = width / frame.shape[1]
            frame = cv2.resize(
                frame, (int(width), max(int(frame.shape[0] * scale), 1)),
                interpolation=cv2.INTER_AREA,
            )
        ok, buf = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(np.clip(quality, 20, 95))]
        )
        return buf.tobytes() if ok else None
