"""Camera capture. Works with one camera or two; nothing above this layer cares.

USB webcams on a Pi 4 need a bit of coaxing:
  * force MJPEG, or the UVC driver falls back to raw YUYV and two 720p streams
    will not fit down the bus at any usable frame rate;
  * kill autofocus and autoexposure. Background differencing assumes the empty
    board looks the same from frame to frame, and an autoexposure hunt as a
    dart flies past will light up the whole frame as "changed".
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    index: int | str = 0
    name: str = "left"
    # A stable path such as /dev/v4l/by-id/usb-..._Camera_SN0001-video-index0,
    # resolved to a numeric index at open. Preferred over `index`: the kernel
    # hands out video0/video2 in enumeration order, so which webcam is which
    # can swap on any reboot or replug -- and swapping the low and high cameras
    # silently swaps their calibrations, which look fine and score wrongly.
    device: str = ""
    name_hint: str = ""  # substring of a by-id name, if the full path is fiddly
    width: int = 1280
    height: int = 720
    fps: int = 15
    autofocus: bool = False
    focus: float | None = None  # 0-255 on most UVC devices; None leaves it alone
    autoexposure: bool = False
    exposure: float | None = None
    extra: dict = field(default_factory=dict)


BY_ID = "/dev/v4l/by-id"


def _resolve_by_id(hint: str) -> str | None:
    """First capture node whose by-id name contains `hint`.

    Only ...-video-index0 is the capture node; index1 on these webcams is a
    metadata node that opens happily and never yields a picture.
    """
    try:
        names = sorted(os.listdir(BY_ID))
    except OSError:
        return None
    for entry in names:
        if hint.lower() in entry.lower() and entry.endswith("index0"):
            return os.path.join(BY_ID, entry)
    return None


class Camera:
    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self.cap: cv2.VideoCapture | None = None
        self.source: int | str = cfg.index
        self.node: str | None = None

        # Latest frame, published by a reader thread. Two cameras read from one
        # thread serialise on each other's blocking reads -- measured here as
        # 10.3 fps alone against 5.7 each together -- and a camera that stops
        # delivering (this hardware does, occasionally) blocks the loop for the
        # *other* camera too, which turns one dead webcam into no scoring at all.
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._seen = 0
        self._cv = threading.Condition()
        self._reader: threading.Thread | None = None
        self._closing = threading.Event()

    # ---- device resolution -------------------------------------------------

    def _resolve(self) -> None:
        path = self.cfg.device
        if not path and self.cfg.name_hint:
            path = _resolve_by_id(self.cfg.name_hint) or ""
            if not path:
                log.warning(
                    "camera %s: no by-id device matching %r; falling back to index %r",
                    self.cfg.name, self.cfg.name_hint, self.cfg.index,
                )
        if path:
            real = os.path.realpath(path)
            match = re.fullmatch(r"/dev/video(\d+)", real)
            if match:
                self.source = int(match.group(1))
                self.node = real
                log.info("camera %s: %s -> %s", self.cfg.name, path, real)
                return
            log.error(
                "camera %s: %s does not resolve to a video node (got %r)",
                self.cfg.name, path, real,
            )
        self.source = self.cfg.index
        self.node = (
            f"/dev/video{self.cfg.index}" if isinstance(self.cfg.index, int) else None
        )

    def _reopen(self) -> cv2.VideoCapture | None:
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _device_path(self) -> str | None:
        return self.node

    def _v4l2_set(self, control: str, value) -> bool:
        """Set a UVC control through v4l2-ctl.

        Preferred over cap.set() for focus and exposure because OpenCV's V4L2
        backend renegotiates the stream when those properties are written, and
        on some webcams that kills frame delivery outright -- measured on the
        onn 4K here as 0 frames out of 25 with cv2.CAP_PROP_FOCUS set, against
        25 out of 25 without it. v4l2-ctl sets the identical control safely.
        """
        dev = self._device_path()
        if dev is None:
            return False
        try:
            done = subprocess.run(
                ["v4l2-ctl", "-d", dev, "--set-ctrl", f"{control}={value}"],
                capture_output=True, timeout=5,
            )
            return done.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    def _usb_port(self) -> str | None:
        """The USB port id (e.g. "1-1.4") backing this video node, if any."""
        dev = self._device_path()
        if dev is None:
            return None
        try:
            real = os.path.realpath(f"/sys/class/video4linux/{os.path.basename(dev)}/device")
        except OSError:
            return None
        # .../usb1/1-1/1-1.4/1-1.4:1.0 -- the interface, whose parent is the port
        for part in (os.path.basename(real), os.path.basename(os.path.dirname(real))):
            if re.fullmatch(r"\d+-[\d.]+", part):
                return part
        return None

    def _usb_reset(self) -> bool:
        """Unbind and rebind the camera's USB port.

        A UVC device left mid-stream by a process that died without releasing it
        opens fine and then delivers nothing, and closing and reopening does not
        clear it -- only a rebind does. Without this the game drops to manual
        entry until someone physically unplugs the webcam.
        """
        port = self._usb_port()
        if port is None:
            return False
        for action in ("unbind", "bind"):
            try:
                done = subprocess.run(
                    ["sudo", "-n", "tee", f"/sys/bus/usb/drivers/usb/{action}"],
                    input=port.encode(), capture_output=True, timeout=10,
                )
                if done.returncode != 0:
                    log.warning("camera %s: USB %s failed", self.cfg.name, action)
                    return False
            except (OSError, subprocess.TimeoutExpired):
                return False
            time.sleep(3.0)
        log.info("camera %s: rebound USB port %s", self.cfg.name, port)
        return True

    def _delivers_frames(self, cap: cv2.VideoCapture, timeout_s: float = 6.0) -> bool:
        """Wait for the first real frame.

        Generous on purpose: a UVC camera takes a second or two to start
        streaming after the controls are set, and longer after a USB rebind. A
        short timeout here reports a perfectly good camera as dead.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            ok, frame = cap.read()
            if ok and frame is not None:
                return True
            time.sleep(0.1)
        return False

    def open(self) -> bool:
        self._resolve()

        # Focus and exposure go through v4l2-ctl *before* the stream starts.
        # Order matters: focus_absolute is ignored while continuous autofocus
        # owns the lens. A fixed-focus camera (the overhead one here) simply has
        # no such control and the calls fail harmlessly.
        self._v4l2_set("focus_automatic_continuous", 1 if self.cfg.autofocus else 0)
        if self.cfg.focus is not None and not self.cfg.autofocus:
            self._v4l2_set("focus_absolute", int(self.cfg.focus))
        self._v4l2_set("auto_exposure", 3 if self.cfg.autoexposure else 1)
        if self.cfg.exposure is not None and not self.cfg.autoexposure:
            self._v4l2_set("exposure_time_absolute", int(self.cfg.exposure))

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            log.error("camera %s: could not open %r", self.cfg.name, self.source)
            return False

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        # Keep the queue shallow so we read the *current* frame, not a stale one.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Opening successfully is not the same as producing pictures, and a
        # camera that opens but never delivers looks identical from above to a
        # board nobody is throwing at. Fail loudly here instead.
        if not self._delivers_frames(cap):
            # A UVC device left mid-stream by a killed process opens fine and
            # then sits silent. Closing and reopening usually clears it, which
            # is worth trying before declaring the camera dead and dropping the
            # game to manual entry for the rest of the evening.
            log.warning(
                "camera %s: no frames after open; closing and retrying once",
                self.cfg.name,
            )
            cap.release()
            time.sleep(2.0)
            cap = self._reopen()
            if cap is None or not self._delivers_frames(cap):
                # Still silent: the device needs a USB rebind, not a reopen.
                if cap is not None:
                    cap.release()
                if not self._usb_reset():
                    log.error(
                        "camera %s: no frames and USB reset unavailable", self.cfg.name
                    )
                    return False
                cap = self._reopen()
                if cap is None or not self._delivers_frames(cap):
                    log.error(
                        "camera %s: opened at %r but delivered no frames "
                        "even after a USB reset", self.cfg.name, self.source,
                    )
                    if cap is not None:
                        cap.release()
                    return False
                log.info("camera %s: recovered after a USB reset", self.cfg.name)

        self.cap = cap
        actual = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info("camera %s: open at %dx%d", self.cfg.name, *map(int, actual))
        self._start_reader()
        return True

    # ---- frame delivery ----------------------------------------------------

    def _start_reader(self) -> None:
        self._closing.clear()
        self._reader = threading.Thread(
            target=self._read_loop, name=f"cam-{self.cfg.name}", daemon=True
        )
        self._reader.start()

    def _read_loop(self) -> None:
        while not self._closing.is_set():
            cap = self.cap
            if cap is None:
                return
            try:
                ok, frame = cap.read()
            except cv2.error:  # device yanked mid-read
                ok, frame = False, None
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            with self._cv:
                self._frame = frame
                self._seq += 1
                self._cv.notify_all()

    def read(self, timeout_s: float = 1.0) -> np.ndarray | None:
        """The next frame this camera has not already handed out.

        Waits for a genuinely new one rather than returning the last one again,
        so the caller keeps the "each read is a fresh observation" contract that
        background differencing and settle detection are both built on. Returns
        None on timeout, which is how a stalled camera stays a *dropped* camera
        instead of a stalled game.
        """
        if self._reader is None:
            return self._grab_direct()
        with self._cv:
            if self._seq == self._seen:
                self._cv.wait(timeout_s)
            if self._frame is None or self._seq == self._seen:
                return None
            self._seen = self._seq
            return self._frame

    def _grab_direct(self) -> np.ndarray | None:
        if self.cap is None:
            return None
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self) -> None:
        self._closing.set()
        reader, self._reader = self._reader, None
        if reader is not None:
            with self._cv:
                self._cv.notify_all()
            reader.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class FileCamera(Camera):
    """Replays a still image or a video. Lets the whole pipeline be developed
    and tested off-Pi, which matters because you cannot iterate on tip
    detection while standing at the oche."""

    def __init__(self, cfg: CameraConfig, path: str):
        super().__init__(cfg)
        self.path = path
        self._still: np.ndarray | None = None

    def open(self) -> bool:
        still = cv2.imread(self.path)
        if still is not None:
            self._still = still
            log.info("camera %s: replaying still %s", self.cfg.name, self.path)
            return True
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            log.error("camera %s: cannot read %s", self.cfg.name, self.path)
            return False
        self.cap = cap
        return True

    def read(self, timeout_s: float = 1.0) -> np.ndarray | None:
        # No reader thread for a file, so Camera.read falls through to a direct
        # grab and the frame pacing stays under this class's control.
        if self._still is not None:
            time.sleep(1.0 / max(self.cfg.fps, 1))
            return self._still.copy()
        frame = super().read(timeout_s)
        if frame is None and self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop
            frame = super().read(timeout_s)
        return frame


def open_cameras(configs: list[CameraConfig], file_sources: dict[str, str] | None = None) -> list[Camera]:
    """Open every configured camera, skipping any that fail.

    A camera that won't open is logged and dropped rather than fatal -- running
    on one camera because the second came unplugged is much better than the
    scoreboard refusing to start mid-game.
    """
    file_sources = file_sources or {}
    cams: list[Camera] = []
    for cfg in configs:
        cam = (
            FileCamera(cfg, file_sources[cfg.name])
            if cfg.name in file_sources
            else Camera(cfg)
        )
        if cam.open():
            cams.append(cam)
    if not cams:
        log.error("no cameras opened; vision disabled, manual entry still works")
    return cams
