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
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    index: int | str = 0
    name: str = "left"
    width: int = 1280
    height: int = 720
    fps: int = 15
    autofocus: bool = False
    focus: float | None = None  # 0-255 on most UVC devices; None leaves it alone
    autoexposure: bool = False
    exposure: float | None = None
    extra: dict = field(default_factory=dict)


class Camera:
    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self.cap: cv2.VideoCapture | None = None

    def open(self) -> bool:
        cap = cv2.VideoCapture(self.cfg.index)
        if not cap.isOpened():
            log.error("camera %s: could not open index %r", self.cfg.name, self.cfg.index)
            return False

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        # Keep the queue shallow so we read the *current* frame, not a stale one.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if self.cfg.autofocus else 0)
        if self.cfg.focus is not None:
            cap.set(cv2.CAP_PROP_FOCUS, self.cfg.focus)
        # 0.25 == manual on most V4L2 UVC drivers, 0.75 == aperture priority.
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if self.cfg.autoexposure else 0.25)
        if self.cfg.exposure is not None:
            cap.set(cv2.CAP_PROP_EXPOSURE, self.cfg.exposure)

        self.cap = cap
        actual = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info("camera %s: open at %dx%d", self.cfg.name, *map(int, actual))
        return True

    def read(self) -> np.ndarray | None:
        if self.cap is None:
            return None
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self) -> None:
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

    def read(self) -> np.ndarray | None:
        if self._still is not None:
            time.sleep(1.0 / max(self.cfg.fps, 1))
            return self._still.copy()
        frame = super().read()
        if frame is None and self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop
            frame = super().read()
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
