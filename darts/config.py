"""Configuration loading. Everything has a working default, so config.yaml is
optional and only needs to contain what you want to change."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .board import BoardGeometry, REGULATION

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass
class GameDefaults:
    default_names: list[str] = field(default_factory=lambda: ["Player 1", "Player 2"])
    start_score: int = 301
    double_out: bool = True
    double_in: bool = False
    auto_advance: bool = True


@dataclass
class AudioConfig:
    enabled: bool = True
    sounds_dir: str = str(ROOT / "sounds")
    # ALSA/Pulse output device. Leave empty for the system default -- but on a
    # Pi that default is HDMI, which fails outright ("audio open error") when no
    # monitor is attached. For the 3.5mm jack use "plughw:2,0"; check the card
    # number with `aplay -l`.
    device: str = ""


@dataclass
class VisionConfig:
    enabled: bool = True
    cameras: list = field(default_factory=list)
    file_sources: dict = field(default_factory=dict)
    geom: BoardGeometry = REGULATION
    detector: Any = None
    yellow: Any = None  # YellowRange; tune with tools/check_calib.py --tune


@dataclass
class AppConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    game: GameDefaults = field(default_factory=GameDefaults)
    audio: AudioConfig = field(default_factory=AudioConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)


def _default_cameras():
    """One camera to the left of the board; add the second when it's mounted."""
    from .vision.camera import CameraConfig

    return [CameraConfig(index=0, name="left-low")]


def load_config(path: str | None = None) -> AppConfig:
    cfg = AppConfig()

    # Import lazily so the server still starts on a machine without OpenCV.
    try:
        from .vision.camera import CameraConfig
        from .vision.detect import DetectorConfig

        cfg.vision.cameras = _default_cameras()
        cfg.vision.detector = DetectorConfig()
    except ImportError:
        log.warning("OpenCV not importable; vision will stay disabled")
        cfg.vision.enabled = False
        CameraConfig = None  # type: ignore[assignment]
        DetectorConfig = None  # type: ignore[assignment]

    candidate = Path(path) if path else ROOT / "config.yaml"
    if not candidate.is_file():
        log.info("no config file at %s; using defaults", candidate)
        return cfg

    try:
        import yaml
    except ImportError:
        log.warning("PyYAML not installed; ignoring %s", candidate)
        return cfg

    with candidate.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    if s := data.get("server"):
        cfg.server = replace(cfg.server, **s)
    if g := data.get("game"):
        cfg.game = replace(cfg.game, **g)
    if a := data.get("audio"):
        cfg.audio = replace(cfg.audio, **a)

    v = data.get("vision") or {}
    cfg.vision.enabled = v.get("enabled", cfg.vision.enabled) and CameraConfig is not None
    cfg.vision.file_sources = v.get("file_sources", {}) or {}

    if (cams := v.get("cameras")) and CameraConfig is not None:
        cfg.vision.cameras = [CameraConfig(**c) for c in cams]
    if (d := v.get("detector")) and DetectorConfig is not None:
        cfg.vision.detector = replace(cfg.vision.detector, **d)
    if (y := v.get("yellow")) and CameraConfig is not None:
        from .vision.calibrate import YellowRange

        cfg.vision.yellow = YellowRange(**y)

    # A board whose double ring measures something other than regulation:
    # give the outer radius in mm and every other ring scales with it.
    if outer := v.get("double_outer_mm"):
        cfg.vision.geom = REGULATION.scaled_to(float(outer))
        log.info("board scaled to a %.1f mm double-outer radius", float(outer))
    if rings := v.get("rings"):
        cfg.vision.geom = replace(cfg.vision.geom, **rings)

    log.info("loaded config from %s", candidate)
    return cfg
