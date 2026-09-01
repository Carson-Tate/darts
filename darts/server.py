"""FastAPI app: phone UI, websocket state push, and the vision bridge.

The game is fully playable with vision switched off -- the camera is a source
of *suggested* darts, never a hard dependency. That is deliberate: an 88%
accurate detector behind a one-tap correction is a good scoreboard, whereas an
88% accurate detector you cannot override is an unusable one.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .audio import Announcer
from .board import hit_from_label
from .config import AppConfig, load_config
from .game import Game, GameConfig

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"


# ---- request bodies --------------------------------------------------------


class NewGameBody(BaseModel):
    names: list[str] = Field(min_length=1, max_length=8)
    start_score: int = 301
    double_out: bool = True
    double_in: bool = False
    auto_advance: bool = True


class ThrowBody(BaseModel):
    label: str


class CorrectBody(BaseModel):
    index: int  # position within the current turn, 0-based
    label: str


# ---- app state -------------------------------------------------------------


class Hub:
    """Owns the game, the announcer, the vision pipeline and the sockets."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.game = Game.new(cfg.game.default_names, GameConfig(
            start_score=cfg.game.start_score,
            double_out=cfg.game.double_out,
            double_in=cfg.game.double_in,
            auto_advance=cfg.game.auto_advance,
        ))
        self.announcer = Announcer(
            cfg.audio.sounds_dir, enabled=cfg.audio.enabled, device=cfg.audio.device
        )
        self.sockets: set[WebSocket] = set()
        self.vision = None  # set in start_vision()
        self.last_detection: dict | None = None
        self.loop: asyncio.AbstractEventLoop | None = None

    # -- broadcasting ----------------------------------------------------

    def snapshot(self) -> dict:
        return {
            "type": "state",
            "game": self.game.to_dict(),
            "vision": self.vision.status() if self.vision else {"state": "disabled"},
            "last_detection": self.last_detection,
        }

    async def broadcast(self) -> None:
        if not self.sockets:
            return
        payload = self.snapshot()
        dead = []
        for ws in list(self.sockets):
            try:
                await ws.send_json(payload)
            except (WebSocketDisconnect, RuntimeError):
                dead.append(ws)
        for ws in dead:
            self.sockets.discard(ws)

    def broadcast_soon(self) -> None:
        """Broadcast from a non-async thread (the vision loop)."""
        if self.loop is None:
            return
        asyncio.run_coroutine_threadsafe(self.broadcast(), self.loop)

    # -- game actions ----------------------------------------------------

    def apply_throw(self, label: str, source: str = "manual", confidence: float = 1.0) -> None:
        hit = hit_from_label(label, self.cfg.vision.geom)
        calls = self.game.throw(hit)
        self.announcer.say(*calls)
        self.last_detection = {
            "label": hit.label,
            "points": hit.points,
            "source": source,
            "confidence": round(confidence, 2),
        }
        # Note: the turn is *locked* after three darts but the player is not
        # changed here. Advancing immediately would clear the turn, and with it
        # any chance of correcting a misread third dart -- which is exactly the
        # dart most likely to need correcting. The turn advances when the
        # camera sees the darts pulled, or when Next Player is tapped.

    # -- vision bridge ---------------------------------------------------

    def start_vision(self) -> None:
        if not self.cfg.vision.enabled:
            log.info("vision disabled in config; manual entry only")
            return
        try:
            from .vision.camera import open_cameras
            from .vision.detect import DetectorConfig
            from .vision.pipeline import PipelineConfig, VisionPipeline
        except ImportError as exc:
            log.warning("vision unavailable (%s); manual entry only", exc)
            return

        cams = open_cameras(self.cfg.vision.cameras, self.cfg.vision.file_sources)
        if not cams:
            return

        self.vision = VisionPipeline(
            cams,
            PipelineConfig(
                geom=self.cfg.vision.geom,
                detector=self.cfg.vision.detector or DetectorConfig(),
                yellow=self.cfg.vision.yellow,
            ),
            on_dart=self._on_dart,
            on_darts_removed=self._on_darts_removed,
            on_status=lambda _s: self.broadcast_soon(),
        )
        self.vision.start()

    def _on_dart(self, event) -> None:
        calls = self.game.throw(event.hit)
        self.announcer.say(*calls)
        self.last_detection = {
            "label": event.hit.label,
            "points": event.hit.points,
            "source": "camera",
            "confidence": round(event.confidence, 2),
            "per_camera": {k: [round(v[0], 1), round(v[1], 1)] for k, v in event.per_camera.items()},
        }
        self.broadcast_soon()

    def _on_darts_removed(self) -> None:
        # Player pulled their darts -- that is the end of the turn.
        if self.game.config.auto_advance and not self.game.finished:
            self.announcer.say(*self.game.next_player())
        self.broadcast_soon()


# ---- app factory -----------------------------------------------------------


def create_app(config_path: str | None = None) -> FastAPI:
    cfg = load_config(config_path)
    hub = Hub(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        hub.loop = asyncio.get_running_loop()
        hub.announcer.start()
        hub.start_vision()
        log.info("darts server ready on http://%s:%d", cfg.server.host, cfg.server.port)
        yield
        if hub.vision:
            hub.vision.stop()
        hub.announcer.stop()

    app = FastAPI(title="Darts", lifespan=lifespan)
    app.state.hub = hub

    if STATIC.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    async def index():
        page = STATIC / "index.html"
        if not page.is_file():
            raise HTTPException(500, "static/index.html is missing")
        return FileResponse(page)

    # -- state ------------------------------------------------------------

    @app.get("/api/state")
    async def get_state():
        return hub.snapshot()

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        hub.sockets.add(ws)
        await ws.send_json(hub.snapshot())
        try:
            while True:
                msg = await ws.receive_json()
                await _handle_ws_command(hub, msg)
        except (WebSocketDisconnect, ValueError):
            pass
        finally:
            hub.sockets.discard(ws)

    # -- game -------------------------------------------------------------

    @app.post("/api/game/new")
    async def new_game(body: NewGameBody):
        names = [n.strip() or f"Player {i+1}" for i, n in enumerate(body.names)]
        hub.game = Game.new(names, GameConfig(
            start_score=body.start_score,
            double_out=body.double_out,
            double_in=body.double_in,
            auto_advance=body.auto_advance,
        ))
        hub.last_detection = None
        if hub.vision:
            hub.vision.reset_background()
        await hub.broadcast()
        return hub.snapshot()

    @app.post("/api/throw")
    async def throw(body: ThrowBody):
        try:
            hub.apply_throw(body.label)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        await hub.broadcast()
        return hub.snapshot()

    @app.post("/api/correct")
    async def correct(body: CorrectBody):
        """Replace a dart in the current turn.

        Implemented as undo-back-to-it then replay, so the score, bust handling
        and check-out hints all stay consistent with however the darts were
        originally entered.
        """
        turn = hub.game.turn
        if not 0 <= body.index < len(turn):
            raise HTTPException(400, f"no dart at index {body.index}")
        try:
            replacement = hit_from_label(body.label, hub.cfg.vision.geom)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

        replay = [h.label for h in turn[body.index + 1:]]
        for _ in range(len(turn) - body.index):
            if not hub.game.undo():
                raise HTTPException(409, "cannot undo far enough to correct that dart")
        hub.game.throw(replacement)
        for label in replay:
            hub.game.throw(hit_from_label(label, hub.cfg.vision.geom))

        hub.last_detection = {
            "label": replacement.label,
            "points": replacement.points,
            "source": "corrected",
            "confidence": 1.0,
        }
        await hub.broadcast()
        return hub.snapshot()

    @app.post("/api/next")
    async def next_player():
        hub.announcer.clear()
        hub.announcer.say(*hub.game.next_player())
        if hub.vision:
            hub.vision.reset_background()
        await hub.broadcast()
        return hub.snapshot()

    @app.post("/api/undo")
    async def undo():
        if not hub.game.undo():
            raise HTTPException(409, "nothing to undo")
        await hub.broadcast()
        return hub.snapshot()

    @app.post("/api/reset")
    async def reset():
        hub.game.reset()
        hub.last_detection = None
        if hub.vision:
            hub.vision.reset_background()
        await hub.broadcast()
        return hub.snapshot()

    # -- vision -----------------------------------------------------------

    @app.get("/api/vision/status")
    async def vision_status():
        return hub.vision.status() if hub.vision else {"state": "disabled"}

    @app.post("/api/vision/recalibrate")
    async def recalibrate():
        if not hub.vision:
            raise HTTPException(409, "vision is not running")
        hub.vision.request_recalibration()
        return {"ok": True, "note": "clear the board -- calibration needs an empty face"}

    @app.post("/api/vision/rebaseline")
    async def rebaseline():
        if not hub.vision:
            raise HTTPException(409, "vision is not running")
        hub.vision.reset_background()
        return {"ok": True}

    @app.post("/api/vision/rotate")
    async def rotate(sectors: int = 1):
        """Nudge the calibrated orientation. See VisionPipeline.nudge_rotation."""
        if not hub.vision:
            raise HTTPException(409, "vision is not running")
        hub.vision.nudge_rotation(sectors)
        await hub.broadcast()
        return {"ok": True}

    @app.get("/api/vision/preview.jpg")
    async def preview(camera: str | None = None, overlay: bool = True):
        if not hub.vision:
            raise HTTPException(409, "vision is not running")
        jpeg = hub.vision.preview_jpeg(camera, overlay)
        if jpeg is None:
            raise HTTPException(503, "no frame available yet")
        return Response(jpeg, media_type="image/jpeg")

    @app.get("/api/vision/stream.mjpg")
    async def stream(camera: str | None = None, overlay: bool = True):
        if not hub.vision:
            raise HTTPException(409, "vision is not running")

        async def frames():
            while True:
                jpeg = hub.vision.preview_jpeg(camera, overlay)
                if jpeg:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                await asyncio.sleep(0.2)

        return StreamingResponse(
            frames(), media_type="multipart/x-mixed-replace; boundary=frame"
        )

    return app


async def _handle_ws_command(hub: Hub, msg: dict) -> None:
    action = msg.get("action")
    if action == "throw":
        try:
            hub.apply_throw(str(msg["label"]))
        except (KeyError, ValueError) as exc:
            log.warning("bad throw over websocket: %s", exc)
    elif action == "next":
        hub.announcer.clear()
        hub.announcer.say(*hub.game.next_player())
        if hub.vision:
            hub.vision.reset_background()
    elif action == "undo":
        hub.game.undo()
    elif action == "reset":
        hub.game.reset()
        hub.last_detection = None
    else:
        log.debug("unknown websocket action %r", action)
        return
    await hub.broadcast()
