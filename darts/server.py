"""FastAPI app: phone UI, websocket state push, and the vision bridge.

The game is fully playable with vision switched off -- the camera is a source
of *suggested* darts, never a hard dependency. That is deliberate: an 88%
accurate detector behind a one-tap correction is a good scoreboard, whereas an
88% accurate detector you cannot override is an unusable one.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .audio import Announcer, phrases_to_text
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
        # What the cameras said for each dart of the current turn, aligned with
        # game.turn. Kept so that correcting dart 1 of 3 can still be paired
        # with the reading that produced it -- see record_correction.
        self.turn_reads: list[dict | None] = []
        self.loop: asyncio.AbstractEventLoop | None = None
        # Monotonic counter, not a queue: the browser speaks a line only when
        # the sequence advances past what it last spoke. That makes the callout
        # idempotent across the redundant state broadcasts and survives a
        # reconnect without replaying a backlog of stale scores.
        self.speech = {"seq": 0, "text": ""}

    # -- broadcasting ----------------------------------------------------

    def announce(self, keys: list[str]) -> None:
        """Speak a set of phrase keys, on the Pi and/or on every phone."""
        if self.cfg.audio.enabled:
            self.announcer.say(*keys)
        if self.cfg.audio.browser and keys:
            text = phrases_to_text(keys, [p.name for p in self.game.players])
            if text:
                self.speech = {"seq": self.speech["seq"] + 1, "text": text}

    def snapshot(self) -> dict:
        return {
            "type": "state",
            "game": self.game.to_dict(),
            "vision": self.vision.status() if self.vision else {"state": "disabled"},
            "last_detection": self.last_detection,
            "speech": self.speech,
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

    # -- correction log --------------------------------------------------

    def _log_jsonl(self, name: str, entry: dict) -> None:
        try:
            path = ROOT / "data" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, default=float) + "\n")
        except OSError as exc:
            log.warning("could not append to %s: %s", name, exc)

    def record_correction(self, index: int, was: str, now: str) -> None:
        """Append a corrected dart to the log, with what each camera saw.

        This is the ground truth the system has no other way of getting. A
        wrong score is only ever visible to the machine as a score it was
        confident about; the tap that fixes it is the one moment a human states
        what was actually true, and pairing that with the per-camera millimetre
        readings that produced it is what makes the failures measurable instead
        of anecdotal. It also answers "which camera is wrong", which cannot be
        settled by looking at either camera on its own.

        Corrections alone are a biased sample: they only ever record darts that
        were *wrong*. Measuring a change against them can show it fixing known
        failures and cannot show it breaking darts that were already right. So
        every camera dart goes to darts.jsonl as well, and a dart that appears
        there without a matching correction is the closest thing available to a
        confirmed-correct example.

        Nothing reads either file yet. They are deliberately raw and
        append-only: the analysis worth doing depends on what the errors turn
        out to look like, and inventing that before there is data is how you
        end up correcting a bias that was never there.
        """
        # Only attach camera data when the stored reading provably belongs to
        # the dart being corrected. Undo and the correct-replay both reshuffle
        # the turn, and a reading paired with the wrong dart is training data
        # that teaches the opposite of the truth -- worse than having none.
        read = self.turn_reads[index] if 0 <= index < len(self.turn_reads) else None
        if read is not None and read.get("label") != was:
            log.debug("correction %d: stored reading is out of step, logging labels only", index)
            read = None
        entry = {
            "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "was": was,
            "truth": now,
            "source": (read or {}).get("source"),
            "confidence": (read or {}).get("confidence"),
            "per_camera": (read or {}).get("per_camera"),
        }
        self._log_jsonl("corrections.jsonl", entry)
        log.info("correction: %s -> %s (cameras: %s)", was, now, entry["per_camera"])

    def apply_throw(self, label: str, source: str = "manual", confidence: float = 1.0) -> None:
        hit = hit_from_label(label, self.cfg.vision.geom)
        calls = self.game.throw(hit)
        self.turn_reads.append(None)  # entered by hand; no camera reading
        self.announce(calls)
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
                debug_dir=Path(self.cfg.vision.debug_dir) if self.cfg.vision.debug_dir else None,
            ),
            on_dart=self._on_dart,
            on_darts_removed=self._on_darts_removed,
            on_status=lambda _s: self.broadcast_soon(),
        )
        self.vision.start()

    def _on_dart(self, event) -> None:
        calls = self.game.throw(event.hit)
        self.announce(calls)
        self.last_detection = {
            "label": event.hit.label,
            "points": event.hit.points,
            "source": "camera",
            "confidence": round(event.confidence, 2),
            "per_camera": {k: [round(v[0], 1), round(v[1], 1)] for k, v in event.per_camera.items()},
        }
        self.turn_reads.append(dict(self.last_detection))
        self._log_jsonl("darts.jsonl", {
            "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "label": event.hit.label,
            "confidence": round(event.confidence, 2),
            "per_camera": self.last_detection["per_camera"],
        })
        self.broadcast_soon()

    def _on_darts_removed(self) -> None:
        """Player pulled their darts -- that is the end of the turn.

        Only if they actually threw this turn, though. Tapping Next Player and
        *then* pulling the darts out used to advance twice, which with two
        players lands you back on the player you just handed over from: the
        scoreboard visibly swapped and swapped back. An empty turn means the
        handover has already happened and this removal is just tidying up.
        """
        if (
            self.game.config.auto_advance
            and not self.game.finished
            and self.game.turn
        ):
            self.announce(self.game.next_player())
            self.turn_reads.clear()
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
        hub.turn_reads.clear()
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

        was = turn[body.index].label
        hub.record_correction(body.index, was, replacement.label)
        replay = [h.label for h in turn[body.index + 1:]]
        for _ in range(len(turn) - body.index):
            if not hub.game.undo():
                raise HTTPException(409, "cannot undo far enough to correct that dart")
        hub.game.throw(replacement)
        for label in replay:
            hub.game.throw(hit_from_label(label, hub.cfg.vision.geom))
        # The replay has reshuffled the turn; the guard in record_correction
        # would reject these anyway, so drop them rather than keep stale pairs.
        hub.turn_reads = [None] * len(hub.game.turn)

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
        hub.turn_reads.clear()
        hub.announce(hub.game.next_player())
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
        hub.turn_reads.clear()
        if hub.vision:
            hub.vision.reset_background()
        await hub.broadcast()
        return hub.snapshot()

    # -- vision -----------------------------------------------------------

    async def _encode(hub, camera, overlay, w, q, crop):
        """Draw the overlay, shrink and JPEG-encode, off the event loop.

        All three are real work -- tens of milliseconds a frame on a Pi -- and
        doing them inline stalls everything else the server is doing for that
        long. With one preview behind a toggle that was survivable. The page now
        holds two continuous streams, so the same blocking would land on the
        websocket that carries the scores, and the scoreboard would judder
        whenever anyone was looking at the cameras.
        """
        return await asyncio.to_thread(
            hub.vision.preview_jpeg, camera, overlay, w, q, crop
        )

    @app.get("/api/vision/status")
    async def vision_status():
        return hub.vision.status() if hub.vision else {"state": "disabled"}

    @app.post("/api/vision/recalibrate")
    async def recalibrate():
        if not hub.vision:
            raise HTTPException(409, "vision is not running")
        hub.vision.request_recalibration()
        return {"ok": True, "note": "clear the board -- calibration needs an empty face"}

    @app.post("/api/vision/forget-orientation")
    async def forget_orientation():
        """Throw away the learned orientation and guess from the numerals again.

        The escape hatch for a template saved at a wrong rotation -- otherwise
        every later calibration would faithfully reproduce the mistake.
        """
        if not hub.vision:
            raise HTTPException(409, "vision is not running")
        hub.vision.forget_orientation()
        return {"ok": True}

    @app.post("/api/vision/rebaseline")
    async def rebaseline():
        if not hub.vision:
            raise HTTPException(409, "vision is not running")
        hub.vision.reset_background()
        return {"ok": True}

    @app.post("/api/vision/rotate")
    async def rotate(sectors: int = 1, camera: str | None = None):
        """Nudge the calibrated orientation. See VisionPipeline.nudge_rotation.

        `camera` rotates just that one. Each camera resolves the board's
        36-degree symmetry against its own view, so they can disagree, and
        rotating both together cannot fix one without breaking the other.
        """
        if not hub.vision:
            raise HTTPException(409, "vision is not running")
        hub.vision.nudge_rotation(sectors, camera)
        await hub.broadcast()
        return {"ok": True}

    @app.get("/api/vision/preview.jpg")
    async def preview(
        camera: str | None = None, overlay: bool = True, w: int = 0,
        q: int = 70, crop: bool = True,
    ):
        if not hub.vision:
            raise HTTPException(409, "vision is not running")
        jpeg = await _encode(hub, camera, overlay, w, q, crop)
        if jpeg is None:
            raise HTTPException(503, "no frame available yet")
        return Response(jpeg, media_type="image/jpeg")

    @app.get("/api/vision/stream.mjpg")
    async def stream(
        camera: str | None = None,
        overlay: bool = True,
        w: int = 0,
        q: int = 70,
        fps: float = 5.0,
        crop: bool = True,
    ):
        """MJPEG preview. `w`/`q`/`fps` exist to keep it affordable.

        The page shows every camera at once now, so this is two concurrent
        streams over a link that has been the slow part of this setup from the
        start. Full-size frames at 5fps each would need roughly 1.5 MB/s; the
        defaults the UI asks for are about a fortieth of that, and the preview
        only has to show whether the overlay sits on the rings.
        """
        if not hub.vision:
            raise HTTPException(409, "vision is not running")
        delay = 1.0 / min(max(fps, 0.5), 15.0)

        async def frames():
            while True:
                jpeg = await _encode(hub, camera, overlay, w, q, crop)
                if jpeg:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                await asyncio.sleep(delay)

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
        hub.announce(hub.game.next_player())
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
