"""Spoken callouts.

Live TTS on a Pi 4 adds a second of latency to every dart, which kills the feel
of the thing. The vocabulary here is bounded and small -- 20 sectors x 3 rings,
the bulls, turn totals 0-180, and a handful of phrases -- so every clip is
rendered to WAV once by tools/render_audio.py and this module just plays files.

Missing clips are logged once and skipped. A missing callout should never stop
a game.
"""

from __future__ import annotations

import logging
import queue
import shutil
import subprocess
import threading
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_PLAYERS = (
    ["aplay", "-q"],  # Raspberry Pi OS / ALSA
    ["paplay"],  # PulseAudio
    ["afplay"],  # macOS, handy for development
)


# How each player names an explicit output device. aplay/paplay take one;
# afplay has no equivalent, so the setting is ignored there.
_DEVICE_FLAG = {"aplay": "-D", "paplay": "--device"}


def _detect_player() -> list[str] | None:
    for cmd in DEFAULT_PLAYERS:
        if shutil.which(cmd[0]):
            return cmd
    return None


def _with_device(player: list[str], device: str) -> list[str]:
    flag = _DEVICE_FLAG.get(player[0]) if device else None
    return [*player, flag, device] if flag else player


class Announcer:
    """Fire-and-forget audio queue.

    Callouts are played on a worker thread: the game loop and the websocket
    broadcast must not wait on the speaker.
    """

    def __init__(
        self,
        sounds_dir: str | Path,
        player: list[str] | None = None,
        enabled: bool = True,
        device: str = "",
    ):
        self.dir = Path(sounds_dir)
        self.enabled = enabled
        self.player = player or _detect_player()
        if self.player and device:
            self.player = _with_device(self.player, device)
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=32)
        self._warned: set[str] = set()
        self._thread: threading.Thread | None = None

        if self.enabled and self.player is None:
            log.warning("no audio player found (tried aplay/paplay/afplay); callouts disabled")
            self.enabled = False
        if self.enabled and not self.dir.is_dir():
            log.warning("sounds directory %s missing; run tools/render_audio.py", self.dir)

    def start(self) -> None:
        if not self.enabled or self._thread:
            return
        self._thread = threading.Thread(target=self._run, name="announcer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread:
            self._queue.put(None)
            self._thread.join(timeout=2.0)
            self._thread = None

    def say(self, *keys: str) -> None:
        if not self.enabled:
            return
        for key in keys:
            try:
                self._queue.put_nowait(key)
            except queue.Full:
                log.debug("audio queue full, dropping %s", key)

    def clear(self) -> None:
        """Drop anything still queued -- used when a turn is cut short."""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def _run(self) -> None:
        while True:
            key = self._queue.get()
            if key is None:
                return
            path = self.dir / f"{key}.wav"
            if not path.is_file():
                if key not in self._warned:
                    self._warned.add(key)
                    log.info("no clip for %r (%s)", key, path.name)
                continue
            try:
                subprocess.run(
                    [*self.player, str(path)],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                log.warning("playback failed for %s: %s", path.name, exc)


# --------------------------------------------------------------------------
# the phrase list -- shared with tools/render_audio.py so the two cannot drift
# --------------------------------------------------------------------------


def phrase_book() -> dict[str, str]:
    """Every clip key mapped to the text that should be spoken."""
    words = {
        1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
        7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven",
        12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
        16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen",
        20: "twenty",
    }
    book: dict[str, str] = {
        "miss": "No score",
        "bullseye": "Bullseye!",
        "outer_bull": "Twenty five",
        "bust": "Bust!",
        "game_shot": "Game shot!",
        "your_throw": "your throw",
        "winner": "Game over",
    }
    for n, w in words.items():
        book[f"single_{n}"] = w
        book[f"double_{n}"] = f"double {w}"
        book[f"triple_{n}"] = f"treble {w}"
    for total in range(0, 181):
        book[f"scored_{total}"] = "One hundred and eighty!" if total == 180 else str(total)
    for p in range(1, 9):
        book[f"player_{p}"] = f"Player {p}"
    return book
