"""301/501 game engine.

Deliberately has no knowledge of cameras, audio or HTTP -- it takes Hits in and
produces state plus a list of callout keys. That keeps it trivially testable and
means the whole thing works as a manual scoreboard with no vision at all.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

from .board import Hit, hit_from_label

DARTS_PER_TURN = 3

TurnEnd = Literal["", "complete", "bust", "win"]
# "not_a_double": landed exactly on zero, but not on a double.
# "overshot":     went past zero.
# "left_one":     would leave 1, which cannot be finished on a double.
BustReason = Literal["", "not_a_double", "overshot", "left_one"]


@dataclass
class Player:
    name: str
    score: int
    started: bool = False  # double-in satisfied
    darts_thrown: int = 0
    total_scored: int = 0

    @property
    def average(self) -> float:
        """Three-dart average, the standard darts stat."""
        if self.darts_thrown == 0:
            return 0.0
        return self.total_scored / self.darts_thrown * 3


@dataclass
class GameConfig:
    start_score: int = 301
    double_out: bool = True
    double_in: bool = False
    # Advance the player when the camera sees the darts pulled out of the board.
    # The turn locks after three darts either way; it just doesn't change hands
    # until the darts come out (or Next Player is tapped), which keeps the last
    # dart on screen and correctable.
    auto_advance: bool = True


@dataclass
class Game:
    config: GameConfig = field(default_factory=GameConfig)
    players: list[Player] = field(default_factory=list)
    current: int = 0
    turn: list[Hit] = field(default_factory=list)
    turn_end: TurnEnd = ""
    # Why the turn busted, so the UI can say so. "Bust" on its own is the most
    # confusing message in darts: hitting your exact remaining score and being
    # told you lost the turn looks like a broken scoreboard unless you already
    # know the double-out rule is what did it.
    bust_reason: BustReason = ""
    winner: int | None = None
    _undo_stack: list[dict] = field(default_factory=list, repr=False)

    # ---- setup -------------------------------------------------------------

    @classmethod
    def new(cls, names: list[str], config: GameConfig | None = None) -> "Game":
        cfg = config or GameConfig()
        if not names:
            raise ValueError("need at least one player")
        return cls(
            config=cfg,
            players=[Player(name=n, score=cfg.start_score) for n in names],
        )

    # ---- play --------------------------------------------------------------

    @property
    def player(self) -> Player:
        return self.players[self.current]

    @property
    def finished(self) -> bool:
        return self.winner is not None

    @property
    def turn_score(self) -> int:
        return sum(h.points for h in self.turn)

    @property
    def darts_left(self) -> int:
        return DARTS_PER_TURN - len(self.turn)

    def throw(self, hit: Hit) -> list[str]:
        """Register a dart. Returns callout keys for the audio layer."""
        if self.finished:
            return []
        if self.turn_end:
            # Turn is over and we're waiting on the Next Player button -- ignore
            # stray detections (usually the player pulling their darts).
            return []

        self._push_undo()
        self.turn.append(hit)

        player = self.player
        player.darts_thrown += 1
        calls = [hit.spoken()]

        # Double-in: nothing counts until the first double lands.
        if self.config.double_in and not player.started:
            if hit.is_double:
                player.started = True
            else:
                return self._maybe_end_turn(calls)

        remaining = player.score - hit.points
        min_finish = 2 if self.config.double_out else 0

        if remaining < 0:
            return self._bust(calls, "overshot")
        if remaining != 0 and remaining < min_finish:
            return self._bust(calls, "left_one")
        if remaining == 0:
            if self.config.double_out and not hit.is_double:
                return self._bust(calls, "not_a_double")
            player.score = 0
            player.total_scored += hit.points
            self.winner = self.current
            self.turn_end = "win"
            calls.append("game_shot")
            return calls

        player.score = remaining
        player.total_scored += hit.points
        return self._maybe_end_turn(calls)

    def throw_label(self, label: str) -> list[str]:
        """Manual entry helper -- "T20", "BULL", "MISS", ..."""
        return self.throw(hit_from_label(label))

    def _bust(self, calls: list[str], reason: BustReason) -> list[str]:
        # Revert every dart in this turn, per standard rules.
        player = self.player
        for h in self.turn[:-1]:
            player.score += h.points
            player.total_scored -= h.points
        self.turn_end = "bust"
        self.bust_reason = reason
        calls.append("bust")
        return calls

    def _maybe_end_turn(self, calls: list[str]) -> list[str]:
        if len(self.turn) >= DARTS_PER_TURN:
            self.turn_end = "complete"
            calls.append(f"scored_{self.turn_score}")
        return calls

    def next_player(self) -> list[str]:
        """End the current turn and move on. Safe to call mid-turn (a stand-down)."""
        if self.finished:
            return []
        self._push_undo()
        self.turn = []
        self.turn_end = ""
        self.bust_reason = ""
        self.current = (self.current + 1) % len(self.players)
        return [f"player_{min(self.current + 1, 8)}", "your_throw"]

    def undo(self) -> bool:
        """Step back one action (dart or player change)."""
        if not self._undo_stack:
            return False
        self._restore(self._undo_stack.pop())
        return True

    def reset(self) -> None:
        """Same players and settings, scores back to the start."""
        self._push_undo()
        for p in self.players:
            p.score = self.config.start_score
            p.started = False
            p.darts_thrown = 0
            p.total_scored = 0
        self.current = 0
        self.turn = []
        self.turn_end = ""
        self.bust_reason = ""
        self.winner = None

    # ---- undo plumbing -----------------------------------------------------

    def _snapshot(self) -> dict:
        return {
            "players": copy.deepcopy(self.players),
            "current": self.current,
            "turn": list(self.turn),
            "turn_end": self.turn_end,
            "bust_reason": self.bust_reason,
            "winner": self.winner,
        }

    def _push_undo(self) -> None:
        self._undo_stack.append(self._snapshot())
        del self._undo_stack[:-100]

    def _restore(self, snap: dict) -> None:
        self.players = snap["players"]
        self.current = snap["current"]
        self.turn = snap["turn"]
        self.turn_end = snap["turn_end"]
        self.bust_reason = snap["bust_reason"]
        self.winner = snap["winner"]

    # ---- serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "config": {
                "start_score": self.config.start_score,
                "double_out": self.config.double_out,
                "double_in": self.config.double_in,
                "auto_advance": self.config.auto_advance,
            },
            "players": [
                {
                    "name": p.name,
                    "score": p.score,
                    "started": p.started,
                    "darts": p.darts_thrown,
                    "average": round(p.average, 1),
                    "checkout": checkout_hint(p.score, self.config.double_out),
                }
                for p in self.players
            ],
            "current": self.current,
            "turn": [
                {"label": h.label, "points": h.points} for h in self.turn
            ],
            "turn_score": self.turn_score,
            "turn_end": self.turn_end,
            "bust_reason": self.bust_reason,
            "darts_left": self.darts_left,
            "winner": self.winner,
            "can_undo": bool(self._undo_stack),
        }


# ---- check-out suggestions -------------------------------------------------

_BULL = ("BULL", 50)
_SINGLES = [(f"S{s}", s) for s in range(1, 21)]
_DOUBLES = [(f"D{s}", s * 2) for s in range(1, 21)] + [_BULL]
_TRIPLES = [(f"T{s}", s * 3) for s in range(1, 21)]
_OUTER_BULL = ("25", 25)
# Highest-value darts first, so the search naturally returns a sensible setup
# shot rather than a technically-valid but silly one.
_SETUP = sorted(_TRIPLES + _DOUBLES + _SINGLES + [_OUTER_BULL], key=lambda p: -p[1])

# Scores that cannot be finished in three darts on a double. Short-circuiting
# these keeps the exhaustive branch from ever running.
_BOGEY = frozenset({169, 168, 166, 165, 163, 162, 159})


@lru_cache(maxsize=512)
def checkout_hint(score: int, double_out: bool = True) -> tuple[str, ...] | None:
    """Cheapest 1-3 dart finish for `score`, or None if it can't be checked out.

    Computed rather than table-driven; it avoids shipping a 170-entry lookup
    that is easy to get subtly wrong. Cached because it's recomputed for every
    player on every state broadcast, and a Pi has better things to do.
    """
    if score <= 0 or score > 170:
        return None
    if double_out and (score == 1 or score in _BOGEY):
        return None

    finishers = _DOUBLES if double_out else _SETUP

    for name, val in finishers:
        if val == score:
            return (name,)
    for n1, v1 in _SETUP:
        for n2, v2 in finishers:
            if v1 + v2 == score:
                return (n1, n2)
    for n1, v1 in _SETUP:
        for n2, v2 in _SETUP:
            for n3, v3 in finishers:
                if v1 + v2 + v3 == score:
                    return (n1, n2, n3)
    return None
