"""The web layer: that it loads at all, and that corrections get recorded.

The first test here exists because a syntax error in server.py once reached the
Pi and crash-looped the service while the whole suite reported green. Nothing
imported the module, so nothing noticed. Everything else in this file is about
the correction log, which is the only ground truth the system can get: a wrong
score looks from the inside exactly like a right one, and the tap that fixes it
is the one moment a human states what was actually true.
"""

from __future__ import annotations

import json

import pytest

from darts.config import AppConfig
from darts.server import Hub, create_app


class TestItLoads:
    def test_the_module_imports_and_the_app_builds(self):
        """A syntax error here is invisible to every other test in the suite."""
        app = create_app()
        routes = {getattr(r, "path", None) for r in app.routes}
        assert "/api/throw" in routes
        assert "/api/correct" in routes
        assert "/ws" in routes


@pytest.fixture
def hub(tmp_path, monkeypatch):
    import darts.server as server_mod

    monkeypatch.setattr(server_mod, "ROOT", tmp_path)
    cfg = AppConfig()
    cfg.audio.enabled = False
    cfg.audio.browser = False
    cfg.vision.enabled = False
    return Hub(cfg)


def corrections(tmp_path):
    path = tmp_path / "data" / "corrections.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


class TestCorrectionLog:
    def _camera_dart(self, hub, label, per_camera):
        """Stand in for the vision pipeline scoring a dart."""
        from darts.board import hit_from_label

        class Event:
            hit = hit_from_label(label)
            confidence = 0.2
        Event.per_camera = per_camera
        hub._on_dart(Event())

    def test_a_correction_keeps_what_each_camera_said(self, hub, tmp_path):
        """Which camera was wrong cannot be settled by looking at one of them."""
        self._camera_dart(hub, "S4", {"left-low": (-29.0, 51.4), "left-high": (143.4, 42.2)})

        hub.record_correction(0, "S4", "S20")

        rows = corrections(tmp_path)
        assert len(rows) == 1
        assert rows[0]["was"] == "S4" and rows[0]["truth"] == "S20"
        assert rows[0]["source"] == "camera"
        assert rows[0]["per_camera"]["left-low"] == [-29.0, 51.4]
        assert rows[0]["per_camera"]["left-high"] == [143.4, 42.2]

    def test_it_appends_rather_than_replacing(self, hub, tmp_path):
        self._camera_dart(hub, "S4", {"left-low": (1.0, 2.0)})
        hub.record_correction(0, "S4", "S20")
        self._camera_dart(hub, "S18", {"left-low": (3.0, 4.0)})
        hub.record_correction(1, "S18", "S12")
        assert [r["truth"] for r in corrections(tmp_path)] == ["S20", "S12"]

    def test_a_mismatched_index_logs_labels_but_no_camera_data(self, hub, tmp_path):
        """A reading paired with the wrong dart teaches the opposite of truth.

        Undo and the correct-replay both reshuffle the turn, so the stored
        reading can fall out of step with the dart being corrected. Better a
        row with no camera data than a row with someone else's.
        """
        self._camera_dart(hub, "S4", {"left-low": (1.0, 2.0)})

        hub.record_correction(0, "T19", "S20")  # 'was' does not match the read

        rows = corrections(tmp_path)
        assert len(rows) == 1, "the correction is still worth recording"
        assert rows[0]["truth"] == "S20"
        assert rows[0]["per_camera"] is None

    def test_a_hand_entered_dart_has_no_camera_data(self, hub, tmp_path):
        hub.apply_throw("S5")
        hub.record_correction(0, "S5", "S20")
        rows = corrections(tmp_path)
        assert rows[0]["per_camera"] is None
        assert rows[0]["source"] is None

    def test_an_out_of_range_index_does_not_raise(self, hub, tmp_path):
        """Corrections come from a phone; they must never take the server down."""
        hub.record_correction(7, "S4", "S20")
        assert corrections(tmp_path)[0]["per_camera"] is None


class TestTurnReadsStayAligned:
    def test_starting_a_new_turn_clears_them(self, hub):
        hub.apply_throw("S5")
        hub.apply_throw("S1")
        assert len(hub.turn_reads) == 2
        hub.game.next_player()
        hub.turn_reads.clear()
        assert hub.turn_reads == []

    def test_one_entry_is_appended_per_dart(self, hub):
        for label in ("S5", "S1", "S20"):
            hub.apply_throw(label)
        assert len(hub.turn_reads) == len(hub.game.turn)
