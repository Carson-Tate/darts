"""Audio device selection.

The Pi's default ALSA device is HDMI. With no monitor attached `aplay` fails
with "audio open error: Unknown error 524", and because playback failures are
deliberately non-fatal, every callout would vanish without the game noticing.
So the device setting has to actually reach the player argv.
"""

from __future__ import annotations

import pytest

from darts.audio import Announcer, phrase_book, phrases_to_text, _with_device


@pytest.mark.parametrize(
    "player, device, expected",
    [
        (["aplay", "-q"], "plughw:2,0", ["aplay", "-q", "-D", "plughw:2,0"]),
        (["paplay"], "alsa_output.x", ["paplay", "--device", "alsa_output.x"]),
        # afplay has no device flag; the setting is ignored rather than mangled.
        (["afplay"], "plughw:2,0", ["afplay"]),
        # No device configured means don't touch the command at all.
        (["aplay", "-q"], "", ["aplay", "-q"]),
    ],
)
def test_device_flag_matches_the_player(player, device, expected):
    assert _with_device(player, device) == expected


def test_announcer_passes_the_device_to_the_player():
    a = Announcer("sounds", player=["aplay", "-q"], device="plughw:2,0")
    assert a.player == ["aplay", "-q", "-D", "plughw:2,0"]


def test_announcer_without_a_device_is_unchanged():
    a = Announcer("sounds", player=["aplay", "-q"])
    assert a.player == ["aplay", "-q"]


def test_disabled_announcer_never_touches_the_queue(tmp_path):
    """say() on a disabled announcer is a no-op, not an error."""
    a = Announcer(tmp_path, player=["aplay"], enabled=False)
    a.say("triple_20", "your_throw")
    assert a._queue.empty()


def test_missing_clips_do_not_raise(tmp_path):
    """A missing WAV must never interrupt a game."""
    a = Announcer(tmp_path, player=["aplay"])
    a.say("no_such_clip")
    a.start()
    a.stop()


# ---------------------------------------------------------------- browser TTS


def test_phrases_render_to_one_line():
    assert phrases_to_text(["triple_20", "scored_60"]) == "treble twenty. 60"


def test_player_keys_become_real_names():
    """The whole point of browser speech: the WAVs can only say "Player 2"."""
    out = phrases_to_text(["player_2", "your_throw"], ["Carson", "Dylan"])
    assert out == "Dylan. your throw"


def test_player_keys_fall_back_when_the_name_is_missing():
    out = phrases_to_text(["player_3", "your_throw"], ["Carson", "Dylan"])
    assert out == "Player 3. your throw"


def test_unknown_keys_are_dropped_not_spoken_raw():
    assert phrases_to_text(["bullseye", "no_such_key"]) == "Bullseye!"


def test_empty_input_is_empty():
    assert phrases_to_text([]) == ""


def test_every_phrase_key_has_speakable_text():
    """A key with no text would be a silent callout in the browser."""
    assert all(text.strip() for text in phrase_book().values())
