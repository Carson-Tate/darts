"""Audio device selection.

The Pi's default ALSA device is HDMI. With no monitor attached `aplay` fails
with "audio open error: Unknown error 524", and because playback failures are
deliberately non-fatal, every callout would vanish without the game noticing.
So the device setting has to actually reach the player argv.
"""

from __future__ import annotations

import pytest

from darts.audio import Announcer, _with_device


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
