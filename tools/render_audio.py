#!/usr/bin/env python3
"""Pre-render every callout to a WAV file.

Uses Piper, which runs comfortably offline on a Pi 4 and sounds far better than
espeak. This takes a couple of minutes once, and then playback at game time is
instant -- no model load, no synthesis latency between "dart lands" and "treble
twenty".

    # one-time, on the Pi
    pip install piper-tts
    python -m piper.download_voices en_GB-alba-medium

    python tools/render_audio.py --voice en_GB-alba-medium

Any TTS that can write a WAV works; --command lets you swap Piper out.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from darts.audio import phrase_book  # noqa: E402


def render_with_piper(text: str, out: Path, voice: str) -> bool:
    proc = subprocess.run(
        [sys.executable, "-m", "piper", "--model", voice, "--output_file", str(out)],
        input=text.encode("utf-8"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        print(f"  piper failed: {proc.stderr.decode(errors='replace').strip()}")
        return False
    return True


def render_with_command(text: str, out: Path, template: str) -> bool:
    cmd = template.replace("{text}", text).replace("{out}", str(out))
    return subprocess.run(cmd, shell=True).returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--out", default="sounds", help="output directory")
    parser.add_argument("--voice", default="en_GB-alba-medium", help="Piper voice name")
    parser.add_argument("--command", help='custom TTS command, e.g. "say -o {out} {text}"')
    parser.add_argument("--force", action="store_true", help="re-render clips that already exist")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    book = phrase_book()
    print(f"rendering {len(book)} clips into {out_dir}/")

    made = skipped = failed = 0
    for key, text in sorted(book.items()):
        path = out_dir / f"{key}.wav"
        if path.exists() and not args.force:
            skipped += 1
            continue
        ok = (
            render_with_command(text, path, args.command)
            if args.command
            else render_with_piper(text, path, args.voice)
        )
        if ok:
            made += 1
            print(f"  {key:<16} {text}")
        else:
            failed += 1
            path.unlink(missing_ok=True)

    print(f"\ndone: {made} rendered, {skipped} already present, {failed} failed")
    if failed:
        print("tip: check the voice is downloaded -- python -m piper.download_voices " + args.voice)
    if not shutil.which("aplay") and not shutil.which("paplay") and not shutil.which("afplay"):
        print("warning: no aplay/paplay/afplay on PATH, so nothing will actually play")


if __name__ == "__main__":
    main()
