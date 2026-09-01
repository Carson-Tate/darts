# Darts

Automatic 301 scoring for a wall-cabinet dartboard. Runs on a Raspberry Pi 4,
watches the board with one or two webcams, calls the scores out loud, and is
controlled from a phone browser.

Built for a specific board: regulation scoring geometry, but with the numbers
printed straight into the double ring instead of on a separate wire number ring,
mounted in a shallow wooden cabinet. That board is why the off-the-shelf options
(autodarts, DeepDarts) don't apply — and also why automatic calibration works
unusually well here, since those big printed numerals are exactly the landmark
the calibrator needs.

## Design stance

**The camera suggests; it never dictates.** Every detected dart lands in the UI
with a one-tap correction. A detector that is 90% right behind a one-tap fix is
a good scoreboard. The same detector with no override is an unusable one. The
whole thing also runs as a pure manual scoreboard with no cameras at all — start
there, it's useful on day one.

## Quick start

```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install -r requirements.txt

python run.py                # then open http://<pi>:8000 on your phone
python run.py --no-vision    # manual scoreboard only
```

On Raspberry Pi OS, install OpenCV from apt rather than pip — the wheel build
takes the better part of an hour:

```bash
sudo apt install python3-opencv python3-numpy
```

### Voice callouts

```bash
pip install piper-tts
python -m piper.download_voices en_GB-alba-medium
python tools/render_audio.py
```

Renders every phrase to `sounds/*.wav` once. Playback at game time is just
`aplay`, so there's no synthesis delay between the dart landing and the call.
Without this the game runs fine, silently.

## Getting the camera working

**Do this before mounting anything.** Take a phone photo from roughly where you
plan to put the camera and run:

```bash
python tools/check_calib.py samples/board.jpg
```

It writes `calibration_check.jpg` with the detected rings and sector numbers
drawn on. If the green rings sit on the real rings and the yellow numbers land
on the real numbers, the hard part is done. If the yellow mask isn't finding the
board under your lighting, `--tune` gives you HSV sliders and prints the config
snippet to paste.

### How calibration works

No clicking, no stored reference photo, re-runs on every startup:

1. **Yellow mask** — the board's paint is far more saturated than the wood
   cabinet, so a colour threshold isolates the face.
2. **Convex hull → `fitEllipse`** — the outer edge of the double ring, as an
   ellipse.
3. **Affine rectify** — ellipse to circle. Rotation is still unknown.
4. **Rotational template match** — 20 sector steps × mirror against a
   *synthetically rendered* reference board, best normalised cross-correlation
   wins.
5. **ECC refinement** — upgrades the affine estimate to a full homography.
   Needed because at ~60 cm from a 340 mm board the weak-perspective assumption
   in step 3 is off by several millimetres at the far rim, which is the
   difference between a treble and a single.

**The one thing to know:** the ring pattern alone repeats every *two* sectors —
alternating sector colours with the ring colours inverted is 36°-symmetric. Ten
of the twenty candidate rotations would score identically. The printed numerals
are the only thing that breaks the tie, so they're rendered into the synthetic
reference too. Calibration reports a **margin** over the runner-up; if it's low,
the UI warns you and the **Rotate** button steps the orientation one sector at a
time. Check the overlay numbers after any recalibration.

### Camera placement

Autodarts' published spec, which is worth following: lens at **35–55° to the
board surface**, and their primary camera sits centred on the 6 or the 11 — the
11 is the left-hand position. Too shallow and the board foreshortens to a sliver
(you lose angular resolution); too face-on and you can't separate the dart's tip
from its barrel.

Manual focus, manual exposure. Autofocus will hunt every time a dart flies past,
and an autoexposure step registers as the entire frame changing:

```bash
v4l2-ctl -d /dev/video0 --set-ctrl=focus_automatic_continuous=0
v4l2-ctl -d /dev/video0 --set-ctrl=focus_absolute=40
v4l2-ctl -d /dev/video0 --set-ctrl=auto_exposure=1
```

Adding the second camera is a config change only — uncomment the block in
`config.yaml`. The pipeline fuses whatever calibrates and reports lower
confidence when the two disagree, which nearly always means one of them mistook
the barrel for the point.

## How detection works

Background differencing, not machine learning. After each scored dart the
background is re-baselined to *include* that dart, so every pass is a
single-new-blob problem — no bookkeeping about which of three overlapping shapes
is new, and an occluded dart degrades to "missed it" rather than "rescored the
last one".

Tip localisation is the crux. A dart sticks 30–40 mm out of the board, so from
an off-axis camera the barrel can appear a centimetre from where the point went
in. The heuristic: fit the blob's principal axis, measure the perpendicular
spread at both ends, and take the **narrower** end — a dart tapers at the point
and flares at the flight. That's geometry-driven, so it holds regardless of
which side of the board the dart landed on.

Turn advance is driven by the same signal: a large change (a person at the
board) followed by the board going quiet means the darts were pulled.

## Tests

```bash
pip install pytest
pytest -v
```

`test_board.py` and `test_game.py` are pure Python — geometry, 301 rules, busts,
check-outs, undo. `test_calibration.py` builds a synthetic board, projects it
through a known homography that mimics a camera to the left and below, and
checks the calibrator recovers board coordinates to within a few millimetres. It
also guards the 36°-symmetry bug described above.

## Layout

```
run.py                    entry point
config.yaml               all tuning lives here
darts/
  board.py                geometry, ring radii, pixel -> score
  game.py                 301 engine, busts, check-outs, undo
  audio.py                callout queue + the phrase list
  server.py               FastAPI, websocket, vision bridge
  config.py               config loading
  vision/
    camera.py             capture, 1 or N cameras
    calibrate.py          automatic calibration
    detect.py             background diff, tip localisation, fusion
    pipeline.py           the state machine
static/                   phone UI
tools/
  check_calib.py          calibration smoke test + HSV tuner
  render_audio.py         pre-render the callouts
```

## Status and known limits

Written but **not yet run** — there was no Python available on the machine it
was written on. Expect to shake out import-level mistakes on first launch. Start
with `pytest`, then `tools/check_calib.py` on a photo, then `run.py --no-vision`,
then the full thing.

Realistic accuracy with a single camera on this board is **85–92%** on total
score. Parallax and dart-on-dart occlusion are the limits, and neither is fixable
in software from one viewpoint — the second camera is what addresses occlusion.
The correction flow is not a fallback for this; it's the design.

Not done yet: cricket and other game modes, per-player stats beyond the 3-dart
average, saving match history, systemd unit files.
