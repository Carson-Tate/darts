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

That's 256 clips and takes roughly 25 minutes on a Pi 4 — run it detached
(`nohup ... &`) rather than over an SSH session you might drop. It skips clips
that already exist, so if it's interrupted just run it again to resume.

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

### Two cameras

Both are configured in `config.yaml`. The pipeline fuses whatever calibrates and
reports lower confidence when they disagree, which nearly always means one of
them mistook the barrel for the point. The first camera listed is the primary:
it drives the trigger, and the game runs on it alone if the other never
calibrates.

Identify cameras by `name_hint`, matched against `/dev/v4l/by-id`, not by index.
`/dev/video0` and `/dev/video2` are handed out in enumeration order, so a reboot
can swap which webcam is which — and nothing looks broken when it happens: both
still calibrate and both still score, each using the other's view of the board.

Three things only matter once there are two:

* **The second camera sees the room.** Ours looks down from about three feet
  above the first and takes in a doorway, a fridge and the dart holders on the
  cabinet doors. To a differencing detector a person walking past is a large,
  dark, elongated blob — the same description as a dart. Calibration knows where
  the board is, so detection is confined to it.
* **They can disagree about orientation.** Each resolves the board's 36° symmetry
  against its own view, so one can lock correctly while the other is two sectors
  out. Rotate is therefore per camera, and so is the confidence reported in the
  UI. When a camera is right but not confident, "Looks right" records that.
* **Exposure settles at its own pace.** A camera with a bright doorway in frame
  can still be hunting when calibration first runs — ours scored 0.27 against a
  0.35 gate at startup and 0.74 a minute later. The pipeline keeps offering a
  straggler another go every 20s, with an empty board, without disturbing the
  camera that already works.

Both views are shown on the site continuously. Tiles are fetched at roughly the
size they're drawn (~14KB a frame against ~150KB for a full one); tapping one
enlarges it and fetches it at a size worth looking closely at.

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

Running on the real board. The game engine, phone UI, spoken callouts, dart
detection and turn-advance-on-removal all work end to end. **Automatic
orientation does not yet.**

The orientation lock is the open problem, and it is a mask problem rather than a
search problem. Measured on the real board:

- Ring radii are exactly regulation — transitions at 96–100, 106–110, 162–164
  and 170–172 mm. The geometry model is right.
- The numerals are **not** in the double ring. On a board with no separate
  number ring they are printed inside the outer single band, at 128–149 mm.
  36-degree self-similarity there is +0.41 against +0.95 twenty millimetres
  further out, which is how they were located. See `number_radius`.
- The yellow mask is the bottleneck. A correct mask of an alternating board
  should read a yellow fraction of 0.50 at every radius; the real one reads
  0.30–0.45, so roughly a quarter of the board's yellow is being thrown away —
  and thin, low-contrast numeral strokes are the first thing lost.

That last point is a vice. `s_lo` low enough to keep the numerals also swallows
the tan cardboard beside the board, and the fitted ellipse then spans the whole
scene; `s_lo` high enough to reject the cardboard erases the numerals. One
global threshold cannot serve both stages — the same shape of bug as the
morphology that used to erase the numerals before the rotation search saw them.

The fix is to decouple them: threshold hard to *find* the board, then crop to
it and threshold adaptively *inside* it, where the cardboard is not a
competitor. Until that lands, the rotation is a coin flip between the ten
symmetry-equivalent orientations and needs the Rotate button.

Realistic accuracy with a single camera on this board is **85–92%** on total
score. Parallax and dart-on-dart occlusion are the limits, and neither is fixable
in software from one viewpoint. The second camera is now mounted and fused, and
it should help with both — a dart occluded from one angle is usually visible
from the other, and the two misjudge a dart's protrusion in opposite directions
so the average beats either alone. That is the argument, not a measurement:
two-camera accuracy on this board has not been measured yet. The correction flow
is not a fallback for any of this; it's the design.

### Can it be trained?

Every correction you make is appended to `data/corrections.jsonl` with the
per-camera millimetre readings that produced it. That is the only ground truth
this system can get: a wrong score looks, from the inside, exactly like a right
one, and the tap that fixes it is the one moment a human says what was true.

Nothing reads that file yet, on purpose. Three levels are possible and they are
worth keeping apart:

* **Fix the inputs.** Most of what has looked like "the model is wrong" was not.
  It was an overhead camera underexposed by 16×, a 170mm threshold with no
  tolerance for a dart in the double, and averaging two estimates that were a
  whole dart length apart. Training on data from an underexposed camera teaches
  a model to be confidently wrong. This is where the wins have been so far.
* **Fit a correction from the log.** With 50–100 corrections you can ask whether
  the residual error is *systematic* — every dart reading a few mm too far out,
  a small rotational bias, one camera consistently worse, errors clustering
  where the view is most oblique. That is a handful of parameters, not a neural
  network, and it is the next thing worth doing once there is data.
* **Learn it end to end.** DeepDarts does this and is open source. It was
  trained on standard boards and its own authors report accuracy falling away
  on setups outside that distribution; this board is not a standard board, so it
  would want thousands of labelled images of *this* board under *this* lighting,
  and a Pi 4 CPU would run it at a couple of frames a second. Not ruled out,
  but not the cheap win it sounds like.

The honest order is: fix the optics, measure, then model — and only model the
part that measurement shows is systematic.

Not done yet: cricket and other game modes, per-player stats beyond the 3-dart
average, saving match history, systemd unit files.
