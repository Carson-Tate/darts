#!/usr/bin/env python3
"""Run auto-calibration against a photo (or a live camera) and show the result.

This is the first thing to run. Take a phone photo from roughly where the
camera will be mounted, point this at it, and look at the overlay: if the green
rings sit on the real rings and the yellow numbers land on the real numbers,
the hard part is already working.

    python tools/check_calib.py samples/board.jpg
    python tools/check_calib.py --camera 0
    python tools/check_calib.py samples/board.jpg --tune   # HSV mask sliders

Exit code is non-zero if calibration failed, so it also works as a smoke test.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from darts.board import REGULATION, score_at  # noqa: E402
from darts.vision.calibrate import (  # noqa: E402
    YellowRange,
    auto_calibrate,
    debug_overlay,
    fit_board_ellipse,
    yellow_mask,
)


def report(calib) -> None:
    print(f"  rotation match : {calib.score:.3f}")
    print(f"  margin         : {calib.margin:.4f} "
          f"({'confident' if calib.rotation_is_confident else 'AMBIGUOUS'})")
    if not calib.rotation_is_confident:
        print("  -> The ring pattern repeats every 2 sectors; only the printed")
        print("     numbers break the tie. Check the overlay numbers carefully.")

    # Spot-check a few landmarks so a silent rotation error is obvious in text
    # as well as in the picture.
    print("  landmark check :")
    for angle, expect in ((0, 20), (90, 6), (180, 3), (270, 11)):
        th = math.radians(angle)
        r = (REGULATION.triple_inner + REGULATION.triple_outer) / 2
        px, py = calib.board_to_image(r * math.sin(th), r * math.cos(th))
        got = score_at(*calib.image_to_board(px, py))
        flag = "ok" if got.sector == expect else "MISMATCH"
        print(f"     {angle:>3}deg -> {got.label:<4} (expected sector {expect}) {flag}")


def tune(image) -> None:
    """Interactive HSV sliders -- for when the mask isn't picking up the board."""
    win = "yellow mask (q to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    for name, val, hi in (("h_lo", 15, 179), ("h_hi", 42, 179), ("s_lo", 70, 255), ("v_lo", 70, 255)):
        cv2.createTrackbar(name, win, val, hi, lambda _v: None)

    while True:
        rng = YellowRange(*[cv2.getTrackbarPos(n, win) for n in ("h_lo", "h_hi", "s_lo", "v_lo")])
        mask = yellow_mask(image, rng)
        shown = cv2.bitwise_and(image, image, mask=mask)
        ellipse = fit_board_ellipse(mask)
        if ellipse is not None:
            cv2.ellipse(shown, ellipse, (0, 255, 0), 2)
        cv2.imshow(win, shown)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            print(f"\nput this in config.yaml under vision:\n"
                  f"  yellow: {{h_lo: {rng.h_lo}, h_hi: {rng.h_hi}, "
                  f"s_lo: {rng.s_lo}, v_lo: {rng.v_lo}}}")
            return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", nargs="?", help="photo of the empty board")
    parser.add_argument("--camera", type=int, help="grab from this camera index instead")
    parser.add_argument("--tune", action="store_true", help="interactive HSV mask sliders")
    parser.add_argument("-o", "--out", default="calibration_check.jpg", help="where to write the overlay")
    parser.add_argument("--show", action="store_true", help="open a window as well as writing the file")
    args = parser.parse_args()

    if args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        for _ in range(10):  # let autoexposure settle
            ok, image = cap.read()
        cap.release()
        if not ok:
            print(f"could not read from camera {args.camera}", file=sys.stderr)
            return 2
    elif args.image:
        image = cv2.imread(args.image)
        if image is None:
            print(f"could not read {args.image}", file=sys.stderr)
            return 2
    else:
        parser.error("give an image path or --camera N")

    print(f"image: {image.shape[1]}x{image.shape[0]}")

    if args.tune:
        tune(image)
        return 0

    mask = yellow_mask(image)
    covered = cv2.countNonZero(mask) / (image.shape[0] * image.shape[1])
    print(f"  yellow mask    : {covered * 100:.1f}% of the frame")
    if covered < 0.02:
        print("  -> Almost nothing matched. Re-run with --tune to fix the HSV window.")

    calib = auto_calibrate(image)
    if calib is None:
        print("\nCALIBRATION FAILED")
        print("Most common causes: board partly out of frame, a dart still in the")
        print("board, or the yellow window not matching your lighting (--tune).")
        cv2.imwrite("calibration_mask.jpg", mask)
        print("wrote calibration_mask.jpg so you can see what was detected")
        return 1

    print("\nCALIBRATED")
    report(calib)

    overlay = debug_overlay(image, calib)
    cv2.imwrite(args.out, overlay)
    print(f"\nwrote {args.out} -- check the green rings and yellow numbers line up")
    if args.show:
        cv2.imshow("calibration", overlay)
        cv2.waitKey(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
