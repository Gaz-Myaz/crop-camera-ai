"""
stereo_capture.py

Live split-view of the DECXIN board's two lenses, and a calibration-image
collector. stereo_probe.py confirmed this board outputs ONE combined
frame that's the left and right lens images side by side -- this script
splits that frame in two, shows both halves live, and saves synced
left/right checkerboard pairs for stereo_calibrate.py.

IMPORTANT -- this board's two lenses are NOT identical: the left lens is
noticeably wider-angle than the right (confirmed by inspecting real
frames). Two consequences when capturing:
  * The right (narrower) lens sees LESS of the scene, so it's the one that
    constrains you -- if the whole checkerboard isn't inside the RIGHT
    half's view, the pair is unusable no matter how good the left looks.
  * The board appears SMALLER in the left/wide half, so don't stand too
    far back. Fill a good part of the right half and the left will follow.

Must live in the same folder as stereo_common.py.

--------------------------------------------------------------------
Setup:
    uv pip install opencv-python numpy pygrabber

1. Look at the live split first, no capturing, to confirm it's sane:

    uv run stereo_capture.py --camera-name DECXIN

2. Get a checkerboard in front of the camera. Either PRINT checkerboard.png
   and mount it on something rigid, or -- if you can't print -- DISPLAY it
   full-screen on a tablet/phone/second monitor. A screen works well: it's
   perfectly flat and rigid, which is exactly what the math wants. Three
   rules if you go the screen route:
     * Display it on a DIFFERENT device than the one running this script,
       or you won't be able to see this preview window.
     * Set screen brightness to roughly half and keep lamps/windows from
       reflecting off the glass. A blown-out or glare-striped board is the
       main way screen calibration fails.
     * Measure the squares ON THE GLASS with a ruler (see step 4) -- the
       displayed size is whatever the screen made it, not 20mm.
   Screen moire (fine ripples from photographing a pixel grid) is handled
   automatically by the detector, so you don't need to do anything about it.

   Then capture pairs -- press SPACE whenever BOTH halves report detected:

    uv run stereo_capture.py --camera-name DECXIN --collect

   Aim for 15-25 pairs, varying distance, tilt, and position in frame
   (including corners, not just centered). Press 'q' when done.

3. If detection keeps failing, press 'd' to dump what the camera is
   actually seeing to debug_frames/ -- those images make it possible to
   diagnose why (out of focus? glare? board cut off? too small?).

4. Measure the REAL square size before calibrating. Lay a ruler across the
   full grid (all 10 squares wide), read the total, divide by 10 -- that
   averages out measurement error instead of multiplying it. Then pass the
   result to stereo_calibrate.py --square-size-mm. This number sets the
   absolute scale of every distance the system will ever report, so a 10%
   error here is a 10% error in every measurement forever.

Note the default capture resolution is 2560x720 (two 1280x720 halves),
not the maximum 3840x1080. That's deliberate: the max mode runs at a low
frame rate over USB, which means more motion blur on a handheld board and
laggy detection feedback. 1280x720 per lens is plenty for calibration.
Whatever resolution you calibrate at, stereo_depth.py will reuse
automatically -- so just don't change it between the two steps.
--------------------------------------------------------------------
"""

import argparse
import os
import time

import cv2

import sys as _sys
from pathlib import Path as _Path
# Make ../../common importable when this script is run directly
# (this folder lives one level down, under experiments/).
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "common"))

import camera_utils as cam
from stereo_common import PATTERN_SIZE, detect_corners, split_frame

OUT_DIR = "calib_images"
DEBUG_DIR = "debug_frames"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--camera-name", type=str, default=None)
    parser.add_argument("--width", type=int, default=2560,
                         help="Combined-frame width (default 2560 = two 1280x720 halves).")
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--collect", action="store_true",
                         help="Enable checkerboard detection + SPACE-to-capture. Without "
                              "this flag it's just a live preview of the split.")
    args = parser.parse_args()

    camera_index = args.camera
    if args.camera_name:
        names = cam.get_camera_names()
        match = next(
            (i for i, n in enumerate(names) if args.camera_name.lower() in n.lower()), None
        )
        if match is None:
            print(f"ERROR: no camera name containing '{args.camera_name}' found.")
            return 1
        camera_index = match
        print(f"Matched '{args.camera_name}' -> camera index {camera_index} ({names[match]})")

    cap = cam.open_camera(camera_index, args.width, args.height)
    if not cap.isOpened():
        print(f"ERROR: could not open camera index {camera_index}.")
        return 1

    for _ in range(10):
        cap.read()
        time.sleep(0.05)

    ok, probe = cap.read()
    if ok:
        print(f"Camera is delivering {probe.shape[1]}x{probe.shape[0]} combined "
              f"-> {probe.shape[1]//2}x{probe.shape[0]} per lens.")

    pair_count = 0
    if args.collect:
        os.makedirs(OUT_DIR, exist_ok=True)
        pair_count = len([f for f in os.listdir(OUT_DIR) if f.startswith("left_")])
        print(f"Collecting into '{OUT_DIR}/' (starting at pair {pair_count}). "
              f"SPACE = capture, 'd' = dump debug frames, 'q' = quit.")
    else:
        print("Live split preview (no capture). 'd' = dump debug frames, 'q' = quit. "
              "Rerun with --collect to capture calibration pairs.")

    debug_count = 0
    consecutive_failures = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                consecutive_failures += 1
                if consecutive_failures > 60:
                    print("WARNING: camera read failing repeatedly, stopping.")
                    break
                time.sleep(0.05)
                continue
            consecutive_failures = 0

            left, right = split_frame(frame)

            left_found = right_found = False
            if args.collect:
                left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
                right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
                left_found, left_corners = detect_corners(left_gray)
                right_found, right_corners = detect_corners(right_gray)

                left_disp, right_disp = left.copy(), right.copy()
                if left_found:
                    cv2.drawChessboardCorners(left_disp, PATTERN_SIZE, left_corners, True)
                if right_found:
                    cv2.drawChessboardCorners(right_disp, PATTERN_SIZE, right_corners, True)
            else:
                left_disp, right_disp = left.copy(), right.copy()

            for disp, name, found in (
                (left_disp, "LEFT (wide)", left_found), (right_disp, "RIGHT (narrow)", right_found)
            ):
                if args.collect:
                    color = (0, 255, 0) if found else (0, 0, 255)
                    text = f"{name}: {'DETECTED' if found else 'not detected'}"
                else:
                    color, text = (255, 255, 255), name
                cv2.putText(disp, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            combined = cv2.hconcat([
                cv2.resize(left_disp, (700, 460)), cv2.resize(right_disp, (700, 460))
            ])
            if args.collect:
                cv2.putText(combined, f"Pairs: {pair_count}   SPACE=capture  d=debug dump  q=quit",
                            (10, combined.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 255), 2)
            cv2.imshow("Stereo split - LEFT | RIGHT", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if key == ord("d"):
                os.makedirs(DEBUG_DIR, exist_ok=True)
                lp = f"{DEBUG_DIR}/debug_left_{debug_count:02d}.jpg"
                rp = f"{DEBUG_DIR}/debug_right_{debug_count:02d}.jpg"
                cv2.imwrite(lp, left, [cv2.IMWRITE_JPEG_QUALITY, 88])
                cv2.imwrite(rp, right, [cv2.IMWRITE_JPEG_QUALITY, 88])
                print(f"  debug dump -> {lp} / {rp}  "
                      f"(left detected={left_found}, right detected={right_found})")
                debug_count += 1

            if key == ord(" ") and args.collect:
                if left_found and right_found:
                    cv2.imwrite(f"{OUT_DIR}/left_{pair_count:03d}.png", left)
                    cv2.imwrite(f"{OUT_DIR}/right_{pair_count:03d}.png", right)
                    print(f"  saved pair {pair_count}")
                    pair_count += 1
                else:
                    missing = []
                    if not left_found:
                        missing.append("LEFT")
                    if not right_found:
                        missing.append("RIGHT")
                    print(f"  skipped -- not detected in: {', '.join(missing)}"
                          f"{'  (right lens is narrower -- is the whole board inside it?)' if 'RIGHT' in missing else ''}")
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if args.collect:
        print(f"\nDone. {pair_count} pairs in '{OUT_DIR}/'. Aim for 15-25 before "
              f"running stereo_calibrate.py.")
    if debug_count:
        print(f"{debug_count} debug dump(s) in '{DEBUG_DIR}/'.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
