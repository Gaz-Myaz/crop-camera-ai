"""
stereo_depth.py

Live distance-to-object using the calibration from stereo_calibrate.py.
Splits the combined frame into left/right, rectifies both (so the same
physical point lands on the same row in each half -- required for
disparity matching to work), computes a disparity map, and converts that
to real-world distance via the calibrated geometry.

Reading the window: LEFT panel is the live (rectified) camera view, RIGHT
panel is the disparity map -- warm colours = near, cool = far, black =
"couldn't match here". A crosshair in the middle of the left panel
continuously reports the distance straight ahead, which is the easiest
way to check readings against a tape measure: point, read, done.

  - Move the mouse / click anywhere in EITHER panel to query that point.
  - Press 'p' to print the current crosshair distance to the console.
  - Press 'q' to quit.
  - --with-detection additionally runs YOLOv8 and labels each detected
    object with its distance, e.g. "cow 3.2m".

Distances are the MEDIAN over a small patch, not a single pixel. A lone
pixel on a disparity map is very noisy; a median over its neighbourhood
is far steadier and is what makes the readout usable.

Must live in the same folder as farm_camera_detect.py, alongside the
stereo_calib.npz produced by stereo_calibrate.py.

--------------------------------------------------------------------
Usage:

    uv run stereo_depth.py --camera-name DECXIN
    uv run stereo_depth.py --camera-name DECXIN --with-detection

Aim at something with visible texture or edges. Blank walls, plain
tabletops and uniform surfaces genuinely cannot be measured by any
stereo camera -- with nothing to match between the two views, there is
no disparity to compute. That is physics, not a fault in the code.
--------------------------------------------------------------------
"""

import argparse
import time

import cv2
import numpy as np

import sys as _sys
from pathlib import Path as _Path
# Make ../../common importable when this script is run directly
# (this folder lives one level down, under experiments/).
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "common"))

import camera_utils as cam

# Size each panel is drawn at in the window. The click handler converts
# window coordinates back into full-resolution image coordinates using
# these, so they can be changed freely without breaking the mapping.
PANEL_W, PANEL_H = 640, 360

_pending_click = None   # set by the mouse callback, consumed once by the loop
_marker = None          # last queried point, kept only for drawing


def on_mouse(event, x, y, flags, param):
    global _pending_click
    if event == cv2.EVENT_LBUTTONDOWN:
        _pending_click = (x, y)


def window_to_image(wx, wy, half_w, half_h):
    """Map a click in the displayed window back to a pixel in the
    full-resolution rectified image.

    The window is two side-by-side panels, each a scaled-down copy of a
    half-frame. Both panels show the same scene, so a click in either one
    maps to the same underlying image pixel -- we just subtract the panel
    offset first. Without this, clicks were being used as raw indices into
    the full-resolution array and silently queried the wrong location.
    """
    if wx >= PANEL_W:          # right-hand (disparity) panel
        wx -= PANEL_W
    ix = int(wx * (half_w / PANEL_W))
    iy = int(wy * (half_h / PANEL_H))
    ix = max(0, min(ix, half_w - 1))
    iy = max(0, min(iy, half_h - 1))
    return ix, iy


def load_calibration(path: str):
    data = np.load(path)
    return {k: data[k] for k in data.files}


def build_rectify_maps(calib):
    image_size = tuple(int(v) for v in calib["image_size"])  # (width, height) of ONE half
    map1x, map1y = cv2.initUndistortRectifyMap(
        calib["mtx_l"], calib["dist_l"], calib["R1"], calib["P1"], image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(
        calib["mtx_r"], calib["dist_r"], calib["R2"], calib["P2"], image_size, cv2.CV_32FC1)
    return image_size, (map1x, map1y), (map2x, map2y)


def make_matcher():
    window_size = 5
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,  # must be divisible by 16
        blockSize=window_size,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
    )


def region_distance_m(points_3d, x0, y0, x1, y1):
    """Median distance (metres) within a box, ignoring invalid pixels."""
    h, w = points_3d.shape[:2]
    x0, y0 = max(int(x0), 0), max(int(y0), 0)
    x1, y1 = min(int(x1), w), min(int(y1), h)
    if x1 <= x0 or y1 <= y0:
        return None
    region = points_3d[y0:y1, x0:x1, 2]  # Z channel, in mm (calibration units)
    valid = region[np.isfinite(region) & (region > 0) & (region < 50000)]  # sane 0-50m
    if valid.size < 5:
        return None
    return float(np.median(valid)) / 1000.0  # mm -> m


def point_distance_m(points_3d, x, y, half=5):
    """Distance at a point, taken as the median of a small patch around it."""
    return region_distance_m(points_3d, x - half, y - half, x + half + 1, y + half + 1)


def patch_median_disparity(disp, x, y, half=5):
    """Median of the valid raw disparities in a patch. Used by --check-at to
    work out the disparity offset correction."""
    h, w = disp.shape
    x0, y0 = max(x - half, 0), max(y - half, 0)
    x1, y1 = min(x + half + 1, w), min(y + half + 1, h)
    region = disp[y0:y1, x0:x1]
    valid = region[region > 0]
    if valid.size < 5:
        return None
    return float(np.median(valid))


def main() -> int:
    global _pending_click, _marker

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--camera-name", type=str, default=None)
    parser.add_argument("--calib", type=str, default="stereo_calib.npz")
    parser.add_argument("--with-detection", action="store_true")
    parser.add_argument("--model-weights", type=str, default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--patch", type=int, default=5,
                         help="Half-size of the median patch used per query "
                              "(5 = an 11x11 window). Larger = steadier but blends "
                              "together objects at different depths.")
    parser.add_argument("--disparity-offset", type=float, default=0.0,
                         help="Subtract this many pixels from every disparity before "
                              "converting to distance. Corrects a residual rectification "
                              "shift, which shows up as distances that read short by a "
                              "PERCENTAGE THAT GROWS WITH RANGE. Find the right value "
                              "with --check-at, then pass it here on every run.")
    parser.add_argument("--check-at", type=float, default=None,
                         help="Distance in METRES to the object under the crosshair. "
                              "Press 'c' and the script prints the --disparity-offset "
                              "that would make the reading match.")
    args = parser.parse_args()

    try:
        calib = load_calibration(args.calib)
    except FileNotFoundError:
        print(f"ERROR: '{args.calib}' not found. Run stereo_capture.py --collect, then "
              f"stereo_calibrate.py, first.")
        return 1

    half_size, (map1x, map1y), (map2x, map2y) = build_rectify_maps(calib)
    half_w, half_h = half_size
    combined_w, combined_h = half_w * 2, half_h
    # focal length x baseline: the constant linking disparity to distance,
    # distance_mm = fB / disparity_px
    fB = float(calib["P1"][0, 0]) * abs(float(calib["P2"][0, 3]) / float(calib["P2"][0, 0]))
    print(f"Calibration loaded: baseline={float(calib['baseline_mm']):.1f}mm, "
          f"stereo RMS={float(calib['rms_stereo']):.3f}px, per-lens frame {half_w}x{half_h}.")
    if args.disparity_offset:
        print(f"Applying disparity offset of {args.disparity_offset:+.2f} px.")

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

    cap = cam.open_camera(camera_index, combined_w, combined_h)
    if not cap.isOpened():
        print(f"ERROR: could not open camera index {camera_index}.")
        return 1

    for _ in range(10):
        cap.read()
        time.sleep(0.05)

    model = None
    if args.with_detection:
        print("Loading detection libraries (can take up to a minute on first run)...",
              flush=True)
        from ultralytics import YOLO
        model = YOLO(args.model_weights)

    matcher = make_matcher()
    cv2.namedWindow("Depth")
    cv2.setMouseCallback("Depth", on_mouse)

    print("Running. The crosshair in the centre shows live distance straight ahead.")
    print("Click anywhere to query a point, 'p' to print the crosshair value, 'q' to quit.")

    cx, cy = half_w // 2, half_h // 2
    center_m = None
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

            left = frame[:, :frame.shape[1] // 2]
            right = frame[:, frame.shape[1] // 2:]
            if (left.shape[1], left.shape[0]) != half_size:
                # Camera came up at a resolution other than the calibrated one.
                left = cv2.resize(left, half_size)
                right = cv2.resize(right, half_size)

            rect_left = cv2.remap(left, map1x, map1y, cv2.INTER_LINEAR)
            rect_right = cv2.remap(right, map2x, map2y, cv2.INTER_LINEAR)

            gray_l = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)
            disparity_raw = matcher.compute(gray_l, gray_r).astype(np.float32) / 16.0
            # Empirical correction for residual rectification shift -- see
            # --disparity-offset. Zero by default, so this is a no-op unless asked for.
            disparity = disparity_raw - args.disparity_offset

            points_3d = cv2.reprojectImageTo3D(disparity, calib["Q"])

            disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
            disp_vis = cv2.applyColorMap(disp_vis.astype(np.uint8), cv2.COLORMAP_JET)

            display = rect_left.copy()

            if model is not None:
                result = model.predict(rect_left, conf=args.conf, verbose=False)[0]
                for box in result.boxes:
                    x0, y0, x1, y1 = map(int, box.xyxy[0])
                    label = result.names[int(box.cls)]
                    # Sample the middle of the box: box corners often sit on the
                    # background rather than the object itself.
                    bw, bh = x1 - x0, y1 - y0
                    dist_m = region_distance_m(points_3d,
                                               x0 + bw // 4, y0 + bh // 4,
                                               x1 - bw // 4, y1 - bh // 4)
                    text = f"{label} {dist_m:.2f}m" if dist_m is not None else f"{label} ?"
                    cv2.rectangle(display, (x0, y0), (x1, y1), (0, 255, 0), 2)
                    cv2.putText(display, text, (x0, max(y0 - 8, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # --- centre crosshair: continuous readout, no clicking needed ---
            center_m = point_distance_m(points_3d, cx, cy, args.patch)
            cv2.drawMarker(display, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 28, 2)
            center_text = f"centre: {center_m:.2f} m" if center_m is not None else \
                          "centre: no match (needs texture)"
            cv2.putText(display, center_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2)

            # --- handle a new click, exactly once ---
            if _pending_click is not None:
                wx, wy = _pending_click
                _pending_click = None
                ix, iy = window_to_image(wx, wy, half_w, half_h)
                d = point_distance_m(points_3d, ix, iy, args.patch)
                if d is not None:
                    print(f"  point ({ix},{iy}): {d:.2f} m")
                    _marker = (ix, iy, f"{d:.2f} m")
                else:
                    print(f"  point ({ix},{iy}): no match here -- aim at something "
                          f"with texture or edges")
                    _marker = (ix, iy, "no match")

            if _marker is not None:
                mx, my, mtext = _marker
                cv2.circle(display, (mx, my), 7, (0, 0, 255), 2)
                cv2.putText(display, mtext, (mx + 12, my), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)

            window = cv2.hconcat([
                cv2.resize(display, (PANEL_W, PANEL_H)),
                cv2.resize(disp_vis, (PANEL_W, PANEL_H)),
            ])
            cv2.putText(window, "live view", (10, PANEL_H - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(window, "disparity: warm=near cool=far black=no match",
                        (PANEL_W + 10, PANEL_H - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("Depth", window)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                if center_m is not None:
                    print(f"  CENTRE: {center_m:.2f} m")
                else:
                    print("  CENTRE: no match (aim at something with texture)")
            if key == ord("c"):
                if args.check_at is None:
                    print("  press 'c' needs --check-at <true distance in metres>")
                else:
                    d_raw = patch_median_disparity(disparity_raw, cx, cy, args.patch)
                    if d_raw is None:
                        print("  no disparity at the crosshair -- aim at something textured")
                    else:
                        needed = fB / (args.check_at * 1000.0)
                        offset = d_raw - needed
                        print(f"  measured disparity {d_raw:.2f} px; {args.check_at:.2f} m "
                              f"needs {needed:.2f} px")
                        print(f"  -> re-run with  --disparity-offset {offset:.2f}")
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
