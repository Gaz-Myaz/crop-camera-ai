"""
ndvi_live.py

Live NDVI from the converted camera: shows the raw view alongside a
false-colour vegetation-health map, and reports how much of the frame is
vegetation and how healthy it looks.

Reading the display: brown/grey = bare ground or non-living material,
yellow = threshold/sparse, green = vegetation, deeper green = more vigorous.
Black means "no valid data" (too dark to measure) -- which is deliberately
distinct from "measured zero", since confusing those two is an easy way to
misread a scene.

Run ndvi_probe.py first; it writes ndvi_config.json telling this script
which channel carries NIR. Override with --nir-channel if you already know.

Must live in the same folder as camera_utils.py and ndvi_common.py.

--------------------------------------------------------------------
Usage:

    uv run ndvi_live.py --camera-name NDVI
    uv run ndvi_live.py --camera 1 --nir-channel R
    uv run ndvi_live.py --camera 1 --lock-exposure

Keys:  'p' print current stats   's' save a snapshot   'q' quit

--lock-exposure attempts to disable auto-exposure and auto white balance.
This matters more than it sounds: NDVI is a ratio between colour channels,
and auto white balance exists precisely to rescale those channels against
each other, so leaving it on means the camera is continuously altering the
quantity being measured. Not every camera or backend honours the request --
the script reports whether it took effect.

Values are RELATIVE, not scientific absolute NDVI -- see ndvi_common.py for
the reasons. Good for comparing plants within a frame or tracking one plant
over time; not comparable to published or satellite NDVI figures.
--------------------------------------------------------------------
"""

import argparse
import time
from datetime import datetime

import cv2
import numpy as np

import sys as _sys
from pathlib import Path as _Path
# Make ../common importable when this script is run directly.
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "common"))

import camera_utils as cam
from ndvi_common import (DEFAULT_VEG_THRESHOLD, colorize_ndvi, compute_ndvi,
                         load_config, vegetation_stats)


def try_lock_exposure(cap) -> None:
    """Best-effort: turn off auto exposure and auto white balance."""
    results = []
    # 0.25 is the magic 'manual' value for many DirectShow/V4L2 drivers.
    ok_ae = cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    results.append(f"auto-exposure off: {'requested' if ok_ae else 'NOT SUPPORTED'}")
    try:
        ok_wb = cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        results.append(f"auto-white-balance off: {'requested' if ok_wb else 'NOT SUPPORTED'}")
    except Exception:
        results.append("auto-white-balance off: NOT SUPPORTED")
    for r in results:
        print(f"  {r}")
    print("  (drivers often accept the call and ignore it -- if NDVI values drift "
          "as you move the camera, set exposure/WB manually in the vendor app)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--camera-name", type=str, default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--nir-channel", type=str, default=None, choices=["R", "B"],
                         help="Override the channel carrying NIR. Normally read "
                              "from ndvi_config.json written by ndvi_probe.py.")
    parser.add_argument("--gamma", type=float, default=2.2,
                         help="Gamma to undo before computing NDVI (default 2.2). "
                              "Pass 0 to skip linearisation.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_VEG_THRESHOLD,
                         help="NDVI above which a pixel counts as vegetation.")
    parser.add_argument("--lock-exposure", action="store_true")
    parser.add_argument("--list-cameras", action="store_true")
    args = parser.parse_args()

    if args.list_cameras:
        print("Probing camera indices 0-7 ...")
        cam.print_camera_list()
        return 0

    nir_channel = args.nir_channel
    if nir_channel is None:
        cfg = load_config()
        if cfg is None:
            print("ERROR: no ndvi_config.json found and --nir-channel not given.")
            print("Run this first:  uv run ndvi_probe.py --camera <index>")
            print("(It determines which channel carries NIR by looking at real "
                  "vegetation -- guessing produces convincing but inverted results.)")
            return 1
        nir_channel = cfg["nir_channel"]
        print(f"Using nir_channel={nir_channel} from ndvi_config.json"
              + (f"  [{cfg.get('notes','')}]" if cfg.get("notes") else ""))

    vis_channel = "B" if nir_channel == "R" else "R"
    print(f"NDVI = ({nir_channel} - {vis_channel}) / ({nir_channel} + {vis_channel})"
          f"   gamma={'off' if args.gamma <= 0 else args.gamma}")

    camera_index = args.camera
    if args.camera_name:
        names = cam.get_camera_names()
        match = next(
            (i for i, n in enumerate(names) if args.camera_name.lower() in n.lower()), None)
        if match is None:
            print(f"ERROR: no camera name containing '{args.camera_name}' found.")
            return 1
        camera_index = match
        print(f"Matched '{args.camera_name}' -> camera index {camera_index} ({names[match]})")

    cap = cam.open_camera(camera_index, args.width, args.height)
    if not cap.isOpened():
        print(f"ERROR: could not open camera index {camera_index}.")
        return 1

    if args.lock_exposure:
        print("Attempting to lock exposure / white balance:")
        try_lock_exposure(cap)

    for _ in range(10):
        cap.read()
        time.sleep(0.05)

    opened_name = cam.describe_camera(camera_index)
    print(f"Opened camera index {camera_index}"
          + (f": {opened_name}" if opened_name else " (name unavailable)"))
    ok, first = cap.read()
    if ok:
        b0, g0, r0 = (float(c.mean()) for c in cv2.split(first))
        lo, hi = min(r0, g0, b0), max(r0, g0, b0)
        print(f"  channel means  R={r0:.0f}  G={g0:.0f}  B={b0:.0f}")
        if hi > 0 and lo / hi > 0.5:
            print("  " + "!" * 58)
            print("  WARNING: these channels are roughly BALANCED, which is what an")
            print("  ORDINARY camera produces. An infrared-converted camera has a")
            print("  strong colour cast -- yours reads about R=243 G=2 B=29.")
            print("  You have most likely opened the wrong camera. USB indices move")
            print("  when devices are plugged in or out; prefer --camera-name.")
            print("  Run:  uv run ndvi_live.py --list-cameras")
            print("  " + "!" * 58)

    print("\nRunning.  'p' = print stats,  's' = save snapshot,  'q' = quit.")

    fails = 0
    stats = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                fails += 1
                if fails > 60:
                    print("WARNING: camera read failing repeatedly, stopping.")
                    break
                time.sleep(0.05)
                continue
            fails = 0

            ndvi, valid = compute_ndvi(frame, nir_channel, args.gamma)
            colored = colorize_ndvi(ndvi, valid)
            stats = vegetation_stats(ndvi, valid, args.threshold)

            overlay = colored.copy()
            if stats and stats["mean_ndvi"] is not None:
                line1 = f"mean NDVI (vegetation only): {stats['mean_ndvi']:+.3f}"
                line2 = (f"coverage: {stats['coverage']*100:.1f}%   "
                         f"range p10-p90: {stats['p10']:+.2f} to {stats['p90']:+.2f}")
            elif stats:
                line1 = "no vegetation detected in frame"
                line2 = f"(nothing above NDVI {args.threshold:+.2f})"
            else:
                line1 = "no valid data -- too dark to measure"
                line2 = ""
            cv2.putText(overlay, line1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (255, 255, 255), 2)
            if line2:
                cv2.putText(overlay, line2, (10, 58), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 2)

            window = cv2.hconcat([
                cv2.resize(frame, (640, 480)),
                cv2.resize(overlay, (640, 480)),
            ])
            cv2.putText(window, "raw camera", (10, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(window, "NDVI  brown=bare  yellow=sparse  green=healthy",
                        (650, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("NDVI", window)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                if stats and stats["mean_ndvi"] is not None:
                    print(f"  mean NDVI {stats['mean_ndvi']:+.3f}  "
                          f"coverage {stats['coverage']*100:.1f}%  "
                          f"p10 {stats['p10']:+.3f}  p90 {stats['p90']:+.3f}")
                else:
                    print("  no vegetation in view")
            if key == ord("s"):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"ndvi_raw_{ts}.png", frame)
                cv2.imwrite(f"ndvi_map_{ts}.png", overlay)
                print(f"  saved ndvi_raw_{ts}.png and ndvi_map_{ts}.png")
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
