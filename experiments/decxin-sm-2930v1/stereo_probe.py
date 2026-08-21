"""
stereo_probe.py

Diagnostic tool for figuring out HOW your 2-lens DECXIN board exposes its
two cameras to the OS -- which determines how distance-to-object
(stereo depth) would actually be implemented.

Binocular/stereo USB modules like this one commonly work one of two ways:

  (a) ONE UVC device that outputs a single wide frame which is actually
      two images stitched side-by-side (left half from one lens, right
      half from the other) -- very common on cheap stereo boards, and
      would explain why only one "DECXIN Camera" showed up in
      --list-cameras earlier.
  (b) TWO separate UVC devices/indices, one per lens, each behaving like
      an independent camera.

This script checks both possibilities without assuming either. Run it,
read the printed output, open the saved snapshot image(s), and tell me
what you see -- that decides which depth-estimation approach to build
next (mode (a) needs splitting one frame in half; mode (b) needs reading
two VideoCaptures in sync).

Must live in the same folder as farm_camera_detect.py.

--------------------------------------------------------------------
Usage:

    uv run stereo_probe.py
        # lists every camera name/index Windows sees, flags how many
        # look like DECXIN entries, and if there's exactly one, probes
        # it for resolutions automatically.

    uv run stereo_probe.py --camera 1
        # probes a specific index directly (use this if the automatic
        # detection above doesn't pick the right one, or to check a
        # second index in mode (b)).
--------------------------------------------------------------------
"""

import argparse
import time

import cv2

import sys as _sys
from pathlib import Path as _Path
# Make ../../common importable when this script is run directly
# (this folder lives one level down, under experiments/).
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "common"))

import camera_utils as cam

# Resolutions worth trying: normal single-lens sizes, and roughly-double-
# width sizes a stitched stereo frame would plausibly use.
CANDIDATE_RESOLUTIONS = [
    (640, 480), (1280, 480), (1280, 720),
    (2560, 720), (2560, 960), (3200, 1200),
    (1920, 1080), (3840, 1080),
]


def probe_names():
    names = cam.get_camera_names()
    if names:
        print("Cameras visible to Windows (by name):")
        for i, n in enumerate(names):
            print(f"  index {i}: {n}")
        decxin_indices = [i for i, n in enumerate(names) if "decxin" in n.lower()]
        print(f"\n-> {len(decxin_indices)} entrie(s) with 'DECXIN' in the name: {decxin_indices}")
        if len(decxin_indices) >= 2:
            print("   Multiple DECXIN entries -- likely mode (b): each lens is its own "
                  "device/index. Try --camera on each of those indices.")
        elif len(decxin_indices) == 1:
            print("   Only one DECXIN entry -- likely mode (a): one device with a combined "
                  "wide frame. Probing its resolutions now...")
        return names, decxin_indices
    print("Camera names unavailable (install 'pygrabber' on Windows, or check manually in "
          "Device Manager -- look under 'Cameras'/'Imaging devices' for how many DECXIN "
          "entries show up). Pass --camera <index> to probe directly.")
    return [], []


def probe_resolutions(camera_index: int):
    print(f"\nProbing resolutions on camera index {camera_index}...")
    cap = cam.open_camera(camera_index, 640, 480)
    if not cap.isOpened():
        print("ERROR: could not open that camera index.")
        return

    for _ in range(10):
        cap.read()
        time.sleep(0.05)

    working = []
    for w, h in CANDIDATE_RESOLUTIONS:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        time.sleep(0.1)
        reported_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        reported_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok, frame = cap.read()
        actual = (frame.shape[1], frame.shape[0]) if ok else None
        print(f"  requested {w}x{h} -> camera reports {reported_w}x{reported_h}"
              f"{f', actual captured frame {actual[0]}x{actual[1]}' if actual else ' (read failed)'}")
        if ok and actual and not any(actual == a for a, _ in working):
            working.append((actual, frame.copy()))

    cap.release()

    if not working:
        print("No resolutions produced a readable frame from this index.")
        return

    widest = max(working, key=lambda item: item[0][0])
    (w, h), frame = widest
    out_path = f"stereo_probe_index{camera_index}_{w}x{h}.png"
    cv2.imwrite(out_path, frame)
    aspect = w / h
    print(f"\nSaved the widest working frame ({w}x{h}, aspect ratio {aspect:.2f}) to "
          f"'{out_path}'.")
    print("Open it and check: if it clearly shows TWO overlapping views of the same scene "
          "side by side, that's a stitched stereo frame (mode a) -- the wide aspect ratio "
          "here is the telltale sign versus a normal single-lens 4:3 or 16:9 shot. If it "
          "just looks like one ordinary photo, this index/resolution is a single lens.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=None,
                         help="Probe this specific camera index directly. If omitted, "
                              "lists names first and auto-probes if exactly one DECXIN "
                              "entry is found.")
    args = parser.parse_args()

    if args.camera is not None:
        probe_resolutions(args.camera)
        return

    names, decxin_indices = probe_names()
    if len(decxin_indices) == 1:
        probe_resolutions(decxin_indices[0])
    elif len(decxin_indices) >= 2:
        for i in decxin_indices:
            probe_resolutions(i)
    elif not names:
        print("\nNo camera names available to auto-detect -- rerun with "
              "--camera <index> (see stereo_probe.py --help).")


if __name__ == "__main__":
    main()
