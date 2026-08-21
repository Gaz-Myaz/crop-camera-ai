"""
ndvi_probe.py

Works out how your NDVI camera is wired -- specifically, which colour
channel carries the near-infrared signal -- by looking at real vegetation
rather than trusting a guess. See ndvi_common.py for why this can't just
be assumed.

The test is simple physics. Healthy leaves reflect near-infrared strongly
and absorb visible red and blue. So point the camera at leaves: whichever
of the RED or BLUE channel is dramatically brighter is the one receiving
NIR. A second sample of something non-living (soil, wall, pavement) acts
as a control -- that same channel gap should mostly disappear there. If it
doesn't, the difference is coming from the camera or the lighting rather
than from chlorophyll, and the result shouldn't be trusted.

Must live in the same folder as camera_utils.py and ndvi_common.py.

--------------------------------------------------------------------
Setup:
    uv pip install opencv-python numpy pygrabber   # pygrabber: Windows only

Find the camera first:
    uv run ndvi_probe.py --list-cameras

Then:
    uv run ndvi_probe.py --camera 1
    uv run ndvi_probe.py --camera-name NDVI

  0. LIGHT MATTERS MORE THAN ANYTHING HERE. Near-infrared has to be
     present in the light source before the camera can see it reflected.
     Daylight is full of it; so are incandescent and halogen bulbs. Most
     indoor LED and fluorescent lighting emits almost NONE. Probing a
     plant under office LEDs can therefore produce a flat, inconclusive
     result even with a perfectly good camera. Work near a window, or
     outdoors, or under a halogen lamp.

  1. Fill the centre box with LEAVES, well lit, then press 'v'.
     Real growing plants work best. Avoid dried, dead or artificial
     foliage -- plastic plants reflect no NIR at all, which is in fact a
     neat demonstration of what NDVI actually measures.
  2. Fill the centre box with something NON-living -- soil, concrete,
     a wall -- then press 'n'.
  3. Press 'a' to analyse. The result is written to ndvi_config.json,
     which ndvi_live.py then picks up automatically.

  'q' quits.

IMPORTANT: if your camera's viewer software offers manual white balance
and exposure, set them manually before probing and leave them fixed
afterwards. NDVI is a ratio between colour channels, and auto white
balance continuously rescales those channels independently -- which is
precisely the measurement being made.
--------------------------------------------------------------------
"""

import argparse
import time

import cv2
import numpy as np

import sys as _sys
from pathlib import Path as _Path
# Make ../common importable when this script is run directly.
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "common"))

import camera_utils as cam
from ndvi_common import save_config

from pathlib import Path as _Path
# Snapshots land next to this script, not in whatever directory you
# happened to launch from.
_HERE = _Path(__file__).resolve().parent

ROI_FRAC = 0.30


def centre_roi(frame, frac=ROI_FRAC):
    h, w = frame.shape[:2]
    bw, bh = int(w * frac), int(h * frac)
    x0, y0 = (w - bw) // 2, (h - bh) // 2
    return x0, y0, x0 + bw, y0 + bh


def clipping_fraction(frame, roi, level: int = 250):
    """Fraction of pixels at/near full scale in each channel.

    A clipped channel has thrown away the very information NDVI depends on:
    once a pixel reads 255 you no longer know whether the true value was 255
    or far beyond it, so the ratio between channels -- which IS the
    measurement -- becomes fiction. Any meaningful clipping invalidates the
    result, however convincing the picture looks.
    """
    x0, y0, x1, y1 = roi
    patch = frame[y0:y1, x0:x1]
    b, g, r = cv2.split(patch)
    n = float(patch.shape[0] * patch.shape[1]) or 1.0
    return {"R": float((r >= level).sum()) / n,
            "G": float((g >= level).sum()) / n,
            "B": float((b >= level).sum()) / n}


def channel_means(frame, roi):
    x0, y0, x1, y1 = roi
    patch = frame[y0:y1, x0:x1].astype(np.float32)
    b, g, r = cv2.split(patch)
    return {"R": float(r.mean()), "G": float(g.mean()), "B": float(b.mean())}


def analyse(veg, nonveg, veg_clip=None):
    """Decide which channel carries NIR, and report the reasoning."""
    if veg_clip and max(veg_clip.values()) > 0.02:
        worst = max(veg_clip, key=veg_clip.get)
        print("\n" + "!" * 62)
        print(f"REFUSING TO TRUST THIS SAMPLE: the {worst} channel is CLIPPED "
              f"({veg_clip[worst]*100:.0f}% of pixels at full scale).")
        print("NDVI is a ratio between channels, and a clipped channel has lost")
        print("the magnitude that ratio depends on. Reduce exposure with '-'")
        print("until no channel is clipping, then take the sample again.")
        print("!" * 62)
        return None
    print("\n" + "=" * 62)
    print("CHANNEL ANALYSIS")
    print("=" * 62)
    print(f"{'':14}{'R':>10}{'G':>10}{'B':>10}")
    print(f"{'vegetation':14}{veg['R']:>10.1f}{veg['G']:>10.1f}{veg['B']:>10.1f}")
    if nonveg:
        print(f"{'non-vegetation':14}{nonveg['R']:>10.1f}{nonveg['G']:>10.1f}{nonveg['B']:>10.1f}")

    eps = 1e-6
    veg_rb = veg["R"] / max(veg["B"], eps)
    print(f"\nOn vegetation, R/B = {veg_rb:.2f}")

    if nonveg:
        non_rb = nonveg["R"] / max(nonveg["B"], eps)
        lift = veg_rb / max(non_rb, eps)
        print(f"On non-vegetation, R/B = {non_rb:.2f}")
        print(f"Vegetation lift = {lift:.2f}x "
              f"(how much MORE lopsided vegetation is than dead material)")
    else:
        lift = None
        print("(no non-vegetation control sample -- press 'n' on soil or a wall "
              "for a stronger conclusion)")

    if veg_rb > 1.25:
        nir = "R"
        verdict = ("RED channel carries NIR -> BLUE-filter ('superblue') conversion.\n"
                   "  NDVI = (R - B) / (R + B)")
    elif veg_rb < 0.80:
        nir = "B"
        verdict = ("BLUE channel carries NIR -> RED-filter (Wratten 25A style) conversion.\n"
                   "  NDVI = (B - R) / (B + R)")
    else:
        nir = None
        verdict = None

    print("-" * 62)
    if nir is None:
        print("INCONCLUSIVE: R and B are too similar on vegetation.")
        print("Vegetation should look strongly lopsided between them. Likely causes:")
        print("  * NO INFRARED IN YOUR LIGHTING -- most indoor LED and fluorescent")
        print("    lamps emit almost no NIR, so there is nothing for the leaf to")
        print("    reflect. Retry in daylight, near a window, or under halogen.")
        print("  * the centre box wasn't actually filled with live foliage")
        print("  * too little light, or the subject is in deep shade")
        print("  * auto white balance is neutralising the very difference we need")
        print("  * this may not be an NDVI-converted camera at all -- an ordinary")
        print("    webcam blocks NIR and would give exactly this result")
        print("\nNothing written. Try again with brighter light and real leaves.")
        return None

    print(f"CONCLUSION: {verdict}")
    if lift is not None and lift < 1.15:
        print("\n  CAUTION: the non-vegetation control shows a similar imbalance,")
        print("  so this may be a camera/lighting colour cast rather than genuine")
        print("  chlorophyll response. Re-probe with brighter, more natural light")
        print("  before relying on the numbers.")
    print("=" * 62)

    notes = (f"veg R/B={veg_rb:.2f}"
             + (f", nonveg R/B={nonveg['R']/max(nonveg['B'],eps):.2f}" if nonveg else "")
             + (f", lift={lift:.2f}x" if lift else ""))
    save_config(nir, notes)
    print(f"\nSaved to ndvi_config.json (nir_channel={nir}).")
    print("Now run:  uv run ndvi_live.py --camera-name <your camera>")
    return nir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--camera-name", type=str, default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--list-cameras", action="store_true")
    parser.add_argument("--set-nir", type=str, default=None, choices=["R", "B"],
                         help="Write ndvi_config.json directly, without probing "
                              "vegetation. Use when you already know the conversion "
                              "type (e.g. inferred from the colour cast) and have no "
                              "plants to hand. Confirm against real foliage later.")
    args = parser.parse_args()

    if args.set_nir:
        save_config(args.set_nir, "set manually, not confirmed against vegetation")
        print(f"Wrote ndvi_config.json with nir_channel={args.set_nir}.")
        print("NOTE: this was NOT verified against real foliage. Re-run the full")
        print("probe on live plants when you can -- if it is wrong, NDVI values")
        print("come out sign-inverted and look entirely plausible.")
        return 0

    if args.list_cameras:
        print("Probing camera indices 0-7 ...")
        for idx, name, res in cam.list_cameras():
            label = name if name else "(name unavailable on this platform)"
            print(f"  index {idx}: {label}  [{res}]")
        return 0

    camera_index = args.camera
    if args.camera_name:
        names = cam.get_camera_names()
        match = next(
            (i for i, n in enumerate(names) if args.camera_name.lower() in n.lower()), None)
        if match is None:
            print(f"ERROR: no camera name containing '{args.camera_name}' found.")
            print("Run with --list-cameras to see what's available.")
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
        h, w = probe.shape[:2]
        opened_name = cam.describe_camera(camera_index)
        print(f"Opened camera index {camera_index}"
              + (f": {opened_name}" if opened_name else " (name unavailable)")
              + f", delivering {w}x{h}.")
        if w >= 2 * h:
            print("  NOTE: unusually wide frame -- this may be a DUAL-SENSOR camera "
                  "(separate NIR and RGB images side by side) rather than a filter "
                  "conversion. Tell me if the preview shows two images and I'll "
                  "handle it differently.")

    print("\nFill the centre box, then:  'v' = vegetation,  'n' = non-vegetation,")
    print("'a' = analyse,  's' = save snapshot,  'q' = quit.")
    print("'-' / '+' = darken / brighten exposure (get every channel out of clipping first)")

    # Switch to manual exposure so '-' and '+' actually do something. Many
    # drivers accept these calls and quietly ignore them.
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    print(f"exposure now {exposure} (manual mode requested)")

    veg = nonveg = None
    veg_clip = None
    fails = 0
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

            roi = centre_roi(frame)
            disp = frame.copy()
            cv2.rectangle(disp, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 255), 2)
            m = channel_means(frame, roi)
            clip = clipping_fraction(frame, roi)
            cv2.putText(disp, f"R={m['R']:.0f}  G={m['G']:.0f}  B={m['B']:.0f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            if max(clip.values()) > 0.02:
                worst = max(clip, key=clip.get)
                cv2.putText(disp, f"{worst} CLIPPED {clip[worst]*100:.0f}% - press '-'",
                            (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(disp,
                        f"vegetation:{'SET' if veg else '--'}  "
                        f"non-veg:{'SET' if nonveg else '--'}   v/n/a/q",
                        (10, disp.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)

            b, g, r = cv2.split(frame)
            grid = cv2.vconcat([
                cv2.hconcat([cv2.resize(disp, (480, 270)),
                             cv2.cvtColor(cv2.resize(r, (480, 270)), cv2.COLOR_GRAY2BGR)]),
                cv2.hconcat([cv2.cvtColor(cv2.resize(g, (480, 270)), cv2.COLOR_GRAY2BGR),
                             cv2.cvtColor(cv2.resize(b, (480, 270)), cv2.COLOR_GRAY2BGR)]),
            ])
            for text, org in (("live", (8, 20)), ("RED channel", (488, 20)),
                              ("GREEN channel", (8, 290)), ("BLUE channel", (488, 290))):
                cv2.putText(grid, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.imshow("NDVI probe - live | R | G | B", grid)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                from datetime import datetime as _dt
                ts = _dt.now().strftime("%Y%m%d_%H%M%S")
                raw_path = str(_HERE / f"ndvi_snapshot_{ts}.png")
                grid_path = str(_HERE / f"ndvi_channels_{ts}.png")
                cv2.imwrite(raw_path, frame)
                cv2.imwrite(grid_path, grid)
                print(f"  saved {raw_path}")
                print(f"  saved {grid_path}  (live | R | G | B montage)")
            if key in (ord("-"), ord("_")):
                exposure -= 1
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
                print(f"  exposure -> {exposure}  (actual: {cap.get(cv2.CAP_PROP_EXPOSURE)})")
            if key in (ord("+"), ord("=")):
                exposure += 1
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
                print(f"  exposure -> {exposure}  (actual: {cap.get(cv2.CAP_PROP_EXPOSURE)})")
            if key == ord("v"):
                veg = m.copy()
                veg_clip = clip.copy()
                print(f"  vegetation sample:     R={m['R']:.1f} G={m['G']:.1f} B={m['B']:.1f}"
                      f"   clipping R/G/B = {clip['R']*100:.0f}/{clip['G']*100:.0f}/{clip['B']*100:.0f}%")
            if key == ord("n"):
                nonveg = m.copy()
                print(f"  non-vegetation sample: R={m['R']:.1f} G={m['G']:.1f} B={m['B']:.1f}")
            if key == ord("a"):
                if veg is None:
                    print("  need a vegetation sample first -- aim at leaves and press 'v'")
                else:
                    analyse(veg, nonveg, veg_clip)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
