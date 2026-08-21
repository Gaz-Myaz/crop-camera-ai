"""
stereo_calibrate.py

Takes the left/right checkerboard image pairs captured by
stereo_capture.py --collect and solves for everything needed to turn
disparity into real-world distance: per-lens focal length and distortion,
the relative rotation/translation between the two lenses (which includes
the physical baseline -- this is computed from the checkerboard geometry,
you don't need to hand-measure it), and rectification maps.

Must live in the same folder as stereo_capture.py's calib_images/ output.

--------------------------------------------------------------------
Usage:

    uv run stereo_calibrate.py
    uv run stereo_calibrate.py --square-size-mm 19.4   # if your printed
                                                          # squares measured
                                                          # differently than
                                                          # the 20mm nominal

Produces 'stereo_calib.npz', consumed by stereo_depth.py.
--------------------------------------------------------------------
"""

import argparse
import glob
import os

import cv2
import numpy as np

# Same detector and pattern definition stereo_capture.py used when deciding
# these frames were worth saving -- shared so the two can't disagree.
from stereo_common import PATTERN_SIZE, detect_corners

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)


def find_pairs(calib_dir: str):
    lefts = sorted(glob.glob(os.path.join(calib_dir, "left_*.png")))
    pairs = []
    for lp in lefts:
        rp = lp.replace("left_", "right_")
        if os.path.exists(rp):
            pairs.append((lp, rp))
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calib-dir", type=str, default="calib_images")
    parser.add_argument("--square-size-mm", type=float, default=20.0,
                         help="REAL size of one checkerboard square in mm -- measure the "
                              "printed board with a ruler and use the actual value, not "
                              "just the nominal default, if they differ.")
    parser.add_argument("--out", type=str, default="stereo_calib.npz")
    args = parser.parse_args()

    pairs = find_pairs(args.calib_dir)
    print(f"Found {len(pairs)} left/right pairs in '{args.calib_dir}'.")
    if len(pairs) < 10:
        print("WARNING: fewer than 10 pairs -- calibration quality will likely be poor. "
              "Capture more with stereo_capture.py --collect (aim for 15-25, varied "
              "distance/angle/position) before trusting the result.")
    if not pairs:
        print("ERROR: no pairs found. Run stereo_capture.py --collect first.")
        return 1

    objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2)
    objp *= args.square_size_mm

    objpoints = []
    imgpoints_left = []
    imgpoints_right = []
    image_size = None

    used = 0
    for lp, rp in pairs:
        left = cv2.imread(lp)
        right = cv2.imread(rp)
        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (left_gray.shape[1], left_gray.shape[0])

        lf, lc = detect_corners(left_gray)
        rf, rc = detect_corners(right_gray)
        if not (lf and rf):
            print(f"  skipping {os.path.basename(lp)}: checkerboard not found in "
                  f"{'left' if not lf else 'right'} image")
            continue

        objpoints.append(objp)
        imgpoints_left.append(lc)
        imgpoints_right.append(rc)
        used += 1

    print(f"Using {used}/{len(pairs)} pairs where the checkerboard was found in both halves.")
    if used < 8:
        print("ERROR: too few usable pairs to calibrate reliably. Capture more.")
        return 1

    print("Calibrating left lens...")
    rms_l, mtx_l, dist_l, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints_left, image_size, None, None)
    print(f"  left reprojection RMS error: {rms_l:.3f} px")

    print("Calibrating right lens...")
    rms_r, mtx_r, dist_r, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints_right, image_size, None, None)
    print(f"  right reprojection RMS error: {rms_r:.3f} px")

    print("Solving stereo geometry (relative position between the two lenses)...")
    stereo_flags = cv2.CALIB_FIX_INTRINSIC
    rms_stereo, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        mtx_l, dist_l, mtx_r, dist_r, image_size,
        criteria=CRITERIA, flags=stereo_flags,
    )

    baseline_mm = float(np.linalg.norm(T))
    print(f"  stereo reprojection RMS error: {rms_stereo:.3f} px")
    print(f"  computed baseline (distance between lenses): {baseline_mm:.1f} mm")
    if rms_stereo > 1.0:
        print("  WARNING: RMS error above 1.0px usually means noisy/unreliable distance "
              "estimates later. Consider capturing more pairs with better variety "
              "(different distances, angles, and positions in frame, sharp focus, "
              "good even lighting), then recalibrating.")

    print("Computing rectification (so left/right rows line up for disparity matching)...")
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtx_l, dist_l, mtx_r, dist_r, image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
    )

    np.savez(
        args.out,
        mtx_l=mtx_l, dist_l=dist_l, mtx_r=mtx_r, dist_r=dist_r,
        R=R, T=T, E=E, F=F, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
        roi1=np.array(roi1), roi2=np.array(roi2),
        image_size=np.array(image_size),
        baseline_mm=baseline_mm, rms_stereo=rms_stereo,
        square_size_mm=args.square_size_mm,
    )
    print(f"\nSaved calibration to '{args.out}'. Sanity-check: measure the real distance "
          f"between the two lens centers on the board with a ruler and compare to the "
          f"computed {baseline_mm:.1f}mm above -- they should be close (within a few mm). "
          f"A big mismatch usually means --square-size-mm was wrong.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
