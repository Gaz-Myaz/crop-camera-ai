"""
stereo_common.py

Shared pieces for the stereo pipeline, so stereo_capture.py (which decides
whether a frame is worth saving) and stereo_calibrate.py (which re-detects
corners in those saved frames) use EXACTLY the same detector. If they
disagree, you get frames that pass at capture time and then get thrown out
at calibration time -- which is confusing and wastes your effort.

Detection notes: this uses OpenCV's findChessboardCornersSB ("sector based")
as the primary detector, falling back to the older findChessboardCorners
only if SB fails. SB is significantly more robust to blur, glare, uneven
lighting, and perspective than the legacy one, and returns sub-pixel
accurate corners directly. The legacy CALIB_CB_FAST_CHECK flag is
deliberately NOT used anywhere -- it's a speed heuristic that produces
false negatives, which is exactly the failure mode we don't want.
"""

import cv2

# 9x6 INTERNAL corners == the 10x7 square grid generate_checkerboard.py draws.
PATTERN_SIZE = (9, 6)

_SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)


def split_frame(frame):
    """This board reports one wide frame that is really the left and right
    lens images side by side (confirmed via stereo_probe.py). Split it."""
    mid = frame.shape[1] // 2
    return frame[:, :mid], frame[:, mid:]


def detect_corners(gray):
    """Find the checkerboard. Returns (found, corners) with sub-pixel corners.

    Tries several strategies in order, cheapest/most-accurate first. All of
    them run automatically -- there are no flags to get wrong, and crucially
    capture and calibration therefore always agree on what's detectable.

      1. SB detector at full resolution.
      2. SB on a 2x-downscaled image (helps when the board is large or soft).
      3. SB after a mild blur -- this is the one that rescues a checkerboard
         DISPLAYED ON A SCREEN rather than printed. A camera photographing a
         monitor picks up the display's pixel grid, which beats against the
         sensor's own grid and produces moire: fine ripples that wreck corner
         detection. A small blur removes that high-frequency interference
         while leaving the checkerboard's much larger squares untouched.
      4. Legacy findChessboardCorners + cornerSubPix, as a last resort.
    """
    try:
        found, corners = cv2.findChessboardCornersSB(
            gray, PATTERN_SIZE,
            flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
        )
        if found:
            return True, corners
    except Exception:
        pass  # very old OpenCV without SB -- fall through to legacy

    # SB on a half-size image, then refine at full resolution.
    try:
        small = cv2.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2))
        found, corners = cv2.findChessboardCornersSB(
            small, PATTERN_SIZE,
            flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
        )
        if found:
            corners = (corners * 2.0).astype("float32")
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), _SUBPIX_CRITERIA)
            return True, corners
    except Exception:
        pass

    # Mild blur to kill screen moire / sensor aliasing, then detect. Corners are
    # always refined against the ORIGINAL sharp image afterwards, so this costs
    # nothing in accuracy -- the blur only helps the detector locate the grid.
    for ksize in (3, 5):
        try:
            blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
            found, corners = cv2.findChessboardCornersSB(
                blurred, PATTERN_SIZE,
                flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
            )
            if found:
                corners = cv2.cornerSubPix(
                    gray, corners.astype("float32"), (11, 11), (-1, -1), _SUBPIX_CRITERIA)
                return True, corners
        except Exception:
            pass

    legacy_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE, flags=legacy_flags)
    if found:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), _SUBPIX_CRITERIA)
        return True, corners

    return False, None
