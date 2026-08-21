"""
ndvi_common.py

Shared NDVI maths and configuration, used by ndvi_probe.py (which works out
how your camera is wired) and ndvi_live.py (which uses that answer).

--------------------------------------------------------------------
WHAT NDVI IS

Healthy vegetation does two distinctive things: chlorophyll absorbs visible
red light strongly (for photosynthesis), while leaf cell structure reflects
near-infrared very strongly. So "bright in NIR, dark in red" is the optical
signature of a healthy leaf. NDVI turns that into one number:

    NDVI = (NIR - VIS) / (NIR + VIS)

Range is -1..+1. Dense healthy vegetation typically lands around 0.6-0.9,
stressed or sparse vegetation lower, and bare soil, stone or man-made
surfaces near 0. Water often goes negative.

This matters for the farm project because it measures plant health
*directly and physically*, rather than inferring "this looks different from
before" the way plant_anomaly_agent.py has to.

--------------------------------------------------------------------
WHY THE CHANNEL MAPPING HAS TO BE DISCOVERED

Affordable "NDVI cameras" are ordinary colour cameras with the internal
infrared-blocking filter removed and a coloured filter fitted instead.
Silicon sensors see NIR perfectly well; it is normally filtered out on
purpose. Which colour channel ends up carrying NIR depends on which filter
was fitted:

  * BLUE filter ("superblue"): passes blue + NIR. The camera's RED channel
    receives NIR; the BLUE channel keeps a real visible band.
        NDVI = (R - B) / (R + B)

  * RED filter (Wratten 25A and similar): passes red + NIR. The camera's
    BLUE channel receives NIR; the RED channel carries visible red.
        NDVI = (B - R) / (B + R)

Both are the same formula with the roles swapped -- which is exactly why
guessing is dangerous. Pick the wrong one and you get a smooth, colourful,
entirely inverted image that looks completely plausible. ndvi_probe.py
determines which applies to your camera from actual vegetation.

--------------------------------------------------------------------
HONEST LIMITS

This is *relative*, uncalibrated NDVI. Treat it as "which plant is doing
better than which", not as a scientific absolute value comparable to
published figures or satellite data. Three reasons:

  1. No reflectance reference. Real NDVI needs targets of known reflectance
     in shot to convert pixel values into reflectance.
  2. Auto white balance and auto exposure actively destroy it -- both
     rescale channels independently between frames, and NDVI is entirely
     a ratio between channels. Lock them if the camera allows it
     (ndvi_live.py --lock-exposure tries).
  3. Camera output is gamma-encoded, not linear in light. We undo that by
     default (--gamma 2.2); without it values are systematically skewed.

Relative NDVI is still genuinely useful: comparing plants in one frame, or
tracking the same plant over time under similar light, is exactly the kind
of question a farm rover needs answered.
"""

import json
import os

import cv2
import numpy as np

CONFIG_FILE = "ndvi_config.json"

# Vegetation vs non-vegetation is conventionally split near here. Tune per
# site -- it is a soft boundary, not a physical constant.
DEFAULT_VEG_THRESHOLD = 0.2


def save_config(nir_channel: str, notes: str = "", path: str = CONFIG_FILE) -> None:
    with open(path, "w") as f:
        json.dump({"nir_channel": nir_channel, "notes": notes}, f, indent=2)


def load_config(path: str = CONFIG_FILE):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def linearize(channel: np.ndarray, gamma: float) -> np.ndarray:
    """Undo the camera's gamma encoding so values are proportional to light.
    NDVI is a ratio of light intensities, so this should be done on values
    that actually represent light."""
    if gamma is None or gamma <= 0:
        return channel
    return np.power(np.clip(channel, 0.0, 1.0), gamma)


def compute_ndvi(frame_bgr: np.ndarray, nir_channel: str, gamma: float = 2.2):
    """Return (ndvi, valid_mask). ndvi is float32 in -1..1; pixels where the
    denominator is too small to be meaningful are masked out rather than
    producing wild values from sensor noise in near-black areas."""
    b, g, r = cv2.split(frame_bgr.astype(np.float32) / 255.0)

    if nir_channel.upper() == "R":
        nir, vis = r, b          # blue-filter conversion
    elif nir_channel.upper() == "B":
        nir, vis = b, r          # red-filter conversion
    else:
        raise ValueError(f"nir_channel must be 'R' or 'B', got {nir_channel!r}")

    nir = linearize(nir, gamma)
    vis = linearize(vis, gamma)

    denom = nir + vis
    valid = denom > 0.02  # below this it's sensor noise, not signal
    ndvi = np.zeros_like(denom, dtype=np.float32)
    np.divide(nir - vis, denom, out=ndvi, where=valid)
    np.clip(ndvi, -1.0, 1.0, out=ndvi)
    return ndvi, valid


def _build_ndvi_lut() -> np.ndarray:
    """256-entry BGR lookup table mapping NDVI -1..+1 onto the palette
    convention used for vegetation indices: brown/grey for bare ground,
    yellow around the vegetation threshold, deepening green for vigour.
    Chosen over a generic rainbow map because it reads correctly at a
    glance -- greener really does mean healthier."""
    stops = [0.00, 0.35, 0.50, 0.62, 0.75, 1.00]     # normalised NDVI position
    rgb = [
        (68, 62, 58),      # very low  -> dark neutral
        (140, 120, 96),    # low       -> brown
        (206, 190, 120),   # ~0        -> tan
        (233, 227, 120),   # slight    -> yellow
        (120, 180, 70),    # moderate  -> green
        (18, 92, 34),      # high      -> deep green
    ]
    xs = np.linspace(0.0, 1.0, 256)
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    for ch in range(3):                       # build in RGB, store as BGR
        vals = np.interp(xs, stops, [c[ch] for c in rgb])
        lut[:, 0, 2 - ch] = vals.astype(np.uint8)
    return lut


NDVI_LUT = _build_ndvi_lut()


def colorize_ndvi(ndvi: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Render an NDVI array as a colour image. Invalid pixels come out black
    so 'no data' is never mistaken for 'measured zero'."""
    norm = ((np.clip(ndvi, -1.0, 1.0) + 1.0) * 0.5 * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(norm, NDVI_LUT)
    colored[~valid] = (0, 0, 0)
    return colored


def vegetation_stats(ndvi: np.ndarray, valid: np.ndarray,
                     threshold: float = DEFAULT_VEG_THRESHOLD):
    """Summarise a frame: how much of it is vegetation, and how healthy.

    Returns a dict, or None when nothing in view looks like vegetation.
    mean_ndvi deliberately covers vegetation pixels ONLY -- averaging over
    the whole frame would just measure how much soil is in shot."""
    veg_mask = valid & (ndvi > threshold)
    total_valid = int(valid.sum())
    veg_count = int(veg_mask.sum())
    if total_valid == 0:
        return None
    coverage = veg_count / total_valid
    if veg_count < 50:
        return {"coverage": coverage, "veg_pixels": veg_count,
                "mean_ndvi": None, "p10": None, "p90": None}
    veg_vals = ndvi[veg_mask]
    return {
        "coverage": coverage,
        "veg_pixels": veg_count,
        "mean_ndvi": float(np.mean(veg_vals)),
        "p10": float(np.percentile(veg_vals, 10)),
        "p90": float(np.percentile(veg_vals, 90)),
    }
