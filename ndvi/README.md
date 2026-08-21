# NDVI camera

Plant-health imaging from an NDVI-converted USB camera.

## What NDVI measures, and why it's the right tool here

Chlorophyll absorbs visible red light strongly for photosynthesis, while the internal structure of a leaf reflects near-infrared very strongly. "Bright in NIR, dark in red" is therefore the optical signature of a healthy leaf, and it weakens as a plant becomes stressed — often before any change is visible to the eye. NDVI condenses that into one number:

```
NDVI = (NIR - VIS) / (NIR + VIS)
```

Range is −1 to +1. Dense healthy vegetation usually lands around 0.6–0.9, stressed or sparse vegetation lower, bare soil and man-made surfaces near 0, water often negative.

This is a genuine physical measurement of plant vigour, which is why it's a better foundation than `common/plant_anomaly_agent.py` — that script can only report "this looks different from what I saw before".

## Why you must run the probe first

Affordable NDVI cameras are ordinary cameras with the internal infrared-blocking filter removed and a coloured filter fitted instead. Silicon sensors see NIR perfectly well; it's normally filtered out deliberately. Which channel ends up carrying NIR depends on the filter:

| Conversion | NIR lands in | Formula |
|---|---|---|
| Blue filter ("superblue") | RED channel | `(R − B) / (R + B)` |
| Red filter (Wratten 25A etc.) | BLUE channel | `(B − R) / (B + R)` |

Same formula, roles swapped — which is exactly the trap. **Choosing wrong doesn't produce an obviously broken image; it produces a smooth, plausible, sign-inverted one.** Tested against a synthetic scene whose true vegetation NDVI was +0.82, the wrong mapping returned −0.82. Stressed plants look healthy and healthy plants look dead, with nothing on screen to suggest anything is wrong.

`ndvi_probe.py` settles it by looking at real leaves.

## Usage

```bash
uv pip install opencv-python numpy
uv pip install pygrabber          # Windows only, for camera names

uv run ndvi_probe.py --list-cameras
uv run ndvi_probe.py --camera <index>
```

In the probe window:

1. Fill the centre box with **live foliage**, well lit → press `v`
2. Fill it with something non-living (soil, concrete, a wall) → press `n`
3. Press `a` to analyse

Use real growing plants. Plastic foliage reflects no NIR at all — which is a neat demonstration of what NDVI actually measures, but useless for calibration.

The result is written to `ndvi_config.json`, which the live view reads automatically:

```bash
uv run ndvi_live.py --camera <index> --lock-exposure
```

You get the raw feed beside a false-colour health map — brown for bare ground, yellow for sparse, green for healthy, black for "no valid data" (deliberately distinct from "measured zero"). It also reports mean NDVI across vegetation pixels only, plus what fraction of the frame is vegetation. Press `p` to log values, `s` to save a snapshot.

## Two things that will corrupt your readings

**Auto white balance.** NDVI is a ratio between colour channels, and auto white balance exists specifically to rescale colour channels against each other. Leaving it on means the camera continuously alters the quantity being measured. `--lock-exposure` requests that both auto-exposure and auto-WB be disabled, but many drivers accept the call and ignore it — if values drift as you move the camera, set them manually in the vendor software.

**Gamma encoding.** Camera output is not proportional to light. NDVI is a ratio of light intensities, so this has to be undone first. It matters more than you'd expect: on a test scene whose true NDVI was +0.818, skipping linearisation gave +0.482 — a 0.34 error, enough to make a healthy plant look stressed. It's on by default (`--gamma 2.2`).

## Honest limits

These are **relative** NDVI values, not scientific absolute ones. Real absolute NDVI needs reflectance calibration targets in shot. What you have here is good for comparing plants within a frame, or tracking the same plant over time under similar light — which is what a rover actually needs. Don't compare the numbers to published or satellite NDVI figures.

## Files

| File | Purpose |
|---|---|
| `ndvi_common.py` | NDVI maths, colour map, vegetation statistics, config handling. |
| `ndvi_probe.py` | Determines which channel carries NIR, from real vegetation. |
| `ndvi_live.py` | Live NDVI view with health statistics. |
| `ndvi_config.json` | Written by the probe, read by the live view. Not in git — it describes *your* camera. |
