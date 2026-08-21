# DECXIN-SM-2930V1 — binocular stereo camera

Distance-to-object measurement from a cheap USB camera board.

There is essentially no published documentation for this module, so this file records what we established by probing an actual unit.

## What this camera actually is

**A standard UVC device.** No vendor driver needed on Windows, macOS, Linux or Raspberry Pi OS — it appears as an ordinary webcam.

**A stereo camera disguised as one camera.** It enumerates as a *single* UVC device, but the frame it returns is two lens views stitched side by side. At its top mode 3840×1080 is really two 1920×1080 images. Split the frame down the middle and you have a stereo pair, which is what makes distance measurement possible — a single lens cannot do this.

**Confirmed resolutions** (combined frame, both lenses):

| Requested | Delivered | Per lens |
|---|---|---|
| 640×480 | 640×240 | 320×240 |
| 1280×480 | 1280×480 | 640×480 |
| 2560×720 | 2560×720 | 1280×720 |
| 2560×960 | 2560×960 | 1280×960 |
| 3840×1080 | 3840×1080 | 1920×1080 |

2560×720 is the practical default: the top mode runs at a low frame rate over USB, which means motion blur on a handheld calibration target and laggy feedback.

**The two lenses are not matched.** The left is visibly wider-angle (subjects smaller, obvious barrel distortion); the right is narrower, more magnified and sharper. Colour and exposure differ slightly too. These modules are usually built for face liveness detection rather than depth sensing.

Stereo depth still works — OpenCV handles differing per-lens intrinsics, verified end to end against synthetic data with deliberately mismatched focal lengths — but there are two practical consequences. The usable region is only where the two views overlap, so **the narrower right lens is what constrains your framing**. And disparity matching is noisier than with a matched pair.

**Measured baseline: ≈64.8 mm**, matching a caliper measurement of 64.75 mm between lens centres.

## The disparity offset — don't skip this

Because the lenses are mismatched, rectification leaves a small constant disparity bias. This makes distances read short by a percentage that **grows with range**, which is a distinctive signature: on this unit, tape-measured 0.5 m read 0.48 m (−4%) while 1.0 m read 0.90 m (−10%).

That pattern rules out the obvious explanations. A wrong calibration scale would give the same percentage error at every distance; measuring the tape from the wrong reference point would give a constant error in metres. Only a constant *disparity* offset explains error that grows with range — because distance is inversely proportional to disparity, so a fixed pixel error matters proportionally more as disparity shrinks with distance.

Correcting by −2.88 px brought both readings within 1.5% of truth.

```bash
uv run stereo_depth.py --camera-name DECXIN --disparity-offset 2.9
```

Find the value for your own unit by putting an object at a known distance:

```bash
uv run stereo_depth.py --camera-name DECXIN --check-at 1.5
# aim the crosshair at it, press 'c'
```

Derive it at a middling distance rather than up close — disparity is smaller there, so the estimate is better conditioned.

## Usage

`stereo_calib.npz` is included, but it describes *this specific board*. Yours will differ.

```bash
# 1. confirm how your board exposes its lenses
uv run stereo_probe.py

# 2. make a calibration target
uv run generate_checkerboard.py

# 3. collect pairs -- SPACE when BOTH panels read DETECTED
uv run stereo_capture.py --camera-name DECXIN --collect

# 4. solve
uv run stereo_calibrate.py --square-size-mm <your measured square size>

# 5. measure distances
uv run stereo_depth.py --camera-name DECXIN --disparity-offset <yours>
```

Print `checkerboard.png` at 100% scale, **or display it full-screen on a tablet or second monitor** — a screen is perfectly flat and rigid, which is what the maths wants, and the detector handles the resulting moiré automatically. Measure the squares with a ruler laid across all 10 and divide by 10; that dilutes measurement error rather than multiplying it. The square size sets the absolute scale of every distance the system will ever report.

Collect 15–25 pairs at varied distances, tilts and positions in frame. Aim for a stereo RMS under 1.0 px.

## Sanity checks

The computed baseline should match a caliper measurement between the lens centres. If it doesn't, your square size was wrong — and the fix is arithmetic, no re-capture needed:

```
corrected square size = current × (measured baseline ÷ computed baseline)
```

Then re-run `stereo_calibrate.py`. Note that afterwards the baseline will agree by construction, so it's no longer independent evidence — validate against a tape measure instead.

## Limits

Stereo cannot measure featureless surfaces. A blank wall offers nothing to match between the two views, so no disparity exists to compute. That's physics, not a bug — aim at texture.

Accuracy falls off with distance as disparity shrinks. With a ~65 mm baseline, expect good results to a few metres and progressively rougher numbers beyond.

## Files

| File | Purpose |
|---|---|
| `stereo_probe.py` | Determines how a binocular board exposes its lenses. |
| `generate_checkerboard.py` | Printable/displayable calibration target. |
| `stereo_capture.py` | Live split view; collects calibration pairs. |
| `stereo_calibrate.py` | Solves intrinsics, distortion, stereo geometry, rectification. |
| `stereo_common.py` | Shared checkerboard detector, robust to blur, glare and screen moiré. |
| `stereo_depth.py` | Live distance measurement; `--with-detection` labels objects with distance. |
| `stereo_calib.npz` | Calibration for this specific unit. |
