# crop-camera-ai

Vision code for an autonomous farm-rover project: stereo distance measurement, NDVI plant-health imaging, object detection, and an LLM agent layer that turns what the cameras see into actions.

Two cameras are supported, each in its own folder, sharing a common core.

```
Camera/
├── common/                 code that isn't specific to any one camera
├── decxin-sm-2930v1/       binocular USB camera -> object distance
├── ndvi/                   NDVI-converted camera -> plant health
└── training/               train a plant detector; per-plant NDVI health
```

---

## The cameras

### `decxin-sm-2930v1/` — binocular stereo camera

A cheap USB board that turns out to be a stereo camera in disguise: it enumerates as one UVC device but returns both lens views stitched side by side in a single wide frame. Split it and you can compute distance to objects.

There's essentially no published documentation for this module, so [`decxin-sm-2930v1/README.md`](decxin-sm-2930v1/README.md) records what we established by probing it — the frame format, working resolutions, the fact that the two lenses are *not* matched, and the measured 64.8 mm baseline.

### `ndvi/` — NDVI plant-health camera

A camera with its infrared-blocking filter replaced, so it can see near-infrared. Healthy leaves reflect NIR strongly while absorbing visible red, which makes NDVI a direct physical measure of plant vigour rather than a guess based on appearance.

Which colour channel carries the infrared signal depends on which filter was fitted, and it has to be determined empirically — see [`ndvi/README.md`](ndvi/README.md).

---

### `training/` — per-plant health

NDVI measures every pixel; a segmentation model says which pixels are *which plant*. Together they turn "this bed averages 0.61" into "plant 7 is stressed while its neighbours are fine" — see [`training/README.md`](training/README.md).

Runs entirely offline: `uv run training/per_plant_ndvi.py --demo` needs no camera, model or dataset.

---

## `common/`

| File | Purpose |
|---|---|
| `camera_utils.py` | Cross-platform camera access: list, identify by name, open with the right backend (Windows/macOS/Linux). |
| `farm_camera_detect.py` | Live YOLOv8 object detection. Works with any camera. |
| `farm_ai_actions.py` | The tools the LLM may call, and the code behind each one. |
| `farm_camera_agent.py` | Object detections → LLM → action. |
| `plant_anomaly_agent.py` | Learns a "healthy" visual baseline and flags deviations → LLM → action. Predates the NDVI camera; NDVI measures the same thing more directly. |

Scripts inside the camera folders add `../common` to their import path automatically, so you can run any of them directly from its own directory.

---

## Setup

```bash
uv venv
uv pip install opencv-python numpy ultralytics openai
uv pip install pygrabber          # Windows only, for camera names
```

Configure the LLM endpoint (the agents need it; the camera tools don't):

```bash
cp .env.example .env      # then edit, or just export the variables
export DEEPSEEK_API_BASE="http://your-server:8888/v1"
export DEEPSEEK_MODEL="your-model-name"
```

Find your cameras:

```bash
cd common && uv run farm_camera_detect.py --list-cameras
```

The stereo board is flagged automatically — it's the only device reporting an unusually wide frame.

---

## Typical use

Scripts live in subfolders. Run them either from the repo root with a path, or from inside the folder — both work, and data files (calibration, NDVI config) always resolve next to the script regardless of where you launched it.

```bash
# from the repo root
uv run ndvi/ndvi_probe.py --list-cameras

# or from inside the folder
cd ndvi && uv run ndvi_probe.py --list-cameras
```

```bash
# distance measurement (calibration included, no need to redo it)
cd decxin-sm-2930v1
uv run stereo_depth.py --camera-name DECXIN --disparity-offset 2.9

# plant health
cd ndvi
uv run ndvi_probe.py --camera <index>     # once, to identify the camera
uv run ndvi_live.py --camera <index>

# object detection
cd common
uv run farm_camera_detect.py --camera-name DECXIN
```

---

## Design notes

**Only text goes to the LLM, never images.** Object detection, depth and NDVI all run locally; the model receives a short description ("cow at 3.2 m", "mean NDVI 0.71 over 40% coverage") and decides what to do. That keeps it fast, cheap, and workable on a model with no vision input.

**Two of the LLM's four actions are deliberate stubs.** `sound_alarm` and `log_event` are real; `call_phone` and `control_smart_device` log their intent rather than acting, pending real credentials. See `common/farm_ai_actions.py`.

**Network addresses live in environment variables, never in this repository.** See `.env.example`.

[`ARCHITECTURE.md`](ARCHITECTURE.md) covers the wider system: rover fleet plans, the LLM serving setup, and current project status.

---

## Licence

MIT — see [LICENSE](LICENSE). Use it, adapt it, ship it; just keep the notice.

That covers **the code in this repository only**. Datasets and pretrained
weights carry their own terms, several of which are non-commercial — see
[`training/DATASETS.md`](training/DATASETS.md) before building on any of them.
