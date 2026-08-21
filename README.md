# crop-camera-ai

Vision stack for an autonomous farm-rover project: stereo distance measurement, NDVI plant-health imaging, object detection, and an LLM agent layer that turns what the cameras see into actions.

The rovers themselves are simple, streaming machines: each carries a camera SoC that captures, encodes (H.264/H.265) and sends video over the farm's WiFi. Everything that *thinks* — detection, NDVI analysis, per-plant statistics, the LLM agents, and model training — runs on one GPU host at the barn. **This repository is the host side of that split** (plus bench tools that run against locally-attached cameras during development).

Every analysis script therefore takes either a local camera (`--camera`, `--camera-name`) **or a network stream** (`--source rtsp://...`), so the same code runs on a bench laptop against USB cameras today and against real rover streams in the field without changes.

```
crop-camera-ai/
├── common/                 code that isn't specific to any one camera
├── ndvi/                   NDVI-converted camera -> plant health
├── training/               train a plant detector; per-plant NDVI health
└── experiments/            bench hardware bring-up (not on the deployment path)
```

---

## Where analysis runs, and why

The split is deliberate, and it follows from what each piece of hardware is good at:

- **A camera SoC is a streamer, not a thinker.** The parts that make cheap rover cameras (IP-camera-class SoCs) are built to do sensor → encode → network and little else. Asking them to run analysis means a rewrite in C on a vendor SDK, for worse results than a GPU host gets in Python.
- **Farm WiFi is fast; fields are small data.** A compressed stream from each rover is well within a WiFi 6 network's budget, and the host gets to see every rover's footage with one pool of compute instead of N small ones.
- **Two latencies, two places.** Crop health changes over hours — analysis with seconds of latency is "real time" for that job, so it can live on the host. Anything that must react in milliseconds (obstacle stop, e-stop) belongs on the rover itself and must never depend on the network. Nothing in this repo is in that loop, on purpose.

The consequence for this codebase: capture is just a source. `common/camera_utils.py` opens a local device with the right backend per OS, or an RTSP/MJPEG stream with reconnect-on-drop — and the scripts downstream don't care which they got.

---

## The cameras

### `ndvi/` — NDVI plant-health camera

A camera with its infrared-blocking filter replaced, so it can see near-infrared. Healthy leaves reflect NIR strongly while absorbing visible red, which makes NDVI a direct physical measure of plant vigour rather than a guess based on appearance.

Which colour channel carries the infrared signal depends on which filter was fitted, and it has to be determined empirically — see [`ndvi/README.md`](ndvi/README.md).

Over a stream, remember exposure can only be locked at the **sender**: an encoder on the rover decides exposure and white balance, and NDVI is a ratio that auto-exposure quietly destroys. That configuration belongs to the rover's camera setup, not this end of the wire.

### `experiments/decxin-sm-2930v1/` — binocular stereo camera (bench)

A cheap USB board that turns out to be a stereo camera in disguise: it enumerates as one UVC device but returns both lens views stitched side by side in a single wide frame. Split it and you can compute distance to objects.

There's essentially no published documentation for this module, so [`experiments/decxin-sm-2930v1/README.md`](experiments/decxin-sm-2930v1/README.md) records what was established by probing it — the frame format, working resolutions, the fact that the two lenses are *not* matched, the measured 64.8 mm baseline, and a disparity-offset correction that brings measured distances within 1.5% of tape-measured truth.

---

### `training/` — per-plant health

NDVI measures every pixel; a segmentation model says which pixels are *which plant*. Together they turn "this bed averages 0.61" into "plant 7 is stressed while its neighbours are fine" — see [`training/README.md`](training/README.md).

Runs entirely offline: `uv run training/per_plant_ndvi.py --demo` needs no camera, model or dataset.

This is also the loop that makes a fleet improve over a season: frames the host flags (status change, uncertain classification) are the frames worth labelling; labelled frames fine-tune the segmentation model; the improved model goes back out to the host. The training data collects itself, next to the GPU.

---

## `common/`

| File | Purpose |
|---|---|
| `camera_utils.py` | Camera access: list local cameras, identify by name, open with the right backend (Windows/macOS/Linux) — or open an RTSP/MJPEG stream with `--source`. |
| `farm_camera_detect.py` | Live YOLOv8 object detection. Works with any camera or stream. |
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

Find your local cameras:

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

A local camera (bench / development):

```bash
# distance measurement (calibration included, no need to redo it)
cd experiments/decxin-sm-2930v1
uv run stereo_depth.py --camera-name DECXIN --disparity-offset 2.9

# plant health
cd ndvi
uv run ndvi_probe.py --camera <index>     # once, to identify the camera
uv run ndvi_live.py --camera <index>

# object detection
cd common
uv run farm_camera_detect.py --camera-name DECXIN
```

A rover's stream (deployment — same scripts, different source):

```bash
uv run farm_camera_detect.py --source rtsp://rover-3.local:8554/cam
uv run ndvi/ndvi_live.py --source rtsp://rover-3.local:8554/ndvi
uv run common/farm_camera_agent.py --source rtsp://rover-3.local:8554/cam
```

No rover yet? Any RTSP source works — e.g. serve a video file locally and point a script at it:

```bash
ffmpeg -re -stream_loop -1 -i field_footage.mp4 -c copy -f rtsp rtsp://localhost:8554/test
uv run farm_camera_detect.py --source rtsp://localhost:8554/test
```

---

## Design notes

**Only text goes to the LLM, never images.** Object detection, depth and NDVI all run locally; the model receives a short description ("cow at 3.2 m", "mean NDVI 0.71 over 40% coverage") and decides what to do. That keeps it fast, cheap, and workable on a model with no vision input.

**Two of the LLM's four actions are deliberate stubs.** `sound_alarm` and `log_event` are real; `call_phone` and `control_smart_device` log their intent rather than acting, pending real credentials. See `common/farm_ai_actions.py`.

**Network addresses live in environment variables, never in this repository.** See `.env.example`.

[`ARCHITECTURE.md`](ARCHITECTURE.md) covers the wider system: the rover→host split, fleet data flow, the LLM serving plan, and current project status.

---

## Licence

MIT — see [LICENSE](LICENSE). Use it, adapt it, ship it; just keep the notice.

That covers **the code in this repository only**. Datasets and pretrained
weights carry their own terms, several of which are non-commercial — see
[`training/DATASETS.md`](training/DATASETS.md) before building on any of them.
