# Farm Rover — Camera & AI Stack

Status as of 21 Aug 2026. This is a living reference for what we're building, why it's built this way, and where things stand — update it as the project moves rather than starting a new doc each time.

## What we're building

The goal is a rover-mounted system that watches a farm through cameras, reasons about what it sees, and takes real action when something needs attention — without a human watching a video feed all day. The production target is a fleet of 6-7 rovers, each carrying 2-3 cameras (12-21 streams total), reporting to a single GPU host at the barn. Two kinds of "seeing" feed into the same decision-making core: general awareness of what's in view (people, animals, vehicles — using an off-the-shelf object detector), and plant/crop health (NDVI measurement, plus a segmentation model that says which pixels are which plant). Either way, what gets sent to the AI is a short text description of the observation — never raw images — and the AI decides whether to log it quietly, sound an alarm, or (once wired up) call a person or trigger a physical device.

Today all of this runs on a Windows laptop with a USB camera standing in for a rover, because the deployment site is not currently accessible. The code was written with that move in mind: every analysis script takes a `--source rtsp://...` network stream exactly as easily as a local camera index, so moving from the bench laptop to the barn host against real rover streams is a configuration change, not a code change.

## Architecture

```
rover (x6-7)                          barn
┌─────────────────────────┐          ┌──────────────────────────────────┐
│ camera SoC(s)           │  WiFi 6  │  GPU host (this repository)      │
│  capture -> ISP ->      │ ───────> │   decode stream(s)               │
│  H.264/H.265 encode     │  RTSP/   │   YOLOv8 detection  NDVI analysis│
│                         │  MJPEG   │   per-plant statistics           │
│  (local reflexes only:  │          │        |                         │
│   obstacle stop, e-stop │          │        v                         │
│   -- never over network)│          │  text observation                │
└─────────────────────────┘          │        v                         │
                                     │  orchestration LLM (tool-calling)│
                                     │        v                         │
                                     │  sound_alarm | log_event |       │
                                     │  (stub) call_phone, smart device │
                                     │  + storage, per-plant records,   │
                                     │    training                      │
                                     └──────────────────────────────────┘
```

## Why the rover streams and the host thinks

The rover cameras are IP-camera-class SoCs (the family being evaluated: SigmaStar SSC338Q/SSC378QE, Rockchip RV1126-class parts) — the chips inside security cameras. Their silicon is a pipeline from sensor to compressed network video, plus a small NPU. They are excellent at capture/encode/stream and poor at everything else: general Python with OpenCV and numpy does not meaningfully run there, and porting to a vendor C SDK would buy worse results than the GPU host gets for free.

So compute concentrates at the host, and the farm's WiFi 6 network carries compressed video — a good trade, because one big pool of GPU beats N small CPU-bound analyzers, and every rover's footage stays observable in one place.

**Two latencies, two places.** Crop health changes over hours, so analysis with seconds of latency is real-time *for agronomy* — it can live behind the network. Anything that must react in milliseconds (obstacle stop, e-stop, geofence) runs on the rover itself and never depends on WiFi or on this stack. If the network drops, rovers must get safer, not dumber. Nothing in this repository is in that control loop, on purpose; the SoC's NPU is the natural home for a local detection reflex later, as a separate C/RKNN project.

Stream consumers reconnect instead of stopping (a rover going out of range and coming back is normal — see `--source` in the scripts), and keep a tiny capture buffer so a slow analysis frame shows fresh footage rather than seconds-old footage.

## Hardware

- **Rover cameras**: IP-camera SoCs with the appropriate sensors — an ordinary colour camera for detection/identity, an IR-converted or mono+NIR pairing for NDVI. The SoC decides exposure and frame rate; for NDVI that matters (the index is a ratio, and auto-exposure quietly rescales it), so locked manual exposure is a rover-side camera configuration requirement, not something the host can fix after the fact.
- **Host**: a single DGX Spark-class machine (unified memory, ~128 GB) near the farm. It is the fleet's collector, analyst, store and training machine, and it is not shared with anything else.
- **Development**: a Windows laptop with USB cameras standing in for rovers. Nothing here is Windows-specific except one optional convenience (camera name lookup).

> Endpoints, hostnames and network addresses are deliberately kept out of this repository. They're supplied at runtime through environment variables — see `.env.example`.

## One known hardware risk: dual-camera NDVI sync

The higher-quality NDVI design uses two cameras — colour for the red band, monochrome with an NIR filter for the NIR band — time-synchronised and registered to each other (the companion two-camera NDVI project measured ~17 ms mean pair skew as workable). On the SoC path, each camera is its own encoder, and "synchronised" degrades to "both clocks were NTP-correct when the frames were stamped". Registration and calibration move to the host either way (that's where the numpy/OpenCV power is). If pairwise sync proves inadequate on real hardware, the fallbacks are (a) the single IR-converted camera approach in `ndvi/` — one stream, no sync problem, slightly noisier NDVI — or (b) one rover carrying a general-purpose compute board for the NDVI pair only. Decide on measured skew, not on speculation.

## AI serving

The analysis scripts are model-agnostic: every script takes `--api-base` and `--model` (or the `DEEPSEEK_API_BASE` / `DEEPSEEK_MODEL` environment variables), against any OpenAI-compatible endpoint. During bring-up a shared development endpoint is in use; the production plan is the farm-side GPU host itself.

The AI's job in this system is narrow: read a short text observation (an object list, an anomaly score, a per-plant health summary) and decide which of a handful of tools to call. That's orchestration/routing, not open-ended reasoning or generation, and doesn't need a frontier-scale model. Nothing in the tool definitions or system prompt (`common/farm_ai_actions.py`) is tied to a specific model or vendor for exactly this reason.

### Model candidates for the orchestration role

Since the job is picking between a handful of tools from a short text description — not writing code or long-form reasoning — there's real headroom to go small. Two tiers worth considering, based on current (Aug 2026) published benchmarks for local tool-calling models:

A **~4-8B tier** (e.g. Qwen3-4B-Instruct-2507, Mistral-7B-Instruct-v0.3) is the leanest option — fast, tiny memory footprint, leaves the vast majority of the host's memory free for concurrency. No hard tool-calling reliability numbers were published for this tier specifically, so it would need to be benchmarked against our actual 4-tool schema (`sound_alarm`, `log_event`, `call_phone`, `control_smart_device`) before trusting it unattended — the task is simple enough that this tier may well be reliable enough, but that's an assumption to verify, not a given.

A **~27-32B tier** (Qwen3-32B, Gemma-4-27B, or GLM-4.7-32B) is the safer default: independently benchmarked at roughly 93-95% well-formed tool calls on general agentic tasks. Still comfortably fits alongside a healthy concurrency budget on ~128 GB of unified memory, with no tensor-parallel/multi-node complexity needed. Qwen's function-calling track record and active maintenance make Qwen3-32B the natural first thing to try; Gemma-4-27B and GLM-4.7-32B are close alternatives if it doesn't work out.

**Recommendation: start with Qwen3-32B, and benchmark a smaller Qwen3-4B/7B-class model against it on our real tool schema before committing.** If the smaller model performs comparably on our specific (fairly simple) task, it's the better long-term choice for a fleet this size — more concurrency headroom per watt on dedicated hardware.

### Memory estimate for `--max-num-seqs 24`

Using each model's real architecture (layers, KV heads, head dim — pulled from their published configs, not guessed), KV cache size per token is `2 × layers × kv_heads × head_dim × bytes_per_element`. Worst case with `--max-model-len 4096`, `--max-num-seqs 24`, fp8 weights, and fp8 KV cache (all 24 slots simultaneously at full length, no prefix-caching credit):

| Model | Layers / KV heads / head_dim | Weights (fp8) | KV cache (24×4096 tok) | Total | Free of 128GB |
|---|---|---|---|---|---|
| Qwen3-32B | 64 / 8 / 128 | 30.5 GB | 12.0 GB | ~45.5 GB | ~82.5 GB |
| Qwen3-14B | 40 / 8 / 128 | 13.8 GB | 7.5 GB | ~24.3 GB | ~103.7 GB |
| Mistral-7B-Instruct-v0.3 | 32 / 8 / 128 | 6.8 GB | 6.0 GB | ~15.8 GB | ~112.2 GB |
| Qwen3-4B-Instruct-2507 | 36 / 8 / 128 | 3.7 GB | 6.8 GB | ~13.5 GB | ~114.5 GB |

All four fit with large headroom even in this worst case — memory isn't the constraint at these sizes, so the model choice should be driven by tool-call reliability and latency, not by what fits. Worth noting: KV cache size depends on layer count and KV-head count, not total parameters, which is why Qwen3-4B's per-token KV cost is actually higher than Mistral-7B's despite being the smaller model — that gap matters once you're running 24 concurrent sequences, even if it's invisible at batch size 1. (Params-based weight figures are the commonly published approximations, not exact down to the embedding table — fine for provisioning, not for the last few hundred MB.)

Suggested launch flags beyond `--max-num-seqs 24`: `--tensor-parallel-size 1`, `--max-model-len 4096` (right-sized for our short prompts — this directly caps worst-case KV memory), `--gpu-memory-utilization 0.85` (generous, since the host isn't shared), `--enable-prefix-caching` (the system prompt + tool schema is identical on every request — real savings), `--kv-cache-dtype fp8` plus an fp8-quantized checkpoint, `--enable-auto-tool-choice` plus a `--tool-call-parser` matched to whichever model is chosen (check vLLM's current supported list for that exact model — worth verifying fresh rather than assuming), and if the model has a "thinking" mode, disabling it by default, since orchestration doesn't need chain-of-thought.

### Concurrency at fleet scale

A 4-32B model has a far smaller KV-cache footprint per request than a frontier-scale MoE, so `--max-num-seqs` can be set high — likely 32-64+ depending on the model chosen and actual memory headroom, enough to comfortably cover bursts from the full rover fleet without queuing. The existing per-camera debounce (only fire on a *change*, minimum seconds between calls) keeps bursts small in the first place; with 12-21 independently-debounced streams it remains worth keeping, not relaxing.

## Fleet data flow

Three tiers, by bandwidth and value:

| Tier | Content | Size | When |
|---|---|---|---|
| 1 — always | structured reports (per-frame health rows, detections, position, heartbeats, model/firmware versions) | KB/s per rover | continuous |
| 2 — on flag | frames or clips around alerts, uncertain classifications, status changes | MB, occasional | when something deserves a second look |
| 3 — never by default | continuous archival video | GB/hour | only if deliberately decided |

Tier 2 is the one that pays for itself: frames the host flags are the frames worth labelling, and they arrive sitting next to the GPU that trains on them. That closes the improve-over-a-season loop — flag → label → fine-tune the segmentation model (`training/`) → the improved model analyses tomorrow's streams.

## Position: RTK and the per-plant record

Each frame should carry centimetre-grade RTK position plus a fix-quality flag (fixed / float / satellite count / HDOP), the same way health reports already carry their calibration and sync state — a position is a measurement, and a measurement without its quality is a claim. With that, positions cluster into a plant registry: each plant gets an ID at first sighting, and "plant #7 declined over three weeks" becomes a query rather than a hope. Fix quality matters most exactly on the frames you'd otherwise trust most — the flagged ones someone walks out to inspect — so it should gate any per-plant statistic that aggregates over position.

## Software stack

Python, managed with `uv` for environments and package installs (faster and less friction than raw pip, especially for a `.venv` per project). Capture and image handling go through OpenCV (`opencv-python`) — including stream ingestion, since OpenCV's FFmpeg backend opens RTSP (H.264/H.265) and MJPEG-over-HTTP sources directly. Object detection uses Ultralytics' YOLOv8 (`ultralytics`), specifically the pretrained `yolov8n.pt` nano model — small and fast enough for real-time use on modest hardware. Plant anomaly detection uses only OpenCV and `numpy` — deliberately no deep learning dependency there, so it stays light. The LLM bridge goes through the official `openai` Python package, since every serving option (vLLM, SGLang, Ollama) exposes an OpenAI-compatible endpoint. On Windows, `pygrabber` is an optional dependency that resolves local camera *names* (not just numeric indices); it is irrelevant for network sources, which have no index to resolve.

## Repository layout

```
crop-camera-ai/
├── common/                 camera access (local + streams), YOLO detection, LLM tools + agents
├── ndvi/                   NDVI plant-health imaging
├── training/               segmentation training + per-plant NDVI
└── experiments/            bench hardware bring-up (decxin stereo board)
```

`common/camera_utils.py` is the single camera layer: list local devices, identify by name, open with the right backend — or open a network stream and keep it alive across drops. `common/farm_ai_actions.py` holds the tools the LLM may call and the code behind each, shared by both agents so every rover behaves identically.

Each camera folder has its own README documenting that hardware. Scripts inside those folders add `../common` to their import path automatically, so any of them can be run directly from its own directory.

## Stereo distance estimation

The `experiments/decxin-sm-2930v1/` folder is a completed bench investigation: a cheap USB board that is secretly a binocular stereo camera, probed, calibrated, and validated against tape-measured distances to ~1.5% after a disparity-offset correction. The core math (calibration recovering a known baseline; disparity-to-distance recovering known depth) was verified against synthetic ground truth before trusting the OpenCV calls. It is not on the deployment path — per-rover obstacle sensing belongs to the rover's own NPU reflex — but the findings are recorded because the module is undocumented anywhere else. See that folder's README.

## Why text, not images

The LLM is used here purely as a text/tool-calling reasoner, not a vision model. Object detection (YOLOv8), NDVI and the per-plant statistics all happen on the host, and only their *output* — a handful of words or a numeric score — reaches the model. This is deliberately cheap and fast: no image encoding, no large payloads, and it works with any tool-calling model regardless of vision support. If a future need requires the AI to actually judge an image (distinguishing look-alike diseases, judging severity from appearance), that's a separate, explicit upgrade — sending a frame to a vision-capable model only when the local analysis already flagged something worth a closer look, to keep bandwidth and host load down.

## Safety posture for actions

Of the four tools the AI can call, two are real and two are intentionally stubbed. `sound_alarm` and `log_event` only touch the machine running the script (a beep through local speakers, a line appended to `events.log`) — safe to let the AI trigger autonomously. `call_phone` and `control_smart_device` currently just log what the AI *would* have done, because they'd require real credentials (a telephony account, a specific smart device's API) that don't exist yet and shouldn't be guessed at. Wiring either up for real is a small, well-scoped addition once a provider and device target are decided; the integration points are already marked with TODOs in `farm_ai_actions.py`.

Longer-term these belong on the host, not on a rover: a fleet-wide alert dispatcher with credentials, retry logic and rate limits is a service, not a per-camera script.

## Current status

Confirmed working: local camera capture and streaming analysis paths on Windows via OpenCV (including warm-up reads and retry tolerance for UVC driver quirks); YOLOv8 object detection runs live with a preview window; the `openai`-client bridge to an OpenAI-compatible endpoint is wired and connects successfully; the stereo bench experiment (see `experiments/`) is calibrated and validated against ground truth; stream sources (`--source rtsp://...`) with reconnect-on-drop are implemented and syntax-verified, but not yet exercised against a real rover encoder.

Paused: `plant_anomaly_agent.py`'s calibration and threshold tuning need a real camera pointed at real (or at least plausible stand-in) plants, which isn't possible until there's access to the farm again. The code is written and the underlying scoring logic has been sanity-tested against synthetic data, but it hasn't been calibrated or validated against an actual plant yet.

Not started: the barn host itself (ingest service for many streams, structured storage, the per-plant registry); real `call_phone` / `control_smart_device` integrations; rover hardware bring-up (SoC selection, encoder configuration, locked-exposure camera setup for NDVI, the local obstacle reflex); field validation of the health thresholds against real crop.

## Open items

Pick and benchmark the orchestration model for the barn host against our actual tool schema (Qwen3-32B is the current working recommendation). Exercise `--source` against a real SoC encoder — an ffmpeg-served test stream proves the path, but only real hardware proves the reconnect behavior. Decide the dual-camera NDVI sync question on measured skew once SoC hardware exists (see "One known hardware risk"). Stand up the host-side ingestion and storage: structured report rows (aligned with the companion NDVI project's report schema), the RTK-gated per-plant registry, and the flag → label → fine-tune loop. Decide on and wire a real phone/SMS provider and a real smart-device target once the brand/API is known. Design how the fleet reports findings centrally before it is deployed — this repository is the intended home of that collector, so its shape should be decided before the first multi-rover run, not retrofitted after.

## Quick reference

```powershell
# One-time environment setup (from the repo root)
uv venv
uv pip install opencv-python numpy ultralytics openai pygrabber

# Object detection, and the agents (local camera)
cd common
uv run farm_camera_detect.py --camera-name DECXIN
uv run farm_camera_agent.py --camera-name DECXIN
uv run plant_anomaly_agent.py --camera-name DECXIN --calibrate

# ...or a network stream (bench-test with an ffmpeg-served file)
uv run farm_camera_detect.py --source rtsp://localhost:8554/test

# NDVI plant health
cd ndvi
uv run ndvi_probe.py --camera <index>          # once, to identify the camera
uv run ndvi_live.py --camera <index>
uv run ndvi_live.py --source rtsp://rover-3.local:8554/ndvi

# Stereo bench experiment
cd experiments/decxin-sm-2930v1
uv run stereo_depth.py --camera-name DECXIN --disparity-offset 2.9

# Per-plant NDVI regression (no hardware needed)
cd training
uv run per_plant_ndvi.py --demo
```
