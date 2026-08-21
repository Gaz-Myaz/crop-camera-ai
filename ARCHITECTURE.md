# Farm Rover — Camera & AI Agent Stack

Status as of 17 Aug 2026. This is a living reference for what we're building, why it's built this way, and where things stand — update it as the project moves rather than starting a new doc each time.

## What we're building

The goal is a rover-mounted system that watches a farm through cameras, reasons about what it sees, and takes real action when something needs attention — without a human watching a video feed all day. The production target is a fleet of 6-7 rovers, each carrying 2-3 cameras (see "Fleet scale" below), though development so far has been on a single camera as a stand-in for one rover. Two kinds of "seeing" feed into the same decision-making core: general awareness of what's in view (people, animals, vehicles — using an off-the-shelf object detector), and plant/crop health (spotting leaves or plants that look different from healthy baseline, since no off-the-shelf model knows what disease or pest damage looks like for this specific crop). Either way, what gets sent to the AI is a short text description of the observation — never raw images — and the AI decides whether to log it quietly, sound an alarm, or (once wired up) call a person or trigger a physical device.

Today all of this runs on a Windows laptop with a USB camera standing in for a rover, because the deployment site is not currently accessible. The code was written with that move in mind: nothing here is Windows-specific except one optional convenience (camera name lookup), and everything is designed to run unmodified on the Raspberry Pi units the real rovers will use.

## Architecture

```
DECXIN camera (USB/UVC)
        |
        v
   OpenCV capture  ──────────────────────────────────────────┐
        |                                                     |
        v                                                     v
 YOLOv8 object detection                     HSV/texture anomaly detection
 (farm_camera_agent.py)                      vs. learned healthy baseline
 "what general objects are visible"          (plant_anomaly_agent.py)
        |                                                     |
        └───────────────────────┬─────────────────────────────┘
                                 v
                    plain-text observation
                                 v
              DeepSeek V4 Flash (tool-calling, text-only)
                served via vLLM on dual NVIDIA DGX Spark
                                 v
                    farm_ai_actions.py dispatch
                 sound_alarm | log_event | (stub) call_phone
                       | (stub) control_smart_device
```

Both detection scripts are independent entry points that converge on the same AI/action layer — you run one or the other (or eventually both, on different rovers or the same one) depending on what you want watched.

## Hardware

The camera is a DECXIN-SM-2930V1, a USB Video Class (UVC) camera module from a small Chinese OEM (Shenzhen Dechuangxin Imaging Technology). No vendor driver is needed — Windows, Linux, and Raspberry Pi OS all handle UVC devices natively. It's currently plugged into a Windows laptop; the production target is a Raspberry Pi per rover. It turns out this board is actually a **binocular (2-lens) stereo camera**, confirmed by probing it: it exposes itself to the OS as a single UVC device, but at its highest-quality mode (3840x1080) the frame is two clean 1920x1080 halves side by side — the left lens's view and the right lens's view, stitched together, not a single wide shot. That two-lens setup is what makes distance-to-object measurement possible (see "Stereo distance estimation" below) — a single camera can't do this on its own.

One important caveat found by inspecting real frames from the board: **the two lenses are not identical.** The left is visibly wider-angle (more of the scene, more barrel distortion, subjects appear smaller) while the right is narrower/more magnified and noticeably sharper; their color and exposure differ slightly too. This is common on cheap "binocular" modules, which are often built for face liveness/anti-spoofing rather than depth sensing. Stereo depth still works — OpenCV's calibration and rectification handle differing per-lens intrinsics, and this was verified end-to-end against synthetic data with deliberately mismatched focal lengths — but it has two practical consequences: the usable stereo region is only where the two views overlap, so the narrower right lens is the binding constraint on framing, and disparity matching will be somewhat noisier than it would be with a matched pair. If depth quality turns out to be inadequate for the rovers, a purpose-built stereo module with matched lenses (or a depth camera) is the fallback, but that decision should wait for real calibrated numbers rather than being made up front.

For AI compute, there are two separate DGX Spark deployments and it's important not to conflate them. A dual-Spark cluster (dev/current) is where development work happens — not near the farm. Separately, a DGX Spark sits physically near the farm itself, and that is the one that will run this system in production. See "AI serving (planned production)" below for how that changes things.

> Endpoints, hostnames and network addresses are deliberately kept out of this repository. They're supplied at runtime through environment variables — see `.env.example`.

## AI serving (current / dev)

The model in current use is DeepSeek V4 Flash (`deepseek-ai/DeepSeek-V4-Flash-0731`, a 284B-parameter MoE with 13B active), served through vLLM with `--tensor-parallel-size 2` across two clustered DGX Spark units, exposed as an OpenAI-compatible endpoint. No real API key is required — the server accepts any non-empty string, which is what these scripts send. It isn't dedicated hardware: the host also runs several other internal services, which is why the server is configured conservatively and why we treat it as shared infrastructure that shouldn't be restarted or hammered with requests.

Two things about this deployment matter for how we use it. First, tool/function calling is enabled server-side (`--tool-call-parser deepseek_v4 --enable-auto-tool-choice`), which is what lets the model actually call `sound_alarm`, `log_event`, and the other functions we've defined rather than just replying with text. Second, it caps concurrent requests at 4 total, shared across every service on the box — our scripts deliberately space out AI calls (5-8 second minimum interval, only firing on a *change* in what's observed rather than every frame) so a farm agent doesn't starve other users of that shared capacity. Connection details and admin notes are kept in a local file that is deliberately excluded from this repository.

**This entire setup is temporary.** It's what's reachable from here during development, but it is not what the deployed system will run on.

## AI serving (planned production)

The actual production target is a DGX Spark located near the farm itself — a single unit, not the dual-node cluster used for development. The plan is to run a considerably smaller model there than DeepSeek V4 Flash, because the AI's job in this system is narrow: read a short text observation (an object list, or an anomaly score) and decide which of a handful of tools to call. That's orchestration/routing, not open-ended reasoning or generation, and doesn't need a frontier-scale model — a smaller model should be faster to respond, cheaper to run continuously on a single Spark with no sharing constraints, and still perfectly capable of reliable tool calling if it's a model with solid function-calling support.

Nothing in the codebase assumes a specific model or endpoint — every script takes `--api-base` and `--model` (or the `DEEPSEEK_API_BASE` / `DEEPSEEK_MODEL` environment variables), so pointing this whole system at the farm's Spark instead of the dev cluster will be a configuration change, not a code change. The `farm_ai_actions.py` tool definitions and system prompt are already written in fairly generic terms (not tied to DeepSeek-specific behavior) for exactly this reason.

### Model candidates for the orchestration role

Since the job is picking between a handful of tools from a short text description — not writing code or long-form reasoning — there's real headroom to go small. Two tiers worth considering, based on current (Aug 2026) published benchmarks for local tool-calling models:

A **~4-8B tier** (e.g. Qwen3-4B-Instruct-2507, Mistral-7B-Instruct-v0.3) is the leanest option — fast, tiny memory footprint, leaves the vast majority of the Spark's 128GB unified memory free for concurrency. No hard tool-calling reliability numbers were published for this tier specifically, so it would need to be benchmarked against our actual 4-tool schema (`sound_alarm`, `log_event`, `call_phone`, `control_smart_device`) before trusting it unattended — the task is simple enough that this tier may well be reliable enough, but that's an assumption to verify, not a given.

A **~27-32B tier** (Qwen3-32B, Gemma-4-27B, or GLM-4.7-32B) is the safer default: independently benchmarked at roughly 93-95% well-formed tool calls on general agentic tasks. Still comfortably fits alongside a healthy concurrency budget on a single Spark's 128GB, with no tensor-parallel/multi-node complexity needed. Qwen's function-calling track record and active maintenance make Qwen3-32B the natural first thing to try; Gemma-4-27B and GLM-4.7-32B are close alternatives if it doesn't work out.

**Recommendation: start with Qwen3-32B, and benchmark a smaller Qwen3-4B/7B-class model against it on our real tool schema before committing.** If the smaller model performs comparably on our specific (fairly simple) task, it's the better long-term choice for a fleet this size — more concurrency headroom per watt on hardware that isn't shared with anything else.

### Memory estimate for `--max-num-seqs 24`

Using each model's real architecture (layers, KV heads, head dim — pulled from their published configs, not guessed), KV cache size per token is `2 × layers × kv_heads × head_dim × bytes_per_element`. Worst case with `--max-model-len 4096`, `--max-num-seqs 24`, fp8 weights, and fp8 KV cache (all 24 slots simultaneously at full length, no prefix-caching credit):

| Model | Layers / KV heads / head_dim | Weights (fp8) | KV cache (24×4096 tok) | Total | Free of 128GB |
|---|---|---|---|---|---|
| Qwen3-32B | 64 / 8 / 128 | 30.5 GB | 12.0 GB | ~45.5 GB | ~82.5 GB |
| Qwen3-14B | 40 / 8 / 128 | 13.8 GB | 7.5 GB | ~24.3 GB | ~103.7 GB |
| Mistral-7B-Instruct-v0.3 | 32 / 8 / 128 | 6.8 GB | 6.0 GB | ~15.8 GB | ~112.2 GB |
| Qwen3-4B-Instruct-2507 | 36 / 8 / 128 | 3.7 GB | 6.8 GB | ~13.5 GB | ~114.5 GB |

All four fit with large headroom even in this worst case — memory isn't the constraint at these sizes on a 128GB Spark, so the model choice should be driven by tool-call reliability and latency, not by what fits. Worth noting: KV cache size depends on layer count and KV-head count, not total parameters, which is why Qwen3-4B's per-token KV cost is actually higher than Mistral-7B's despite being the smaller model — that gap matters once you're running 24 concurrent sequences, even if it's invisible at batch size 1. (Params-based weight figures are the commonly published approximations, not exact down to the embedding table — fine for provisioning, not for the last few hundred MB.)

Suggested launch flags beyond `--max-num-seqs 24`: `--tensor-parallel-size 1` (fits on one Spark, no need to split), `--max-model-len 4096` (right-sized for our short prompts — this directly caps worst-case KV memory), `--gpu-memory-utilization 0.85` (can be more generous than the dev box's 0.75 since this Spark isn't shared), `--enable-prefix-caching` (the system prompt + tool schema is identical on every request — real savings), `--kv-cache-dtype fp8` plus an fp8-quantized checkpoint (GB10 handles low precision natively — the dev deployment already does this with `nvfp4_ds_mla`), `--enable-auto-tool-choice` plus a `--tool-call-parser` matched to whichever model is chosen (check vLLM's current supported list for that exact model — worth verifying fresh rather than assuming), and if the model has a "thinking" mode, disabling it by default the same way the dev deployment does, since orchestration doesn't need chain-of-thought.

### Concurrency at fleet scale

Unlike the dev box's 4-slot cap (shared with other internal services), the farm's Spark will be dedicated to this system alone, and a 4-32B model has a far smaller KV-cache footprint per request than the 284B-parameter DeepSeek V4 Flash. That means `--max-num-seqs` can be set much higher — likely 32-64+ depending on the model chosen and actual memory headroom, enough to comfortably cover bursts from the full rover fleet (see below) without the queuing behavior we had to design around on the dev box.

## Fleet scale

The production deployment is 6-7 rovers, each carrying 2-3 cameras — 12-21 camera streams total, all ultimately reporting to the single farm-side DGX Spark. This is a meaningful step up from the one-camera prototype built so far, and it changes a few things worth planning for even before hardware is available: each rover will likely run its own local process(es) per camera (extending `farm_camera_agent.py` / `plant_anomaly_agent.py` rather than one process handling 2-3 cameras serially), all pointed at the same `--api-base`; the existing per-camera debounce (only fire on a *change*, minimum seconds between calls) becomes more important, not less, since 12-21 independently-debounced streams can still produce a meaningful burst of simultaneous requests; and it's worth deciding whether each rover should log/report locally (as today) or whether there should be a lightweight central collector so a human can see fleet-wide status in one place instead of SSH-ing into each rover's `events.log`. That last point is flagged again under Open Items — worth designing once there's a second physical rover to plan against, rather than guessing at a topology now.

## Software stack

Python, managed with `uv` for environments and package installs (faster and less friction than raw pip, especially for a `.venv` per project). Camera capture and image handling go through OpenCV (`opencv-python`). Object detection uses Ultralytics' YOLOv8 (`ultralytics`), specifically the pretrained `yolov8n.pt` nano model — small and fast enough for real-time use on a laptop or a Pi 4/5. Plant anomaly detection uses only OpenCV and `numpy` — deliberately no deep learning dependency there, so it stays light enough for a GPU-less Raspberry Pi. Talking to DeepSeek V4 Flash goes through the official `openai` Python package, since vLLM's endpoint is OpenAI-API-compatible. On Windows, `pygrabber` is an optional dependency that resolves camera *names* (not just numeric indices) so the DECXIN can be selected reliably even with other cameras (like a laptop's built-in webcam) present.

## Repository layout

```
Camera/
├── common/                 camera access, YOLO detection, LLM tools + agents
├── decxin-sm-2930v1/       binocular stereo camera -> object distance
└── ndvi/                   NDVI-converted camera -> plant health
```

`common/camera_utils.py` is the single cross-platform camera layer (list, identify by name, open with the right backend). `common/farm_ai_actions.py` holds the tools the LLM may call and the code behind each, shared by both agents so every rover behaves identically. `common/farm_camera_detect.py` is live YOLOv8 detection, and the two agents (`farm_camera_agent.py`, `plant_anomaly_agent.py`) sit alongside it — all three work with any camera, which is why they're in `common/` rather than either camera's folder.

Each camera folder has its own README documenting that hardware, since neither module is meaningfully documented by its manufacturer. Scripts inside those folders add `../common` to their import path automatically, so any of them can be run directly from its own directory.

`DEEPSE_1.MD` (not in this repository — see `.gitignore`) is the private connection reference for the LLM deployment. Keep it locally; it is intentionally never committed.

## Stereo distance estimation

The DECXIN board's two lenses (see "Hardware" above) enable measuring how far away something is, the same way human binocular vision does: an object's position shifts by a different amount between the left and right view depending on how far away it is (its "disparity") — close things shift a lot, far things barely shift. Once calibrated, `distance = (focal_length × baseline) / disparity` turns that shift into a real-world distance. Getting there needs three things: knowing exactly how the two lenses are exposed (confirmed — one combined frame, split cleanly in half), a calibration pass with a checkerboard target to solve for lens distortion, per-lens focal length, and the two lenses' true relative geometry (including the baseline — the software derives this automatically from the checkerboard's known real-world size rather than needing it hand-measured), and rectification so the same physical point lands on the same image row in both halves before matching them up.

The pipeline, in the order you'd run it: `stereo_probe.py` (done — confirmed the board outputs one 3840x1080 frame that's two clean 1920x1080 halves, left and right). `generate_checkerboard.py` produces a printable calibration target (print at 100%/actual size, mount on something rigid). `stereo_capture.py --collect` shows the live split with real-time checkerboard-detection feedback and saves left/right image pairs on demand (aim for 15-25, varied distance/angle/position). `stereo_calibrate.py` solves the calibration from those pairs and reports its own quality metric (reprojection RMS error — flags a warning above 1.0px) plus the baseline it computed, which is worth cross-checking against a ruler measurement of the board as a sanity check, not a hard requirement. `stereo_depth.py` then does live distance estimation — a colorized depth map you can click to query a distance, or with `--with-detection`, YOLOv8 running on the rectified left frame with each detected object labeled by its estimated distance (e.g. "cow 3.2m").

Before shipping this, the core math (calibration correctly recovering a known baseline, and disparity-to-distance reprojection correctly recovering a known depth) was verified against synthetic data with ground-truth values, not just trusted from the OpenCV calls looking right — both passed. What hasn't been validated yet is calibration quality against the *real* camera and a *real* printed checkerboard, which needs you to actually run `stereo_capture.py --collect` and `stereo_calibrate.py` and check the reported RMS error.

Once distances are available, the natural next step is folding them into `farm_camera_agent.py`'s observations to the AI — "cow detected 3.2m away" carries more decision-relevant signal than "cow detected" alone (something close may warrant more urgency than the same thing far off). That integration hasn't been built yet; it's a reasonable follow-up once `stereo_depth.py --with-detection` is confirmed working on the real board.

## Why text, not images

DeepSeek V4 Flash is being used here purely as a text/tool-calling reasoner, not a vision model. Object detection (YOLOv8) and anomaly scoring (the HSV/texture comparison) both happen locally, on-device, and only their *output* — a handful of words or a numeric score — gets sent over the network. This is deliberately cheap and fast: no image encoding, no large payloads, and it fits a deployment that isn't currently configured for vision input anyway. If a future need requires the AI to actually judge an image (distinguishing look-alike issues, judging severity from appearance), that's a separate, explicit upgrade — sending a base64 frame to a vision-capable model only when the local detector already flagged something worth a closer look, to keep bandwidth and shared-server load down.

## Safety posture for actions

Of the four tools the AI can call, two are real and two are intentionally stubbed. `sound_alarm` and `log_event` only touch the machine running the script (a beep through local speakers, a line appended to `events.log`) — safe to let the AI trigger autonomously. `call_phone` and `control_smart_device` currently just log what the AI *would* have done, because they'd require real credentials (a Twilio account, a specific smart device's API) that don't exist yet and that we shouldn't guess at. Wiring either up for real is a small, well-scoped addition once you decide on a phone/SMS provider or tell us the smart device's brand — the integration points are already marked with TODOs in `farm_ai_actions.py`.

## Current status

Confirmed working: the DECXIN camera connects and streams reliably on Windows via OpenCV (after adding a warm-up read and retry tolerance for a driver quirk where the first frame or two can fail); YOLOv8 object detection runs live with a preview window; the `openai`-client bridge to DeepSeek V4 Flash at `<your-llm-server>:8888` is wired in with the correct model name and connects successfully; the board is confirmed to be a stereo (2-lens) camera exposing one clean side-by-side frame, and the stereo calibration/depth math has been verified against synthetic ground-truth data.

Paused: `plant_anomaly_agent.py`'s calibration and threshold tuning need a real camera pointed at real (or at least plausible stand-in) plants, which isn't possible until there's access to the farm again. The code is written and the underlying scoring logic has been sanity-tested against synthetic data, but it hasn't been calibrated or validated against an actual plant yet.

Not started: real calibration of `stereo_depth.py` against the actual camera (needs a printed checkerboard and captured image pairs — code is ready, hasn't been run against real hardware), folding distance estimates into `farm_camera_agent.py`'s AI observations, real `call_phone` / `control_smart_device` integrations, Raspberry Pi field deployment, and any design for multiple rovers reporting to a shared place rather than each running an isolated loop.

## Open items

Pick and benchmark the smaller orchestration model for the farm's DGX Spark against our actual tool schema (Qwen3-32B is the current working recommendation, see "AI serving (planned production)" above) — this is the biggest architectural decision still open. Run the stereo calibration pipeline against the real DECXIN board (print `checkerboard.png`, capture pairs with `stereo_capture.py --collect`, calibrate, then check `stereo_depth.py`'s distances against a tape measure) and fold distances into the AI's observations once trustworthy. Decide on and wire a real phone/SMS provider (Twilio is the sketched-out default) and a real smart-device target once you know the brand/API. Validate the whole stack on an actual Raspberry Pi rather than assuming portability holds. Once farm access resumes, calibrate `plant_anomaly_agent.py` on the real crop and tune its alert threshold from real data instead of the synthetic test. Design how the 6-7 rovers (2-3 cameras each) report findings — one shared collector vs. each rover logging independently — before the fleet is actually deployed, since retrofitting that after the fact is more disruptive than deciding it up front.

## Quick reference

```powershell
# One-time environment setup (from the repo root)
uv venv
uv pip install opencv-python numpy ultralytics openai pygrabber

# Object detection, and the agents (camera-agnostic)
cd common
uv run farm_camera_detect.py --camera-name DECXIN
uv run farm_camera_agent.py --camera-name DECXIN
uv run plant_anomaly_agent.py --camera-name DECXIN --calibrate

# Stereo distance
cd decxin-sm-2930v1
uv run stereo_depth.py --camera-name DECXIN --disparity-offset 2.9

# NDVI plant health
cd ndvi
uv run ndvi_probe.py --camera <index>
uv run ndvi_live.py --camera <index>
```
