"""
plant_anomaly_agent.py

Rover-style anomaly watcher for scanning leaves/plants. YOLO's fixed 80
COCO classes (used in farm_camera_agent.py) have zero categories for
"diseased leaf" or "pest damage" -- they're built for people/cars/animals,
not crop health. Since there's no labeled disease data to train on yet,
this script takes a different approach: it learns what NORMAL/healthy
looks like for whatever it's pointed at, then flags anything that looks
meaningfully different. No labeled examples required.

Two phases:
  1. Calibrate (--calibrate): point the camera at healthy plants for a
     short burst. It learns the normal range of variation.
  2. Monitor (default): compares every live frame against that learned
     baseline. When something deviates enough, it's described to
     the LLM in plain text (a score, not an image -- same
     text-only approach as farm_camera_agent.py), which decides whether
     to act.

How "normal" is measured: an HSV color histogram (catches color changes
like browning/yellowing/spots) plus Laplacian variance, a cheap texture
proxy (catches surface changes like holes, lesions, wilting). Compared
via Bhattacharyya distance. This is intentionally simple and
dependency-light -- just OpenCV and numpy, both already needed -- so it
runs fine on a Raspberry Pi with no GPU. If it proves too coarse for your
crop once you have real examples of problems, swap in a trained
classifier or embedding model later; the calibrate/monitor structure and
the AI hand-off stay the same either way.

Must live in the same folder as farm_camera_detect.py and
farm_ai_actions.py.

--------------------------------------------------------------------
Setup (uv, Windows):

    uv pip install opencv-python numpy openai pygrabber

Step 1 -- calibrate on a HEALTHY plant/leaf (redo this if you move to a
very different plant, scene, or lighting -- the baseline is specific to
what it was calibrated against):

    uv run plant_anomaly_agent.py --camera-name DECXIN --calibrate

Step 2 -- monitor (uses the baseline saved by step 1):

    uv run plant_anomaly_agent.py --camera-name DECXIN ^
        --api-base $DEEPSEEK_API_BASE

Press 'q' to quit either phase.
--------------------------------------------------------------------
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np

import camera_utils as cam
import farm_ai_actions as actions

BASELINE_FILE = "plant_baseline.json"


def frame_signature(frame) -> dict:
    """Cheap descriptor of 'what does this frame look like': an HSV color
    histogram plus a texture measure. Deliberately simple -- no trained
    model, no labeled data needed."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    texture = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {"hist": hist, "texture": texture}


def distance(sig_a: dict, sig_b: dict) -> float:
    """0 = identical, larger = more different. Combines color-histogram
    distance (Bhattacharyya) with normalized texture difference."""
    hist_dist = cv2.compareHist(sig_a["hist"], sig_b["hist"], cv2.HISTCMP_BHATTACHARYYA)
    tex_a, tex_b = sig_a["texture"], sig_b["texture"]
    tex_dist = abs(tex_a - tex_b) / max(tex_a, tex_b, 1.0)
    return 0.7 * hist_dist + 0.3 * min(tex_dist, 1.0)


def nearest_distance(sig: dict, baseline_sigs: list) -> float:
    return min(distance(sig, b) for b in baseline_sigs)


def save_baseline(signatures: list, threshold: float, path: str = BASELINE_FILE) -> None:
    data = {
        "threshold": threshold,
        "hists": [s["hist"].flatten().tolist() for s in signatures],
        "textures": [s["texture"] for s in signatures],
        "hist_shape": list(signatures[0]["hist"].shape),
        "calibrated_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(path, "w") as f:
        json.dump(data, f)


def load_baseline(path: str = BASELINE_FILE):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    shape = tuple(data["hist_shape"])
    sigs = [
        {"hist": np.array(h, dtype=np.float32).reshape(shape), "texture": t}
        for h, t in zip(data["hists"], data["textures"])
    ]
    return sigs, data["threshold"]


def run_calibration(cap, seconds: float, no_display: bool) -> None:
    print(f"Calibrating: point the camera at NORMAL/healthy plants now. "
          f"Capturing for {seconds:.0f}s -- move/rotate slightly to cover the "
          f"range of normal angles and lighting you'll actually see.", flush=True)
    signatures = []
    start = time.time()
    while time.time() - start < seconds:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        signatures.append(frame_signature(frame))
        if not no_display:
            preview = frame.copy()
            cv2.putText(preview, f"Calibrating... {len(signatures)} frames captured",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Calibrating - point at healthy plants, press q to stop early",
                       preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        time.sleep(0.2)  # a few samples per second is plenty

    if not no_display:
        cv2.destroyAllWindows()

    if len(signatures) < 5:
        print("ERROR: not enough frames captured to calibrate "
              "(need at least 5). Try again, or check the camera.")
        return

    # How much do genuinely normal frames vary from each other? Use that to
    # auto-set a threshold instead of asking you to hand-tune one blind.
    pairwise = [
        distance(signatures[i], signatures[j])
        for i in range(len(signatures))
        for j in range(i + 1, len(signatures))
    ]
    mean, std = float(np.mean(pairwise)), float(np.std(pairwise))
    threshold = mean + 3 * std

    save_baseline(signatures, threshold)
    print(f"Calibration done: {len(signatures)} reference frames saved to "
          f"'{BASELINE_FILE}'.")
    print(f"Auto-threshold = {threshold:.3f} (normal frames varied by "
          f"{mean:.3f} +/- {std:.3f} among themselves). Override with "
          f"--threshold if this alerts too much/little once you're monitoring.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--camera-name", type=str, default=None)
    parser.add_argument("--source", type=str, default=None,
                        help="Network stream URL (rtsp:// or http://) from a remote "
                             "camera -- e.g. what a rover's vision SoC sends. Overrides "
                             "--camera / --camera-name.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--calibrate", action="store_true",
                         help="Run calibration instead of monitoring: point the camera at "
                              "healthy/normal plants and it learns what 'normal' looks like.")
    parser.add_argument("--calibrate-seconds", type=float, default=20.0)
    parser.add_argument("--threshold", type=float, default=None,
                         help="Override the auto-computed anomaly threshold from calibration.")
    parser.add_argument("--api-base", type=str,
                         default=os.environ.get("DEEPSEEK_API_BASE", ""),
                         help="OpenAI-compatible endpoint of your LLM server. Can also "
                              "be set via DEEPSEEK_API_BASE.")
    parser.add_argument("--model", type=str,
                         default=os.environ.get("DEEPSEEK_MODEL", ""),
                         help="Model name AS REGISTERED ON YOUR SERVER. Can also be set "
                              "via DEEPSEEK_MODEL.")
    parser.add_argument("--min-interval", type=float, default=8.0,
                         help="Minimum seconds between AI calls for a new anomaly.")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    camera_index = cam.resolve_camera(args.camera, args.camera_name)
    if camera_index is None:
        return 1

    source = args.source or camera_index
    cap = cam.open_camera(source, args.width, args.height)
    if not cap.isOpened():
        print(f"ERROR: could not open {args.source or f'camera index {camera_index}'}.")
        return 1

    # Warm-up: discard the first few frames while auto-exposure/white-balance
    # settle (see farm_camera_detect.py for why this matters on this camera).
    for _ in range(10):
        cap.read()
        time.sleep(0.05)

    if args.calibrate:
        run_calibration(cap, args.calibrate_seconds, args.no_display)
        cap.release()
        cv2.destroyAllWindows()
        return 0

    baseline = load_baseline()
    if baseline is None:
        print(f"ERROR: no baseline found ('{BASELINE_FILE}' doesn't exist yet).")
        print("Run this first, pointing the camera at normal/healthy plants:")
        print("  uv run plant_anomaly_agent.py --camera-name DECXIN --calibrate")
        cap.release()
        return 1
    baseline_sigs, threshold = baseline
    if args.threshold is not None:
        threshold = args.threshold

    if not args.api_base:
        print("ERROR: no AI endpoint configured. Pass --api-base or set DEEPSEEK_API_BASE.")
        cap.release()
        return 1
    if not args.model:
        print("ERROR: no model name configured. Pass --model or set DEEPSEEK_MODEL.")
        cap.release()
        return 1
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: the 'openai' package is not installed. Install: uv pip install openai")
        cap.release()
        return 1
    client = OpenAI(base_url=args.api_base, api_key="not-needed")

    print(f"Monitoring against baseline ({len(baseline_sigs)} reference frames, "
          f"threshold={threshold:.3f}). Press 'q' to quit.", flush=True)

    consecutive_failures = 0
    last_ai_call = 0.0
    in_anomaly = False  # avoid re-alerting every single frame while still anomalous

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                consecutive_failures += 1
                if consecutive_failures > 60:
                    if args.source:
                        # A network source going quiet and coming back is normal
                        # (rover out of WiFi range, encoder restarting) --
                        # reconnect rather than stopping.
                        print(f"WARNING: stream unreadable for a while -- reconnecting "
                              f"to {args.source} ...")
                        cap.release()
                        cap = cam.open_camera(args.source, args.width, args.height)
                        if cap.isOpened():
                            consecutive_failures = 0
                            continue
                        print("  (reconnect failed; will keep retrying)")
                    else:
                        print("WARNING: camera read failing repeatedly, stopping. Is another "
                              "program still holding the camera open?")
                        break
                time.sleep(0.05)
                continue
            consecutive_failures = 0

            sig = frame_signature(frame)
            score = nearest_distance(sig, baseline_sigs)
            anomalous = score > threshold

            now = time.time()
            if anomalous and not in_anomaly and (now - last_ai_call) >= args.min_interval:
                print(f"Anomaly detected: score={score:.3f} (threshold={threshold:.3f})")
                content = (
                    f"Plant/leaf camera anomaly detected. Visual difference score "
                    f"{score:.3f} vs. a learned healthy-plant baseline (alert threshold "
                    f"{threshold:.3f} -- the further above threshold, the more confident "
                    f"this is a real difference, not noise). Possible causes: disease, "
                    f"pest damage, wilting, foreign debris, or just a lighting/angle "
                    f"change -- the detector can't tell which, only that this looks "
                    f"different from normal.\n"
                    f"Timestamp: {datetime.now().isoformat(timespec='seconds')}"
                )
                actions.ask_ai(client, args.model, content)
                last_ai_call = now
                in_anomaly = True
            elif not anomalous:
                in_anomaly = False

            if not args.no_display:
                color = (0, 0, 255) if anomalous else (0, 200, 0)
                label = f"{'ANOMALY' if anomalous else 'normal'}  score={score:.3f}"
                cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.imshow("Plant Anomaly Monitor", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
