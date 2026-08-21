"""
farm_camera_agent.py

Watches a camera, runs YOLOv8 object detection, and periodically tells an
AI model (DeepSeek V4 Flash, served OpenAI-compatible from your dual DGX
Spark) what it currently sees -- as plain text (object labels), not
images. The AI can then call a tool to trigger an action on the machine
running this script.

Must live in the SAME FOLDER as farm_camera_detect.py and
farm_ai_actions.py -- this script reuses the camera-opening helpers from
the former and the AI/tool-calling logic from the latter (shared with
plant_anomaly_agent.py, so every rover behaves consistently).

--------------------------------------------------------------------
Setup (uv, Windows):

    uv pip install opencv-python ultralytics pygrabber openai

Point it at your DeepSeek V4 Flash server -- either pass it every run:

    uv run farm_camera_agent.py --camera-name DECXIN ^
        --api-base $DEEPSEEK_API_BASE --model deepseek-v4-flash-dspark

...or set it once so you don't have to retype it:

    setx DEEPSEEK_API_BASE "http://<your-llm-server>:8888/v1"
    setx DEEPSEEK_MODEL "deepseek-v4-flash-dspark"
    # close/reopen the terminal after setx, then just:
    uv run farm_camera_agent.py --camera-name DECXIN

No API key is needed for this server -- it accepts any non-empty string,
which is what this script already sends.

Note this deployment caps concurrent requests at 4 total, shared with
other services sharing the same box -- the default
--min-interval below (5s between AI calls) is deliberately conservative
so this agent doesn't hog a shared, resource-constrained server.

--------------------------------------------------------------------
What the AI can actually do -- see farm_ai_actions.py:

  - sound_alarm / log_event      -> real (beep + local log file)
  - call_phone / control_smart_device -> stubs with TODOs; tell me the
    phone service or device/brand you want and I'll wire them up for real.

Press 'q' in the preview window to quit.
--------------------------------------------------------------------
"""

import argparse
import os
import sys
import time
from datetime import datetime

import cv2

# Reuse the camera-opening / name-matching helpers from the detection script,
# and the AI/tool-calling logic shared with plant_anomaly_agent.py.
import camera_utils as cam
import farm_ai_actions as actions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--camera-name", type=str, default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--model-weights", type=str, default="yolov8n.pt",
                         help="YOLO weights used for object detection")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--api-base", type=str,
                         default=os.environ.get("DEEPSEEK_API_BASE", ""),
                         help="Your DeepSeek V4 Flash OpenAI-compatible endpoint, e.g. "
                              "http://<dgx-spark-ip>:8000/v1. Can also be set via the "
                              "DEEPSEEK_API_BASE environment variable.")
    parser.add_argument("--model", type=str,
                         default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash-dspark"),
                         help="Model name as registered on your server (default matches "
                              "your deployment, e.g. deepseek-v4-flash-dspark). Can "
                              "also be set via the DEEPSEEK_MODEL environment variable.")
    parser.add_argument("--min-interval", type=float, default=5.0,
                         help="Minimum seconds between AI calls, even if detections keep "
                              "changing -- avoids spamming the model on every frame.")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    if not args.api_base:
        print("ERROR: no AI endpoint configured.")
        print("Pass --api-base http://<dgx-spark-ip>:8000/v1, or set DEEPSEEK_API_BASE.")
        return 1

    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: the 'openai' package is not installed.")
        print("Install it with: uv pip install openai")
        return 1

    client = OpenAI(base_url=args.api_base, api_key="not-needed")

    camera_index = args.camera
    if args.camera_name:
        names = cam.get_camera_names()
        match = next(
            (i for i, n in enumerate(names) if args.camera_name.lower() in n.lower()), None
        )
        if match is None:
            print(f"ERROR: no camera name containing '{args.camera_name}' found.")
            return 1
        camera_index = match
        print(f"Matched '{args.camera_name}' -> camera index {camera_index} ({names[match]})")

    cap = cam.open_camera(camera_index, args.width, args.height)
    if not cap.isOpened():
        print(f"ERROR: could not open camera index {camera_index}.")
        return 1

    # Warm-up: some UVC drivers (this DECXIN included) return a failed read
    # or garbage on the first grab or two while auto-exposure/white-balance
    # settle. Discard a handful of frames before trusting the stream.
    for _ in range(10):
        cap.read()
        time.sleep(0.05)

    print("Loading detection libraries (PyTorch import -- can take up to a minute "
          "on first run). Please wait...", flush=True)
    from ultralytics import YOLO
    model = YOLO(args.model_weights)

    print("Camera + AI bridge running. Press 'q' in the preview window to quit.", flush=True)

    last_labels = set()
    last_ai_call = 0.0
    consecutive_failures = 0
    # Tolerate transient read failures (common right after opening, or a brief
    # driver hiccup) instead of giving up on the very first one.
    MAX_CONSECUTIVE_FAILURES = 60

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                consecutive_failures += 1
                if consecutive_failures == 1:
                    print("  (frame read failed, retrying...)")
                if consecutive_failures > MAX_CONSECUTIVE_FAILURES:
                    print("WARNING: failed to read frame from camera repeatedly, stopping. "
                          "Is another program (a leftover farm_camera_detect.py window, "
                          "the Windows Camera app, etc.) still holding the camera open?")
                    break
                time.sleep(0.05)
                continue
            consecutive_failures = 0

            results = model.predict(frame, conf=args.conf, verbose=False)
            result = results[0]
            labels = sorted({result.names[int(box.cls)] for box in result.boxes})

            now = time.time()
            changed = set(labels) != last_labels
            due = (now - last_ai_call) >= args.min_interval
            if labels and changed and due:
                print(f"Detected change: {labels}")
                content = (
                    f"Camera detections right now: {', '.join(labels)}\n"
                    f"Timestamp: {datetime.now().isoformat(timespec='seconds')}"
                )
                actions.ask_ai(client, args.model, content)
                last_ai_call = now
                last_labels = set(labels)
            elif not labels:
                last_labels = set()

            if not args.no_display:
                annotated = result.plot()
                cv2.imshow("Farm Camera + AI Agent", annotated)
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
