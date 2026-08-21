"""
farm_camera_agent.py

Watches a camera, runs YOLOv8 object detection, and periodically tells an
AI model behind an OpenAI-compatible endpoint what it currently sees --
as plain text (object labels), not images. The AI can then call a tool to
trigger an action on the machine running this script.

Must live in the SAME FOLDER as farm_camera_detect.py and
farm_ai_actions.py -- this script reuses the camera-opening helpers from
the former and the AI/tool-calling logic from the latter (shared with
plant_anomaly_agent.py, so every rover behaves consistently).

--------------------------------------------------------------------
Setup (uv, Windows):

    uv pip install opencv-python ultralytics pygrabber openai

Point it at your LLM server -- either pass it every run:

    uv run farm_camera_agent.py --camera-name DECXIN ^
        --api-base $DEEPSEEK_API_BASE --model $DEEPSEEK_MODEL

...or set it once so you don't have to retype it:

    setx DEEPSEEK_API_BASE "http://<your-llm-server>:8888/v1"
    setx DEEPSEEK_MODEL "your-model-name"
    # close/reopen the terminal after setx, then just:
    uv run farm_camera_agent.py --camera-name DECXIN

A placeholder API key is sent; local serving stacks (vLLM, SGLang,
Ollama) ignore it, and anything that does enforce auth should be reached
through the standard OpenAI-client environment variables instead.

If the endpoint is shared with other users, keep --min-interval
conservative (the default, 5 s between AI calls) so one agent can't hog
it. A dedicated host can afford a lower value.

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
    parser.add_argument("--source", type=str, default=None,
                        help="Network stream URL (rtsp:// or http://) from a remote "
                             "camera -- e.g. what a rover's vision SoC sends. Overrides "
                             "--camera / --camera-name; reads that fail repeatedly "
                             "trigger a reconnect instead of stopping.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--model-weights", type=str, default="yolov8n.pt",
                         help="YOLO weights used for object detection")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--api-base", type=str,
                         default=os.environ.get("DEEPSEEK_API_BASE", ""),
                         help="OpenAI-compatible endpoint of your LLM server (vLLM, "
                              "SGLang, Ollama, ...), e.g. http://<llm-server>:8000/v1. "
                              "Can also be set via the DEEPSEEK_API_BASE environment "
                              "variable.")
    parser.add_argument("--model", type=str,
                         default=os.environ.get("DEEPSEEK_MODEL", ""),
                         help="Model name AS REGISTERED ON YOUR SERVER, not the "
                              "HuggingFace path. Can also be set via the DEEPSEEK_MODEL "
                              "environment variable.")
    parser.add_argument("--min-interval", type=float, default=5.0,
                         help="Minimum seconds between AI calls, even if detections keep "
                              "changing -- avoids spamming the model on every frame.")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    if not args.api_base:
        print("ERROR: no AI endpoint configured.")
        print("Pass --api-base http://<llm-server>:8000/v1, or set DEEPSEEK_API_BASE.")
        return 1
    if not args.model:
        print("ERROR: no model name configured.")
        print("Pass --model <name-as-registered-on-your-server>, or set DEEPSEEK_MODEL.")
        return 1

    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: the 'openai' package is not installed.")
        print("Install it with: uv pip install openai")
        return 1

    client = OpenAI(base_url=args.api_base, api_key="not-needed")

    camera_index = cam.resolve_camera(args.camera, args.camera_name)
    if camera_index is None:
        return 1

    source = args.source or camera_index
    cap = cam.open_camera(source, args.width, args.height)
    if not cap.isOpened():
        print(f"ERROR: could not open {args.source or f'camera index {camera_index}'}.")
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
                    if args.source:
                        # A network source going quiet and coming back is normal
                        # (rover out of WiFi range, encoder restarting) -- recover
                        # by reconnecting rather than stopping.
                        print("WARNING: stream unreadable for a while -- reconnecting "
                              f"to {args.source} ...")
                        cap.release()
                        cap = cam.open_camera(args.source, args.width, args.height)
                        if cap.isOpened():
                            consecutive_failures = 0
                            continue
                        print("  (reconnect failed; will keep retrying)")
                    else:
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
