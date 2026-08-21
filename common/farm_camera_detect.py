"""
farm_camera_detect.py

Real-time object detection for a DECXIN-SM-2930V1 (or any other UVC/USB)
camera using OpenCV + a pretrained YOLOv8 model (via the `ultralytics`
package).

Works today on Windows, and needs no code changes to run later on Linux
or a Raspberry Pi -- only the `--backend` default changes automatically,
and camera indices/paths are auto-detected.

--------------------------------------------------------------------
Setup (Windows, using uv):

    uv venv
    uv pip install opencv-python ultralytics pygrabber
    # pygrabber is optional but recommended on Windows: it lets this script
    # show camera NAMES (e.g. "DECXIN USB Camera") next to their index, so
    # you can tell the built-in laptop webcam apart from the DECXIN one.

Setup (Linux / Raspberry Pi, using uv):

    uv venv
    uv pip install opencv-python ultralytics
    # Raspberry Pi: prefer the lighter "opencv-python-headless" build if
    # you don't need cv2.imshow() (e.g. running headless over SSH):
    #   uv pip install opencv-python-headless ultralytics

(No `uv`? Swap `uv pip install X` for `pip install X` after activating the
venv with `.venv\\Scripts\\activate` on Windows or `source .venv/bin/activate`
on Linux/macOS -- everything else below is identical.)

Run (uv run works whether or not the venv is "activated"):

    uv run farm_camera_detect.py --list-cameras     # find/identify your cameras first
    uv run farm_camera_detect.py --camera 1          # pick by index
    uv run farm_camera_detect.py --camera-name DECXIN  # pick by name (Windows + pygrabber)
    uv run farm_camera_detect.py --source rtsp://rover-3.local:8554/cam
                                                    # a rover's stream instead of a
                                                    # local camera (RTSP/H.265 or
                                                    # MJPEG-over-HTTP)
    uv run farm_camera_detect.py --model yolov8n.pt --conf 0.4

Press 'q' to quit the preview window.
--------------------------------------------------------------------
"""

import argparse
import sys
import time

import cv2


from camera_utils import (get_camera_names, list_cameras,  # noqa: F401
                          open_camera, resolve_camera)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0,
                         help="Camera index (default: 0). Use --list-cameras to find it.")
    parser.add_argument("--source", type=str, default=None,
                         help="Network stream URL (rtsp:// or http://) from a remote "
                              "camera -- e.g. what a rover's vision SoC sends. Overrides "
                              "--camera / --camera-name. Reads that fail repeatedly "
                              "trigger a reconnect instead of stopping, because a rover "
                              "dropping off the network and coming back is normal.")
    parser.add_argument("--width", type=int, default=1280, help="Requested capture width")
    parser.add_argument("--height", type=int, default=720, help="Requested capture height")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                         help="Ultralytics model to use. yolov8n.pt is the fastest/smallest "
                              "pretrained model, good for real-time on a laptop or Raspberry Pi. "
                              "Swap in a custom-trained .pt file later for farm-specific classes.")
    parser.add_argument("--conf", type=float, default=0.35, help="Minimum detection confidence")
    parser.add_argument("--camera-name", type=str, default=None,
                         help="Auto-select the camera whose name contains this text "
                              "(case-insensitive), e.g. --camera-name DECXIN. "
                              "Windows-only, requires 'pygrabber'. Overrides --camera.")
    parser.add_argument("--list-cameras", action="store_true",
                         help="Probe camera indices and exit (use this first if unsure "
                              "which index the DECXIN camera is on).")
    parser.add_argument("--no-display", action="store_true",
                         help="Don't open a preview window (e.g. running headless over SSH). "
                              "Detections are printed to the console instead.")
    args = parser.parse_args()

    if args.list_cameras:
        print("Probing camera indices 0-7 ...")
        found = list_cameras()
        if not found:
            print("No cameras found. Check the USB connection / drivers.")
        else:
            for idx, name, res in found:
                label = name if name else "(name unavailable on this platform)"
                wide = ""
                w, h = (int(v) for v in res.split("x"))
                if w >= 2 * h:
                    wide = "   <-- WIDE frame: this is the stereo board"
                print(f"  index {idx}: {label}  [{res}]{wide}")
        return 0

    camera_index = resolve_camera(args.camera, args.camera_name)
    if camera_index is None:
        return 1

    source = args.source or camera_index
    cap = open_camera(source, args.width, args.height)
    if not cap.isOpened():
        print(f"ERROR: could not open {args.source or f'camera index {camera_index}'}.")
        if args.source:
            print("Check the URL, and that the sender is actually streaming "
                  "(e.g. ffprobe <url>, or open it in VLC).")
        else:
            print("Try: uv run farm_camera_detect.py --list-cameras")
        return 1

    # Warm-up: some UVC drivers (this DECXIN included) return a failed read
    # or garbage on the first grab or two while auto-exposure/white-balance
    # settle. Discard a handful of frames before trusting the stream. Harmless
    # on a network source too -- it lets the decoder settle the same way.
    for _ in range(10):
        cap.read()
        time.sleep(0.05)

    # Import ultralytics lazily so --list-cameras doesn't require it installed.
    print("Loading detection libraries (PyTorch is a large import -- this can take "
          "30-90 seconds on the very first run, especially with antivirus scanning "
          "enabled). Please wait, this is normal...", flush=True)
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: the 'ultralytics' package is not installed.")
        print("Install it with: uv pip install ultralytics")
        return 1

    print(f"Loading model '{args.model}' (first run also downloads ~6MB of "
          f"pretrained weights)...", flush=True)
    model = YOLO(args.model)

    print("Camera opened. Press 'q' in the preview window to quit "
          "(or Ctrl+C in this terminal if running with --no-display). "
          "If no window appears, check the taskbar / Alt+Tab -- it can open "
          "without stealing focus.", flush=True)

    prev_time = time.time()
    consecutive_failures = 0
    # Tolerate transient read failures instead of giving up on the first one.
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
                        cap = open_camera(args.source, args.width, args.height)
                        if cap.isOpened():
                            consecutive_failures = 0
                            continue
                        print("  (reconnect failed; will keep retrying)")
                    else:
                        print("WARNING: failed to read frame from camera repeatedly, stopping. "
                              "Is another program (another instance of this script, the "
                              "Windows Camera app, etc.) still holding the camera open?")
                        break
                time.sleep(0.05)
                continue
            consecutive_failures = 0

            results = model.predict(frame, conf=args.conf, verbose=False)
            result = results[0]

            # Names of everything detected in this frame, e.g. ['person', 'dog', 'apple']
            labels = [result.names[int(box.cls)] for box in result.boxes]
            if labels:
                print(f"Detected: {', '.join(labels)}")

            if not args.no_display:
                annotated = result.plot()  # draws boxes + labels on the frame

                now = time.time()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now
                cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow(args.source or "camera - Object Detection", annotated)
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
