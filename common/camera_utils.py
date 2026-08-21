"""
camera_utils.py

Cross-platform camera access: find local cameras, identify them by name, and
open them with the right backend for the current OS -- or open a network
stream from a remote camera instead (see is_stream_url).

This is a deliberate, self-contained copy of the equivalent helpers in the
stereo-camera project rather than a shared import. Each camera's code stays
independently deployable -- a rover carrying only the NDVI camera can run
this directory on its own, with nothing else present. The duplication is a
few dozen lines and is worth it for that.
"""

import platform

import cv2


def get_camera_names() -> list[str]:
    """Camera friendly names, ordered so names[i] matches camera index i.
    Returns [] when names can't be resolved on this platform -- callers must
    read that as "fall back to numeric indices", not "no cameras found".

      Windows: DirectShow enumeration via the optional `pygrabber` package;
               order matches cv2.CAP_DSHOW exactly, so this is reliable.
      macOS:   parsed from `system_profiler`. Order usually matches
               AVFoundation but Apple doesn't guarantee it -- confirm by eye.
      Linux:   not implemented; use --list-cameras and pick by index.
    """
    system = platform.system()

    if system == "Windows":
        try:
            from pygrabber.dshow_graph import FilterGraph
        except ImportError:
            return []
        try:
            return FilterGraph().get_input_devices()
        except Exception:
            return []

    if system == "Darwin":
        import json
        import subprocess
        try:
            out = subprocess.run(
                ["system_profiler", "-json", "SPCameraDataType"],
                capture_output=True, text=True, timeout=15,
            )
            data = json.loads(out.stdout)
            return [item.get("_name", "") for item in data.get("SPCameraDataType", [])]
        except Exception:
            return []

    return []


def list_cameras(max_index: int = 8) -> list[tuple[int, str, str]]:
    """Probe indices 0..max_index-1. Returns (index, name, resolution) for
    each that opens and delivers a frame."""
    names = get_camera_names()
    found = []
    for i in range(max_index):
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
        cap = cv2.VideoCapture(i, backend)
        if cap is not None and cap.isOpened():
            ok, frame = cap.read()
            if ok:
                name = names[i] if i < len(names) else ""
                found.append((i, name, f"{frame.shape[1]}x{frame.shape[0]}"))
            cap.release()
    return found


def print_camera_list(max_index: int = 8) -> None:
    found = list_cameras(max_index)
    if not found:
        print("No cameras found. Check the USB connection.")
        return
    for idx, name, res in found:
        label = name if name else "(name unavailable on this platform)"
        print(f"  index {idx}: {label}  [{res}]")


def describe_camera(index: int) -> str:
    """Friendly name for an already-chosen index, or '' if unavailable.
    Used to report which camera actually got opened -- USB indices shift
    whenever devices are plugged or unplugged, so an index that was right
    yesterday can silently point at a different camera today."""
    names = get_camera_names()
    return names[index] if 0 <= index < len(names) else ""


def resolve_camera(index: int, name_fragment: str | None):
    """Turn a --camera/--camera-name pair into a concrete index.
    Returns the index, or None if a name was given and didn't match."""
    if not name_fragment:
        return index
    names = get_camera_names()
    match = next(
        (i for i, n in enumerate(names) if name_fragment.lower() in n.lower()), None)
    if match is None:
        print(f"ERROR: no camera name containing '{name_fragment}' found.")
        if names:
            print(f"Available: {names}")
        else:
            print("No camera names available on this platform -- use --camera <index>.")
        print("Run with --list-cameras to see what's connected.")
        return None
    print(f"Matched '{name_fragment}' -> camera index {match} ({names[match]})")
    return match


def is_stream_url(source) -> bool:
    """True when `source` is a network stream URL (rtsp://, rtmp://, http://,
    https://) rather than a local camera index. This is how a rover's camera
    arrives: its vision SoC encodes H.264/H.265 and serves it as RTSP, or
    MJPEG over HTTP, and the analysis machine opens it like a camera."""
    return isinstance(source, str) and source.lower().startswith(
        ("rtsp://", "rtmp://", "http://", "https://"))


def open_stream(url: str, timeout_ms: int = 10_000) -> cv2.VideoCapture:
    """Open a network stream (RTSP with H.264/H.265, or MJPEG over HTTP).

    Three things differ from a local camera, all worth knowing:

    - There is nothing to warm up or configure. Exposure, resolution and
      frame rate were decided by whatever encodes the stream; CAP_PROP
      writes that seem to succeed are silently ignored by the sender.
    - The capture buffer is deliberately kept tiny. OpenCV queues frames
      as they arrive, so a consumer that takes 200 ms per frame on the
      default buffer ends up showing seconds-old footage -- the view
      falls further behind reality instead of skipping ahead. A buffer
      of 1 trades smoothness for freshness, which is the right trade
      here: a stale frame is a wrong measurement.
    - Opening a DEAD sender blocks for a while before failing. The open/
      read timeout properties below bound HTTP-family sources, but RTSP
      ignores them in current OpenCV builds -- measured: a 30 s hang on
      an unreachable rtsp:// URL either way. To bound RTSP too, set
      OPENCV_FFMPEG_CAPTURE_OPTIONS="timeout;10000000" (microseconds)
      in the shell BEFORE starting Python -- OpenCV reads it at import
      time, so setting it from code after cv2 is loaded has no effect.
      The reconnect loops in the scripts tolerate this; it just makes
      each retry cycle slower than it looks.
    """
    cap = cv2.VideoCapture()
    try:
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms)
    except Exception:
        pass  # properties unknown to this OpenCV build -- defaults apply
    cap.open(url, cv2.CAP_FFMPEG)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass  # not every backend honours it; the stream still works
    return cap


def open_camera(index_or_url, width: int, height: int) -> cv2.VideoCapture:
    """Open a camera with the backend that works best on this OS.

    Windows: DirectShow is the most reliable for generic UVC devices.
    Linux:   V4L2 talks directly to /dev/video<index>.
    macOS:   default backend (AVFoundation).

    Passing a stream URL string (see is_stream_url) opens that network
    stream instead -- width/height are then ignored, because the sender's
    encoder decides the frame size.
    """
    if is_stream_url(index_or_url):
        return open_stream(index_or_url)

    index = index_or_url
    system = platform.system()
    if system == "Windows":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    elif system == "Linux":
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def warm_up(cap, frames: int = 10, delay: float = 0.05) -> None:
    """Discard the first few frames. Many UVC drivers fail the first read or
    two, and auto-exposure/white-balance need a moment to settle."""
    import time
    for _ in range(frames):
        cap.read()
        time.sleep(delay)
