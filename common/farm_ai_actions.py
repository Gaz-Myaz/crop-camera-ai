"""
farm_ai_actions.py

Shared "brain" for the farm AI agents: the tools DeepSeek V4 Flash is
allowed to call, what they actually do, and the plumbing to send it an
observation (plain text -- no images) and dispatch whatever it decides
to do.

Imported by:
  - farm_camera_agent.py   (triggered by YOLO object-detection changes)
  - plant_anomaly_agent.py (triggered by visual anomaly vs. a learned
                             healthy-plant baseline)

Keeping this in one shared file means every rover behaves consistently,
and there's exactly one place to wire in real actions (a real phone
call, a real smart-device webhook) once you're ready -- instead of N
copies drifting apart across scripts/machines.
"""

import json
import time
from datetime import datetime

try:
    import winsound

    def _beep() -> None:
        winsound.Beep(1000, 400)
except ImportError:
    # winsound is Windows-only; fall back to a terminal bell on Linux/macOS/
    # Raspberry Pi so rovers running headless don't just crash on this.
    def _beep() -> None:
        print("\a", end="", flush=True)


# ---------------------------------------------------------------------------
# Actions the AI is allowed to trigger. Kept safe-by-default: the two that
# actually DO something (sound_alarm, log_event) only touch the local
# machine. The other two are stubs until real credentials/URLs are wired in.
# ---------------------------------------------------------------------------

LOG_FILE = "events.log"


def sound_alarm(reason: str = "") -> str:
    for _ in range(3):
        _beep()
        time.sleep(0.1)
    return f"Alarm sounded on this machine ({reason})"


def log_event(message: str) -> str:
    line = f"{datetime.now().isoformat(timespec='seconds')}\t{message}\n"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line)
    return f"Logged: {message}"


def call_phone(number: str, message: str = "") -> str:
    """STUB. Wire this to Twilio (or any call/SMS API) when ready, e.g.:

        from twilio.rest import Client
        client = Client(os.environ["TWILIO_SID"], os.environ["TWILIO_TOKEN"])
        client.calls.create(
            to=number, from_=os.environ["TWILIO_FROM"],
            twiml=f"<Response><Say>{message}</Say></Response>",
        )

    Needs a Twilio account + a phone number -- can't be wired up without
    your credentials. For now this just logs the AI's intent, so you can
    see it actually deciding to do it.
    """
    return log_event(f"[STUB call_phone] would call {number}: {message}")


def control_smart_device(device: str, action: str) -> str:
    """STUB. Wire this to whatever your smart device speaks -- a
    Shelly/Tasmota/Home Assistant webhook is usually a one-line POST:

        import requests
        requests.post(f"http://<device-ip>/relay/0?turn={action}")

    Tell me the device/brand and I'll fill this in for real.
    """
    return log_event(f"[STUB control_smart_device] would set {device} -> {action}")


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "sound_alarm",
            "description": "Sound an audible alarm on the machine that's watching. Use for "
                            "anything that needs immediate human attention right now.",
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "log_event",
            "description": "Record a note about what's happening, without alarming anyone. "
                            "Use for routine or low-confidence observations worth keeping.",
            "parameters": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "call_phone",
            "description": "Place a phone call to alert a human. Reserve for genuinely "
                            "urgent, high-confidence events (not yet wired to a real line).",
            "parameters": {
                "type": "object",
                "properties": {
                    "number": {"type": "string"},
                    "message": {"type": "string"},
                },
                "required": ["number", "message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "control_smart_device",
            "description": "Trigger a smart device such as a light, siren, or relay "
                            "(not yet wired to a real device).",
            "parameters": {
                "type": "object",
                "properties": {
                    "device": {"type": "string"},
                    "action": {"type": "string"},
                },
                "required": ["device", "action"],
            },
        },
    },
]

DISPATCH = {
    "sound_alarm": lambda a: sound_alarm(a.get("reason", "")),
    "log_event": lambda a: log_event(a.get("message", "")),
    "call_phone": lambda a: call_phone(a.get("number", ""), a.get("message", "")),
    "control_smart_device": lambda a: control_smart_device(a.get("device", ""), a.get("action", "")),
}

SYSTEM_PROMPT = (
    "You are a monitoring assistant for an autonomous farm rover. You'll be told an "
    "observation from the rover's cameras/sensors -- either which general object "
    "categories a detector currently sees, or a visual-anomaly score comparing the "
    "current view to a learned 'healthy/normal' baseline (higher score = more "
    "different from normal; treat scores well past the stated threshold as more "
    "confident than ones just barely over it). Decide if the situation needs any "
    "action, then call the ONE most appropriate tool, or none at all if nothing "
    "notable is happening. Don't call sound_alarm or call_phone for routine, "
    "expected, or borderline/low-confidence observations -- reserve those for "
    "something that actually looks unusual, unsafe, or worth a human's immediate "
    "attention. Prefer log_event for anything lower-stakes than that."
)


def ask_ai(client, model: str, content: str) -> None:
    """Send a plain-text observation to the model and execute whatever
    tool(s) it decides to call, if any. No images are sent -- see the
    module docstring."""
    print(f"  -> asking AI: {content}")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            tools=TOOLS,
            tool_choice="auto",
        )
    except Exception as e:
        print(f"  -> AI request failed: {e}")
        return

    msg = response.choices[0].message
    if not msg.tool_calls:
        print(f"  -> AI (no action): {msg.content.strip() if msg.content else '(none)'}")
        return

    for call in msg.tool_calls:
        name = call.function.name
        try:
            fn_args = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            fn_args = {}
        handler = DISPATCH.get(name)
        if handler is None:
            print(f"  -> AI called unknown tool '{name}', ignoring")
            continue
        result = handler(fn_args)
        print(f"  -> ACTION: {name}({fn_args}) -> {result}")
