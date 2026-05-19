"""set_camera_power — turn the robot's camera on or off.

Off releases the camera device (the capture light goes out) and the
assistant can no longer see; vision questions won't work until it is
turned back on. The camera's background capture thread keeps running
either way, so toggling is fast and the web UI stays connected.
"""

import logging
import re
from typing import Any, Dict, Optional

from app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


_OFF_VERB = (
    r"(turn(ed|ing)?\s+(it\s+|the\s+\w+\s+)?off|shut\s+(it\s+)?off|"
    r"switch(ed|ing)?\s+(it\s+|the\s+\w+\s+)?off|power\s+(it\s+)?off|"
    r"disable|deactivate|mute|cover|kill)"
)
_ON_VERB = (
    r"(turn(ed|ing)?\s+(it\s+|the\s+\w+\s+)?(back\s+)?on|"
    r"switch(ed|ing)?\s+(it\s+|the\s+\w+\s+)?on|power\s+(it\s+)?on|"
    r"enable|reactivate|activate|resume|unmute)"
)


def camera_power_intent(text: str) -> Optional[bool]:
    """Deterministically detect an unambiguous camera on/off command.

    Returns True (turn on), False (turn off), or None (not a clear
    camera-power command — let the LLM handle it). Used as a pre-LLM
    intercept because small models inconsistently emit the tool call
    for "turn off the camera" and just narrate instead.

    Conservative: requires the word camera/webcam AND an explicit power
    verb, or the exact "camera on/off" phrasing. Locational "on camera"
    (e.g. "what do you see on camera") is intentionally NOT matched.
    """
    t = (text or "").lower()
    if not re.search(r"\b(camera|webcam)\b", t):
        return None
    off = re.search(_OFF_VERB, t) is not None
    on = re.search(_ON_VERB, t) is not None
    if off and not on:
        return False
    if on and not off:
        return True
    # Bare "camera off" / "camera on" (not "on camera").
    if re.search(r"\bcamera\s+off\b", t):
        return False
    if re.search(r"\bcamera\s+on\b", t):
        return True
    return None


class SetCameraPower(Tool):
    """Turn the camera on or off."""

    name = "set_camera_power"
    description = (
        "Turn the camera on (on=true) or off (on=false). You MUST call "
        "this tool for BOTH directions — to enable/turn on/resume the "
        "camera AND to disable/turn off/mute/cover it. Turning a camera "
        "back ON only happens by calling this tool with on=true; never "
        "just say the camera is on without calling it. Off releases the "
        "camera (light off) so vision requests fail until on=true. Use "
        "for any enable/disable, mute/unmute, cover, or privacy request."
    )
    example = "Turn the camera back on."
    parameters_schema = {
        "type": "object",
        "properties": {
            "on": {
                "type": "boolean",
                "description": "True to turn the camera on, False to turn it off.",
            },
        },
        "required": ["on"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        cam = deps.camera
        if cam is None:
            return {"error": "no camera available"}
        if not hasattr(cam, "set_enabled"):
            return {"error": "this camera does not support power control"}

        on = bool(kwargs.get("on", True))
        logger.info("Tool call: set_camera_power on=%s", on)
        state = cam.set_enabled(on)

        # Let the web UI reflect the new state (placeholder vs live feed).
        if deps.broadcaster is not None:
            try:
                deps.broadcaster.send({"type": "camera_power", "on": bool(state)})
            except Exception:
                pass

        return {
            "status": "ok",
            "camera_on": bool(state),
            "message": f"Camera turned {'on' if state else 'off'}.",
        }
