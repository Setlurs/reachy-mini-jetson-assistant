"""set_camera_power — turn the robot's camera on or off.

Off releases the camera device (the capture light goes out) and the
assistant can no longer see; vision questions won't work until it is
turned back on. The camera's background capture thread keeps running
either way, so toggling is fast and the web UI stays connected.
"""

import logging
from typing import Any, Dict

from app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class SetCameraPower(Tool):
    """Turn the camera on or off."""

    name = "set_camera_power"
    description = (
        "Turn the camera on or off. Turning it off releases the camera "
        "(its light turns off) and you can no longer see — vision/'what "
        "do you see' requests will not work until it is turned back on. "
        "Use when the user asks to disable/enable, mute/unmute, or cover "
        "the camera, or asks for privacy."
    )
    example = "Turn off the camera."
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
