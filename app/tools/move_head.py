"""move_head — point the head in one of five canonical directions."""

import asyncio
import logging
import re
from typing import Any, Dict, Optional, Tuple

from app.tools.core_tools import Tool, ToolDependencies
from app.movements import _head_pose


# Action verbs and directional tokens for the deterministic intercept.
_ACTION = (
    r"\b(move|look|turn|face|tilt|point|head|gaze|aim|peer|swivel|"
    r"glance|rotate)\b"
)
_DIR_MAP = {
    "left": "left", "right": "right",
    "up": "up", "upward": "up", "upwards": "up", "above": "up",
    "down": "down", "downward": "down", "downwards": "down", "below": "down",
    "front": "front", "forward": "front", "forwards": "front",
    "ahead": "front", "center": "front", "centre": "front",
    "straight": "front", "neutral": "front",
}
_DIR_RE = r"\b(" + "|".join(map(re.escape, _DIR_MAP.keys())) + r")\b"


def move_head_intent(text: str) -> Optional[str]:
    """Detect a head-direction command. Returns left/right/up/down/front
    or None. Conservative: requires both an action verb and a direction
    so ordinary sentences containing "left" don't move the head.
    """
    t = re.sub(r"[^\w\s]", " ", (text or "").lower()).strip()
    if not t:
        return None
    if not re.search(_ACTION, t):
        return None
    m = re.search(_DIR_RE, t)
    if not m:
        return None
    return _DIR_MAP[m.group(1)]


logger = logging.getLogger(__name__)


class MoveHead(Tool):
    """Move the head in a given direction."""

    name = "move_head"
    description = "Move your head: left, right, up, down, or front (center)."
    example = "Look to the left."
    parameters_schema = {
        "type": "object",
        "properties": {
            "direction": {
                "type": "string",
                "enum": ["left", "right", "up", "down", "front"],
            },
        },
        "required": ["direction"],
    }

    # Roll/pitch/yaw deltas for each direction.
    DELTAS: Dict[str, Tuple[float, float, float]] = {
        "left": (0.0, 0.0, 40.0),
        "right": (0.0, 0.0, -40.0),
        "up": (0.0, -25.0, 0.0),
        "down": (0.0, 25.0, 0.0),
        "front": (0.0, 0.0, 0.0),
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        direction = kwargs.get("direction")
        if direction not in self.DELTAS:
            return {"error": f"unknown direction: {direction!r}"}

        if deps.reachy is None:
            return {"error": "robot not connected"}

        roll, pitch, yaw = self.DELTAS[direction]
        pose = _head_pose(roll=roll, pitch=pitch, yaw=yaw)
        duration = float(deps.motion_duration_s or 1.0)

        # Engage the manual-head override BEFORE moving so any in-flight
        # emotion sequence is cancelled+joined — otherwise its trailing
        # goto_target(neutral) overrides our new pose right after we
        # send it (the "fleeting first move" bug).
        mc = getattr(deps, "movement_controller", None)
        if mc is not None and hasattr(mc, "set_manual_head"):
            mc.set_manual_head(direction != "front")

        def _move():
            try:
                deps.reachy.goto_target(pose, duration=duration)
                if deps.antenna_rest is not None:
                    deps.reachy.set_target_antenna_joint_positions(deps.antenna_rest)
            except Exception as e:
                raise RuntimeError(f"goto_target failed: {e}") from e

        try:
            await asyncio.to_thread(_move)
        except Exception as e:
            logger.exception("move_head failed")
            return {"error": str(e)}

        logger.info("Tool call: move_head direction=%s", direction)
        return {"status": f"looking {direction}"}
