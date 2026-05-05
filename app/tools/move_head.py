"""move_head — point the head in one of five canonical directions."""

import asyncio
import logging
from typing import Any, Dict, Tuple

from app.tools.core_tools import Tool, ToolDependencies
from app.movements import _head_pose


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
