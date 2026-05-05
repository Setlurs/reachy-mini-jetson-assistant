"""stop_emotion — cancel any in-flight emotion sequence and reset to neutral."""

import asyncio
import logging
from typing import Any, Dict

from app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class StopEmotion(Tool):
    """Stop the current emotion and return the head to neutral."""

    name = "stop_emotion"
    description = "Stop any current emotion sequence and return the head to neutral."
    example = "Stop and return to neutral."
    parameters_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        if deps.movement_controller is None:
            return {"error": "movement controller unavailable"}

        await asyncio.to_thread(deps.movement_controller.reset)
        logger.info("Tool call: stop_emotion")
        return {"status": "stopped"}
