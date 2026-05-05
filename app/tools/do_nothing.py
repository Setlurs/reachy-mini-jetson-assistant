"""do_nothing — explicit no-op tool the LLM can pick when it should stay still."""

import logging
from typing import Any, Dict

from app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class DoNothing(Tool):
    """Choose to do nothing — stay still and silent."""

    name = "do_nothing"
    description = (
        "Choose to do nothing — stay still and silent. Use when you want to "
        "be contemplative, save energy, or just chill."
    )
    example = "Just stay still for a moment."
    parameters_schema = {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Optional reason for doing nothing.",
            },
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        reason = kwargs.get("reason", "just chilling")
        logger.info("Tool call: do_nothing reason=%s", reason)
        return {"status": "doing nothing", "reason": reason}
