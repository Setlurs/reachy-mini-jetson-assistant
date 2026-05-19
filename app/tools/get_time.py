"""get_time — current local date and time.

Small models have no clock; without this they answer "I don't have access
to the current time/date". This returns the host's local time so the LLM
can speak it.
"""

import logging
from datetime import datetime
from typing import Any, Dict

from app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class GetTime(Tool):
    """Get the current local date and time."""

    name = "get_time"
    description = (
        "Get the current local date and time. Use whenever the user asks "
        "what time it is, what day or date it is, the day of the week, or "
        "anything that depends on the current date or time."
    )
    example = "What time is it?"
    parameters_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        now = datetime.now().astimezone()
        tz = now.tzname() or ""
        logger.info("Tool call: get_time -> %s", now.isoformat())
        # 12-hour clock without a leading zero, portable across platforms.
        hour12 = now.strftime("%I").lstrip("0") or "12"
        spoken = now.strftime(f"%A, %B %-d, %Y at {hour12}:%M %p").strip()
        return {
            "status": "ok",
            "iso": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "weekday": now.strftime("%A"),
            "timezone": tz,
            "spoken": f"{spoken} {tz}".strip(),
        }
