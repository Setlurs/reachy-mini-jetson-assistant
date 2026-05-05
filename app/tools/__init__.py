"""Tool-calling registry for the Jetson assistant.

Adapted from reachy_mini_conversation_app's profile-driven design, simplified
to a static set of imports. Each tool subclasses Tool and is auto-registered
via __init_subclass__ side effect when its module is imported below.
"""

from app.tools.core_tools import (
    Tool,
    ToolDependencies,
    ALL_TOOLS,
    dispatch_tool_call,
    get_tool_specs,
    get_tools_info,
    initialize_tools,
)

# Importing each module triggers its Tool subclass registration.
from app.tools import (  # noqa: F401
    do_nothing,
    move_head,
    play_emotion,
    stop_emotion,
    analyze_image,
    web_search,
    air_quality,
    water_data,
    wildfire_activity,
)

initialize_tools()


__all__ = [
    "Tool",
    "ToolDependencies",
    "ALL_TOOLS",
    "dispatch_tool_call",
    "get_tool_specs",
    "get_tools_info",
]
