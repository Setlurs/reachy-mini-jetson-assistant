"""Tool ABC, dependency container, and dispatch.

Mirrors reachy_mini_conversation_app/tools/core_tools.py but skips the
profile-driven dynamic import — modules are imported explicitly in
app/tools/__init__.py and concrete subclasses self-register via
__init_subclass__.
"""

from __future__ import annotations

import abc
import inspect
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)

ALL_TOOLS: Dict[str, "Tool"] = {}
ALL_TOOL_SPECS: List[Dict[str, Any]] = []
_REGISTERED_CLASSES: List[type] = []
_INITIALIZED = False


@dataclass
class ToolDependencies:
    """External dependencies injected into tools."""

    reachy: Any = None              # ReachyMini handle (or None when robot absent)
    movement_controller: Any = None  # app.movements.MovementController
    camera: Any = None               # app.camera.Camera or RobotCamera
    llm: Any = None                  # app.llm.LLM (for analyze_image)
    broadcaster: Any = None          # app.web.Broadcaster (optional, for UI events)
    antenna_rest: Optional[List[float]] = None
    motion_duration_s: float = 1.0


class Tool(abc.ABC):
    """Base abstraction for tools used in function-calling.

    Each tool must define name, description, and parameters_schema (JSON
    Schema for the function's arguments).
    """

    name: str
    description: str
    parameters_schema: Dict[str, Any]
    example: str = ""  # Sample user utterance that should invoke this tool.

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not inspect.isabstract(cls):
            _REGISTERED_CLASSES.append(cls)

    def spec(self) -> Dict[str, Any]:
        """OpenAI function-calling spec."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }

    @abc.abstractmethod
    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Async tool execution entrypoint."""
        raise NotImplementedError


def initialize_tools() -> None:
    """Instantiate every registered Tool subclass exactly once."""
    global _INITIALIZED
    if _INITIALIZED:
        return
    for cls in _REGISTERED_CLASSES:
        try:
            tool = cls()  # type: ignore[abstract]
        except Exception as e:
            logger.warning("Failed to instantiate tool %s: %s", cls.__name__, e)
            continue
        ALL_TOOLS[tool.name] = tool
        ALL_TOOL_SPECS.append(tool.spec())
        logger.info("tool registered: %s — %s", tool.name, tool.description)
    _INITIALIZED = True


def get_tool_specs(enabled: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Return OpenAI tool specs, optionally filtered to a name allowlist."""
    if enabled is None:
        return list(ALL_TOOL_SPECS)
    allow = set(enabled)
    return [s for s in ALL_TOOL_SPECS if s.get("function", {}).get("name") in allow]


def get_tools_info(enabled: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """Plain (name, description, example) summary for UI display."""
    if enabled is None:
        items = list(ALL_TOOLS.values())
    else:
        allow = set(enabled)
        items = [t for t in ALL_TOOLS.values() if t.name in allow]
    return [
        {
            "name": t.name,
            "description": t.description,
            "example": getattr(t, "example", "") or "",
        }
        for t in items
    ]


def _safe_load_args(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        logger.warning("bad tool args=%r", raw)
        return {}


async def dispatch_tool_call(
    tool_name: str, args: Any, deps: ToolDependencies
) -> Dict[str, Any]:
    """Run a tool by name with JSON-or-dict args. Always returns a dict."""
    tool = ALL_TOOLS.get(tool_name)
    if tool is None:
        return {"error": f"unknown tool: {tool_name}"}
    parsed = _safe_load_args(args)
    try:
        return await tool(deps, **parsed)
    except Exception as e:
        logger.exception("Tool error in %s", tool_name)
        return {"error": f"{type(e).__name__}: {e}"}
