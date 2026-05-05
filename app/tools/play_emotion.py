"""play_emotion — trigger one of the built-in emotion sequences."""

import asyncio
import logging
from typing import Any, Dict

from app.emotion import Emotion
from app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


_EMOTION_NAMES = [e.value for e in Emotion if e is not Emotion.NEUTRAL]


class PlayEmotion(Tool):
    """Play a pre-recorded emotion (head + antenna sequence)."""

    name = "play_emotion"
    description = (
        "Play a pre-recorded emotion sequence on the robot. Available emotions: "
        + ", ".join(_EMOTION_NAMES)
        + "."
    )
    example = "Show me you're excited."
    parameters_schema = {
        "type": "object",
        "properties": {
            "emotion": {
                "type": "string",
                "enum": _EMOTION_NAMES,
                "description": "Name of the emotion to play.",
            },
        },
        "required": ["emotion"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        name = (kwargs.get("emotion") or "").strip().lower()
        if name not in _EMOTION_NAMES:
            return {"error": f"unknown emotion: {name!r}. Available: {_EMOTION_NAMES}"}

        if deps.movement_controller is None:
            return {"error": "movement controller unavailable"}

        emotion = Emotion(name)

        def _react() -> bool:
            return deps.movement_controller.react(emotion, confidence=1.0)

        triggered = await asyncio.to_thread(_react)
        logger.info("Tool call: play_emotion emotion=%s triggered=%s", name, triggered)
        if not triggered:
            return {"status": "suppressed", "emotion": name}
        return {"status": "playing", "emotion": name}
