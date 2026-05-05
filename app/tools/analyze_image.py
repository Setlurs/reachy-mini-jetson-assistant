"""analyze_image — answer a question about the latest camera frame.

The pipeline already keeps a continuous ring buffer of frames; this tool
pulls the most recent one and runs a one-shot multimodal call against the
same LLM to produce a short description for the outer model to weave into
its spoken reply.
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional

from app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


_DEFAULT_PROMPT = (
    "Describe what you see in one short sentence. Be direct."
)
_VISION_SYSTEM_PROMPT = (
    "You are a vision assistant. Look at the attached image and answer the "
    "user's question in one short sentence. No markdown or formatting."
)


def _grab_frame(camera, retries: int = 4, delay_s: float = 0.15) -> Optional[str]:
    """Pull a JPEG-b64 frame from camera with a short retry budget.

    Tries the ring buffer (capture_single) first because it's cheapest, then
    a fresh hardware read (read_live) so we don't fail when the ring is
    momentarily empty. A few short retries cover startup races where neither
    surface has populated yet.
    """
    for attempt in range(retries):
        try:
            b64 = camera.capture_single()
        except Exception:
            b64 = None
        if b64:
            return b64
        try:
            b64 = camera.read_live()
        except Exception:
            b64 = None
        if b64:
            return b64
        time.sleep(delay_s)
    return None


class AnalyzeImage(Tool):
    """Look at the latest camera frame and answer a question about it."""

    name = "analyze_image"
    description = (
        "Look at the live camera frame from the robot and answer a question "
        "about what is visible. Use this when the user asks what you see, "
        "asks about colors, objects, people, or the scene around you."
    )
    example = "What do you see in front of you?"
    parameters_schema = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": (
                    "What to look for or describe. Defaults to a brief scene "
                    "description if omitted."
                ),
            },
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        if deps.camera is None:
            return {"error": "camera unavailable"}
        if deps.llm is None:
            return {"error": "vision LLM unavailable"}

        question = (kwargs.get("question") or "").strip() or _DEFAULT_PROMPT

        # Try the ring buffer first, fall back to a fresh hardware read,
        # with a couple of brief retries to absorb pipeline warmup races.
        frame_b64 = await asyncio.to_thread(_grab_frame, deps.camera)
        if not frame_b64:
            return {"error": "no camera frame available"}

        def _ask() -> str:
            chunks = []
            for content, _meta in deps.llm.generate_stream(
                prompt=question,
                system_prompt=_VISION_SYSTEM_PROMPT,
                images_b64=[frame_b64],
            ):
                if content:
                    chunks.append(content)
            return "".join(chunks).strip()

        try:
            answer = await asyncio.to_thread(_ask)
        except Exception as e:
            logger.exception("analyze_image LLM call failed")
            return {"error": f"vision call failed: {e}"}

        if not answer:
            return {"error": "vision model returned no text"}

        logger.info("Tool call: analyze_image question=%s answer=%s", question, answer[:120])
        return {"status": "ok", "question": question, "description": answer}
