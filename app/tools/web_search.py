"""web_search — DuckDuckGo lookup for fresh information.

Ported from reachy_mini_conversation_app/tools/web_search.py.
"""

import asyncio
import logging
from typing import Any, Dict

from app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class WebSearch(Tool):
    """Search the web for current information using DuckDuckGo."""

    name = "web_search"
    description = (
        "Search the web for current, real-time information. Use when the user "
        "asks about recent events, news, sports scores, weather, or anything "
        "that requires up-to-date information you do not already know."
    )
    example = "What's the latest news about NASA Artemis?"
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query.",
            },
        },
        "required": ["query"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        query = (kwargs.get("query") or "").strip()
        if not query:
            return {"error": "no query provided"}

        try:
            from ddgs import DDGS  # noqa: WPS433
        except ImportError:
            return {"error": "ddgs not installed; pip install ddgs"}

        def _search():
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=5))

        try:
            results = await asyncio.to_thread(_search)
        except Exception as e:
            logger.warning("web_search failed: %s", e)
            return {"error": f"search failed: {e}"}

        if not results:
            return {"status": "no_results", "query": query, "summary": "No results."}

        snippets = [
            f"- {r.get('title', '')}: {r.get('body', '')}" for r in results
        ]
        logger.info("Tool call: web_search query=%s n=%d", query, len(results))
        return {"status": "ok", "query": query, "results": "\n".join(snippets)}
