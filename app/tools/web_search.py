"""web_search — DuckDuckGo lookup for fresh information.

Keyless Google (scraping) was tried but is reliably blocked/empty, so
DuckDuckGo is the only engine. For Google-quality results a paid
Programmable Search / SerpAPI key would be required.

Ported from reachy_mini_conversation_app/tools/web_search.py.
"""

import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict

from app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


def _resolve_relative_dates(query: str) -> str:
    """Rewrite relative time words to explicit dates.

    The model has no clock, so "who's winning the NBA game tonight" or
    "what happened yesterday" reach the search engine undated. We resolve
    those words against the host's local date and inline the explicit
    date so results are pinned to the right day, e.g.:

        "the NBA game tonight"  -> "the NBA game tonight (May 18, 2026)"
        "yesterday's results"   -> "yesterday (May 17, 2026)'s results"

    The original word is kept so the phrasing still reads naturally.
    """
    today = datetime.now().astimezone()

    def fmt(d: datetime) -> str:
        return d.strftime("%B %d, %Y").replace(" 0", " ")

    # word (case-insensitive, whole-word) -> date offset in days
    offsets = {
        "today": 0, "tonight": 0, "now": 0, "currently": 0,
        "this morning": 0, "this afternoon": 0, "this evening": 0,
        "yesterday": -1, "tomorrow": 1,
    }
    out = query
    for word, delta in offsets.items():
        date_str = fmt(today + timedelta(days=delta))
        out = re.sub(
            rf"\b{re.escape(word)}\b",
            lambda m: f"{m.group(0)} ({date_str})",
            out,
            flags=re.IGNORECASE,
        )
    # If nothing relative was mentioned but the ask is clearly time-
    # sensitive, still anchor it to today.
    if out == query and re.search(
        r"\b(latest|recent|score|scores|news|today'?s|happening|right now)\b",
        query, re.IGNORECASE,
    ):
        out = f"{query} (as of {fmt(today)})"
    return out


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

        original_query = query
        query = _resolve_relative_dates(query)
        if query != original_query:
            logger.info("web_search date-resolved: %r -> %r", original_query, query)

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
            f"- {r.get('title', '')}: {r.get('body', '')}"
            for r in results
            if r.get("title") or r.get("body")
        ]
        logger.info("Tool call: web_search query=%s n=%d", query, len(snippets))
        return {"status": "ok", "query": query, "results": "\n".join(snippets)}
