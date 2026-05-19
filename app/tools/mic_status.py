"""mic_status — report or change the microphone mute state.

"Muted" means the pipeline ignores everything you say (web UI shows
Muted, the PTT button is the toggle). The single source of truth is
the web Broadcaster's push-to-talk flag: ptt_active == not muted.

This tool lets the LLM answer "are you muted?" and mute/unmute on
request. A deterministic spoken-command matcher (mic_command_intent)
also runs in run_web_vision_chat — including while muted — so
"hey reachy unmute" / "hey reachy mute" always works without relying
on the model.
"""

import logging
import re
from typing import Any, Dict, Optional

from app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


def mic_status_query(text: str) -> bool:
    """True if `text` is asking about the mic / mute / listening state.

    Lets the muted branch answer "are you muted right now?" without
    unmuting — purely deterministic, no LLM and no wake word needed.
    """
    t = (text or "").lower()
    return bool(
        re.search(r"\bare you (muted|listening|on mute|there|awake)\b", t)
        or re.search(r"\b(is|are) (the )?(mic|microphone) (on|off|muted|working|live)\b", t)
        or re.search(r"\bare you (still )?listening\b", t)
        or re.search(r"\bmuted right now\b", t)
        or re.search(r"\bcan you hear me\b", t)
        or re.search(r"\b(what'?s|what is) your (mic|microphone|mute) (status|state)\b", t)
    )


def _wake_tokens(wake_model: str) -> list:
    """Spoken form of a wake-model id, as STT would transcribe it.

    "alexa" -> ["alexa"]; "hey_jarvis" -> ["hey jarvis", "jarvis"];
    "okay_nabu" -> ["okay nabu", "nabu"]. Keep the bare last word so a
    clipped transcription ("...jarvis mute") still matches.
    """
    spoken = wake_model.replace("_", " ").strip().lower()
    parts = spoken.split()
    toks = [spoken]
    if len(parts) > 1:
        toks.append(parts[-1])
    return toks


def mic_command_intent(text: str, wake_model: str = "alexa") -> Optional[bool]:
    """Detect a spoken mic mute/unmute command.

    Returns True (unmute), False (mute), or None (not a command).
    Requires the configured wake word (so ordinary conversation that
    happens to contain "mute" never toggles the mic) OR a bare command
    ("mute" / "unmute" on its own — clearly addressed to the device).
    """
    t = re.sub(r"[^\w\s]", " ", (text or "").lower()).strip()
    if not t:
        return None

    is_unmute = re.search(r"\b(unmute|wake up|start listening|resume listening)\b", t) is not None
    is_mute = (not is_unmute) and re.search(
        r"\b(mute|stop listening|go to sleep|be quiet)\b", t
    ) is not None
    if not (is_unmute or is_mute):
        return None

    has_wake = any(tok in t for tok in _wake_tokens(wake_model))
    if not has_wake and len(t.split()) > 3:
        # No wake word and too long to be a bare "mute"/"unmute" command.
        return None
    return True if is_unmute else False


class MicStatus(Tool):
    """Report or change whether the microphone is muted."""

    name = "mic_status"
    description = (
        "Report or change the microphone mute state. action='status' "
        "tells whether the mic is currently muted or listening; "
        "action='mute' stops the assistant from listening; "
        "action='unmute' resumes listening. Use when the user asks "
        "whether you are muted/listening, or asks you to mute or unmute."
    )
    example = "Are you muted right now?"
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["status", "mute", "unmute"],
                "description": "What to do (default: status).",
            },
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        bc = deps.broadcaster
        if bc is None:
            return {"error": "mute state unavailable (no web broadcaster)"}

        action = (kwargs.get("action") or "status").lower()
        if action == "mute":
            bc.set_ptt(False)
            bc.send({"type": "status", "stage": "muted"})
        elif action == "unmute":
            bc.set_ptt(True)
            bc.send({"type": "status", "stage": "listening"})

        muted = not bc.ptt_active
        logger.info("Tool call: mic_status action=%s -> muted=%s", action, muted)
        return {
            "status": "ok",
            "action": action,
            "muted": muted,
            "message": (
                "The microphone is muted." if muted
                else "The microphone is on and listening."
            ),
        }
