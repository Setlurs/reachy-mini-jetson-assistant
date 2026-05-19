"""play_video — play / control a named YouTube clip in the web UI.

A registry maps names ("school", …) to YouTube URLs (from
config.videos.clips). "play school video" cues the matching clip in
the web UI's panel (the same panel the maps use); voice commands can
advance / rewind / pause / resume / stop it.

Both a deterministic transcript intercept (video_command_intent, used
by run_web_vision_chat so it works without the LLM) and an LLM tool
(PlayVideo) drive the same broadcast message.
"""

import logging
import re
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# name -> YouTube URL. Populated at startup from config.videos.clips.
VIDEO_REGISTRY: Dict[str, str] = {}

_SEEK_SECONDS = 10


def register_videos(mapping: Dict[str, str]) -> None:
    VIDEO_REGISTRY.clear()
    for k, v in (mapping or {}).items():
        VIDEO_REGISTRY[k.strip().lower()] = v


def parse_youtube(url: str) -> Tuple[Optional[str], int]:
    """Return (video_id, start_seconds) from a YouTube URL.

    Handles watch?v=, youtu.be/<id>, /embed/<id>, and a t= / start=
    timestamp like "20", "20s", "1m30s".
    """
    if not url:
        return None, 0
    u = urlparse(url)
    vid = None
    if "youtu.be" in (u.netloc or ""):
        vid = u.path.lstrip("/").split("/")[0] or None
    else:
        qs = parse_qs(u.query or "")
        if "v" in qs:
            vid = qs["v"][0]
        elif "/embed/" in (u.path or ""):
            vid = u.path.split("/embed/")[1].split("/")[0]

    start = 0
    qs = parse_qs(u.query or "")
    raw = (qs.get("t") or qs.get("start") or ["0"])[0]
    if raw.isdigit():
        start = int(raw)
    else:
        m = re.findall(r"(\d+)([hms])", raw.lower())
        if m:
            mult = {"h": 3600, "m": 60, "s": 1}
            start = sum(int(n) * mult[u_] for n, u_ in m)
    return vid, start


def video_command_intent(text: str, names) -> Optional[Dict[str, Any]]:
    """Detect a spoken video command. Returns a dict or None.

    {"action": "play", "name": "school"} |
    {"action": "forward"|"back", "seconds": N} |
    {"action": "pause"|"resume"|"stop"}
    """
    t = re.sub(r"[^\w\s]", " ", (text or "").lower()).strip()
    if not t:
        return None

    # play <name> video / play the <name> video / play video <name>
    if "video" in t and re.search(r"\b(play|start|show|open)\b", t):
        for name in names:
            if re.search(rf"\b{re.escape(name.lower())}\b", t):
                return {"action": "play", "name": name}
        # "play the video" with a single registered clip → use it
        if len(names) == 1:
            return {"action": "play", "name": next(iter(names))}

    if re.search(r"\b(advance|forward|skip ahead|fast forward|skip forward)\b", t):
        return {"action": "forward", "seconds": _SEEK_SECONDS}
    if re.search(r"\b(rewind|go back|skip back|back up|skip backward)\b", t):
        return {"action": "back", "seconds": _SEEK_SECONDS}
    if re.search(r"\b(pause|hold) (the )?video\b", t) or t in ("pause", "pause video"):
        return {"action": "pause"}
    if re.search(r"\b(resume|continue|unpause|play) (the )?video\b", t):
        return {"action": "resume"}
    if re.search(r"\b(stop|close|exit|end|dismiss) (the )?video\b", t):
        return {"action": "stop"}
    return None


def resolve_clip(name: str) -> Optional[Dict[str, Any]]:
    url = VIDEO_REGISTRY.get((name or "").strip().lower())
    if not url:
        return None
    vid, start = parse_youtube(url)
    if not vid:
        return None
    return {"name": name, "video_id": vid, "start": start, "url": url}


class PlayVideo(Tool):
    """Play or control a named how-to video in the on-screen panel."""

    name = "play_video"
    description = (
        "Play or control a named instructional video in the on-screen "
        "panel (e.g. 'play school video'). action='play' needs the clip "
        "name; 'forward'/'back' seek, 'pause'/'resume'/'stop' control "
        "playback. Use when the user asks to play, advance, rewind, "
        "pause, or stop a video."
    )
    example = "Play school video."
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["play", "forward", "back", "pause", "resume", "stop"],
                "description": "What to do (default: play).",
            },
            "name": {
                "type": "string",
                "description": "Clip name for action=play, e.g. 'school'.",
            },
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        if deps.broadcaster is None:
            return {"error": "video playback needs the web UI"}
        action = (kwargs.get("action") or "play").lower()
        msg: Dict[str, Any] = {"type": "video", "action": action}

        if action == "play":
            name = kwargs.get("name") or (
                next(iter(VIDEO_REGISTRY)) if len(VIDEO_REGISTRY) == 1 else ""
            )
            clip = resolve_clip(name)
            if clip is None:
                return {"error": f"unknown video {name!r}; "
                                 f"known: {sorted(VIDEO_REGISTRY)}"}
            msg.update(video_id=clip["video_id"], start=clip["start"],
                       name=clip["name"])
            spoken = f"Playing the {clip['name']} video."
        elif action in ("forward", "back"):
            msg["seconds"] = int(kwargs.get("seconds") or _SEEK_SECONDS)
            spoken = f"Skipping {action}."
        else:
            spoken = f"Video {action}."

        deps.broadcaster.send(msg)
        logger.info("Tool call: play_video %s", msg)
        return {"status": "ok", "message": spoken, **{k: msg[k] for k in msg if k != "type"}}
