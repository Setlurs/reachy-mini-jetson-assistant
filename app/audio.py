"""Audio utilities — PulseAudio management and ALSA device discovery."""

import subprocess
import time
from pathlib import Path
from typing import Optional


def kill_pulseaudio() -> bool:
    """Kill PulseAudio and prevent respawning (ALSA fallback only)."""
    pa_conf = Path.home() / ".config" / "pulse" / "client.conf"
    pa_conf.parent.mkdir(parents=True, exist_ok=True)
    if not pa_conf.exists() or "autospawn = no" not in pa_conf.read_text():
        pa_conf.write_text("autospawn = no\n")
    subprocess.run(["pulseaudio", "--kill"], capture_output=True)
    subprocess.run(["pkill", "-9", "pulseaudio"], capture_output=True)
    time.sleep(0.5)
    return subprocess.run(["pgrep", "-x", "pulseaudio"], capture_output=True).returncode != 0


def find_alsa_device(
    name_hint: str = "USB Audio",
    direction: str = "input",
) -> Optional[tuple[int, int, str]]:
    """Find ALSA device by name substring. Returns (card, device, name) or None."""
    cmd = "arecord" if direction == "input" else "aplay"
    try:
        r = subprocess.run([cmd, "-l"], capture_output=True, text=True, timeout=5)
    except Exception:
        return None
    needle = name_hint.lower()
    for line in r.stdout.splitlines():
        if "card" in line.lower() and needle in line.lower():
            try:
                parts = line.split()
                card = int(parts[parts.index("card") + 1].rstrip(":,"))
                dev = int(parts[parts.index("device") + 1].rstrip(":,"))
                name = line.split("[")[1].split("]")[0] if "[" in line else name_hint
                return card, dev, name
            except (ValueError, IndexError):
                continue
    return None
