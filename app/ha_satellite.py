"""Home Assistant ESPHome voice-satellite sidecar.

When config.ha.enabled is true, run_web_vision_chat spawns the
reachy_mini_home_assistant package as a subprocess. The satellite owns
its own ReachyMini SDK instance and its own ESPHome server; HA's
existing ESPHome integration auto-discovers it via mDNS — no token, no
URL plumbing on our side.

The shutdown discipline mirrors app/tts.py: graceful SIGINT first
(__main__.py catches KeyboardInterrupt and calls service.stop()), then
SIGTERM, then SIGKILL, with explicit pipe close so the parent doesn't
leave orphaned subprocess pipes.
"""

from __future__ import annotations

import importlib.util
import os
import signal
import subprocess
import sys
from typing import Optional


def is_satellite_installed() -> bool:
    """True if reachy_mini_home_assistant is importable in the current venv."""
    return importlib.util.find_spec("reachy_mini_home_assistant") is not None


class HASatellite:
    """Manage the lifecycle of the HA voice-satellite sidecar."""

    def __init__(
        self,
        wake_model: str = "okay_nabu",
        log_level: str = "info",
        daemon_url: str = "",
    ):
        self.wake_model = wake_model
        self.log_level = (log_level or "info").lower()
        self.daemon_url = daemon_url
        self._proc: Optional[subprocess.Popen] = None

    def start(self) -> bool:
        """Spawn the satellite. Returns False if the package isn't installed
        or the subprocess fails to launch.
        """
        if self._proc is not None and self._proc.poll() is None:
            return True

        if not is_satellite_installed():
            print(
                "  HA satellite: reachy_mini_home_assistant not installed in venv. "
                "Run `pip install -e Reachy_Mini_For_Home_Assistant/` (or your "
                "satellite source) and restart, or set ha.enabled: false."
            )
            return False

        # Spawn through our motor-stubbing wrapper so the satellite's own
        # MovementManager can't actuate the head/antennas. This project
        # owns motor control end-to-end; the satellite is audio + ESPHome
        # only. CLI args pass through to the satellite via sys.argv.
        cmd = [
            sys.executable, "-m", "app.ha_satellite_wrapper",
            "--wake-model", self.wake_model,
        ]
        if self.log_level == "debug":
            cmd.append("--debug")

        env = os.environ.copy()
        # Subprocess inherits OS_ACTIVITY_MODE / PYTHONWARNINGS from the
        # launcher. We also override REACHY_DAEMON_URL so the satellite's
        # volume / status HTTP calls go to the robot rather than its
        # default 127.0.0.1, which only works in on-device mode.
        if self.daemon_url:
            env["REACHY_DAEMON_URL"] = self.daemon_url

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=None,    # inherit parent stdout for visibility
                stderr=None,    # inherit parent stderr (gets log-noise filtered)
                env=env,
            )
        except Exception as e:
            print(f"  HA satellite: spawn failed: {e}")
            return False

        return True

    def stop(self) -> None:
        """Reap the satellite process. SIGINT → SIGTERM → SIGKILL ladder."""
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        if proc.poll() is not None:
            return  # already exited

        # The satellite's __main__.py catches KeyboardInterrupt and calls
        # service.stop() for clean WebRTC + ESPHome shutdown.
        try:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            pass
        except Exception:
            pass

        try:
            proc.terminate()
            proc.wait(timeout=3)
            return
        except subprocess.TimeoutExpired:
            pass
        except Exception:
            pass

        try:
            proc.kill()
            proc.wait(timeout=2)
        except Exception:
            pass

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None
