# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reachy Mini connection helpers.

Shared across run_vision_chat.py, run_web_vision_chat.py, and any future
entry point that needs robot control.

Two connection modes:

* Wired (Reachy Mini Lite): USB-tethered, daemon spawned locally,
  media_backend="no_media". The app reads the local USB camera/audio.
* Wireless (Reachy Mini Wireless CM4): no USB tether possible. The daemon
  runs on the robot itself; we connect over the network and route camera/
  mic/speaker through robot.media — WebRTC by default, or GStreamer when
  the app happens to run on the robot's own CM4 (--on-device).
"""

import os
import signal
import subprocess
import time
from typing import Optional, Tuple

from rich.console import Console

from app.platform_utils import is_linux

try:
    from reachy_mini import ReachyMini
    import psutil
    HAS_REACHY = True
except ImportError:
    HAS_REACHY = False
    ReachyMini = None  # type: ignore[assignment,misc]
    psutil = None


def is_daemon_running() -> bool:
    """Check if a reachy-mini-daemon process exists."""
    if not psutil:
        return False
    for proc in psutil.process_iter(["cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            for part in cmdline:
                if "reachy-mini-daemon" in part:
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
            continue
    return False


def kill_daemon(console: Console) -> bool:
    """Kill a stale reachy-mini-daemon process. Returns True if one was found."""
    if not psutil:
        return False
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            for part in cmdline:
                if "reachy-mini-daemon" in part:
                    pid = proc.pid
                    console.print(f"  [yellow]Killing stale Reachy daemon (PID {pid})[/yellow]")
                    os.kill(pid, signal.SIGKILL)
                    time.sleep(2)
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
            continue
    return False


def kill_stale_camera_holders(device: int, console: Console) -> None:
    """Kill any process holding /dev/video<device> (except ourselves).

    Linux-only: uses `fuser`, which doesn't exist on macOS. Also irrelevant
    in wireless mode — there is no local /dev/video* in that case.
    """
    if not is_linux():
        return
    try:
        r = subprocess.run(
            ["fuser", f"/dev/video{device}"],
            capture_output=True, text=True, timeout=5,
        )
        pids = r.stdout.strip().split()
        my_pid = str(os.getpid())
        for pid in pids:
            pid = pid.strip().rstrip("m")
            if pid and pid != my_pid:
                console.print(f"  [yellow]Killing stale process {pid} holding /dev/video{device}[/yellow]")
                os.kill(int(pid), signal.SIGKILL)
        if pids:
            time.sleep(0.5)
    except Exception:
        pass


def _resolve_connection_mode(rcfg) -> Tuple[str, str, bool]:
    """Translate (wireless, on_device, media_backend) into ReachyMini kwargs.

    Returns (media_backend, connection_mode, spawn_daemon) where
    connection_mode is one of "auto" / "localhost_only" / "network" — the
    SDK 1.7+ replacement for the deprecated `localhost_only` bool.
    """
    wireless = bool(getattr(rcfg, "wireless", False))
    on_device = bool(getattr(rcfg, "on_device", False))

    if not wireless:
        # Wired USB (Reachy Mini Lite) — keep existing behavior.
        return rcfg.media_backend, "localhost_only", rcfg.spawn_daemon

    if on_device:
        # Wireless robot, daemon co-located with the app (e.g. running on
        # the CM4 itself). GStreamer talks to the local pipeline directly.
        return "gstreamer", "localhost_only", rcfg.spawn_daemon

    # Fully remote — daemon runs on the robot, we talk over the network.
    return "webrtc", "network", False


def is_wireless(config) -> bool:
    """Used by entry points to pick local vs robot-media camera + mic."""
    return bool(getattr(config.reachy, "wireless", False))


def apply_cli_overrides(config, args) -> None:
    """Apply --wireless / --on-device CLI flags onto config.reachy.

    Both flags are tri-state via argparse (default None = keep config). This
    keeps the YAML defaults usable while CLI flags can flip them at runtime.
    """
    if getattr(args, "wireless", None) is not None:
        config.reachy.wireless = bool(args.wireless)
    if getattr(args, "on_device", None) is not None:
        config.reachy.on_device = bool(args.on_device)


def add_connection_args(parser) -> None:
    """Add --wireless / --on-device flags to an argparse parser.

    --wireless / --no-wireless and --on-device / --no-on-device let users
    override config.reachy.wireless / on_device per-invocation. Default is
    None so we can tell "user didn't pass it" from "user said False".
    """
    g = parser.add_argument_group("Reachy Mini connection")
    w = g.add_mutually_exclusive_group()
    w.add_argument("--wireless", dest="wireless", action="store_true", default=None,
                   help="Use Reachy Mini Wireless (CM4) — media via WebRTC/GStreamer")
    w.add_argument("--no-wireless", dest="wireless", action="store_false",
                   help="Force wired USB mode (Reachy Mini Lite)")
    o = g.add_mutually_exclusive_group()
    o.add_argument("--on-device", dest="on_device", action="store_true", default=None,
                   help="App is running on the robot itself — use GStreamer instead of WebRTC")
    o.add_argument("--off-device", dest="on_device", action="store_false",
                   help="App is on a separate host — use WebRTC (default for --wireless)")


def build_camera(config, console, robot):
    """Build the wired or wireless camera depending on config.reachy.wireless.

    Lives in app.reachy because choosing media routing is a connection-mode
    decision; placing it here also avoids a circular import between
    app.camera and app.pipeline. Returns the (already constructed, not
    started) camera, or None if camera support is unavailable.
    """
    from app.camera import Camera, RobotCamera

    if is_wireless(config):
        if robot is None:
            console.print("[red]Wireless mode requires a connected robot for camera access.[/red]")
            return None
        return RobotCamera(
            robot=robot,
            width=config.vision.width,
            height=config.vision.height,
            jpeg_quality=config.vision.jpeg_quality,
            capture_fps=config.vision.capture_fps,
        )

    return Camera(
        device=config.vision.camera_device,
        width=config.vision.width,
        height=config.vision.height,
        jpeg_quality=config.vision.jpeg_quality,
        capture_fps=config.vision.capture_fps,
    )


def build_mic(config, console, robot, chunk_ms: int):
    """Build the wired or wireless mic recorder.

    Returns the (already constructed, not started) recorder, or None on
    failure. The wired path also needs the ALSA hw string and mic hint, so
    we surface those in the return tuple alongside the recorder.

    Returns: (recorder, hw, mic_hint) where hw and mic_hint are None for
    the wireless path (unused).
    """
    from app.audio import find_alsa_device
    from app.pipeline import MicRecorder, RobotMicRecorder

    if is_wireless(config):
        if robot is None:
            console.print("[red]Wireless mode requires a connected robot for mic access.[/red]")
            return None, None, None
        return RobotMicRecorder(console, robot=robot, chunk_ms=chunk_ms), None, None

    hint = config.audio.input_device or "Reachy Mini Audio"
    result = find_alsa_device(name_hint=hint)
    if not result:
        console.print("[red]No mic found![/red]")
        return None, None, None
    card, dev, mic_name = result
    hw = f"hw:{card},{dev}"
    console.print(f"  Mic: {hw} ({mic_name})")
    return MicRecorder(console, chunk_ms=chunk_ms), hw, hint


def connect(config, console: Console) -> Optional["ReachyMini"]:
    """Connect to Reachy Mini using config.reachy settings.

    Handles daemon discovery, retries, wake-up, antenna positioning, and
    wired/wireless backend selection.
    Returns a connected ReachyMini instance, or None if unavailable.
    """
    if not HAS_REACHY or not config.reachy.enabled:
        return None

    rcfg = config.reachy
    media_backend, connection_mode, spawn_daemon = _resolve_connection_mode(rcfg)
    daemon_already_running = is_daemon_running()

    if rcfg.wireless and rcfg.on_device:
        mode_label = "wireless / gstreamer (on-device)"
    elif rcfg.wireless:
        mode_label = "wireless / webrtc (remote daemon)"
    else:
        mode_label = "wired USB"

    for attempt in range(rcfg.daemon_retry_attempts):
        try:
            if attempt == 0:
                console.print(f"  Connecting to Reachy Mini [{mode_label}]...")
            elif attempt == 1:
                console.print(f"  [dim]Daemon may still be starting, waiting {rcfg.daemon_startup_wait:.0f}s...[/dim]")
                time.sleep(rcfg.daemon_startup_wait)
                console.print("  Retrying connection to Reachy Mini...")
            else:
                if spawn_daemon:
                    kill_daemon(console)
                console.print("  Retrying connection (fresh daemon)...")

            reachy = ReachyMini(
                spawn_daemon=spawn_daemon,
                use_sim=False,
                timeout=rcfg.timeout,
                media_backend=media_backend,
                connection_mode=connection_mode,
            )

            reachy.enable_motors()
            if rcfg.wake_on_start:
                if daemon_already_running:
                    console.print("  Ensuring Reachy Mini is awake...")
                else:
                    console.print("  Waking up Reachy Mini...")
                reachy.wake_up()
                time.sleep(0.5)
                try:
                    reachy.set_target_antenna_joint_positions(rcfg.antenna_rest_position)
                    time.sleep(0.2)
                except Exception:
                    pass
                console.print("  [green]✓ Reachy Mini awake (head up, camera ready)[/green]")
            else:
                console.print("  [green]✓ Reachy Mini connected (wake_on_start=false)[/green]")
            return reachy

        except Exception as e:
            err_msg = str(e)
            if ("localhost and network" in err_msg or "both localhost" in err_msg.lower()) and attempt < rcfg.daemon_retry_attempts - 1:
                continue
            console.print(f"  [yellow]⚠ Reachy Mini unavailable: {e}[/yellow]")
            console.print("  [yellow]  Continuing without robot control[/yellow]")
            return None
    return None
