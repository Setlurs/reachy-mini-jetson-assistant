#!/usr/bin/env python3
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

"""
Web Vision Chat — browser UI + terminal output simultaneously.
Mic -> Silero/energy VAD -> [camera] -> STT -> VLM -> TTS -> Speaker
               + WebSocket broadcast to connected browsers.

Usage:
  python3 run_web_vision_chat.py                 # default 0.0.0.0:8090
  python3 run_web_vision_chat.py --port 9000
  python3 run_web_vision_chat.py --host 127.0.0.1
"""

import argparse
import os
import queue
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from app.config import Config
from app.stt import STT
from app.llm import LLM
from app.tts import create_tts
from app.monitor import get_system_stats, get_jetson_model
from app.pipeline import (
    SAMPLE_RATE, TTS_BREAKS, warmup_stt, vad_loop,
    tts_player, load_silero,
)
from app.reachy import (
    add_connection_args, apply_cli_overrides, build_camera, build_mic,
    kill_stale_camera_holders, connect as connect_reachy, is_wireless,
    is_local_media,
)
from app.emotion import EmotionDetector
from app.movements import MovementController
from app.web import Broadcaster, start_web_server
from app.tools import (
    ToolDependencies, dispatch_tool_call, get_tool_specs, get_tools_info,
)
from app.tools.camera_power import camera_power_intent
from app.tools.mic_status import mic_command_intent, mic_status_query
from app.ha_satellite import HASatellite, is_satellite_installed
from app.wake_word import try_create_detector as create_wake_word_detector
from rich.console import Console
from rich.panel import Panel

console = Console()


# ── Background threads ───────────────────────────────────────────

def _frame_broadcast_thread(cam: Camera, broadcaster: Broadcaster, fps: float = 10.0):
    """Stream camera frames to browsers at UI fps via direct hardware reads.

    Uses cam.read_live() (bypasses the 3fps VLM ring buffer) so the browser
    gets a smooth video feed without affecting VLM frame selection.
    """
    interval = 1.0 / fps
    while cam.health_check():
        if broadcaster.client_count > 0:
            b64 = cam.read_live()
            if b64:
                broadcaster.send({"type": "frame", "data": b64})
        time.sleep(interval)


def _stats_broadcast_thread(
    broadcaster: Broadcaster,
    models: dict,
    reachy,
    interval: float = 2.0,
):
    """Periodically send system stats + robot status to all WebSocket clients."""
    while True:
        try:
            s = get_system_stats()
            msg = {
                "type": "stats",
                "cpu": round(s.cpu_percent, 1),
                "ram_used": round(s.ram_used_mb / 1024, 1),
                "ram_total": round(s.ram_total_mb / 1024, 1),
                "disk_used": round(s.disk_used_gb, 1),
                "disk_total": round(s.disk_total_gb, 1),
                "models": models,
                "clients": broadcaster.client_count,
            }
            if s.gpu_percent is not None:
                msg["gpu"] = round(s.gpu_percent, 1)
            broadcaster.send(msg)

            broadcaster.send({
                "type": "robot",
                "connected": reachy is not None,
                "motors": True if reachy else False,
                "head": "Up" if reachy else "N/A",
            })
        except Exception:
            pass
        time.sleep(interval)


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Vision Chat with Web UI")
    parser.add_argument("--host", default=None, help="Web server bind address")
    parser.add_argument("--port", type=int, default=None, help="Web server port")
    add_connection_args(parser)
    parser.add_argument("--camera-device", type=int, default=None,
                        help="Camera index (overrides config; macOS built-in is usually 0)")
    parser.add_argument("--list-cameras", action="store_true",
                        help="Probe and list available local camera indices, then exit")
    args = parser.parse_args()

    if args.list_cameras:
        from app.local_media import list_local_cameras
        cams = list_local_cameras()
        if cams:
            for idx, w, h in cams:
                console.print(f"  camera #{idx}: {w}x{h}")
        else:
            console.print("  [yellow]No cameras found[/yellow]")
        return

    config = Config.load()
    apply_cli_overrides(config, args)
    if args.camera_device is not None:
        config.vision.camera_device = args.camera_device
    web_host = args.host or config.web.host
    web_port = args.port or config.web.port
    broadcaster = Broadcaster()

    console.print(Panel.fit(
        "[bold cyan]Web Vision Chat[/bold cyan]\n"
        "Speak anytime — camera captures when you speak\n"
        f"[dim]Web UI: http://{{host}}:{web_port}  |  Ctrl-C to quit[/dim]",
        border_style="cyan",
    ))

    # ── Reachy Mini ──────────────────────────────────────────────
    reachy = connect_reachy(config, console)

    # ── Camera setup ─────────────────────────────────────────────
    if not is_wireless(config):
        kill_stale_camera_holders(config.vision.camera_device, console)

    # Local-media: probe attached cameras once (before opening one, so the
    # probe doesn't fight the live capture). Used for the fallback below
    # and to populate the web UI's camera dropdown.
    local_cams: list = []
    if is_local_media(config):
        from app.local_media import list_local_cameras
        local_cams = list_local_cameras()
        avail = [i for i, _, _ in local_cams]
        if local_cams and config.vision.camera_device not in avail:
            console.print(
                f"  [yellow]camera #{config.vision.camera_device} not available; "
                f"using #{local_cams[0][0]} (switch in the web UI)[/yellow]"
            )
            config.vision.camera_device = local_cams[0][0]

    cam = build_camera(config, console, reachy)
    if cam is None or not cam.start():
        if is_wireless(config):
            console.print("[red]  ✗ Robot camera unavailable (WebRTC media not ready?).[/red]")
        else:
            console.print("[red]  ✗ Camera not found! Check USB webcam.[/red]")
        return
    if is_local_media(config):
        console.print(
            f"  ✓ Camera local webcam #{config.vision.camera_device} "
            f"({config.vision.width}x{config.vision.height}, "
            f"{config.vision.capture_fps} fps ring buffer)"
        )

        def _send_camera_list(current: int) -> None:
            broadcaster.send({
                "type": "camera_list",
                "cameras": [
                    {"index": i, "label": f"Camera {i} ({w}×{h})"}
                    for i, w, h in local_cams
                ],
                "current": current,
            })

        def _switch_camera(index: int) -> bool:
            ok = cam.switch_device(index) if hasattr(cam, "switch_device") else False
            if ok:
                config.vision.camera_device = index
                console.print(f"  [cyan]Camera switched to #{index}[/cyan]")
            _send_camera_list(config.vision.camera_device)
            return ok

        broadcaster.set_camera_switch(_switch_camera)
        _send_camera_list(config.vision.camera_device)
    elif is_wireless(config):
        console.print(
            f"  ✓ Camera robot.media "
            f"({config.vision.capture_fps} fps ring buffer)"
        )
    else:
        console.print(
            f"  ✓ Camera /dev/video{config.vision.camera_device} "
            f"({config.vision.width}x{config.vision.height}, "
            f"{config.vision.capture_fps} fps ring buffer)"
        )

    # ── Pre-declare variables for cleanup closure ───────────────
    mic = None
    stt = None
    llm = None
    tts = None
    silero_model = None
    ha_satellite: Optional[HASatellite] = None

    # ── Cleanup handler ──────────────────────────────────────────
    _cleanup_done = threading.Event()

    def _do_cleanup():
        if _cleanup_done.is_set():
            return
        _cleanup_done.set()
        console.print("\n[yellow]Shutting down...[/yellow]")
        if mic:
            try:
                mic.stop()
            except Exception:
                pass
        cam.close()
        # TTS is a subprocess holding its own stdin/stdout pipes (and onnx
        # runtime resources). Explicitly tear it down so the process is
        # reaped and pipes are closed before the parent exits — without
        # this, multiprocessing's resource tracker reports a leaked
        # semaphore and Python prints "subprocess still running" warnings.
        if tts:
            try:
                tts.unload()
            except Exception:
                pass
        # Reap the HA satellite sidecar before the parent exits so its
        # WebRTC session and ESPHome server are torn down cleanly. Without
        # this the satellite would either be left as a zombie or get
        # SIGHUP'd by the kernel without finishing its own cleanup.
        if ha_satellite is not None:
            try:
                ha_satellite.stop()
            except Exception:
                pass
        if reachy and config.reachy.sleep_on_exit:
            try:
                signal.signal(signal.SIGINT, signal.SIG_IGN)
            except OSError:
                pass
            try:
                console.print("  Putting Reachy Mini to sleep...")
                reachy.goto_sleep()
                time.sleep(0.5)
                reachy.disable_motors()
                time.sleep(0.3)
            except Exception as e:
                console.print(f"  [dim]Sleep failed: {e}[/dim]")

    def _sig_cleanup(signum=None, frame=None):
        _do_cleanup()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _sig_cleanup)
    signal.signal(signal.SIGTSTP, _sig_cleanup)
    signal.signal(signal.SIGTERM, _sig_cleanup)
    signal.signal(signal.SIGHUP, _sig_cleanup)

    # ── Load models ──────────────────────────────────────────────
    console.print("\n[bold]Loading...[/bold]")

    stt = STT(
        model=config.stt.model, device=config.stt.device,
        compute_type=config.stt.compute_type, language=config.stt.language,
        beam_size=config.stt.beam_size,
    )
    stt.load()
    console.print(f"  ✓ STT (faster-whisper, {config.stt.model})")
    console.print("    CUDA warmup...", end=" ")
    console.print(f"done ({warmup_stt(stt):.1f}s)")

    if config.vad.use_silero:
        silero_model = load_silero(console)
    else:
        console.print("  [dim]Silero VAD disabled, using energy-only VAD[/dim]")

    vision_system_prompt = config.vision.system_prompt
    vision_few_shot = config.vision.few_shot or []

    # When tool calling is on we don't auto-attach the camera frame anymore
    # (analyze_image is the way to see). The default vision prompt assumes
    # an image is always present, which leads small models to hallucinate
    # scenes. Prepend a tool-aware preamble — generated from the live
    # registry so adding/removing a tool updates the prompt automatically.
    if config.llm.tools_enabled:
        _tool_lines = [
            f"- {t['name']}: {t['description']}"
            for t in get_tools_info(config.llm.enabled_tools or None)
        ]
        if _tool_lines:
            vision_system_prompt = (
                "You have function-calling tools listed below. No image, web "
                "data, or sensor reading is available to you unless you call "
                "the matching tool yourself — never invent visual details, "
                "weather, or facts you would otherwise need a tool for. Use "
                "your own knowledge only when no tool is appropriate.\n"
                "For anything about current events, news, sports scores, "
                "prices, weather, or the date/time, you MUST call the "
                "matching tool (e.g. web_search, get_time) every time it is "
                "asked, including follow-up questions. Never reply that you "
                "lack access to real-time information, the internet, sports "
                "scores, or the current date/time — call the tool instead. "
                "Do not refuse a request you answered with a tool earlier "
                "just because you see a similar earlier reply.\n"
                "To enable or disable the camera you MUST call "
                "set_camera_power (on=true to turn on, on=false to turn "
                "off) — turning it back on requires the call, so never "
                "just say it is on/off without calling the tool.\n"
                "Tools:\n"
                + "\n".join(_tool_lines)
                + "\n\n"
                + vision_system_prompt
            )
    llm = LLM(
        model=config.llm.model, base_url=config.llm.base_url,
        backend=config.llm.backend, max_tokens=config.llm.max_tokens,
        temperature=config.llm.temperature, timeout=config.llm.timeout,
        system_prompt=vision_system_prompt,
        history_turns=config.llm.history_turns,
    )
    llm.load()
    console.print(f"  ✓ VLM ({llm.model})")

    tts = create_tts(
        backend=config.tts.backend,
        voice=config.tts.voice,
        speed=config.tts.speed,
        lang=config.tts.lang,
        xtts_speaker_wav=config.tts.xtts_speaker_wav,
        xtts_language=config.tts.xtts_language,
        xtts_temperature=config.tts.xtts_temperature,
    )
    if tts.load():
        console.print(f"  ✓ TTS ({tts.backend_name}, {tts.voice})")
    elif config.tts.backend.lower() == "xtts":
        # Graceful fallback: if XTTS fails to load (missing deps, missing
        # speaker WAV, model download failure), drop back to Kokoro so the
        # demo still has a voice instead of going silent.
        console.print("  [yellow]⚠ XTTS unavailable; falling back to Kokoro[/yellow]")
        tts = create_tts(
            backend="kokoro",
            voice=config.tts.voice,
            speed=config.tts.speed,
            lang=config.tts.lang,
        )
        tts = tts if tts.load() else None
        if tts:
            console.print(f"  ✓ TTS ({tts.backend_name}, {tts.voice})")
        else:
            console.print("  ⚠ TTS unavailable")
    else:
        tts = None
        console.print("  ⚠ TTS unavailable")

    emotion_detector = None
    mover = None
    if config.emotion.enabled:
        emotion_detector = EmotionDetector()
        if emotion_detector.load():
            console.print("  ✓ Emotion (distilbert-sst2, CPU)")
            if reachy:
                mover = MovementController(reachy, config.reachy.antenna_rest_position)
                console.print("  ✓ Emotion movements enabled")
        else:
            console.print("  ⚠ Emotion detector unavailable")
            emotion_detector = None

    # ── Start mic (wired or wireless via robot.media) ────────────
    effective_chunk_ms = 32 if silero_model else config.vad.chunk_ms
    mic, hw, hint = build_mic(config, console, reachy, chunk_ms=effective_chunk_ms)
    if mic is None or not mic.start(hw, hint):
        console.print("[red]Cannot start recording! Check mic.[/red]")
        cam.close()
        return

    # ── HA voice-satellite sidecar (opt-in) ──────────────────────
    # When config.ha.enabled, spawn the reachy_mini_home_assistant
    # ESPHome satellite alongside our pipeline. HA auto-discovers via
    # mDNS — no token, no URL config needed. The satellite owns its own
    # ReachyMini SDK instance; in wireless mode the daemon's WebRTC
    # stream supports both subscribers in parallel.
    if config.ha.enabled and is_local_media(config):
        console.print("  [dim]HA satellite skipped (--local-media: no robot daemon)[/dim]")
    elif config.ha.enabled:
        if not is_satellite_installed():
            console.print(
                "  [yellow]⚠ HA satellite enabled but reachy_mini_home_assistant "
                "is not installed. Run `pip install -e Reachy_Mini_For_Home_Assistant/` "
                "into this venv, or set ha.enabled: false.[/yellow]"
            )
        else:
            ha_satellite = HASatellite(
                wake_model=config.ha.wake_model,
                log_level=config.ha.satellite_log_level,
                daemon_url=config.ha.daemon_url,
            )
            if ha_satellite.start():
                console.print(
                    f"  ✓ HA satellite (wake: {config.ha.wake_model}) — "
                    "watch HA Settings → Devices & Services for auto-discovery"
                )
            else:
                ha_satellite = None

    # Wake-word filler-skip detector. When the satellite hears its wake
    # word, our LLM pipeline must not also process the same utterance —
    # otherwise both pipelines respond. We run the same model the
    # satellite uses (default okay_nabu) on each captured segment and
    # drop matching ones before STT/LLM. Detector is None when ha.enabled
    # is off or pymicro_wakeword isn't installed; in either case the
    # main loop falls through to its existing behavior.
    ha_wake_detector = (
        create_wake_word_detector(config.ha.wake_model)
        if config.ha.enabled else None
    )
    if ha_wake_detector is not None:
        console.print(
            f"  ✓ Wake-word filler skip armed ({config.ha.wake_model})"
        )

    # Local wake-word unmute — independent of HA. The tflite model runs
    # on-device (CPU, offline), so this works in --local-media with HA
    # disabled. Reuses the HA detector instance when the models match to
    # avoid loading the same model twice.
    unmute_wake_detector = None
    if config.mic.wake_unmute_enabled:
        # Always build a dedicated detector so we can apply the
        # sensitivity override + debug probabilities (don't share the
        # HA one, whose cutoff must stay strict).
        unmute_wake_detector = create_wake_word_detector(
            config.mic.wake_model,
            probability_cutoff=config.mic.wake_sensitivity,
            debug=True,
        )
        if unmute_wake_detector is not None:
            console.print(
                f"  ✓ Wake-word unmute armed ({config.mic.wake_model}, "
                f"cutoff {config.mic.wake_sensitivity}) — "
                "say it while muted to start listening"
            )
        else:
            console.print(
                "  [yellow]⚠ Wake-word unmute unavailable "
                "(pip install pymicro_wakeword) — use the UI button[/yellow]"
            )

    # ── Start web server + background threads ────────────────────
    web_thread = start_web_server(broadcaster, host=web_host, port=web_port)
    time.sleep(0.5)
    console.print(f"  ✓ Web UI  →  [bold]http://{web_host}:{web_port}[/bold]")

    threading.Thread(
        target=_frame_broadcast_thread,
        args=(cam, broadcaster, config.web.ui_fps),
        daemon=True, name="frame-broadcaster",
    ).start()

    model_info = {
        "stt": f"faster-whisper ({config.stt.model})",
        "vlm": llm.model,
        "tts": f"{tts.backend_name} ({tts.voice})" if tts else "unavailable",
        "vad": "Silero" if silero_model else "Energy",
    }

    threading.Thread(
        target=_stats_broadcast_thread,
        args=(broadcaster, model_info, reachy),
        daemon=True, name="stats-broadcaster",
    ).start()

    platform_name = get_jetson_model()
    config_info = {
        "max_tokens": config.llm.max_tokens,
        "temperature": config.llm.temperature,
        "vision_frames": config.vision.frames,
        "capture_fps": config.vision.capture_fps,
        "ui_fps": config.web.ui_fps,
        "jpeg_quality": config.vision.jpeg_quality,
        "resolution": f"{config.vision.width}x{config.vision.height}",
        "silero_threshold": config.vad.silero_threshold if config.vad.use_silero else None,
        "beam_size": config.stt.beam_size,
    }
    enabled_tools = config.llm.enabled_tools or None
    tools_info_payload = {
        "enabled": bool(config.llm.tools_enabled),
        "tools": get_tools_info(enabled_tools) if config.llm.tools_enabled else [],
    }
    broadcaster.send({
        "type": "info",
        "models": model_info,
        "platform": platform_name,
        "config": config_info,
        "tools": tools_info_payload,
    })

    tool_specs = get_tool_specs(enabled_tools) if config.llm.tools_enabled else []
    tool_deps = ToolDependencies(
        reachy=reachy,
        movement_controller=mover,
        camera=cam,
        llm=llm,
        broadcaster=broadcaster,
        antenna_rest=config.reachy.antenna_rest_position,
    )

    def _emit_tool_call(name: str, args: dict, result: dict):
        broadcaster.send({
            "type": "tool_call",
            "name": name,
            "args": args,
            "result": result,
        })
        sys.stdout.write(f"\n  [tool] {name}({args}) -> {str(result)[:160]}\n")
        sys.stdout.flush()

    async def _tool_dispatcher(name: str, raw_args):
        return await dispatch_tool_call(name, raw_args, tool_deps)

    n_frames = config.vision.frames
    n_fewshot = len(vision_few_shot) // 2
    first_chunk_chars = config.tts.first_chunk_chars
    max_chunk_chars = config.tts.max_chunk_chars

    from app.tts import StreamingChunker, clean_text_for_speech

    def _idle_status() -> str:
        """The resting pipeline stage — reflects mute so the GUI never
        shows 'Waiting for speech…' while actually muted."""
        return "listening" if broadcaster.ptt_active else "muted"

    def _say(msg: str) -> None:
        """Speak a short canned line + mirror it to the web UI."""
        broadcaster.send({"type": "status", "stage": "speaking"})
        broadcaster.send({"type": "token", "text": msg})
        console.print(f"  [magenta]Assistant:[/magenta] {msg}")
        if tts:
            _q: queue.Queue = queue.Queue()
            _t = threading.Thread(
                target=tts_player, args=(tts, _q, mic.pa_sink), daemon=True,
            )
            _t.start()
            _q.put(clean_text_for_speech(msg))
            _q.put(None)
            _t.join()
        broadcaster.send({"type": "done", "ttft": None,
                          "vlm_time": 0, "tokens": len(msg.split())})

    # Only one query (voice or typed) generates at a time — they share
    # the LLM, TTS subprocess, and speaker.
    _process_lock = threading.Lock()

    def _respond(text, captured_frames, dt_stt=0.0, dt_cam=0.0,
                 n_imgs=0, emotion_tag=""):
        """Run the VLM/tool stream for `text`, stream TTS, and broadcast.

        Shared by the voice loop and the web text-query worker so a typed
        question behaves exactly like a spoken one.
        """
        broadcaster.send({"type": "status", "stage": "thinking"})
        console.print("  [magenta]Assistant:[/magenta] ", end="")
        sys.stdout.flush()

        tts_q = None
        tts_thread = None
        if tts:
            tts_q = queue.Queue()
            tts_thread = threading.Thread(
                target=tts_player, args=(tts, tts_q, mic.pa_sink), daemon=True,
            )
            tts_thread.start()

        full_resp = ""
        t_llm = time.perf_counter()
        ttft = None
        chunker = StreamingChunker(
            first_chunk_chars=first_chunk_chars,
            max_chunk_chars=max_chunk_chars,
        ) if tts_q is not None else None

        def _emit_tts(s: str):
            cleaned = clean_text_for_speech(s)
            if cleaned and tts_q is not None:
                tts_q.put(cleaned)

        if config.llm.tools_enabled and tool_specs:
            stream_iter = llm.generate_with_tools(
                prompt=text,
                tools=tool_specs,
                dispatcher=_tool_dispatcher,
                system_prompt=vision_system_prompt,
                few_shot=vision_few_shot if vision_few_shot else None,
                on_tool_call=_emit_tool_call,
            )
        else:
            stream_iter = llm.generate_stream(
                prompt=text, system_prompt=vision_system_prompt,
                images_b64=captured_frames if captured_frames else None,
                few_shot=vision_few_shot if vision_few_shot else None,
            )

        for chunk_data in stream_iter:
            content, meta = chunk_data if isinstance(chunk_data, tuple) else (chunk_data, {})
            if content:
                if ttft is None:
                    ttft = time.perf_counter() - t_llm
                    broadcaster.send({"type": "status", "stage": "speaking"})
                sys.stdout.write(content)
                sys.stdout.flush()
                full_resp += content
                broadcaster.send({"type": "token", "text": content})
                if chunker is not None:
                    for ready in chunker.feed(content):
                        _emit_tts(ready)

        dt_llm = time.perf_counter() - t_llm

        if tts_q is not None:
            if chunker is not None:
                tail = chunker.flush()
                if tail:
                    _emit_tts(tail)
            tts_q.put(None)
            tts_thread.join()

        console.print()
        llm.add_turn(text, full_resp)

        toks = len(full_resp.split())
        timing = f"  [dim]STT {dt_stt:.1f}s | CAM {dt_cam*1000:.0f}ms ({n_imgs} img from buf)"
        if ttft is not None:
            timing += f" | TTFT {ttft:.1f}s | VLM {dt_llm:.1f}s ~{toks/(dt_llm or 1):.0f}w/s"
        else:
            timing += " | VLM no response"
        timing += emotion_tag
        timing += "[/dim]"
        console.print(timing)

        broadcaster.send({
            "type": "done",
            "ttft": round(ttft, 2) if ttft else None,
            "vlm_time": round(dt_llm, 2),
            "tokens": toks,
        })
        broadcaster.send({"type": "status", "stage": _idle_status()})

    def _text_query_worker():
        """Drain typed queries from the web UI and answer them like speech."""
        while True:
            q = broadcaster.poll_text_query(timeout=0.5)
            if not q:
                continue
            frames = []
            try:
                f = cam.capture_single() if cam else None
                if f:
                    frames = [f]
            except Exception:
                pass
            console.print(f'  [green]You (typed):[/green] "{q}"')
            broadcaster.send({"type": "transcript", "text": q,
                              "stt_time": 0, "duration": 0, "emotion": None})
            with _process_lock:
                _respond(q, frames)

    threading.Thread(target=_text_query_worker, daemon=True,
                      name="text-query-worker").start()

    console.print(
        f"\n[green bold]Ready — speak anytime! "
        f"({config.vision.capture_fps} fps, {n_frames} frame{'s' if n_frames > 1 else ''} "
        f"per query{f', {n_fewshot} few-shot pairs' if n_fewshot else ''})[/green bold]\n"
    )

    if broadcaster.ptt_active:
        broadcaster.send({"type": "status", "stage": "listening"})
    else:
        broadcaster.send({"type": "status", "stage": "muted"})

    # HA exchange grace window: when the satellite's wake word fires, open
    # a window during which all VAD segments are dropped. The user's
    # follow-up question ("what's the temperature?") usually arrives as a
    # separate segment after the wake word and would otherwise leak into
    # our LLM pipeline. The window auto-extends on each dropped segment
    # so multi-turn HA exchanges stay protected, then closes naturally
    # once the user goes quiet.
    HA_GRACE_INITIAL_SECS = 15.0
    HA_GRACE_EXTEND_SECS = 8.0
    ha_grace_until = 0.0

    # ── Main loop ────────────────────────────────────────────────
    try:
        for segment in vad_loop(mic, console, vad_cfg=config.vad, silero=silero_model):
            if not broadcaster.ptt_active:
                # Muted: run the on-device wake-word model on the raw
                # audio; the wake word unmutes, everything else is
                # dropped. DEBUG: also STT the utterance and log level +
                # detector decision so we can see exactly what the muted
                # branch receives and why it did/didn't unmute.
                _raw = b"".join(segment.raw_chunks)
                _heard = ""
                try:
                    _dbg = stt.transcribe(segment.audio, sample_rate=SAMPLE_RATE)
                    _heard = (_dbg.get("text") or "").strip()
                except Exception as _e:
                    _heard = f"<stt err: {_e}>"
                _fired = bool(
                    unmute_wake_detector is not None
                    and unmute_wake_detector.contains(_raw)
                )
                _peak = getattr(unmute_wake_detector, "last_peak", 0.0)
                console.print(
                    f"[dim]  (muted: heard \"{_heard}\" | {segment.duration:.1f}s "
                    f"rms={segment.rms:.4f} bytes={len(_raw)} | "
                    f"wake[{config.mic.wake_model}]={_fired} "
                    f"peak={_peak:.3f}/{config.mic.wake_sensitivity})[/dim]"
                )
                _wake_say = config.mic.wake_model.replace("_", " ").title()
                if _fired:
                    broadcaster.set_ptt(True)
                    _say(
                        f"Microphone on. Say '{_wake_say} stop listening' "
                        f"to mute."
                    )
                    broadcaster.send({"type": "status", "stage": _idle_status()})
                elif mic_status_query(_heard):
                    # Answer a status question while staying muted.
                    _say(
                        f"I am muted, so I am only listening for the wake "
                        f"word. Say '{_wake_say} start listening' to unmute."
                    )
                    broadcaster.send({"type": "status", "stage": "muted"})
                else:
                    broadcaster.send({"type": "status", "stage": "muted"})
                mic.resume()
                continue

            now = time.monotonic()

            # If we're inside an HA exchange window, drop the segment
            # without running STT/LLM and extend the window so any further
            # follow-up turns stay covered.
            if now < ha_grace_until:
                console.print(
                    f"[dim]  (HA exchange in progress — dropping "
                    f"{segment.duration:.1f}s)[/dim]"
                )
                ha_grace_until = now + HA_GRACE_EXTEND_SECS
                broadcaster.send({"type": "status", "stage": _idle_status()})
                mic.resume()
                continue

            # Detect the wake word in this segment. If present, open the
            # grace window — the question itself usually comes in a
            # separate segment a moment later.
            if ha_wake_detector is not None:
                raw_audio = b"".join(segment.raw_chunks)
                if ha_wake_detector.contains(raw_audio):
                    ha_grace_until = now + HA_GRACE_INITIAL_SECS
                    console.print(
                        f"[dim]  (wake word — handing off to HA, "
                        f"{segment.duration:.1f}s; LLM muted ~{HA_GRACE_INITIAL_SECS:.0f}s)[/dim]"
                    )
                    broadcaster.send({"type": "status", "stage": _idle_status()})
                    mic.resume()
                    continue

            broadcaster.send({"type": "status", "stage": "transcribing"})

            t_cam = time.perf_counter()
            captured_frames = cam.get_speech_frames(
                speech_start=segment.start_time,
                speech_end=segment.end_time,
                max_frames=n_frames,
            )
            dt_cam = time.perf_counter() - t_cam

            t_stt = time.perf_counter()
            result = stt.transcribe(segment.audio, sample_rate=SAMPLE_RATE)
            text = result.get("text", "").strip()
            dt_stt = time.perf_counter() - t_stt

            if not text:
                err = result.get("error", "")
                console.print(
                    f"[dim]  (not recognized — {segment.duration:.1f}s, "
                    f"rms={segment.rms:.4f}{', err='+err if err else ''})[/dim]"
                )
                broadcaster.send({"type": "status", "stage": _idle_status()})
                mic.resume()
                continue

            # Filler threshold: in chat-only mode, drop very short non-questions
            # (coughs, partial words, "uh"). With tools enabled, two-word
            # imperatives like "Look left" or "Be happy" are valid commands —
            # only filter true single-word grunts.
            filler_max = 1 if config.llm.tools_enabled else 2
            word_count = len(text.split())
            if word_count <= filler_max and "?" not in text:
                console.print(f"[dim]  (skipped filler: \"{text}\")[/dim]")
                broadcaster.send({"type": "status", "stage": _idle_status()})
                mic.resume()
                continue

            emotion_tag = ""
            if emotion_detector:
                emo = emotion_detector.detect(text)
                moved = mover.react(emo.emotion, emo.confidence) if mover else False
                emotion_tag = (
                    f" | {emo.emotion.value} ({emo.confidence:.0%}, {emo.inference_ms:.0f}ms)"
                    f"{'*' if moved else ''}"
                )

            n_imgs = len(captured_frames)
            console.print(
                f'  [green]You:[/green] "{text}" '
                f'[dim]({n_imgs} frame{"s" if n_imgs != 1 else ""} captured)[/dim]'
            )
            broadcaster.send({
                "type": "transcript",
                "text": text,
                "stt_time": round(dt_stt, 2),
                "duration": round(segment.duration, 1),
                "emotion": emo.emotion.value if emotion_detector else None,
            })

            # ── Deterministic mic mute intercept ─────────────────
            # "<wake word> mute" while listening, e.g. "Alexa mute".
            # (Unmute while muted is handled by the wake-word detector
            # in the muted branch above.)
            _mic_cmd = mic_command_intent(text, wake_model=config.mic.wake_model)
            console.print(
                f"[dim]  (mic-intent: \"{text}\" wake={config.mic.wake_model} "
                f"-> {_mic_cmd!r})[/dim]"
            )
            if _mic_cmd is False:
                broadcaster.set_ptt(False)
                console.print(f'  [green]You:[/green] "{text}"')
                # Announce the wake word so demo passers-by know how to
                # bring it back without touching the UI.
                _wake_say = config.mic.wake_model.replace("_", " ").title()
                _muted_msg = (
                    f"Microphone muted. Say '{_wake_say} start listening' "
                    f"to unmute."
                )
                _say(_muted_msg)
                llm.add_turn(text, _muted_msg)
                broadcaster.send({"type": "status", "stage": "muted"})
                mic.resume()
                continue
            if _mic_cmd is True:
                _say("The microphone is already on.")
                broadcaster.send({"type": "status", "stage": _idle_status()})
                mic.resume()
                continue

            # ── Deterministic camera on/off intercept ────────────
            # Small models inconsistently emit set_camera_power for
            # "turn off the camera" (they narrate instead). For
            # unambiguous power commands, call the tool directly and
            # skip the LLM entirely so it always works.
            cam_intent = (
                camera_power_intent(text)
                if config.llm.tools_enabled else None
            )
            if cam_intent is not None:
                import asyncio
                _loop = asyncio.new_event_loop()
                try:
                    result = _loop.run_until_complete(
                        _tool_dispatcher("set_camera_power", {"on": cam_intent})
                    )
                finally:
                    _loop.close()
                _emit_tool_call("set_camera_power", {"on": cam_intent}, result)
                msg = result.get("message") or (
                    "Camera turned on." if cam_intent else "Camera turned off."
                )
                broadcaster.send({"type": "status", "stage": "speaking"})
                broadcaster.send({"type": "token", "text": msg})
                console.print(f"  [magenta]Assistant:[/magenta] {msg}")
                if tts:
                    q = queue.Queue()
                    th = threading.Thread(
                        target=tts_player, args=(tts, q, mic.pa_sink), daemon=True,
                    )
                    th.start()
                    q.put(clean_text_for_speech(msg))
                    q.put(None)
                    th.join()
                llm.add_turn(text, msg)
                broadcaster.send({"type": "done", "ttft": None,
                                  "vlm_time": 0, "tokens": len(msg.split())})
                broadcaster.send({"type": "status", "stage": _idle_status()})
                mic.resume()
                continue

            # ── VLM streaming with TTS + WebSocket broadcast ─────
            # Shared with the typed-query worker; the lock keeps a voice
            # turn and a typed turn from running the LLM/TTS at once.
            with _process_lock:
                _respond(text, captured_frames, dt_stt, dt_cam,
                         n_imgs, emotion_tag)

            mic.resume()

    except (KeyboardInterrupt, SystemExit):
        pass
    except Exception:
        pass

    _do_cleanup()
    if mover:
        mover.reset()
    try:
        if stt:
            stt.unload()
        if llm:
            llm.unload()
        if tts:
            tts.unload()
        if emotion_detector:
            emotion_detector.unload()
    except Exception:
        pass
    console.print("[yellow]Goodbye![/yellow]")
    os._exit(0)


if __name__ == "__main__":
    main()
