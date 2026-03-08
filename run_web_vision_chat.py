#!/usr/bin/env python3
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
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.config import Config
from app.audio import find_alsa_device
from app.stt import STT
from app.llm import LLM
from app.tts import create_tts
from app.camera import Camera
from app.monitor import get_system_stats, get_jetson_model
from app.pipeline import (
    SAMPLE_RATE, TTS_BREAKS, MicRecorder, warmup_stt, vad_loop,
    tts_player, load_silero,
)
from app.web import Broadcaster, start_web_server
from rich.console import Console
from rich.panel import Panel

try:
    from reachy_mini import ReachyMini
    import psutil
    HAS_REACHY = True
except ImportError:
    HAS_REACHY = False
    psutil = None

console = Console()


# ── Reachy helpers (same as run_vision_chat.py) ──────────────────

def _is_reachy_daemon_running() -> bool:
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


def _kill_reachy_daemon() -> bool:
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


def _kill_stale_camera_holders(device: int = 0):
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


def _connect_reachy(config):
    if not HAS_REACHY or not config.reachy.enabled:
        return None

    rcfg = config.reachy
    daemon_already_running = _is_reachy_daemon_running()

    for attempt in range(rcfg.daemon_retry_attempts):
        try:
            if attempt == 0:
                console.print("  Connecting to Reachy Mini...")
            elif attempt == 1:
                console.print(f"  [dim]Daemon may still be starting, waiting {rcfg.daemon_startup_wait:.0f}s...[/dim]")
                time.sleep(rcfg.daemon_startup_wait)
                console.print("  Retrying connection to Reachy Mini...")
            else:
                _kill_reachy_daemon()
                console.print("  Retrying connection (fresh daemon)...")

            reachy = ReachyMini(
                spawn_daemon=rcfg.spawn_daemon,
                use_sim=False,
                timeout=rcfg.timeout,
                media_backend=rcfg.media_backend,
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
    args = parser.parse_args()

    config = Config.load()
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
    reachy = _connect_reachy(config)

    # ── Audio setup ──────────────────────────────────────────────
    result = find_alsa_device(name_hint=config.audio.input_device or "Reachy Mini Audio")
    if not result:
        console.print("[red]No mic found![/red]")
        return
    card, dev, mic_name = result
    hw = f"hw:{card},{dev}"
    console.print(f"  Mic: {hw} ({mic_name})")

    # ── Camera setup ─────────────────────────────────────────────
    _kill_stale_camera_holders(config.vision.camera_device)

    cam = Camera(
        device=config.vision.camera_device,
        width=config.vision.width,
        height=config.vision.height,
        jpeg_quality=config.vision.jpeg_quality,
        capture_fps=config.vision.capture_fps,
    )
    if cam.start():
        console.print(
            f"  ✓ Camera /dev/video{config.vision.camera_device} "
            f"({config.vision.width}x{config.vision.height}, "
            f"{config.vision.capture_fps} fps ring buffer)"
        )
    else:
        console.print("[red]  ✗ Camera not found! Check USB webcam.[/red]")
        return

    # ── Pre-declare variables for cleanup closure ───────────────
    mic = None
    stt = None
    llm = None
    tts = None
    silero_model = None

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
    llm = LLM(
        model=config.llm.model, base_url=config.llm.base_url,
        backend=config.llm.backend, max_tokens=config.llm.max_tokens,
        temperature=config.llm.temperature, timeout=config.llm.timeout,
        system_prompt=vision_system_prompt,
    )
    llm.load()
    console.print(f"  ✓ VLM ({llm.model})")

    tts = create_tts(
        backend=config.tts.backend, voice=config.tts.voice,
        speed=config.tts.speed, piper_voice=config.tts.piper_voice,
        lang=config.tts.lang,
    )
    tts = tts if tts.load() else None
    if tts:
        console.print(f"  ✓ TTS ({tts.backend_name}, {tts.voice})")
    else:
        console.print("  ⚠ TTS unavailable")

    # ── Start mic ────────────────────────────────────────────────
    effective_chunk_ms = 32 if silero_model else config.vad.chunk_ms
    mic = MicRecorder(console, chunk_ms=effective_chunk_ms)
    if not mic.start(hw, config.audio.input_device or "Reachy Mini Audio"):
        console.print("[red]Cannot start recording! Check mic.[/red]")
        cam.close()
        return

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
    broadcaster.send({
        "type": "info",
        "models": model_info,
        "platform": platform_name,
        "config": config_info,
    })

    n_frames = config.vision.frames
    n_fewshot = len(vision_few_shot) // 2
    first_chunk_words = config.tts.first_chunk_words
    max_chunk_words = config.tts.max_chunk_words

    console.print(
        f"\n[green bold]Ready — speak anytime! "
        f"({config.vision.capture_fps} fps, {n_frames} frame{'s' if n_frames > 1 else ''} "
        f"per query{f', {n_fewshot} few-shot pairs' if n_fewshot else ''})[/green bold]\n"
    )

    if broadcaster.ptt_active:
        broadcaster.send({"type": "status", "stage": "listening"})
    else:
        broadcaster.send({"type": "status", "stage": "muted"})

    # ── Main loop ────────────────────────────────────────────────
    try:
        for segment in vad_loop(mic, console, vad_cfg=config.vad, silero=silero_model):
            if not broadcaster.ptt_active:
                broadcaster.send({"type": "status", "stage": "muted"})
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
                broadcaster.send({"type": "status", "stage": "listening"})
                mic.resume()
                continue

            word_count = len(text.split())
            if word_count <= 2 and "?" not in text:
                console.print(f"[dim]  (skipped filler: \"{text}\")[/dim]")
                broadcaster.send({"type": "status", "stage": "listening"})
                mic.resume()
                continue

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
            })

            # ── VLM streaming with TTS + WebSocket broadcast ─────
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
            tts_buf = ""
            first_tts_sent = False
            t_llm = time.perf_counter()
            ttft = None

            for chunk_data in llm.generate_stream(
                prompt=text, system_prompt=vision_system_prompt,
                images_b64=captured_frames if captured_frames else None,
                few_shot=vision_few_shot if vision_few_shot else None,
            ):
                content, meta = chunk_data if isinstance(chunk_data, tuple) else (chunk_data, {})
                if content:
                    if ttft is None:
                        ttft = time.perf_counter() - t_llm
                        broadcaster.send({"type": "status", "stage": "speaking"})
                    sys.stdout.write(content)
                    sys.stdout.flush()
                    full_resp += content

                    broadcaster.send({"type": "token", "text": content})

                    if tts_q is not None:
                        tts_buf += content
                        words = len(tts_buf.split())
                        limit = first_chunk_words if not first_tts_sent else max_chunk_words
                        hit_break = any(c in content for c in TTS_BREAKS) and words >= 2
                        if hit_break or words >= limit:
                            tts_q.put(tts_buf.strip())
                            tts_buf = ""
                            first_tts_sent = True

            dt_llm = time.perf_counter() - t_llm

            if tts_q is not None:
                if tts_buf.strip():
                    tts_q.put(tts_buf.strip())
                tts_q.put(None)
                tts_thread.join()

            console.print()

            toks = len(full_resp.split())
            timing = f"  [dim]STT {dt_stt:.1f}s | CAM {dt_cam*1000:.0f}ms ({n_imgs} img from buf)"
            if ttft is not None:
                timing += f" | TTFT {ttft:.1f}s | VLM {dt_llm:.1f}s ~{toks/(dt_llm or 1):.0f}w/s"
            else:
                timing += " | VLM no response"
            timing += "[/dim]"
            console.print(timing)

            broadcaster.send({
                "type": "done",
                "ttft": round(ttft, 2) if ttft else None,
                "vlm_time": round(dt_llm, 2),
                "tokens": toks,
            })
            broadcaster.send({"type": "status", "stage": "listening"})

            mic.resume()

    except (KeyboardInterrupt, SystemExit):
        pass
    except Exception:
        pass

    _do_cleanup()
    try:
        if stt:
            stt.unload()
        if llm:
            llm.unload()
        if tts:
            tts.unload()
    except Exception:
        pass
    console.print("[yellow]Goodbye![/yellow]")
    os._exit(0)


if __name__ == "__main__":
    main()
