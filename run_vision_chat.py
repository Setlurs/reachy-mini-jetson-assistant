#!/usr/bin/env python3
"""
Vision Chat — speak + see, the VLM describes what it sees.
Mic → energy VAD → [camera capture] → STT → VLM (text + images) → TTS → Speaker

Usage:
  python3 run_vision_chat.py
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.config import Config
from app.audio import find_alsa_device
from app.stt import STT
from app.llm import LLM
from app.tts import create_tts
from app.camera import Camera
from app.pipeline import SAMPLE_RATE, MicRecorder, warmup_stt, vad_loop, stream_and_speak
from rich.console import Console
from rich.panel import Panel

try:
    from reachy_mini import ReachyMini
    HAS_REACHY = True
except ImportError:
    HAS_REACHY = False

console = Console()


def _kill_stale_camera_holders(device: int = 0):
    """Kill any leftover processes holding /dev/videoN (e.g. from Ctrl-Z)."""
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


def main():
    config = Config.load()

    console.print(Panel.fit(
        "[bold cyan]Vision Chat[/bold cyan]\n"
        "Speak anytime — camera captures when you speak\n"
        "[dim]Ctrl-C to quit[/dim]",
        border_style="cyan",
    ))

    # ── Reachy Mini — wake up so head/camera is upright ──────────
    reachy = None
    if HAS_REACHY:
        try:
            console.print("  Connecting to Reachy Mini...")
            reachy = ReachyMini(spawn_daemon=True, use_sim=False, timeout=30.0, media_backend="no_media")
            reachy.enable_motors()
            console.print("  Waking up Reachy Mini...")
            reachy.wake_up()
            time.sleep(0.5)
            console.print("  [green]✓ Reachy Mini awake (head up, camera ready)[/green]")
        except Exception as e:
            console.print(f"  [yellow]⚠ Reachy Mini unavailable: {e}[/yellow]")
            console.print("  [yellow]  Continuing without robot control[/yellow]")
            reachy = None

    # ── Audio setup ───────────────────────────────────────────────
    result = find_alsa_device(name_hint=config.audio.input_device or "Reachy Mini")
    if not result:
        console.print("[red]No mic found![/red]")
        return
    card, dev, mic_name = result
    hw = f"hw:{card},{dev}"
    console.print(f"  Mic: {hw} ({mic_name})")

    # ── Camera setup (background ring buffer) ────────────────────
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

    # ── Register cleanup for all exit signals (including Ctrl-Z) ─
    def _cleanup(signum=None, frame=None):
        console.print("\n[yellow]Cleaning up...[/yellow]")
        cam.close()
        mic.stop()
        if reachy:
            try:
                reachy.goto_sleep()
                reachy.disable_motors()
            except Exception:
                pass
        sys.exit(0)

    signal.signal(signal.SIGTSTP, _cleanup)  # Ctrl-Z
    signal.signal(signal.SIGTERM, _cleanup)  # kill
    signal.signal(signal.SIGHUP, _cleanup)   # terminal closed

    # ── Load models ───────────────────────────────────────────────
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

    vision_system_prompt = config.vision.system_prompt
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

    # ── Start mic ─────────────────────────────────────────────────
    mic = MicRecorder(console)
    if not mic.start(hw, config.audio.input_device or "Reachy Mini"):
        console.print("[red]Cannot start recording! Check mic.[/red]")
        cam.close()
        return

    n_frames = config.vision.frames
    console.print(
        f"\n[green bold]Ready — speak anytime! "
        f"({config.vision.capture_fps} fps, {n_frames} frame{'s' if n_frames > 1 else ''} "
        f"per query, speech-window capture)[/green bold]\n"
    )

    # ── Main loop ─────────────────────────────────────────────────
    try:
        for segment in vad_loop(mic, console):
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
                mic.resume()
                continue

            n_imgs = len(captured_frames)
            console.print(
                f'  [green]You:[/green] "{text}" '
                f'[dim]({n_imgs} frame{"s" if n_imgs != 1 else ""} captured)[/dim]'
            )

            console.print("  [magenta]Assistant:[/magenta] ", end="")
            sys.stdout.flush()

            full_resp, dt_llm, ttft = stream_and_speak(
                llm, tts, text, vision_system_prompt, mic.pa_sink,
                images_b64=captured_frames if captured_frames else None,
            )
            console.print()

            timing = f"  [dim]STT {dt_stt:.1f}s | CAM {dt_cam*1000:.0f}ms ({n_imgs} img from buf)"
            if ttft is not None:
                toks = len(full_resp.split())
                timing += f" | TTFT {ttft:.1f}s | VLM {dt_llm:.1f}s ~{toks/(dt_llm or 1):.0f}w/s"
            else:
                timing += " | VLM no response"
            timing += "[/dim]"
            console.print(timing)

            mic.resume()

    except KeyboardInterrupt:
        console.print("\n[yellow]Goodbye![/yellow]")
    finally:
        mic.stop()
        cam.close()
        stt.unload()
        llm.unload()
        if tts:
            tts.unload()
        if reachy:
            try:
                console.print("  Reachy Mini going to sleep...")
                reachy.goto_sleep()
                reachy.disable_motors()
            except Exception:
                pass


if __name__ == "__main__":
    main()
