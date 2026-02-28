"""Pipeline — shared audio I/O, VAD, TTS streaming, and mic recording.

Extracts the common infrastructure used by both run_voice_chat.py and
run_vision_chat.py so each entry point only contains its unique logic.
"""

import sys
import time
import wave
import subprocess
import threading
import queue
from collections import deque
from dataclasses import dataclass
from typing import Optional, Iterator

import numpy as np
from pathlib import Path
from rich.console import Console

from app.audio import kill_pulseaudio

# Suppress noisy ALSA error messages (underrun warnings etc.)
# The callback reference must be kept alive to avoid segfault from GC.
_ALSA_ERR_T = None
_alsa_handler = None
try:
    import ctypes
    _ALSA_ERR_T = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int,
                                    ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)
    _alsa_handler = _ALSA_ERR_T(lambda *_: None)
    ctypes.cdll.LoadLibrary('libasound.so.2').snd_lib_error_set_handler(_alsa_handler)
except Exception:
    pass


# ── Audio constants ───────────────────────────────────────────────

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_MS = 30
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)
CHUNK_BYTES = CHUNK_SAMPLES * CHANNELS * 2

SPEECH_THRESH = 0.008
SILENCE_DUR_MS = 500
LOOKBACK_MS = 250
MAX_SPEECH_SECS = 15

SILENCE_CHUNKS = int(SILENCE_DUR_MS / CHUNK_MS)
LOOKBACK_CHUNKS = int(LOOKBACK_MS / CHUNK_MS)
MAX_CHUNKS = int(MAX_SPEECH_SECS * 1000 / CHUNK_MS)

TTS_BREAKS = frozenset('.,;:!?\n')
FIRST_CHUNK_WORDS = 3
MAX_CHUNK_WORDS = 8


# ── Audio helpers ─────────────────────────────────────────────────

def chunk_rms(raw: bytes) -> float:
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(pcm ** 2)))


def save_wav(chunks: list[bytes], path: str):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(chunks))


def warmup_stt(stt_obj) -> float:
    """Run a dummy transcription to warm up CUDA. Returns elapsed seconds."""
    path = "/tmp/_warmup.wav"
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(np.zeros(SAMPLE_RATE // 2, dtype=np.int16).tobytes())
    t0 = time.perf_counter()
    stt_obj.transcribe(path, sample_rate=SAMPLE_RATE)
    Path(path).unlink(missing_ok=True)
    return time.perf_counter() - t0


def _pa_match(needle: str, haystack: str) -> bool:
    """Match a name hint against a PulseAudio device name, ignoring space/underscore differences."""
    n = needle.lower().replace(" ", "_")
    h = haystack.lower().replace(" ", "_")
    return n in h


def find_pa_source(name_hint: str) -> Optional[str]:
    """Find a PulseAudio input source matching name_hint."""
    try:
        r = subprocess.run(["pactl", "list", "short", "sources"],
                           capture_output=True, text=True, timeout=5)
        for line in r.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) >= 2 and _pa_match(name_hint, parts[1]) and "monitor" not in parts[1].lower():
                return parts[1]
    except Exception:
        pass
    return None


def find_pa_sink(name_hint: str) -> Optional[str]:
    """Find a PulseAudio output sink matching name_hint."""
    try:
        r = subprocess.run(["pactl", "list", "short", "sinks"],
                           capture_output=True, text=True, timeout=5)
        for line in r.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) >= 2 and _pa_match(name_hint, parts[1]):
                return parts[1]
    except Exception:
        pass
    return None


def play_audio(audio: np.ndarray, sample_rate: int, sink: Optional[str] = None):
    """Play int16 audio via paplay (PulseAudio) or aplay fallback."""
    raw = audio.astype(np.int16).tobytes()
    try:
        if sink:
            cmd = ["paplay", f"--device={sink}", "--format=s16le",
                   f"--rate={sample_rate}", "--channels=1", "--raw"]
        else:
            cmd = ["aplay", "-f", "S16_LE", "-r", str(sample_rate),
                   "-c", "1", "-t", "raw", "-q"]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        p.stdin.write(raw)
        p.stdin.close()
        p.wait(timeout=30)
    except Exception:
        pass


def tts_player(tts_obj, tts_q: queue.Queue, sink: Optional[str] = None):
    """Background thread target: synthesize + play each sentence as it arrives."""
    while True:
        text = tts_q.get()
        if text is None:
            return
        r = tts_obj.synthesize(text)
        if r.get("audio") is not None:
            play_audio(r["audio"], r["sample_rate"], sink=sink)


# ── Speech segment ────────────────────────────────────────────────

@dataclass
class SpeechSegment:
    """A completed speech utterance from the VAD."""
    audio: np.ndarray
    raw_chunks: list
    duration: float
    rms: float
    start_time: float
    end_time: float


# ── Mic recorder ──────────────────────────────────────────────────

class MicRecorder:
    """Manages mic recording via parecord/arecord with a background reader thread."""

    def __init__(self, console: Console):
        self.console = console
        self.audio_q: queue.Queue[bytes] = queue.Queue()
        self.listening = threading.Event()
        self.listening.set()
        self.alive = True
        self._proc: Optional[subprocess.Popen] = None
        self.pa_source: Optional[str] = None
        self.pa_sink: Optional[str] = None

    def start(self, hw: str, mic_hint: str) -> bool:
        """Start recording. Returns True on success."""
        subprocess.run(["pkill", "-9", "parecord"], capture_output=True)
        subprocess.run(["pkill", "-9", "arecord"], capture_output=True)
        time.sleep(0.3)

        self.pa_source = find_pa_source(mic_hint)
        self.pa_sink = find_pa_sink(mic_hint)

        if self.pa_source:
            self.console.print(f"  PA source: {self.pa_source.split('.')[-2]}")
            rec_cmd = ["parecord", "-d", self.pa_source, "--format=s16le",
                       f"--rate={SAMPLE_RATE}", f"--channels={CHANNELS}", "--raw"]
        else:
            self.console.print("  [yellow]PA source not found, using ALSA direct[/yellow]")
            kill_pulseaudio()
            time.sleep(0.5)
            plughw = hw.replace("hw:", "plughw:")
            rec_cmd = ["arecord", "-D", plughw, "-f", "S16_LE", "-r", str(SAMPLE_RATE),
                       "-c", str(CHANNELS), "-t", "raw"]

        for attempt in range(3):
            self._proc = subprocess.Popen(rec_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(0.5)
            if self._proc.poll() is None:
                break
            err = self._proc.stderr.read().decode(errors="replace").strip()
            self.console.print(f"  [red]Mic attempt {attempt+1} failed: {err}[/red]")
            time.sleep(1)

        if self._proc is None or self._proc.poll() is not None:
            return False

        threading.Thread(target=self._reader, daemon=True).start()

        time.sleep(0.5)
        test_chunks = []
        for _ in range(10):
            try:
                test_chunks.append(self.audio_q.get(timeout=0.5))
            except queue.Empty:
                break
        if test_chunks:
            r = chunk_rms(b"".join(test_chunks))
            if r > 0.003:
                self.console.print("  Mic: [green]✓ live[/green]")
            else:
                self.console.print("  Mic: [red]✗ silent — unmute![/red]")
        else:
            self.console.print(
                f"  [red]Mic: no audio data! arecord running: {self._proc.poll() is None}[/red]"
            )

        return True

    def _reader(self):
        while self.alive:
            raw = self._proc.stdout.read(CHUNK_BYTES)
            if not raw:
                if self._proc.poll() is not None:
                    err = self._proc.stderr.read().decode(errors="replace").strip()
                    if err:
                        self.console.print(f"\n  [red]arecord died: {err}[/red]")
                break
            if self.listening.is_set():
                self.audio_q.put(raw)

    def flush(self):
        while not self.audio_q.empty():
            try:
                self.audio_q.get_nowait()
            except queue.Empty:
                break

    def pause(self):
        """Stop queuing audio and drain the buffer."""
        self.listening.clear()
        self.flush()

    def resume(self):
        """Drain any stale audio and resume queuing."""
        self.flush()
        self.listening.set()

    def stop(self):
        self.alive = False
        if self._proc:
            self._proc.terminate()
            self._proc.wait(timeout=2)


# ── VAD loop ──────────────────────────────────────────────────────

def vad_loop(mic: MicRecorder, console: Console) -> Iterator[SpeechSegment]:
    """Yields SpeechSegment each time a complete utterance is detected.

    The caller is responsible for calling mic.resume() after processing
    each segment (so audio stays paused during STT/LLM/TTS).
    """
    lookback: deque[bytes] = deque(maxlen=LOOKBACK_CHUNKS)
    speech_raw: list[bytes] = []
    is_speaking = False
    silence_count = 0
    speech_start_t: float = 0.0

    while mic.alive:
        try:
            raw = mic.audio_q.get(timeout=0.1)
        except queue.Empty:
            continue

        rms = chunk_rms(raw)

        if rms > SPEECH_THRESH:
            silence_count = 0
            if not is_speaking:
                is_speaking = True
                speech_start_t = time.monotonic()
                speech_raw = list(lookback)
                sys.stdout.write("  🎤 Listening...\r")
                sys.stdout.flush()
            speech_raw.append(raw)
            if len(speech_raw) < MAX_CHUNKS:
                continue
        else:
            if is_speaking:
                speech_raw.append(raw)
                silence_count += 1
                if silence_count < SILENCE_CHUNKS:
                    continue
            else:
                lookback.append(raw)
                continue

        is_speaking = False
        captured = speech_raw
        speech_raw = []
        silence_count = 0
        lookback.clear()
        speech_end_t = time.monotonic()

        dur_s = len(captured) * CHUNK_MS / 1000
        cap_rms = chunk_rms(b"".join(captured))

        sys.stdout.write("                              \r")
        sys.stdout.flush()
        mic.pause()

        if dur_s < 0.3 or cap_rms < 0.005:
            console.print(f"[dim]  (noise: {dur_s:.1f}s, rms={cap_rms:.4f})[/dim]")
            mic.resume()
            continue

        raw_audio = b"".join(captured)
        audio_np = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0

        yield SpeechSegment(
            audio=audio_np,
            raw_chunks=captured,
            duration=dur_s,
            rms=cap_rms,
            start_time=speech_start_t,
            end_time=speech_end_t,
        )


# ── LLM streaming with TTS ───────────────────────────────────────

def stream_and_speak(
    llm,
    tts_obj,
    prompt: str,
    system_prompt: str,
    pa_sink: Optional[str] = None,
    images_b64: Optional[list[str]] = None,
) -> tuple[str, float, Optional[float]]:
    """Stream LLM response while chunking text to TTS for real-time playback.

    Returns (full_response, elapsed_seconds, time_to_first_token).
    """
    tts_q = None
    tts_thread = None
    if tts_obj:
        tts_q = queue.Queue()
        tts_thread = threading.Thread(
            target=tts_player, args=(tts_obj, tts_q, pa_sink), daemon=True,
        )
        tts_thread.start()

    full_resp = ""
    tts_buf = ""
    first_tts_sent = False
    t_llm = time.perf_counter()
    ttft = None

    for chunk_data in llm.generate_stream(
        prompt=prompt, system_prompt=system_prompt,
        images_b64=images_b64,
    ):
        content, meta = chunk_data if isinstance(chunk_data, tuple) else (chunk_data, {})
        if content:
            if ttft is None:
                ttft = time.perf_counter() - t_llm
            sys.stdout.write(content)
            sys.stdout.flush()
            full_resp += content

            if tts_q is not None:
                tts_buf += content
                words = len(tts_buf.split())
                limit = FIRST_CHUNK_WORDS if not first_tts_sent else MAX_CHUNK_WORDS
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

    return full_resp, dt_llm, ttft
