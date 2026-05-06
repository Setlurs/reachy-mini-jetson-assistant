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
from app.config import VADConfig
from app.platform_utils import is_linux

# Suppress noisy ALSA error messages (underrun warnings etc.)
# The callback reference must be kept alive to avoid segfault from GC.
# Linux-only — libasound doesn't exist on macOS, and the wireless path
# never touches ALSA anyway, so this whole block can be skipped off-Linux.
_ALSA_ERR_T = None
_alsa_handler = None
if is_linux():
    try:
        import ctypes
        _ALSA_ERR_T = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int,
                                        ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)
        _alsa_handler = _ALSA_ERR_T(lambda *_: None)
        ctypes.cdll.LoadLibrary('libasound.so.2').snd_lib_error_set_handler(_alsa_handler)
    except Exception:
        pass


# ── Audio constants (fixed by hardware, not user-tunable) ────────

SAMPLE_RATE = 16000
SILERO_CHUNK_SAMPLES = 512  # Silero VAD requires exactly 512 samples (32ms) at 16kHz
CHANNELS = 1

TTS_BREAKS = frozenset('.,;:!?\n')


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


def play_audio(audio: np.ndarray, sample_rate: int, sink=None):
    """Play int16 audio.

    `sink` is the routing hint:
      * None or PulseAudio sink name (str) → wired path: paplay/aplay.
      * A ReachyMini robot instance → wireless path: push float32 samples
        to the robot's speakers via robot.media.push_audio_sample().

    The two paths are chosen by isinstance check rather than a flag so
    callers stay generic (they just hand off whatever the MicRecorder
    exposes as pa_sink).
    """
    # Wireless path — push to the robot speaker over WebRTC/GStreamer.
    #
    # The reachy_mini_conversation_app reference pushes audio in many
    # small chunks because Kokoro's stream_tts_sync yields ~10ms slices
    # as synthesis progresses. Our tts.py uses kokoro.create() which
    # returns the whole utterance at once, so a naive single push of
    # ~3 s of audio overflows the WebRTC pipeline buffer and the tail
    # gets dropped. Slicing into ~0.5 s pieces and pushing them
    # sequentially lands us in the same regime as the reference: each
    # push is small enough that the buffer accepts it cleanly.
    if sink is not None and not isinstance(sink, str):
        robot = sink
        try:
            target_sr = robot.media.get_output_audio_samplerate() or sample_rate
        except Exception:
            target_sr = sample_rate
        try:
            float_pcm = audio.astype(np.float32) / 32768.0
            if target_sr and target_sr > 0 and target_sr != sample_rate and float_pcm.size > 1:
                # Linear resample. Quality is fine for TTS; matches the
                # reference repo's scipy.signal.resample length-based call.
                n_in = float_pcm.shape[0]
                n_out = max(1, int(round(n_in * target_sr / sample_rate)))
                xp = np.linspace(0.0, 1.0, n_in, endpoint=False, dtype=np.float32)
                xq = np.linspace(0.0, 1.0, n_out, endpoint=False, dtype=np.float32)
                float_pcm = np.interp(xq, xp, float_pcm).astype(np.float32)

            slice_seconds = 0.5
            slice_samples = max(1, int((target_sr or sample_rate) * slice_seconds))
            total = float_pcm.shape[0]
            for start in range(0, total, slice_samples):
                end = min(start + slice_samples, total)
                robot.media.push_audio_sample(float_pcm[start:end])
                # Pace at slightly under realtime so the pipeline can
                # drain a slice before we feed the next one. This avoids
                # the buffer overflow that drops the tail of long pushes.
                if end < total:
                    time.sleep((end - start) / float(target_sr or sample_rate) * 0.9)
        except Exception as e:
            print(f"  [audio] robot push failed: {e}", file=sys.stderr)
        return

    # Wired path (Linux/Jetson): paplay (PulseAudio) or aplay fallback.
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


# ── Silero VAD ────────────────────────────────────────────────────

class SileroVAD:
    """Thin wrapper around the Silero VAD ONNX model."""

    def __init__(self):
        from silero_vad import load_silero_vad
        import torch
        self._model = load_silero_vad(onnx=True)
        self._torch = torch

    def __call__(self, raw_audio: bytes) -> float:
        """Return speech probability for raw int16 PCM audio at 16 kHz."""
        pcm = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
        tensor = self._torch.from_numpy(pcm)
        return self._model(tensor, SAMPLE_RATE).item()

    def reset(self):
        self._model.reset_states()


def load_silero(console: Optional[Console] = None) -> Optional[SileroVAD]:
    """Try to load Silero VAD. Returns wrapper or None on failure."""
    try:
        t0 = time.perf_counter()
        vad = SileroVAD()
        dt = time.perf_counter() - t0
        if console:
            console.print(f"  ✓ Silero VAD (ONNX, loaded in {dt:.1f}s)")
        return vad
    except ImportError:
        if console:
            console.print("  [yellow]⚠ silero-vad not installed (pip install silero-vad), using energy VAD[/yellow]")
        return None
    except Exception as e:
        if console:
            console.print(f"  [yellow]⚠ Silero VAD failed to load: {e}, using energy VAD[/yellow]")
        return None


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

    def __init__(self, console: Console, chunk_ms: int = 30):
        self.console = console
        self.chunk_ms = chunk_ms
        self.chunk_samples = int(SAMPLE_RATE * chunk_ms / 1000)
        self.chunk_bytes = self.chunk_samples * CHANNELS * 2
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
            raw = self._proc.stdout.read(self.chunk_bytes)
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


# ── Robot-media mic recorder (wireless mode) ──────────────────────

class RobotMicRecorder:
    """Mic recorder that pulls audio from the Reachy Mini SDK's media backend.

    Used in wireless mode where there is no local USB audio device — the
    robot's microphones are the only ones available, exposed by the SDK
    as float32 samples over WebRTC (or GStreamer when on-device).

    Exposes the same surface as MicRecorder (audio_q, pause/resume/stop,
    listening, pa_sink, alive) so vad_loop and stream_and_speak don't
    need to know which backend is in use. pa_sink is set to the robot
    instance so play_audio() routes TTS back through robot.media.
    """

    def __init__(self, console: Console, robot, chunk_ms: int = 32):
        self.console = console
        self.robot = robot
        self.chunk_ms = chunk_ms
        self.chunk_samples = int(SAMPLE_RATE * chunk_ms / 1000)
        self.chunk_bytes = self.chunk_samples * CHANNELS * 2
        self.audio_q: queue.Queue[bytes] = queue.Queue()
        self.listening = threading.Event()
        self.listening.set()
        self.alive = True
        self._thread: Optional[threading.Thread] = None
        self._input_sr: int = SAMPLE_RATE
        # play_audio() inspects this — non-string, non-None means "push to
        # the robot speaker via SDK".
        self.pa_source: Optional[str] = None
        self.pa_sink = robot

    def start(self, *_args, **_kwargs) -> bool:
        """Start the robot's audio pipelines and the reader thread.

        Accepts ignored positional/keyword args for surface compatibility
        with MicRecorder.start(hw, mic_hint), so the entry-point factory
        can call rec.start(hw, hint) generically.
        """
        try:
            self.robot.media.start_recording()
            try:
                self.robot.media.start_playing()
            except Exception:
                pass
            self._input_sr = self.robot.media.get_input_audio_samplerate() or SAMPLE_RATE
        except Exception as e:
            self.console.print(f"  [red]Robot mic start failed: {e}[/red]")
            return False

        self.console.print(
            f"  Mic: robot.media (input_sr={self._input_sr}Hz → {SAMPLE_RATE}Hz)"
        )

        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

        # Quick liveness check — wait briefly for first frames.
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
                self.console.print("  Mic: [yellow]quiet — speak into the robot[/yellow]")
        else:
            self.console.print("  [yellow]Mic: no frames yet (WebRTC may still be negotiating)[/yellow]")
        return True

    def _reader(self):
        """Pull float32 samples from robot.media, mix to mono, resample to
        16 kHz, and emit fixed-size int16 PCM chunks into audio_q."""
        leftover = np.zeros(0, dtype=np.float32)
        target_per_chunk = self.chunk_samples
        in_sr = self._input_sr or SAMPLE_RATE
        ratio = SAMPLE_RATE / float(in_sr) if in_sr > 0 else 1.0

        while self.alive:
            try:
                frame = self.robot.media.get_audio_sample()
            except Exception:
                time.sleep(0.01)
                continue
            if frame is None:
                time.sleep(0.005)
                continue

            arr = np.asarray(frame, dtype=np.float32)
            # Mix down to mono if multi-channel.
            if arr.ndim == 2:
                # sounddevice convention: (samples, channels)
                if arr.shape[1] < arr.shape[0]:
                    arr = arr.mean(axis=1)
                else:
                    arr = arr.mean(axis=0)
            elif arr.ndim != 1:
                arr = arr.flatten()

            # Resample to 16 kHz with linear interpolation if needed.
            if abs(ratio - 1.0) > 1e-3 and arr.size > 1:
                n_out = max(1, int(round(arr.size * ratio)))
                xp = np.linspace(0.0, 1.0, arr.size, endpoint=False, dtype=np.float32)
                xq = np.linspace(0.0, 1.0, n_out, endpoint=False, dtype=np.float32)
                arr = np.interp(xq, xp, arr).astype(np.float32)

            leftover = np.concatenate([leftover, arr])

            while leftover.size >= target_per_chunk:
                chunk = leftover[:target_per_chunk]
                leftover = leftover[target_per_chunk:]
                if not self.listening.is_set():
                    continue
                pcm = (np.clip(chunk, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
                self.audio_q.put(pcm)

    def flush(self):
        while not self.audio_q.empty():
            try:
                self.audio_q.get_nowait()
            except queue.Empty:
                break

    def pause(self):
        self.listening.clear()
        self.flush()

    def resume(self):
        self.flush()
        self.listening.set()

    def stop(self):
        self.alive = False
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        try:
            self.robot.media.stop_recording()
        except Exception:
            pass
        try:
            self.robot.media.stop_playing()
        except Exception:
            pass


# ── VAD loop ──────────────────────────────────────────────────────

def vad_loop(
    mic: MicRecorder,
    console: Console,
    vad_cfg: Optional[VADConfig] = None,
    silero: Optional[SileroVAD] = None,
) -> Iterator[SpeechSegment]:
    """Yields SpeechSegment each time a complete utterance is detected.

    When *silero* is provided, speech detection uses the neural model's
    probability (much better at rejecting non-speech sounds like coughs,
    keyboard clicks, and ambient noise).  RMS is still used as a cheap
    pre-filter to skip dead silence without invoking the model.

    The caller is responsible for calling mic.resume() after processing
    each segment (so audio stays paused during STT/LLM/TTS).
    """
    cfg = vad_cfg or VADConfig()
    chunk_ms = cfg.chunk_ms
    silence_chunks = int(cfg.silence_duration_ms / chunk_ms)
    lookback_chunks = int(cfg.lookback_ms / chunk_ms)
    max_chunks = int(cfg.max_speech_secs * 1000 / chunk_ms)

    use_silero = silero is not None
    silero_thresh = cfg.silero_threshold
    rms_silence_floor = 0.002  # below this, skip Silero inference entirely

    lookback: deque[bytes] = deque(maxlen=lookback_chunks)
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

        if use_silero:
            if rms < rms_silence_floor:
                is_speech = False
            else:
                is_speech = silero(raw) > silero_thresh
        else:
            is_speech = rms > cfg.speech_threshold

        if is_speech:
            silence_count = 0
            if not is_speaking:
                is_speaking = True
                speech_start_t = time.monotonic()
                speech_raw = list(lookback)
                sys.stdout.write("  🎤 Listening...\r")
                sys.stdout.flush()
            speech_raw.append(raw)
            if len(speech_raw) < max_chunks:
                continue
        else:
            if is_speaking:
                speech_raw.append(raw)
                silence_count += 1
                if silence_count < silence_chunks:
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

        if use_silero:
            silero.reset()

        dur_s = len(captured) * chunk_ms / 1000
        cap_rms = chunk_rms(b"".join(captured))

        sys.stdout.write("                              \r")
        sys.stdout.flush()
        mic.pause()

        if dur_s < cfg.min_utterance_secs or cap_rms < cfg.min_utterance_rms:
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
    few_shot: Optional[list[dict]] = None,
    first_chunk_chars: int = 60,
    max_chunk_chars: int = 150,
) -> tuple[str, float, Optional[float]]:
    """Stream LLM response while chunking text to TTS for real-time playback.

    Uses the waterfall StreamingChunker so chunks fall on sentence/clause
    boundaries instead of arbitrary word counts — that's what gives Kokoro
    natural prosody. Each chunk is also cleaned of markdown noise before
    being queued to the TTS subprocess.

    Returns (full_response, elapsed_seconds, time_to_first_token).
    """
    from app.tts import StreamingChunker, clean_text_for_speech

    tts_q = None
    tts_thread = None
    chunker: Optional[StreamingChunker] = None
    if tts_obj:
        tts_q = queue.Queue()
        tts_thread = threading.Thread(
            target=tts_player, args=(tts_obj, tts_q, pa_sink), daemon=True,
        )
        tts_thread.start()
        chunker = StreamingChunker(
            first_chunk_chars=first_chunk_chars,
            max_chunk_chars=max_chunk_chars,
        )

    full_resp = ""
    t_llm = time.perf_counter()
    ttft = None

    def _emit(text: str) -> None:
        cleaned = clean_text_for_speech(text)
        if cleaned and tts_q is not None:
            tts_q.put(cleaned)

    for chunk_data in llm.generate_stream(
        prompt=prompt, system_prompt=system_prompt,
        images_b64=images_b64, few_shot=few_shot,
    ):
        content, meta = chunk_data if isinstance(chunk_data, tuple) else (chunk_data, {})
        if content:
            if ttft is None:
                ttft = time.perf_counter() - t_llm
            sys.stdout.write(content)
            sys.stdout.flush()
            full_resp += content
            if chunker is not None:
                for ready in chunker.feed(content):
                    _emit(ready)

    dt_llm = time.perf_counter() - t_llm

    if tts_q is not None:
        if chunker is not None:
            tail = chunker.flush()
            if tail:
                _emit(tail)
        tts_q.put(None)
        tts_thread.join()

    return full_resp, dt_llm, ttft
