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

"""Host-local media backend — laptop camera / mic / speakers.

Selected by the --local-media CLI flag. This bypasses the Reachy Mini
robot and its GStreamer/WebRTC media proxy entirely so the voice and
vision apps can run on a plain laptop (e.g. a MacBook) for development.

Three classes mirror the surfaces the rest of the app already expects:

* ``LocalMicRecorder``  — same surface as ``MicRecorder`` /
  ``RobotMicRecorder`` (audio_q, listening, pause/resume/stop, pa_sink).
  Uses ``sounddevice`` instead of arecord/parecord subprocesses.
* ``LocalCamera``       — a ``Camera`` subclass that opens the default
  OS webcam directly (AVFoundation on macOS, V4L2 on Linux) and skips
  the Reachy SDK camera probe.
* ``LocalSpeaker``      — the ``pa_sink`` sentinel; ``play_audio()`` in
  app.pipeline routes here via the duck-typed ``play_local`` hook.
"""

import queue
import threading
import time
from typing import Optional

import numpy as np
import sounddevice as sd
from rich.console import Console

from app.camera import Camera
from app.pipeline import CHANNELS, SAMPLE_RATE, chunk_rms


def list_local_cameras(max_index: int = 8) -> list[tuple[int, int, int]]:
    """Probe camera indices 0..max_index-1 with OpenCV's default backend.

    Returns [(index, width, height), ...] for the ones that open and
    deliver a frame. OpenCV on macOS (AVFoundation) can't select a
    camera by name, so index probing is the only way to tell the
    built-in FaceTime camera apart from virtual cams like OBS — which
    typically take a higher index than the built-in (index 0).
    """
    import cv2

    found: list[tuple[int, int, int]] = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        try:
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    h, w = frame.shape[:2]
                    found.append((idx, w, h))
        finally:
            cap.release()
    return found


class LocalSpeaker:
    """Plays TTS audio out the host's default output device.

    Used as the ``pa_sink`` value on LocalMicRecorder. play_audio() in
    app.pipeline detects it by the ``play_local`` attribute (duck-typed
    to avoid a circular import) and hands the utterance here.

    A single OutputStream is held open for the speaker's lifetime and
    chunks are written into it. The earlier sd.play() approach allocated
    and tore down a fresh CoreAudio stream per TTS chunk; since a reply
    is split into several waterfall chunks, each cold start under-ran
    its buffer and produced an audible click at every chunk boundary.
    Keeping one primed stream removes that crackle.
    """

    def __init__(self):
        self._stream: Optional[sd.OutputStream] = None
        self._sr: Optional[int] = None
        self._lock = threading.Lock()

    def _ensure_stream(self, sample_rate: int) -> None:
        # Reopen only if the rate changes (it doesn't within a session —
        # Kokoro and XTTS are both fixed 24 kHz — but stay correct if it
        # ever does).
        if self._stream is not None and self._sr == sample_rate:
            return
        self._close_stream()
        self._stream = sd.OutputStream(
            samplerate=sample_rate, channels=1, dtype="int16"
        )
        self._stream.start()
        self._sr = sample_rate

    def _close_stream(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
            self._sr = None

    def play_local(self, audio: np.ndarray, sample_rate: int) -> None:
        try:
            pcm = np.ascontiguousarray(
                np.asarray(audio, dtype=np.int16)
            ).reshape(-1, 1)
            with self._lock:
                self._ensure_stream(sample_rate)
                # Blocking write provides natural backpressure and keeps
                # the buffer primed across consecutive chunks.
                self._stream.write(pcm)
        except Exception as e:  # pragma: no cover - device dependent
            print(f"  [audio] local speaker playback failed: {e}")

    def close(self) -> None:
        with self._lock:
            self._close_stream()


class LocalMicRecorder:
    """Microphone capture from the host's default input device.

    Emits the same fixed-size mono int16 PCM chunks at SAMPLE_RATE that
    MicRecorder produces, so vad_loop and stream_and_speak don't care
    which backend is in use. pa_sink is a LocalSpeaker so TTS plays back
    through the host speakers.
    """

    def __init__(self, console: Console, chunk_ms: int = 30):
        self.console = console
        self.chunk_ms = chunk_ms
        self.chunk_samples = int(SAMPLE_RATE * chunk_ms / 1000)
        self.audio_q: queue.Queue[bytes] = queue.Queue()
        self.listening = threading.Event()
        self.listening.set()
        self.alive = True
        self._stream: Optional[sd.InputStream] = None
        # play_audio() inspects these. pa_source unused locally; pa_sink
        # is the host-speaker sentinel.
        self.pa_source: Optional[str] = None
        self.pa_sink = LocalSpeaker()

    def _callback(self, indata, _frames, _time_info, status):
        if status:
            # Overflows are non-fatal; just note once in a while.
            pass
        if not self.alive or not self.listening.is_set():
            return
        # indata is int16 (samples, CHANNELS); the pipeline wants mono.
        if indata.ndim == 2 and indata.shape[1] > 1:
            mono = indata.mean(axis=1).astype(np.int16)
        else:
            mono = indata.reshape(-1).astype(np.int16)
        self.audio_q.put(mono.tobytes())

    def start(self, *_args, **_kwargs) -> bool:
        """Open the input stream. Accepts/ignores (hw, mic_hint) for
        surface compatibility with MicRecorder.start()."""
        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
                blocksize=self.chunk_samples,
                callback=self._callback,
            )
            self._stream.start()
        except Exception as e:
            self.console.print(f"  [red]Local mic start failed: {e}[/red]")
            return False

        try:
            dev = sd.query_devices(kind="input")
            self.console.print(
                f"  Mic: local sounddevice ({dev['name']} → {SAMPLE_RATE}Hz)"
            )
        except Exception:
            self.console.print(f"  Mic: local sounddevice ({SAMPLE_RATE}Hz)")

        # Quick liveness check.
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
                self.console.print("  Mic: [yellow]quiet — speak up / check input[/yellow]")
        else:
            self.console.print("  [yellow]Mic: no audio data yet[/yellow]")
        return True

    def flush(self):
        while not self.audio_q.empty():
            try:
                self.audio_q.get_nowait()
            except queue.Empty:
                break

    def pause(self):
        # Genuinely stop the capture device (not just gate the queue) so
        # playback runs half-duplex — a live input stream contending with
        # the output stream on CoreAudio adds startup jitter/crackle.
        self.listening.clear()
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:
                pass
        self.flush()

    def resume(self):
        self.flush()
        if self._stream is not None:
            try:
                self._stream.start()
            except Exception:
                pass
        self.listening.set()

    def stop(self):
        self.alive = False
        try:
            self.pa_sink.close()
        except Exception:
            pass
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None


class LocalCamera(Camera):
    """Host webcam via OpenCV's default backend.

    The base Camera.open() first probes the Reachy SDK's get_video_device()
    (which targets the robot's specific USB camera) and then tries a
    V4L2 device index — neither is right on a laptop. This override goes
    straight to cv2.VideoCapture(device), which uses AVFoundation on
    macOS and the default backend elsewhere. Everything else (ring
    buffer, get_speech_frames, read_live) is inherited unchanged.
    """

    def open(self) -> bool:
        import cv2

        if self._cap is not None and self._cap.isOpened():
            return True

        # On macOS the first VideoCapture triggers a TCC permission
        # prompt and returns un-opened while the dialog is up. Retry for
        # a few seconds so a freshly-granted permission can take effect
        # within this run instead of failing instantly.
        deadline = time.monotonic() + 8.0
        attempt = 0
        while time.monotonic() < deadline:
            attempt += 1
            cap = cv2.VideoCapture(self.device)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self._cap = cap
                return True
            cap.release()
            if attempt == 1:
                print(
                    "  [camera] waiting for OS camera permission — if a "
                    "dialog appeared, click Allow (System Settings → "
                    "Privacy & Security → Camera → enable your terminal)"
                )
            time.sleep(1.0)

        self._cap = None
        return False

    def switch_device(self, index: int) -> bool:
        """Hot-swap to a different camera index while the capture thread
        keeps running. Returns True on success (old device kept on failure).

        The new capture is fully opened *before* swapping it in under
        _cap_lock and self._cap is never set to None, so _capture_loop
        (which breaks on a None/closed cap) is not torn down mid-switch.
        """
        import cv2

        if index == self.device and self._cap is not None and self._cap.isOpened():
            return True
        new = cv2.VideoCapture(index)
        if not new.isOpened():
            new.release()
            return False
        new.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        new.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        new.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        with self._cap_lock:
            old = self._cap
            self._cap = new
            self.device = index
        if old is not None:
            old.release()
        return True
