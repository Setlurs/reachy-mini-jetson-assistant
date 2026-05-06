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
#
# TTS — subprocess-isolated Kokoro TTS.
#
# kokoro-onnx depends on phonemizer-fork (GPL-3.0) and espeak-ng (GPL-3.0).
# To avoid loading GPL code into the same process as NVIDIA CUDA libraries,
# synthesis runs in a separate subprocess (app/tts_worker.py) that
# communicates via JSON lines over stdin/stdout.

import re
import sys
import json
import wave
import base64
import subprocess
from typing import Dict, Any, List, Optional
from pathlib import Path

import numpy as np


# ── Text cleaning + waterfall chunking ───────────────────────────
#
# Two helpers ported from reachy_mini_conversation_app_local — they're
# what makes Kokoro sound polished there. The cleaner strips markdown
# noise the LLM occasionally emits (asterisks, bracketed stage
# directions, etc.); the chunker buffers token chunks and only emits
# them on natural punctuation boundaries instead of arbitrary word
# counts. Both are used by app/pipeline.py and run_web_vision_chat.py.


try:
    from num2words import num2words as _num2words  # type: ignore
    _HAS_NUM2WORDS = True
except ImportError:
    _HAS_NUM2WORDS = False

# Numbers Kokoro should hear as words: 32,575 / 100 / 18.4 (matches comma
# groupings or plain int/decimal). The lookarounds guard against digits
# embedded in alphanumeric identifiers like "PM2.5" or "PM10" — those are
# names, not quantities, and should be left alone for the model to spell.
# Years like 2026 still match — fine for our use case ("two thousand
# twenty six" reads cleanly).
# Lookbehind rejects letters, digits, and '.' so identifiers like
# 'PM2.5', 'PM10', or 'v1.0' aren't mangled — only standalone numbers
# preceded by whitespace or punctuation get vocalized. Lookahead rejects
# trailing letters so '32K' or '5GB' stay alphanumeric.
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z.\d])"
    r"(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?![A-Za-z])"
)


def _spoken_number(match: "re.Match[str]") -> str:
    raw = match.group()
    if not _HAS_NUM2WORDS:
        return raw
    cleaned = raw.replace(",", "")
    try:
        n = float(cleaned) if "." in cleaned else int(cleaned)
        out = _num2words(n)
    except Exception:
        return raw
    # Smooth num2words output: drop hyphens, commas, and the "and"
    # connector so Kokoro sees a flat word sequence.
    return out.replace("-", " ").replace(",", "").replace(" and ", " ")


def clean_text_for_speech(text: str) -> str:
    """Strip markdown / bracketed asides and turn digit groups into words.

    Goals:
      - LLM occasionally emits markdown (asterisks, brackets, headers) —
        strip so Kokoro doesn't read them literally.
      - Numbers like '32,575' should be vocalized as
        'thirty two thousand five hundred seventy five', not stumbled
        over digit-by-digit.
    """
    text = re.sub(r"\([^)]*\)", "", text)        # (parentheticals)
    text = re.sub(r"\[[^\]]*\]", "", text)       # [stage directions]
    text = re.sub(r"\{[^}]*\}", "", text)        # {tags}
    text = re.sub(r"\*+", "", text)              # *emphasis* / **bold**
    text = re.sub(r"_+", "", text)               # _underscores_
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)  # markdown headers
    text = _NUMBER_RE.sub(_spoken_number, text)  # 32,575 → thirty two thousand…
    text = re.sub(r"\s+", " ", text)             # collapse whitespace
    return text.strip()


# Waterfall break points, strongest natural pause first.
_CHUNK_BREAKS: List[re.Pattern] = [
    re.compile(r'([.!?…]+["\'\)]?\s+)'),  # sentence end (with trailing close-quote/paren)
    re.compile(r"([:;]\s+)"),              # clause separator
    re.compile(r"([,—]\s+)"),              # phrase separator
    re.compile(r"(\s+)"),                  # last-resort: any whitespace
]


def _find_break(buf: str, target: int, slack: int = 20) -> Optional[int]:
    """Return the best break index in buf around `target` chars, or None.

    Searches up to `target + slack` so a sentence end *just past* the
    target can still anchor the chunk. Falls through the waterfall:
    sentence > clause > phrase > whitespace, taking the latest match in
    each priority that fits within the window.
    """
    window = buf[: target + slack]
    for pattern in _CHUNK_BREAKS:
        matches = list(pattern.finditer(window))
        if not matches:
            continue
        # Prefer the latest match within the soft target.
        for m in reversed(matches):
            if m.end() <= target + slack and m.end() > 20:
                return m.end()
    return None


class StreamingChunker:
    """Buffer tokens from the LLM and emit TTS-ready chunks on natural breaks.

    The first chunk uses a smaller char target so the robot starts
    speaking quickly; subsequent chunks aim for full sentences (~max_chars)
    to keep prosody coherent. Mirrors the reference project's waterfall
    splitter, adapted for incremental token feed instead of a finished
    string.
    """

    def __init__(self, first_chunk_chars: int = 60, max_chunk_chars: int = 150):
        self.first_chunk_chars = max(20, first_chunk_chars)
        self.max_chunk_chars = max(self.first_chunk_chars, max_chunk_chars)
        self._buf = ""
        self._first_emitted = False

    def feed(self, token: str) -> List[str]:
        """Append a token; return any chunks ready for synthesis."""
        if not token:
            return []
        self._buf += token
        out: List[str] = []
        while True:
            chunk = self._try_extract()
            if chunk is None:
                break
            out.append(chunk)
        return out

    def flush(self) -> Optional[str]:
        """Return whatever's left in the buffer (call at end of stream)."""
        rest = self._buf.strip()
        self._buf = ""
        if rest:
            self._first_emitted = True
            return rest
        return None

    def _try_extract(self) -> Optional[str]:
        target = self.first_chunk_chars if not self._first_emitted else self.max_chunk_chars
        if len(self._buf) < target:
            return None
        idx = _find_break(self._buf, target)
        if idx is None:
            # No natural break in the window — let more text accumulate
            # rather than slice mid-word. Bail out at hard ceiling.
            if len(self._buf) < target * 2:
                return None
            # Hard cap reached: split at the last space we can find.
            sp = self._buf.rfind(" ", 0, target * 2)
            idx = sp if sp > 20 else target * 2
        chunk = self._buf[:idx].strip()
        self._buf = self._buf[idx:]
        if chunk:
            self._first_emitted = True
        return chunk or None


VOICES_DIR = Path(__file__).resolve().parent.parent / "voices"

KOKORO_MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
KOKORO_VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"


def _download_kokoro_models_if_missing() -> bool:
    """Download Kokoro model and voices to voices/ if not present."""
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    model_path = VOICES_DIR / "kokoro-v1.0.onnx"
    voices_path = VOICES_DIR / "voices-v1.0.bin"
    needed = []
    if not model_path.exists():
        needed.append((KOKORO_MODEL_URL, model_path, "kokoro-v1.0.onnx (~311 MB)"))
    if not voices_path.exists():
        needed.append((KOKORO_VOICES_URL, voices_path, "voices-v1.0.bin (~30 MB)"))
    if not needed:
        return True
    try:
        import httpx
    except ImportError:
        print("Kokoro: install httpx to auto-download models (pip install httpx)")
        return False
    for url, path, label in needed:
        print(f"Downloading {label} to {path} ...")
        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0)) or None
                done = 0
                with open(path, "wb") as f:
                    for chunk in r.iter_bytes(chunk_size=262144):
                        f.write(chunk)
                        done += len(chunk)
                        if total and total > 0:
                            pct = 100 * done / total
                            sys.stdout.write(f"\r  {label}: {pct:.0f}%\r")
                            sys.stdout.flush()
            if total:
                print()
            print(f"  Saved {path}")
        except Exception as e:
            print(f"  Download failed: {e}")
            return False
    return True


class KokoroTTS:
    """Kokoro TTS client — synthesis runs in a subprocess for GPL isolation."""

    def __init__(self, voice: str = "af_sarah", speed: float = 1.0, lang: str = "en-us"):
        self.voice = voice
        self.speed = speed
        self.lang = lang
        self._proc: Optional[subprocess.Popen] = None
        self._sample_rate = 24000
        self.backend_name = "Kokoro"
        self.provider = "unknown"

    def load(self) -> bool:
        model_path = VOICES_DIR / "kokoro-v1.0.onnx"
        voices_path = VOICES_DIR / "voices-v1.0.bin"
        if not model_path.exists() or not voices_path.exists():
            if not _download_kokoro_models_if_missing():
                return False

        worker = Path(__file__).parent / "tts_worker.py"
        try:
            self._proc = subprocess.Popen(
                [sys.executable, str(worker),
                 "--model-dir", str(VOICES_DIR),
                 "--voice", self.voice,
                 "--speed", str(self.speed),
                 "--lang", self.lang],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,  # inherit parent's stderr for log visibility
                text=True,
                bufsize=1,
            )
        except Exception as e:
            print(f"TTS worker spawn failed: {e}")
            return False

        line = self._proc.stdout.readline()
        if not line:
            print("TTS worker exited before signalling ready")
            return False

        try:
            resp = json.loads(line)
        except json.JSONDecodeError:
            print(f"TTS worker sent invalid init response: {line!r}")
            return False

        if resp.get("status") == "ready":
            self.provider = resp.get("provider", "unknown")
            return True

        print(f"TTS worker error: {resp.get('error', 'unknown')}")
        return False

    def _send(self, req: dict) -> Optional[dict]:
        if not self._proc or self._proc.poll() is not None:
            return None
        try:
            self._proc.stdin.write(json.dumps(req) + "\n")
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
            if not line:
                return None
            return json.loads(line)
        except (BrokenPipeError, json.JSONDecodeError, OSError):
            return None

    def synthesize(self, text: str) -> Dict[str, Any]:
        if not text.strip():
            return {"audio": None, "error": "Empty"}

        resp = self._send({
            "cmd": "synthesize",
            "text": text,
            "voice": self.voice,
            "speed": self.speed,
            "lang": self.lang,
        })
        if resp is None:
            return {"audio": None, "error": "Worker not running"}
        if "error" in resp:
            return {"audio": None, "error": resp["error"]}

        audio_bytes = base64.b64decode(resp["audio_b64"])
        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        return {"audio": audio, "sample_rate": resp["sample_rate"]}

    def synthesize_to_file(self, text: str, path: str) -> bool:
        r = self.synthesize(text)
        if r.get("audio") is None:
            return False
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(r["sample_rate"])
            wf.writeframes(r["audio"].tobytes())
        return True

    def health_check(self) -> bool:
        resp = self._send({"cmd": "health"})
        return resp is not None and resp.get("healthy", False)

    def unload(self):
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        # Politely ask the worker to shut down; fall back to terminate, then
        # kill, so the process is reaped before our parent exits. Closing
        # stdin/stdout explicitly avoids ResourceWarnings about unclosed
        # subprocess pipes that get leaked when the parent dies first.
        try:
            if proc.poll() is None:
                try:
                    proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                    proc.stdin.flush()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        try:
                            proc.wait(timeout=1)
                        except Exception:
                            pass
        finally:
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass


class XTTSCloningTTS:
    """XTTS-v2 voice-cloning client — synthesis runs in a subprocess.

    Mirrors KokoroTTS's surface (load / synthesize / synthesize_to_file /
    health_check / unload) so the rest of the pipeline is backend-agnostic.
    The worker holds a reference WAV (the user's voice clone) and the
    XTTS-v2 model in memory; each synthesize() call is a JSON-over-stdin
    round-trip that returns base64-encoded int16 PCM at 24 kHz.
    """

    def __init__(
        self,
        speaker_wav: str,
        language: str = "en",
        temperature: float = 0.7,
    ):
        self.speaker_wav = speaker_wav
        self.language = language
        self.temperature = temperature
        self._proc: Optional[subprocess.Popen] = None
        self._sample_rate = 24000
        self.backend_name = "XTTS-v2"
        # Surfaced in the startup banner; users see "voice clone of <name>"
        # rather than a generic backend name.
        self.voice = f"clone of {Path(speaker_wav).stem}"
        self.provider = "unknown"

    def _resolve_speaker_wav(self) -> Path:
        """Resolve relative speaker_wav paths against the project root so
        the YAML default `voices/clone.wav` works regardless of CWD."""
        p = Path(self.speaker_wav)
        if p.is_absolute():
            return p
        return Path(__file__).resolve().parent.parent / p

    def load(self) -> bool:
        speaker_path = self._resolve_speaker_wav()
        if not speaker_path.exists():
            print(
                f"XTTS speaker WAV not found: {speaker_path} — "
                "place a 6–15s clean speech recording at this path or "
                "update tts.xtts_speaker_wav in settings.yaml."
            )
            return False

        worker = Path(__file__).parent / "tts_xtts_worker.py"
        try:
            self._proc = subprocess.Popen(
                [sys.executable, str(worker),
                 "--speaker-wav", str(speaker_path),
                 "--language", self.language,
                 "--temperature", str(self.temperature)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
                bufsize=1,
            )
        except Exception as e:
            print(f"XTTS worker spawn failed: {e}")
            return False

        line = self._proc.stdout.readline()
        if not line:
            print("XTTS worker exited before signalling ready (model load failed?)")
            return False

        try:
            resp = json.loads(line)
        except json.JSONDecodeError:
            print(f"XTTS worker sent invalid init response: {line!r}")
            return False

        if resp.get("status") == "ready":
            self.provider = resp.get("provider", "unknown")
            self._sample_rate = resp.get("sample_rate", 24000)
            return True

        print(f"XTTS worker error: {resp.get('error', 'unknown')}")
        return False

    def _send(self, req: dict) -> Optional[dict]:
        if not self._proc or self._proc.poll() is not None:
            return None
        try:
            self._proc.stdin.write(json.dumps(req) + "\n")
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
            if not line:
                return None
            return json.loads(line)
        except (BrokenPipeError, json.JSONDecodeError, OSError):
            return None

    def synthesize(self, text: str) -> Dict[str, Any]:
        if not text.strip():
            return {"audio": None, "error": "Empty"}

        resp = self._send({
            "cmd": "synthesize",
            "text": text,
            "language": self.language,
            "temperature": self.temperature,
        })
        if resp is None:
            return {"audio": None, "error": "Worker not running"}
        if "error" in resp:
            return {"audio": None, "error": resp["error"]}

        audio_bytes = base64.b64decode(resp["audio_b64"])
        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        return {"audio": audio, "sample_rate": resp["sample_rate"]}

    def synthesize_to_file(self, text: str, path: str) -> bool:
        r = self.synthesize(text)
        if r.get("audio") is None:
            return False
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(r["sample_rate"])
            wf.writeframes(r["audio"].tobytes())
        return True

    def health_check(self) -> bool:
        resp = self._send({"cmd": "health"})
        return resp is not None and resp.get("healthy", False)

    def unload(self):
        # Identical teardown discipline to KokoroTTS.unload — graceful
        # shutdown cmd, then terminate, then kill; explicit pipe close
        # so the subprocess is reaped before the parent exits.
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                try:
                    proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                    proc.stdin.flush()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        try:
                            proc.wait(timeout=1)
                        except Exception:
                            pass
        finally:
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass


def create_tts(
    backend: str = "kokoro",
    *,
    voice: str = "",
    speed: float = 1.0,
    lang: str = "en-us",
    xtts_speaker_wav: str = "",
    xtts_language: str = "en",
    xtts_temperature: float = 0.7,
    **_kwargs,
):
    """Build the configured TTS backend.

    Routes by `backend`. Both backends expose the same client surface
    (load / synthesize / synthesize_to_file / health_check / unload),
    so the rest of the pipeline doesn't need to know which one is in
    use. Callers should still .load() the returned object and check the
    return value — if XTTS dependencies aren't installed or the speaker
    WAV is missing, load() returns False and the caller can fall back
    to Kokoro for the session.
    """
    backend = (backend or "kokoro").strip().lower()

    if backend == "xtts":
        return XTTSCloningTTS(
            speaker_wav=xtts_speaker_wav or "voices/clone.wav",
            language=xtts_language or "en",
            temperature=float(xtts_temperature),
        )

    # Default: Kokoro. Unknown backend names also fall here (logged by
    # the caller's load() check, not silently misrouted).
    return KokoroTTS(voice=voice or "af_sarah", speed=speed, lang=lang)
