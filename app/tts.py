"""TTS — pluggable text-to-speech backends (Piper and Kokoro)."""

import wave
from typing import Dict, Any
from pathlib import Path
import numpy as np


VOICES_DIR = Path(__file__).resolve().parent.parent / "voices"


class PiperTTS:
    """Piper neural TTS (CPU, lightweight, ~61 MB model)."""

    def __init__(self, voice: str = "en_US-lessac-medium", speed: float = 1.0):
        self.voice = voice
        self.speed = speed
        self._piper = None
        self._sample_rate = 22050
        self.backend_name = "Piper"

    def load(self) -> bool:
        try:
            from piper import PiperVoice

            search = [
                VOICES_DIR / f"{self.voice}.onnx",
                Path(self.voice),
                Path(f"{self.voice}.onnx"),
                Path.home() / ".local" / "share" / "piper" / "voices" / f"{self.voice}.onnx",
            ]
            for p in search:
                try:
                    if p.exists():
                        cfg = p.with_suffix(".json")
                        self._piper = PiperVoice.load(str(p), config_path=str(cfg) if cfg.exists() else None)
                        if hasattr(self._piper, "config") and self._piper.config:
                            self._sample_rate = getattr(self._piper.config, "sample_rate", 22050)
                        return True
                except Exception:
                    continue
            print(f"TTS voice not found: {self.voice}")
            return False
        except ImportError:
            print("piper-tts not installed")
            return False

    def synthesize(self, text: str) -> Dict[str, Any]:
        if self._piper is None:
            return {"audio": None, "error": "Not loaded"}
        if not text.strip():
            return {"audio": None, "error": "Empty"}
        try:
            chunks = []
            for chunk in self._piper.synthesize(text):
                chunks.append((chunk.audio_float_array * 32767).astype(np.int16))
            if not chunks:
                return {"audio": None, "error": "No audio"}
            return {"audio": np.concatenate(chunks), "sample_rate": self._sample_rate}
        except Exception as e:
            return {"audio": None, "error": str(e)}

    def synthesize_to_file(self, text: str, path: str) -> bool:
        r = self.synthesize(text)
        if r.get("audio") is None:
            return False
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self._sample_rate)
            wf.writeframes(r["audio"].tobytes())
        return True

    def health_check(self) -> bool:
        return self._piper is not None

    def unload(self):
        if self._piper:
            del self._piper
            self._piper = None


class KokoroTTS:
    """Kokoro neural TTS (ONNX Runtime, high quality, ~300 MB model)."""

    def __init__(self, voice: str = "af_sarah", speed: float = 1.0, lang: str = "en-us"):
        self.voice = voice
        self.speed = speed
        self.lang = lang
        self._kokoro = None
        self._sample_rate = 24000
        self.backend_name = "Kokoro"

    def load(self) -> bool:
        try:
            import os
            import onnxruntime as ort

            model_path = VOICES_DIR / "kokoro-v1.0.onnx"
            voices_path = VOICES_DIR / "voices-v1.0.bin"

            if not model_path.exists():
                print(f"Kokoro model not found: {model_path}")
                return False
            if not voices_path.exists():
                print(f"Kokoro voices not found: {voices_path}")
                return False

            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                os.environ["ONNX_PROVIDER"] = "CUDAExecutionProvider"
            elif "TensorrtExecutionProvider" in available:
                os.environ["ONNX_PROVIDER"] = "TensorrtExecutionProvider"

            from kokoro_onnx import Kokoro
            self._kokoro = Kokoro(str(model_path), str(voices_path))

            provider_used = self._kokoro.sess.get_providers()[0]
            print(f"Kokoro TTS loaded — ONNX provider: {provider_used}")
            return True
        except ImportError:
            print("kokoro-onnx not installed (pip install kokoro-onnx)")
            return False
        except Exception as e:
            print(f"Kokoro load error: {e}")
            return False

    def synthesize(self, text: str) -> Dict[str, Any]:
        if self._kokoro is None:
            return {"audio": None, "error": "Not loaded"}
        if not text.strip():
            return {"audio": None, "error": "Empty"}
        try:
            samples, sample_rate = self._kokoro.create(
                text, voice=self.voice, speed=self.speed, lang=self.lang,
            )
            audio_int16 = (samples * 32767).astype(np.int16)
            return {"audio": audio_int16, "sample_rate": sample_rate}
        except Exception as e:
            return {"audio": None, "error": str(e)}

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
        return self._kokoro is not None

    def unload(self):
        if self._kokoro:
            del self._kokoro
            self._kokoro = None


# Keep backward compat: TTS = PiperTTS
TTS = PiperTTS


def create_tts(backend: str = "piper", voice: str = "", speed: float = 1.0,
               piper_voice: str = "en_US-lessac-medium", lang: str = "en-us"):
    """Factory: create the right TTS backend."""
    if backend == "kokoro":
        return KokoroTTS(voice=voice or "af_sarah", speed=speed, lang=lang)
    else:
        return PiperTTS(voice=voice or piper_voice, speed=speed)
