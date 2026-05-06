"""Wake-word detection for the HA filler-skip path.

When config.ha.enabled is true, the LLM pipeline runs the same wake-word
model the satellite uses (okay_nabu via pymicro_wakeword). If a captured
speech segment contains the wake word, our pipeline drops it as filler —
HA's satellite is handling the response, and we don't want to double up.

The detector is offline / batch-mode: it processes a complete utterance's
worth of audio bytes in one call rather than streaming live. That keeps
the integration trivial — feed `b"".join(segment.raw_chunks)` and read
the boolean back.
"""

from __future__ import annotations

import logging
from typing import Optional


logger = logging.getLogger(__name__)


class WakeWordDetector:
    """Offline wake-word check for one utterance at a time."""

    def __init__(self, model_id: str = "okay_nabu"):
        from pymicro_wakeword import MicroWakeWord, Model

        attr = model_id.upper()
        if not hasattr(Model, attr):
            raise ValueError(
                f"Unknown wake-word model: {model_id!r}. "
                f"Try: okay_nabu, alexa, hey_jarvis, hey_mycroft."
            )
        self._ww = MicroWakeWord.from_builtin(getattr(Model, attr))
        self.model_id = model_id

    def contains(self, audio_bytes: bytes) -> bool:
        """True if `audio_bytes` (int16 PCM, 16 kHz mono) contains the wake word.

        Internally resets the model's sliding-window state before scanning so
        each call is independent. Errors are logged but don't crash the
        caller — a failure here just means the utterance falls through to
        the LLM as today.
        """
        from pymicro_wakeword import MicroWakeWordFeatures

        if not audio_bytes:
            return False

        try:
            self._ww.reset()
        except Exception:
            pass

        features = MicroWakeWordFeatures()
        try:
            for frame in features.process_streaming(audio_bytes):
                if self._ww.process_streaming(frame) is True:
                    return True
        except Exception as e:
            logger.warning("Wake-word detection error (model=%s): %s", self.model_id, e)
            return False
        return False


def try_create_detector(model_id: str) -> Optional[WakeWordDetector]:
    """Best-effort factory. Returns None if pymicro_wakeword isn't installed
    in the venv or the model id is unknown — callers treat None as "skip
    the wake-word check entirely."
    """
    try:
        return WakeWordDetector(model_id=model_id)
    except ImportError:
        logger.info(
            "pymicro_wakeword not installed; wake-word filler skip disabled. "
            "Install reachy_mini_home_assistant or `pip install pymicro_wakeword` "
            "to enable."
        )
        return None
    except Exception as e:
        logger.warning("Wake-word detector init failed: %s", e)
        return None
