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


class _PeakProbHandler(logging.Handler):
    """Captures the highest 'mean prob' microWakeWord logs for an utterance.

    microwakeword emits `"<wake word> mean prob: <float>"` at DEBUG when
    debug_probabilities is on. We tap that to expose how close a missed
    utterance got to the cutoff — the decisive signal for "raise/lower
    the threshold" vs "the model never recognized it at all".
    """

    def __init__(self):
        super().__init__()
        self.peak = 0.0

    def emit(self, record):
        msg = record.getMessage()
        i = msg.rfind("mean prob:")
        if i != -1:
            try:
                self.peak = max(self.peak, float(msg[i + len("mean prob:"):].strip()))
            except ValueError:
                pass


class WakeWordDetector:
    """Offline wake-word check for one utterance at a time."""

    def __init__(
        self,
        model_id: str = "okay_nabu",
        probability_cutoff: Optional[float] = None,
        debug: bool = False,
    ):
        from pymicro_wakeword import MicroWakeWord, Model

        attr = model_id.upper()
        if not hasattr(Model, attr):
            raise ValueError(
                f"Unknown wake-word model: {model_id!r}. "
                f"Try: okay_nabu, alexa, hey_jarvis, hey_mycroft."
            )
        self._ww = MicroWakeWord.from_builtin(getattr(Model, attr))
        self.model_id = model_id
        self.default_cutoff = float(self._ww.probability_cutoff)
        # The builtin cutoff (0.97) is very strict; allow an override so a
        # clean utterance that the model scores ~0.7 still triggers.
        if probability_cutoff is not None:
            self._ww.probability_cutoff = float(probability_cutoff)
        self.cutoff = float(self._ww.probability_cutoff)
        # Peak mean-probability from the most recent contains() call.
        self.last_peak = 0.0
        self._peak_handler: Optional[_PeakProbHandler] = None
        if debug:
            self._ww.debug_probabilities = True
            self._peak_handler = _PeakProbHandler()
            plog = logging.getLogger("pymicro_wakeword")
            plog.setLevel(logging.DEBUG)
            plog.addHandler(self._peak_handler)

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
        if self._peak_handler is not None:
            self._peak_handler.peak = 0.0

        features = MicroWakeWordFeatures()
        hit = False
        try:
            for frame in features.process_streaming(audio_bytes):
                if self._ww.process_streaming(frame) is True:
                    hit = True
                    break
        except Exception as e:
            logger.warning("Wake-word detection error (model=%s): %s", self.model_id, e)
            return False
        if self._peak_handler is not None:
            self.last_peak = self._peak_handler.peak
        return hit


def try_create_detector(
    model_id: str,
    probability_cutoff: Optional[float] = None,
    debug: bool = False,
) -> Optional[WakeWordDetector]:
    """Best-effort factory. Returns None if pymicro_wakeword isn't installed
    in the venv or the model id is unknown — callers treat None as "skip
    the wake-word check entirely."
    """
    try:
        return WakeWordDetector(
            model_id=model_id,
            probability_cutoff=probability_cutoff,
            debug=debug,
        )
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
