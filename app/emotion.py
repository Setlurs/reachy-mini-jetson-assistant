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

"""Emotion detection — lightweight ONNX sentiment classifier on user speech.

Runs a quantized DistilBERT (~65 MB int8) on CPU to classify user sentiment
before the VLM starts generating, so the robot can react immediately.

Model: distilbert-base-uncased-finetuned-sst-2 (binary: NEGATIVE/POSITIVE)
combined with text heuristics for richer emotion categories.
"""

import re
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "emotion"

MODEL_REPO = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
MODEL_FILES = {
    "model": f"https://huggingface.co/{MODEL_REPO}/resolve/main/onnx/model.onnx",
    "tokenizer": f"https://huggingface.co/{MODEL_REPO}/resolve/main/onnx/tokenizer.json",
}


class Emotion(Enum):
    HAPPY = "happy"
    SAD = "sad"
    CURIOUS = "curious"
    EXCITED = "excited"
    SURPRISED = "surprised"
    GREETING = "greeting"
    FAREWELL = "farewell"
    GRATEFUL = "grateful"
    NEUTRAL = "neutral"


@dataclass
class EmotionResult:
    emotion: Emotion
    confidence: float
    sentiment: str
    sentiment_score: float
    inference_ms: float


_GREETING_RE = re.compile(
    r"\b(hi|hello|hey|howdy|good\s+(morning|afternoon|evening)|what'?s\s+up|yo)\b", re.I
)
_FAREWELL_RE = re.compile(
    r"\b(bye|goodbye|see\s+you|later|good\s*night|take\s+care)\b", re.I
)
_GRATEFUL_RE = re.compile(
    r"\b(thanks?|thank\s+you|appreciate|grateful)\b", re.I
)


def _download_file(url: str, path: Path, label: str) -> bool:
    try:
        import httpx
    except ImportError:
        print("Emotion: install httpx to auto-download model (pip install httpx)")
        return False
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
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        if path.exists():
            path.unlink()
        return False


def _ensure_model_files() -> bool:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "model.onnx"
    tokenizer_path = MODELS_DIR / "tokenizer.json"

    needed = []
    if not model_path.exists():
        needed.append((MODEL_FILES["model"], model_path, "emotion model (~268 MB)"))
    if not tokenizer_path.exists():
        needed.append((MODEL_FILES["tokenizer"], tokenizer_path, "tokenizer.json"))

    if not needed:
        return True

    print("Downloading emotion model...")
    for url, path, label in needed:
        if not _download_file(url, path, label):
            return False
        print(f"  Saved {path}")
    return True


class EmotionDetector:
    """Lightweight sentiment-based emotion detector using ONNX Runtime.

    Runs on CPU to keep GPU free for VLM. Typical inference: ~5-15ms.
    """

    def __init__(self):
        self._session = None
        self._tokenizer = None
        self._labels = ["NEGATIVE", "POSITIVE"]

    def load(self) -> bool:
        if not _ensure_model_files():
            return False
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            model_path = MODELS_DIR / "model.onnx"
            tokenizer_path = MODELS_DIR / "tokenizer.json"

            self._session = ort.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"],
            )
            self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
            self._tokenizer.enable_truncation(max_length=128)
            self._tokenizer.enable_padding(length=128)
            return True
        except ImportError as e:
            print(f"Emotion: missing dependency — {e}")
            print("  pip install tokenizers")
            return False
        except Exception as e:
            print(f"Emotion: load error — {e}")
            return False

    def detect(self, text: str) -> EmotionResult:
        """Classify the emotion of user text. Returns in ~5-15ms on CPU."""
        if not text.strip():
            return EmotionResult(Emotion.NEUTRAL, 0.0, "NEUTRAL", 0.5, 0.0)

        t0 = time.perf_counter()

        sentiment, score = self._classify_sentiment(text)
        emotion, confidence = self._map_emotion(text, sentiment, score)

        dt = (time.perf_counter() - t0) * 1000
        return EmotionResult(emotion, confidence, sentiment, score, dt)

    def _classify_sentiment(self, text: str) -> tuple[str, float]:
        if self._session is None or self._tokenizer is None:
            return "NEUTRAL", 0.5

        encoded = self._tokenizer.encode(text)
        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)

        outputs = self._session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        })

        logits = outputs[0][0]
        probs = _softmax(logits)
        idx = int(np.argmax(probs))
        return self._labels[idx], float(probs[idx])

    def _map_emotion(
        self, text: str, sentiment: str, score: float
    ) -> tuple[Emotion, float]:
        """Combine binary sentiment with text heuristics for richer categories.

        Priority: greeting/farewell/thanks (pattern match) > strong sentiment
        (score > 0.85) > question fallback > weak sentiment > neutral.
        """
        if _GREETING_RE.search(text):
            return Emotion.GREETING, 0.95

        if _FAREWELL_RE.search(text):
            return Emotion.FAREWELL, 0.95

        if _GRATEFUL_RE.search(text):
            return Emotion.GRATEFUL, 0.90

        has_exclamation = "!" in text

        if sentiment == "POSITIVE" and score > 0.85:
            if has_exclamation:
                return Emotion.EXCITED, score
            return Emotion.HAPPY, score

        if sentiment == "NEGATIVE" and score > 0.85:
            return Emotion.SAD, score

        is_question = text.rstrip().endswith("?")
        if is_question:
            return Emotion.CURIOUS, 0.70

        if sentiment == "POSITIVE" and score > 0.6:
            return Emotion.HAPPY, score

        if sentiment == "NEGATIVE" and score > 0.6:
            return Emotion.SAD, score

        return Emotion.NEUTRAL, 0.5

    def health_check(self) -> bool:
        return self._session is not None and self._tokenizer is not None

    def unload(self):
        if self._session:
            del self._session
            self._session = None
        self._tokenizer = None


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()
