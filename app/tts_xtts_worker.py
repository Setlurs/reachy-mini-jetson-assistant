#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# XTTS-v2 subprocess worker — voice cloning sibling of app/tts_worker.py.
#
# Loads the XTTS-v2 model once at startup with a reference WAV (the user's
# voice clone), then synthesizes utterances on demand. Same JSON-over-stdin
# protocol as the Kokoro worker so the XTTSCloningTTS client class drives
# both identically.
#
# Note: XTTS-v2 model weights are licensed under Coqui's CPML
# (non-commercial). Suitable for demos and personal use; consult the
# model card for production licensing implications.

import argparse
import base64
import json
import os
import sys
from pathlib import Path

import numpy as np


def _respond(obj: dict):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _log(msg: str):
    print(msg, file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(description="XTTS-v2 subprocess worker")
    parser.add_argument(
        "--speaker-wav", required=True,
        help="Path to reference voice WAV (~6-15s of clean speech)",
    )
    parser.add_argument("--language", default="en")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    speaker_wav = Path(args.speaker_wav)
    if not speaker_wav.exists():
        _respond({"status": "error",
                  "error": f"speaker_wav not found: {speaker_wav}"})
        sys.exit(1)

    # Coqui's TTS package emits a EULA prompt on first model load that
    # blocks stdin. Suppress it — the user is already opting in by
    # flipping the backend flag.
    os.environ.setdefault("COQUI_TOS_AGREED", "1")

    try:
        # Import inside the try block so missing deps surface as a clean
        # error response instead of a Python traceback splatted to stderr.
        try:
            from TTS.api import TTS  # type: ignore
        except ImportError:
            try:
                # Maintained fork after Coqui's company shutdown.
                from coqui_tts.api import TTS  # type: ignore
            except ImportError as e:
                _respond({
                    "status": "error",
                    "error": (
                        "Coqui TTS not installed. Run "
                        "`pip install -r requirements-xtts.txt` into the venv."
                    ),
                })
                sys.exit(1)

        import torch  # noqa: WPS433
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            # MPS works for XTTS but some ops fall back to CPU. Honor an
            # opt-out env var so users hitting MPS-specific bugs can pin
            # to CPU without code changes.
            device = "cpu" if os.environ.get("XTTS_DISABLE_MPS") else "mps"
        else:
            device = "cpu"

        # On first run Coqui silently downloads ~1.5 GB of model weights
        # from Hugging Face into ~/Library/Application Support/tts/ (or
        # the platform equivalent). That can take several minutes with
        # no output — print a heartbeat so the parent's startup banner
        # doesn't look frozen.
        _log("XTTS-v2: loading model (first run downloads ~1.5 GB, "
             "this can take several minutes)...")
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        sample_rate = 24000  # XTTS-v2 native output rate

        _log(f"XTTS-v2 loaded on {device} (speaker={speaker_wav.name}, "
             f"lang={args.language})")
        _respond({
            "status": "ready",
            "provider": device,
            "sample_rate": sample_rate,
        })

    except Exception as e:
        _respond({"status": "error", "error": str(e)})
        sys.exit(1)

    speaker_wav_str = str(speaker_wav)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            _respond({"error": "Invalid JSON"})
            continue

        cmd = req.get("cmd", "synthesize")

        if cmd == "synthesize":
            text = req.get("text", "")
            language = req.get("language", args.language)
            temperature = float(req.get("temperature", args.temperature))

            if not text.strip():
                _respond({"error": "Empty text"})
                continue

            try:
                # tts.tts returns a list[float] in [-1, 1] at 24 kHz.
                samples = tts.tts(
                    text=text,
                    speaker_wav=speaker_wav_str,
                    language=language,
                    temperature=temperature,
                )
                audio_f32 = np.asarray(samples, dtype=np.float32)
                audio_int16 = (np.clip(audio_f32, -1.0, 1.0) * 32767).astype(np.int16)
                audio_b64 = base64.b64encode(audio_int16.tobytes()).decode("ascii")
                _respond({"audio_b64": audio_b64, "sample_rate": sample_rate})
            except Exception as e:
                _respond({"error": str(e)})

        elif cmd == "health":
            _respond({"healthy": True})

        elif cmd == "shutdown":
            _respond({"status": "shutdown"})
            break


if __name__ == "__main__":
    main()
