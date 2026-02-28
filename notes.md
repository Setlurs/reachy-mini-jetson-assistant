# Benchmarks — Jetson Orin Nano 8GB (JetPack 6)

All benchmarks on NVIDIA Jetson Orin Nano 8GB, CUDA 12.6, Ubuntu 22.04.

---

## TTS — Kokoro v1.0 (82M params, ONNX Runtime GPU)

Test sentence: *"The quick brown fox jumps over the lazy dog. This is a benchmark test for comparing different model quantizations on the Jetson Orin Nano."* (~8.4s of audio @ 24kHz, voice: `af_sarah`)

| Model | File Size | Inference | RTF | Notes |
|-------|-----------|-----------|-----|-------|
| **FP32** `kokoro-v1.0.onnx` | 311 MB | **1.00s** | 0.12x | **Best on GPU** — 8x faster than realtime |
| **FP16** `kokoro-v1.0.fp16.onnx` | 169 MB | **1.63s** | 0.19x | Saves 142 MB RAM, 5x faster than realtime |
| **INT8** `kokoro-v1.0.int8.onnx` | 88 MB | **12.72s** | 1.50x | Slower than realtime — 550 memcpy nodes kill GPU perf |

- RTF = Real-Time Factor (< 1.0 means faster than realtime)
- FP32 is the clear winner on GPU. FP16 is the only viable alternative if RAM-constrained.
- INT8 is only useful on CPU-only systems (no GPU memcpy overhead).
- All three use CUDAExecutionProvider. Warmup run excluded from timing.

**Decision:** Using **FP32** — best latency, RAM is acceptable (~311 MB model + runtime overhead).

---

## VLM — Cosmos-Reason2-2B (Q4_K_M GGUF, llama.cpp Docker)

Model: `Kbenkhaled/Cosmos-Reason2-2B-GGUF:Q4_K_M`, GPU-offloaded via llama.cpp, context 2048.
Architecture: **Qwen3-VL** (not Qwen2-VL). Input: images only via llama.cpp (video not yet supported).

### Architecture & Input Format

- Cosmos-Reason2-2B is built on **Qwen3-VL** and natively supports **both images and video** when run via vLLM/transformers.
- The GGUF version in **llama.cpp only supports image input** — video support is planned but not merged ([Issue #18389](https://github.com/ggml-org/llama.cpp/issues/18389), still open as of Feb 2026).
- Qwen3-VL image support was merged into llama.cpp Oct 2025 ([Issue #16207](https://github.com/ggml-org/llama.cpp/issues/16207)).
- With proper video input the model would get **temporal positional encoding (3D M-RoPE)** — it knows frame ordering in time. Our multi-image workaround sends N separate pictures without that temporal embedding.
- vLLM deployment requires ~24 GB GPU — does not fit on 8 GB Jetson.

### VLM Benchmark (single image, 640x480 JPEG, Q4_K_M)

12 rounds, continuous capture, `"What do you see?"` prompt, `max_tokens=128`.

| Metric | Value |
|--------|-------|
| RAM (model loaded) | ~550 MB steady-state |
| TTFT (image encode + prefill) | **0.79–0.84s** (very consistent) |
| Total response time | 1.2–3.6s (depends on response length) |
| Generation speed | ~9–19 words/s (longer responses faster) |
| JPEG frame size | 5–53 KB (depends on scene complexity) |

- TTFT ~0.8s is the fixed cost per image (vision encoder + prompt processing).
- No RAM spikes during inference — llama.cpp does in-place GPU computation on shared memory.
- Model accurately described person, clothing, gestures, office environment, whiteboard, plants.
- First frame was black (5 KB) due to camera warmup — detected correctly as "completely black."

### Multi-image Workaround (ring buffer approach)

Since llama.cpp doesn't support video, we use a background ring buffer:
- Capture at ~3 FPS continuously into a deque (1.5s lookback).
- On speech end, grab 3 evenly-spaced frames from the buffer.
- Send as separate `image_url` entries in the chat API.
- Model sees a low-rate "slideshow" — works for scene description, misses motion/temporal reasoning.

---

## LLM — Cosmos-Reason2-2B (Q8_0 GGUF, llama.cpp Docker)

Model: `Kbenkhaled/Cosmos-Reason2-2B-GGUF:Q8_0`, GPU-offloaded via llama.cpp, context 2048.

### Server-side (llama.cpp logs)

| Metric | Value | Notes |
|--------|-------|-------|
| Prompt eval (prefill) | **430–840 tok/s** | Varies with prompt cache hit; 1.2–2.3 ms/tok |
| Token generation | **24–29 tok/s** | ~36–41 ms/tok |

### End-to-end (run_voice_chat.py, --no-rag mode)

| Query | STT | TTFT | LLM Total | Words/s | Response Length |
|-------|-----|------|-----------|---------|-----------------|
| "Who are you?" | 0.8s | 0.5s | 1.4s | ~12 w/s | Short (1 sentence) |
| "Tell me about Nvidia" | 0.6s | 0.3s | 4.2s | ~21 w/s | Long paragraph |
| "In short, tell me about Ulama" | 0.8s | 0.2s | 2.4s | ~21 w/s | Medium paragraph |
| "Coughing" | 0.5s | 0.2s | 2.1s | ~19 w/s | Medium paragraph |
| "Thank you" | 0.5s | 0.2s | 0.8s | ~13 w/s | Short (1 sentence) |
| "Tell me about Jetson Orin Nano" | 0.9s | 0.3s | 3.6s | ~19 w/s | Long paragraph |
| "Give me a big para about Nvidia" | 0.8s | 0.2s | 5.4s | ~19 w/s | Long paragraph (128 tok) |

- **STT:** faster-whisper `small.en` on CUDA — consistently 0.5–0.9s
- **TTFT (Time To First Token):** 0.2–0.5s — fast prefill
- **LLM throughput:** ~19–21 words/s for longer outputs, ~12–14 w/s for short replies
- **Token generation:** ~25–28 tok/s (server-side)

---

## STT — faster-whisper (CTranslate2 + CUDA)

Model: `small.en` (GPU), CUDA warmup ~1.2–2.0s on first load.

| Metric | Value |
|--------|-------|
| Typical transcription latency | 0.5–0.9s |
| CUDA warmup (first load) | 1.2–2.0s |

---

## Typical End-to-End Latency (no-RAG mode)

For a spoken question → spoken answer:

| Stage | Typical Time |
|-------|-------------|
| STT (speech → text) | 0.5–0.9s |
| LLM TTFT (first token) | 0.2–0.5s |
| LLM generation | 0.8–5.4s (depends on response length) |
| TTS (text → speech, FP32 GPU) | ~1.0s per 8s of audio |
| **Total (short answer)** | **~1.5–2.0s** |
| **Total (long answer)** | **~5–7s** |
