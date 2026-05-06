# Setup Guide

Full installation instructions. Two host platforms are supported — pick one:

* **NVIDIA Jetson** (the original target). Continue with the sections below.
* **macOS / Apple Silicon** — jump to [macOS Install](#macos--apple-silicon) at the bottom; the Jetson sections do not apply.

## Prerequisites

### Hardware

- **NVIDIA Jetson Orin Nano** (8GB) — other Jetson modules may work but are untested
- **[Reachy Mini Lite](https://huggingface.co/docs/reachy_mini/platforms/reachy_mini_lite/get_started)** — the developer version, USB connection to your computer. Provides camera, microphone, speaker, and 9-DOF motor control in one cable. [Buy Reachy Mini](https://www.hf.co/reachy-mini/)
- **NVMe SSD** recommended — for swap space and model storage

If you're new to Reachy Mini, start with the [official getting started guide](https://huggingface.co/docs/reachy_mini/index) and the [Reachy Mini Lite setup](https://huggingface.co/docs/reachy_mini/platforms/reachy_mini_lite/get_started). The [Python SDK documentation](https://huggingface.co/docs/reachy_mini/SDK/readme) covers movement, camera, audio, and AI integrations.

### Software

- **JetPack 6.x** (L4T r36.x, Ubuntu 22.04, CUDA 12.6)
- **Python 3.10** (ships with JetPack 6 Ubuntu 22.04)
- **Docker** with NVIDIA runtime (`nvidia-container-toolkit`)
- **PulseAudio** (for mic/speaker multiplexing)

> **Important:** This project requires **Python 3.10** specifically. The Jetson ONNX Runtime GPU wheels, CTranslate2 builds, and Reachy Mini SDK are all built against Python 3.10 on JetPack 6. Using a different Python version will cause compatibility issues.

## Hardware Setup

### Reachy Mini Lite

1. Connect Reachy Mini Lite to your Jetson via USB. The robot provides camera, microphone, speaker, and motor control over a single USB connection.

2. Add udev rules so the SDK can access the robot's serial ports without root:

```bash
echo 'SUBSYSTEM=="tty", ATTRS{idVendor}=="2e8a", ATTRS{idProduct}=="000a", MODE="0666", SYMLINK+="reachy_mini"' \
  | sudo tee /etc/udev/rules.d/99-reachy-mini.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

3. Add your user to the `dialout` group and reboot:

```bash
sudo usermod -aG dialout $USER
sudo reboot
```

4. Verify the device is visible:

```bash
ls -la /dev/ttyACM*
# Should show /dev/ttyACM0, /dev/ttyACM1, etc.
```

### NVMe Swap (Required for 8GB Jetson)

Running STT + VLM + TTS simultaneously exceeds 8GB RAM. Setting up swap on NVMe prevents OOM kills:

```bash
sudo fallocate -l 8G /mnt/nvme/swapfile   # adjust path to your NVMe mount
sudo chmod 600 /mnt/nvme/swapfile
sudo mkswap /mnt/nvme/swapfile
sudo swapon /mnt/nvme/swapfile

# Persist across reboots:
echo '/mnt/nvme/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## Installation

### Step 1: System Dependencies

```bash
sudo apt-get update
sudo apt-get install -y \
  python3.10-venv \
  portaudio19-dev \
  libasound2-dev \
  pulseaudio-utils \
  libcudnn9-dev-cuda-12
```

### Step 2: Clone and Create Virtual Environment

```bash
git clone https://github.com/NVIDIA-AI-IOT/reachy-mini-jetson-assistant
cd reachy-mini-jetson-assistant
python3.10 -m venv venv
source venv/bin/activate
```

### Step 3: Install Python Packages

```bash
pip install --upgrade pip wheel
pip install -r requirements.txt
```

### Step 4: Install ONNX Runtime GPU (Jetson-Specific)

The default `onnxruntime` from pip is CPU-only. For GPU inference (Kokoro TTS, Silero VAD) on Jetson:

```bash
pip install onnxruntime-gpu --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126
```

> If `CUDAExecutionProvider` isn't listed after install, uninstall the CPU version first: `pip uninstall onnxruntime && pip install onnxruntime-gpu --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126`

### Step 5: Install Reachy Mini SDK

```bash
pip install reachy-mini
```

### Step 6: Pin NumPy (Compatibility Fix)

The Jetson `onnxruntime-gpu` wheel requires NumPy 1.x:

```bash
pip install "numpy==1.26.4"
```

### Step 7: Build CTranslate2 with CUDA (GPU-Accelerated STT)

The pip `ctranslate2` package is CPU-only. For GPU-accelerated speech-to-text on Jetson, build from source:

```bash
pip install pybind11

cd ~
git clone --depth 1 https://github.com/OpenNMT/CTranslate2.git
cd CTranslate2
git submodule update --init --recursive

mkdir build && cd build
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
cmake .. -DWITH_CUDA=ON -DWITH_CUDNN=ON -DCMAKE_BUILD_TYPE=Release \
         -DCUDA_ARCH_LIST="8.7" -DOPENMP_RUNTIME=NONE -DWITH_MKL=OFF

make -j$(nproc)
cmake --install . --prefix ~/.local

export LD_LIBRARY_PATH=~/.local/lib:$LD_LIBRARY_PATH
cd ../python
pip install .
```

Persist the library path in your venv activation script:

```bash
echo 'export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH' >> ~/reachy-mini-jetson-assistant/venv/bin/activate
```

### Verify Installation

```bash
source venv/bin/activate
python3 -c "
import ctranslate2; print('CTranslate2 CUDA devices:', ctranslate2.get_cuda_device_count())
import onnxruntime; print('ONNX providers:', onnxruntime.get_available_providers())
from reachy_mini import ReachyMini; print('Reachy Mini SDK: OK')
import faster_whisper; print('faster-whisper: OK')
import kokoro_onnx; print('kokoro-onnx: OK')
"
```

Expected output:
```
CTranslate2 CUDA devices: 1
ONNX providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
Reachy Mini SDK: OK
faster-whisper: OK
kokoro-onnx: OK
```

## Models

### LLM / VLM (served via llama.cpp Docker)

Models download automatically from HuggingFace on first launch. No manual download needed.

| Model | Use | Launch Command |
|-------|-----|----------------|
| Cosmos-Reason2-2B (Q4_K_M) | Vision VLM | `NP=1 ./run_llama_cpp.sh Kbenkhaled/Cosmos-Reason2-2B-GGUF:Q4_K_M` |
| Gemma 3 1B (Q8) | Text LLM | `./run_llama_cpp.sh ggml-org/gemma-3-1b-it-GGUF:Q8_0` |
| bge-small-en-v1.5 (Q8) | RAG embeddings | `./run_llama_embedding.sh ggml-org/bge-small-en-v1.5-Q8_0-GGUF:Q8_0` |

Models are cached in `~/.cache/huggingface` and reused across runs.

### TTS Voices

**Kokoro TTS** (default) downloads automatically on first run (~340 MB). No manual step needed.

To pre-download for offline use:

```bash
wget -P voices/ https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
wget -P voices/ https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

Configure voice in `config/settings.yaml`:

```yaml
tts:
  voice: "af_sarah"    # kokoro voices: af_sarah, af_bella, am_adam, bf_emma, bm_george
```

### Emotion Model

The emotion classifier (DistilBERT SST-2, ~268 MB) downloads automatically on first run. No manual step needed.

To pre-download for offline use:

```bash
mkdir -p models/emotion
wget -O models/emotion/model.onnx \
  "https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/onnx/model.onnx"
wget -O models/emotion/tokenizer.json \
  "https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/onnx/tokenizer.json"
```

## Troubleshooting

**`CUDAExecutionProvider` not available:**
Uninstall CPU onnxruntime and reinstall the GPU version:
```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126
```

**CTranslate2 not finding CUDA:**
Make sure the library path is set: `export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH`

**VLM server not responding:**
Check the Docker container is running: `docker ps`. View logs: `docker logs assistant-llm`

**Process won't exit / robot stays awake after Ctrl+C:**
The app handles Ctrl+C cleanly — the robot should go to sleep. If the process is stuck, run `pkill -9 -f run_web_vision_chat` and `pkill -f reachy-mini-daemon`.

**Port 8090 already in use:**
A previous instance is still running. Kill it: `lsof -ti :8090 | xargs kill -9`

**Camera not found:**
Check the device is available: `ls /dev/video*`. If another process holds it: `fuser -k /dev/video0`

---

## Connection Modes

| Mode | Robot | Host | Camera / Mic / Speaker |
|------|-------|------|------------------------|
| Wired (default for Reachy Mini Lite) | Reachy Mini Lite | Jetson | Local USB (V4L2 / ALSA) |
| Wireless (default everywhere else) | Reachy Mini Wireless (CM4) | Jetson **or** macOS | Robot streams via `robot.media` over WebRTC |
| Wireless on-device | Reachy Mini Wireless (CM4) | The robot's CM4 itself | `robot.media` via GStreamer |

The default in `config/settings.yaml` is wireless against a remote daemon — i.e. the assistant runs on a Jetson or Mac, the daemon (`reachy-mini-daemon` ≥ 1.7.0) runs on the robot. To use a wired Reachy Mini Lite instead, set:

```yaml
reachy:
  wireless: false
  spawn_daemon: true
  media_backend: "no_media"
```

Or pass `--no-wireless` on the command line.

---

## macOS / Apple Silicon

Tested on macOS 14+ with Apple Silicon. macOS only supports the **wireless** connection mode — the assistant connects to the daemon running on the robot.

### Install

```bash
git clone https://github.com/Setlurs/reachy-mini-jetson-assistant
cd reachy-mini-jetson-assistant
./install_mac.sh
```

`install_mac.sh` creates a `venv/`, installs `requirements.txt`, then folds in the macOS dependency workarounds for the Reachy Mini SDK:

* `reachy_mini` is installed `--no-deps` because it pins `libusb_package>=1.0.26.3` which doesn't exist on PyPI (only matters for wired mode anyway).
* `gst-signalling`, `reachy_mini_dances_library`, and `reachy_mini_toolbox` are installed `--no-deps` to bypass overly-strict version caps and avoid scipy source builds.
* The Homebrew Python framework is symlinked into the python.org layout so `gstreamer-bundle`'s `libgstpython.dylib` can find it (sudo prompt during install).

These workarounds came from `saket424/reachy_mini_conversation_app_local`'s `install_mac_wireless.sh`. Re-run `install_mac.sh` after a Python upgrade to refresh the symlink.

### LLM / VLM server (Metal)

```bash
brew install llama.cpp
./run_llama_cpp_mac.sh Kbenkhaled/Cosmos-Reason2-2B-GGUF:Q4_K_M
```

The OpenAI-compatible HTTP surface is the same as the Jetson Docker variant, so `app/llm.py` does not branch.

### What's different vs Jetson

| Thing | Jetson | macOS |
|-------|--------|-------|
| LLM server | `run_llama_cpp.sh` (Docker, CUDA) | `run_llama_cpp_mac.sh` (native, Metal) |
| STT (faster-whisper) | CUDA via CTranslate2 | CPU (CTranslate2 has no Metal target) |
| TTS (Kokoro ONNX) | CUDA Execution Provider | CoreML Execution Provider |
| Mic / camera / speaker | USB ALSA + V4L2 (wired) **or** `robot.media` (wireless) | `robot.media` (wireless only) |
| GPU stats in web UI | nvidia-smi / sysfs | Skipped (Apple GPU not exposed without root) |
| Platform name | "Jetson Orin Nano …" | "MacBook Pro (Apple M3 …)" via `sysctl` |

### macOS troubleshooting

**`Library not loaded: /Library/Frameworks/Python.framework/...`**
Re-run `./install_mac.sh`. The script creates a symlink from that path to the Homebrew Python framework, which `gstreamer-bundle`'s `libgstpython.dylib` was built against.

**WebRTC stays "negotiating" forever / no camera frames**
Verify the `reachy-mini-daemon` is reachable on the network (the Mac and the robot need to be on the same LAN, or you need port-forwarding). Check the daemon logs on the CM4.

**Kokoro TTS falls back to CPU**
That's fine on Apple Silicon, but you can verify CoreML by checking the worker's stderr — it logs the active ONNX provider on startup.

## Home Assistant integration (optional, wireless only)

When `config/settings.yaml::ha.enabled` is `true`, `run_web_vision_chat.py` spawns the
[reachy_mini_home_assistant](https://github.com/ha-china/Reachy_Mini_For_Home_Assistant)
ESPHome voice satellite as a sidecar. Home Assistant auto-discovers the
device via mDNS — no token or URL configuration is required on this side.

### Bring-up

1. Clone the satellite repo somewhere, then install it into the same venv:

   ```bash
   pip install -e /path/to/Reachy_Mini_For_Home_Assistant/
   ```

2. Flip the flag in `config/settings.yaml`:

   ```yaml
   ha:
     enabled: true
     wake_model: "okay_nabu"
   ```

3. Restart `./run`. The console prints `✓ HA satellite (wake: okay_nabu) — watch HA Settings → Devices & Services for auto-discovery`.

4. In Home Assistant: **Settings → Devices & Services → Add Integration → ESPHome**. The Reachy Mini appears in the discovered list; adopt it.

### Notes

- Wireless mode only. The wired Lite SKU has an exclusive local mic and can't share with the satellite.
- The satellite owns its own ReachyMini SDK instance. The Reachy Mini daemon's WebRTC media stream supports both subscribers (our LLM pipeline + the satellite) in parallel.
- Both processes are reaped on Ctrl-C — no orphans, no stuck WebRTC sessions.
- Your existing PTT path on `:8090` and tool calling are unchanged. The satellite handles only utterances that begin with its wake word.

## Voice cloning with XTTS-v2 (optional)

Kokoro is the fast default TTS (~0.2 s time-to-first-audio) and ships
fixed voices. To make the robot speak in *your* voice instead, switch
to the XTTS-v2 backend. Trade-off: ~1.5–2.5 s extra synthesis latency
and ~1.5 GB more RAM, but the voice cloning quality is the headline.

### One-time setup

1. **Capture a reference WAV** of your voice — 6 to 15 seconds of clean
   conversational speech in a quiet room. The
   [rainbow passage](https://en.wikipedia.org/wiki/The_rainbow_passage)
   read naturally is a common choice. Trim to ~10 s, mono, 22050 Hz or
   24000 Hz, save as `voices/clone.wav` (or any path you like).

2. **Install XTTS dependencies** into the same venv. They live in a
   separate requirements file so default installs stay light:

   ```bash
   pip install -r requirements-xtts.txt
   ```

   First synthesis triggers a ~1.5 GB model download into the TTS
   cache — that takes a few minutes. Subsequent runs use the cache.

3. **Flip the backend in `config/settings.yaml`**:

   ```yaml
   tts:
     backend: "xtts"
     xtts_speaker_wav: "voices/clone.wav"
     xtts_language: "en"
     xtts_temperature: 0.7
   ```

4. **Restart `./run`**. You'll see:

   ```
   ✓ TTS (XTTS-v2, clone of clone)
   ```

   on startup. Every TTS chunk now plays in the cloned voice.

### Notes

- If XTTS fails to load (missing deps, missing WAV, model download
  failure), the launcher falls back to Kokoro for the session and
  prints `⚠ XTTS unavailable; falling back to Kokoro` — the demo
  doesn't go silent.
- License: XTTS-v2 model weights are under
  [Coqui's CPML](https://coqui.ai/cpml) (non-commercial). Suitable for
  demos and personal use; consult the model card before any
  commercial deployment.
- Apple Silicon: XTTS works on MPS but some ops fall back to CPU. If
  you hit MPS-specific issues, `XTTS_DISABLE_MPS=1 ./run` pins it to
  CPU.
- To switch back to Kokoro at any time, set `tts.backend: "kokoro"`
  and restart — no rebuild, no other config changes needed.
