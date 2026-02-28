# Reachy Mini Jetson Assistant

A low-latency, fully on-device voice and vision assistant for [Reachy Mini Lite](https://www.pollen-robotics.com/reachy-mini/) on NVIDIA Jetson. Runs entirely on-device with GPU acceleration — no cloud needed.

## Pipelines

### Voice Chat (text LLM)

```
[Mic] → [Energy VAD] → [faster-whisper STT] → [RAG] → [LLM stream] → [TTS stream] → [Speaker]
```

### Vision Chat (VLM + camera)

```
[Mic] → [Energy VAD] → [faster-whisper STT] ──┐
[USB Cam] → [Frame Capture] ──────────────────┼→ [VLM stream] → [TTS stream] → [Speaker]
```

| Component | Library | Runs On | Notes |
|-----------|---------|---------|-------|
| **LLM** | llama.cpp (Docker) | GPU | OpenAI-compatible API, Gemma 3 1B recommended |
| **VLM** | llama.cpp (Docker) | GPU | Cosmos-Reason2-2B with mmproj, multimodal |
| **STT** | faster-whisper | GPU (CUDA) | CTranslate2 with CUDA, base.en / small.en |
| **TTS** | Kokoro (default) or Piper | GPU (CUDA) / CPU | Kokoro uses ONNX Runtime GPU. Piper is CPU-only fallback. |
| **Camera** | OpenCV (V4L2) / Reachy SDK | CPU | Reachy Mini camera or USB webcam |
| **RAG** | ChromaDB + llama.cpp embeddings | GPU | bge-small-en-v1.5 GGUF embeddings |
| **VAD** | Energy-based (RMS) | CPU | Continuous listen, no wake word needed |
| **Robot** | Reachy Mini SDK | CPU | Head pose, antenna wiggle, wake/sleep |

## Prerequisites

- **NVIDIA Jetson** (tested on Orin Nano 8GB, JetPack 6, CUDA 12.6)
- **Docker** with NVIDIA runtime (`nvidia-container-toolkit`)
- **Reachy Mini Lite** connected via USB-C
- PulseAudio (for mic/speaker multiplexing)

## Hardware Setup

### Reachy Mini Lite

1. Connect the Reachy Mini Lite to your Jetson via USB-C.

2. Add udev rules so the SDK can access the robot's USB serial ports without root:

```bash
echo 'SUBSYSTEM=="tty", ATTRS{idVendor}=="2e8a", ATTRS{idProduct}=="000a", MODE="0666", SYMLINK+="reachy_mini"' \
  | sudo tee /etc/udev/rules.d/99-reachy-mini.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

3. Add your user to the `dialout` group:

```bash
sudo usermod -aG dialout $USER
```

4. Reboot for the group change to take effect, then verify:

```bash
ls -la /dev/ttyACM*
# Should show /dev/ttyACM0, /dev/ttyACM1, etc.
```

### NVMe Swap (recommended for 8GB Jetson)

Running all components simultaneously can exceed 8GB RAM. Setting up swap on NVMe prevents OOM kills:

```bash
sudo fallocate -l 8G /mnt/nvme/swapfile   # adjust path to your NVMe mount
sudo chmod 600 /mnt/nvme/swapfile
sudo mkswap /mnt/nvme/swapfile
sudo swapon /mnt/nvme/swapfile
# Persist across reboots:
echo '/mnt/nvme/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## Installation

### 1. System dependencies

```bash
sudo apt-get update
sudo apt-get install -y \
  python3.10-venv \
  portaudio19-dev \
  libasound2-dev \
  pulseaudio-utils \
  libcudnn9-dev-cuda-12
```

### 2. Clone and create virtual environment

```bash
git clone https://github.com/adsahu-nv/reachy-mini-jetson-assistant.git
cd reachy-mini-jetson-assistant
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python packages

```bash
pip install --upgrade pip wheel
pip install -r requirements.txt
```

### 4. Install ONNX Runtime GPU (Jetson-specific)

The default `onnxruntime` is CPU-only. For GPU inference (Kokoro TTS) on Jetson:

```bash
pip install onnxruntime-gpu --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126
```

> If `CUDAExecutionProvider` isn't listed, uninstall the CPU version first: `pip uninstall onnxruntime`

### 5. Install Reachy Mini SDK

```bash
pip install reachy-mini
```

### 6. Pin numpy (compatibility fix)

The Jetson `onnxruntime-gpu` binary requires numpy 1.x. Pin it after all other installs:

```bash
pip install "numpy==1.26.4"
```

Verify everything loads:

```bash
python3 -c "
import onnxruntime; print('ONNX providers:', onnxruntime.get_available_providers())
from reachy_mini import ReachyMini; print('reachy-mini: OK')
import faster_whisper; print('faster-whisper: OK')
import kokoro_onnx; print('kokoro-onnx: OK')
"
# Should show: CUDAExecutionProvider in ONNX providers
```

### 7. Build CTranslate2 with CUDA (GPU-accelerated STT)

The pip `ctranslate2` package is CPU-only. For GPU on Jetson, build from source:

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

Persist the library path in your venv:

```bash
echo 'export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH' >> venv/bin/activate
```

Verify:

```bash
python3 -c "import ctranslate2; print('CUDA devices:', ctranslate2.get_cuda_device_count())"
# Should show: CUDA devices: 1
```

## Download Models

### LLM / VLM (GGUF — served via Docker)

Models are downloaded automatically by `run_llama_cpp.sh` from HuggingFace on first use. No manual download needed.

Recommended models:

| Model | Use | Command |
|-------|-----|---------|
| Gemma 3 1B (Q8) | Text LLM | `./run_llama_cpp.sh ggml-org/gemma-3-1b-it-GGUF:Q8_0` |
| Cosmos-Reason2-2B (Q4_K_M) | Vision VLM | `./run_llama_cpp.sh Kbenkhaled/Cosmos-Reason2-2B-GGUF:Q4_K_M` |
| bge-small-en-v1.5 (Q8) | RAG embeddings | `./run_llama_embedding.sh ggml-org/bge-small-en-v1.5-Q8_0-GGUF:Q8_0` |

Models are cached in `~/.cache/huggingface` and reused across runs.

### TTS voice models

Download to the `voices/` directory:

```bash
# Kokoro TTS (default — higher quality, ~340 MB total)
wget -P voices/ https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
wget -P voices/ https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin

# Piper TTS (lighter alternative, ~61 MB)
wget -O voices/en_US-lessac-medium.onnx \
  "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
```

Switch TTS backend in `config/settings.yaml`:

```yaml
tts:
  backend: "kokoro"    # or "piper"
  voice: "af_sarah"    # kokoro voices: af_sarah, af_bella, am_adam, bf_emma, bm_george
```

## Running

### Step 1: Start the LLM server

```bash
./run_llama_cpp.sh ggml-org/gemma-3-1b-it-GGUF:Q8_0
```

### Step 2: Start the embedding server (for RAG)

```bash
./run_llama_embedding.sh ggml-org/bge-small-en-v1.5-Q8_0-GGUF:Q8_0
```

### Step 3: Run the voice assistant

```bash
source venv/bin/activate
python3 run_voice_chat.py           # with RAG
python3 run_voice_chat.py --no-rag  # without RAG
```

### Vision chat (camera + VLM)

Start the VLM server (mmproj auto-downloads):

```bash
NP=1 ./run_llama_cpp.sh Kbenkhaled/Cosmos-Reason2-2B-GGUF:Q4_K_M
```

Run the vision assistant:

```bash
source venv/bin/activate
python3 run_vision_chat.py
```

### Text-only chat (no mic/speaker)

```bash
python3 main.py chat -t
```

### Other CLI commands

```bash
python3 main.py ask "What is the Jetson Orin Nano?"
python3 main.py info
python3 main.py rag-status
python3 main.py rag-search "Jetson GPU specs"
```

### Test robot movement

```bash
python3 test_reachy_movement.py
```

## Configuration

All settings in `config/settings.yaml`. Key sections:

| Section | What it controls |
|---------|-----------------|
| `llm` | LLM server URL, model, temperature, max tokens, system prompt |
| `stt` | Whisper model size, CUDA device, beam size |
| `tts` | Backend (kokoro/piper), voice, speed |
| `audio` | Sample rate, input device name |
| `vision` | Camera device, resolution, frames per utterance |
| `rag` | Embedding backend, knowledge directory, chunk settings |

## Project Structure

```
reachy-mini-jetson-assistant/
├── app/
│   ├── pipeline.py      # Shared audio I/O, VAD, TTS streaming, mic recording
│   ├── config.py        # Configuration dataclasses + YAML loader
│   ├── llm.py           # LLM/VLM client (OpenAI-compatible, multimodal)
│   ├── stt.py           # faster-whisper STT
│   ├── tts.py           # TTS backends (Kokoro GPU / Piper CPU)
│   ├── camera.py        # USB webcam capture (OpenCV, V4L2, Reachy SDK)
│   ├── rag.py           # ChromaDB + embeddings + retriever
│   ├── audio.py         # PulseAudio/ALSA device helpers
│   ├── monitor.py       # System resource monitor
│   └── cli.py           # Typer CLI (chat, ask, rag-*)
├── config/
│   └── settings.yaml    # All configuration
├── knowledge_base/
│   └── jetson_orin_nano.md
├── models/              # GGUF models (gitignored)
├── voices/              # TTS voice models (large files gitignored)
├── run_voice_chat.py    # Voice chat entry point
├── run_vision_chat.py   # Vision chat entry point (VLM + camera)
├── test_reachy_movement.py  # Robot movement test
├── run_llama_cpp.sh     # Docker LLM server launcher
├── run_llama_embedding.sh   # Docker embedding server launcher
├── main.py              # CLI entry point
├── notes.md             # Benchmarks and performance notes
└── requirements.txt
```

## Stopping

```bash
docker stop assistant-llm assistant-embed
```

## License

Apache-2.0
