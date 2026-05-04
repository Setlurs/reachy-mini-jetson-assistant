#!/usr/bin/env bash
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
# install_mac.sh
#
# Installs reachy-mini-jetson-assistant on macOS (Apple Silicon) for use
# with a Reachy Mini Wireless (CM4). The wireless robot streams
# camera/audio over WebRTC, so this install does NOT need any local USB
# audio or V4L2 support — what we DO need is the GStreamer Python
# bindings reachy_mini uses to talk to the daemon.
#
# Workarounds folded in here come from saket424's reference repo:
#   - reachy_mini pins libusb_package>=1.0.26.3 which doesn't exist on
#     PyPI; we install reachy_mini --no-deps and pull its actual runtime
#     deps separately. (libusb is wired-only, harmless in wireless mode.)
#   - reachy_mini_dances_library / reachy_mini_toolbox have dep chains
#     that try to rebuild scipy from source on Python 3.14; --no-deps
#     skips that since scipy is already a wheel.
#   - gst-signalling has overly strict numpy / PyGObject upper bounds.
#   - gstreamer-bundle's libgstpython.dylib was built against the
#     python.org framework; Homebrew Python lives elsewhere, so we
#     symlink the framework path.
#
# Tested: macOS 14+ (Apple Silicon), Python 3.10–3.14.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d "venv" ]; then
    echo "==> Creating virtualenv (venv/) ..."
    python3 -m venv venv
fi
source venv/bin/activate
echo "==> Python: $(python3 --version) at $(which python3)"

echo "==> Upgrading pip ..."
pip install --upgrade pip wheel

# Project core deps (no Jetson wheels, no onnxruntime-gpu, no faster-whisper
# CUDA build — see SETUP.md for the macOS notes).
echo "==> Installing project requirements ..."
pip install -r requirements.txt

# reachy_mini pins libusb_package>=1.0.26.3, but PyPI only has 1.0.26.1.
# --no-deps lets us install the SDK without that block.
echo "==> Installing reachy_mini (--no-deps) ..."
pip install --upgrade reachy_mini --no-deps

# As of reachy_mini 1.7.x the SDK does an unconditional
# `from libusb_package import get_libusb1_backend` at module-load time
# (audio_control_utils.py), so the package has to be importable even in
# wireless mode where USB is never touched. PyPI's 1.0.26.1 satisfies the
# import — only the version pin is wrong.
echo "==> Installing libusb_package (1.0.26.1; pin in SDK is >=1.0.26.3 but unavailable) ..."
pip install libusb_package

echo "==> Installing reachy_mini runtime deps (excluding libusb_package) ..."
pip install \
    aiohttp \
    asgiref \
    fastapi \
    huggingface-hub \
    jinja2 \
    log-throttling \
    numpy \
    psutil \
    pyserial \
    python-multipart \
    pyusb \
    pyyaml \
    questionary \
    reachy-mini-rust-kinematics \
    reachy_mini_motor_controller \
    requests \
    rich \
    rustypot \
    scipy \
    starlette \
    toml \
    tornado \
    uvicorn \
    websockets \
    zeroconf

echo "==> Installing GStreamer bundle (WebRTC media) ..."
pip install gstreamer-bundle

echo "==> Installing gst-signalling (--no-deps to bypass numpy / PyGObject caps) ..."
pip install "gst-signalling>=1.1.2" --no-deps

echo "==> Installing reachy_mini_dances_library + reachy_mini_toolbox (--no-deps) ..."
pip install reachy_mini_dances_library --no-deps
pip install reachy_mini_toolbox --no-deps

# Optional: the wireless-mode media path also needs aiortc when the SDK
# falls back to its pure-Python WebRTC client. Harmless if unused.
pip install aiortc "eclipse-zenoh~=1.7.0"

# Symlink the Homebrew Python framework into the python.org layout that
# gstreamer-bundle's libgstpython.dylib expects.
PYTHON_MINOR=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
FRAMEWORK_TARGET="/Library/Frameworks/Python.framework/Versions/${PYTHON_MINOR}/Python"
HOMEBREW_FRAMEWORK="/opt/homebrew/Frameworks/Python.framework/Versions/${PYTHON_MINOR}/Python"

if [ ! -f "$FRAMEWORK_TARGET" ] && [ -f "$HOMEBREW_FRAMEWORK" ]; then
    echo "==> Symlinking GStreamer Python dylib (sudo required) ..."
    sudo mkdir -p "$(dirname "$FRAMEWORK_TARGET")"
    sudo ln -s "$HOMEBREW_FRAMEWORK" "$FRAMEWORK_TARGET"
    echo "    Linked $FRAMEWORK_TARGET -> $HOMEBREW_FRAMEWORK"
elif [ -f "$FRAMEWORK_TARGET" ]; then
    echo "==> Python framework dylib already at $FRAMEWORK_TARGET — skipping."
else
    echo "==> WARNING: $HOMEBREW_FRAMEWORK not found."
    echo "   GStreamer's libgstpython.dylib may fail to load."
    echo "   If you installed Python from python.org this is fine."
fi

echo "==> Verifying imports ..."
python3 -c "
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
print('  [OK] GStreamer (gi.repository.Gst)')

from gst_signalling import GstSignallingProducer
print('  [OK] gst-signalling')

from reachy_mini import ReachyMini
print('  [OK] reachy_mini SDK')

from reachy_mini.media.media_manager import MediaBackend
print('  [OK] reachy_mini.media (WebRTC + GStreamer)')
"

echo ""
echo "==> macOS install complete."
echo ""
echo "Next steps:"
echo "  1. Make sure reachy-mini-daemon (>= 1.7.0) is running on the robot."
echo "  2. Start the LLM/VLM server in a second terminal:"
echo "       brew install llama.cpp"
echo "       ./run_llama_cpp_mac.sh Kbenkhaled/Cosmos-Reason2-2B-GGUF:Q4_K_M"
echo "  3. Start the assistant:"
echo "       source venv/bin/activate"
echo "       python3 run_web_vision_chat.py    # defaults to wireless"
echo ""
