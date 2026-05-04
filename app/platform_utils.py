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

"""Host platform detection helpers.

These are deliberately small and orthogonal to the wired/wireless
connection mode. The wired/wireless split lives in app.reachy and the
Camera/MicRecorder factories; this module only knows about the host OS.
Used to pick acceleration providers (CUDA vs CoreML), label the platform
in the web UI, and skip Linux-only syscalls (libasound, fuser).
"""

import platform
import sys
from functools import lru_cache


@lru_cache(maxsize=1)
def is_macos() -> bool:
    return sys.platform == "darwin"


@lru_cache(maxsize=1)
def is_linux() -> bool:
    return sys.platform.startswith("linux")


@lru_cache(maxsize=1)
def is_apple_silicon() -> bool:
    return is_macos() and platform.machine().lower() in ("arm64", "aarch64")


@lru_cache(maxsize=1)
def is_jetson() -> bool:
    """True if running on an NVIDIA Jetson (detected via device-tree)."""
    if not is_linux():
        return False
    try:
        with open("/proc/device-tree/model") as f:
            text = f.read().lower()
        return "jetson" in text or "tegra" in text
    except Exception:
        return False
