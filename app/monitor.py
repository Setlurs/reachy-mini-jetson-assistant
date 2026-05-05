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

"""System monitor — CPU, RAM, GPU stats. Linux/Jetson + macOS."""

import subprocess
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from app.platform_utils import is_macos


@dataclass
class SystemStats:
    cpu_percent: float
    ram_used_mb: float
    ram_total_mb: float
    ram_percent: float
    # `available` is psutil's "memory that can be given to processes without
    # swapping" — on macOS this excludes the file-system cache, so it tracks
    # real headroom rather than the inflated `used` figure that includes
    # reclaimable cache.
    ram_available_mb: float = 0.0
    disk_used_gb: float = 0.0
    disk_total_gb: float = 0.0
    disk_percent: float = 0.0
    gpu_percent: Optional[float] = None


def get_system_stats() -> SystemStats:
    cpu = _cpu()
    used, total, pct, avail = _ram()
    d_used, d_total, d_pct = _disk()
    return SystemStats(
        cpu_percent=cpu,
        ram_used_mb=used,
        ram_total_mb=total,
        ram_percent=pct,
        ram_available_mb=avail,
        disk_used_gb=d_used,
        disk_total_gb=d_total,
        disk_percent=d_pct,
        gpu_percent=_gpu(),
    )


def format_stats(s: SystemStats) -> str:
    parts = [f"CPU: {s.cpu_percent:.1f}%", f"RAM: {s.ram_used_mb:.0f}/{s.ram_total_mb:.0f}MB ({s.ram_percent:.1f}%)"]
    if s.gpu_percent is not None:
        parts.append(f"GPU: {s.gpu_percent:.1f}%")
    return " | ".join(parts)


def _cpu() -> float:
    try:
        import psutil
        return psutil.cpu_percent(interval=0.1)
    except ImportError:
        try:
            with open("/proc/stat") as f:
                p = f.readline().split()
            idle, total = int(p[4]), sum(int(x) for x in p[1:])
            return 100.0 * (1 - idle / total) if total else 0.0
        except Exception:
            return 0.0


def _disk() -> tuple[float, float, float]:
    """Root-filesystem usage in GB and a percent. Falls back to (0,0,0)
    on platforms without psutil and without `shutil.disk_usage` access."""
    try:
        import shutil
        u = shutil.disk_usage("/")
        gb = 1024 ** 3
        used_gb = u.used / gb
        total_gb = u.total / gb
        pct = (u.used / u.total * 100.0) if u.total else 0.0
        return used_gb, total_gb, pct
    except Exception:
        return 0.0, 0.0, 0.0


def _ram() -> tuple[float, float, float, float]:
    try:
        import psutil
        m = psutil.virtual_memory()
        avail = getattr(m, "available", m.total - m.used)
        return m.used / 1048576, m.total / 1048576, m.percent, avail / 1048576
    except ImportError:
        try:
            info = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    k, v = line.split()[:2]
                    info[k.rstrip(":")] = int(v)
            total = info.get("MemTotal", 0) / 1024
            avail = info.get("MemAvailable", 0) / 1024
            used = total - avail
            return used, total, (used / total * 100) if total else 0, avail
        except Exception:
            return 0.0, 0.0, 0.0, 0.0


_GPU_SYSFS_PATHS = [
    "/sys/devices/platform/gpu.0/load",
    "/sys/devices/platform/17000000.gpu/load",
    "/sys/devices/gpu.0/load",
]


@lru_cache(maxsize=1)
def get_jetson_model() -> str:
    """Return a clean platform name.

    On Jetson: 'Jetson Orin Nano Super' from /proc/device-tree.
    On macOS:  e.g. 'MacBook Pro (Apple M3 Pro)' from sysctl.
    Otherwise: 'Jetson' as a last-resort fallback (kept for backwards
    compatibility with code that imports this name).
    """
    if is_macos():
        return _macos_model()
    try:
        with open("/proc/device-tree/model") as f:
            raw = f.read().strip().rstrip("\x00")
        name = raw.replace("NVIDIA ", "").replace(" Engineering Reference Developer Kit", "")
        return name
    except Exception:
        return "Jetson"


def get_platform_name() -> str:
    """Friendlier alias; same as get_jetson_model()."""
    return get_jetson_model()


def _macos_model() -> str:
    try:
        model = subprocess.run(
            ["sysctl", "-n", "hw.model"],
            capture_output=True, text=True, timeout=2,
        ).stdout.strip()
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=2,
        ).stdout.strip()
        if model and chip:
            return f"{model} ({chip})"
        return model or chip or "Mac"
    except Exception:
        return "Mac"


def _gpu() -> Optional[float]:
    # Apple Silicon GPU usage isn't accessible without `powermetrics` (root).
    # Skip on macOS; the UI handles a missing value gracefully.
    if is_macos():
        return None
    for path in _GPU_SYSFS_PATHS:
        try:
            with open(path) as f:
                return int(f.read().strip()) / 10.0
        except Exception:
            continue
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        )
        if r.returncode == 0:
            return float(r.stdout.strip())
    except Exception:
        pass
    return None
