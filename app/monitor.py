"""System monitor — CPU, RAM, GPU stats for Jetson."""

import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class SystemStats:
    cpu_percent: float
    ram_used_mb: float
    ram_total_mb: float
    ram_percent: float
    gpu_percent: Optional[float] = None


def get_system_stats() -> SystemStats:
    cpu = _cpu()
    used, total, pct = _ram()
    return SystemStats(cpu_percent=cpu, ram_used_mb=used, ram_total_mb=total, ram_percent=pct, gpu_percent=_gpu())


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


def _ram() -> tuple[float, float, float]:
    try:
        import psutil
        m = psutil.virtual_memory()
        return m.used / 1048576, m.total / 1048576, m.percent
    except ImportError:
        try:
            info = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    k, v = line.split()[:2]
                    info[k.rstrip(":")] = int(v)
            total = info.get("MemTotal", 0) / 1024
            used = total - info.get("MemAvailable", 0) / 1024
            return used, total, (used / total * 100) if total else 0
        except Exception:
            return 0.0, 0.0, 0.0


def _gpu() -> Optional[float]:
    try:
        with open("/sys/devices/gpu.0/load") as f:
            return int(f.read().strip()) / 10.0
    except Exception:
        pass
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
