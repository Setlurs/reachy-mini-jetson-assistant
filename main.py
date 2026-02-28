#!/usr/bin/env python3
"""
Reachy Mini Jetson Assistant
On-device, GPU-accelerated for NVIDIA Jetson

Usage:
    python main.py chat -t -m /path/to/model.gguf   # Text chat
    python main.py ask "question" -m /path/to/model.gguf  # Single question
    python main.py info  # System info
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.cli import main

if __name__ == "__main__":
    main()
