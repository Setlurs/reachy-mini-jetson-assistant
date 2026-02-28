#!/bin/bash
# Run llama.cpp embedding server (GPU) — wrapper around run_llama_cpp.sh.
#
# Usage:
#   ./run_llama_embedding.sh ggml-org/bge-small-en-v1.5-Q8_0-GGUF:Q8_0
#   ./run_llama_embedding.sh ./models/bge-small-en-v1.5-q8_0.gguf
#
# Stop:
#   docker stop assistant-embed

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${PORT:-8081}" NAME="${NAME:-assistant-embed}" EMBED=1 \
    "$SCRIPT_DIR/run_llama_cpp.sh" "${1:?Usage: $0 <user/repo:quant or path/to/model.gguf>}"
