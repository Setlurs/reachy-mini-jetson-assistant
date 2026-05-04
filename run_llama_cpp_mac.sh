#!/bin/bash
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

# Run llama.cpp server on macOS (Apple Silicon, Metal-accelerated).
#
# Unlike run_llama_cpp.sh (which uses NVIDIA Docker on Jetson), this calls
# the native llama-server binary that ships with Homebrew's llama.cpp,
# which is built with Metal support. The OpenAI-compatible HTTP surface
# is identical, so app/llm.py needs no changes.
#
# Prereq:
#   brew install llama.cpp
#
# Usage:
#   ./run_llama_cpp_mac.sh ggml-org/gemma-3-1b-it-GGUF:Q8_0
#   ./run_llama_cpp_mac.sh Kbenkhaled/Cosmos-Reason2-2B-GGUF:Q4_K_M
#   ./run_llama_cpp_mac.sh ./models/gemma-3-1b-it-Q8_0.gguf
#
# Options (env vars):
#   PORT=8080  CTX=4096  NP=1  EMBED=1
#
# Stop with Ctrl-C (the server runs in the foreground).

set -e

if ! command -v llama-server > /dev/null 2>&1; then
    echo "llama-server not found. Install with: brew install llama.cpp" >&2
    exit 1
fi

MODEL="${1:?Usage: $0 <user/repo:quant or path/to/model.gguf>}"
PORT="${PORT:-8080}"
CTX="${CTX:-4096}"
NP="${NP:-1}"

EXTRA_ARGS=()
if [ "${EMBED:-0}" = "1" ]; then
    EXTRA_ARGS+=("--embeddings")
fi

# -ngl 99 offloads everything possible to Metal. Flash attention and KV
# cache reuse mirror the Jetson launcher.
COMMON_ARGS=(
    --host 0.0.0.0 --port "$PORT"
    -ngl 99 -c "$CTX" -np "$NP"
    -fa on --cache-reuse 256
)

if [ -f "$MODEL" ]; then
    echo "Model : $MODEL (local)"
    echo "Port  : $PORT"
    echo ""
    exec llama-server -m "$MODEL" "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"
else
    echo "Model : $MODEL (HuggingFace)"
    echo "Port  : $PORT"
    echo ""
    exec llama-server -hf "$MODEL" "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"
fi
