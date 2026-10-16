#!/usr/bin/env bash
# Per-operation GPU time for a short decode on the Vulkan backend (GGML_VK_PERF_LOGGER). Tells you which kernels
# eat the token time; see docs/05-bolt-speedups.md for the Bolt profile. Usage: MODEL=file.gguf ./perf-logger-vulkan.sh
set -u
. "$(dirname "$0")/common.sh"
export LD_LIBRARY_PATH=$LLAMA_RT
GGML_VK_PERF_LOGGER=1 GGML_VK_PERF_LOGGER_FREQUENCY=100000 "$LLAMA_RT/llama-bench" -m "$MODELS_DIR/${MODEL:?}" -ngl 99 -fa 1 -p 0 -n 32 -r 1 -t 16 2>&1 | grep -vE '^\s*$' | tail -80
