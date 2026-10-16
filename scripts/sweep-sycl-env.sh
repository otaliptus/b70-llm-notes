#!/usr/bin/env bash
# llama-bench the SYCL build under the environment toggles that sometimes move decode speed on Intel GPUs.
# Result on the B70 for Bolt IQ2_XXS: all within 29.5-30.2 tok/s (flat). Usage: MODEL=file.gguf ./sweep-sycl-env.sh
set -u
. "$(dirname "$0")/common.sh"
RT=${SYCL_RT:-$LLAMA_RT}; MODEL=${MODEL:?gguf file name under MODELS_DIR}
run() { local label=$1; shift
  echo "=== $label"
  docker run --rm --device /dev/dri -v "$RT:/llama:ro" -v "$MODELS_DIR:/models:ro" -v "$JIT_CACHE:/root/.cache" \
    -e ONEAPI_DEVICE_SELECTOR=level_zero:0 "$@" --entrypoint bash "$SYCL_IMAGE" -lc \
    'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; export LD_LIBRARY_PATH=/llama:$LD_LIBRARY_PATH; /llama/llama-bench -m /models/'"$MODEL"' -ngl 99 -fa 1 -p 512 -n 64 -r 2 -t 16 2>&1 | grep -E "^\|.*(pp512|tg64)|error|Error"'
}
run default
run "Level Zero v1 adapter" -e SYCL_UR_USE_LEVEL_ZERO_V2=0
run "no oneDNN flash-attn" -e GGML_SYCL_FA_ONEDNN=0
run "both" -e SYCL_UR_USE_LEVEL_ZERO_V2=0 -e GGML_SYCL_FA_ONEDNN=0
run "no mmap" -e LLAMA_ARG_NO_MMAP=1
echo "=== sweep done"
