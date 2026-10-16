#!/usr/bin/env bash
# Sweep the SYCL backend's runtime knobs with llama-bench (pp512 / tg128) for one GGUF.
#   MODEL=file.gguf scripts/sweep-sycl-knobs.sh   (MODELS_DIR, LLAMA_RT = SYCL runtime, SYCL_IMAGE, JIT_CACHE from common.sh)
# Each line of the sweep is: name, environment to export inside the container, extra llama-bench flags.
set -u
. "$(dirname "$0")/common.sh"
MODEL=${MODEL:?GGUF file name under MODELS_DIR}
COMMON="-m /models/$MODEL -ngl 99 -t 16 -b 2048 -ub 2048 -p 512 -n 128 -r 2 -o md"
run() { local name=$1 envs=$2 flags=$3
  echo "### $name  [$envs] [$flags]"
  docker run --rm --device /dev/dri --no-healthcheck -v "$LLAMA_RT:/llama:ro" -v "$MODELS_DIR:/models:ro" -v "$JIT_CACHE:/root/.cache" \
    -e ONEAPI_DEVICE_SELECTOR=level_zero:0 --entrypoint bash "$SYCL_IMAGE" -lc \
    "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; export LD_LIBRARY_PATH=/llama:\$LD_LIBRARY_PATH; export $envs; exec /llama/llama-bench $COMMON $flags 2>&1" | grep -E "pp512|tg128"
}
KV="-fa 1 -ctk q8_0 -ctv q8_0"
run base         "X=1" "$KV"
run graph        "GGML_SYCL_ENABLE_GRAPH=1" "$KV"
run kv-f16       "X=1" "-fa 1"
run fa-off       "X=1" "-fa 0 -ctk q8_0 -ctv q8_0"
run fa-no-onednn "GGML_SYCL_FA_ONEDNN=0" "$KV"
run fa-no-mkl    "GGML_SYCL_ENABLE_MKL_FA=0" "$KV"
run no-fusion    "GGML_SYCL_ENABLE_FUSION=0" "$KV"
run no-esimd     "GGML_SYCL_ENABLE_ESIMD=0" "$KV"
run mmvq-narrow  "GGML_SYCL_MMVQ_WIDE=0" "$KV"
run dmmv         "GGML_SYCL_PRIORITIZE_DMMV=1" "$KV"
run no-opt       "GGML_SYCL_ENABLE_OPT=0" "$KV"
run prec-bf16    "GGML_SYCL_DYNAMIC_PRECISION=1" "$KV"
run prec-f32     "GGML_SYCL_DYNAMIC_PRECISION=3" "$KV"
run base-d16k    "X=1" "$KV -d 16384 -p 0"
