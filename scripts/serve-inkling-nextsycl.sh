#!/usr/bin/env bash
# Single-stream B70 decode of the AtomicChat Inkling checkpoint.
set -euo pipefail
if [[ ${1:-} == --help ]]; then
    echo 'Set RUNTIME, MODELS and API_KEY_FILE, then run serve-inkling-nextsycl.sh.'
    echo 'RUNTIME is a directory containing nextsycl, libnextsycl-llm.so and silo/libnextsycl-inkling2.so.'
    echo 'INKLING_NATIVE_PORT defaults to 18081. CTX defaults to 131072. SYCL_IMAGE selects the oneAPI runtime image.'
    echo 'WIDE_PREFILL defaults to 1 and lets the runtime select a memory-fitting prompt batch. Set it to 0 for the original policy.'
    echo 'PREFILL_CHUNK pins the prompt batch size explicitly.'
    echo 'Set DRAFT_WINDOW=2048 with runtime patch 0060 to limit long-context proposal attention. The default is 0.'
    echo 'KV defaults to f16 (bit-exact); set KV=q8 with patch 0063 for symmetric INT8 full-attention KV.'
    echo 'PROMPT_CACHE_MIB sets the host checkpoint cache budget in MiB and defaults to 6144.'
    exit 0
fi
[[ $# == 0 ]] || { echo 'See --help for usage.' >&2; exit 2; }
runtime=$(realpath -e "${RUNTIME:?set RUNTIME to the native runtime directory}")
models=$(realpath -e "${MODELS:?set MODELS to the directory containing the GGUF files}")
key_file=$(realpath -e "${API_KEY_FILE:?set API_KEY_FILE to a bearer key file}")
target="$models/Inkling-2-Mini-Uncensored-AD-Q4_K.gguf"
draft="$models/dspark-Inkling-2-Mini-Q8_0-sub32k-unc.gguf"
port=${INKLING_NATIVE_PORT:-18081}
context=${CTX:-131072}
for file in "$runtime/nextsycl" "$runtime/libnextsycl-llm.so" "$runtime/silo/libnextsycl-inkling2.so" "$target" "$draft" "$key_file"; do
    [[ -s "$file" ]] || { echo "Missing file: $file" >&2; exit 1; }
done
exec docker run --rm --init --name "inkling-unc-native-$port" --stop-timeout 10 \
    --device /dev/dri/renderD128 -p "127.0.0.1:$port:18081" \
    --mount "type=bind,src=$runtime,dst=/runtime,readonly" \
    --mount "type=bind,src=$target,dst=/models/inkling.gguf,readonly" \
    --mount "type=bind,src=$draft,dst=/models/dspark.gguf,readonly" \
    --mount "type=bind,src=$key_file,dst=/run/inference.key,readonly" \
    -e ONEAPI_DEVICE_SELECTOR=level_zero:0 -e NS_IN_IMAGE=1 -e NS_LIB_FROM_DISK=1 \
    -e NEXTSYCL_LIB=/runtime/libnextsycl-llm.so -e NS_SILO_DIR=/runtime/silo -e NS_SILO_REQUIRED=1 \
    -e NS_INK2_DRAFT_WINDOW="${DRAFT_WINDOW:-0}" \
    -e NS_INK2_KV="${KV:-f16}" \
    -e NS_INK2_WIDE_PREFILL="${WIDE_PREFILL:-1}" -e NS_INK2_CHUNK="${PREFILL_CHUNK:-}" \
    -e NS_API_KEY_FILE=/run/inference.key -e NS_FULL_VOCAB=1 \
    -e NS_INK2_DRAFT=/models/dspark.gguf -e NS_INK2_DRAFTS=6 -e NS_INK2_DRAFT_PMIN=0.65 \
    --entrypoint bash "${SYCL_IMAGE:-strata-sycl-dev}" -lc '
        source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
        exec /runtime/nextsycl serve /models/inkling.gguf --gpu 0 --name inkling-2-mini \
            --ctx "$1" --prompt-cache-mib "$2" --parallel 1 --host 0.0.0.0 --port 18081 --effort low
    ' -- "$context" "${PROMPT_CACHE_MIB:-6144}"
