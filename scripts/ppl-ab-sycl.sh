#!/usr/bin/env bash
# Perplexity A/B of two SYCL runtimes on the same text, at ubatch 1 (single-column kernels) and 4 (multi-column
# kernels, the speculative-verify path). Differences beyond float reordering mean a kernel bug.
#   TEXT=corpus.txt MODEL=file.gguf scripts/ppl-ab-sycl.sh /path/runtime-A /path/runtime-B [chunks]
set -u
. "$(dirname "$0")/common.sh"
A=$1; B=$2; CH=${3:-8}; TEXT=${TEXT:?text file}; MODEL=${MODEL:?GGUF file name under MODELS_DIR}
run() { docker run --rm --device /dev/dri --no-healthcheck -v "$1:/llama:ro" -v "$MODELS_DIR:/models:ro" -v "$TEXT:/corpus.txt:ro" -v "$JIT_CACHE:/root/.cache" \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 --entrypoint bash "$SYCL_IMAGE" -lc "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; export LD_LIBRARY_PATH=/llama:\$LD_LIBRARY_PATH; exec /llama/llama-perplexity -m /models/$MODEL -f /corpus.txt -ngl 99 -t 16 -fa 1 -c 512 --chunks $CH -b $2 -ub $2 2>&1" | grep -E "Final estimate"; }
for ub in 1 4; do
  echo "== ub=$ub  $(basename "$A")"; run "$A" $ub
  echo "== ub=$ub  $(basename "$B")"; run "$B" $ub
done
