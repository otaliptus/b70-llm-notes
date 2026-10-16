#!/usr/bin/env bash
# Nemotron 3.5 Bolt on the official llama.cpp SYCL build inside the oneAPI container (~29 tok/s on the B70; kept for
# comparison, the Vulkan path is faster for this model). Same arguments as serve-bolt-vulkan.sh.
#   SYCL_RT = runtime dir for the SYCL build (needs >= b11669 for this architecture).
#   SPEC dflash only works with the patched build (patches/0001, scripts/build-sycl-patched.sh) and is slower than none.
set -u
. "$(dirname "$0")/common.sh"
B=${1:--1}; SPEC=${2:-none}; MF=${3:-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-IQ2_XXS.gguf}
DRAFT=${DRAFT_FILE:-dflash-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-NVFP4.gguf}
extra=()
[ "$B" != "-1" ] && extra+=(--reasoning-budget "$B" --reasoning-budget-message "Thinking budget reached; answer now with the best current plan.")
case "$SPEC" in
  dflash) extra+=(--spec-type draft-dflash --spec-draft-model "/models/$DRAFT" --spec-draft-n-max "${DRAFT_N:-3}") ;;
  ngram)  extra+=(--spec-type ngram-mod) ;;
  none)   ;;
esac
keyf=(); [ -n "$API_KEY_FILE" ] && keyf=(--api-key-file /run/inference.key)
exec "$(dirname "$0")/serve-sycl-container.sh" bolt-sycl "127.0.0.1:$PORT" -- \
  --model "/models/$MF" --alias nemotron-3.5-bolt-74b-a9b --offline --no-webui --no-agent --no-ui-mcp-proxy --no-mmproj \
  --gpu-layers 99 --ctx-size "${CTX:-98304}" --parallel 1 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
  --threads 16 --threads-batch 16 --batch-size 2048 --ubatch-size 2048 \
  --reasoning on --reasoning-preserve "${extra[@]}" ${EXTRA_ARGS:-} --temp 1.0 --top-p 0.95 --port "$PORT" "${keyf[@]}"
