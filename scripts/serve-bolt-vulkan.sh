#!/usr/bin/env bash
# Nemotron 3.5 Bolt 74B-A9B on the host Vulkan build (Mesa ANV). This is the fast path on the Arc Pro B70:
# ~42 tok/s plain, ~50 tok/s with the DFlash drafter at draft length 3 (see docs/05-bolt-speedups.md).
#
#   serve-bolt-vulkan.sh BUDGET [dflash|ngram|none] [MODEL_FILE]
#   BUDGET: -1 unlimited thinking, 0 thinking off, N = hard cap of N thinking tokens (llama.cpp --reasoning-budget).
#   DRAFT_N (default 3) = --spec-draft-n-max; CTX (default 98304); EXTRA_ARGS appended verbatim.
set -u
. "$(dirname "$0")/common.sh"
B=${1:--1}; SPEC=${2:-dflash}; MF=${3:-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-IQ2_XXS.gguf}
DRAFT=${DRAFT_FILE:-dflash-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-NVFP4.gguf}
extra=()
[ "$B" != "-1" ] && extra+=(--reasoning-budget "$B" --reasoning-budget-message "Thinking budget reached; answer now with the best current plan.")
case "$SPEC" in
  dflash) extra+=(--spec-type draft-dflash --spec-draft-model "$MODELS_DIR/$DRAFT" --spec-draft-n-max "${DRAFT_N:-3}") ;;
  ngram)  extra+=(--spec-type ngram-mod) ;;
  none)   ;;
esac
export LD_LIBRARY_PATH=$LLAMA_RT
exec "$LLAMA_RT/llama-server" --host 127.0.0.1 --port "$PORT" $(keyargs "$API_KEY_FILE") \
  --model "$MODELS_DIR/$MF" --alias nemotron-3.5-bolt-74b-a9b --offline --no-webui --no-agent --no-ui-mcp-proxy --no-mmproj \
  --gpu-layers 99 --ctx-size "${CTX:-98304}" --parallel 1 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
  --threads 16 --threads-batch 16 --batch-size 2048 --ubatch-size 2048 \
  --reasoning on --reasoning-preserve "${extra[@]}" ${EXTRA_ARGS:-} --temp 1.0 --top-p 0.95
