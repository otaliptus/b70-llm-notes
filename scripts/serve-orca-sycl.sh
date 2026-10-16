#!/usr/bin/env bash
# OrcaSAQ-2-Cyber-27B (dense Qwen3-based, Q4_K_S) on llama.cpp SYCL with the flags that measured best on the B70:
# 262k context with an 8-bit KV cache, flash attention, ubatch 2048, MTP + n-gram drafting (draft length 3,
# probabilistic draft sampling: +17 % decode over greedy drafts at temperature 1.0, exact target distribution).
#   serve-orca-sycl.sh BUDGET [MODEL_FILE]   (BUDGET as in serve-bolt-vulkan.sh; SPEC_ARGS replaces the drafting flags)
set -u
. "$(dirname "$0")/common.sh"
B=${1:--1}; MF=${2:-OrcaSAQ-2-27B-Uncensored-Q4_K_S.gguf}
extra=()
[ "$B" != "-1" ] && extra=(--reasoning-budget "$B" --reasoning-budget-message "Thinking budget reached; answer now with the best current plan.")
keyf=(); [ -n "$API_KEY_FILE" ] && keyf=(--api-key-file /run/inference.key)
exec "$(dirname "$0")/serve-sycl-container.sh" orca-sycl "127.0.0.1:$PORT" -- \
  --model "/models/$MF" --alias "${ALIAS:-OrcaSAQ-2-Cyber-27B}" --offline --no-mmproj --no-agent --no-ui-mcp-proxy --no-webui \
  --gpu-layers 99 --ctx-size "${CTX:-262144}" --parallel 1 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
  --threads 16 --threads-batch 16 --batch-size 2048 --ubatch-size 2048 ${SPEC_ARGS:---spec-type draft-mtp,ngram-mod --spec-draft-n-max 3 --spec-draft-sampling probabilistic} \
  --reasoning on --reasoning-preserve "${extra[@]}" --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --port "$PORT" "${keyf[@]}"
