#!/usr/bin/env bash
# LFM2.5-24B-A2B (Q6_K) on llama.cpp SYCL: ~128 tok/s decode on the B70. Vendor sampling is temp 0.1 / top-p 1 / top-k 50 /
# repeat 1.05 (set by the client). It is an agent model: without a system prompt saying "no tools" it may emit tool calls.
#   serve-lfm-sycl.sh BUDGET [MODEL_FILE]
set -u
. "$(dirname "$0")/common.sh"
B=${1:--1}; MF=${2:-LFM2.5-24B-A2B-Q6_K.gguf}
extra=()
[ "$B" != "-1" ] && extra=(--reasoning-budget "$B" --reasoning-budget-message "Thinking budget reached; answer now with the best current plan.")
keyf=(); [ -n "$API_KEY_FILE" ] && keyf=(--api-key-file /run/inference.key)
exec "$(dirname "$0")/serve-sycl-container.sh" lfm-sycl "127.0.0.1:$PORT" -- \
  --model "/models/$MF" --alias lfm2.5-24b-a2b --offline --no-webui --no-agent --no-ui-mcp-proxy \
  --gpu-layers 99 --ctx-size "${CTX:-98304}" --parallel 1 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
  --threads 16 --threads-batch 16 --batch-size 2048 --ubatch-size 2048 --spec-type ngram-mod \
  --reasoning on "${extra[@]}" --port "$PORT" "${keyf[@]}"
