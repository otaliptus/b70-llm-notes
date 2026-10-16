#!/usr/bin/env bash
# Perplexity of several GGUF variants on the same text (24 x 2048-token chunks), to judge a requantization.
#   TEXT=/path/to/corpus.txt ./ppl-compare.sh a.gguf b.gguf ...   (files under MODELS_DIR; Vulkan runtime in LLAMA_RT)
set -u
. "$(dirname "$0")/common.sh"
TEXT=${TEXT:?plain-text corpus}
export LD_LIBRARY_PATH=$LLAMA_RT
for f in "$@"; do
  echo "=== $f"
  "$LLAMA_RT/llama-perplexity" -m "$MODELS_DIR/$f" -f "$TEXT" -ngl 99 -fa on -c 2048 -b 2048 --chunks 24 -t 16 2>&1 | grep -E "Final estimate|rror" | tail -2
done
