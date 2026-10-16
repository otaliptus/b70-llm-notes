#!/usr/bin/env bash
# llama-bench the Vulkan build under every GGML_VK_* toggle plus KV type / ubatch / threads variations.
# Result on the B70 for Bolt IQ2_XXS: decode 41.7-42.1 tok/s for every row (flat); only prefill reacts
# (coopmat off: 777 -> 156 pp512, f16 off: -> 411). Usage: MODEL=file.gguf ./sweep-vulkan-env.sh
set -u
. "$(dirname "$0")/common.sh"
M=$MODELS_DIR/${MODEL:?gguf file name under MODELS_DIR}
export LD_LIBRARY_PATH=$LLAMA_RT
run() { local label=$1; shift
  echo "=== $label"
  "$LLAMA_RT/llama-bench" -m "$M" -ngl 99 -p 512 -n 64 -r 2 -t 16 "$@" 2>&1 | grep -E "^\|.*(pp512|tg64)|ggml_vulkan: 0 =|error" | sed -E 's/^\| [^|]*\|[^|]*\|[^|]*\| Vulkan +\| +99 \| +//'
}
run "baseline fa=1 f16 kv" -fa 1
run "fa=0" -fa 0
run "q8 kv" -fa 1 -ctk q8_0 -ctv q8_0
run "ub 1024" -fa 1 -ub 1024
run "ub 2048 b 2048" -fa 1 -ub 2048 -b 2048
run "threads 8" -fa 1 -t 8
GGML_VK_DISABLE_COOPMAT=1 run "no coopmat" -fa 1
GGML_VK_DISABLE_COOPMAT2=1 run "no coopmat2" -fa 1
GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1 run "no int dot" -fa 1
GGML_VK_FORCE_MMVQ=1 run "force mmvq" -fa 1
GGML_VK_DISABLE_MMVQ=1 run "disable mmvq" -fa 1
GGML_VK_DISABLE_F16=1 run "no f16" -fa 1
GGML_VK_DISABLE_FUSION=1 run "no fusion" -fa 1
GGML_VK_DISABLE_GRAPH_OPTIMIZE=1 run "no graph opt" -fa 1
GGML_VK_DISABLE_DOT2=1 run "no dot2" -fa 1
GGML_VK_PREFER_HOST_MEMORY=1 run "prefer host mem" -fa 1
echo "=== sweep done"
