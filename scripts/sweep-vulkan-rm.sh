#!/usr/bin/env bash
# Sweep the Intel rows-per-workgroup knobs exposed by patches/0002 (from-source Vulkan build, run in the sid container).
# Result on the B70 for Bolt IQ2_XXS: defaults (stdq 2, kq 2) are best at 43.8 tok/s; kq=4 drops to 39.4.
#   LLAMA_RT=/path/to/patched-vulkan-build MODEL=file.gguf ./sweep-vulkan-rm.sh
set -u
. "$(dirname "$0")/common.sh"
MODEL=${MODEL:?gguf file name under MODELS_DIR}
run() { local label=$1; shift
  echo "=== $label"
  docker run --rm --device /dev/dri -v "$LLAMA_RT:/llama:ro" -v "$MODELS_DIR:/models:ro" -e LD_LIBRARY_PATH=/llama "$@" llama-vk-sid \
    /llama/llama-bench -m "/models/$MODEL" -ngl 99 -fa 1 -p 512 -n 64 -r 2 -t 16 2>&1 | grep -E "^\|.*(pp512|tg64)|rror" | sed -E 's/^\| [^|]*\|[^|]*\|[^|]*\| Vulkan +\| +99 \| +//'
}
run "baseline (stdq 2, kq 2)"
run "kq=4" -e GGML_VK_INTEL_RM_KQ=4
run "kq=1" -e GGML_VK_INTEL_RM_KQ=1
run "stdq=4" -e GGML_VK_INTEL_RM_STDQ=4
run "stdq=1" -e GGML_VK_INTEL_RM_STDQ=1
run "rm_id=4" -e GGML_VK_INTEL_RM_ID=4
run "rm_id=2" -e GGML_VK_INTEL_RM_ID=2
run "kq=4 stdq=4 rm_id=4" -e GGML_VK_INTEL_RM_KQ=4 -e GGML_VK_INTEL_RM_STDQ=4 -e GGML_VK_INTEL_RM_ID=4
echo "=== sweep done"
