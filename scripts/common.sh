# Shared defaults for every script here. Override with environment variables.
#   MODELS_DIR    directory holding the GGUF files                       (default /data/models)
#   LLAMA_RT      llama.cpp runtime directory (llama-server, llama-bench, libggml-*.so)
#   SYCL_IMAGE    docker image with the Intel oneAPI runtime (setvars.sh); used by the SYCL scripts
#   API_KEY_FILE  optional file with the API key llama-server should require
#   PORT          port to serve on                                        (default 18081)
#   JIT_CACHE     persistent SYCL JIT cache directory                     (default ~/.cache/llama-sycl-jit)
MODELS_DIR=${MODELS_DIR:-/data/models}
LLAMA_RT=${LLAMA_RT:-$HOME/llama.cpp/runtime-vulkan}
SYCL_IMAGE=${SYCL_IMAGE:-oneapi-runtime}
API_KEY_FILE=${API_KEY_FILE:-}
PORT=${PORT:-18081}
JIT_CACHE=${JIT_CACHE:-$HOME/.cache/llama-sycl-jit}
keyargs() { [ -n "$API_KEY_FILE" ] && [ -r "$API_KEY_FILE" ] && echo --api-key-file "$1"; true; }
