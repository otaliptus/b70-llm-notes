#!/usr/bin/env bash
# Build llama.cpp Vulkan from source inside a Debian sid container (newest Mesa + glslc) and run it in the matching
# runtime container. Needed when the host distro's glibc/Mesa are older than what the build links against.
#   docker build -t llama-vk-sid-build -f mesa-sid/Dockerfile.build mesa-sid
#   docker build -t llama-vk-sid       -f mesa-sid/Dockerfile       mesa-sid
#   SRC=/path/to/llama.cpp OUT=/path/to/output ./build-vulkan-sid.sh
# Run:  docker run --rm --device /dev/dri -v $OUT:/llama:ro -v $MODELS_DIR:/models:ro -e LD_LIBRARY_PATH=/llama llama-vk-sid /llama/llama-bench ...
set -euo pipefail
SRC=${SRC:?source tree}; OUT=${OUT:?output dir}
mkdir -p "$OUT"
docker run --rm -v "$SRC:/src" -v "$OUT:/out" llama-vk-sid-build bash -c '
  git config --global --add safe.directory /src; cd /src
  cmake -B build-vk -G Ninja -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_NATIVE=ON
  cmake --build build-vk --target llama-server llama-bench llama-perplexity -j "$(nproc)"
  cp -a build-vk/bin/* /out/
  chown -R '"$(id -u):$(id -g)"' /out build-vk'
echo "=== vk build done: $OUT"
