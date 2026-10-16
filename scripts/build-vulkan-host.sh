#!/bin/bash
# Build the patched llama.cpp Vulkan tree in a trixie (host-glibc) image -> runtime-vulkan-src-host/ runs on the host.
set -e
cd "$(dirname "$0")"
OUT=${OUT:-runtime-vulkan-src-host}
if pgrep -f "$PWD/$OUT/llama-server" >/dev/null; then
  echo "a llama-server from $OUT is running; refusing to overwrite its libraries (set OUT=other-dir)"; exit 1
fi
docker image inspect llama-vk-trixie-build >/dev/null 2>&1 || docker build -t llama-vk-trixie-build -f mesa-trixie/Dockerfile.build mesa-trixie
docker run --rm -v "$PWD/src-b11669:/src" -v "$PWD:/out" llama-vk-trixie-build bash -c '
  git config --global --add safe.directory /src; cd /src
  cmake -B build-vk-host -G Ninja -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_NATIVE=ON > /tmp/cmake.log 2>&1 || { tail -20 /tmp/cmake.log; exit 1; }
  grep -E "GLSLC_SUPPORT|glslc" build-vk-host/CMakeCache.txt | head -5
  cmake --build build-vk-host --target llama-server llama-bench test-backend-ops llama-perplexity -j 24 2>&1 | grep -E "error|FAILED|Linking CXX executable" | head -20
  mkdir -p /out/'"$OUT"' && cp -a build-vk-host/bin/* /out/'"$OUT"'/
  chown -R '"$(id -u):$(id -g)"' /out/'"$OUT"' build-vk-host'
echo "=== trixie host build done -> $OUT"
