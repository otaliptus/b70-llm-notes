#!/bin/bash
# Reconfigure the from-source Vulkan build with tests on and build test-backend-ops + llama-bench (sid build image).
set -e
cd "$(dirname "$0")"
docker run --rm -v "$PWD/src-b11669:/src" llama-vk-sid-build bash -c '
  git config --global --add safe.directory /src; cd /src
  cmake -B build-vk -G Ninja -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_NATIVE=ON >/dev/null
  cmake --build build-vk --target test-backend-ops llama-bench -j 24 2>&1 | tail -3
  chown -R '"$(id -u):$(id -g)"' build-vk'
echo "=== tests build done"
