#!/usr/bin/env bash
# Build llama.cpp (SYCL, fp16) from a source tree inside the oneAPI image, e.g. tag b11696 with patches/0001 and
# patches/0008-0010 applied (git apply patches/00{01,08,09,10}-*.patch). Also builds the tests and batched-bench.
#   SRC=/path/to/llama.cpp OUT=/path/to/output-dir ./build-sycl-patched.sh
# The image must contain icx/icpx, cmake and ninja (oneAPI base-kit images do). -DGGML_SYCL_DNN=OFF because the
# runtime-only image has oneDNN libraries but no headers; oneDNN only affects the flash-attention prefill path.
set -euo pipefail
. "$(dirname "$0")/common.sh"
SRC=${SRC:?source tree}; OUT=${OUT:?output dir}
mkdir -p "$OUT"
docker run --rm -v "$SRC:/src" -v "$OUT:/out" --entrypoint bash "$SYCL_IMAGE" -lc '
  source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
  git config --global --add safe.directory /src; cd /src
  cmake -B build-sycl -G Ninja -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL_F16=ON -DGGML_SYCL_GRAPH=ON \
        -DCMAKE_BUILD_TYPE=Release -DGGML_SYCL_DNN=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_NATIVE=ON
  cmake --build build-sycl --target llama-server llama-bench llama-batched-bench llama-perplexity test-backend-ops -j "$(nproc)"
  cp -a build-sycl/bin/* /out/
  chown -R '"$(id -u):$(id -g)"' /out build-sycl'
echo "=== build done: $OUT"
