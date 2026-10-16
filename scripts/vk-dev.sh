#!/bin/bash
# Expects the patched llama.cpp tree at ./src-b11669 next to this script (see build-vulkan-sid-tests.sh for the first build).
# Dev loop for the Vulkan backend on the B70: `vk-dev.sh build` (incremental, sid build image), `vk-dev.sh perf [filter]`
# (test-backend-ops perf on the Bolt shapes), `vk-dev.sh check [filter]` (correctness vs CPU), `vk-dev.sh bench [args]` (llama-bench).
set -e
cd "$(dirname "$0")"
BIN=$PWD/src-b11669/build-vk/bin
case "$1" in
  build) docker run --rm -v "$PWD/src-b11669:/src" llama-vk-sid-build bash -c 'git config --global --add safe.directory /src; cd /src && cmake --build build-vk --target test-backend-ops llama-bench llama-server -j 24 2>&1 | grep -E "error|warning: unused|FAILED|Linking" | head -40; chown -R '"$(id -u):$(id -g)"' build-vk' ;;
  perf)  while docker ps --format '{{.Names}}' | grep -q '^vkdev-bench$'; do sleep 5; done
         docker run --rm --device /dev/dri -v "$BIN:/llama:ro" -e LD_LIBRARY_PATH=/llama -e BOLT_SHAPES=1 "${ENV[@]}" ${DOCKER_ENV:-} llama-vk-sid /llama/test-backend-ops perf -b Vulkan0 ${2:+-p "$2"} 2>&1 | grep -E 'us/run|error|rror' ;;
  check) while docker ps --format '{{.Names}}' | grep -q '^vkdev-bench$'; do sleep 5; done
         docker run --rm --device /dev/dri -v "$BIN:/llama:ro" -e LD_LIBRARY_PATH=/llama ${DOCKER_ENV:-} llama-vk-sid /llama/test-backend-ops test -b Vulkan0 -o "${2:-MUL_MAT}" ${3:+-p "$3"} 2>&1 | grep -E 'OK|FAIL|rror|Backend' | grep -vE 'not supported' | tail -${4:-25} ;;
  bench) shift; docker run --rm --name vkdev-bench --device /dev/dri -v "$BIN:/llama:ro" -v /data/models-gguf:/models:ro -e LD_LIBRARY_PATH=/llama ${DOCKER_ENV:-} llama-vk-sid /llama/llama-bench -m /models/${MODEL:-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-IQ2_XXS.gguf} -ngl 99 -fa 1 -p 512 -n 64 -r 2 -t 16 "$@" 2>&1 | grep -E '^\|.*(pp512|tg64)|rror' | sed -E 's/^\| [^|]*\|[^|]*\|[^|]*\| Vulkan +\| +99 \| +//' ;;
esac
