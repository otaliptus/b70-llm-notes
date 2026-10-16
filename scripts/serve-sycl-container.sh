#!/usr/bin/env bash
# Run an official llama.cpp SYCL release inside a oneAPI runtime container on an Intel Arc GPU (no host oneAPI install).
#
#   serve-sycl-container.sh CONTAINER_NAME HOST_BIND:PORT -- <llama-server args>
#
# Inside the container: runtime at /llama (ro), models at /models (ro), API key at /run/inference.key (ro),
# SYCL JIT cache persisted in $JIT_CACHE so later starts are fast. Only the given host address is published.
# The image needs /opt/intel/oneapi/setvars.sh (any oneAPI base/runtime image works). Level Zero reaches the GPU via /dev/dri.
# Gotcha: images whose ENTRYPOINT is `bash -lc` swallow a plain `docker run image cmd`; always pass --entrypoint bash ... -lc.
set -euo pipefail
. "$(dirname "$0")/common.sh"
name=$1; bind=$2; shift 2
[ "${1:-}" = "--" ] && shift
RT=${SYCL_RT:-$LLAMA_RT}
port=${bind##*:}
mkdir -p "$JIT_CACHE"
docker rm -f "$name" >/dev/null 2>&1 || true   # a container left behind by a killed server
keymount=()
[ -n "$API_KEY_FILE" ] && [ -r "$API_KEY_FILE" ] && keymount=(-v "$API_KEY_FILE:/run/inference.key:ro")
# Not exec'd: a signal to this script stops the container (a signal to the docker client alone does not reliably do that).
trap 'docker stop -t 30 "$name" >/dev/null 2>&1 || true' INT TERM EXIT
docker run --rm --name "$name" --device /dev/dri --stop-timeout 30 --oom-score-adj 1000 --no-healthcheck \
    -p "$bind:$port/tcp" \
    -v "$RT:/llama:ro" -v "$MODELS_DIR:/models:ro" -v "$JIT_CACHE:/root/.cache" "${keymount[@]}" \
    -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
    --entrypoint bash "$SYCL_IMAGE" -lc \
    'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; export LD_LIBRARY_PATH=/llama:$LD_LIBRARY_PATH; exec /llama/llama-server --host 0.0.0.0 "$@"' -- "$@" &
wait $! || true
