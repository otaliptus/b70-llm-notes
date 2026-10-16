#!/usr/bin/env bash
# Per-kernel timings on the SYCL backend with test-backend-ops perf. With ORCA_SHAPES=1 (patch 0010) only the
# Qwen3.5-27B-class decode shapes run (n = 1..8 for the dominant matrices, delta-net, conv); BOLT_SHAPES=1 (patch 0006)
# gives the Nemotron 3.5 Bolt shapes. Extra arguments go to test-backend-ops (e.g. -o MUL_MAT).
#   ORCA_SHAPES=1 LLAMA_RT=/path/to/dev-runtime scripts/perf-shapes-sycl.sh -o MUL_MAT
# PERF_ENV="A=1 B=2" exports knobs inside the container; PERF_CACHE picks another JIT cache (use one per compile option).
set -u
. "$(dirname "$0")/common.sh"
docker run --rm --device /dev/dri --no-healthcheck -v "$LLAMA_RT:/llama:ro" -v "${PERF_CACHE:-$JIT_CACHE}:/root/.cache" \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 -e ORCA_SHAPES="${ORCA_SHAPES:-}" -e BOLT_SHAPES="${BOLT_SHAPES:-}" --entrypoint bash "$SYCL_IMAGE" -lc \
  "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; export LD_LIBRARY_PATH=/llama:\$LD_LIBRARY_PATH; export ${PERF_ENV:-X=1}; exec /llama/test-backend-ops perf -b SYCL0 $* 2>&1"
