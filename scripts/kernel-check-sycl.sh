#!/usr/bin/env bash
# Correctness of the SYCL MUL_MAT kernels against the CPU backend (test-backend-ops test), for a runtime built with
# the tests (build-sycl-patched.sh). Run after every kernel change; the ncols 1..8 cases cover the multi-column paths.
#   LLAMA_RT=/path/to/dev-runtime scripts/kernel-check-sycl.sh [-o OP]
set -u
. "$(dirname "$0")/common.sh"
docker run --rm --device /dev/dri --no-healthcheck -v "$LLAMA_RT:/llama:ro" -v "$JIT_CACHE:/root/.cache" \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 --entrypoint bash "$SYCL_IMAGE" -lc \
  "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; export LD_LIBRARY_PATH=/llama:\$LD_LIBRARY_PATH; exec /llama/test-backend-ops test -b SYCL0 ${*:--o MUL_MAT} 2>&1" | grep -E "tests passed|FAIL|Backend SYCL0"
