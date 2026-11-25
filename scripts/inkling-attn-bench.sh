#!/usr/bin/env bash
# Exactness and speed of the nextsycl inkling2 decode attention (silo attn.cpp and the dec.cpp merges) between two
# nextsycl source trees, on synthetic data. Builds inkling-attn-bench.cpp against each tree (AOT for the B70 with the
# silo's own compiler flags), runs the two binaries alternately, compares every case's outputs byte for byte with the
# base tree's, and prints the fastest attention + merge time of each case (microseconds a call).
#
#   SYCL_IMAGE=<oneAPI image that has icpx> scripts/inkling-attn-bench.sh BASE_TREE NEW_TREE [ROUNDS]
#
# ROUNDS (default 2) alternate the order of the two binaries. BENCH_WIDE=1 adds every row count 1-8, the window (1,024)
# and long-path (4,096) boundaries, the windowed ring's wrap (9,215 slots), 9-16 rows and the drafter's block attention.
# NEW_ENV="VAR=VAL ..." is passed to the new tree's runs only (for example NEW_ENV=NS_INK2_ATTN_ROWS=0 with patch 0061).
# WORK (default ./attn-bench) holds the binaries, the outputs and the per-round timings.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
image=${SYCL_IMAGE:?set SYCL_IMAGE to a oneAPI image that has icpx}
base=$(realpath -e "${1:?usage: inkling-attn-bench.sh BASE_TREE NEW_TREE [ROUNDS]}")
new=$(realpath -e "${2:?usage: inkling-attn-bench.sh BASE_TREE NEW_TREE [ROUNDS]}")
rounds=${3:-2}
work=$(realpath -m "${WORK:-attn-bench}")
mkdir -p "$work"
cp "$here/inkling-attn-bench.cpp" "$work/"
user=(--user "$(id -u):$(id -g)" -e HOME=/tmp)
gpu=(--device /dev/dri --group-add "$(stat -c %g /dev/dri/renderD128)" -e ONEAPI_DEVICE_SELECTOR=level_zero:0)

build() { # TREE NAME
    docker run --rm "${user[@]}" -v "$1:/src:ro" -v "$work:/b" -w /b --entrypoint bash "$image" -c '
        source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
        set -e
        S=/src/kernels
        icpx -g0 -fsycl -std=c++20 -O3 -fp-model=precise -fsycl-default-sub-group-size=32 \
            -fsycl-device-code-split=per_kernel -fsycl-targets=spir64_gen \
            -Xsycl-target-backend=spir64_gen "-device bmg-g31 -options -cl-fp32-correctly-rounded-divide-sqrt" \
            -I$S/strata/include-sycl -I$S/strata/include -I$S/ns -I$S/llm/inkling2/silo \
            inkling-attn-bench.cpp $S/llm/inkling2/silo/attn.cpp $S/llm/inkling2/silo/dec.cpp \
            $S/llm/inkling2/silo/mv.cpp -qmkl=sequential -o '"$2"
    echo "built $2 from $1"
}

run() { # NAME ROUND
    local env=()
    [[ -n ${BENCH_WIDE:-} ]] && env+=(-e BENCH_WIDE=1)
    if [[ $1 == new ]]; then for kv in ${NEW_ENV:-}; do env+=(-e "$kv"); done; fi
    mkdir -p "$work/out-$1"
    docker run --rm "${user[@]}" "${gpu[@]}" "${env[@]}" -v "$work:/b" -w /b --entrypoint bash "$image" -c \
        "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; ./$1 out-$1 200" > "$work/t-$1-$2.txt"
}

build "$base" base
build "$new" new
for i in $(seq 1 "$rounds"); do
    if (( i % 2 )); then run base "$i"; run new "$i"; else run new "$i"; run base "$i"; fi
done
diffs=0
for f in "$work"/out-base/*.bin; do cmp -s "$f" "$work/out-new/$(basename "$f")" || diffs=$((diffs + 1)); done
echo "outputs: $(ls "$work"/out-base/*.bin | wc -l) cases, $diffs differ"
python3 - "$work" "$rounds" <<'PY'
import collections, sys
work, n = sys.argv[1], int(sys.argv[2])
def load(arm):
    d = collections.OrderedDict()
    for i in range(1, n + 1):
        for line in open(f"{work}/t-{arm}-{i}.txt"):
            f = line.split()
            if len(f) == 7 and f[0] in ("win", "full", "bwin", "bfull", "wwrap", "bwrap"):
                d.setdefault((f[0], int(f[1]), int(f[2])), []).append(float(f[6]))
    return d
b, w = load("base"), load("new")
print(f"{'case':5s} {'rows':>4s} {'p0':>6s} {'base_us':>9s} {'new_us':>9s}")
for k in b:
    print(f"{k[0]:5s} {k[1]:4d} {k[2]:6d} {min(b[k]):9.1f} {min(w[k]):9.1f}")
PY
