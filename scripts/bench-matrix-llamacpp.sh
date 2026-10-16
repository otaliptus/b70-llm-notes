#!/usr/bin/env bash
# Run the whole benchmark matrix against a llama.cpp server whose thinking budget is a *server* flag: restart the
# server per budget. SERVE = one of the serve-*.sh scripts here; the model alias must match --model.
#   SERVE=./serve-bolt-vulkan.sh MODEL_ID=nemotron-3.5-bolt-74b-a9b PREFIX=bolt ./bench-matrix-llamacpp.sh
set -u
. "$(dirname "$0")/common.sh"
HERE=$(cd "$(dirname "$0")" && pwd); BENCH=$HERE/../bench
SERVE=${SERVE:?serve script}; MODEL_ID=${MODEL_ID:?}; PREFIX=${PREFIX:?result-name prefix}
KEY=$([ -n "$API_KEY_FILE" ] && cat "$API_KEY_FILE" || true)
SAMP=(--temperature "${TEMP:-1.0}" --top-p "${TOP_P:-0.95}" --top-k "${TOP_K:-0}")
COMMON=(--url "http://127.0.0.1:$PORT/v1" --key "$KEY" --model "$MODEL_ID")
LOG=$BENCH/results/run_$PREFIX.log
pid=
serve() { setsid nohup "$SERVE" "$@" > "$BENCH/results/$PREFIX-server.log" 2>&1 & pid=$!; }
stop() { [ -n "$pid" ] && kill "$pid" 2>/dev/null; docker stop -t 30 bolt-sycl dense-sycl lfm-sycl >/dev/null 2>&1; sleep 5; pid=; }
run() { local name=$1; shift
  echo "=== $(date '+%H:%M') $name" >> "$LOG"
  python3 "$@" --name "$name" "${COMMON[@]}" "${SAMP[@]}" --wait 900 >> "$BENCH/results/$name.log" 2>&1
  echo "=== $(date '+%H:%M') $name done rc=$?" >> "$LOG"
}
cd "$BENCH"
serve 2048; run $PREFIX-smoke run_bench.py --only c09_lis,k02_http
run $PREFIX-b2048 run_bench.py
run $PREFIX-convo run_convo.py --suite convo
run $PREFIX-repair run_convo.py --suite repair
run $PREFIX-long run_convo.py --suite long
stop
serve 0;    run $PREFIX-nothink run_bench.py; stop
serve 1024; run $PREFIX-b1024 run_bench.py; stop
serve 4096; run $PREFIX-b4096 run_bench.py; stop
serve -1;   run $PREFIX-unlim run_bench.py; stop
python3 report.py > /dev/null
echo "=== $(date '+%F %T') all done" >> "$LOG"
