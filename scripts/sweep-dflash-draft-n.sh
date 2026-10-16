#!/usr/bin/env bash
# Speculative-decoding draft length sweep: serve Bolt with the DFlash drafter at several --spec-draft-n-max values and
# run a 4-task speed subset of the benchmark against each. Prints the server's acceptance lines too.
# Result on the B70 (Vulkan): n_max 3 best (45-58 tok/s), 4 similar, 5 and 2 worse, 7 (upstream default) 27-47.
#   ./sweep-dflash-draft-n.sh [2 3 4 5]
set -u
. "$(dirname "$0")/common.sh"
HERE=$(cd "$(dirname "$0")" && pwd); BENCH=$HERE/../bench
KEY=$([ -n "$API_KEY_FILE" ] && cat "$API_KEY_FILE" || true)
TASKS=c09_lis,r05_table,c01_duration,k07_dns
for n in "${@:-3 4 5}"; do
  rm -f "$BENCH/results/dflash$n.json"
  DRAFT_N=$n setsid nohup "$HERE/serve-bolt-vulkan.sh" 2048 dflash > "/tmp/dflash$n.log" 2>&1 &
  pid=$!
  for i in $(seq 1 150); do curl -sf -H "Authorization: Bearer $KEY" "http://127.0.0.1:$PORT/v1/models" >/dev/null && break; sleep 2; done
  echo "=== dflash n_max=$n up after ${i}x2s"
  python3 "$BENCH/run_bench.py" --name "dflash$n" --url "http://127.0.0.1:$PORT/v1" --key "$KEY" --model nemotron-3.5-bolt-74b-a9b \
    --temperature 1.0 --top-p 0.95 --top-k 0 --only $TASKS 2>&1 | grep -E "PASS|FAIL"
  grep -oE "draft acceptance = [0-9.]+ \([^)]*\), mean len = +[0-9.]+" "/tmp/dflash$n.log" | tail -4
  kill "$pid"; sleep 5
done
echo "=== draft-n sweep done"
