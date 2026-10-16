#!/usr/bin/env bash
# Speculative-decoding sweep for a llama.cpp server: restart the server per configuration, run the benchmark's 4-task
# speed subset, report median decode tok/s and the server's draft-acceptance lines.
#   SERVE="scripts/serve-orca-sycl.sh 2048" MODEL_ID=OrcaSAQ-2-Cyber-27B scripts/spec-sweep-llamacpp.sh
# SERVE must accept extra llama-server arguments after its own (serve-orca-sycl.sh does: SPEC_ARGS env).
set -u
. "$(dirname "$0")/common.sh"
HERE=$(cd "$(dirname "$0")/.." && pwd)
SERVE=${SERVE:?server command}; MODEL_ID=${MODEL_ID:?model id}
KEY=$( [ -n "$API_KEY_FILE" ] && cat "$API_KEY_FILE" || echo none )
TASKS=c09_lis,r05_table,c01_duration,k07_dns
LOG=${LOG:-spec-sweep.log}
cfg() { local name=$1; shift
  echo "### $(date '+%H:%M:%S') $name  [$*]" | tee -a "$LOG"
  (SPEC_ARGS="$*" setsid nohup $SERVE > "server-$name.log" 2>&1 &)
  for i in $(seq 1 150); do curl -sf -H "Authorization: Bearer $KEY" "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break; sleep 2; done
  rm -f "$HERE/bench/results/spec-$name.json"
  python3 "$HERE/bench/run_bench.py" --name "spec-$name" --url "http://127.0.0.1:$PORT/v1" --key "$KEY" --model "$MODEL_ID" \
     --temperature 1.0 --top-p 0.95 --top-k 20 --only $TASKS --wait 120 2>&1 | grep -E 'PASS|FAIL|never' | tee -a "$LOG"
  python3 - "$HERE/bench/results/spec-$name.json" <<'PY' | tee -a "$LOG"
import json,sys,statistics
d=json.load(open(sys.argv[1])); sp=[v["decode_tps"] for k,v in d.items() if k!="_meta" and isinstance(v,dict) and v.get("decode_tps")]
print(f"median tok/s {statistics.median(sp):.1f}  mean {statistics.mean(sp):.1f}  n={len(sp)}")
PY
  grep -h "draft acceptance" "server-$name.log" | tail -4 | tee -a "$LOG"
  pkill -INT -f "llama-server.*--port $PORT" 2>/dev/null; sleep 5
}
cfg s3-greedy  --spec-type draft-mtp,ngram-mod --spec-draft-n-max 3 --spec-draft-sampling greedy
cfg s3-prob    --spec-type draft-mtp,ngram-mod --spec-draft-n-max 3 --spec-draft-sampling probabilistic
cfg s4-prob    --spec-type draft-mtp,ngram-mod --spec-draft-n-max 4 --spec-draft-sampling probabilistic
cfg s2-prob    --spec-type draft-mtp,ngram-mod --spec-draft-n-max 2 --spec-draft-sampling probabilistic
cfg mtp3-only  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-sampling probabilistic
cfg nospec     --spec-type none
