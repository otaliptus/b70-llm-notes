# The benchmark harness (`bench/`)

A small, local, reproducible benchmark aimed at *coding and technical comprehension*, deliberately mid-difficulty
so that 20-80B models separate, with a few tasks written to punish sloppy reading. Everything is stdlib Python plus
Docker for the code sandbox. It talks to any OpenAI-compatible `/v1/chat/completions` endpoint that streams.

## Single-turn suite: 39 tasks (`tasks.py`)
| tier | code (hidden tests) | comprehension | knowledge |
|---|---|---|---|
| base (24) | 10 | 6 | 8 |
| hard (15) | 6 | 4 | 5 |

- **Code**: the model must return one function/class in a code block. The block is extracted, appended to hidden
  asserts, and executed in a locked-down container (`--network none`, read-only, 2 GB, 2 CPUs, nobody user, 90 s).
  `reference.py` holds a passing solution for every task and runs them through the same sandbox, so the tests are
  provably satisfiable.
- **Comprehension**: a log excerpt, a diff, a config, a trace or a table is pasted and a question asked whose answer
  requires actually reading it. Graded by regex (`require` / `accept` / `reject`) or a multiple-choice letter.
- **Knowledge**: concrete technical questions (TCP, HTTP, DNS, TLS 1.3, cron, SQL, Unicode, PID 1, floating point).

`grading.py` strips thinking, extracts the last code block, and never trusts the model's own claims.

## Conversation suites (`convo_tasks.py`, `run_convo.py`)
- **convo**: 8 conversations, 24 graded turns. Code that grows over turns and must keep earlier behaviour, rules
  set in turn 1 checked in turn 4, running state with removals, a pasted file edited two turns later, a bug hunt then
  a refactor, a spec change, a recommendation that must flip when a constraint changes, arithmetic redone after a
  correction. Assistant turns are carried forward as content only (thinking stripped), like a real chat client.
- **repair**: every coding task as an agentic loop. On failure the test output is fed back as the next user turn
  ("fix it"), up to 2 fixes. Records pass@1 and pass@3.
- **long**: the same questions asked at the end of a synthetic prior conversation of 8k / 32k / 64k tokens (doc
  sections + one-line summaries) plus needle questions about facts planted at 10 % / 50 % / 15 %+70 %. Records real
  prompt tokens, time-to-first-token cold and warm (prefix cache), decode rate at depth. The filler text comes from
  `BENCH_LONG_CORPUS` or this repo's `docs/`.

## Scenarios (`matrix.py`)
Thinking budget is the main knob: `b1024`, `b2048`, `b4096` (hard caps), `unlim`, `low` (soft `reasoning_effort`),
`nothink`, plus `b2048-t06` (temperature 0.6), `b2048-rep` (repeat, for the noise floor) and `proj` (Strata's
shipped control vector on). For llama.cpp servers the budget is a server flag, so `scripts/bench-matrix-llamacpp.sh`
restarts the server per scenario; for Strata it is per request and `matrix.py` drives it directly.

## Metrics recorded per task
Verdict and grader note, wall time, time-to-first-token, decode tok/s (from streamed usage), completion tokens,
estimated thinking tokens, finish reason. `cap` in the reports = answers lost because the model was still thinking
at the 16k output limit.

## Running it
```sh
cp bench/profiles.example.json bench/profiles.json   # endpoints, keys, model ids
python3 bench/reference.py                           # sanity: all reference solutions pass
python3 bench/run_bench.py --name mymodel-b2048 --url http://127.0.0.1:18081/v1 --key ... --model ... \
        --temperature 1.0 --top-p 0.95 --top-k 20                  # llama.cpp: budget set on the server
python3 bench/run_bench.py --name fn-b2048 --extra '{"reasoning_budget_tokens": 2048}' ...   # Strata: per request
python3 bench/run_convo.py --suite convo|repair|long --name ... --url ... --model ...
python3 bench/matrix.py --profile sw                 # a whole scenario list for one profile
python3 bench/report.py                              # -> bench/results/report.md
```
Runs resume (tasks already in the result file are skipped) and `--regrade` re-scores stored text answers after a
grader fix without re-querying the model. The sandbox image is `BENCH_SANDBOX_IMAGE` (any image with `python3`).

## Noise floor
Two identical b2048 runs disagreed on 2-4 of 39 tasks for every model tried. Differences that small between any two
rows are not meaningful; differences of 5+ are.
