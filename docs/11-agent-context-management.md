# Keeping a local agent fast over long sessions

The decode speed of every model here falls with context depth (Orca 27B: 28 tok/s shallow, 13 at 64k, 9 at 128k;
prefill 1,259 / 682 / 527). A harness that lets the conversation grow to half the window pays that tax on every
token of the second half of a session. None of the kernel work in 09 / 10 recovers as much as not being deep in the
first place. What the lab harness does now (plain-stdlib Python agent talking to an OpenAI-compatible server):

- **Compact at ~48k tokens, not at half the window.** The agent asks the model for a terse handoff note (goals,
  decisions, files touched with paths, commands and results, open problems, next step) and continues from the note.
  The threshold is a flag; 24k / 32k / 48k are all reasonable and the right one depends on how often the task
  needs to look back. Measure retained accuracy with a long-session task list, not just speed.
- **Durable notes.** Every handoff note is also appended to `<workspace>/.lab/notes.md`, so requirements and
  decisions survive further compactions and sessions, and the model can `read_file` the history when it needs it.
- **Full tool logs on disk, excerpts in context.** A tool result longer than the kept head + tail (12k chars) is
  saved whole under `<workspace>/.lab/tool-logs/NNNN-<tool>.txt`, and the truncated result says so; `read_file`
  with a line range retrieves the part that matters. Same for over-long file reads (the reply says how many lines the
  file has and asks for a narrower range).
- **Stable prompt prefixes.** The server reuses the KV cache for an unchanged prefix; rewriting early messages on
  every step (trimming old tool outputs, injecting stats) forces a full re-read. The harness touches history only
  when over budget (trim) or at compaction, and never edits the system prompt.
- **Per-request thinking budgets.** llama-server (>= b11696) and Strata accept `reasoning_budget_tokens` in the
  request, so the cap no longer has to be global: a smaller cap for tool-result continuations, a larger one for user
  turns. In the harness: `--think-budget-user N`, `--think-budget-tool N` (default: server setting). Judge by
  completed tasks per minute including retries, not by tokens per second (see 10, section 5: on the repair suite
  512 reached the same final pass count as 2048 in less than half the wall time; repeats pending).
- **Count what costs time.** Tool calls, malformed tool-call arguments (recovered or not), tool errors (non-zero exit,
  timeouts), long outputs spilled, compactions, per-turn wall. A scripted mode (`--script FILE` + `--stats-json`)
  replays the same user turns non-interactively, so two runtimes or two settings can be compared from the same VM
  snapshot and task list; comparing live sessions with different commands and filesystem state measures the
  session, not the change.

## Mid-task evidence ledger (2026-11-23)

Compaction between turns does not help a single long instruction such as "review this repository for bugs": the whole
review is one user turn with hundreds of tool calls. The harness then only shortened older tool outputs to 1,200
characters (its "drop the oldest exchange" step needs an earlier user message, which one review request does not have),
so the context kept growing, the middle of every older tool output was lost, and nothing was written down. Its budget
check also lagged a step behind: it used the server's token count from the previous request, so eight files read in
one step went ~7k tokens over budget unnoticed.

What the harness does now when a turn crosses its budget:

- The model writes an **evidence ledger** with fixed sections: confirmed findings (file, function or line, failure
  condition, the code or output that proves it), ruled out (with the reason, so it is not investigated again), open
  questions (with what would settle them), covered (files, areas, commands and key results), next steps. At most 900
  words; earlier ledgers are merged, not repeated.
- The ledger is appended to `<workspace>/.lab/notes.md`; the tool exchanges that leave the context are archived whole
  in `.lab/tool-logs/turn-<session>-ledgerN.md`, so `read_file` can bring any of them back.
- The context becomes: system prompt, earlier turns unchanged, the current request verbatim plus the latest ledger, and
  the newest exchanges. The newest one always stays (its outputs may not be analysed yet); older ones only up to three
  and half the budget. If it is still over budget, the old trimming runs on top.
- A new ledger needs 25 % of the budget added since the last one and at least 3 tool rounds; otherwise the harness only
  trims. Without this, a step whose outputs alone exceed the budget wrote a ledger on every step.
- The ledger request reuses the turn's tool list and thinking settings with `tool_choice: "none"`, so the rendered prompt
  starts exactly like the one the server just processed and its prompt cache covers the history. Measured on an
  8.2k-token history: 260 tokens re-read (0.85 s) instead of 7,922 (13.2 s) for a request without tools.
- The budget check counts the messages appended since the last request.
- If the ledger request fails or returns nothing, the old trimming runs; `--no-ledger` keeps the old behaviour.

Tests: a mock-server unit test (repeated compactions keep the request verbatim, one ledger in the head, every tool
result after its call, ledgers carried forward, files written, context near the budget; batched reads trigger it; a
second turn does not drop the first; one huge step per round is bounded; empty ledger falls back). End-to-end with
Inkling 2 Mini (docs/14) on a synthetic 24-module repository with 3 planted bugs and 2 suspicious-but-correct decoys, at
a deliberately tiny 8,000-token budget, agent in a container on an internal Docker network shared only with the server:
all 3 bugs found (and verified by running code), no decoy flagged, 4-7 ledgers per run, 330-400 s per run. Not yet
measured: real repositories, the realistic 49k budget, false-positive rates.

### Comparing servers for review work

Short-prompt decode speed is the wrong metric for this workload: prompt reading dominates (every file read is prompt),
and the reading speed at depth differs far more between stacks than decode speed does. The evaluation design used:

- Same harness, sampling, reasoning effort, time allowance and sandbox for every server.
- Cases built from real repositories at the parent of a known bug-fix commit, with the fix hidden (key outside the
  workspace), in three sizes (under 32k tokens, 32-64k, over 100k so that at least two compactions happen), plus clean
  cases (fixed code, or documented suspicious-but-correct patterns) where any finding is a false positive. Two seeds per
  case and server, cold prompt cache per case.
- Scored by a person or a checked script, never by the model under test: confirmed-correct findings (file + function +
  same failure condition), partial credit, false positives; whether findings and exclusions stated in a ledger survive
  to the final answer; wall time to the first correct finding and to the end, split into prompt reading, decode, ledger
  writing and tool time; results by case size and number of compactions.
- Switch only if a candidate gives more correct findings per hour without more false positives or worse evidence
  survival.
