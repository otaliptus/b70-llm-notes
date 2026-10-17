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
