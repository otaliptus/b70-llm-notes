# Finding: a hard thinking budget beats unlimited thinking

This was the single most useful result of the whole exercise, and it held for every reasoning model tried.

## What happened with unlimited thinking
With a 16k output limit and no budget, the Strata models and Orca 27B regularly *thought until the
limit and never answered*. Flash-Next lost 9 of 39 answers that way, Swift 6, Orca 12. Those runs took 69-120
minutes against 15-35 minutes with a budget. Nearly every miss in the first 24-task run was this failure, not a
wrong answer.

## What a hard budget does
`reasoning_budget_tokens` (Strata, per request) or `--reasoning-budget N` (llama.cpp, server flag) stops the
thinking at N tokens, injects a short message ("budget reached; answer now with the best current plan") and lets the
model answer. Scores:

| model | 1024 | 2048 | 4096 | unlimited |
|---|---|---|---|---|
| Swift 1.5 | 35 | 37 | 32 | 32 |
| Orca 27B | 35 | 36 | 36 | 25 |
| Flash-Next | 33 | 34 | 38 | 29 |
| LFM2.5 | 24 | 28 | 27 | 29 |

Any budget between 1024 and 4096 beats unlimited for the three Qwen-family models; 2048 is the sweet spot for
speed (Swift's median thinking at b2048 was only ~1000 tokens, so the cap rarely bites and mostly removes the
runaway tail). LFM2.5 is the exception that proves the rule: it always uses its whole budget, so its score tracks
the budget size and unlimited is marginally best, at twice the wall time.

## What does not substitute for a hard cap
- The soft `reasoning_effort: low` lowers the median thinking but leaves the tail: 2-3 capped answers remain and
  the wall time stays long (Flash-Next 37 min vs 24 at b2048).
- Turning thinking off entirely costs 2-14 tasks depending on the model (Swift -4, Flash-Next -2, Orca -14,
  Bolt -8, LFM -11) but is 3x faster; fine for chat, not for the coding tasks.
- Temperature 0.6 instead of 1.0 at b2048: +4 for Flash-Next, -4 for Swift, i.e. noise.

## How to apply
Set a ~2048 budget as the default in every reasoning-model config and let the client raise it per request when a
task deserves it. On the model side, make sure the budget message is set so the model answers instead of stopping.
