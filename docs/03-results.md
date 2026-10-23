# Results

All at the 39-task single-turn suite unless noted. Full per-task grids, conversation, repair and long-context tables:
[`bench/results/report.md`](../bench/results/report.md). Dates: 2026-10-14 to 2026-10-16 unless a section says otherwise.

## Models and how they were served

| short name | model | file | engine on the B70 | decode tok/s |
|---|---|---|---|---|
| Flash-Next | Qwen3.8-Flash-Next (MoE) | IQ3_S | Strata 0.1.44, SYCL engine, 262k ctx | 44-47 |
| ThinkingCap | bottlecapai ThinkingCap-Qwen3.8-Flash-Next (Flash-Next fine-tune, section at the end) | IQ4_XS (vendor GGUF, MTP block rewritten out, see 07) | Strata 0.1.45, SYCL engine, 131k ctx | 35 |
| Swift 1.5 | Swift 1.5 (MoE) | IQ3_S | Strata 0.1.44, SYCL engine | 45-47 |
| Orca 27B | OrcaSAQ-2-Cyber-27B (orcarouter), dense Qwen3-based | Q4_K_S (imatrix requant) | llama.cpp b11541 SYCL, MTP + n-gram drafting | 37 |
| LFM2.5 | LiquidAI LFM2.5-24B-A2B | Q6_K | llama.cpp b11541 SYCL, n-gram drafting | 128 |
| Bolt | NVIDIA Nemotron 3.5 Bolt 74B-A9B (hybrid Mamba-2 + MoE) | IQ2_XXS (unsloth) | llama.cpp b11669, SYCL then Vulkan + DFlash | 29 -> 50 |

## Headline: score by thinking budget

| model | b1024 | **b2048** | b4096 | unlimited | nothink | `low` | b2048 t=0.6 |
|---|---|---|---|---|---|---|---|
| Swift 1.5 | 35 | **37** | 32 (1 cap) | 32 (6 cap, 69 min) | 33 | 35 (2 cap) | 33 |
| Orca 27B | 35 | **36** | 36 | 25 (12 cap, 120 min) | 22 | – | – |
| Flash-Next | 33 | **34** (repeat 34) | 38 | 29 (9 cap, 84 min) | 32 | 34 (3 cap) | 38 |
| Bolt (SYCL) | – | **34** | – | – | 26 | – | – |
| Bolt (Vulkan + DFlash) | – | **33** | – | – | – | – | – |
| LFM2.5 | 24 | **28** | 27 | 29 (2 cap) | 17 | – | – |

Wall time at b2048: Swift 22 min, Flash-Next 24, Orca 34, Bolt 42 (SYCL) / 25 (Vulkan+DFlash), LFM2.5 15.
The unlimited column is a Strata 0.1.44 / llama.cpp result. Strata 0.1.45 closes a no-budget thinking block at
three quarters of `max_tokens` (12k with this suite's cap); re-run that way Flash-Next scores 37 with no cap hits at
three times the wall of b2048. See the note at the end of the ThinkingCap section below.

## Multi-turn, repair, long context (budget 2048)

| model | convo (8 conversations / 24 turns) | repair pass@1 / pass@3 of 16 | long 8k/32k/64k | cold 64k prefill |
|---|---|---|---|---|
| Swift 1.5 | 8/8, 24/24 | 13 / 15 | 7/7 each | 66 s |
| Flash-Next | 8/8, 24/24 | 13 / 14 | 7/7 each | 67 s |
| Orca 27B | 8/8, 24/24 | 11 / 16 | – | – |
| Bolt | 7/8, 23/24 | 13 / 14 | 7/7 each | 101 s (decode 24 tok/s at 64k) |
| LFM2.5 | 7/8, 23/24 | 8 / 11 | 7/7 each | 26 s (decode 51 tok/s at 64k) |

Both 7/8 results lose the same conversation: the three-turn growing `Inventory` class with an atomic `transfer` and
an `undo`.

## Per-model notes
- **Swift 1.5** is the best stock choice for coding on this bench; it also tolerates `nothink` best (33).
- **Flash-Next** reaches 38 at b4096 or at b2048 with temperature 0.6; b2048 at temperature 1.0 is 34 twice.
- **Orca 27B** matches the Strata models once given a 2048 budget (36). Unlimited thinking is a disaster for it
  (12 of 39 answers never arrived within 16k tokens).
- **LFM2.5-24B-A2B** is 3x faster than everything else and ~9 tasks weaker. Gaps are knowledge and hard code. It is
  an agent model: without a system prompt it emitted tool calls on 4 plain questions; with "no tools are available"
  that stopped (score 27, within noise). It always spends its whole thinking budget.
- **Bolt** at 2 bits per weight scores with the 3-4 bit models (34), 13/13 on knowledge, and 21/21 on long context.
  Its misses are small-careless (`from typing import dict`) rather than conceptual. Nothink costs it 8 tasks.

## Speed summary
| model / engine | prefill tok/s | decode tok/s | notes |
|---|---|---|---|
| Strata (Flash-Next / Swift) | 1,180 at 15k prompt (was 250-525 before 0.1.44) | 47-48 at 0-15k depth | per-request budgets, control vector optional |
| Orca 27B, llama.cpp Vulkan b10430 (start) | 517 (pp2048) | 25 | |
| Orca 27B, llama.cpp SYCL b11541 | 1,259 (pp2048), 1,009 at 32k depth | 28 plain, 37 with MTP + n-gram | see 06 |
| Orca 27B, llama.cpp SYCL b11696 | 1,126 (pp512; b11541: 906) | 29.6 plain, 36-49 with MTP + n-gram | "wide stores for the q4_K/q5_K dequant" landed upstream; Vulkan (patched) is 686 / 27.4, so SYCL stays |
| LFM2.5 Q6_K, SYCL b11696 | 1,951 (pp512) | 148.6 (llama-bench) | Vulkan: 2,605 / 126; SYCL stays for decode |
| LFM2.5 Q6_K, SYCL | – | 128 | ttft 0.2 s |
| Bolt, SYCL b11669 | 624 (pp512) | 29.7 | flat under every toggle |
| Bolt, Vulkan b11669 host | 777 | 41.9 | |
| Bolt, Vulkan + DFlash n=3 | 777 | 50 median over 39 tasks | see 05 |
| Bolt, Vulkan + DFlash, ssm tensors Q4_0 | 966 | 56 | +13 % perplexity, opt-in |
| Bolt, patched Vulkan kernels (09) | 789 | 51.0 plain, **64.2 median with DFlash n=3** | 34/39, perplexity unchanged |

## Spark-X2.6-4B (XHToken, released 2026-10-19), measured 2026-10-20/21

A 4B dense on-device agent model (36 layers, sliding window 512 with every fourth layer full, 1M context, no MTP or
drafter; llama.cpp architecture `spark2_5`). Q8_0 on the SYCL XMX runtime with f16 KV: 5552 tok/s prefill, 102.7 tok/s
decode fresh, 68.2 at 32k depth (Q4_K_M: 120.6 / 75.7). Sampling per the card (temperature 1.0, top-p 0.95, no top-k),
n-gram drafting on but accepting only 6-31 %. 39-task suite, one pass per row, 16k output cap:

| thinking setting | pass / 39 | code | compr. | knowl. | wall | completion tokens | hit the cap |
|---|---|---|---|---|---|---|---|
| off | 12 | 1/16 | 2/10 | 9/13 | 5.7 min | 32k | 1 |
| budget 512 | 20 | 3/16 | 5/10 | 12/13 | 17.9 min | 95k | 4 |
| budget 2048 | 20 | 4/16 | 6/10 | 10/13 | 22.8 min | 120k | 3 |
| template effort "low", no budget | 24 | 6/16 | 8/10 | 10/13 | 49.4 min | 256k | 9 |
| effort "high", budget 8192 | 23 | 6/16 | 7/10 | 10/13 | 44.4 min | 233k | 2 |
| effort "high", unlimited | 15 | 1/16 | 6/10 | 8/13 | 73.8 min | 378k | 18 |

Same tasks and budget 2048 for scale: LFM2.5-24B-A2B 28 in 15 min, the two 27B cyber models 32 in 10-18 min. Repair
suite at budget 512: pass@1 4/16, pass@3 6/16 in 30 min (the 27Bs: 12-14/16 pass@3 in 5-12 min). Long-context suite:
7/7 at 8k and 32k, 6/7 at 64k, decoding at 61 and 48 tok/s.

Reading: the thinking behaviour is bimodal. A hard budget cuts nearly every coding task mid-thought (2,050-2,080 thinking
tokens at a 2,048 budget, then wrong code); no budget runs 18 of 39 tasks into the 16k output cap with a median of 13.8k
thinking tokens. The template's own "low" effort is the best compromise and still costs 3-5x the wall time of a 27B for
eight fewer passes. Raw decode speed is fine; tokens per task are 3-8x those of the 27Bs. The card's "matches 9B models"
claim does not reproduce on these tasks, and the 24B-A2B MoE beats it at every setting at similar speed.

<a id="thinkingcap"></a>
## ThinkingCap-Qwen3.8-Flash-Next (bottlecapai), measured 2026-10-23

A fine-tune of Qwen3.8-Flash-Next trained to think less (vendor card: ~40 % fewer reasoning tokens at the same
scores, MTP head unchanged). Run from the vendor's IQ4_XS GGUF (88 GiB after the MTP-block rewrite described in
[07](07-strata-notes.md)) on Strata 0.1.45 with the base model's MTP runtime as the drafter and otherwise the base
flags (expert streaming, int8 KV, `--spec 4`). The base Flash-Next IQ3_S (78 GiB) was re-run the same evening on the
same binary as the control, because the 2026-10-14 base numbers above were taken on 0.1.44 and the two releases
treat a no-budget request differently (note at the end). Sampling 1.0 / 0.95 / 20 as on the card, 16k output cap,
one run per row.

| run | pass / 39 | code | compr. | knowl. | median tok/s | wall | completion tokens | median thinking / task | hit the cap |
|---|---|---|---|---|---|---|---|---|---|
| ThinkingCap, budget 2048 | 34 | 12/16 | 9/10 | 13/13 | 35.5 | 24.6 min | 50k | 0.9k | 0 |
| base, budget 2048, same evening | 36 | 13/16 | 10/10 | 13/13 | 46.2 | 23.2 min | 62k | 1.6k | 0 |
| base, budget 2048, 2026-10-14 (0.1.44; two runs) | 34, 34 | 12/16 | 9-10/10 | 12-13/13 | 44 | 24 min | 61k | 1.4k | 0 |
| ThinkingCap, no budget | 35 | 13/16 | 9/10 | 13/13 | 35.2 | 65.6 min | 128k | 0.9k | 0 |
| base, no budget, same evening | 37 | 14/16 | 10/10 | 13/13 | 45.4 | 72.0 min | 183k | 1.5k | 0 |
| base, no budget, 2026-10-14 (0.1.44) | 29 | 7/16 | 10/10 | 12/13 | 42.5 | 84.4 min | 204k | 1.6k | 8 |

Repair suite at budget 2048 (16 coding tasks, up to two repairs each): ThinkingCap pass@1 12/16, pass@3 14/16,
7 repairs, 31 min; the base the same evening 13/16, 15/16, 4 repairs, 21 min. ThinkingCap's two unrepaired tasks are
the JSON parser and the regex engine; the base misses only the regex engine.

Per task the differences sit at the noise floor (2-4 tasks between two runs of one model): at budget 2048 the two
models differ on 2 tasks, both base passes (a diff viewer and the JSON parser); with no budget on 4 (ThinkingCap
loses an expression evaluator, a spec question and a cron parser, and wins the JSON parser).

Reading:
- The vendor's token claim reproduces. ThinkingCap's median thinking per task is 0.9k against 1.5-1.6k for the
  base, the tail is shorter (5 tasks reach the 12k close against 10; the code tasks' median thinking with no budget
  is 4.1k against 10.1k), and total completion tokens are 19 % lower at budget 2048 and 30 % lower with no budget.
- The score does not move: 34 vs 36 and 35 vs 37 against the same-evening base, 14 vs 15 of 16 on repair. All
  inside the noise floor of a single run, so the honest reading is "no better than the base", not "worse".
- The token savings do not become wall time on this GPU because the IQ4_XS file decodes 23 % slower than the
  IQ3_S base (35 vs 46 tok/s median): with 32 GB of VRAM the expert cache holds 10,110 experts instead of 12,478,
  and 15,477 experts (35 GiB) instead of 13,298 (25 GiB) sit in pinned host memory and are read over PCIe. That is
  the quant, not the fine-tune; a like-for-like speed comparison needs the same quant on both sides. Wall per passed
  task: 0.72 vs 0.65 min at budget 2048, 1.87 vs 1.95 min with no budget.
- A hard budget of 2048 still removes most of the thinking tail at no measurable cost for both models (ThinkingCap
  34 in 25 min vs 35 in 66 min).

**Note on "no budget" and Strata 0.1.45.** The same-evening base control scored 37 with no cap hits where the
2026-10-14 run scored 29 with 8 (9 in the headline table, which also counted a task that stopped at 15k). Strata
0.1.45 (commit d0e990ba, "max_tokens counts the thinking") closes a no-budget thinking block at three quarters of
`max_tokens` so a quarter is left for the answer; the config key `answer_room: false` opts out. With this suite's
16k cap that is a 12k close: 10 of the control's 39 tasks and 5 of ThinkingCap's stop at 11.8-12.1k thinking tokens
and then answer, which is where the 8 lost answers of 10-14 went. So on 0.1.45 "unlimited" is a 12k soft budget plus
a forced answer, and for Flash-Next it scores the same as budget 2048 (37 vs 36, one run each) at three times the
wall. The wall-time half of the budget finding stands; the score half is specific to engines that let the thinking
run into the output cap (llama.cpp without `--reasoning-budget`, Strata before 0.1.45 or with `answer_room: false`).
