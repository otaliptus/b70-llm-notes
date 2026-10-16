# Results

All at the 39-task single-turn suite unless noted. Full per-task grids, conversation, repair and long-context tables:
[`bench/results/report.md`](../bench/results/report.md). Dates: 2026-10-14 to 2026-10-16.

## Models and how they were served

| short name | model | file | engine on the B70 | decode tok/s |
|---|---|---|---|---|
| Flash-Next | Qwen3.8-Flash-Next (MoE) | IQ3_S | Strata 0.1.44, SYCL engine, 262k ctx | 44-47 |
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
| LFM2.5 Q6_K, SYCL | – | 128 | ttft 0.2 s |
| Bolt, SYCL b11669 | 624 (pp512) | 29.7 | flat under every toggle |
| Bolt, Vulkan b11669 host | 777 | 41.9 | |
| Bolt, Vulkan + DFlash n=3 | 777 | 50 median over 39 tasks | see 05 |
| Bolt, Vulkan + DFlash, ssm tensors Q4_0 | 966 | 56 | +13 % perplexity, opt-in |
| Bolt, patched Vulkan kernels (09) | 789 | 51.0 plain, **64.2 median with DFlash n=3** | 34/39, perplexity unchanged |
