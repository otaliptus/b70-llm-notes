# Local LLMs on an Intel Arc Pro B70: benchmark, speedups, notes

Working notes from two weeks of running 20-80B open-weight models on a single Intel Arc Pro B70 (32 GB) with
llama.cpp (SYCL and Vulkan backends) and Strata, plus the small benchmark harness used to compare them. Everything
here is generic: no personal paths, keys or hostnames; scripts take their locations from environment variables.

## Seven results worth knowing
1. **A hard thinking budget (~2048 tokens) beats unlimited thinking** for every Qwen-family reasoning model tried:
   higher score *and* 2-4x less wall time, because unlimited runs regularly think to the output limit and never
   answer. [`docs/04-thinking-budget.md`](docs/04-thinking-budget.md). Measured on Strata 0.1.44 and llama.cpp;
   Strata 0.1.45 closes a no-budget thinking block at three quarters of `max_tokens`, which removes the lost
   answers but not the wall time (Flash-Next: 37 with no budget vs 36 at b2048, one run each, 72 vs 23 min;
   [`docs/03-results.md`](docs/03-results.md#thinkingcap)).
2. **The faster backend depends on the model.** For Orca 27B (dense) Q4_K_S, SYCL gives 2.4x the prefill and +10-30 %
   decode over Vulkan. For Nemotron 3.5 Bolt (hybrid Mamba-2 + MoE, IQ2_XXS + NVFP4) Vulkan gives +45 % decode and
   the DFlash drafter at draft length 3 another +20 %: 29 -> 50 tok/s, same score, 1.7x less wall time.
   [`docs/05-bolt-speedups.md`](docs/05-bolt-speedups.md), [`docs/06-orca-27b-sycl.md`](docs/06-orca-27b-sycl.md)
4. **The Vulkan kernels had another +21 % in them.** An integer-dot path for IQ2_XXS, a dedicated NVFP4 shader and a
   one-line class of bug in the fused router (runtime loop bounds pushing register arrays to scratch) take plain decode
   from 42 to 51 tok/s, and the served configuration (DFlash drafter) from a 50 to a 64 tok/s median, with unchanged
   perplexity. [`docs/09-vulkan-kernels.md`](docs/09-vulkan-kernels.md)
5. **Speculative decoding on SYCL: draft probabilistically, and fix the multi-column mat-vecs.** For Orca 27B the
   MTP drafter was accepting 41-50 % of its greedy drafts at temperature 1.0; `--spec-draft-sampling probabilistic`
   (exact target distribution) takes it to 51-69 % and 36 -> 42 tok/s. The 4-token verify then cost 1.8x a single
   token because the K-quant multi-column kernels were issue-bound; four rows per sub-group, uniform scale loads and
   an un-fused gate/up at n >= 3 make 4-column mat-vecs 20-40 % faster (deterministic: batched-bench B=4 +12 %).
   Validated on the full 39-task suite in the served configuration (768 budget), two alternating pairs: 48.8 / 47.5
   vs 41.8 / 42.8 tok/s median, 13.2 / 13.3 vs 14.8 / 15.3 min wall, same pass rate; long-context suite 21/21 on both
   with +11-15 % decode at 8-64k; plain decode and perplexity unchanged. One trap: build the SYCL backend *with*
   oneDNN or prefill at depth loses 16 %. Thinking budget, three rounds each on a 16-task repair suite: 512 passes
   as many tasks as 2048 (13-14 of 16 after repairs) in 44 % of the wall time, so the agent now runs at 512.
   [`docs/10-sycl-speculative-kquant.md`](docs/10-sycl-speculative-kquant.md)
6. **Upstream caught up and overtook: put the verify step on the XMX units.** llama.cpp PR #29864 runs the multi-column
   K-quant mat-vecs on int8 DPAS and its cost is flat from 2 to 8 columns (Q4_K 17408x5120, n = 4: 91 us vs 133 with our
   kernels and 157 on master). Full suite at budget 512, two alternating pairs: 64.6 / 64.9 vs 49.6 / 49.3 tok/s median,
   7.8 / 8.3 vs 10.2 / 10.2 min, pass counts within the run-to-run spread, long suite 21/21 both; op tests and perplexity
   unchanged. Strata 0.1.45 is +2-4 % decode with identical greedy answers; its opt-in KREG kernels another +4 % but the
   answers change. Seven of eleven suggested upstream items did not apply to this model once the GGUF and the served
   flags were checked. Then f16 KV (the SYCL attention decode kernel at head size 256 is 1.6x slower on q8_0 KV), the
   Battlemage FA vector-kernel PR and the official DFlash2 block drafter together: another +20 % on the full suite
   (78 vs 65 tok/s median, 11 % less wall, same long-context pass rate, +59 % decode at 64k depth). Served Orca went
   from 50 to 78 tok/s in one day. [`docs/12-sycl-xmx-upstream-pass.md`](docs/12-sycl-xmx-upstream-pass.md)
7. **Inkling 2 Mini on SYCL: 28 -> 58 tok/s from kernels, not settings.** The NVFP4 expert mat-vec was compute-bound
   (emulated FP4 table plus `pow` for every scale); decoding with SWAR and building the scale from its bits made it 3.3x
   faster. The SYCL flash-attention kernels learned a per-head mask (the model's relative-position bias), and nine
   model-specific node-group fusions cut decode launches from 7,609 to ~2,250. Server decode 28.4 -> 57.7 tok/s, pp512
   343 -> 889, 103k-token prompt prefill 142 -> 265 and decode at that depth 14.1 -> 37.7, KLD unchanged within noise.
   Also found: the unfused relative-position path breaks past ~2^29 elements (~87k tokens at micro-batch 256).
   [`docs/13-inkling-2-mini.md`](docs/13-inkling-2-mini.md)
8. **Inkling 2 Mini Uncensored on ggml-org master: 58.9 -> 69 tok/s with drafting.** Unsloth's DSpark drafter made
   decoding slower on SYCL (a single-work-item argmax over 200k logits, two lm-head passes per drafted position, the
   whole block always chained); fixed, ported to master, and given a reduced-vocabulary draft head (32k rows of the
   target's lm head: draft step 2.4 -> 1.3 ms), plus SYCL graph replay for the 2-8 token verify batches. Sweep 69.1
   tok/s (75.0 at temperature 0), 103k needle still found. The drafter's confidence predicts rejection (89 % accepted
   above p 0.9), but gating on it is eaten by graph rebuilds when the batch size changes; open items are listed for
   whoever continues. [`docs/14-inkling-uncensored-dspark.md`](docs/14-inkling-uncensored-dspark.md); agent-side
   (mid-task evidence ledger, how to compare servers for review work) in [`docs/11`](docs/11-agent-context-management.md)
9. **Native Inkling decode: one tile for every row of a check, 105 -> 115 tok/s, bit for bit.** On the nextsycl native
   engine, decode attention grew from 1 to 10.6 ms a step between a short prompt and 100K. Check steps re-read each
   key once a row. Patch 0061 loads each 64-key block once for all 1-8 rows, issues small kernels' reads first and
   casts job-table pointers to global memory, with the same lanes, rotations, fma chains and summation order, so
   full-vocabulary scores are byte-identical to 0059. Decode +6 to +13 % at fixed seed; 39-task suite 105.0 -> 115.1
   tok/s pooled over three interleaved runs each. Ideas from steveseguin's b70-optimization-lab inkling2 engine.
   [`docs/17-inkling-decode-attention.md`](docs/17-inkling-decode-attention.md)
10. **Native Inkling long & drafted decode: 96.4–96.8 tok/s at 100K and 147.2 tok/s short bit for bit, plus opt-in INT8 KV to 155.7 tok/s at 39K, 110–128 tok/s at 100K, and 688,128 max context.** Patches
   0062–0063 replace two-pass long full-attention with single-launch L1/SLM-aware ESIMD FlashDecoding, couple
   temperature-1 DSpark drafting to the target's inverse-CDF draw over INT8 `fold` + INT8 Layer-1 KV + dynamic
   session-OOV rows, capture decode/draft/sampling passes in SYCL command graphs, parallelize MoE `tanh`/`exp` SFU lanes,
   and fuse Q8_0 loop controllers, router scans, and MoE combine+conv. Default `f16` stays 100 % byte-identical across all
   18 teacher-forced score specs and 14 chat replay hashes (`96.4 tok/s` at 100K, `141.0` at 39K, `133.8` at 7.3K, `147.2`
   short-1024, `90.6` plain short, `113.5` 14-turn chat, `375,808` max ctx). Opt-in `NS_INK2_KV=q8` (`--opt-kv q8`) adds
   symmetric INT8 target full-attention KV with hybrid XMX-prefix + SLM-tail FlashDecoding (`2.45x` faster 100K
   attention+merge, `0.042` nats/tok 100K KLD), reaching `152.4–155.7 tok/s` at 39K, `108.3–113.5 tok/s` sampled /
   `128.1` greedy / `64.5` plain at 100K, `1,899 tok/s` 100K prefill, `115.7 tok/s` chat, and `688,128` (`672K`) max
   context with DSpark enabled (`733,184` without) on one 32 GB B70. [`docs/18-inkling-neon-decode.md`](docs/18-inkling-neon-decode.md)
3. **On this GPU avoid IQ-type quants for the big tensors** (IQ4_XS decodes at 13 tok/s where Q4_K_S does 30) and
   requantize with an importance matrix; and do not re-block NVFP4 into Q4_0/Q5_0 unless you accept +13 % perplexity.

## Layout
| path | what |
|---|---|
| [`docs/01-environment.md`](docs/01-environment.md) | hardware, drivers, the two backends, container setup, gotchas |
| [`docs/02-benchmark.md`](docs/02-benchmark.md) | the 39-task + multi-turn + repair + long-context harness and how to run it |
| [`docs/03-results.md`](docs/03-results.md) | all models, all scenarios, speeds |
| [`docs/04-thinking-budget.md`](docs/04-thinking-budget.md) | the budget finding |
| [`docs/05-bolt-speedups.md`](docs/05-bolt-speedups.md) | SYCL -> Vulkan -> DFlash -> requant on Nemotron 3.5 Bolt, with the per-op profile |
| [`docs/06-orca-27b-sycl.md`](docs/06-orca-27b-sycl.md) | Orca 27B (OrcaSAQ-2-Cyber-27B, dense): backend, MTP drafting, imatrix quant study, 262k KV, power cap |
| [`docs/07-strata-notes.md`](docs/07-strata-notes.md) | Strata engine on Battlemage: AOT build, kv-resident 262k, sampling, control vector |
| [`docs/08-dead-ends.md`](docs/08-dead-ends.md) | everything measured and rejected, with numbers |
| [`docs/09-vulkan-kernels.md`](docs/09-vulkan-kernels.md) | llama.cpp Vulkan kernel work: IQ2_XXS int8 dot, dedicated NVFP4 shader, router fix; 42 -> 51 tok/s plain decode |
| [`docs/10-sycl-speculative-kquant.md`](docs/10-sycl-speculative-kquant.md) | llama.cpp SYCL: where a speculative step goes, probabilistic MTP drafting, faster multi-column K-quant mat-vecs, full-suite validation, the oneDNN build trap; 36 -> 48 tok/s served |
| [`docs/11-agent-context-management.md`](docs/11-agent-context-management.md) | keeping a local agent fast over long sessions: compaction point, durable notes, tool logs on disk, stable prefixes, per-request thinking budgets, what to count; the mid-task evidence ledger for one long review turn; how to compare servers for review work |
| [`docs/12-sycl-xmx-upstream-pass.md`](docs/12-sycl-xmx-upstream-pass.md) | upstream pass 2026-10-18/19: llama.cpp PR #29864 XMX multi-column mat-vecs (+30 % full-suite tok/s over part 2, per-kernel table, flat verify cost), Strata 0.1.45 and KREG A/Bs, draft length revisited, and the backlog items that did not apply |
| [`docs/13-inkling-2-mini.md`](docs/13-inkling-2-mini.md) | Inkling 2 Mini (looped MoE, NVFP4 experts) on SYCL: NVFP4-Q8_0 requant, NVFP4 mat-vec decode fix, per-head flash-attention mask, Inkling node-group fusions, the long-context overflow, where the time goes; 28 -> 58 tok/s served |
| [`docs/14-inkling-uncensored-dspark.md`](docs/14-inkling-uncensored-dspark.md) | Inkling 2 Mini Uncensored (AtomicChat AD-Q4_K) on ggml-org master: patch order, DSpark drafter fixes and port, reduced-vocabulary draft head, SYCL graphs for verify batches, where a speculative step goes, the confidence gate, what did not help, open items in order; 58.9 -> 69 tok/s |
| [Native Inkling on the B70](docs/15-inkling-nextsycl.md) | AtomicChat AD-Q4_K on nextsycl, full-vocabulary sampling, graded tasks, chat, compaction, 100K and 126K throughput, patches 0054–0059 and optional long-context drafting in 0060 |
| [November 23 Inkling experiment record](docs/16-inkling-b70-experiment-log.md) | Exact models and runtime builds, graded-suite replay, experiments that stayed out of the release, and locations of the original measurements |
| [`docs/17-inkling-decode-attention.md`](docs/17-inkling-decode-attention.md) | Native Inkling decode attention (patch 0061): shared check-step tile, reads-first kernels, global pointer casts; exactness checks, kernel and decode numbers, draft settings re-checked, what was not ported; 105 -> 115 tok/s |
| [`docs/18-inkling-neon-decode.md`](docs/18-inkling-neon-decode.md) | Native Inkling decode stack (patches 0062–0063): single-launch ESIMD FlashDecoding, coupled INT8 DSpark + INT8 KV + dynamic session-OOV head, SYCL command graphs, parallel MoE SFU, Q8_0 loop controllers, and opt-in symmetric INT8 full-attention KV (`--opt-kv q8`); 96.4 tok/s exact / 147.2 tok/s short (`f16`) and 155.7 tok/s @ 39K / 110–128 tok/s @ 100K / 688,128 max ctx (`q8`) |
| [`bench/`](bench/) | the harness (stdlib Python + Docker sandbox) and the generated [`report.md`](bench/results/report.md) |
| [`scripts/`](scripts/) | serve / build / sweep / profile / requant / perplexity scripts; DSpark reduced-vocabulary draft head (`make-dspark-subvocab.py`, `token-freq.py`, see 14); nextsycl inkling2 attention kernel bench (`inkling-attn-bench.sh`, see 17) |
| [`patches/`](patches/) | llama.cpp patches (0001-0010) and Strata patches (0011-0018: in-file MTP head packer, Intel drafter path check, Intel server hook (0.2.10 only), SYCL DFlash2 planar Q6_K head gather, drafter check on the engine binary, launcher zero allowance, A/B drafter arm, tuned-DFlash2 test switch; 0011-0012 and 0014-0018 against v0.2.15). b11669: SYCL `out_prod` strided f32 fix (DFlash on SYCL); Vulkan: IQ2_XXS/NVFP4 integer-dot paths, dedicated NVFP4 mat-vec, top-k router fix, Intel policy + knobs, perf shapes. b11696: SYCL K-quant multi-column mat-vecs (0008), MTP draft timing (0009), Qwen3.5-class perf shapes (0010). On PR #29864 / master b11721 apply 0009-0010 only (0008 is superseded there, see 12). 0019-0029: Inkling 2 Mini on Unsloth's b12572-mix (SCALE strided source, get_rows launch split, NVFP4 SWAR decode, per-node GPU profiler, per-head FA mask, `inkling-fuse.cpp` fusions; see 13). 0030-0031: same tree, small-batch MoE and the DSpark drafter fixes. 0032-0053: Inkling 2 Mini Uncensored on ggml-org master 19bc41d (+ upstream #32799): ports of 0019-0029, Q5_0 MoE mat-vec, master-graph fusions, small-batch MoE, DSpark port, reduced-vocabulary draft head, q8_1 input cache, SYCL graphs for verify batches, confidence gate (off), debug aids; see 14. Patches 0054–0060 cover [native nextsycl on the B70](docs/15-inkling-nextsycl.md#reproduce-the-runtime); 0061 is the decode attention of [17](docs/17-inkling-decode-attention.md); 0062–0063 are the ESIMD FlashDecoding, coupled INT8 DSpark, SYCL-graph, parallel MoE SFU, and opt-in INT8 full-attention KV stack of [18](docs/18-inkling-neon-decode.md). |

## Quick start
```sh
export MODELS_DIR=/path/to/ggufs LLAMA_RT=/path/to/llama.cpp-vulkan-release API_KEY_FILE=~/inference.key
scripts/serve-bolt-vulkan.sh 2048 dflash              # Bolt, budget 2048, DFlash n=3, port 18081
cp bench/profiles.example.json bench/profiles.json    # endpoints and model ids
python3 bench/reference.py                            # the hidden tests are satisfiable
python3 bench/run_bench.py --name bolt-b2048 --url http://127.0.0.1:18081/v1 --key "$(cat ~/inference.key)" \
        --model nemotron-3.5-bolt-74b-a9b --temperature 1.0 --top-p 0.95 --top-k 0
python3 bench/report.py                               # bench/results/report.md
```
Models used: Qwen3.8 Flash-Next and Swift 1.5 (IQ3_S, Strata), OrcaSAQ-2-Cyber-27B (Q4_K_S), LiquidAI
LFM2.5-24B-A2B (Q6_K), NVIDIA Nemotron 3.5 Bolt 74B-A9B (IQ2_XXS + DFlash drafter). llama.cpp releases b11541,
b11669, b11696 and PR #29864 on b11721; Strata 0.1.44.1 and 0.1.45. Dates 2026-10-10 to 2026-10-19. Later: Strata
0.2.x (2026-11), Thinking Machines Inkling 2 Mini (NVFP4 experts + Q8_0, Unsloth llama.cpp b12572-mix, 2026-11-20/21).

## License
MIT, see [`LICENSE`](LICENSE).
