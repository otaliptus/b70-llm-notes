# Orca 27B on SYCL, part 2: where the speculative step goes, and faster multi-column K-quant mat-vecs

Follow-up to `06-orca-27b-sycl.md`. Same model (OrcaSAQ-2-Cyber-27B, dense Qwen3.5-class hybrid: 48 gated
delta-net layers + 16 full-attention layers, 14.73 GiB of Q4_K / Q5_K / Q6_K weights), llama.cpp b11696 SYCL,
Arc Pro B70. Everything measured on 2026-10-16 with the 4-task "speed subset" of the benchmark (temperature 1.0,
2048-token thinking budget) unless stated otherwise.

## 1. Plain decode is already at the memory wall; the knobs do nothing
`test-backend-ops perf` with the model's own shapes (`ORCA_SHAPES=1`, patch 0010) says every single-token
mat-vec runs at 570-600 GB/s on this card (Q6_K lm head 995 MiB: 598 GB/s; Q4_K 17408x5120: 570; Q5_K
5120x17408: 570). 14.73 GiB at ~590 GB/s is 26.8 ms; the measured token is 33.9 ms (29.5 tok/s); the rest is
~1600 small launches per token. There is nothing left in the kernels for n = 1.

Every runtime knob this backend exposes was swept with `llama-bench` (`scripts/sweep-sycl-knobs.sh`): SYCL graphs
(`GGML_SYCL_ENABLE_GRAPH=1`, compiled in), fusion off, ESIMD off, wide MMVQ off, DMMV priority, reorder off, the
dynamic-precision modes, oneDNN / MKL flash attention off, f16 vs q8_0 KV, flash attention off (fails for this
architecture). All within +-0.5 % of 29.4 tok/s, except fusion off (-4 %) and ESIMD off (-6 %), which confirms the
defaults.

## 2. The speculative step: drafting is cheap, verifying is not
The lab serves Orca with the model's own MTP head as drafter (`--spec-type draft-mtp,ngram-mod`). Instrumenting the
drafter (patch 0009, `SPEC_TIMING=1`) gives, per step at draft length 3:

| part | time |
|---|---|
| MTP `process()` over the verified batch | 1.2 ms |
| 3 x MTP draft decode (1 layer + 995 MiB lm head each) | 3 x 2.5 ms |
| draft sampling (backend) | ~0 |
| **verify 4 tokens through the target** | **~60 ms** (a single token is 34 ms) |

(Measured on the unpatched kernels; the split has not been re-recorded on the deployed build.) So the 4-token verify costs 1.77x a 1-token step, and `llama-batched-bench` says the same about any 4-column batch
(B=1: 25.1 t/s, B=4: 62.7 t/s aggregate = 1.6x the step time). Per kernel, the multi-column reorder mat-vecs were
far from "weights read once": Q4_K 17408x5120 at n=4 took 1.79x the n=1 time, Q5_K 5120x17408 2.2x, Q6_K lm head
1.54x. Beyond 8 columns the backend leaves MMVQ for the dequant + XMX GEMM path and collapses (B=12: 33 t/s).

## 3. Drafting mode: probabilistic beats greedy by 17 %
`--spec-draft-sampling probabilistic` lets the target accept drafts by rejection sampling (exact target
distribution) instead of matching the greedy draft token against a temperature-1.0 sample:

| config (MTP + n-gram) | median tok/s | draft acceptance |
|---|---|---|
| no speculation | 28.5 | |
| n-max 3, greedy draft (previous lab setting) | 36.0 | 41-50 % |
| MTP only, n-max 3 | 36.1 | |
| n-max 2, greedy | 37.3 | |
| **n-max 3, probabilistic** | **42.1** | 51-69 % |
| n-max 4, probabilistic / greedy | 38.5 / 32.0 | |
| n-max 5, probabilistic | 33.0 | |

The n-gram drafter adds nothing measurable for this model. Longer drafts lose because each extra verified column
costs ~25 % of a token (section 2), more than the extra accepted length returns.

## 4. Faster multi-column K-quant mat-vecs (patch 0008)
What the ISA dumps (`IGC_ShaderDumpEnable=1`) showed for the n=4 Q4_K kernel: SIMD16, 128 GRF, no spills, but
~1.5x the instructions per row of the n=1 kernel, a data-dependent branch per block for the scale decode, and every
address derived from the lane id (so nothing is sub-group uniform). Changes, all in `ggml-sycl/mmvq.cpp` and
`vecdotq.hpp`:

- **Pre-decoded weights.** `load()` now returns the 4-bit/5-bit values already extracted into dot-product operands
  and the scale/min already multiplied by the block's d/dmin; the activation-only term (`d8 * sum(u)`) is computed
  once per column in `load_activations()`. `apply()` is 4 dp4a + 4 fma per sub-block pair.
- **Uniform scale loads.** The 12-byte K-quant scale block is loaded as three dwords (same address for every lane)
  and the per-lane (scale, min) pairs are extracted branch-free, replacing the `if (j < 2)` divergent region.
- **Compile-time loop start.** When one block spans the sub-group the first block index is 0 for every lane; saying
  so lets the compiler keep block offsets and scale/dm addresses uniform.
- **Rows per sub-group.** For Q4_K / Q5_K at 3..8 columns, four rows per sub-group when the matrix has >= 4096
  rows (amortizes the six scattered activation loads per column), two otherwise; Q6_K gets a `load / apply` split
  (it re-read its weights once per column before) and two rows.
- **GLU fusion only for 1..2 columns.** The fused gate+up kernel holds two weight sets per lane; with paired rows it
  spills (up to 3.7 KB, 7-8k instructions) and loses to the unfused kernels at n >= 3 (B=4: 65.4 vs 67.7 tok/s).
  At n = 1 the fusion still wins (25.1 vs 24.3), so it is gated by `GGML_SYCL_GLU_FUSE_MAX_NCOLS` (default 2).

Per kernel (us/run, `test-backend-ops perf`, same card):

| shape | n | before | after |
|---|---|---|---|
| Q4_K 17408x5120 (ffn gate/up) | 1 / 4 | 88.0 / 157.6 | 87.7 / 130.4 |
| Q4_K 12288x5120 (attn q) | 4 | 101.2 | 90.0 |
| Q4_K 6144x5120 / 5120x6144 | 4 | 56.0 / 64.2 | 32.9 / 37.1 |
| Q5_K 5120x17408 (ffn down) | 4 | 239.3 | 158.8 |
| Q5_K 10240x5120 (attn qkv) | 4 | 103.5 | 92.8 |
| Q6_K 248320x5120 (lm head) | 4 | 2682 | 2053 |

End to end, deterministic (`llama-batched-bench`, aggregate t/s): B=4 62.7 -> 70.0, B=6 65.6 -> 75.8, B=8
66.8 -> 74.8; B=1 unchanged (25.1).

Served configuration (probabilistic MTP, n-max 3), **4-task speed subset, temperature 1.0, 2048-token budget, one
run per runtime, back to back**: median 41.9 -> 47.7 tok/s, total wall 160 -> 135 s, 4/4 passed on both. Against
the day's starting configuration (greedy drafts, official kernels: 36.0) that is +32 % on this subset. Caveats that
the number does not cover: the subset's run-to-run noise at temperature 1.0 is a few tok/s (repeat runs of the same
configuration: 41.9 / 42.0 / 42.1 / 42.8 / 43.5 for the probabilistic baseline); the full 39-task suite, deep
contexts (where the 16 attention layers add verify cost) and the agent configuration (768 budget, tool output in the
prompt) have not been measured yet. Draft length 4 re-tested on the new kernels: 42.0, so n-max 3 stays.

No regression detected: `test-backend-ops test -o MUL_MAT` 2273/2273 against the CPU backend; perplexity over the
same 8x512 tokens, official vs patched: 6.6087 vs 6.6117 at ubatch 1, 6.6217 vs 6.6241 at ubatch 4. The kernels
reorder float accumulation by construction, so results are close, not bit-identical; eight chunks bound gross errors
only.

## 5. Things that did not work (details in 08)
Four rows per sub-group everywhere (Q6_K and small matrices got slower; the fused GLU kernel spilled), the large
register file JIT option (helps n=8 by 30 % but not the dominant n=4 shapes; also halves resident threads), and the
instruction-count hypothesis on its own: pre-decoding alone changed nothing for Q4_K at n=4 because the compiler had
already hoisted that work; the gains came from the rows-per-sub-group change and the GLU gate.

## Reproduce
```sh
# build b11696 + patches 0001, 0008-0010 (SRC = a llama.cpp checkout at tag b11696 with the patches applied)
SRC=... OUT=... scripts/build-sycl-patched.sh
# per-kernel timings and correctness of the dev runtime
LLAMA_RT=$OUT scripts/kernel-check-sycl.sh
ORCA_SHAPES=1 LLAMA_RT=$OUT scripts/perf-shapes-sycl.sh -o MUL_MAT
# drafting sweep against a running server (serve-orca-sycl.sh now uses probabilistic drafting)
scripts/spec-sweep-llamacpp.sh
# perplexity A/B at ubatch 1 and 4
TEXT=corpus.txt scripts/ppl-ab-sycl.sh /path/official-runtime /path/patched-runtime
```
