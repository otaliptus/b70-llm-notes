# Orca 27B on SYCL, part 3: upstream XMX multi-column mat-vecs, Strata 0.1.45 and KREG, and a backlog triage

Dates 2026-10-18/19. Same machine and containers as [01](01-environment.md); the Orca served configuration is the one
validated in [10](10-sycl-speculative-kquant.md): Q4_K_S, 262k context, q8_0 KV, flash attention, MTP + n-gram drafter at
draft length 3 with probabilistic sampling, temperature 1.0, agent thinking budget 512. All speeds are tok/s as reported by
the server's own timings; "speed subset" = 4 tasks of the 39-task suite at budget 2048, one pass per row; "full suite" =
all 39 tasks; "long suite" = 7 tasks x 3 depths (8k/32k/64k of filler before the question).

The input was an 11-item list of upstream changes suggested by another model. Checked against the actual GGUF, the actual
served flags and the actual upstream state, 4 applied and 7 did not (section 5). The one that mattered is a llama.cpp pull
request that puts the multi-column K-quant mat-vec on the XMX units: **+30 % median tok/s on the full suite over the
hand-tuned kernels of part 2, same pass rate, 20-24 % less wall time.**

## 1. llama.cpp PR #29864: XMX (int8 DPAS, ESIMD) multi-column `mul_mat_vec_q`

Part 2 found that the speculative verify step (2-8 columns through the K-quant mat-vecs) was issue-bound and shaved
20-40 % off it with better SYCL kernels (patch 0008). PR #29864 ("sycl: XMX multi-column mmvq for reordered quants")
replaces that path for n = 2..80 columns with ESIMD kernels that run the int8 dot products on the DPAS units. The
verify window therefore costs nearly the same for 2 and for 8 columns. Knobs: `GGML_SYCL_ENABLE_XMX_ESIMD` (1; 0 restores
the plain path), `GGML_SYCL_XMX_MAX_COLS` (80), `GGML_SYCL_XMX_GLU_MAX_COLS` (16).

Build: check out the PR head (9ac7826ee at the time: the PR branch with master b11721 merged in, so it already contains
#30374 and #30239), apply patches 0009 and 0010 (timing instrumentation and perf shapes; **0008 does not apply and is not
needed**, the PR supersedes it), then `SRC=... OUT=... scripts/build-sycl-patched.sh` with oneDNN on as in part 2. The
same script builds any source tree, a PR checkout included.

### Per-kernel cost (test-backend-ops perf, Orca shapes, us per run, lower is better)

| weight | shape m x k | n | XMX (PR) | master b11721 (XMX off) | b11696 + patch 0008 |
|---|---|---|---|---|---|
| Q4_K | 17408 x 5120 | 2 | 89 | 109 | 106 |
| Q4_K | 17408 x 5120 | 3 | 91 | 134 | 120 |
| Q4_K | 17408 x 5120 | 4 | 91 | 157 | 133 |
| Q4_K | 17408 x 5120 | 6 | 92 | 271 | 180 |
| Q4_K | 17408 x 5120 | 8 | 101 | 342 | 252 |
| Q4_K | 6144 x 5120 | 4 | 13 | 55 | 33 |
| Q5_K | 5120 x 17408 | 2 | 109 | 157 | 152 |
| Q5_K | 5120 x 17408 | 4 | 111 | 254 | 146 |
| Q5_K | 5120 x 17408 | 8 | 118 | 473 | 319 |
| Q6_K | 248320 x 5120 (output head) | 2 | 1733 | 1829 | 1782 |
| Q6_K | 248320 x 5120 | 4 | 1753 | 2663 | 2088 |
| Q6_K | 248320 x 5120 | 8 | 1787 | 5700 | 3905 |

n = 1 is the same on all three builds (not shown). The XMX numbers are flat in n; the other two grow with n, which is why
part 2 could not go past draft length 3.

### Correctness and plain speed (no regression detected)

- `test-backend-ops -o MUL_MAT`: 2273/2273 against the CPU reference.
- Perplexity, 8 x 512-token chunks of the usual text, ubatch 1: 6.623 (XMX) vs 6.612 (previous runtime); ubatch 4:
  6.622 vs 6.624; the estimate's own error is +/-0.35.
- llama-bench, 3 runs each: pp512 1053 vs 1065, tg128 29.5 vs 29.4, pp512 at depth 32768 753 vs 755, tg128 at depth
  32768 18.7 vs 18.5. (One cold run of pp512 on the XMX runtime printed 744 +/- 560: the first call JIT-compiles the ESIMD
  kernels; the repeat gave 1053.)

### End to end

Speed subset, budget 2048, one pass per row, same session, GPU otherwise idle:

| runtime | median tok/s |
|---|---|
| master b11721, XMX off (`GGML_SYCL_ENABLE_XMX_ESIMD=0`) | 41.0 |
| master b11721 + patch 0008 ported | 46.6 |
| b11696 + patches 0008-0010 (part 2's result, served until now) | 46.2, 46.8 |
| PR #29864 (XMX) + 0009/0010 | 61.5, 61.4 |

So the newer base alone brings nothing for this model, and our 0008 on top of it reproduces part 2; the XMX path is the
whole gain. Draft acceptance is unchanged (0.54-0.69 per task), as it must be: only the verify cost moved.

Full 39-task suite in the served configuration at budget 512, alternating XMX / previous / XMX / previous, 2026-10-19
01:08-01:48, each row one pass:

| run | median tok/s | wall | passes / 39 | tokens |
|---|---|---|---|---|
| XMX, pair 1 | 64.6 | 7.8 min | 28 | 29.0k |
| previous, pair 1 | 49.6 | 10.2 min | 31 | 28.9k |
| XMX, pair 2 | 64.9 | 8.3 min | 30 | 31.1k |
| previous, pair 2 | 49.3 | 10.2 min | 29 | 29.4k |

Pass counts differ by 3 in the first pair and by 1 the other way in the second; the identical-configuration spread in
this and earlier A/Bs is 2 passes, so no accuracy effect is detectable at this sample size (the kernels are exact on the
op tests and the perplexity matches). Long suite right after, one pass each: 21/21 on both; decode median 46.4 vs 37.4;
per depth XMX 52-69 at 8k, 39-57 at 32k, 30-42 at 64k against 40-55, 33-39, 25-37. A leftover sandbox container from a
hidden test pinned one CPU core throughout the four full-suite runs, equally for both sides.

Adopted as the served runtime. Part 2's runtime stays as the fallback.

### Draft length, revisited

With a flat verify cost the draft-length trade-off from part 2 (n = 4 and 5 lost to 3) should move. Speed subset,
budget 2048, on the XMX runtime: three runs per setting, alternating (each row one pass of the 4 tasks):

| draft length | median tok/s per run | acceptance range |
|---|---|---|
| 3 (served) | 61.5, 61.4, 63.4 | 0.52-0.69 |
| 4 | 65.0, 63.8, 63.0 | 0.48-0.57 |
| 5 | 62.0 (one run) | 0.40-0.65 |

Mean 63.9 vs 62.1 for 4 vs 3 (+3 %), with the third pair reversed (63.0 vs 63.4), one sampling fail in the first n = 4
run, and a four-task subset whose run-to-run spread is about 2 tok/s. Not enough to move the served setting; draft
length 3 stays, and the next look should be a full-suite pair rather than more subset runs.

## 2. Strata 0.1.45

Release notes claim +3.5/+4.2 % decode on this GPU for the IQ3_S models with identical answers. Engine rebuilt from the
tag inside the same image as before, `setup.sh --backend sycl --update --yes`, configs untouched. Measured with the
upstream `tools/ab_engine.py` (restarts the engine per arm, alternates arms, 3 rounds, greedy, three short chat prompts of
~30 tokens with 200-256-token answers plus one 6345-token prompt), new vs old engine binary on the same config:

| model | prompt | decode new | decode old | change |
|---|---|---|---|---|
| Flash-Next IQ3_S | chat 0/1/2 | 58.2 / 51.5 / 51.1 | 56.9 / 49.5 / 49.2 | +2.3 / +4.0 / +3.9 % |
| Flash-Next IQ3_S | 6345-token prompt | 46.7 | 45.1 | +3.5 % |
| Swift 1.5 IQ3_S | chat 0/1/2 | 58.3 / 51.4 / 49.9 | 56.5 / 49.4 / 48.0 | +3.2 / +4.0 / +4.0 % |
| Swift 1.5 IQ3_S | 6345-token prompt | 49.2 | 47.4 | +3.8 % |

Prompt processing unchanged within noise (71-84 tok/s on the short prompts, ~1300-1340 on the long one). All 12 greedy
answers per model are byte-identical between the two engines. Adopted. Two gotchas: `setup.sh` asks a question and dies
on EOF unless `--yes` is given; the engine build script prints its own `BUILD EXIT n` line, grep that one and not the
status of whatever wrapper ran after it.

## 3. Strata PR #2804 "KREG" (opt-in register-blocked mmvq)

Register-blocked Q4_K/Q5_K/Q6_K/IQ4_XS mat-vec for 1-8 columns, behind `STRATA_MMVQ_KREG=1`. It conflicts with 0.1.45's
planar Q6_K copy path in `native_q6_k_mmvq`; resolved by trying the planar path first, then KREG. Built as a second engine
binary in a git worktree so it can be selected per service. Same A/B as above, KREG vs plain 0.1.45:

| model | chat 0/1/2 | long |
|---|---|---|
| Flash-Next | 61.0 / 55.9 / 52.7 vs 58.2 / 51.4 / 51.1 (+4.8 / +8.8 / +3.1 %) | 47.2 vs 46.7 (+1.1 %) |
| Swift 1.5 | 60.8 / 53.3 / 51.8 vs 58.3 / 51.4 / 49.9 (+4.3 / +3.7 / +3.8 %) | 50.9 vs 49.2 (+3.5 %) |

The greedy answers are **not** identical (Flash-Next 6 of 12, Swift 3 of 12 unchanged), which the PR states up front:
the accumulation order differs. Kept opt-in, not default, because the lab's bench baselines are greedy-reproducible and
+4 % is not worth losing that.

## 4. What the two gains have in common

Both are the same bottleneck seen in part 2 from two sides: on this GPU the memory wall sets single-token decode (29.5
tok/s for the 27B at Q4_K_S, unchanged by any of this), and everything above it comes from doing more columns per weight
read without paying issue slots for them. The XMX units are the right place for that work; register blocking in plain
SYCL is a weaker version of the same idea.

## 5. The rest of the list, and why not

| item | verdict | reason |
|---|---|---|
| #30374 Q4_0/Q8_0 reorder prefill fix, #30775 Q8_0 dequant | n/a | the Orca GGUF is Q4_K (89.5 % of tensors), Q5_K (12), Q6_K (1); no Q4_0/Q8_0 weights. Check with `gguf-dump` before believing a quant-specific PR applies |
| #30239 ESIMD FFN | included | already in the PR head above (merged in master) |
| Strata #2566 message-boundary cache | n/a | needs `--batch >= 2` and `STRATA_CACHE_MESSAGE_BOUNDARY=1`; single-user service runs batch 1 |
| #30745 slot save/restore | n/a | not used by the agent |
| #30738 Vulkan | out of scope | Bolt/Vulkan not in this pass |
| #27210 adaptive draft length | done before | measured and rejected in [08](08-dead-ends.md) |
| new llama.cpp base b11721 alone | measured | 46.6 with our kernels, 41.0 without: nothing for this model beyond the PR |

## Reproduce
```sh
# XMX runtime: PR head + patches 0009/0010, oneDNN on (see scripts/build-sycl-patched.sh header for DNNL_DEV)
git fetch origin pull/29864/head && git checkout FETCH_HEAD && git apply patches/00{09,10}-*.patch
SRC=$PWD OUT=/path/runtime-sycl-xmx DNNL_DEV=/path/dnnl-devel scripts/build-sycl-patched.sh
# per-kernel timings, correctness and perplexity A/B
LLAMA_RT=/path/runtime-sycl-xmx scripts/perf-shapes-sycl.sh; scripts/kernel-check-sycl.sh; scripts/ppl-ab-sycl.sh
# XMX off for a same-binary control
SERVE_SYCL_ENV=GGML_SYCL_ENABLE_XMX_ESIMD=0 scripts/serve-orca-sycl.sh
# Strata engine A/B (upstream tool; use a loopback config without api_key)
python tools/ab_engine.py cfg.json new:exe=/path/new/strata old:exe=/path/old/strata --rounds 3
python tools/ab_engine.py cfg.json kreg:STRATA_MMVQ_KREG=1 base:STRATA_MMVQ_KREG=0 --rounds 3
```

## 6. What is left on the table (measured 2026-10-19, not adopted yet)

**Attention at depth is now the dominant cost.** Plain decode sits at 466 GB/s effective over the 15.8 GB of weights
(peak 608 GB/s), but `llama-bench` tg128 falls from 29.5 at depth 0 to 22.8 at 16k and 18.7 at 32k with q8_0 KV. The
Gated DeltaNet layers are noise (4.5 us per layer). The flash-attention decode kernel at this model's shape (head size
256, 4 KV heads, 6 queries per KV head) is what costs it, per `test-backend-ops perf`:

| KV type | kv length | us per call | KV bytes read | effective GB/s |
|---|---|---|---|---|
| f16 | 16384 | 420 | 67 MB | 160 |
| f16 | 65536 | 1636 | 268 MB | 164 |
| q8_0 (2 KV heads, GQA 12) | 32768 | 702 | 36 MB | 51 |
| f16 (2 KV heads, GQA 12) | 32768 | 440 | 67 MB | 153 |
| f16, GQA 1 (reference) | 32768 | 474 | 268 MB | 567 |

q8_0 KV takes 1.6x the time of f16 for half the bytes. Sixteen attention layers at 1.2-1.4 ms each is the whole gap
between depth 0 and depth 32k. Two things follow:

- **f16 KV instead of q8_0**: tg128 29.6 / 24.7 / 21.4 at 0 / 16k / 32k against 29.5 / 22.8 / 18.7 (+8 % at 16k,
  +15 % at 32k), prefill unchanged. f16 at a 131k window costs the same VRAM as q8_0 at 262k, so if 131k is enough
  (the agent compacts at 48k) this is a free change.
- **PR #30606** routes single-query decode at head size 256 on Battlemage to the FA vector kernel (f16 KV only); its
  author measured Qwen3.8-27B tg128 at 32k 18.6 to 20.6. Stacks with the above. Not built here yet.

**DFlash2 block drafter.** An official DFlash2 drafter for the base model exists as GGUF (1.1 GB at Q4_K_M) and the
DFlash2 code is in llama.cpp since August. On the XMX runtime the verify cost is flat to 80 columns, so a 7-token block
costs what a 3-token one did. The SYCL `out_prod` strided-f32 patch (0001) was applied for safety, though the drafter
loaded without it. Speed subset, budget 2048, one pass per row, MTP control on the same binary:

| drafter | draft length | median tok/s | accepted per step |
|---|---|---|---|
| MTP + n-gram (served) | 3 | 63.2 (earlier 61.5, 61.4, 63.4) | 2.7 |
| DFlash2 | 5 | 67.5 | 3.0 |
| DFlash2 | 7 | 66.8, 74.8 | 3.1-3.2 |
| DFlash2 | 12 | 64.7 | 3.0 |
| DFlash2 + n-gram | 7 | 64.5 | 3.2 |
| DFlash2, p-min 0.5 | 7 | 61.4 | 3.4 |

Acceptance is 0.25-0.40 per drafted token, against 5.3 accepted per step the model card reports on the base model: the
drafter was trained for the base and this is a fine-tune. Per task the gain ranges from flat (the DNS question) to
+25 % (the short table task). The two drafters cannot be combined (`failed to create MTP context`). Verdict: +6-13 % on
the subset, worth a full-suite alternating pair before switching; a drafter fine-tuned on the target would be the
bigger win.

**Checked and dropped**: output head Q6_K is 1.0 GB per token (5 % of decode), Q4_K would buy ~2 % for a perplexity
cost; the Q4_K 5-8 column row pairing PR is superseded by XMX; SYCL graph record/replay is reported negligible on
Battlemage by its author; prompt-prefix reuse across agent turns already works (15-107 new prompt tokens evaluated per
turn in a three-turn session), so the server's checkpoint handling is not a cost here.

## 7. Adopted the same afternoon: f16 KV, the vector-kernel PR, DFlash2

All three of section 6 went in together, validated as one configuration against the morning's served one.

- Context 131072 with f16 KV instead of 262144 with q8_0 (same ~9 GB of KV).
- PR #30606 on top of the XMX runtime (plus the strided `out_prod` patch 0001 for the drafter graph). llama-bench
  tg128 with f16 KV: 30.0 / 26.3 / 23.5 / 18.5 at depth 0 / 16k / 32k / 64k, against 29.5 / 22.8 / 18.7 for the
  served q8_0 setup (+26 % at 32k). One caveat, stated as measured: `test-backend-ops` shows 11 additional
  `FLASH_ATTN_EXT` tolerance misses with the PR, all head size 256, 4 KV heads, kv >= 32767, 1-5 queries, NMSE
  7e-4 to 1.6e-3 against the 5e-4 threshold (2.6-4 % RMS deviation of the attention output at that depth). The two
  ERR ~1.0 failures are pre-existing on every build (upstream issue #30757). Perplexity at 32768-token chunks is
  identical to four decimals with and without the PR (2.7678; q8_0 2.7682), but that exercises the batched prefill
  path, not the single-query kernel; the long-context suite below is the practical check.
- DFlash2 drafter at draft length 7 instead of MTP + n-gram at 3.

Full 39-task suite, agent budget 512, alternating fast / served / fast / served, each row one pass:

| configuration | median tok/s | wall | passes / 39 | accepted per step | acceptance |
|---|---|---|---|---|---|
| fast, pair 1 | 78.1 | 6.9 min | 29 | 3.7 | 0.39 |
| served, pair 1 | 65.1 | 7.8 min | 30 | 3.1 | 0.60 |
| fast, pair 2 | 77.1 | 7.2 min | 30 | 3.6 | 0.38 |
| served, pair 2 | 64.6 | 8.1 min | 32 | 3.1 | 0.60 |

+20 % tok/s and 11 % less wall. Passes average 29.5 against 31 with an identical-configuration spread of 2 seen in
earlier pairs, so a small accuracy cost is possible and not excluded; it will be watched over more runs. Long suite,
one pass each, 21/21 on both:

| depth | fast decode tok/s | served decode tok/s |
|---|---|---|
| 8k | 75.2 | 61.3 |
| 32k | 66.1 | 48.7 |
| 64k | 57.7 | 36.3 |

Time to first token is unchanged (prefill is the same code). Over the day, the served Orca went from 49.6 to 78.1
tok/s median on the full suite at the same budget: XMX kernels (+30 %), then f16 KV, the vector kernel and the
block drafter (+20 %).

## 8. A newer model from the same publisher, through the same pipeline (2026-10-19)

OrcaCyber Nano 1.0 (released 2026-10-16) is a second 27B cyber fine-tune on the same hybrid skeleton as Orca, shipped
with its own DFlash2 drafter and a vision projector. The published GGUFs are IQ mixes (the 15.7 GB file is mostly
IQ4_XS) and decode at 20.5 tok/s on this GPU; the same imatrix + Q4_K_S requant as in 06 brings it to 29.9 (perplexity
ratio 1.009 to the mix, same top token 94.8 %). With the fast configuration of section 7 and its matched drafter:

| budget | model | pass / 39 | median tok/s | wall | tokens | accepted per step |
|---|---|---|---|---|---|---|
| 512 | Nano | 30 | 86.3 | 5.3 min | 25k | 4.3 |
| 512 | Orca (two runs) | 29, 30 | 78.1, 77.1 | 6.9, 7.2 min | 30-32k | 3.7 |
| 2048 | Nano | 32 | 83.9 | 10.4 min | 47k | 4.1 |
| 2048 | Orca | 32 | 67.3 | 17.5 min | 68k | 3.3 |

One pass per row. Accuracy is a tie (at 2048 each passes two tasks the other fails); the repair suite gives Nano 14/16
pass@3 like Orca. The wall-time gap is real and comes from two places: the drafter trained for the model accepts 0.44-0.48
per drafted token against 0.33-0.39 for the base-model drafter on the fine-tune, and Nano writes 20-30 % fewer tokens per
task. Neither model refused a reverse shell, a stack-overflow proof of concept or a banner-grabbing scanner, despite the
newer one's "defense" branding. Lesson that generalizes: a publisher's matched DFlash2 drafter is worth more than any
kernel knob once the verify step is flat, and an IQ-type release should be requantized before judging speed on Arc.
