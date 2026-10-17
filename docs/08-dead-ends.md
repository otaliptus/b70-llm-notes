# Dead ends, so nobody redoes them

Measured and rejected, with the number that killed each one.

## Bolt (hybrid Mamba-2 + MoE, IQ2_XXS + NVFP4)
- Every SYCL env toggle (`SYCL_UR_USE_LEVEL_ZERO_V2=0`, `GGML_SYCL_FA_ONEDNN=0`, no-mmap): 29.5-30.2 tok/s, flat.
- Every Vulkan toggle (`GGML_VK_DISABLE_COOPMAT/COOPMAT2/INTEGER_DOT_PRODUCT/F16/FUSION/GRAPH_OPTIMIZE/DOT2`,
  `FORCE_MMVQ`, `DISABLE_MMVQ`, `PREFER_HOST_MEMORY`), fa on/off, q8_0 KV, ubatch 1024/2048, 8 vs 16 threads:
  41.7-42.1 tok/s. Only prefill reacts (coopmat and f16 matter there).
- Mesa 26.2.4 (sid container) vs 26.1.6: 42.4 vs 41.9 decode, 669 vs 777 prefill. Not an upgrade.
- Intel rows-per-workgroup constants (`rm_stdq`, `rm_kq`, MUL_MAT_ID rows): defaults best; `kq=4` -10 %.
- DFlash on SYCL: asserts without the out_prod patch; with it, 14-27 tok/s vs 29 without a drafter.
- DFlash draft length 7 (upstream default): 27-47 tok/s, below no-drafter on long thinking; 5 and 2 both worse than 3.
- n-gram drafter: no gain (28 vs 29).
- The earlier "DSpark" drafter file from the first upload never loaded; the repo was rewritten a day later.
- Requantizing the Mamba projections to Q5_0: same +12-13 % perplexity as Q4_0 for 1.6 tok/s less. The loss is the
  block re-layout (16-element NVFP4 scales -> 32-element blocks), not the bit width.
- Upstream Intel Vulkan PRs for mat-vec geometry: stale drafts; the mergeable one is flash-attention prefill only.
  The PREC_F32 MUL_MAT_ID CPU-fallback fix applies to a different architecture.

## Vulkan kernel work (see 09)
- Subgroup-lane lookup table via `subgroupShuffle` for the NVFP4 decode: 2.7x slower than arithmetic on Intel.
- Integer-dot NVFP4 with shared-memory tables: 920 "sends", slower than the float path.
- Integer-dot NVFP4 with SIMD-within-register decode: kernel 15 % faster, model 2.5 % slower (extra Q8_1 quantize dispatch per projection).
- Arithmetic decode inside the generic mat-vec, dropping the shared-table init barrier, 2-block unroll with hoisted loads, subgroup 32, large workgroups, every rows-per-workgroup setting: no change for NVFP4.
- Q8_0 on the integer path at n=1 (upstream default for k >= 2048 on Intel): 4 µs slower per call than the float path.
- `test-backend-ops perf` without L2-defeating batch sizes: 2x optimistic for anything under ~25 MB.
- `GGML_VK_PERF_LOGGER` inside the Mesa 26.2.4 container: op times inflated 3-10x (timestamps distort); profile on the host driver.

## Orca 27B (OrcaSAQ-2-Cyber-27B, dense)
- IQ4_XS and IQ3_M quants on Intel: 11-13 tok/s decode (IQ dequant paths are slow on both backends here).
- Q4_0 and Q3_K_M requants: 3-4 % perplexity for no speed gain over Q4_K_S.
- Flash attention off: prefill at depth drops 847 -> 614.
- Draft length 5: slower than 3.
- f16 KV at 262k: does not fit beside the model.
- `--cache-reuse`: unsupported for the hybrid-attention architecture.

## SYCL speculative path / K-quant kernels (see 10)
- Every SYCL runtime knob (graphs, fusion, ESIMD, wide MMVQ, DMMV priority, reorder, dynamic precision, oneDNN/MKL
  flash attention, f16 KV) for Orca plain decode: all within +-0.5 % of 29.4 tok/s; fusion off -4 %, ESIMD off -6 %.
  SYCL graphs are compiled in and enabled by the env var but change nothing here.
- Draft length 4 or 5 (greedy or probabilistic): slower than 3 both before and after the kernel work; each verified
  column costs ~25 % of a token.
- Pre-decoding Q4_K weights and hoisting the activation-sum term alone: Q6_K -24 %, Q5_K -12 %, Q4_K +-0 % at n=4 (the
  compiler had already hoisted it); the Q4_K gains needed more rows per sub-group.
- Four rows per sub-group for every type and size: Q6_K n=3..6 +11..43 % slower, 48..1024-row matrices +35 %, and the
  fused gate+up kernel spills (up to 3.7 KB); kept only for Q4_K/Q5_K with >= 4096 rows, GLU fusion limited to <= 2
  columns.
- `-ze-opt-large-register-file` at JIT time: n=8 shapes -30 %, the dominant n=4 shapes +-0; not adopted.
- Draft length 2 (probabilistic): 44.9 vs 47.7 / 46.5 tok/s for length 3 on the speed subset.
- Adaptive draft length by expected tokens per step time (patch 0009, `SPEC_ADAPTIVE=1`): 45.9 (cap 3) and 47.7
  (cap 4) vs fixed 3 at 47.7 / 46.5; the fixed length was already at the model's optimum for this acceptance rate.
- Building the SYCL backend without oneDNN (`GGML_SYCL_DNN=OFF`, the easy way in a runtime-only image): prefill at
  32k depth 692 vs 911 tok/s. Mount the oneDNN devel headers and build with DNN on.
- The dequant + XMX GEMM path for 9..64 columns: 3x slower per token than MMVQ at 8 (batched-bench B=12 collapses
  to 33 t/s aggregate); keep speculative batches at <= 8 tokens on this backend.

## Strata
- Draft length 3 or 5 instead of 4: within noise (41-50 tok/s on the same prompts).
- Updating to 0.1.44.1: no engine change for the B70; speed identical.

## Thinking budgets
- Unlimited thinking: 6-12 of 39 answers lost to the output limit, 2-4x the wall time.
- Soft `reasoning_effort: low`: tail of capped answers remains, wall time barely improves.
- Temperature 0.6 vs 1.0 at b2048: ±4 tasks in opposite directions for two models, i.e. noise.

## Tooling
- A home-grown parallel HTTP range downloader that re-requested ranges on retry produced duplicated bytes; the
  Hugging Face CLI with `hf_xet` was reliable (and still bound by the 11 MB/s line).
- `pkill -f` with a pattern that matches the calling shell kills the calling shell.
