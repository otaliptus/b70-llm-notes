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

## Dense 27B
- IQ4_XS and IQ3_M quants on Intel: 11-13 tok/s decode (IQ dequant paths are slow on both backends here).
- Q4_0 and Q3_K_M requants: 3-4 % perplexity for no speed gain over Q4_K_S.
- Flash attention off: prefill at depth drops 847 -> 614.
- Draft length 5: slower than 3.
- f16 KV at 262k: does not fit beside the model.
- `--cache-reuse`: unsupported for the hybrid-attention architecture.

## Thinking budgets
- Unlimited thinking: 6-12 of 39 answers lost to the output limit, 2-4x the wall time.
- Soft `reasoning_effort: low`: tail of capped answers remains, wall time barely improves.
- Temperature 0.6 vs 1.0 at b2048: ±4 tasks in opposite directions for two models, i.e. noise.

## Tooling
- A home-grown parallel HTTP range downloader that re-requested ranges on retry produced duplicated bytes; the
  Hugging Face CLI with `hf_xet` was reliable (and still bound by the 11 MB/s line).
- `pkill -f` with a pattern that matches the calling shell kills the calling shell.
