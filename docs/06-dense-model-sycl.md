# A dense 27B on the B70: Vulkan -> SYCL, drafting, quant, KV, power

The model here is a dense Qwen3-based 27B fine-tune distributed as IQ4_XS (imatrix). Everything applies to any
dense 20-35B GGUF on this card.

## Backend: SYCL wins for dense models
Same card, same model (Q4_K_M), all 65 layers on GPU:

| | host Vulkan b10430 | SYCL b11541 in the oneAPI container (ubatch 2048, FA on) |
|---|---|---|
| prefill pp2048 | 517 tok/s | 1,259 tok/s |
| prefill at 32k depth | 130 | 1,009 |
| decode, no draft | 25.2 | 28.0 |
| decode at 32k depth | 10.6 | 20.3 |
| decode with MTP drafting | 15-27 | 36.7 (49 % acceptance) |

The Vulkan build of the *same* release (b11541) got to 647 pp2048 / 25.8 tg, so most of the gap is the backend,
not the version. Flash attention off: 614 vs 847 tok/s prefill at depth, so keep it on. Draft length 5 was slower
than 3. Note the contrast with `05-bolt-speedups.md`: for a hybrid model with IQ2_XXS/NVFP4 tensors the ranking
flips, so measure both backends per model.

## Drafting
`--spec-type draft-mtp,ngram-mod --spec-draft-n-max 3`: MTP (the model's own multi-token-prediction head) gives
+30 % decode, adding the n-gram drafter another +7 %. Prompt-cache reuse (`--cache-reuse`) is unsupported for this
hybrid-attention model and was dropped.

## Quantization study (imatrix requant from IQ4_XS)
IQ4_XS decodes at 13 tok/s on Vulkan/BMG and IQ3_M at 11 on SYCL: the IQ dequant paths are slow on Intel. A
K-quant requant of the already-quantized file, made with an importance matrix from 24x2048 tokens of a technical
corpus, fixed that. Quality measured as perplexity ratio and same-top-token agreement against the IQ4_XS source on
8 held-out chunks (`llama-perplexity --kl-divergence`):

| candidate | ppl ratio | same top token | decode tok/s | pp2048 | verdict |
|---|---|---|---|---|---|
| **Q4_K_S (imatrix)** | **1.0085** | **95.2 %** | **29.6** | **1,284** | chosen |
| Q4_K_M (no imatrix) | 1.0154 | 94.0 % | 28.0 | 1,259 | |
| Q4_0 | 1.0326 | 90.3 % | 29.1 | | rejected: quality |
| Q3_K_M (imatrix) | 1.0375 | 90.0 % | 28.8 | | rejected: quality, no speed gain |
| IQ3_M (imatrix) | 1.0285 | 89.3 % | 11.2 | | rejected: IQ decode slow on SYCL |

Lesson: on this card prefer K-quants (or Q8_0 for small tensors) over IQ types, requantize *with* an imatrix, and
measure KL/perplexity, because the bench scores are too coarse to see a 1-3 % quality change.

## Context and KV cache
262,144 context with an 8-bit KV cache (8 GiB) fits beside the ~16 GB model; f16 KV does not. q8_0 KV costs
~12 % decode at depth. Speed falls with depth: prefill 1,259 fresh / 682 at 64k / 527 at 128k; decode 28 / 13 / 9.
A chat harness that auto-compacts at half the window keeps the depth available without paying for it by default.

## Power cap (sysfs `power1_cap`, SYCL backend)
| cap | pp2048 | decode | average draw |
|---|---|---|---|
| 230 W (firmware max) | 1,153 | 28.1 | 185 W (peak 240) |
| 190 W | 1,047 | 27.0 | 164 W |
| 160 W | 909 | 24.0 | 144 W |
| 130 W | 697 | 17.4 | 128 W |

190 W costs ~4 % decode and ~9 % prefill for a noticeably quieter box. The cap resets at boot.

## Greedy decoding loops
A server that decodes greedily when the client sends no sampling parameters produces sentence loops with these
models. Always send the model card's sampling (temperature 1.0, top_p 0.95, top_k 20, presence penalty 1.5 for the
Qwen3 family); penalties cost 1-11 % decode in Strata. A client-side guard that cuts a reply after the same
sentence repeats four times is a cheap backstop.
