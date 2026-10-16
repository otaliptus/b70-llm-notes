# Local LLMs on an Intel Arc Pro B70: benchmark, speedups, notes

Working notes from two weeks of running 20-80B open-weight models on a single Intel Arc Pro B70 (32 GB) with
llama.cpp (SYCL and Vulkan backends) and Strata, plus the small benchmark harness used to compare them. Everything
here is generic: no personal paths, keys or hostnames; scripts take their locations from environment variables.

## Five results worth knowing
1. **A hard thinking budget (~2048 tokens) beats unlimited thinking** for every Qwen-family reasoning model tried:
   higher score *and* 2-4x less wall time, because unlimited runs regularly think to the output limit and never
   answer. [`docs/04-thinking-budget.md`](docs/04-thinking-budget.md)
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
   an un-fused gate/up at n >= 3 make 4-column mat-vecs 20-40 % faster: 42 -> 48 tok/s served, plain decode and
   perplexity unchanged. [`docs/10-sycl-speculative-kquant.md`](docs/10-sycl-speculative-kquant.md)
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
| [`docs/10-sycl-speculative-kquant.md`](docs/10-sycl-speculative-kquant.md) | llama.cpp SYCL: where a speculative step goes, probabilistic MTP drafting, faster multi-column K-quant mat-vecs; 36 -> 48 tok/s served |
| [`bench/`](bench/) | the harness (stdlib Python + Docker sandbox) and the generated [`report.md`](bench/results/report.md) |
| [`scripts/`](scripts/) | serve / build / sweep / profile / requant / perplexity scripts |
| [`patches/`](patches/) | llama.cpp patches. b11669: SYCL `out_prod` strided f32 fix (DFlash on SYCL); Vulkan: IQ2_XXS/NVFP4 integer-dot paths, dedicated NVFP4 mat-vec, top-k router fix, Intel policy + knobs, perf shapes. b11696: SYCL K-quant multi-column mat-vecs (0008), MTP draft timing (0009), Qwen3.5-class perf shapes (0010) |

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
b11669 and b11696. Dates 2026-10-10 to 2026-10-16.

## License
MIT, see [`LICENSE`](LICENSE).
