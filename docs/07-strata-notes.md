# Strata on the B70 (Qwen3.8 Flash-Next and Swift 1.5)

Strata is the engine used for the two Qwen3.8-era MoE models. Notes specific to running it on this card:

- **Build**: the SYCL engine is built ahead-of-time for the Battlemage target (`bmg-g31`) inside the same oneAPI
  image used for llama.cpp, following the project's Intel build notes. The 0.1.44 build with the project's SYCL
  expert profile raised prompt reading on a 15.5k-token prompt from 250-525 tok/s to ~1,180 tok/s; decode 47-48 tok/s
  at 0-15k depth (was 43-46 at 66k on the previous version).
- **262k context**: `--kv-resident 32768` streams the KV cache from ~3 GiB of pinned RAM with `--vram-reserve-mib
  2048` and `--prefill 4096`. Measured 859 tok/s prompt reading at 25k and 848 at 74k, decode 52 short / 47 after a
  74k prompt, 2.2 GB VRAM spare. The project's docs warn that 262k *without* kv-resident livelocked a B70; keep it.
- **Thinking budget** is per request (`reasoning_budget_tokens`), which is what made the budget matrix cheap to run
  on these two models (no server restart). The same key at the top level of `strata-<model>.json` makes it the server
  default (a request's own value wins, 0 = none); both configs now carry 2048, see `04-thinking-budget.md`.
- **0.1.44.1** (2026-10-16) is a hotfix release (Metal launch sizes, CPU NVFP4 ISA guards, Responses API honours
  `reasoning_effort`, setup re-run keeps the model size); nothing in the SYCL engine changed, so no speed change on the
  B70. Updating is: `git fetch --tags && git checkout <tag>`, rebuild the engine with the documented container command
  (about 7 minutes, AOT for `bmg-g31`), `./setup.sh --backend sycl --update`.
- **Sampling**: Strata decodes greedily when the request has no sampling block, which loops. Put the Qwen
  recommendation (temp 1.0, top_p 0.95, top_k 20, presence 1.5) in the config and send it from the client.
- **Control vector**: Strata ships a per-layer control vector (`--cvec-mode project`, layers 4-44) that removes one
  direction at runtime. The `proj` benchmark scenario measured it: 36/39 vs 34 with it off, within noise, i.e. no
  measurable cost on these tasks (the vector's own metadata reports mean KL 0.19 vs the stock model). Requests can
  opt out per call with `"experimental_speed_projection": false`, which the bench does for the "stock" rows.
- **MTP draft length** (`--spec N`): 3, 4 and 5 measured 41-50 tok/s on the same four prompts with the 2048 budget;
  the differences are noise, so the default 4 stays (unlike llama.cpp's DFlash drafter on Bolt, where 3 was clearly best).
- **Engine updates** (2026-10-19): 0.1.45 is +2-4 % decode on the IQ3_S models with byte-identical greedy answers; the
  opt-in KREG kernels (PR #2804, `STRATA_MMVQ_KREG=1`) another +4 % with changed answers. A/B method, tables and the
  two `setup.sh`/build-log gotchas in [12](12-sycl-xmx-upstream-pass.md).
- **Engine update to 0.2.6 / engine 0.1.51** (2026-10-26; 14 tags after 0.1.45: the 0.2.x number is the app and the
  Terminal, the Qwen engine inside is 0.1.50-0.1.51). Build and setup went as before: a fresh AOT build in the unchanged
  image (25 minutes), then `setup.sh --backend sycl --update --vision no --yes`; setup's closing warning that the engine
  "is still 0.1.12" is it reading the version out of the SYCL shell wrapper named as the config's exe, cosmetic. Same A/B
  method as for 0.1.45 (`tools/ab_engine.py`, greedy, alternating fresh engines; 3 rounds per IQ3_S model at a
  6,345-token prompt, then two 4-arm runs on Flash-Next at 6,345 and 21,102 tokens): decode unchanged on both models
  (51.4 vs 51.5 tok/s on short chats, 46-50 after the long prompts). The new Battlemage default for prompt reading
  (fused int8 expert products, `STRATA_PREFILL_MMQ` unset = auto) read the 6,345-token prompt 15 % slower here (1,112 vs
  1,313 tok/s) and its greedy answer on that prompt differed and varied between restarts: auto times the int8 and FP16
  chains at every start, and the cache file it writes lands in the container's home, gone with `--rm`
  (`STRATA_CACHE_DIR` on the mounted folder would keep it). `STRATA_PREFILL_MMQ=off` is the FP16 chain and matched
  0.1.45 exactly: the same answers on all 8 prompts, 1,316 vs 1,313 tok/s at 6.3k and 1,400 vs 1,396 at 21k. `=1`
  forces int8: +4.5 % at 6.3k, +2.5 % at 21k, deterministic, long-prompt answers differ (the project's own numbers:
  KL 0.0002 nats, top-1 99.7 %); `--prefill 8192` on top was slower here (1,145-1,322 at 21k, decode after it 3-7 %
  down). Two gotchas: the launcher `strata-sycl.sh` drops every `STRATA_*` variable whose value is `0`, so the
  documented `=0` never reaches the engine (a non-digit value such as `off` is parsed as 0 and gets through); and the
  server appends to a reused engine log, so a grep over the file also shows the previous session's lines. The configs
  here carry `"env": {"STRATA_PREFILL_MMQ": "off"}`: parity with 0.1.45 by default, int8 as an opt-in, the same call
  as for the KREG kernels. Not reported upstream yet.
- **A GGUF that keeps the MTP head inside the file** (2026-10-23, bottlecapai's ThinkingCap-Qwen3.8-Flash-Next-GGUF:
  `qwen4exp.block_count = 49`, `nextn_predict_layers = 1`, 32 `blk.48.*` tensors across shards 3-7, 49-entry
  `attention.compress_ratios` / `recurrent_layers`; llama.cpp's own converter drops the head, so no earlier Flash-Next
  GGUF looked like this) does not load on 0.1.45 as shipped, in three steps: `tools/iq_pack.py` writes a 49th row into
  `native_experts.txt` which the engine refuses as "a malformed line"; with that fixed, `read_geometry` checks all 49
  `compress_ratios` entries against the 48-layer interval rule ("compress_ratios[48] = 4 ... makes layer 48 linear
  attention"); with the header trimmed, the native-dense loader's layer filter has no upper bound and refuses
  "tensor absent from canonical table: blk.48.ffn_gate_shexp.weight". The engine does model an in-file MTP block
  (`n_mtp_layers`, subtracted in `family_layout.hpp`); these three places just do not use it. What worked without
  rebuilding the engine: [`patches/0011`](../patches/0011-strata-iq-pack-skip-in-file-mtp-head-block.patch) for the
  packer (layers = `block_count - nextn_predict_layers`; the head's block is left out of the expert table and the
  dense index; the packer's 24 unit tests pass) and
  [`scripts/gguf-strip-mtp-block.py`](../scripts/gguf-strip-mtp-block.py), which rewrites the eight shards as the
  48-layer trunk (headers fixed, `split.tensors.count` 1256 -> 1224, the 32 head tensors dropped, every kept tensor
  sha256-verified, 5 minutes for 90 GiB on an NVMe). Pack the rewritten set with `iq_pack.py --compat-bf16` as in the
  project's compatibility notes (IQ4_XS experts are served natively; 363 small projections rounded to BF16, max error
  0.014; under half a minute) and serve it with the base model's `--mtp` runtime, which the vendor card says is the unchanged base
  head. Results in [03](03-results.md#thinkingcap). Not reported upstream yet.
