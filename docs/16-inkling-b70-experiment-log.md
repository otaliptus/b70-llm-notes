# Inkling on the B70, November 23 experiment record

Use **0059** as the reference for the four-hour evening session. On one Intel Arc Pro B70 it passed 36/39 on the graded suite at 105.59 decode tok/s. The 14-turn conversation averaged 81.91 decode tok/s, and the three-turn 100K source task averaged 53.87. The optional **0060** build reached 55.09 on the 100K task and slowed the 40K source task from 81.28 to 74.83. It reproduced all 34 fixed-seed responses checked across everyday tasks, chat, source reading and compaction. The graded-suite score belongs to 0059. [Release handoff and per-workload prefill, decode and quality results](15-inkling-nextsycl.md)

## Identify a measured build

The target was AtomicChat Inkling-2-Mini-Uncensored AD-Q4_K. DSpark used a Q8_0 drafter with a 32,768-row proposal head. The target scored all 200,058 vocabulary entries. The model declares a 1,048,576-token context. The evening runs configured 131,072 tokens and measured up to 126,458 tokens of actual input. [Build instructions and context configuration](15-inkling-nextsycl.md#reproduce-the-runtime)

| Model file | Size in bytes | SHA-256 of the installed GGUF |
|---|---:|---|
| `Inkling-2-Mini-Uncensored-AD-Q4_K.gguf` | 16,525,559,776 | `36e99f1f56ca33cb356f19d7e3822938f65af1cde978ad13170238928d4c955f` |
| `dspark-Inkling-2-Mini-Q8_0-sub32k-unc.gguf` | 499,403,136 | `654e4661941374a9f9cef6324109a23192fab9a49b055ca82bcdf0457b6cb1a0` |

Both builds start at nextsycl commit `e3f1fd891b7fda6f478a326d913d28893a4e9811`. Applying the published patches in order produced the source trees below. The binary hashes identify the files used on the benchmark host. Its oneAPI Docker image was tagged `strata-sycl-dev` with local image ID `sha256:ef55c795435ff7aba43e8ea7da2c8e5902af29925020f27c43b1dceef5fb5a26`. [Source patches](../patches/)

| Artifact | 0059 default | 0060 optional |
|---|---|---|
| Patched Git tree | `661e2787db9f74e854df9f0bd22ab82f78fe92a0` | `a184395d07d9b28c077ef73f41ea59220ff1dd9e` |
| `nextsycl` SHA-256 | `95bc554f8d7a0d6ea67df395e1bdb5ad220a05d7cb0372e0bdff318491cef45e` | `640f078a32c116353bfe914283b558966af6cbd4c0cdba573bf4e58341c35c33` |
| `libnextsycl-llm.so` SHA-256 | `6ef76b8cc944047eb2e5f8cbbb55d2222fd380b167f80bdd6f78a15a132e5ed8` | `6ef76b8cc944047eb2e5f8cbbb55d2222fd380b167f80bdd6f78a15a132e5ed8` |
| `silo/libnextsycl-inkling2.so` SHA-256 | `ae66e2815da167a2ccd2ea80fe766946a206aac19cc8b974fcc4ed3e1101b9c3` | `fe73b204761b581e365d8defa39b567b5bf63698ab97fd3b4902b837fc75f68c` |

## Repeat the graded suite

On the benchmark host, the workspace launcher `coop-lab/bench/inkling-unc-native-serve.sh` selects 0059 by default. Setting `DRAFT_WINDOW=2048` selects the installed 0060 build. The portable launcher takes an explicit `RUNTIME` directory for either build. The measured 0059 configuration selected an 8,192-token prefill chunk and a 6,144 MiB host checkpoint budget. [Portable launcher](../scripts/serve-inkling-nextsycl.sh)

With 0059 listening on the default port and `API_KEY_FILE` pointing to the server's key, run this from the notes checkout. The original runner, task definitions and client match the public copies byte for byte. The public grader makes the Python container image configurable. The command selects the image used in the original run and gives each replay a fresh name because the runner resumes completed tasks by name. [Graded runner](../bench/run_bench.py) · [Tasks](../bench/tasks.py) · [Grader](../bench/grading.py)

```sh
BENCH_SANDBOX_IMAGE=strata-sycl-dev python3 bench/run_bench.py \
    --name "inkling-0059-$(date -u +%Y%m%dT%H%M%SZ)" \
    --url http://127.0.0.1:18081/v1 --key "$(cat "$API_KEY_FILE")" \
    --model inkling-2-mini --temperature 1 --top-p 1 --top-k 0 \
    --extra '{"chat_template_kwargs":{"reasoning_effort":"low"}}' --wait 120
```

The 39-task run samples fresh answers. For a fixed-seed comparison, the public context client can replay the conversation, 100K and 126K source tasks, branching and compaction. The release guide gives the fixture revision and the cold-cache procedure. [Context replay commands](15-inkling-nextsycl.md#replay-the-long-conversations)

## Experiments that stayed out of the release

These comparisons used the same B70. A kernel timing, a fixed-length decode, and a served conversation each measure a different part of inference. The log names below are relative to the evening archive on the benchmark host.

| Experiment and measurement | Outcome | Local evidence |
|---|---|---|
| Expand groups of consecutive experts during wide prefill, keeping each matrix product's reference row grouping | Reading 99,978 fresh tokens took 59.978 s versus 56.499 s on the reference. All three source replies matched in visible text, reasoning hash and length. | `logs/packexp-priority`, `results/packexp/100k/` |
| Assign two or four long-attention splits to one workgroup | Across six target and drafter kernel cases, relative throughput was 0.9972× and 0.9377× respectively. Float outputs and quantized outputs matched byte for byte. The experiment ended at the kernel comparison. | `logs/querytiles-priority`, `results/tile-reference/` |
| Pack the Q8 controllers together with the Q8 output-head optimization | All 28,311,552 controller coefficients and 38,411,136 tested target scores reproduced their references exactly. On one short prompt, 0059 → combined decode was 118.99 → 119.79, 123.97 → 124.84 and 126.19 → 126.99 tok/s over three seeds. Only the output-head change went into 0060. | `logs/qctrl-model-weights`, `logs/headctrl-exact`, `logs/headctrl-pair` |
| Group dense projections | Across the same three seeds, decode fell from 118.34 → 117.00, 123.99 → 122.01 and 126.22 → 124.19 tok/s versus the experimental `stable2` control. All three output hashes matched. | `logs/densegroup-pair` |
| Host allocation changes, router union, and link-time optimization | Each individual change measured within 0.07 tok/s of its paired `stable2` control across all three seeds. | `logs/host-pair`, `logs/uroute-pair`, `logs/lto-pair` |
| Group host checkpoint transfers | Across six everyday requests, both builds averaged 103.34 decode tok/s. The fourteen-turn chat moved from 81.90 to 81.92 and the two 40K source turns from 81.33 to 81.32. | `logs/transfer-priority`, `results/{stable2,transfer}/chat/realchat.jsonl` |
| Reduce the drafter head further | On the same experimental runtime the 16,384-row head measured 102.15 tok/s on everyday tasks and 80.47 on chat. The 8,192-row head measured 99.96 and 79.30. The shipped head has 32,768 rows. | `logs/heads-grid`, `results/head-{16384,8192}/chat/` |
| Vary draft count and confidence on a 100K conversation | Five completed three-turn runs ranged from 40.06 tok/s with seven proposals and a 0.25 threshold to 53.68 with four and 0.50. These were runs on the experimental `conv` build. | `logs/long-drafts`, `results/sweep-conv-*/100k/` |

The three-seed decode comparisons used a 56-token prompt, a fixed 768-token generation and seeds 404, 917 and 188. The harness alternated which binary ran first. For proposal-window tuning, the preliminary 100K grid contains four completed window sizes. The release guide reports the separate installed configuration, its 40K regression and the 4.25 GiB checkpoint-budget test. [Optional 0060 results](15-inkling-nextsycl.md#optional-long-context-drafting-with-patch-0060)

## Find the original evidence

The host archive is `/data/inkling-perf-lab/nov23-evening/`. GitHub carries the source patches, the portable launcher and clients, and the per-request counters and digests. The archive retains full graded responses, transcripts, raw binary score arrays, experimental source trees and local build logs. [Published measurement files](../bench/results/)

| Question | Location inside the archive |
|---|---|
| Which requests made up the 0059 score and near-limit timings? | `results/deploy59/bench39/results/graded.json` and `results/deploy59/126k/`. The 0059 wide-prefill experiments with an explicit 8,192-token chunk used `results/precision/`. |
| Where are the six everyday tasks and the 40K source pair? | `results/precision/chat/realchat.jsonl` records the requests, answers, checks and timings. The tasks cover a customer reply, shift handoff, invoice reconciliation, calendar overlap fix, engineering brief and source walkthrough. The local driver `e2e.py` loads `../realbench-MZQgWO/realchat.py`. |
| Which runs went through the installed release and public client? | `results/public-replay-chat/` and `results/public-stable-100k/`. The matched-prompt external comparison is in `results/public-xpu-comparison/`. |
| Where are the optional build and storage controls? | `results/combo60/` holds the 0060 source, chat, compaction and score runs. `results/fast-economy/126k/` and `results/lean-economy/126k/` hold the two 4.25 GiB controls with a 2,048-token prefill chunk. |
| Where are the target score checks? | `results/control/score-ref/` contains reference arrays. `results/precision/{scores,edge}/` holds wide-prefill and boundary cases. `logs/quality-depot` records the independent logistics fixture. |
| Which other trials already have measurements? | Drafting and proposal attention include `logs/{draft-grid,adaptive-real,spec-grid,race-grid,window-sweep,sink-workloads,lookup-retry}`. Driver settings and shared memory compression are in `logs/{driver-grid,usm-compression}`. Some sweeps ended early. Completed per-case counters are under `results/`. |
| Which comparisons finished without a measurement? | The queued `group-micro`, `groupgem-workloads` and `transfer-disk` logs are empty. The extra `stable2` graded repeat was retired. `retired-cases/` records the individual cases taken out of the queue. |
| How can I verify the source used for each release? | `prepare-patch59.sh` and `prepare-option60.sh` replay the release series into separate Git indexes. Applying the public patches to the pinned upstream commit reproduces both tree IDs above. |

For 0059, five core tests and two Inkling tests passed. On the host the serving suite first failed one speech test with FFmpeg missing. A second host attempt failed two video tests. In the oneAPI image all 238 serving tests passed. For 0060, all three Inkling tests passed, including the proposal-window boundary test. Each installed build also served an authenticated 63-token completion with the expected answer. The logs are `logs/deploy59-units`, `logs/deploy59-units-ffmpeg`, `logs/deploy59-units-in-image`, `logs/combo60-units`, `logs/installed-0059-smoke` and `logs/optional60-launcher-smoke`.
