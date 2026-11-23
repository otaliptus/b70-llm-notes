#!/usr/bin/env python3
"""Add a reduced-vocabulary draft head to the Inkling 2 DSpark drafter GGUF (see docs/14, patch 0048).

The DSpark markov head runs one lm-head pass over the full 201k vocabulary per drafted position (492 MB of Q8_0 for
the uncensored target). This copies the target's lm-head rows (and the drafter's markov_w1 rows) for the N most
frequent output tokens into the drafter file:
  dspark_out_sub.weight    Q8_0 [n_embd, N]   rows of the TARGET's output.weight (byte copies, no requant)
  dspark_markov_w1_sub     BF16 [rank, N]     rows of markov_w1 (the previous-token embedding of the markov head)
  dspark_out_sub_ids       F32  [N]           full-vocabulary token id of each row (exact in f32: ids < 2^24)
Token order: by count in a frequency file (json list of [id, count], e.g. from scripts/token-freq.py over the target
model's own answers), then the lowest unused ids (BPE merge order ~ frequency) up to N.

usage: make-dspark-subvocab.py DRAFT.gguf TARGET.gguf FREQ.json N OUT.gguf
Run with GGUF_PY=<llama.cpp tree>/gguf-py and a Python that has numpy (python -I).
"""
import json
import os
import sys

import numpy as np
sys.path.insert(0, os.environ["GGUF_PY"])   # gguf-py of the llama.cpp tree (run with python -I)
import gguf

draft_path, target_path, freq_path, n_sub, out_path = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]

draft = gguf.GGUFReader(draft_path)
target = gguf.GGUFReader(target_path)
tt = {t.name: t for t in target.tensors}
dt = {t.name: t for t in draft.tensors}
out_w = tt["output.weight"]
assert out_w.tensor_type == gguf.GGMLQuantizationType.Q8_0, out_w.tensor_type
w1 = dt["markov_w1.weight"]
assert w1.tensor_type == gguf.GGMLQuantizationType.BF16, w1.tensor_type

n_vocab_unpadded = int(draft.fields["dflash.vocab_size_unpadded"].contents())
freq = json.load(open(freq_path))
ids = [int(i) for i, _ in freq if 0 <= int(i) < n_vocab_unpadded]
seen = set(ids)
for i in range(n_vocab_unpadded):
    if len(ids) >= n_sub:
        break
    if i not in seen:
        ids.append(i)
        seen.add(i)
ids = np.array(sorted(ids[:n_sub]), dtype=np.int64)   # sorted: argmax ties resolve to the lowest full id
print(f"subset: {len(ids)} tokens, max id {ids.max()}")

out_rows = np.asarray(out_w.data)          # [n_vocab_padded, bytes_per_row] uint8 (Q8_0 blocks)
w1_rows = np.asarray(w1.data)              # [n_vocab, rank] uint16 view of BF16 (or raw bytes)
print("output.weight", out_rows.shape, out_rows.dtype, "markov_w1", w1_rows.shape, w1_rows.dtype)
sub_out = np.ascontiguousarray(out_rows[ids])
sub_w1 = np.ascontiguousarray(w1_rows[ids])

w = gguf.GGUFWriter(out_path, arch=draft.fields["general.architecture"].contents())
for name, field in draft.fields.items():
    if name.startswith("GGUF.") or name == "general.architecture":
        continue
    vtype = field.types[0]
    if vtype == gguf.GGUFValueType.ARRAY:
        sub = field.types[-1] if len(field.types) > 1 else gguf.GGUFValueType.INT32
        w.add_key_value(name, field.contents(), vtype, sub_type=sub)
    else:
        w.add_key_value(name, field.contents(), vtype)
w.add_key_value("dflash.dspark.sub_vocab", len(ids), gguf.GGUFValueType.UINT32)
for t in draft.tensors:
    w.add_tensor(t.name, np.asarray(t.data), raw_dtype=t.tensor_type)
w.add_tensor("dspark_out_sub.weight", sub_out, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
w.add_tensor("dspark_markov_w1_sub.weight", sub_w1, raw_dtype=gguf.GGMLQuantizationType.BF16)
w.add_tensor("dspark_out_sub_ids.weight", ids.astype(np.float32))
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("wrote", out_path)
