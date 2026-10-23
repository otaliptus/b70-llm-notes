#!/usr/bin/env python3
"""gguf-strip-mtp-block.py - rewrite a split qwen4exp GGUF that carries its MTP head as an extra block
(block_count 49, nextn_predict_layers 1; bottlecapai's ThinkingCap-Qwen3.8-Flash-Next-GGUF is one) as the plain
48-layer trunk, so Strata v0.1.45 loads it: every shard gets block_count 48, the per-layer arrays cut to 48, the
nextn key dropped and split.tensors.count corrected, and the blk.<HEAD>.* tensors are left out.  Kept tensors are
copied byte for byte and verified by sha256; shards after the first keep having no general.architecture key
(Strata treats a shard that carries one as a first shard).  The draft layer then comes from Strata's --mtp runtime
(the vendor states the head is the base model's, unchanged).

    GGUF_PY=/path/to/llama.cpp/gguf-py python3 -I gguf-strip-mtp-block.py SRC_DIR DST_DIR 48

Needs the model's size free in DST_DIR (90 GiB for the IQ4_XS file; about 5 minutes on an NVMe).  Why this is
needed and what Strata refuses otherwise: docs/07-strata-notes.md.
"""
import sys, hashlib, pathlib, time, os
sys.path.insert(0, os.environ.get("GGUF_PY", "third_party/llama.cpp/gguf-py"))
import gguf
src_dir, dst_dir, head = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), int(sys.argv[3])
pre = "blk.%d." % head
shards = sorted(src_dir.glob("*-of-0000*.gguf"))
readers = [gguf.GGUFReader(p) for p in shards]
keep_total = sum(1 for r in readers for t in r.tensors if not t.name.startswith(pre))
print("shards", len(shards), "tensors kept", keep_total, "dropped", sum(len(r.tensors) for r in readers) - keep_total, flush=True)
def h(a): return hashlib.sha256(a.tobytes()).digest()
for p, r in zip(shards, readers):
    t0 = time.time()
    arch_f = r.fields.get("general.architecture"); arch = arch_f.contents() if arch_f else None   # shards 2+ have none
    bc = r.fields[arch + ".block_count"].contents() if arch and arch + ".block_count" in r.fields else None
    mtp = r.fields[arch + ".nextn_predict_layers"].contents() if arch and arch + ".nextn_predict_layers" in r.fields else 0
    n = None if bc is None else bc - mtp
    tmp = dst_dir / (p.name + ".tmp")
    w = gguf.GGUFWriter(tmp, arch or "qwen4exp", endianess=r.endianess)
    if arch is None: w.kv_data[0].pop("general.architecture", None)   # the writer adds it; a later shard must not carry it
    notes = []
    for f in r.fields.values():
        if f.name == "general.architecture" or f.name.startswith("GGUF."): continue
        if arch and f.name == arch + ".nextn_predict_layers": notes.append("drop " + f.name); continue
        vt = f.types[0]; st = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        val = f.contents()
        if arch and f.name == arch + ".block_count": val = n; notes.append("block_count %d->%d" % (bc, n))
        elif f.name == "split.tensors.count": notes.append("split.tensors.count %d->%d" % (val, keep_total)); val = keep_total
        elif vt == gguf.GGUFValueType.ARRAY and bc is not None and len(val) == bc: notes.append("trim %s %d->%d" % (f.name, bc, n)); val = val[:n]
        w.add_key_value(f.name, val, vt, sub_type=st)
    kept = [t for t in r.tensors if not t.name.startswith(pre)]
    for t in kept: w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    hashes = []
    for t in kept:
        w.write_tensor_data(t.data, tensor_endianess=r.endianess); hashes.append(h(t.data))
    w.close()
    r2 = gguf.GGUFReader(tmp)
    if [t.name for t in r2.tensors] != [t.name for t in kept]: sys.exit("name list differs in " + p.name)
    if ("general.architecture" in r2.fields) != (arch is not None): sys.exit("architecture key presence changed in " + p.name)
    bad = [u.name for u, t, hh in zip(r2.tensors, kept, hashes) if u.tensor_type != t.tensor_type or list(u.shape) != list(t.shape) or h(u.data) != hh]
    if bad: sys.exit("byte mismatch in %s: %s" % (p.name, bad[:5]))
    del r2
    os.replace(tmp, dst_dir / p.name)
    print("%s: %d -> %d tensors, %.1f GB, %.0f s; %s" % (p.name[-24:], len(r.tensors), len(kept), sum(t.n_bytes for t in kept) / 1e9, time.time() - t0, "; ".join(notes)), flush=True)
print("strip done", flush=True)
