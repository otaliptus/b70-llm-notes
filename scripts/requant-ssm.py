#!/usr/bin/env python3
"""Rewrite a GGUF so that selected tensors are requantized to a type with a fast mat-vec kernel on the target GPU,
copying everything else byte for byte. Written for Nemotron 3.5 Bolt, whose Mamba-2 projections (blk.*.ssm_in /
ssm_out) ship as NVFP4, a format with no hardware path on Intel Arc (Vulkan dequantizes pair by pair).

    requant-ssm.py IN.gguf OUT.gguf Q4_0|Q5_0|Q8_0 [--match ssm_in.weight,ssm_out.weight] [--from NVFP4]

Needs the gguf-py package from the llama.cpp tree (pip install -e llama.cpp/gguf-py, or set GGUF_PY to that dir).
Result on the B70 (Bolt IQ2_XXS, 80 tensors): Q4_0 48 tok/s plain / 56 with DFlash (vs 42 / 50) but perplexity
8.90 -> 10.09 (+13 %); Q5_0 46.3 tok/s and 9.99. NVFP4 uses 16-element blocks with e4m3 scales, so re-blocking into
32-element Q4_0/Q5_0 blocks loses more than the bit width suggests. Treat the output as a speed-over-quality option."""
import argparse, os, sys
import numpy as np
if os.environ.get("GGUF_PY"):
    sys.path.insert(0, os.environ["GGUF_PY"])
import gguf
from gguf.quants import quantize, dequantize

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("qtype")
ap.add_argument("--match", default="ssm_in.weight,ssm_out.weight", help="comma list of tensor-name substrings to convert")
ap.add_argument("--from", dest="src_type", default="NVFP4", help="only convert tensors currently of this type")
a = ap.parse_args()
qtype = getattr(gguf.GGMLQuantizationType, a.qtype)
src_type = getattr(gguf.GGMLQuantizationType, a.src_type)
subs = [s for s in a.match.split(",") if s]

r = gguf.GGUFReader(a.src)
arch = r.fields["general.architecture"].contents()
w = gguf.GGUFWriter(a.dst, arch, use_temp_file=False)
for f in r.fields.values():
    if f.name == "general.architecture" or f.name.startswith("GGUF."):
        continue
    vt = f.types[0]; st = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
    w.add_key_value(f.name, f.contents(), vt, sub_type=st)
plan = [(t, t.tensor_type == src_type and any(s in t.name for s in subs)) for t in r.tensors]
conv = {}
for t, do in plan:
    if do:
        shape = list(reversed([int(x) for x in t.shape]))   # gguf lists ne0 first; numpy wants rows first
        f32 = dequantize(t.data, t.tensor_type).reshape(shape)
        q = quantize(f32, qtype)
        conv[t.name] = q
        w.add_tensor_info(t.name, shape, np.dtype(np.float32), q.nbytes, qtype)
    else:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
n = 0
for t, do in plan:
    w.write_tensor_data(conv[t.name] if do else t.data); n += do
w.close()
print(f"wrote {a.dst}: {n} tensors -> {a.qtype}, {os.path.getsize(a.dst)/1e9:.2f} GB")
