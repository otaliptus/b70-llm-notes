#!/usr/bin/env python3
"""Token frequencies of a model's own output, for scripts/make-dspark-subvocab.py.

    token-freq.py URL API_KEY_FILE OUT.json TEXT_FILE...

Tokenizes each text file (e.g. answers and reasoning the target model produced on your workload) with a running
llama-server's /tokenize endpoint (no special tokens), counts token ids, prints the coverage of the top 8k-64k ids and
writes a json list of [id, count] sorted by count. Standard library only.
"""
import collections
import json
import sys
import urllib.request

url, key_file, out = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3]
key = open(key_file).read().strip() if key_file != "-" else ""


def tokenize(text):
    req = urllib.request.Request(url + "/tokenize", json.dumps({"content": text, "add_special": False}).encode(),
                                 {"Content-Type": "application/json", "Authorization": "Bearer " + key})
    return json.load(urllib.request.urlopen(req, timeout=300))["tokens"]


cnt = collections.Counter()
total = 0
for path in sys.argv[4:]:
    text = open(path, encoding="utf-8", errors="replace").read()
    for i in range(0, len(text), 20000):
        ids = tokenize(text[i:i + 20000])
        cnt.update(ids)
        total += len(ids)
print(f"{total} tokens, {len(cnt)} distinct")
for n in (8192, 16384, 32768, 65536):
    print(f"top {n}: coverage {sum(c for _, c in cnt.most_common(n)) / max(total, 1):.4f}")
json.dump(cnt.most_common(), open(out, "w"))
