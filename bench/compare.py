#!/usr/bin/env python3
"""Per-task matrix across result files: python3 compare.py [name ...] (default: every results/*.json)."""
import glob, json, os, sys
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
names = sys.argv[1:] or sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(D, "*.json")))
runs = {n: json.load(open(os.path.join(D, n + ".json"))) for n in names}
tasks = []
for r in runs.values():
    for t in r:
        if t not in tasks: tasks.append(t)
w = max(len(n) for n in names) + 2
print("task".ljust(16) + "".join(n.ljust(w) for n in names))
for t in tasks:
    row = t.ljust(16)
    for n in names:
        x = runs[n].get(t)
        if not x: row += "-".ljust(w); continue
        mark = "PASS" if x["verdict"] else ("THINK" if x["finish"] == "length" or "no answer" in (x["note"] or "") else "FAIL")
        row += f"{mark} {x['wall_s']:.0f}s".ljust(w)
    print(row)
print()
print("totals".ljust(16) + "".join(f"{sum(1 for x in runs[n].values() if x['verdict'])}/{len(runs[n])}".ljust(w) for n in names))
print("sum wall".ljust(16) + "".join(f"{sum(x['wall_s'] or 0 for x in runs[n].values())/60:.0f} min".ljust(w) for n in names))
print("tok total".ljust(16) + "".join(f"{sum(x['completion_tokens'] for x in runs[n].values())//1000}k".ljust(w) for n in names))
print("THINK = thought until the 16k token limit and never answered")
