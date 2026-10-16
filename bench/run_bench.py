#!/usr/bin/env python3
"""Single-turn bench: every task in tasks.py against one OpenAI-compatible endpoint, graded.

  run_bench.py --name NAME --url http://127.0.0.1:PORT/v1 --key KEY --model MODEL
               [--extra '{"reasoning_budget_tokens": 2048}'] [--temperature 1.0 --top-p 0.95 --top-k 20]
               [--max-tokens 16000] [--timeout 1500] [--only id,id] [--tier base|hard|all] [--regrade]

Results: results/NAME.json, one record per task (answer, thinking, timing incl. time-to-first-token and decode
tok/s, usage, verdict). Tasks already in the file are skipped (resume). Code answers run in a locked-down container."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tasks as T
from client import Endpoint, think_tokens
from grading import grade

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--url", default=""); ap.add_argument("--key", default="")
ap.add_argument("--model", default=""); ap.add_argument("--extra", default="{}")
ap.add_argument("--temperature", type=float, default=1.0); ap.add_argument("--top-p", type=float, default=0.95)
ap.add_argument("--top-k", type=int, default=20)
ap.add_argument("--max-tokens", type=int, default=16000); ap.add_argument("--timeout", type=int, default=1500)
ap.add_argument("--only", default=""); ap.add_argument("--tier", default="all")
ap.add_argument("--wait", type=int, default=600)
ap.add_argument("--system", default="", help="optional system prompt sent first")
ap.add_argument("--regrade", action="store_true", help="re-score stored non-code answers with the current rules")
a = ap.parse_args()
EXTRA = json.loads(a.extra)
SAMPLING = {"temperature": a.temperature, "top_p": a.top_p, "top_k": a.top_k}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", a.name + ".json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
results = json.load(open(OUT)) if os.path.exists(OUT) else {}


def save():
    json.dump(results, open(OUT, "w"), indent=1)


def regrade():
    changed = 0
    for task in T.ALL:
        r = results.get(task["id"])
        if not r or task["cat"] == "code" or not r.get("content"):
            continue
        ok, note = grade(task, r["content"])
        if ok != r["verdict"]:
            print(f"{task['id']}: {r['verdict']} -> {ok} ({note})"); changed += 1
        r["verdict"], r["note"] = ok, note
    save(); print(f"regraded, {changed} verdict(s) changed")


def main():
    if a.regrade:
        regrade(); summary(); return
    ep = Endpoint(a.url, a.key, a.model, a.timeout); ep.wait_ready(a.wait)
    only = set(a.only.split(",")) if a.only else None
    meta = {"model": a.model, "url": a.url, "sampling": SAMPLING, "extra": EXTRA, "max_tokens": a.max_tokens,
            "started": time.strftime("%Y-%m-%d %H:%M")}
    results.setdefault("_meta", meta)
    for task in T.ALL:
        if only and task["id"] not in only: continue
        if a.tier != "all" and task["tier"] != a.tier: continue
        if task["id"] in results and results[task["id"]].get("verdict") is not None: continue
        print(f"[{a.name}] {task['id']} ...", end=" ", flush=True)
        try:
            msgs = ([{"role": "system", "content": a.system}] if a.system else []) + [{"role": "user", "content": task["prompt"]}]
            ans = ep.chat(msgs, a.max_tokens, SAMPLING, EXTRA)
        except Exception as e:
            ans = {"content": "", "think": "", "wall_s": None, "ttft_s": None, "decode_tps": None, "usage": {},
                   "finish": f"error: {e}", "prompt_tokens": None, "completion_tokens": 0}
        ok, note = grade(task, ans["content"])
        results[task["id"]] = {"cat": task["cat"], "tier": task["tier"], "verdict": ok, "note": note,
                               "wall_s": ans["wall_s"], "ttft_s": ans["ttft_s"], "decode_tps": ans["decode_tps"],
                               "prompt_tokens": ans["prompt_tokens"], "completion_tokens": ans["completion_tokens"],
                               "think_tokens_est": think_tokens(ans), "think_chars": len(ans["think"]),
                               "finish": ans["finish"], "content": ans["content"], "think": ans["think"]}
        save()
        print(f"{'PASS' if ok else 'FAIL'} ({(note or '').splitlines()[-1][:70] if note else ''}) {ans['wall_s']}s, "
              f"{ans['completion_tokens']} tok (think ~{think_tokens(ans)}), {ans['decode_tps']} tok/s", flush=True)
    summary()


def summary():
    rows = {k: v for k, v in results.items() if not k.startswith("_")}
    print(f"\n== {a.name} ==")
    for tier in ("base", "hard"):
        for cat in ("code", "comprehension", "knowledge"):
            rs = [r for r in rows.values() if r["cat"] == cat and r.get("tier", "base") == tier]
            if not rs: continue
            p = sum(1 for r in rs if r["verdict"])
            walls = sorted(r["wall_s"] for r in rs if r["wall_s"]); th = sorted(r["think_tokens_est"] or 0 for r in rs)
            print(f"{tier:5s} {cat:14s} {p}/{len(rs)}  median wall {walls[len(walls)//2] if walls else '-'}s  median think ~{th[len(th)//2]} tok")
    tot = sum(1 for r in rows.values() if r["verdict"])
    print(f"total {tot}/{len(rows)}  wall {sum(r['wall_s'] or 0 for r in rows.values())/60:.0f} min  "
          f"tokens {sum(r['completion_tokens'] or 0 for r in rows.values())//1000}k")


if __name__ == "__main__":
    main()
