#!/usr/bin/env python3
"""Conversation suites against one endpoint.

  run_convo.py --suite convo|repair|long --name NAME --url URL --key KEY --model MODEL [--extra JSON] [sampling flags]

convo : the multi-turn conversations in convo_tasks.py (each checked turn graded; a conversation passes if all do).
repair: every coding task from tasks.py as an agentic loop: the model answers, the hidden tests run, and on failure
        the test output goes back as the next user turn ("fix it"), up to --repairs extra turns. Records pass@1 and
        pass@N and how many repairs were needed.
long  : tasks asked at the end of a long prior conversation (synthetic doc-summarising turns) of --lengths tokens,
        plus needle questions about facts planted early / in the middle. Records real prompt tokens, time-to-first-
        token, cached tokens, decode rate.
Results: results/NAME.json, resumable per conversation."""
import argparse, json, os, sys, time, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tasks as T, convo_tasks as C
from client import Endpoint, think_tokens
from grading import grade, extract_code, run_sandbox

ap = argparse.ArgumentParser()
ap.add_argument("--suite", required=True, choices=["convo", "repair", "long"])
ap.add_argument("--name", required=True); ap.add_argument("--url", required=True); ap.add_argument("--key", default="")
ap.add_argument("--model", required=True); ap.add_argument("--extra", default="{}")
ap.add_argument("--temperature", type=float, default=1.0); ap.add_argument("--top-p", type=float, default=0.95)
ap.add_argument("--top-k", type=int, default=20)
ap.add_argument("--max-tokens", type=int, default=16000); ap.add_argument("--timeout", type=int, default=1500)
ap.add_argument("--only", default=""); ap.add_argument("--wait", type=int, default=600)
ap.add_argument("--repairs", type=int, default=2, help="repair suite: extra fix-it turns after a failure")
ap.add_argument("--lengths", default="8000,32000,64000", help="long suite: target prior-conversation sizes in tokens")
ap.add_argument("--long-items", default="needle_early,needle_mid,needle_two,c01_duration,r05_table,k04_git,c09_lis")
ap.add_argument("--system", default="", help="optional system prompt sent first")
ap.add_argument("--regrade", action="store_true")
a = ap.parse_args()
EXTRA = json.loads(a.extra)
SAMPLING = {"temperature": a.temperature, "top_p": a.top_p, "top_k": a.top_k}
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", a.name + ".json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
results = json.load(open(OUT)) if os.path.exists(OUT) else {}
only = set(a.only.split(",")) if a.only else None


def save():
    json.dump(results, open(OUT, "w"), indent=1)


def turn_record(ans, ok, note, user):
    return {"user": user if len(user) < 4000 else user[:2000] + "\n...[trimmed]...\n" + user[-1500:],
            "verdict": ok, "note": note, "wall_s": ans["wall_s"], "ttft_s": ans["ttft_s"], "decode_tps": ans["decode_tps"],
            "prompt_tokens": ans["prompt_tokens"], "cached_tokens": ((ans.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
            "completion_tokens": ans["completion_tokens"], "think_tokens_est": think_tokens(ans), "finish": ans["finish"],
            "content": ans["content"], "think": ans["think"]}


def chat(ep, messages):
    try:
        if a.system and not (messages and messages[0]["role"] == "system"):
            messages = [{"role": "system", "content": a.system}] + messages
        return ep.chat(messages, a.max_tokens, SAMPLING, EXTRA)
    except Exception as e:
        return {"content": "", "think": "", "wall_s": None, "ttft_s": None, "decode_tps": None, "usage": {},
                "finish": f"error: {e}", "prompt_tokens": None, "completion_tokens": 0}


# ------------------------------------------------------------------------------------------------- convo
def run_convo(ep):
    for conv in C.CONVOS:
        if only and conv["id"] not in only: continue
        if conv["id"] in results and results[conv["id"]].get("verdict") is not None: continue
        print(f"[{a.name}] {conv['id']} ", end="", flush=True)
        msgs, turns, all_ok = [], [], True
        for turn in conv["turns"]:
            msgs.append({"role": "user", "content": turn["user"]})
            ans = chat(ep, msgs)
            msgs.append({"role": "assistant", "content": ans["content"]})
            if turn.get("check"):
                ok, note = grade(turn["check"], ans["content"])
            else:
                ok, note = None, "unchecked"
            if ok is False: all_ok = False
            turns.append(turn_record(ans, ok, note, turn["user"]))
            print(f"{'P' if ok else ('-' if ok is None else 'F')}", end="", flush=True)
        results[conv["id"]] = {"what": conv["what"], "verdict": all_ok, "turns": turns,
                               "passed_turns": sum(1 for t in turns if t["verdict"]), "checked_turns": sum(1 for t in turns if t["verdict"] is not None),
                               "wall_s": round(sum(t["wall_s"] or 0 for t in turns), 1),
                               "completion_tokens": sum(t["completion_tokens"] or 0 for t in turns)}
        save()
        print(f" -> {'PASS' if all_ok else 'FAIL'} {results[conv['id']]['wall_s']}s", flush=True)


# ------------------------------------------------------------------------------------------------- repair
def run_repair(ep):
    for task in [t for t in T.ALL if t["cat"] == "code"]:
        if only and task["id"] not in only: continue
        if task["id"] in results and results[task["id"]].get("verdict") is not None: continue
        print(f"[{a.name}] {task['id']} ", end="", flush=True)
        msgs, turns = [{"role": "user", "content": task["prompt"]}], []
        ok, repairs_used = False, 0
        for attempt in range(a.repairs + 1):
            ans = chat(ep, msgs)
            msgs.append({"role": "assistant", "content": ans["content"]})
            code = extract_code(ans["content"])
            if not code.strip():
                ok, note = False, "no code block"
            else:
                ok, note = run_sandbox(code, task["tests"])
            turns.append(turn_record(ans, ok, note, msgs[-2]["content"]))
            print("P" if ok else "F", end="", flush=True)
            if ok: break
            if attempt < a.repairs:
                repairs_used += 1
                fb = ("Your code did not pass the tests. This is the output of running it against the tests:\n\n```\n"
                      + (note or "")[-1200:] + "\n```\n\nFix the problem and return the complete corrected code in one ```python block.")
                msgs.append({"role": "user", "content": fb})
        results[task["id"]] = {"tier": task["tier"], "verdict": ok, "pass_at_1": turns[0]["verdict"], "repairs_used": repairs_used,
                               "turns": turns, "wall_s": round(sum(t["wall_s"] or 0 for t in turns), 1),
                               "completion_tokens": sum(t["completion_tokens"] or 0 for t in turns)}
        save()
        print(f" -> {'PASS' if ok else 'FAIL'} after {repairs_used} repair(s), {results[task['id']]['wall_s']}s", flush=True)


# ------------------------------------------------------------------------------------------------- long
NEEDLES = {
    "needle_early": {"facts": [(0.10, "Note for later: the staging database listens on port 5433 and the service account for it is svc-ledger-7.")],
                     "ask": "Earlier in this conversation I told you the staging database's port and its service account name. What were they? Answer as `port: N, account: NAME`.",
                     "check": {"require": [r"port:\s*\**5433\b", r"account:\s*\**`?svc-ledger-7\b"]}},
    "needle_mid": {"facts": [(0.50, "By the way, remember this: the rollback tag we agreed on is rel-2026.10-hotfix3 and the on-call phone is extension 4471.")],
                   "ask": "What rollback tag and on-call extension did I mention earlier in this conversation? Answer as `tag: X, extension: N`.",
                   "check": {"require": [r"tag:\s*\**`?rel-2026\.10-hotfix3\b", r"extension:\s*\**4471\b"]}},
    "needle_two": {"facts": [(0.15, "For reference: the cluster API endpoint is https://api.int.example:8443."),
                             (0.70, "Also: the deploy token expires on 2026-11-03.")],
                   "ask": "Two facts I gave you earlier: the cluster API endpoint, and the deploy-token expiry date. Repeat them as `endpoint: URL, expires: YYYY-MM-DD`.",
                   "check": {"require": [r"endpoint:\s*\**`?https://api\.int\.example:8443", r"expires:\s*\**`?2026-11-03\b"]}},
}


def build_filler(target_tokens, seed="x"):
    """Synthetic prior conversation: the user pastes doc sections and asks for one-line summaries. ~3.3 chars/token."""
    # Any long technical text works as filler; set BENCH_LONG_CORPUS=/path/to/file.md or it uses this repo's docs/.
    import glob
    path = os.environ.get("BENCH_LONG_CORPUS")
    files = [path] if path else sorted(glob.glob(os.path.join(HERE, "..", "docs", "*.md")))
    src = "\n\n".join(open(f, encoding="utf-8", errors="replace").read() for f in files)
    paras = [p.strip() for p in src.split("\n\n") if len(p.strip()) > 200]
    chunks, cur = [], ""
    for p in paras:
        cur += p + "\n\n"
        if len(cur) > 1800: chunks.append(cur); cur = ""
    h = int(hashlib.md5(seed.encode()).hexdigest(), 16) % max(1, len(chunks) - 1)
    chunks = chunks[h:] + chunks[:h]
    msgs, chars, k = [], 0, 0
    while chars < target_tokens * 3.6:
        c = chunks[k % len(chunks)]; k += 1
        u = f"Doc section {k}:\n\n{c}\nSummarise this section in one sentence."
        s = "Summary: " + c.split("\n")[0][:160].rstrip(".") + "."
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": s}]
        chars += len(u) + len(s)
    return msgs


def run_long(ep):
    lengths = [int(x) for x in a.lengths.split(",")]
    items = a.long_items.split(",")
    for L in lengths:
        filler = build_filler(L, seed=str(L))
        for item in items:
            key = f"{item}@{L//1000}k"
            if only and key not in only and item not in only: continue
            if key in results and results[key].get("verdict") is not None: continue
            print(f"[{a.name}] {key} ", end="", flush=True)
            msgs = list(filler)
            if item in NEEDLES:
                spec = NEEDLES[item]
                for frac, fact in sorted(spec["facts"], reverse=True):
                    at = int(len(msgs) * frac) // 2 * 2
                    msgs[at:at] = [{"role": "user", "content": fact}, {"role": "assistant", "content": "Noted, I'll remember that."}]
                ask, check = spec["ask"], spec["check"]
            else:
                t = T.BY_ID[item]; ask, check = t["prompt"], t
            msgs.append({"role": "user", "content": ask})
            ans = chat(ep, msgs)
            ok, note = grade(check, ans["content"])
            rec = turn_record(ans, ok, note, ask)
            rec.update({"item": item, "target_tokens": L})
            results[key] = rec; save()
            print(f"-> {'PASS' if ok else 'FAIL'} prompt {ans['prompt_tokens']} tok (cached {rec['cached_tokens']}), ttft {ans['ttft_s']}s, "
                  f"{ans['decode_tps']} tok/s, wall {ans['wall_s']}s", flush=True)


def regrade():
    changed = 0
    if a.suite == "convo":
        for cid, r in results.items():
            if cid.startswith("_"): continue
            conv = C.BY_ID[cid]; all_ok = True
            for turn, rec in zip(conv["turns"], r["turns"]):
                if not turn.get("check") or turn["check"].get("tests"): continue
                ok, note = grade(turn["check"], rec["content"])
                if ok != rec["verdict"]: changed += 1; print(f"{cid} turn: {rec['verdict']} -> {ok} ({note})")
                rec["verdict"], rec["note"] = ok, note
            all_ok = all(t["verdict"] is not False for t in r["turns"])
            r["verdict"] = all_ok; r["passed_turns"] = sum(1 for t in r["turns"] if t["verdict"])
    elif a.suite == "long":
        for key, r in results.items():
            if key.startswith("_"): continue
            item = r["item"]
            check = NEEDLES[item]["check"] if item in NEEDLES else T.BY_ID[item]
            if check.get("tests"): continue
            ok, note = grade(check, r["content"])
            if ok != r["verdict"]: changed += 1; print(f"{key}: {r['verdict']} -> {ok} ({note})")
            r["verdict"], r["note"] = ok, note
    save(); print(f"regraded, {changed} changed")


def summary():
    rows = {k: v for k, v in results.items() if not k.startswith("_")}
    print(f"\n== {a.name} ({a.suite}) ==")
    if a.suite == "convo":
        print(f"conversations {sum(1 for r in rows.values() if r['verdict'])}/{len(rows)}, turns "
              f"{sum(r['passed_turns'] for r in rows.values())}/{sum(r['checked_turns'] for r in rows.values())}, "
              f"wall {sum(r['wall_s'] for r in rows.values())/60:.0f} min")
    elif a.suite == "repair":
        print(f"pass@1 {sum(1 for r in rows.values() if r['pass_at_1'])}/{len(rows)}, "
              f"pass@{a.repairs+1} {sum(1 for r in rows.values() if r['verdict'])}/{len(rows)}, "
              f"repairs used {sum(r['repairs_used'] for r in rows.values())}, wall {sum(r['wall_s'] for r in rows.values())/60:.0f} min")
    else:
        for L in sorted({r["target_tokens"] for r in rows.values()}):
            rs = [r for r in rows.values() if r["target_tokens"] == L]
            pt = [r["prompt_tokens"] for r in rs if r["prompt_tokens"]]
            print(f"{L//1000:3d}k: {sum(1 for r in rs if r['verdict'])}/{len(rs)}  prompt ~{max(pt) if pt else '-'} tok  "
                  f"ttft {[r['ttft_s'] for r in rs]}  decode {[r['decode_tps'] for r in rs]}")


if __name__ == "__main__":
    if a.regrade:
        regrade(); summary(); sys.exit()
    ep = Endpoint(a.url, a.key, a.model, a.timeout); ep.wait_ready(a.wait)
    results.setdefault("_meta", {"suite": a.suite, "model": a.model, "sampling": SAMPLING, "extra": EXTRA,
                                 "started": time.strftime("%Y-%m-%d %H:%M")})
    {"convo": run_convo, "repair": run_repair, "long": run_long}[a.suite](ep)
    summary()
