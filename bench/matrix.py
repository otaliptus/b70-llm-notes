#!/usr/bin/env python3
"""Run a list of scenarios for one model profile, one after the other (each one resumes if interrupted).

  matrix.py --profile NAME --scenarios b2048,nothink,...   (default: the profile's full list)

Profiles know the endpoint, the key and the model id; scenarios set the thinking budget / effort, the sampling and
the suite. Output: results/<profile>-<scenario>.json + .log via run_bench.py / run_convo.py."""
import argparse, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
# Endpoints live in profiles.json next to this file (see profiles.example.json): {"name": {"url", "key" | "key_file", "model", "base_extra"}}
PROFILES = json.load(open(os.path.join(HERE, "profiles.json")))
for _p in PROFILES.values():
    if "key_file" in _p:
        _kf = os.path.expanduser(_p.pop("key_file"))
        _p["key"] = open(_kf).read().strip() if os.path.exists(_kf) else ""   # no file = server without a key
    _p.setdefault("key", ""); _p.setdefault("base_extra", {})
B = lambda n: {"reasoning_budget_tokens": n}
THINK = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}           # the Strata config default
SCENARIOS = {
    # single-turn, all 39 tasks
    "b2048":      dict(kind="single", extra=B(2048), sampling=THINK),
    "nothink":    dict(kind="single", extra={"reasoning_effort": "none"}, sampling={"temperature": 0.7, "top_p": 0.8, "top_k": 20}),
    "b1024":      dict(kind="single", extra=B(1024), sampling=THINK),
    "b4096":      dict(kind="single", extra=B(4096), sampling=THINK),
    "low":        dict(kind="single", extra={"reasoning_effort": "low"}, sampling=THINK),      # soft level, no hard cap
    "b2048-t06":  dict(kind="single", extra=B(2048), sampling={"temperature": 0.6, "top_p": 0.95, "top_k": 20}),
    "unlim":      dict(kind="single", extra={}, sampling=THINK),
    "b2048-rep":  dict(kind="single", extra=B(2048), sampling=THINK),                        # repeat: sampling noise
    "proj-b2048": dict(kind="single", extra=B(2048), sampling=THINK, no_base_extra=True),     # fn only: lab's control vector on
    # conversation suites, all at the 2048 budget
    "convo":      dict(kind="convo", extra=B(2048), sampling=THINK),
    "repair":     dict(kind="repair", extra=B(2048), sampling=THINK),
    "long":       dict(kind="long", extra=B(2048), sampling=THINK),
}
ORDER = {"fn": ["b2048", "nothink", "b1024", "b4096", "convo", "repair", "long", "b2048-t06", "low", "proj-b2048", "unlim", "b2048-rep"],
         "sw": ["b2048", "nothink", "b1024", "b4096", "convo", "repair", "long", "b2048-t06", "low", "unlim", "b2048-rep"]}

ap = argparse.ArgumentParser()
ap.add_argument("--profile", required=True, choices=list(PROFILES))
ap.add_argument("--scenarios", default="")
ap.add_argument("--dry", action="store_true")
a = ap.parse_args()
P = PROFILES[a.profile]
names = a.scenarios.split(",") if a.scenarios else ORDER.get(a.profile, ORDER["sw"])
for n in names:
    S = SCENARIOS[n]
    extra = dict({} if S.get("no_base_extra") else P["base_extra"], **S["extra"])
    name = f"{a.profile}-{n}"
    common = ["--name", name, "--url", P["url"], "--key", P["key"], "--model", P["model"], "--extra", json.dumps(extra),
              "--temperature", str(S["sampling"]["temperature"]), "--top-p", str(S["sampling"]["top_p"]), "--top-k", str(S["sampling"]["top_k"])]
    if S["kind"] == "single":
        cmd = [sys.executable, os.path.join(HERE, "run_bench.py")] + common
    else:
        cmd = [sys.executable, os.path.join(HERE, "run_convo.py"), "--suite", S["kind"]] + common
    log = os.path.join(HERE, "results", name + ".log")
    print(f"=== {time.strftime('%H:%M')} {name}: extra={json.dumps(extra)} sampling={S['sampling']}", flush=True)
    if a.dry: continue
    t0 = time.time()
    with open(log, "a") as lf:
        rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)
    print(f"=== {time.strftime('%H:%M')} {name} done rc={rc} in {(time.time()-t0)/60:.0f} min", flush=True)
