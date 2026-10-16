"""Graders shared by the runners: code runs against hidden tests in a locked-down container; text answers are matched
with regexes or a multiple-choice letter."""
import re, subprocess

import os
SANDBOX_IMAGE = os.environ.get("BENCH_SANDBOX_IMAGE", "python:3.12-slim")   # any image with python3; no network, read-only
CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(content):
    blocks = CODE_BLOCK.findall(content)
    if not blocks:
        return content if "def " in content or "class " in content else ""
    defs = [b for b in blocks if re.search(r"^\s*(def|class|import|from) ", b, re.M)]
    return "\n\n".join(defs) if defs else blocks[-1]


def run_sandbox(code, tests, timeout=90, image=None):
    prog = code + "\n\n# ---- tests ----\n" + tests + "\nprint('ALL TESTS PASSED')\n"
    cmd = ["docker", "run", "--rm", "-i", "--network", "none", "--memory", "2g", "--cpus", "2", "--pids-limit", "256",
           "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--read-only", "--tmpfs", "/tmp",
           "-u", "65534:65534", "-e", "HOME=/tmp", "--entrypoint", "python3", image or SANDBOX_IMAGE, "-I", "-"]
    try:
        p = subprocess.run(cmd, input=prog.encode(), capture_output=True, timeout=timeout)
        out = (p.stdout + p.stderr).decode("utf-8", "replace")
        return p.returncode == 0 and "ALL TESTS PASSED" in out, out[-1500:]
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT ({timeout}s)"


def mc_answer(content):
    cands = []
    for line in content.splitlines():
        m = re.match(r"^\W*([A-D])\W*$", line.strip())
        if m: cands.append(m.group(1))
    cands += re.findall(r"answer(?: is|:)?\W*\**([A-D])\b", content, re.I)
    cands += re.findall(r"\b([A-D])\)", content)
    return cands[-1] if cands else None


def grade_text(check, content):
    """check: dict with optional mc / reject / require / accept. Returns (ok, note)."""
    c = content
    if check.get("mc"):
        got = mc_answer(c)
        return got == check["mc"], f"picked {got}"
    flat = " ".join(c.split())
    for p in check.get("reject", []):
        if re.search(p, flat, re.I):
            return False, f"reject pattern matched: {p[:40]}"
    for p in check.get("require", []):
        if not re.search(p, flat, re.I | re.S):
            return False, f"required pattern missing: {p[:40]}"
    acc = check.get("accept", [])
    if acc and not any(re.search(p, flat, re.I | re.S) or re.search(p, c.strip(), re.I | re.S) for p in acc):
        return False, "no accept pattern matched"
    return True, "ok"


def grade(check, content, image=None):
    """check: a task or a per-turn check: {'tests': ...} runs code; else text rules. Empty content = no answer."""
    if not content.strip():
        return False, "no answer"
    if any(k in check for k in ("mc", "accept", "require", "reject")):
        ok, note = grade_text(check, content)
        if not ok:
            return ok, note
    if check.get("tests"):
        code = extract_code(content)
        if not code.strip():
            return False, "no code block"
        return run_sandbox(code, check["tests"], image=image)
    return True, "ok"
