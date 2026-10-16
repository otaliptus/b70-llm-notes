"""Reference solutions for the coding tasks: `python3 reference.py` runs each through the same sandbox as the
benchmark, to prove the hidden tests are satisfiable."""
import os, re, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tasks as T

REF = {}

REF["c01_duration"] = r'''
import re
def parse_duration(s: str) -> int:
    if not s or not re.fullmatch(r"(\d+[dhms])+", s):
        raise ValueError(s)
    mult = {"d": 86400, "h": 3600, "m": 60, "s": 1}
    seen, total = set(), 0
    for num, unit in re.findall(r"(\d+)([dhms])", s):
        if unit in seen:
            raise ValueError(s)
        seen.add(unit); total += int(num) * mult[unit]
    return total
'''

REF["c02_toposort"] = r'''
import heapq
def topo_sort(deps):
    nodes = set(deps)
    for v in deps.values(): nodes.update(v)
    indeg = {n: 0 for n in nodes}; out = {n: [] for n in nodes}
    for n, ds in deps.items():
        for d in ds:
            indeg[n] += 1; out[d].append(n)
    heap = sorted(n for n in nodes if indeg[n] == 0); res = []
    while heap:
        n = heapq.heappop(heap); res.append(n)
        for m in out[n]:
            indeg[m] -= 1
            if indeg[m] == 0: heapq.heappush(heap, m)
    return res if len(res) == len(nodes) else None
'''

REF["c03_lru"] = r'''
class LRU:
    class _N:
        __slots__ = ("k", "v", "p", "n")
        def __init__(self, k=None, v=None): self.k, self.v, self.p, self.n = k, v, None, None
    def __init__(self, capacity):
        self.cap = capacity; self.d = {}
        self.h = self._N(); self.t = self._N(); self.h.n = self.t; self.t.p = self.h
    def _unlink(self, x): x.p.n = x.n; x.n.p = x.p
    def _front(self, x): x.n = self.h.n; x.p = self.h; self.h.n.p = x; self.h.n = x
    def get(self, key):
        x = self.d.get(key)
        if x is None: return -1
        self._unlink(x); self._front(x); return x.v
    def put(self, key, value):
        x = self.d.get(key)
        if x is not None:
            x.v = value; self._unlink(x); self._front(x); return
        if len(self.d) >= self.cap:
            last = self.t.p; self._unlink(last); del self.d[last.k]
        x = self._N(key, value); self.d[key] = x; self._front(x)
'''

REF["c04_expr"] = r'''
def evaluate(expr: str) -> float:
    s = expr; pos = 0
    if not s.strip(): raise ValueError("empty")
    def peek():
        nonlocal pos
        while pos < len(s) and s[pos] == " ": pos += 1
        return s[pos] if pos < len(s) else ""
    def number():
        nonlocal pos
        st = pos
        while peek() and (peek().isdigit() or peek() == "."): pos += 1
        if st == pos: raise ValueError("number expected at %d" % pos)
        return float(s[st:pos])
    def factor():
        nonlocal pos
        if peek() == "-": pos += 1; return -factor()
        if peek() == "(":
            pos += 1; v = expr_()
            if peek() != ")": raise ValueError("missing )")
            pos += 1; return v
        return number()
    def term():
        nonlocal pos
        v = factor()
        while peek() in ("*", "/"):
            op = s[pos]; pos += 1; r = factor()
            v = v * r if op == "*" else v / r
        return v
    def expr_():
        nonlocal pos
        v = term()
        while peek() in ("+", "-"):
            op = s[pos]; pos += 1; r = term()
            v = v + r if op == "+" else v - r
        return v
    v = expr_()
    if peek() != "": raise ValueError("trailing input")
    return v
'''

REF["c05_glob"] = r'''
def glob_match(pattern: str, text: str) -> bool:
    p = t = 0; star = -1; mark = 0
    while t < len(text):
        if p < len(pattern) and pattern[p] in ("?", text[t]):
            p += 1; t += 1
        elif p < len(pattern) and pattern[p] == "*":
            star = p; mark = t; p += 1
        elif star != -1:
            p = star + 1; mark += 1; t = mark
        else:
            return False
    while p < len(pattern) and pattern[p] == "*": p += 1
    return p == len(pattern)
'''

REF["c06_cidr"] = r'''
def in_cidr(ip: str, cidr: str) -> bool:
    def to_int(a):
        parts = a.split(".")
        if len(parts) != 4 or not all(p.isdigit() and 0 <= int(p) <= 255 for p in parts): raise ValueError(a)
        n = 0
        for p in parts: n = (n << 8) | int(p)
        return n
    net, _, plen = cidr.partition("/")
    if not plen.isdigit() or not 0 <= int(plen) <= 32: raise ValueError(cidr)
    plen = int(plen); mask = ((1 << 32) - 1) ^ ((1 << (32 - plen)) - 1)
    return to_int(ip) & mask == to_int(net) & mask
'''

REF["c07_roman"] = r'''
_T = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
def to_roman(n: int) -> str:
    if not isinstance(n, int) or not 1 <= n <= 3999: raise ValueError(n)
    out = ""
    for v, r in _T:
        while n >= v: out += r; n -= v
    return out
def from_roman(s: str) -> int:
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    if not s or any(ch not in vals for ch in s): raise ValueError(s)
    total = 0
    for i, ch in enumerate(s):
        v = vals[ch]
        if i + 1 < len(s) and vals[s[i + 1]] > v: total -= v
        else: total += v
    if not 1 <= total <= 3999 or to_roman(total) != s: raise ValueError(s)
    return total
'''

REF["c08_ini"] = r'''
def parse_ini(text: str):
    out = {"": {}}; cur = ""
    for no, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line[0] in "#;": continue
        if line.startswith("[") and line.endswith("]"):
            cur = line[1:-1].strip(); out.setdefault(cur, {}); continue
        if "=" in line:
            k, v = line.split("=", 1); k = k.strip(); v = v.strip()
            if len(v) >= 2 and v[0] == v[-1] == '"':
                v = v[1:-1]
            else:
                for mark in (" #", " ;"):
                    i = v.find(mark)
                    if i != -1: v = v[:i]
                v = v.strip()
            out.setdefault(cur, {})[k] = v; continue
        raise ValueError("line %d: %r" % (no, raw))
    return out
'''

REF["c09_lis"] = r'''
from bisect import bisect_left
def lis_length(nums):
    tails = []
    for x in nums:
        i = bisect_left(tails, x)
        if i == len(tails): tails.append(x)
        else: tails[i] = x
    return len(tails)
'''

REF["c10_logparse"] = r'''
import re
def slowest_endpoints(log: str, n: int):
    pat = re.compile(r"^\S+ (\S+) (\S+) \d{3} (\d+(?:\.\d+)?)(ms|s)$")
    acc = {}
    for line in log.splitlines():
        m = pat.match(line.strip())
        if not m: continue
        method, path, dur, unit = m.groups()
        ms = float(dur) * (1000 if unit == "s" else 1)
        norm = "/".join("{id}" if seg.isdigit() else seg for seg in path.split("/"))
        acc.setdefault(f"{method} {norm}", []).append(ms)
    rows = [(k, sum(v) / len(v)) for k, v in acc.items()]
    rows.sort(key=lambda kv: (-kv[1], kv[0]))
    return rows[:n]
'''

REF["c11_json"] = r"""
def parse_json(s: str):
    n = len(s); i = 0
    def err(pos): raise ValueError("error at %d" % pos)
    def ws():
        nonlocal i
        while i < n and s[i] in " \t\n\r": i += 1
    def value():
        nonlocal i
        ws()
        if i >= n: err(i)
        c = s[i]
        if c == "{": return obj()
        if c == "[": return arr()
        if c == '"': return string()
        if c == "-" or c.isdigit(): return number()
        for lit, v in (("true", True), ("false", False), ("null", None)):
            if s.startswith(lit, i): i += len(lit); return v
            k = 0
            while k < len(lit) and i + k < n and s[i + k] == lit[k]: k += 1
            if k > 0: err(i + k)
        err(i)
    def obj():
        nonlocal i
        i += 1; out = {}; ws()
        if i < n and s[i] == "}": i += 1; return out
        while True:
            ws()
            if i >= n or s[i] != '"': err(i)
            k = string(); ws()
            if i >= n or s[i] != ":": err(i)
            i += 1; out[k] = value(); ws()
            if i >= n: err(i)
            if s[i] == ",": i += 1; continue
            if s[i] == "}": i += 1; return out
            err(i)
    def arr():
        nonlocal i
        i += 1; out = []; ws()
        if i < n and s[i] == "]": i += 1; return out
        while True:
            out.append(value()); ws()
            if i >= n: err(i)
            if s[i] == ",": i += 1; continue
            if s[i] == "]": i += 1; return out
            err(i)
    def string():
        nonlocal i
        i += 1; buf = []
        esc = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}
        while True:
            if i >= n: err(i)
            c = s[i]
            if c == '"': i += 1; return "".join(buf)
            if c == "\\":
                if i + 1 >= n: err(i + 1)
                e = s[i + 1]
                if e in esc: buf.append(esc[e]); i += 2; continue
                if e == "u":
                    def hex4(p):
                        h = s[p:p + 4]
                        if len(h) < 4 or any(ch not in "0123456789abcdefABCDEF" for ch in h):
                            for k, ch in enumerate(h):
                                if ch not in "0123456789abcdefABCDEF": err(p + k)
                            err(p + len(h))
                        return int(h, 16)
                    cp = hex4(i + 2); i += 6
                    if 0xD800 <= cp < 0xDC00 and s.startswith("\\u", i):
                        lo = hex4(i + 2)
                        if 0xDC00 <= lo < 0xE000:
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00); i += 6
                    buf.append(chr(cp)); continue
                err(i + 1)
            if ord(c) < 0x20: err(i)
            buf.append(c); i += 1
    def number():
        nonlocal i
        st = i
        if s[i] == "-": i += 1
        if i >= n: err(i)
        if s[i] == "0": i += 1
        elif s[i].isdigit():
            while i < n and s[i].isdigit(): i += 1
        else: err(i)
        isf = False
        if i < n and s[i] == ".":
            i += 1; isf = True
            if i >= n or not s[i].isdigit(): err(i)
            while i < n and s[i].isdigit(): i += 1
        if i < n and s[i] in "eE":
            i += 1; isf = True
            if i < n and s[i] in "+-": i += 1
            if i >= n or not s[i].isdigit(): err(i)
            while i < n and s[i].isdigit(): i += 1
        return float(s[st:i]) if isf else int(s[st:i])
    v = value(); ws()
    if i != n: err(i)
    return v
"""

REF["c12_dijkstra"] = r"""
import heapq
def shortest_path(graph, src, dst):
    if src == dst: return (0, [src])
    best = {src: (0, [src])}
    heap = [(0, [src])]
    while heap:
        d, path = heapq.heappop(heap)
        u = path[-1]
        if best.get(u, (None,))[0] != d or best[u][1] != path: continue
        if u == dst: return (d, path)
        for v, w in graph.get(u, []):
            nd = d + w; np = path + [v]
            cur = best.get(v)
            if cur is None or nd < cur[0] or (nd == cur[0] and np < cur[1]):
                best[v] = (nd, np); heapq.heappush(heap, (nd, np))
    return None
"""

REF["c13_regex"] = r"""
def re_match(pattern, text):
    atoms = []; i = 0; anch_s = anch_e = False
    if pattern.startswith("^"): anch_s = True; i = 1
    p = pattern
    if p.endswith("$") and not p.endswith("\\$"): anch_e = True; p = p[:-1]
    while i < len(p):
        c = p[i]
        if c == "\\": atom = ("lit", p[i + 1]); i += 2
        elif c == ".": atom = ("any", None); i += 1
        elif c == "[":
            j = i + 1; neg = False
            if j < len(p) and p[j] == "^": neg = True; j += 1
            items = []
            while j < len(p) and p[j] != "]":
                if p[j] == "\\": items.append((p[j + 1], p[j + 1])); j += 2; continue
                if j + 2 < len(p) and p[j + 1] == "-" and p[j + 2] != "]":
                    items.append((p[j], p[j + 2])); j += 3
                else: items.append((p[j], p[j])); j += 1
            atom = ("cls", (neg, items)); i = j + 1
        else: atom = ("lit", c); i += 1
        q = None
        if i < len(p) and p[i] in "*+?": q = p[i]; i += 1
        atoms.append((atom, q))
    def ok(atom, ch):
        k, v = atom
        if k == "any": return True
        if k == "lit": return ch == v
        neg, items = v
        hit = any(a <= ch <= b for a, b in items)
        return hit != neg
    def match_from(start):
        states = {0}
        def close(st):
            out = set(); stack = list(st)
            while stack:
                k = stack.pop()
                if k in out: continue
                out.add(k)
                if k < len(atoms) and atoms[k][1] in ("*", "?"): stack.append(k + 1)
            return out
        states = close(states)
        for pos in range(start, len(text) + 1):
            if len(atoms) in states and (not anch_e or pos == len(text)): return True
            if pos == len(text): break
            ch = text[pos]; nxt = set()
            for k in states:
                if k >= len(atoms): continue
                atom, q = atoms[k]
                if ok(atom, ch):
                    if q in ("*", "+"): nxt.add(k); nxt.add(k + 1)
                    else: nxt.add(k + 1)
                    if q == "+": nxt.add(k + 1)
            states = close(nxt)
            if not states: return False
        return len(atoms) in states
    if anch_s: return match_from(0)
    return any(match_from(st) for st in range(len(text) + 1))
"""

REF["c14_bucket"] = r"""
from fractions import Fraction
class TokenBucket:
    def __init__(self, capacity, rate):
        self.cap = Fraction(str(capacity)); self.rate = Fraction(str(rate)); self.tokens = self.cap; self.last = None
    def allow(self, now):
        now = Fraction(str(now))
        if self.last is not None:
            self.tokens = min(self.cap, self.tokens + self.rate * (now - self.last))
        self.last = now
        if self.tokens >= 1:
            self.tokens -= 1; return True
        return False
"""

REF["c15_editscript"] = r"""
def edit_script(a, b):
    n, m = len(a), len(b)
    L = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        Li, Ln, ai = L[i], L[i + 1], a[i]
        for j in range(m - 1, -1, -1):
            Li[j] = Ln[j + 1] + 1 if ai == b[j] else (Ln[j] if Ln[j] >= Li[j + 1] else Li[j + 1])
    ops = []; i = j = 0
    while i < n or j < m:
        if i < n and j < m and a[i] == b[j]: ops.append(("keep", a[i])); i += 1; j += 1
        elif j < m and (i == n or L[i][j + 1] >= L[i + 1][j]): ops.append(("ins", b[j])); j += 1
        else: ops.append(("del", a[i])); i += 1
    return ops
"""

REF["c16_cron"] = r"""
from datetime import datetime, timedelta
import calendar
def next_run(expr, after):
    f = expr.split()
    def parse(field, lo, hi):
        out = set()
        for part in field.split(","):
            step = 1
            if "/" in part: part, step = part.split("/"); step = int(step)
            if part == "*": a, b = lo, hi
            elif "-" in part: a, b = map(int, part.split("-"))
            else: a = int(part); b = hi if step > 1 else a
            out.update(range(a, b + 1, step))
        return out
    mins, hrs, doms, mons, dows = (parse(f[0], 0, 59), parse(f[1], 0, 23), parse(f[2], 1, 31), parse(f[3], 1, 12), parse(f[4], 0, 7))
    if 7 in dows: dows.add(0)
    dom_any, dow_any = f[2] == "*", f[4] == "*"
    def day_ok(d):
        a = d.day in doms; w = (d.weekday() + 1) % 7 in dows
        if dom_any and dow_any: return True
        if dom_any: return w
        if dow_any: return a
        return a or w
    t = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
    limit = after + timedelta(days=366 * 8)
    while t < limit:
        if t.month not in mons or not day_ok(t):
            t = (t + timedelta(days=1)).replace(hour=0, minute=0); continue
        if t.hour not in hrs:
            t = (t + timedelta(hours=1)).replace(minute=0); continue
        if t.minute not in mins:
            t += timedelta(minutes=1); continue
        return t
    return None
"""

if __name__ == "__main__":
    bad = 0
    for t in T.CODE + T.HARD_CODE:
        prog = REF[t["id"]] + "\n\n" + t["tests"] + "\nprint('ALL TESTS PASSED')\n"
        cmd = ["docker", "run", "--rm", "-i", "--network", "none", "--memory", "2g", "--cpus", "2", "--pids-limit", "256",
               "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--read-only", "--tmpfs", "/tmp",
               "-u", "65534:65534", "-e", "HOME=/tmp", "--entrypoint", "python3", "strata-sycl-dev", "-I", "-"]
        p = subprocess.run(cmd, input=prog.encode(), capture_output=True, timeout=120)
        out = (p.stdout + p.stderr).decode()
        ok = p.returncode == 0 and "ALL TESTS PASSED" in out
        bad += not ok
        print(t["id"], "OK" if ok else "FAIL\n" + out[-800:])
    sys.exit(bad)
