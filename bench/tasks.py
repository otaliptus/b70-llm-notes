"""Small local benchmark: coding (executed against hidden tests in a sandbox), reading comprehension of technical
material, and technical knowledge. Deliberately mid-difficulty so models separate; a few are designed to punish
sloppy reading. Grading: code = tests pass; others = keyword / regex match or a multiple-choice letter."""

CODE = [
    {
        "id": "c01_duration",
        "prompt": "Write a Python function `parse_duration(s: str) -> int` that converts strings like '1h30m15s', '2d', '45m', '90s', '1d2h' into a total number of seconds. Units: d (days), h, m, s; they may appear in any order but at most once each; whitespace is not allowed. Raise ValueError for anything else (empty string, unknown unit, missing number, repeated unit). Return the function in a single ```python code block, no example usage.",
        "tests": """
assert parse_duration('1h30m15s') == 5415
assert parse_duration('2d') == 172800
assert parse_duration('45m') == 2700
assert parse_duration('90s') == 90
assert parse_duration('1d2h') == 93600
assert parse_duration('15s1h') == 3615
for bad in ['', 'h', '1x', '1h1h', '1h 2m', '1.5h', 'abc']:
    try:
        parse_duration(bad); raise SystemExit('no error for %r' % bad)
    except ValueError: pass
""",
    },
    {
        "id": "c02_toposort",
        "prompt": "Write a Python function `topo_sort(deps: dict[str, list[str]]) -> list[str] | None`. `deps` maps a node to the nodes it depends on (which must come before it). Nodes that only appear as dependencies count too. Return a list containing every node exactly once in a valid order, or None if there is a cycle. The order among independent nodes must be deterministic: when several nodes are ready at once, emit them in sorted (alphabetical) order. Return only the function in one ```python block.",
        "tests": """
assert topo_sort({'a': ['b'], 'b': ['c'], 'c': []}) == ['c', 'b', 'a']
assert topo_sort({'x': ['a', 'b'], 'a': [], 'b': []}) == ['a', 'b', 'x']
assert topo_sort({'a': ['b'], 'b': ['a']}) is None
assert topo_sort({'q': ['p']}) == ['p', 'q']
r = topo_sort({'d': ['b', 'c'], 'b': ['a'], 'c': ['a'], 'a': []})
assert r == ['a', 'b', 'c', 'd'], r
assert topo_sort({}) == []
assert topo_sort({'a': ['a']}) is None
""",
    },
    {
        "id": "c03_lru",
        "prompt": "Implement a Python class `LRU` with `__init__(self, capacity: int)`, `get(self, key)` returning the value or -1, and `put(self, key, value)`. Both operations must be O(1) on average; when full, evict the least recently used key (a `get` counts as a use, and so does `put` on an existing key). Do not use functools.lru_cache or OrderedDict; use a dict plus a doubly linked list. Return only the class in one ```python block.",
        "tests": """
c = LRU(2)
c.put(1, 1); c.put(2, 2)
assert c.get(1) == 1
c.put(3, 3)
assert c.get(2) == -1
c.put(4, 4)
assert c.get(1) == -1 and c.get(3) == 3 and c.get(4) == 4
c = LRU(1); c.put('a', 'A'); c.put('a', 'B'); assert c.get('a') == 'B'; c.put('b', 1); assert c.get('a') == -1
import time
c = LRU(1000)
t = time.time()
for i in range(200000):
    c.put(i, i); c.get(i - 500)
assert time.time() - t < 5, 'too slow'
""",
    },
    {
        "id": "c04_expr",
        "prompt": "Write a Python function `evaluate(expr: str) -> float` that evaluates an arithmetic expression containing integers and decimals, + - * /, parentheses, unary minus, and arbitrary spaces, with normal precedence and left-to-right associativity. Do not use eval, exec, ast, or any library; write a recursive-descent parser. Raise ValueError on malformed input (unbalanced parentheses, trailing characters, empty). Division by zero may raise ZeroDivisionError. Return only the function(s) in one ```python block.",
        "tests": """
assert evaluate('1 + 2 * 3') == 7
assert evaluate('(1 + 2) * 3') == 9
assert evaluate('-3 + 5') == 2
assert evaluate('2 * -(3 + 4)') == -14
assert abs(evaluate('10 / 4') - 2.5) < 1e-9
assert evaluate('8 - 3 - 2') == 3
assert evaluate('100 / 10 / 2') == 5
assert abs(evaluate('2.5 * 4') - 10) < 1e-9
assert evaluate('((2))') == 2
assert evaluate('-(-2)') == 2
for bad in ['', '(1 + 2', '1 + 2)', '1 +', '2 3', 'abc']:
    try:
        evaluate(bad); raise SystemExit('no error for %r' % bad)
    except ValueError: pass
""",
    },
    {
        "id": "c05_glob",
        "prompt": "Write a Python function `glob_match(pattern: str, text: str) -> bool` implementing wildcard matching where `*` matches any sequence (including empty) and `?` matches exactly one character; every other character matches itself. It must run in polynomial time (a pattern like '*a*a*a*a*a*a*a*a*b' against 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' must finish in well under a second). Do not use fnmatch, re, or any library. Return only the function in one ```python block.",
        "tests": """
assert glob_match('*.py', 'main.py')
assert not glob_match('*.py', 'main.pyc')
assert glob_match('a?c', 'abc') and not glob_match('a?c', 'ac')
assert glob_match('*', '') and glob_match('', '') and not glob_match('?', '')
assert glob_match('a*b*c', 'axxbyyc') and not glob_match('a*b*c', 'axxcyyb')
assert glob_match('**x', 'x')
import time
t = time.time()
assert not glob_match('*a*a*a*a*a*a*a*a*b', 'a' * 80)
assert time.time() - t < 1.0
""",
    },
    {
        "id": "c06_cidr",
        "prompt": "Write a Python function `in_cidr(ip: str, cidr: str) -> bool` for IPv4, e.g. in_cidr('10.1.2.3', '10.1.0.0/16') -> True. Do not use the ipaddress module. Raise ValueError for an invalid IP (wrong octet count, octet > 255, non-numeric) or an invalid prefix length (not 0..32). A /0 matches everything. Return only the function in one ```python block.",
        "tests": """
assert in_cidr('10.1.2.3', '10.1.0.0/16')
assert not in_cidr('10.2.2.3', '10.1.0.0/16')
assert in_cidr('192.168.1.255', '192.168.1.0/24')
assert not in_cidr('192.168.2.0', '192.168.1.0/24')
assert in_cidr('1.2.3.4', '0.0.0.0/0')
assert in_cidr('172.16.5.5', '172.16.5.5/32') and not in_cidr('172.16.5.6', '172.16.5.5/32')
assert in_cidr('10.0.0.129', '10.0.0.128/25') and not in_cidr('10.0.0.127', '10.0.0.128/25')
for bad in [('1.2.3', '1.2.3.0/24'), ('1.2.3.256', '1.2.3.0/24'), ('a.b.c.d', '1.2.3.0/24'), ('1.2.3.4', '1.2.3.0/33'), ('1.2.3.4', '1.2.3.0/-1')]:
    try:
        in_cidr(*bad); raise SystemExit('no error for %r' % (bad,))
    except ValueError: pass
""",
    },
    {
        "id": "c07_roman",
        "prompt": "Write two Python functions: `to_roman(n: int) -> str` for 1..3999 (standard subtractive notation: 4 = IV, 9 = IX, 40 = XL, 90 = XC, 400 = CD, 900 = CM) raising ValueError outside that range, and `from_roman(s: str) -> int` that accepts only canonical numerals and raises ValueError for anything else (e.g. 'IIII', 'VX', 'IM', lower case, empty). Return only the functions in one ```python block.",
        "tests": """
assert to_roman(1994) == 'MCMXCIV' and to_roman(3999) == 'MMMCMXCIX' and to_roman(4) == 'IV' and to_roman(58) == 'LVIII'
assert from_roman('MCMXCIV') == 1994 and from_roman('XLII') == 42 and from_roman('MMMCMXCIX') == 3999
for n in range(1, 4000):
    assert from_roman(to_roman(n)) == n
for bad in ['IIII', 'VX', 'IM', 'iv', '', 'MMMM', 'XCX', 'IL']:
    try:
        from_roman(bad); raise SystemExit('no error for %r' % bad)
    except ValueError: pass
for n in [0, 4000, -5]:
    try:
        to_roman(n); raise SystemExit('no error for %r' % n)
    except ValueError: pass
""",
    },
    {
        "id": "c08_ini",
        "prompt": "Write a Python function `parse_ini(text: str) -> dict[str, dict[str, str]]`. Rules: lines `[section]` start a section; `key = value` lines belong to the current section (keys before any section go under the section named ''); keys and values are stripped of surrounding whitespace; a value wrapped in double quotes has the quotes removed and may then contain '#' or ';' literally; otherwise everything from the first ' #' or ' ;' (a space followed by # or ;) onward is a comment and is removed; blank lines and lines starting with # or ; are ignored; a later duplicate key overwrites; a line that is neither of these raises ValueError with the line number. Return only the function in one ```python block.",
        "tests": """
cfg = parse_ini('''
# top comment
name = demo
[server]
host = 0.0.0.0   ; the bind address
port=8080
motd = "hello; world # not a comment"
[server]
port = 9090
[empty]
''')
assert cfg[''] == {'name': 'demo'}
assert cfg['server'] == {'host': '0.0.0.0', 'port': '9090', 'motd': 'hello; world # not a comment'}, cfg['server']
assert cfg['empty'] == {}
try:
    parse_ini('[a]\\nthis is not valid\\n'); raise SystemExit('no error')
except ValueError as e:
    assert '2' in str(e), str(e)
""",
    },
    {
        "id": "c09_lis",
        "prompt": "Write a Python function `lis_length(nums: list[int]) -> int` returning the length of the longest strictly increasing subsequence, in O(n log n) time. It will be tested on a list of 200,000 integers and must finish in under 2 seconds. Return only the function in one ```python block.",
        "tests": """
assert lis_length([10, 9, 2, 5, 3, 7, 101, 18]) == 4
assert lis_length([]) == 0 and lis_length([5]) == 1 and lis_length([3, 3, 3]) == 1
assert lis_length([1, 2, 3, 4, 5]) == 5 and lis_length([5, 4, 3, 2, 1]) == 1
import random, time
random.seed(7)
big = [random.randint(0, 10**9) for _ in range(200000)]
t = time.time(); r = lis_length(big); dt = time.time() - t
assert dt < 2.0, 'too slow: %.2fs' % dt
assert 800 < r < 1100, r
""",
    },
    {
        "id": "c10_logparse",
        "prompt": "Write a Python function `slowest_endpoints(log: str, n: int) -> list[tuple[str, float]]`. Each log line looks like `2026-10-13T21:44:35Z GET /api/users/42 200 123ms` (ISO time, method, path, status, duration with 'ms' or 's' suffix). Normalise paths by replacing every path segment that is purely numeric with `{id}`. Return the n endpoints (as 'METHOD /normalised/path') with the highest mean duration in milliseconds, as (endpoint, mean_ms) tuples sorted by mean descending, ties broken alphabetically by endpoint; ignore lines that do not match the format. Return only the function in one ```python block.",
        "tests": """
log = '''2026-10-13T21:44:35Z GET /api/users/42 200 100ms
2026-10-13T21:44:36Z GET /api/users/7 200 300ms
2026-10-13T21:44:37Z POST /api/users 201 1.5s
garbage line here
2026-10-13T21:44:38Z GET /health 200 2ms
2026-10-13T21:44:39Z GET /api/orders/1/items/22 200 50ms
2026-10-13T21:44:40Z GET /api/orders/9/items/3 500 150ms
'''
r = slowest_endpoints(log, 3)
assert r == [('POST /api/users', 1500.0), ('GET /api/users/{id}', 200.0), ('GET /api/orders/{id}/items/{id}', 100.0)], r
assert slowest_endpoints('', 5) == []
assert slowest_endpoints(log, 10)[-1] == ('GET /health', 2.0)
""",
    },
]

# Comprehension: a passage and a question with a definite answer. `accept` = list of regexes; any match = correct.
COMPREHENSION = [
    {
        "id": "r01_journal",
        "prompt": """Below is a systemd unit and a journal excerpt. Answer the question at the end in one short sentence.

[Unit]
Description=Inference server
After=network-online.target
[Service]
ExecStart=/usr/bin/env -i HOME=/srv/inf PATH=/usr/bin:/bin /usr/bin/python3 /srv/inf/server.py
Restart=on-failure
RestartSec=3
StartLimitIntervalSec=120
StartLimitBurst=5
TimeoutStopSec=45
KillSignal=SIGINT

Journal:
22:32:11 systemd[1]: Started inference.service.
22:32:14 python3[4410]: loading model from /srv/inf/models/m.gguf
22:33:55 systemd[1]: Stopping inference.service...
22:33:55 python3[4410]: KeyboardInterrupt
22:34:06 systemd[1]: Stopped inference.service.
22:34:06 systemd[1]: Started inference.service.
22:34:09 python3[4471]: loading model from /srv/inf/models/m.gguf
22:36:01 python3[4471]: listening on 127.0.0.1:18080

Question: What produced the "KeyboardInterrupt" line, and was it a crash?""",
        "require": [r"SIGINT|KillSignal|systemd\W+(sent|sending|stopp)|stop(ping|ped)?\W+(the )?(service|it)", r"not (a |an )?(crash|bug|error)|no(t a)? crash|isn.t a crash|wasn.t a crash|rather than a crash|expected|normal|intentional|deliberate|clean|graceful|orderly"],
    },
    {
        "id": "r02_nft",
        "prompt": """nftables ruleset:

table inet lab {
  chain fwd {
    type filter hook forward priority 0; policy drop;
    iifname "br0" ip daddr 172.16.0.0/24 drop
    iifname "br0" ip daddr 10.0.0.0/8 drop
    iifname "br0" ip daddr 100.64.0.0/10 drop
    iifname "br0" ip daddr 192.168.0.0/16 drop
    iifname "br0" ip daddr != 172.16.0.1 accept
    ct state established,related accept
  }
}

A guest VM on br0 (address 172.16.0.2) sends a new TCP SYN to 100.80.221.20 port 22. Is the packet forwarded? Answer "yes" or "no" and name the rule that decides it.""",
        "accept": [r"^\W*no\b", r"\bno\b.*100\.64\.0\.0/10", r"100\.64\.0\.0/10.*\bdrop"],
        "reject": [r"^\W*yes\b"],
    },
    {
        "id": "r03_trace",
        "prompt": """What does this Python program print? Give only the exact output, one value per line.

def f(xs, acc=[]):
    for x in xs:
        if x % 2:
            acc.append(x * x)
    return acc

a = f([1, 2, 3])
b = f([4, 5])
c = f([], [])
print(len(a), len(b), a is b, c)
d = {k: v for k, v in zip("abc", range(3))}
d.update(b=10)
print(sorted(d.items())[-1])
print(sum(1 for _ in filter(None, [0, "", [], "x", 2, None, 0.0])))""",
        "accept": [r"3 3 True \[\]\s*\n\s*\('c', 2\)\s*\n\s*2"],
    },
    {
        "id": "r04_spec",
        "prompt": """A rate limiter spec says: "Each client has a bucket of capacity C tokens that refills at R tokens per second, continuously. A request costs 1 token and is rejected if the bucket holds less than 1 token. Buckets start full. Refill is computed lazily at the time of each request from the elapsed time since the previous request of that client; the bucket never exceeds C."

C = 5, R = 2. A client sends requests at these times (seconds): 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.5, 1.6, 1.7, 1.8, 1.9. How many requests are rejected? Give the number and list the rejected request times.""",
        "accept": [r"(four|4)\s+(requests?\s+)?(are\s+|were\s+|get\s+|got\s+)?rejected", r"rejected( requests?)?\W+(\w+\W+){0,3}(four|4)\b", r"\b4\b[^.]{0,40}0\.6\W+1\.7\W+1\.8\W+(and\s+)?1\.9"],
        "reject": [r"(three|3)\s+(requests?\s+)?(are\s+|were\s+|get\s+|got\s+)?rejected", r"rejected( requests?)?\W+(\w+\W+){0,3}(three|3)\b"],
    },
    {
        "id": "r05_table",
        "prompt": """Inventory (one device per line: name, zone, table entries, uptime hours):
edge-1, A, 61, 912
edge-2, A, 88, 143
core-1, B, 90, 990
core-2, B, 90, 240
dmz-1, C, 12, 1004
dmz-2, C, 75, 77

Question: Among the devices whose uptime is below 500 hours, which has the most table entries, and what is the total of table entries in its zone?""",
        "accept": [r"core[-\u2010-\u2015]2.*\b180\b"],
    },
    {
        "id": "r06_diff",
        "prompt": """Here is a unified diff:

--- a/retry.py
+++ b/retry.py
@@ -1,9 +1,10 @@
 import time
-def fetch(url, attempts=3, delay=1.0):
+def fetch(url, attempts=3, delay=1.0, backoff=2.0):
     last = None
     for i in range(attempts):
         try:
             return _get(url)
         except TimeoutError as e:
             last = e
-            time.sleep(delay)
+            time.sleep(delay)
+            delay *= backoff
     raise last

With attempts=3, delay=1.0, backoff=2.0 and _get timing out every time: how many seconds does the new fetch() sleep in total before raising, and how many did the old one sleep? Answer as "new: X, old: Y".""",
        "accept": [r"new:?\s*7(\.0)?\b.*old:?\s*3(\.0)?\b"],
    },
]

# Technical knowledge: definite short answers. Multiple choice where phrasing could vary.
KNOWLEDGE = [
    {
        "id": "k01_tcp",
        "prompt": "A host sends a TCP SYN to a port on a Linux machine where nothing is listening and no firewall interferes. What single TCP segment does the Linux kernel send back? Name the flags set.",
        "accept": [r"\bRST\b"],
        "reject": [r"SYN[- /]?ACK"],
    },
    {
        "id": "k02_http",
        "prompt": "An HTTP client sends GET with the header If-None-Match: \"abc\" and the resource's current ETag is \"abc\". Which status code should a compliant server return? Give only the number.",
        "accept": [r"\b304\b"],
    },
    {
        "id": "k03_bash",
        "prompt": "In a bash script, what does `set -o pipefail` change about a pipeline such as `cmd1 | cmd2 | cmd3`? One sentence.",
        "accept": [r"(status|code|value)[^.]*(last|rightmost|right-?most)[^.]*(fail|non-?zero)", r"(last|rightmost|right-?most)[^.]*(command|one|stage)[^.]*(fail|non-?zero)", r"(non-?zero|fail)[^.]*any[^.]*(command|stage|part)", r"any (command|stage|part)[^.]*(fail|non-?zero)"],
    },
    {
        "id": "k04_git",
        "prompt": "Multiple choice. After `git checkout v1.2.3` (a tag), `git pull --ff-only` fails. Why?\nA) Tags cannot be pulled\nB) HEAD is detached, so there is no branch with an upstream to pull into\nC) --ff-only is not valid with tags\nD) The tag is unsigned\nAnswer with the letter only.",
        "accept": [r"^\W*B\b", r"\bB\)"],
        "mc": "B",
    },
    {
        "id": "k05_mem",
        "prompt": "A process maps a 50 GB file with mmap on a machine with 16 GB of RAM and reads it sequentially once. Does the kernel need to swap, does the file need to fit in RAM, and what does the kernel do with the pages that have already been read? Answer in two sentences.",
        "require": [r"(no|does ?n.t|doesn.t|not|never|needn.t|won.t)\W+(\w+\W+){0,6}swap|swap\W+(\w+\W+){0,4}(not|unnecessary|needed|no)", r"page ?cache|evict|reclaim|drop|discard|free"],
        "reject": [r"(?<!not )(?<!n't )(?<!never )(must|needs?|has) to fit"],
    },
    {
        "id": "k06_crypto",
        "prompt": "Multiple choice. Which statement about AES-GCM is correct?\nA) Reusing a nonce with the same key is harmless because GCM is authenticated\nB) Reusing a nonce with the same key breaks confidentiality of the XOR of the plaintexts and lets an attacker forge tags\nC) The authentication tag is computed with HMAC-SHA256\nD) GCM requires the plaintext length to be a multiple of 16 bytes\nAnswer with the letter only.",
        "accept": [r"^\W*B\b", r"\bB\)"],
        "mc": "B",
    },
    {
        "id": "k07_dns",
        "prompt": "A DNS resolver queries for www.example.com and receives a CNAME pointing to cdn.example.net plus an A record for cdn.example.net in the same response. What is the TTL that governs how long the resolver may cache the final IP for www.example.com, and why? Two sentences.",
        "accept": [r"(min(imum)?|lower|smaller|shorter).*(ttl|of the two|both)", r"(cname|a record).*(whichever|lower|smaller|shorter|minimum)"],
    },
    {
        "id": "k08_sql",
        "prompt": "In SQL, `SELECT COUNT(*)` and `SELECT COUNT(col)` on the same table can return different numbers. When, exactly? One sentence.",
        "accept": [r"null"],
    },
]


# ---------------------------------------------------------------------------------------------------------------
# HARD tier: added 2026-10-14. Same categories, more steps, more room to get subtly wrong. Text answers ask for a
# fixed answer line so grading is exact rather than a keyword hunt.

HARD_CODE = [
    {
        "id": "c11_json",
        "prompt": """Write a Python function `parse_json(s: str)` that parses a JSON document and returns Python values: objects -> dict (duplicate keys: last wins), arrays -> list, strings -> str, numbers -> int when the literal has no fraction or exponent else float, true/false/null -> True/False/None. Support the full JSON string escapes (\\" \\\\ \\/ \\b \\f \\n \\r \\t \\uXXXX, including UTF-16 surrogate pairs) and JSON number syntax (optional minus, no leading zeros, optional fraction, optional exponent). Whitespace (space, tab, newline, CR) is allowed between tokens. Do not use the json module, eval, ast or regexes.

On malformed input raise ValueError with a message that begins with `error at N` where N is the 0-based index of the first character the parser cannot accept (for unexpected end of input, N = len(s)). Trailing commas, single-quoted strings, leading zeros, and anything after the top-level value are errors.

Return only the function(s) in one ```python block.""",
        "tests": """
assert parse_json('{"a": [1, 2.5, -3e2, "x\\\\ny", true, null]}') == {"a": [1, 2.5, -300.0, "x\\ny", True, None]}
v = parse_json('10'); assert v == 10 and type(v) is int
v = parse_json('1E2'); assert v == 100.0 and type(v) is float
assert parse_json('"\\\\u00e9\\\\ud83d\\\\ude00"') == "\\u00e9\\U0001F600"
assert parse_json('  {"k":{"k":{"k":[[]]}}}  ') == {"k": {"k": {"k": [[]]}}}
assert parse_json('{"a":1,"a":2}') == {"a": 2}
assert parse_json('[]') == [] and parse_json('{}') == {} and parse_json('-0') == 0
assert parse_json('"a\\\\"b\\\\\\\\c\\\\/d\\\\t"') == 'a"b\\\\c/d\\t'
import re
def off(bad):
    try:
        parse_json(bad)
    except ValueError as e:
        m = re.match(r"error at (\\d+)", str(e))
        assert m, "bad message %r for %r" % (str(e), bad)
        return int(m.group(1))
    raise SystemExit("no error for %r" % bad)
assert off('[1,2,]') == 5, off('[1,2,]')
assert off('{"a":1') == 6, off('{"a":1')
assert off('"abc') == 4, off('"abc')
assert off("{'a':1}") == 1, off("{'a':1}")
assert off('nul') == 3, off('nul')
assert off('[1 2]') == 3, off('[1 2]')
assert off('') == 0
assert off('1 2') == 2, off('1 2')
assert off('[01]') == 2, off('[01]')
assert off('{"a" 1}') == 5, off('{"a" 1}')
""",
    },
    {
        "id": "c12_dijkstra",
        "prompt": """Write a Python function `shortest_path(graph: dict[str, list[tuple[str, int]]], src: str, dst: str) -> tuple[int, list[str]] | None` for a directed graph with non-negative integer edge weights (`graph[u]` lists `(v, w)` edges; nodes may appear only as edge targets). Return `(total_cost, path)` where path is the list of nodes from src to dst inclusive, or None if dst is unreachable. If src == dst return `(0, [src])`.

Tie-break rule: among all minimum-cost paths return the one whose node list is lexicographically smallest when compared as Python lists (so `['A', 'B', 'C', 'D']` beats `['A', 'B', 'D']` because 'C' < 'D' at index 2 — a longer path can win). It must run in O((V + E) log V) time: a chain of 20,000 nodes must be solved well under a second. Return only the function(s) in one ```python block.""",
        "tests": """
g = {'A': [('B', 1), ('C', 1)], 'B': [('D', 1)], 'C': [('D', 1)], 'D': []}
assert shortest_path(g, 'A', 'D') == (2, ['A', 'B', 'D'])
g = {'A': [('B', 1), ('C', 2)], 'B': [('D', 5)], 'C': [('D', 4)]}
assert shortest_path(g, 'A', 'D') == (6, ['A', 'B', 'D'])
g = {'A': [('B', 1)], 'B': [('C', 1), ('D', 2)], 'C': [('D', 1)]}
assert shortest_path(g, 'A', 'D') == (3, ['A', 'B', 'C', 'D']), shortest_path(g, 'A', 'D')
g = {'A': [('B', 1)], 'B': [('A', 1)]}
assert shortest_path(g, 'A', 'C') is None
assert shortest_path(g, 'A', 'A') == (0, ['A'])
g = {'S': [('X', 0), ('Y', 0)], 'X': [('T', 0)], 'Y': [('T', 0)]}
assert shortest_path(g, 'S', 'T') == (0, ['S', 'X', 'T'])
g = {'A': [('Z', 1), ('M', 1)], 'Z': [('E', 1)], 'M': [('E', 1)], 'E': []}
assert shortest_path(g, 'A', 'E') == (2, ['A', 'M', 'E'])
import time
n = 20000
g = {f'n{i:05d}': [(f'n{i+1:05d}', 1)] for i in range(n - 1)}
g[f'n{n-1:05d}'] = []
t = time.time(); r = shortest_path(g, 'n00000', f'n{n-1:05d}')
assert r[0] == n - 1 and len(r[1]) == n and r[1][0] == 'n00000', (r[0], len(r[1]))
assert time.time() - t < 5, 'too slow'
""",
    },
    {
        "id": "c13_regex",
        "prompt": """Write a Python function `re_match(pattern: str, text: str) -> bool` implementing a small regex engine from scratch (no `re` module). Supported syntax: literal characters; `.` (any single character); character classes `[abc]`, ranges `[a-z0-9]`, negated classes `[^...]`; quantifiers `*`, `+`, `?` that apply to the single preceding atom (a literal, `.`, or a class); anchors `^` (only meaningful at the start of the pattern) and `$` (only at the end); and `\\` escaping the next character so it is matched literally (`\\.`, `\\*`, `\\[`, `\\\\`). No groups, no alternation. Semantics are like `re.search`: the pattern may match anywhere in the text unless anchored; return True if any match exists. Matching must not blow up exponentially: `re_match('^a*a*a*a*a*a*a*a*b$', 'a' * 30)` must return False in well under a second (memoize or run states in parallel). Return only the function(s) in one ```python block.""",
        "tests": """
assert re_match('a*b', 'aaab') is True
assert re_match('^a*b$', 'aaab') is True
assert re_match('^a*b$', 'aaaba') is False
assert re_match('x.z', 'axyzb') is True
assert re_match('[0-9]+\\\\.[0-9]+', 'v1.25') is True
assert re_match('[0-9]+\\\\.[0-9]+', 'v125') is False
assert re_match('[^a-c]+d', 'abcd') is False
assert re_match('^[^a-c]+d$', 'xyd') is True
assert re_match('a?b?c?', '') is True
assert re_match('^\\\\.$', '.') is True and re_match('^\\\\.$', 'x') is False
assert re_match('a+', '') is False
assert re_match('^ab*c$', 'ac') is True and re_match('^ab*c$', 'abbbc') is True and re_match('^ab*c$', 'abbbd') is False
assert re_match('colou?r', 'the color red') is True and re_match('colou?r', 'the colour red') is True
assert re_match('^[a-f0-9][a-f0-9]*$', 'deadbeef') is True and re_match('^[a-f0-9][a-f0-9]*$', 'deadbeefg') is False
assert re_match('\\\\*+', 'a**b') is True and re_match('^\\\\[x\\\\]$', '[x]') is True
assert re_match('^.*$', 'anything at all') is True
import time
t = time.time()
assert re_match('^a*a*a*a*a*a*a*a*b$', 'a' * 30) is False
assert re_match('^a*a*a*a*a*a*a*a*a*a*$', 'a' * 40) is True
assert time.time() - t < 2, 'exponential blow-up'
""",
    },
    {
        "id": "c14_bucket",
        "prompt": """Implement a Python class `TokenBucket(capacity: float, rate: float)` with one method `allow(now: float) -> bool`. `now` is a monotonic timestamp in seconds, non-decreasing across calls. Semantics: the bucket starts full; on each call it is first refilled by `rate * (now - time_of_previous_call)` (the first call refills nothing), capped at `capacity`; then the request costs 1 token and is allowed only if the bucket holds at least 1 token (exactly 1 counts as enough); a rejected request consumes nothing but still updates the time of the previous call. Use exact arithmetic (`fractions.Fraction`) internally so that comparisons like "exactly 1 token" are not broken by floating point. Return only the class in one ```python block.""",
        "tests": """
b = TokenBucket(5, 2)
res = [b.allow(t) for t in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.5, 1.6, 1.7, 1.8, 1.9]]
assert res == [True, True, True, True, True, True, False, True, True, False, False, False], res
b = TokenBucket(1, 1)
assert b.allow(0) is True and b.allow(0.5) is False and b.allow(1.0) is True and b.allow(1.999) is False and b.allow(2.0) is True
b = TokenBucket(2, 10)
assert [b.allow(0), b.allow(0), b.allow(0), b.allow(0.1)] == [True, True, False, True]
b = TokenBucket(3, 0.5)
assert all(b.allow(0) for _ in range(3)) and b.allow(100) is True and b.allow(100) is True and b.allow(100) is True and b.allow(100) is False
""",
    },
    {
        "id": "c15_editscript",
        "prompt": """Write a Python function `edit_script(a: list[str], b: list[str]) -> list[tuple[str, str]]` that returns a minimal line-based edit script turning `a` into `b`, as a list of operations `('keep', line)`, `('del', line)`, `('ins', line)` in order. Applying the script (emit kept and inserted lines, skip deleted ones) must reproduce `b` exactly, every line of `a` must appear exactly once as keep or del in order, and the number of del + ins operations must be minimal, i.e. `len(a) + len(b) - 2 * LCS(a, b)`. Use an O(len(a) * len(b)) dynamic program (or better); two lists of 1,500 lines must finish in a few seconds. Return only the function(s) in one ```python block.""",
        "tests": """
def lcs(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]
def check(a, b):
    ops = edit_script(a, b)
    out = [l for op, l in ops if op in ('keep', 'ins')]
    assert out == b, (a, b, ops)
    assert [l for op, l in ops if op in ('keep', 'del')] == a, ops
    cost = sum(1 for op, _ in ops if op != 'keep')
    assert cost == len(a) + len(b) - 2 * lcs(a, b), (cost, ops)
check(['a', 'b', 'c'], ['a', 'c'])
check(['a', 'b', 'c'], ['a', 'x', 'b', 'c', 'y'])
check([], ['q']); check(['q'], []); check([], [])
check(['x', 'y', 'z'], ['z', 'y', 'x'])
check(list('kitten'), list('sitting'))
check(['same'] * 5, ['same'] * 3)
import random, time
random.seed(7)
a = [str(random.randint(0, 50)) for _ in range(1500)]
b = a[:]
for _ in range(300):
    i = random.randrange(len(b)); b[i] = str(random.randint(0, 50))
for _ in range(100):
    b.insert(random.randrange(len(b)), 'new'); del b[random.randrange(len(b))]
t = time.time(); check(a, b); assert time.time() - t < 30, 'too slow'
""",
    },
    {
        "id": "c16_cron",
        "prompt": """Write a Python function `next_run(expr: str, after: datetime) -> datetime` that returns the first time strictly after `after` (seconds and microseconds ignored, so start checking at the next whole minute) matching a standard 5-field cron expression `minute hour day-of-month month day-of-week`. Each field supports `*`, single values, lists (`1,15`), ranges (`1-5`), and steps (`*/15`, `1-10/3`). Day-of-week is 0-6 with 0 = Sunday, and 7 also means Sunday. Standard cron rule: when both day-of-month and day-of-week are restricted (neither is `*`), a day matches if EITHER field matches; otherwise the restricted one must match. Months without the requested day (e.g. Feb 29) are skipped. Naive datetimes, no timezones, no DST. The search may have to go more than a year ahead; make it efficient enough that `('30 2 29 2 *', datetime(2026, 1, 1))` returns quickly (iterate minutes smartly, or skip whole days that cannot match). Return only the function(s) and imports in one ```python block.""",
        "tests": """
from datetime import datetime
assert next_run('*/15 * * * *', datetime(2026, 10, 14, 10, 7, 30)) == datetime(2026, 10, 14, 10, 15)
assert next_run('0 9 * * 1-5', datetime(2026, 10, 16, 9, 0)) == datetime(2026, 10, 19, 9, 0)
assert next_run('30 2 29 2 *', datetime(2026, 1, 1)) == datetime(2028, 2, 29, 2, 30)
assert next_run('0 0 1 * 1', datetime(2026, 10, 14, 12, 0)) == datetime(2026, 10, 19, 0, 0)
assert next_run('10-40/10 * * * *', datetime(2026, 10, 14, 10, 41)) == datetime(2026, 10, 14, 11, 10)
assert next_run('5 4 * * 7', datetime(2026, 10, 14)) == datetime(2026, 10, 18, 4, 5)
assert next_run('0 0 31 * *', datetime(2026, 10, 31, 0, 0)) == datetime(2026, 12, 31, 0, 0)
assert next_run('59 23 * 12 *', datetime(2026, 12, 31, 23, 59)) == datetime(2027, 12, 1, 23, 59)
assert next_run('0 12 15 * 3', datetime(2026, 10, 14, 12, 0)) == datetime(2026, 10, 15, 12, 0)
assert next_run('0 12 15 * 3', datetime(2026, 10, 15, 12, 0)) == datetime(2026, 10, 21, 12, 0)
import time
t = time.time(); next_run('30 2 29 2 *', datetime(2026, 1, 1)); assert time.time() - t < 10, 'too slow'
""",
    },
]

HARD_COMPREHENSION = [
    {
        "id": "r07_nginx",
        "prompt": """nginx server block:

location = /static/app.js { return 200 "exact"; }
location ^~ /static/ { return 200 "prefix-static"; }
location ~* \\.(js|css)$ { return 200 "regex-asset"; }
location /static/js/ { return 200 "prefix-js"; }
location / { return 200 "root"; }
location ~ ^/api/v[0-9]+/ { return 200 "regex-api"; }
location /api/ { return 200 "prefix-api"; }

Using nginx's real location selection rules (exact match first; then the longest matching prefix, which wins outright only if it carries `^~`; otherwise the regex locations are tried in file order and the first match wins; if no regex matches, the longest prefix is used), which string is returned for each request?
1. /static/app.js
2. /static/js/main.js
3. /api/v2/users
4. /static/img/logo.png
5. /about.css

Answer with five lines exactly like `1: name`.""",
        "require": [r"1:\s*\**exact\b", r"2:\s*\**regex-asset", r"3:\s*\**regex-api", r"4:\s*\**prefix-static", r"5:\s*\**regex-asset"],
    },
    {
        "id": "r08_trace2",
        "prompt": """What does this Python 3 program print? Give only the exact output.

def make():
    fs = []
    for i in range(3):
        fs.append(lambda x=i: x * 2)
    gs = [lambda: i for i in range(3)]
    return [f() for f in fs], [g() for g in gs]
print(make())

def gen():
    try:
        yield 1
        yield 2
    finally:
        print("fin")
g = gen()
print(next(g))
g.close()

class A:
    x = 1
    def __init__(self): self.x += 1
a = A(); b = A()
A.x = 10
print(a.x, b.x, A().x)

d = {}
d.setdefault("k", []).append(1)
d.setdefault("k", []).append(2)
print(d)
print(0.1 + 0.2 == 0.3, round(2.5), round(3.5), -7 // 2, -7 % 2)""",
        "accept": [r"\(\[0, 2, 4\], \[2, 2, 2\]\)\s*\n\s*1\s*\n\s*fin\s*\n\s*2 2 11\s*\n\s*\{'k': \[1, 2\]\}\s*\n\s*False 2 4 -4 1"],
    },
    {
        "id": "r09_timeline",
        "prompt": """A backup job logs every step with its own local timestamp. Europe switched from CET (+01:00) to CEST (+02:00) during this job; the uploader logs in UTC.

2026-03-29T01:30:00+01:00  snapshot started
2026-03-29T03:05:00+02:00  snapshot finished
2026-03-29T03:05:00+02:00  upload started
2026-03-29T02:50:00Z       upload finished
2026-03-29T02:50:00Z       verify started
2026-03-29T05:10:00+02:00  verify finished

Convert everything to UTC first. How many minutes did the whole job take from the first to the last line, and which step took the longest (with its duration in minutes)? Answer as `total: N min, slowest: STEP (M min)`.""",
        "require": [r"total:\s*\**160\s*min", r"slowest:\s*\**upload\**\s*\(\s*105\s*min\s*\)"],
    },
    {
        "id": "r10_statemachine",
        "prompt": """An order state machine:

States: NEW, PAID, PACKED, SHIPPED, DELIVERED, REFUNDED, CANCELLED.
Transitions (state --event--> state):
  NEW --pay--> PAID
  NEW --cancel--> CANCELLED
  PAID --pack--> PACKED
  PAID --cancel--> REFUNDED
  PACKED --ship--> SHIPPED
  PACKED --cancel--> REFUNDED
  SHIPPED --deliver--> DELIVERED   (record the delivery day)
  SHIPPED --return--> REFUNDED
  DELIVERED --return--> REFUNDED   only if the return happens no later than 14 days after the delivery day
Any event with no transition from the current state (or failing the 14-day condition) is ignored and the state is unchanged.

Events, as day:event, applied in order:
1:pack  1:pay  2:pack  2:deliver  3:ship  4:pay  6:deliver  19:cancel  21:return

What is the final state, and how many events were ignored? Answer as `final: STATE, ignored: N`.""",
        "require": [r"final:\s*\**delivered\b", r"ignored:\s*\**5\b"],
    },
]

HARD_KNOWLEDGE = [
    {
        "id": "k09_tls13",
        "prompt": "Multiple choice. Which statement about TLS 1.3 is correct?\nA) RSA key transport is still allowed as a fallback key exchange\nB) The server's certificate is sent encrypted, after the key exchange\nC) Session resumption works only with server-side session ID caches\nD) Renegotiation is used to request client authentication after the handshake\nAnswer with the letter only.",
        "accept": [r"^\W*B\b", r"\bB\)"],
        "mc": "B",
    },
    {
        "id": "k10_rename",
        "prompt": "On Linux, process A has /srv/app/config.json open for reading. Process B writes a complete new file config.json.tmp in the same directory and then calls rename(\"config.json.tmp\", \"config.json\"). (1) Can any other process ever observe config.json missing or half-written? (2) If A keeps reading from its already-open file descriptor, does it get the old contents or the new ones? Answer as `1: yes|no - reason. 2: old|new - reason.`",
        "require": [r"1:\s*\**no\b", r"2:\s*\**old\b"],
    },
    {
        "id": "k11_double",
        "prompt": "What is the smallest positive integer that cannot be represented exactly as an IEEE 754 double-precision float? Give the exact decimal number.",
        "accept": [r"9[, ]?007[, ]?199[, ]?254[, ]?740[, ]?993"],
        "reject": [r"9[, ]?007[, ]?199[, ]?254[, ]?740[, ]?992\s*(is|$)"],
    },
    {
        "id": "k12_pid1",
        "prompt": "Multiple choice. A container's entrypoint is a shell script whose last line is `exec myserver`. `docker stop` sends SIGTERM to the container's PID 1, and myserver installs no signal handlers. What happens?\nA) The shell receives SIGTERM, forwards it to myserver, and myserver exits cleanly\nB) myserver is PID 1; SIGTERM is ignored because the kernel does not apply default signal actions to PID 1, so Docker sends SIGKILL after the grace period\nC) PID 1 cannot receive signals from outside its PID namespace at all, so nothing happens until the host reboots\nD) SIGTERM terminates myserver immediately because the default action always applies\nAnswer with the letter only.",
        "accept": [r"^\W*B\b", r"\bB\)"],
        "mc": "B",
    },
    {
        "id": "k13_unicode",
        "prompt": "In Python 3, `s1 = 'e\\u0301'` (e followed by COMBINING ACUTE ACCENT) and `s2 = '\\u00e9'` (precomposed é). What are `len(s1)` and `len(s2)`, and which unicodedata normalization form, applied to both, makes `s1 == s2` true? Answer as `lengths: X and Y; form: NAME`.",
        "require": [r"lengths:\s*\**2\**\s*(and|,|/)\s*\**1\b", r"form:\s*\**NF[CD]K?"],
    },
]

HARD = [dict(t, cat="code", tier="hard") for t in HARD_CODE] + \
       [dict(t, cat="comprehension", tier="hard") for t in HARD_COMPREHENSION] + \
       [dict(t, cat="knowledge", tier="hard") for t in HARD_KNOWLEDGE]
BASE = [dict(t, cat="code", tier="base") for t in CODE] + [dict(t, cat="comprehension", tier="base") for t in COMPREHENSION] + \
       [dict(t, cat="knowledge", tier="base") for t in KNOWLEDGE]
ALL = BASE + HARD
BY_ID = {t["id"]: t for t in ALL}
