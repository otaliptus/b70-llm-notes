"""Multi-turn conversations. Each turn is a user message plus an optional `check` (same rules as tasks.py: `tests`
runs the code block, `accept`/`require`/`reject` match the text, `mc` a letter). A conversation passes when every
checked turn passes; per-turn verdicts are recorded too. Assistant replies are carried forward as content only
(thinking stripped), the way a normal chat client does."""

CONVOS = [
    {
        "id": "m01_evolve",
        "what": "code that grows over three turns; later turns must keep earlier behaviour",
        "turns": [
            {"user": "Write a Python class `Inventory` with `add(item: str, qty: int)`, `remove(item: str, qty: int)` (raise ValueError if the item is unknown or has fewer than qty), and `count(item: str) -> int` (0 for unknown items). Return only the class in one ```python block.",
             "check": {"tests": """
inv = Inventory(); inv.add('bolt', 10); inv.remove('bolt', 3)
assert inv.count('bolt') == 7 and inv.count('nut') == 0
for bad in [('bolt', 8), ('nut', 1)]:
    try: inv.remove(*bad); raise SystemExit('no error')
    except ValueError: pass
"""}},
            {"user": "Add a method `transfer(self, other: 'Inventory', item: str, qty: int)` that moves qty of item from this inventory to `other`. It must be atomic: if this inventory lacks enough, raise ValueError and leave BOTH inventories unchanged. Return the full updated class.",
             "check": {"tests": """
a, b = Inventory(), Inventory(); a.add('bolt', 5); a.transfer(b, 'bolt', 2)
assert a.count('bolt') == 3 and b.count('bolt') == 2
try: a.transfer(b, 'bolt', 9); raise SystemExit('no error')
except ValueError: pass
assert a.count('bolt') == 3 and b.count('bolt') == 2, 'not atomic'
a.remove('bolt', 3); assert a.count('bolt') == 0
"""}},
            {"user": "Now add `undo(self)`: it reverts the most recent *successful* add, remove or transfer performed by this inventory (a transfer is recorded on the source inventory and undoing it reverts both inventories). Repeated undo keeps stepping back; undo with nothing left to undo raises ValueError. Failed operations are not recorded. Return the full class again.",
             "check": {"tests": """
a, b = Inventory(), Inventory()
a.add('bolt', 5); a.add('nut', 2); a.remove('nut', 1); a.transfer(b, 'bolt', 4)
assert a.count('bolt') == 1 and b.count('bolt') == 4
try: a.transfer(b, 'bolt', 50)
except ValueError: pass
a.undo(); assert a.count('bolt') == 5 and b.count('bolt') == 0, 'undo transfer'
a.undo(); assert a.count('nut') == 2, 'undo remove'
a.undo(); assert a.count('nut') == 0, 'undo add'
a.undo(); assert a.count('bolt') == 0
try: a.undo(); raise SystemExit('no error')
except ValueError: pass
"""}},
        ],
    },
    {
        "id": "m02_rules",
        "what": "a formatting rule set in turn 1 must still hold in turn 4",
        "turns": [
            {"user": "For the rest of this conversation follow two rules: (1) every reply must end with the exact line `-- END --` as its last line; (2) never use the words 'simply' or 'simple'. Acknowledge in one short line.",
             "check": {"require": [r"-- END --\s*$"], "reject": [r"\bsimpl[ey]\b"]}},
            {"user": "What does `chmod 640 file` grant to the owner, the group, and others? One line each.",
             "check": {"require": [r"-- END --\s*$", r"other[s]?\W+(\w+\W+){0,6}(no|none|nothing|zero|cannot|can.t|---|no access|no permission)"], "reject": [r"\bsimpl[ey]\b"]}},
            {"user": "Write a Python function `is_palindrome(s: str) -> bool` that ignores case and every non-alphanumeric character. One ```python block.",
             "check": {"tests": """
assert is_palindrome('A man, a plan, a canal: Panama') and is_palindrome('') and not is_palindrome('ab')
assert is_palindrome('No lemon, no melon!') and not is_palindrome('abc1')
""", "require": [r"-- END --\s*$"], "reject": [r"\bsimpl[ey]\b"]}},
            {"user": "What is 17 * 23 + 4?",
             "check": {"accept": [r"\b395\b"], "require": [r"-- END --\s*$"], "reject": [r"\bsimpl[ey]\b"]}},
        ],
    },
    {
        "id": "m03_cart",
        "what": "running state across turns with removals and a partial discount",
        "turns": [
            {"user": "Track a shopping cart for me. Add 3 apples at 0.50 each and 2 breads at 2.25 each. What is the running total? Answer as `total: X`.",
             "check": {"accept": [r"total:\s*\**\$?6(\.00?)?\b"]}},
            {"user": "Remove one bread, then apply a 10% discount to the apples only. New total? Same format.",
             "check": {"accept": [r"total:\s*\**\$?3\.60?\b"]}},
            {"user": "Add one milk at 1.80 and two eggs at 0.35 each, then remove one apple (the discount still applies to the remaining apples). Final total? Same format.",
             "check": {"accept": [r"total:\s*\**\$?5\.65\b"]}},
        ],
    },
    {
        "id": "m04_compose",
        "what": "questions about a file pasted in turn 1, then an edit of it in turn 3",
        "turns": [
            {"user": """Here is a docker-compose.yml:

services:
  db:
    image: postgres:16
    ports: ["127.0.0.1:5432:5432"]
    healthcheck: {test: ["CMD", "pg_isready"], interval: 5s}
  cache:
    image: redis:7
    expose: ["6379"]
  api:
    build: .
    ports: ["8080:8000"]
    depends_on:
      db: {condition: service_healthy}
      cache: {condition: service_started}
    environment: {DATABASE_URL: "postgres://db:5432/app"}

Which services can be reached from outside the host, which only from the host itself, and which not at all? Name the host ports.""",
             "check": {"require": [r"\b8080\b", r"\b5432\b", r"cache\W+(\w+\W+){0,8}(not|no|isn.t|unreachable|internal|only)"]}},
            {"user": "Without me pasting the file again: which services have no healthcheck, and what exactly does api's depends_on wait for on each dependency?",
             "check": {"require": [r"\bcache\b", r"\bapi\b", r"healthy", r"start(ed)?\b"]}},
            {"user": "Give me the full modified file: the db port must no longer be published to the host at all, and api must run with `deploy: {replicas: 2}`. Keep everything else the same. Return it in one ```yaml block.",
             "check": {"require": [r"replicas:\s*2\b", r"6379", r"service_healthy", r"8080:8000"], "reject": [r"5432:5432"]}},
        ],
    },
    {
        "id": "m05_review",
        "what": "find two bugs, fix them, then refactor without regressing",
        "turns": [
            {"user": """Review this code. There are exactly two bugs; name them in two bullet points, no fixes yet.

def chunked(xs, n):
    out = []
    for i in range(0, len(xs), n):
        out.append(xs[i:i+n-1])
    return out

def dedupe(xs, seen=set()):
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x); out.append(x)
    return out""",
             "check": {"require": [r"n\s*-\s*1|i\s*\+\s*n\s*-\s*1|last (element|item)|drops|loses|one (element|item) short|off.by.one", r"mutable default|shared|default (argument|parameter|value|set)|seen\s*=\s*set\(\)|persist|between calls|across calls"]}},
            {"user": "Fix both bugs and give the full corrected code in one ```python block.",
             "check": {"tests": """
assert chunked([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
assert chunked([], 3) == []
assert dedupe([1, 1, 2, 3, 2]) == [1, 2, 3]
assert dedupe([1, 2]) == [1, 2], 'state leaked between calls'
assert dedupe(['b', 'a', 'b']) == ['b', 'a']
"""}},
            {"user": "Now make `chunked` lazy (a generator) and make it raise ValueError unless n is a positive int (bool does not count). Keep dedupe as it is. Full code again in one ```python block.",
             "check": {"tests": """
import types
g = chunked([1, 2, 3, 4, 5], 2); assert isinstance(g, types.GeneratorType), 'not a generator'
assert list(g) == [[1, 2], [3, 4], [5]]
assert list(chunked([], 3)) == []
for bad in [0, -1, '2', 2.0, True]:
    try: list(chunked([1, 2], bad)); raise SystemExit('no error for %r' % (bad,))
    except ValueError: pass
assert dedupe([1, 1, 2]) == [1, 2] and dedupe([1, 2]) == [1, 2]
"""}},
        ],
    },
    {
        "id": "m06_specchange",
        "what": "implement a spec, then absorb a requirements change",
        "turns": [
            {"user": "Write a Python function `slugify(s: str) -> str`: lowercase the text, replace every run of characters that are not ASCII letters or digits with a single '-', and strip leading and trailing '-'. One ```python block.",
             "check": {"tests": """
assert slugify('Hello, World!') == 'hello-world'
assert slugify('  --x--  ') == 'x' and slugify('A  B') == 'a-b' and slugify('') == ''
assert slugify('Ünïcode café') == 'n-code-caf'
"""}},
            {"user": "Requirements changed. (a) Keep Unicode letters and digits (anything for which `str.isalnum()` is true), not just ASCII. (b) Add a `max_len: int = 20` parameter: if the result is longer than max_len, cut it at the last '-' that is at or before position max_len when there is one, otherwise hard-cut at max_len; the result must never end with '-'. Give the full function again in one ```python block.",
             "check": {"tests": """
assert slugify('Hello, World!') == 'hello-world'
assert slugify('Ünïcode café') == 'ünïcode-café'
assert slugify('a b c d e f g h i j k l m n o p', max_len=10) == 'a-b-c-d-e', slugify('a b c d e f g h i j k l m n o p', max_len=10)
assert slugify('abcdefghijklmnopqrstuvwxyz') == 'abcdefghijklmnopqrst'
assert slugify('abcdefghij-klmnopqrstuv', max_len=12) == 'abcdefghij'
assert slugify('--x--') == 'x' and slugify('') == ''
assert slugify('short one', max_len=20) == 'short-one'
"""}},
        ],
    },
    {
        "id": "m07_decision",
        "what": "a recommendation that must flip when a constraint changes",
        "turns": [
            {"user": "I'm building a small desktop tool: a single process on one laptop, one user, a few thousand rows, no network, must work offline, zero ops. SQLite or PostgreSQL? One word, then one sentence why.",
             "check": {"accept": [r"sqlite"]}},
            {"user": "Update: it is now a web service with 40 worker processes across 3 machines writing concurrently, needs row-level locking, roles and online backups. Does your recommendation change? One word, then one sentence why.",
             "check": {"accept": [r"postgres"]}},
            {"user": "Final answer for the current requirements, one word only: which database?",
             "check": {"accept": [r"postgres"], "reject": [r"sqlite"]}},
        ],
    },
    {
        "id": "m08_capacity",
        "what": "arithmetic that must be redone after a correction two turns later",
        "turns": [
            {"user": "I have 3 servers with 64 GB RAM each. Each runs 5 VMs of 8 GB plus a 4 GB hypervisor reserve. How much RAM is free per server? Answer as `free: N GB`.",
             "check": {"accept": [r"free:\s*\**20\s*GB"]}},
            {"user": "Correction: one of the three servers actually has 128 GB, not 64. What is the total free RAM across all three now? Answer as `total free: N GB`.",
             "check": {"accept": [r"total free:\s*\**124\s*GB"]}},
            {"user": "If I keep adding 8 GB VMs, one at a time, to whichever server still has room, until no server can fit another, how many VMs do I add in total? Answer as `added: N`.",
             "check": {"accept": [r"added:\s*\**14\b"]}},
        ],
    },
]
BY_ID = {c["id"]: c for c in CONVOS}
