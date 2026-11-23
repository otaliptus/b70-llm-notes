"""Replay the Inkling B70 conversations and compaction checks against nextsycl."""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.request
from pathlib import Path


SOURCE_FILES = (
    ("llm/inkling2/src/engine.rs", 100410),
    ("llm/inkling2/src/engine/dspark.rs", 29504),
    ("crates/gguf/src/lib.rs", 21961),
    ("docs/ports/inkling2.md", 26342),
    ("kernels/llm/inkling2/silo/mv.cpp", 45418),
    ("kernels/llm/inkling2/silo/dec.cpp", 21298),
)
EXTRA_FILES = (
    ("glue/serve/src/cache.rs", 16967),
    ("llm/inkling2/src/lib.rs", 7960),
    ("llm/inkling2/src/ffi.rs", 18722),
    ("kernels/llm/inkling2/silo/attn.cpp", 7843),
    ("kernels/llm/inkling2/silo/pattn.cpp", 15516),
)
FIXTURE_HASH = {
    "100k": "eee6ae1863b4c73801cd0d03ca4c89336f2c60aabdba9a0145067150687555ba",
    "126k": "bb77dabc12aa699c37d94b33cfbbfe1040a0f323eb186db10f758dffd3a040f8",
}
INTRO = (
    "I need to hand this Inkling loader to our on-call. Read the source and give me a short operational note: "
    "when the caller leaves the context unset what value is used, in which order are prompt chunk sizes tried "
    "automatically, and how much memory is reserved as a margin? Give the actual numbers and separate context size "
    "from chunk size. The material below contains the loader and its supporting implementation.\n\n"
)


CHAT_TURNS = (
    ("hey, i'm trying to get a little zine swap going at our community room. never hosted "
     "anything before. is this doable with just two of us?"),
    ("We're thinking Saturday 6 June, 2–4 pm at Maple Room. Free entry. Room limit 30, but that "
     "includes me and Jules who are hosting. I'd like a quiet corner. Can you sketch something "
     "simple?"),
    ("Nice. How many visitor places can I actually offer? I always mess up the bit about whether "
     "the hosts count."),
    ("I can spend £75. Printing is £18, tea and biscuits £22, and sign holders £9. What's left? "
     "I'd like to keep a bit for things I forget."),
    ("Let's turn that into a friendly invite. Short enough to send in the group chat, no big "
     "promotional voice, and make clear you can just browse."),
    ("Actually the caretaker just texted: “Maple is fine but maximum 24 people including any "
     "organisers. Door opens to you at 1:30. Keep the side exit clear.” What do I need to change?"),
    ("Can you undo the old number in the invite and send me just the new wording? Don't mention "
     "the old cap."),
    ("Unrelated for a sec, what's a zine in plain English? My neighbour asked and I realised my "
     "explanation made it sound way more complicated than it is."),
    ("Right, back to Saturday. Jules can only stay until 3. Priya will take over at 3, with no "
     "overlap in the room. Give me the simplest host handover, in three bullets."),
    ("We don't need sign holders after all, Jules is borrowing some. Tea and biscuits actually "
     "came to £27. Printing is still £18. What can I spend now without going over budget?"),
    ("I got a question from Lee: “I don't have anything to trade, can I still come? And I find "
     "loud rooms rough.” Could you help me reply without promising total silence?"),
    ("Okay, now a one-line reminder for my phone for setup time. Include the little detail about "
     "the exit because I know I'll forget."),
    ("I'm going to lose this chat. Write a handoff for Priya in under 80 words. Keep the final "
     "date, public hours, room, visitor limit, swap time for the hosts and money left. This is "
     "for her, so drop the backstory."),
    ("Quick check from memory: if all the visitor places are booked and both hosts are inside at "
     "2:15, is there room for two walk-ins? Just tell me what to say at the door."),
)


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_prompt(source, case):
    size = "126k" if case == "126k" else "100k"
    files = SOURCE_FILES + (EXTRA_FILES if size == "126k" else ())
    parts = [INTRO]
    for name, length in files:
        body = (source / name).read_text(encoding="utf-8")
        parts.append((f"\n\nFile {name}\n\n" + body)[:length])
    prompt = "".join(parts)
    if digest(prompt) != FIXTURE_HASH[size]:
        raise ValueError(f"{size} input hash differs. Use the pinned nextsycl checkout with patch 0054 applied.")
    return prompt


def ask(url, key, name, messages, seed, cap):
    body = {"messages": messages, "effort": "low", "stream": True, "temperature": 1.0,
            "top_p": 1.0, "top_k": 0, "min_p": 0.0, "seed": seed, "max_tokens": cap}
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    request = urllib.request.Request(url.rstrip("/") + "/api/chat", json.dumps(body).encode(), headers)
    chunks = []
    reasoning = hashlib.sha256()
    first = visible = end = None
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=900) as response:
        for line in response:
            if not line.strip():
                continue
            item = json.loads(line)
            if "error" in item:
                raise RuntimeError(item["error"])
            if item.get("done"):
                end = item
                break
            thought, content = item.get("thinking", ""), item.get("content", "")
            elapsed = time.perf_counter() - start
            if (thought or content) and first is None:
                first = elapsed
            if content:
                if visible is None:
                    visible = elapsed
                chunks.append(content)
            reasoning.update(thought.encode("utf-8"))
    if end is None or first is None:
        raise RuntimeError(f"incomplete response for {name}")
    content = "".join(chunks)
    row = {"turn": name, "seed": seed, "max_tokens": cap, "finish": end["finish"],
           "prompt_tokens": end["prompt_tokens"], "fresh_tokens": end["prompt_tokens"] - end["reused"],
           "reused_tokens": end["reused"], "generated_tokens": end["generated"],
           "read_seconds": end["read_seconds"], "generate_seconds": end["generate_seconds"],
           "first_token_seconds": first, "first_visible_seconds": visible,
           "visible_sha256": digest(content), "reasoning_sha256": reasoning.hexdigest()}
    print(json.dumps(row), flush=True)
    return {"role": "assistant", "content": content}, row


def user(text):
    return {"role": "user", "content": text}


def compaction_fixture(seed, count=4300):
    rng = random.Random(seed)
    order = f"ZK-{rng.randrange(10000, 100000)}"
    booking = f"PK-{rng.randrange(100000, 1000000)}"
    original = f"B-{rng.randrange(100, 400)}"
    current = f"B-{rng.randrange(500, 800)}"
    pallet = f"PL-{rng.randrange(1000, 10000)}"
    expected = dict(order=order, booking=booking, current_bay=current, pallet=pallet, owner="Neri")

    def closed_checks():
        return [f"Routine check {i}: crew {rng.randrange(20, 95)} returned {rng.randrange(1, 90)} trays "
                f"to bay {rng.randrange(10, 99)}. Closed." for i in range(count)]

    first = closed_checks()
    first.insert(count // 13, f"Open order {order} is for the violet lamp shipment. Its carrier booking is {booking}.")
    first.insert(count * 3 // 4, f"Booking {booking} currently departs from {original}. Its owner is Neri.")
    change = closed_checks()
    change.insert(count // 3, f'Carrier correction for booking {booking}: departure is now {current}. The earlier bay was superseded.')
    change.insert(count * 4 // 5, f"Dispatch assigned pallet {pallet} to booking {booking}.")
    instruction = ("Write a handoff of at most 80 words for the next shift. Preserve all open orders, identifiers, "
                   "references, owners and changes that affect an open order. Omit closed routine checks. Only this handoff "
                   "will be available after the log closes. Return the handoff itself.")
    return "\n".join(first + [instruction]), "\n".join(change + [instruction]), expected


def compaction(request):
    for seed in (177, 178):
        first, second, expected = compaction_fixture(seed)
        one = request(f"{seed}-0", [user(first)], seed, 2048)["content"]
        two = request(f"{seed}-1", [user(f"Previous handoff:\n{one}\nNew shift records:\n{second}")],
                      seed + 1, 2048)["content"]
        question = (f"Available handoff:\n{two}\nFor the violet lamp shipment, return a JSON object "
                    "with keys order, booking, current_bay, pallet and owner. Output only JSON.")
        final = request(f"{seed}-2", [user(question)], seed + 2, 1536)["content"]
        found = re.search(r"\{[^{}]*\}", final)
        try:
            answer = json.loads(found.group()) if found else {}
        except ValueError:
            answer = {}
        fields = {key: answer.get(key) == value for key, value in expected.items()}
        print(json.dumps({"case": "compaction", "seed": seed, "exact_fields": fields,
                          "summary_words": [len(one.split()), len(two.split())]}), flush=True)


def conversation(request):
    history = []
    previous = ""
    for index, text in enumerate(CHAT_TURNS):
        if index == 4 and "£26" in previous:
            text = "Good, I’ll leave that £26 alone for now. " + text
        if index == 10 and ("£30" in previous or "30 pounds" in previous.lower()):
            text = "Great, £30 is enough cushion. " + text
        history.append(user(text))
        reply = request(f"turn-{index + 1:02}", history, 630 + index, 1800)
        history.append(reply)
        previous = reply["content"]
        if index + 1 < len(CHAT_TURNS):
            time.sleep((1.1, 1.8, 1.4, 2.2)[index % 4])


def replay(case, prompt, request):
    if case == "branches":
        root = [user(prompt[:98138] + "\nSummarize the main point in one sentence.")]
        root.append(request("base", root, 805, 768))
        a = root + [user("Read appendix A. " + prompt[11000:22000] + "\nName one Rust function.")]
        b = root + [user("Read appendix B. " + prompt[123000:134000] + "\nName one identifier.")]
        a.append(request("A", a, 806, 768))
        b.append(request("B", b, 807, 768))
        request("revisit-A", a + [user("Give its name in a single sentence.")], 808, 768)
        request("revisit-B", b + [user("Describe the contents of appendix B in ten words.")], 809, 768)
        return
    history = []
    turns = (
        ("cold", prompt, 990),
        ("followup", "One follow-up while you still have the source: which environment variable would I set to "
         "override the automatic chunk size, and what happens if every candidate is too big? Two sentences for the runbook.", 991),
        ("handoff", "Turn that into a concise checklist for a new teammate who wants to measure prompt-read speed "
         "and generated tokens per second separately. Include how to record fresh and reused tokens and how to compare "
         "a cold request to a follow-up fairly.", 992),
    )
    for name, text, seed in turns:
        history.append(user(text))
        history.append(request(name, history, seed, 1152))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="fixture source checkout with only patch 0054 applied")
    parser.add_argument("--case", choices=("100k", "126k", "branches", "compaction", "chat"), default="100k")
    parser.add_argument("--server", default="http://127.0.0.1:18081", help="nextsycl server root URL")
    parser.add_argument("--key-file", type=Path, default=os.getenv("API_KEY_FILE"))
    parser.add_argument("--prepare-only", action="store_true", help="verify the source fixture and exit before contacting the server")
    args = parser.parse_args()
    if args.case == "chat":
        if args.prepare_only:
            print(digest("\n".join(CHAT_TURNS)))
            return
        prompt = ""
    elif args.case == "compaction":
        if args.prepare_only:
            for seed in (177, 178):
                first, second, _ = compaction_fixture(seed)
                print(json.dumps({"seed": seed, "first_shift": digest(first), "second_shift": digest(second)}))
            return
        prompt = ""
    else:
        if not args.source:
            parser.error("--source is required for source-reading and branching cases")
        prompt = source_prompt(args.source, args.case)
        if args.prepare_only:
            print(digest(prompt))
            return
    key = args.key_file.read_text().strip() if args.key_file else ""
    rows = []

    def request(name, history, seed, limit):
        reply, row = ask(args.server, key, name, history, seed, limit)
        rows.append(row)
        return reply

    if args.case == "chat":
        conversation(request)
    elif args.case == "compaction":
        compaction(request)
    else:
        replay(args.case, prompt, request)
    read = sum(r["read_seconds"] for r in rows)
    generation = sum(r["generate_seconds"] for r in rows)
    print(json.dumps({"case": args.case, "requests": len(rows),
                      "fresh_prefill_tps": sum(r["fresh_tokens"] for r in rows) / read if read else None,
                      "decode_tps": sum(r["generated_tokens"] for r in rows) / generation if generation else None}),
          file=sys.stderr)


if __name__ == "__main__":
    main()
