"""OpenAI-compatible chat client used by every bench runner. Streams so it can measure time-to-first-token and the
decode rate; falls back to a plain request if the server does not stream usage."""
import json, time, urllib.request, urllib.error

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


class Endpoint:
    def __init__(self, url, key="", model="", timeout=1500):
        self.url, self.key, self.model, self.timeout = url.rstrip("/"), key, model, timeout

    def _req(self, path, body=None, timeout=30, stream=False):
        r = urllib.request.Request(self.url + path, data=None if body is None else json.dumps(body).encode(),
                                   headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
        resp = urllib.request.build_opener(urllib.request.ProxyHandler({})).open(r, timeout=timeout)
        return resp if stream else json.load(resp)

    def wait_ready(self, seconds=600):
        t0 = time.time()
        while time.time() - t0 < seconds:
            try:
                ids = [m["id"] for m in self._req("/models", timeout=5)["data"]]
                print(f"endpoint ready after {time.time() - t0:.0f}s: {ids}", flush=True); return ids
            except Exception:
                time.sleep(5)
        raise SystemExit("endpoint never became ready")

    def chat(self, messages, max_tokens=16000, sampling=None, extra=None):
        """Returns dict(content, think, wall_s, ttft_s, decode_tps, usage, finish, prompt_tokens, completion_tokens)."""
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "stream": True,
                "stream_options": {"include_usage": True}, **(sampling or {}), **(extra or {})}
        t0 = time.time(); first = None; last = None
        content, think, usage, finish = [], [], {}, None
        in_think = False
        resp = self._req("/chat/completions", body, timeout=self.timeout, stream=True)
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"): continue
            data = line[5:].strip()
            if data == "[DONE]": break
            ev = json.loads(data)
            if ev.get("usage"): usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                rc = d.get("reasoning_content") or d.get("reasoning")
                c = d.get("content")
                if rc or c:
                    now = time.time()
                    if first is None: first = now
                    last = now
                if rc: think.append(rc)
                if c:
                    # inline <think> tags (llama.cpp with reasoning off): split them out
                    if THINK_OPEN in c or in_think or THINK_CLOSE in c:
                        buf = c
                        while buf:
                            if in_think:
                                i = buf.find(THINK_CLOSE)
                                if i == -1: think.append(buf); buf = ""
                                else: think.append(buf[:i]); buf = buf[i + len(THINK_CLOSE):]; in_think = False
                            else:
                                i = buf.find(THINK_OPEN); j = buf.find(THINK_CLOSE)
                                if j != -1 and (i == -1 or j < i):
                                    # a closing tag with no opening one: the template pre-filled <think>, so
                                    # everything streamed so far was thinking
                                    think.append("".join(content) + buf[:j]); content.clear(); buf = buf[j + len(THINK_CLOSE):]
                                elif i == -1: content.append(buf); buf = ""
                                else: content.append(buf[:i]); buf = buf[i + len(THINK_OPEN):]; in_think = True
                    else:
                        content.append(c)
                if ch.get("finish_reason"): finish = ch["finish_reason"]
        wall = time.time() - t0
        comp = usage.get("completion_tokens") or 0
        decode_s = (last - first) if (first and last and last > first) else None
        return {"content": "".join(content).strip(), "think": "".join(think), "wall_s": round(wall, 1),
                "ttft_s": round(first - t0, 2) if first else None,
                "decode_tps": round(comp / decode_s, 1) if (decode_s and comp) else None,
                "usage": usage, "finish": finish, "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": comp}


def think_tokens(ans):
    """Thinking tokens: from usage if the server reports them, else pro-rated by characters."""
    u = ans.get("usage") or {}
    det = u.get("completion_tokens_details") or {}
    if det.get("reasoning_tokens") is not None:
        return det["reasoning_tokens"]
    comp = ans.get("completion_tokens") or 0
    tc, cc = len(ans.get("think") or ""), len(ans.get("content") or "")
    return round(comp * tc / (tc + cc)) if comp and (tc + cc) else None
