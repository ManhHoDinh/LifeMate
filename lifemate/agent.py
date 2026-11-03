"""Agent loop: one LLM turn = one tool call, executed on the sample's ToolEnv, result fed back.

Agents are step-driven state machines so a backend can batch one generation step of many samples at once.
Guards that protect the score (every env.call is part of the submission):
- env.validate() before env.call(): schema errors go back to the model as feedback and are never executed;
- a repeated identical read call is answered from cache instead of calling the env again;
- sys.ask_missing_info ends the turn; hard caps on executed calls and on total generations.
"""
from __future__ import annotations

import json
import re

from .prompt import build_messages, tool_result_message

DONE_NAMES = {"done", "", None, "none", "stop", "finish"}
_THINK = re.compile(r"<think>.*?</think>", re.S)


def extract_json(text: str):
    """First balanced {...} object in the text (thinking blocks and code fences ignored); None if there is none."""
    text = _THINK.sub("", text or "")
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


class Agent:
    def __init__(self, sample: dict, env, tools: dict, apps: dict, request_text: str, history_texts: list[str],
                 max_calls: int = 8, max_generations: int = 12):
        self.sample, self.env, self.tools = sample, env, tools
        self.messages = build_messages(sample, tools, apps, request_text, history_texts)
        self.max_calls, self.max_generations = max_calls, max_generations
        self.n_generations = 0
        self.n_executed = 0
        self.cache: dict[str, dict] = {}
        # every tool result seen so far (history + this turn): x-ref IDs must come from here
        self.observed = json.dumps([h.get("tool_results") for h in sample["history"]], ensure_ascii=False)
        self.done = False
        self.trace: list[dict] = []  # raw model outputs and what the agent did with them

    def _finish(self, why: str):
        self.done = True
        self.trace.append({"event": "finish", "why": why})

    def _feedback(self, raw: str, msg: str):
        self.messages.append({"role": "assistant", "content": raw})
        self.messages.append({"role": "user", "content": msg})

    def feed(self, raw: str):
        """Consume one model output."""
        self.n_generations += 1
        obj = extract_json(raw)
        self.trace.append({"event": "output", "raw": raw})
        if not isinstance(obj, dict):
            self._feedback(raw, 'Lỗi: hãy trả về đúng một đối tượng JSON {"name": ..., "arguments": {...}} hoặc {"name": "done"}.')
            return self._check_budget()
        name = obj.get("name")
        args = obj.get("arguments") or {}
        if isinstance(name, str):
            name = name.strip()
        if name in DONE_NAMES or (isinstance(name, str) and name.lower() in DONE_NAMES):
            return self._finish("done")
        if not isinstance(args, dict):
            args = {}
        err = self.env.validate(name, args)
        if not err["ok"]:
            self.trace.append({"event": "invalid", "name": name, "arguments": args, "error": err["error"]})
            self._feedback(raw, f"Lỗi schema (chưa thực hiện): {err['error']}. Sửa lại lời gọi, hoặc {{\"name\": \"done\"}}.")
            return self._check_budget()
        unseen = self._unseen_refs(name, args)
        if unseen:
            self.trace.append({"event": "unseen_ref", "name": name, "arguments": args, "params": unseen})
            self._feedback(raw, f"Lỗi (chưa thực hiện): {', '.join(unseen)} không có trong kết quả tool nào. "
                                "Gọi tool đọc để lấy ID đúng, hoặc sys.ask_missing_info nếu không xác định được.")
            return self._check_budget()
        call_json = json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
        key = json.dumps({"name": name, "arguments": args}, ensure_ascii=False, sort_keys=True)
        # only reads are deduplicated: gold repeats identical writes on purpose (e.g. redeem the same reward twice)
        if key in self.cache and self.tools.get(name, {}).get("kind") == "read":
            self.trace.append({"event": "duplicate", "name": name})
            result = {**self.cache[key], "note": "đã gọi trước đó, không thực hiện lại"}
        else:
            result = self.env.call(name, args)
            self.observed += json.dumps(result, ensure_ascii=False)
            self.cache[key] = result
            self.n_executed += 1
            self.trace.append({"event": "call", "name": name, "arguments": args, "result": result})
        if name == "sys.ask_missing_info":
            return self._finish("ask_missing_info")
        if self.n_executed >= self.max_calls:
            return self._finish("max_calls")
        self.messages.append({"role": "assistant", "content": call_json})
        self.messages.append(tool_result_message(result))
        return self._check_budget()

    def _unseen_refs(self, name: str, args: dict) -> list[str]:
        """x-ref arguments whose value never appeared in a tool result: invented IDs (on gold calls this rule
        flags 1 of 1278 x-ref values, on zero-shot Qwen3-4B it caught IDs behind 52 of 516 val samples)."""
        props = (self.tools.get(name, {}).get("parameters") or {}).get("properties", {})
        out = []
        for k, v in args.items():
            if not props.get(k, {}).get("x-ref"):
                continue
            for x in v if isinstance(v, list) else [v]:
                if isinstance(x, str) and json.dumps(x, ensure_ascii=False) not in self.observed:
                    out.append(f"{k}={x!r}")
        return out

    def _check_budget(self):
        if not self.done and self.n_generations >= self.max_generations:
            self._finish("max_generations")


def run_agents(agents: list[Agent], backend, log_every: int = 0, wave: int = 0) -> None:
    """Step all agents until done, batching one generation step of every active agent per backend call.

    wave > 0 runs the agents in groups of `wave`, each group to completion: a small KV cache then keeps every agent's
    prompt prefix cached between its steps (on a T4, 200 agents at once evicted it and re-prefilled ~1M tokens a step).
    """
    if wave:
        import time

        t0 = time.time()
        every = max(1, len(agents) // (wave * 20)) * wave  # ~20 progress lines per run
        for i in range(0, len(agents), wave):
            run_agents(agents[i:i + wave], backend)
            done = min(i + wave, len(agents))
            if done % every == 0 or done == len(agents):
                el = time.time() - t0
                print(f"  {done}/{len(agents)} samples, {el:.0f}s ({el / done:.2f}s/sample)", flush=True)
        return
    step = 0
    while True:
        active = [a for a in agents if not a.done]
        if not active:
            return
        outs = backend.generate(active)
        for a, out in zip(active, outs):
            if out is None:  # backend could not serve this request (e.g. prompt too long)
                a._finish("backend_error")
            else:
                a.feed(out)
        step += 1
        if log_every and step % log_every == 0:
            print(f"  step {step}: {len(active)} active", flush=True)
