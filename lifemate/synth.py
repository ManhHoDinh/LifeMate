"""Synthetic SFT data for apps the training set does not cover, verified by execution.

    python -m lifemate.synth --split train --targets held_out --out synth/held_out      # experiment: held-out train apps
    python -m lifemate.synth --split public_test --targets new --out synth/new_apps     # final: the apps new in public_test

Only the split's tool catalog and its zaic_tools code are used for the target apps; their app data (state) is generated,
never read from <split>_environment. Context and companion-app states come from train samples (train_environment).
Three LLM stages, run in rounds of --chunk states (each round batched; its accepted samples are appended to
<out>/kept.jsonl and the outputs are rebuilt, so a killed run keeps its data and resumes at the next round):
1. state: for each (target app, base context) the model writes the app's state from the app's source code + schemas;
   kept if it loads in the simulator and every argument-free read tool returns ok.
2. task: the model writes a spoken Vietnamese request of a given category and its gold call sequence, in the style
   of train examples shown as few-shot; kept if every gold call runs ok on the simulator (no rejected call, no
   no-op write) and the target app is used; a failing gold gets --repairs turns that show the simulator results + error.
3. solve: the request alone is given to the agent (lifemate.agent, same prompt as inference) on a fresh simulator;
   kept only if its final state hash and its write / ask calls match the gold ones. The SFT row is the gold trajectory.
A slice of the accepted samples (--val-ratio) is written apart as an evaluation set for the target apps.
"""
from __future__ import annotations

import argparse
import collections
import copy
import importlib
import inspect
import json
import random
import time
import zlib
from pathlib import Path

from .agent import Agent, extract_json, run_agents
from .data import catalog_index, data_root, load_catalog, load_samples, load_zaic_tools, write_jsonl
from .prompt import SYSTEM_RULES, render_context, render_tool
from .scorer import category
from .sft import HELPER_APPS, trajectory_messages

CATEGORIES = ["single", "multi", "multi", "ask_missing"]  # per state; out_of_scope comes from the drop-app copies
CATEGORY_TEXT = {
    "single": "exactly ONE tool call in total (a single write that needs no lookup, or a single read that answers a question)",
    "multi": "several tool calls (2-6): the reads needed to find IDs / check conditions, then one or more writes; it may also "
             "use a second installed app (e.g. message a contact, add a reminder or calendar event)",
    "ask_missing": "a request that is missing one REQUIRED detail (amount, time, which item, ...): do the parts that are "
                   "possible, then end with sys.ask_missing_info naming what is missing",
}


class Chat:
    """chat(list of message lists, temperature, max_tokens) -> list of texts, on one in-process vLLM engine."""

    def __init__(self, model: str, tp: int = 1, max_model_len: int = 16384, gpu_mem: float = 0.9, seed: int = 0,
                 quantization: str | None = None):
        from vllm import LLM

        self.llm = LLM(model=model, tensor_parallel_size=tp, max_model_len=max_model_len, gpu_memory_utilization=gpu_mem,
                       seed=seed, enable_prefix_caching=True, quantization=quantization)
        self.seed = seed

    def __call__(self, convs, temperature=0.0, max_tokens=1024):
        from vllm import SamplingParams

        if not convs:
            return []
        p = SamplingParams(temperature=temperature, top_p=0.95 if temperature else 1.0, max_tokens=max_tokens, seed=self.seed)
        return [r.outputs[0].text for r in self.llm.chat(convs, p, use_tqdm=True, chat_template_kwargs={"enable_thinking": False})]


class ChatBackend:
    """Agent backend (lifemate.backends interface) on top of a Chat."""

    def __init__(self, chat):
        self.chat = chat

    def generate(self, agents):
        return self.chat([a.messages for a in agents], 0.0, 512)


def sim_sample(context: dict, states: dict) -> dict:
    """Sample dict the zaic_tools Simulator takes: installed_apps entries carry their tools and state."""
    ctx = copy.deepcopy(context)
    for app, a in ctx["installed_apps"].items():
        a["state"] = copy.deepcopy(states.get(app, {}))
    return {"id": "synth", "context": ctx, "history": []}


def run_calls(zt, catalog, context, states, calls, tools=None):
    """Execute calls on a fresh simulator -> (results, simulator, error or None). With `tools`, every x-ref argument must
    have appeared in an earlier result of the sequence (the agent's own rule: IDs come from tool results, never guessed)."""
    sim = zt.executor.Simulator(sim_sample(context, states), catalog)
    results, observed = [], ""
    for c in calls:
        if not isinstance(c, dict) or not isinstance(c.get("name"), str) or not isinstance(c.get("arguments", {}), dict):
            return results, sim, f"malformed call {c!r}"
        if tools is not None:
            props = (tools.get(c["name"], {}).get("parameters") or {}).get("properties", {})
            for k, v in (c.get("arguments") or {}).items():
                for x in v if isinstance(v, list) else [v]:
                    if props.get(k, {}).get("x-ref") and isinstance(x, str) and json.dumps(x, ensure_ascii=False) not in observed:
                        return results, sim, f"{c['name']}: {k}={x!r} not looked up first"
        r = sim.call(c["name"], c.get("arguments") or {})
        results.append(r)
        observed += json.dumps(r, ensure_ascii=False)
        if not r.get("ok"):
            return results, sim, f"{c['name']}: {r.get('error')}"
        if sim.kind(c["name"]) == "write" and r.get("done") is False:
            return results, sim, f"{c['name']}: no-op ({r.get('message')})"
        if "Error:" in str(r.get("message", "")):
            return results, sim, f"{c['name']}: {r.get('message')}"
    return results, sim, None


def base_contexts(root, ids, drop_apps, keep_apps, tools):
    """(context, states) of train samples: companion apps limited to keep_apps and minus drop_apps."""
    out = []
    env = data_root(root) / "train" / "train_environment"
    for s in load_samples("train", root):
        if ids is not None and s["id"] not in ids:
            continue
        states = json.loads((env / s["id"] / "state.json").read_text(encoding="utf-8"))
        ctx = copy.deepcopy(s["context"])
        ctx["installed_apps"] = {a: {"tools": [t for t in v["tools"] if t in tools]} for a, v in ctx["installed_apps"].items()
                                 if a in keep_apps and a not in drop_apps and a in states}
        out.append((ctx, {a: states[a] for a in ctx["installed_apps"]}))
    return out


def fewshot_pool(root, ids, drop_apps):
    """Train requests (no history) with their gold calls, by category; samples touching drop_apps excluded."""
    pool = collections.defaultdict(list)
    for s in load_samples("train", root):
        if s["history"] or (ids is not None and s["id"] not in ids):
            continue
        calls = s["label"]["tool_calls"]
        if any(c["name"].split(".")[0] in drop_apps for c in calls):
            continue
        pool[category(s)].append({"request": s["audio_text"], "calls": calls})
    return pool


def state_prompt(app, catalog_app, source, ctx):
    tools = "\n".join(render_tool(t) for t in catalog_app["tools"])
    return [{"role": "system", "content": "You write realistic test data for a Vietnamese voice assistant simulator. Answer with one JSON object only."},
            {"role": "user", "content": f"""App `{app}` ({catalog_app.get('name_vi', '')}): {catalog_app.get('description', '')}

Tools:
{tools}

Simulator source of the app (its docstring describes the state; the methods show every field they read or write):
```python
{source}
```

User context:
{render_context(ctx)}

Write the app's state as one JSON object with exactly the structure this code expects. Vietnamese names and text, realistic
for a user in Vietnam; 4-10 items per list (vary the count), ids like "<prefix>_01", "<prefix>_02" as the code uses;
dates and times consistent with the current time above (some in the past, most upcoming when the app lists future items);
prices in VND as integers. Make items varied (similar names, different prices / times) so that requests need a lookup."""}]


def task_prompt(app, catalog_app, ctx, states, cat, shots):
    others = [a for a in ctx["installed_apps"] if a != app]
    companions = {a: states[a] for a in others if a in ("contacts", "chat_ott", "calendar", "reminders", "notes", "sms",
                                                           "ewallet", "banking", "shopping_list", "tasks", "email")}
    comp = "\n".join(f"- {a}: {json.dumps(st, ensure_ascii=False)[:1500]}" for a, st in companions.items())
    tools = "\n".join(render_tool(t) for t in catalog_app["tools"])
    ex = "\n\n".join(f"Request: {e['request']}\nCalls: {json.dumps(e['calls'], ensure_ascii=False)}" for e in shots)
    return [{"role": "system", "content": "You write training data for a Vietnamese voice assistant that acts by calling tools. "
                                          "Answer with one JSON object only."},
            {"role": "user", "content": f"""The assistant follows these rules (they also define what the correct calls are):
{SYSTEM_RULES}

User context:
{render_context(ctx)}
Installed apps: {', '.join(ctx['installed_apps'])}

Target app `{app}` ({catalog_app.get('name_vi', '')}), tools:
{tools}
Its current data:
{json.dumps(states[app], ensure_ascii=False)}

Data of other installed apps (truncated):
{comp or '(none shown)'}

Examples of real requests and their correct calls (other apps; copy their spoken style and their call pattern):
{ex}

Write ONE new request that a user would say to the assistant, mainly about `{app}`, of this kind: {CATEGORY_TEXT[cat]}.
The request is spoken Vietnamese (casual, like the examples; numbers may be spelled out as spoken), it refers to things by
name / description as a person would (never by ID), and it must be answerable from the data above.
Then give the exact correct call sequence: read calls first to obtain IDs or check conditions (only the reads needed),
arguments exactly as the tool schemas require, IDs copied from the data, times in ISO-8601 with +07:00.
Answer: {{"request": "...", "calls": [{{"name": "app.tool", "arguments": {{...}}}}, ...]}}"""}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--split", required=True, help="split whose catalog + zaic_tools define the target apps")
    ap.add_argument("--targets", default="new", help="'new' (apps not in train), 'held_out' (splits/train_split.json) or a,b,c")
    ap.add_argument("--split-file", default="splits/train_split.json")
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--states-per-app", type=int, default=25)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--drop-app", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=12, help="states per round")
    ap.add_argument("--max-minutes", type=float, default=0, help="start no new round after this many minutes (0: no limit)")
    ap.add_argument("--repairs", type=int, default=1, help="repair turns for a gold that fails on the simulator")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    zt = load_zaic_tools(args.split, args.root)
    catalog = load_catalog(args.split, args.root)
    tools, apps = catalog_index(catalog)
    cat_apps = {a["id"]: a for a in catalog["apps"]}
    train_apps = {a["id"] for a in load_catalog("train", args.root)["apps"]}
    split = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
    if args.targets == "new":
        targets = sorted(set(cat_apps) - train_apps)
    elif args.targets == "held_out":
        targets = sorted(split["held_out_apps"])
    else:
        targets = args.targets.split(",")
    # experiment on held-out train apps: nothing from val_seen / val_unseen, and the held-out apps never seen as data
    ids = set(split["train"]) if args.targets == "held_out" else None
    drop = set(split["held_out_apps"]) if args.targets == "held_out" else set(targets)
    keep = set(cat_apps) - {"sys"}
    bases = base_contexts(args.root, ids, drop, keep, tools)
    pool = fewshot_pool(args.root, ids, drop | set(targets))
    print(f"targets {targets}; {len(bases)} base contexts; few-shot pool {({k: len(v) for k, v in pool.items()})}", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = collections.Counter()

    # every (target app, base context) job is fixed up front, so a resumed run continues with the same jobs
    jobs = []
    for app in targets:
        src = inspect.getsource(importlib.import_module(f"zaic_tools.apps.{app}"))
        for ctx, states in rng.sample(bases, args.states_per_app):
            ctx = copy.deepcopy(ctx)
            ctx["installed_apps"][app] = {"tools": [t["name"] for t in cat_apps[app]["tools"]]}
            items = list(ctx["installed_apps"].items())
            rng.shuffle(items)
            ctx["installed_apps"] = dict(items)
            jobs.append({"app": app, "ctx": ctx, "states": dict(states), "src": src})
    order = list(range(len(jobs)))
    rng.shuffle(order)  # rounds mix the target apps, so a partial run still covers all of them
    rounds = [order[i:i + args.chunk] for i in range(0, len(order), args.chunk)]
    done = set()
    if (out / "rounds.jsonl").exists():
        for line in (out / "rounds.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add(r["round"])
            stats.update(r["stats"])
    print(f"{len(jobs)} jobs in {len(rounds)} rounds; {len(done)} already done", flush=True)

    chat = Chat(args.model, tp=args.tp, max_model_len=args.max_model_len, gpu_mem=args.gpu_mem, seed=args.seed,
                quantization=args.quantization)
    t0 = time.time()
    for ri, idx in enumerate(rounds):
        if ri in done:
            continue
        if args.max_minutes and (time.time() - t0) / 60 > args.max_minutes:
            print(f"time budget reached: stopping before round {ri + 1}/{len(rounds)}", flush=True)
            break
        rstats = collections.Counter()
        kept = run_round(chat, zt, catalog, tools, apps, cat_apps, pool, [jobs[i] for i in idx],
                         random.Random(args.seed * 100003 + ri), rstats, args.repairs)
        with open(out / "kept.jsonl", "a", encoding="utf-8") as f:
            for g in kept:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
        with open(out / "rounds.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"round": ri, "stats": dict(rstats)}) + "\n")
        stats.update(rstats)
        write_outputs(out, tools, apps, targets, stats, args)
        print(f"round {ri + 1}/{len(rounds)}: kept {len(kept)}; totals {dict(stats)}", flush=True)
    write_outputs(out, tools, apps, targets, stats, args)


def run_round(chat, zt, catalog, tools, apps, cat_apps, pool, jobs, rng, stats, repairs=1):
    """state -> task -> solve for one round of jobs; returns the accepted gold samples."""
    # 1. states
    outs = chat([state_prompt(j["app"], cat_apps[j["app"]], j["src"], j["ctx"]) for j in jobs], 0.8, 3000)
    ok_states = []
    for j, raw in zip(jobs, outs):
        st = extract_json(raw)
        stats["state_generated"] += 1
        if not isinstance(st, dict) or not st:
            continue
        j = {"app": j["app"], "ctx": j["ctx"], "states": {**j["states"], j["app"]: st}}
        reads = [t["name"] for t in cat_apps[j["app"]]["tools"]
                 if t.get("kind") == "read" and not (t.get("parameters") or {}).get("required")
                 and not (t.get("parameters") or {}).get("x-one-of")]
        _, _, err = run_calls(zt, catalog, j["ctx"], j["states"], [{"name": n, "arguments": {}} for n in reads])
        if err is None:
            ok_states.append(j)
            stats["state_ok"] += 1
    print(f"states: {len(ok_states)}/{len(jobs)} ok", flush=True)

    # 2. tasks (one per category slot per state)
    tasks = []
    for j in ok_states:
        for cat in CATEGORIES:
            shots = rng.sample(pool[cat], min(3, len(pool[cat]))) + rng.sample(pool["multi"], 1)
            tasks.append({**j, "cat": cat, "prompt": task_prompt(j["app"], cat_apps[j["app"]], j["ctx"], j["states"], cat, shots)})
    # a gold that fails gets one repair turn: the model sees the simulator results and the error, and answers again
    golds = []
    for attempt in range(1 + repairs):
        outs = chat([t["prompt"] for t in tasks], 0.9 if attempt == 0 else 0.3, 1200)
        retry = []
        for t, raw in zip(tasks, outs):
            stats["task_generated" if attempt == 0 else "task_repair"] += 1
            obj = extract_json(raw)
            if not isinstance(obj, dict) or not isinstance(obj.get("request"), str) or not isinstance(obj.get("calls"), list):
                stats["task_bad_json"] += 1
                continue
            calls = [{"name": c.get("name"), "arguments": c.get("arguments") or {}} for c in obj["calls"] if isinstance(c, dict)]
            results, err = [], None
            if not calls or len(calls) > 8 or not any(str(c["name"]).startswith(t["app"] + ".") for c in calls):
                err, kind = f"the calls must use `{t['app']}` and be 1-8 calls", "task_bad_shape"
            elif t["cat"] == "ask_missing" and calls[-1]["name"] != "sys.ask_missing_info":
                err, kind = "this kind of request must end with sys.ask_missing_info", "task_bad_shape"
            else:
                results, sim, err = run_calls(zt, catalog, t["ctx"], t["states"], calls, tools)
                kind = "task_exec_error"
            if err is not None:
                stats[kind] += 1
                ran = "\n".join(f"{json.dumps(c, ensure_ascii=False)} -> {json.dumps(r, ensure_ascii=False)[:1200]}"
                                for c, r in zip(calls, results))
                retry.append({**t, "prompt": t["prompt"] + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": f"Run on the simulator:\n{ran or '(nothing ran)'}\nProblem: {err}\n"
                                                "Fix it: give the corrected request and calls (IDs only from results above "
                                                "or the data, each needed read before the write). Answer with one JSON object only."}]})
                continue
            request = obj["request"].strip()
            gid = f"synth_{t['app']}_{zlib.crc32((request + json.dumps(calls, ensure_ascii=False)).encode()):08x}"
            stats["task_ok" if attempt == 0 else "task_repaired"] += 1
            golds.append({**{k: v for k, v in t.items() if k != "prompt"}, "id": gid, "request": request, "calls": calls,
                          "results": results, "hash": zt.state_hash(sim.states)})
        tasks = retry
    print(f"tasks: {len(golds)} execute cleanly ({stats['task_repaired']} after repair)", flush=True)

    # 3. solve with the agent, keep matching ones
    agents = []
    for g in golds:
        sim = zt.executor.Simulator(sim_sample(g["ctx"], g["states"]), catalog)
        agents.append(Agent({"context": g["ctx"], "history": []}, sim, tools, apps, g["request"], []))
    run_agents(agents, ChatBackend(chat), wave=64)

    def key_calls(calls):
        return sorted(c["name"] for c in calls if tools.get(c["name"], {}).get("kind") != "read")

    kept = []
    for g, a in zip(golds, agents):
        got = [{"name": c["name"], "arguments": c["arguments"]} for c in a.env.log]
        same = zt.state_hash(a.env.states) == g["hash"] and key_calls(got) == key_calls(g["calls"])
        if same and not key_calls(g["calls"]):  # read-only request: the reads themselves are the answer
            same = {c["name"] for c in got} == {c["name"] for c in g["calls"]}
        stats["solve_match" if same else "solve_mismatch"] += 1
        if same:
            kept.append(g)
    print(f"solve: {len(kept)}/{len(golds)} agent runs match the gold", flush=True)
    return kept


def write_outputs(out, tools, apps, targets, stats, args):
    """SFT rows (+ drop-app copies), held-out evaluation samples and per-app stats from all of <out>/kept.jsonl.
    The val / drop-app choices hash the sample id, so they do not change as rounds are added."""
    kept, seen = [], set()
    if (out / "kept.jsonl").exists():
        for line in (out / "kept.jsonl").read_text(encoding="utf-8").splitlines():
            g = json.loads(line)
            if g["id"] not in seen:
                seen.add(g["id"])
                kept.append(g)

    def frac(key):
        return zlib.crc32(key.encode()) % 10000 / 10000

    val, train_rows = [], []
    for g in kept:
        if frac(g["id"]) < args.val_ratio:
            val.append(g)
            continue
        s = {"context": g["ctx"], "history": []}
        train_rows.append({"id": g["id"], "messages": trajectory_messages(s, tools, apps, g["request"], [], g["calls"], g["results"])})
        main_apps = {c["name"].split(".")[0] for c in g["calls"]} - HELPER_APPS
        if main_apps == {g["app"]} and frac(g["id"] + "#drop") < args.drop_app:
            d = copy.deepcopy(s)
            d["context"]["installed_apps"].pop(g["app"])
            train_rows.append({"id": f"{g['id']}#drop_{g['app']}", "messages": trajectory_messages(d, tools, apps, g["request"], [], [], [])})
    random.Random(0).shuffle(train_rows)
    write_jsonl(out / "sft.jsonl", train_rows)
    write_jsonl(out / "val.jsonl", [{"id": g["id"], "app": g["app"], "category": g["cat"], "request": g["request"], "context": g["ctx"],
                                     "states": g["states"], "calls": g["calls"], "hash": g["hash"]} for g in val])
    write_jsonl(out / "states.jsonl", [{"app": g["app"], "state": g["states"][g["app"]]} for g in kept])
    per_app = collections.Counter(g["app"] for g in kept)
    per_cat = collections.Counter(category({"label": {"tool_calls": g["calls"]}}) for g in kept)
    summary = {"targets": targets, "stats": dict(stats), "kept": len(kept), "val": len(val), "sft_rows": len(train_rows),
               "per_app": dict(per_app), "per_category": dict(per_cat), "model": args.model}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
