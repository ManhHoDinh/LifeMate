"""SFT data: the gold trajectories of train rendered in exactly the agent's prompt format.

One example = one sample's whole conversation: system + request, then for every gold call the assistant JSON and the
tool result message the agent would feed back, closed by {"name": "done"} (no closing turn after
sys.ask_missing_info, where the agent stops). Loss goes on assistant turns only (see lifemate.train).

    python -m lifemate.sft --ids train --out sft/train.jsonl            # prompts hide the held-out apps
    python -m lifemate.sft --ids train --asr asr/train.json --asr-ratio 0.5 --out sft/train_asr.jsonl

Augmentations (deterministic, seed 0):
- installed apps shuffled, held-out apps (splits/train_split.json) removed from every prompt;
- --drop-app P: with probability P a sample whose gold calls touch exactly one non-helper app is copied with that app
  uninstalled and an empty target (teaches "no installed app can do it");
- --asr: request / history texts replaced by ASR transcripts with probability --asr-ratio (robustness to ASR errors).
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

from .data import catalog_index, load_catalog, load_samples, write_jsonl
from .prompt import build_messages, tool_result_message
from .run import select
from .split import apps_used

DONE = json.dumps({"name": "done"}, ensure_ascii=False)
HELPER_APPS = {"contacts", "sys"}  # used alongside the main app, never the reason a request is in scope


def trajectory_messages(sample, tools, apps, request_text, history_texts, calls, results) -> list[dict]:
    msgs = build_messages(sample, tools, apps, request_text, history_texts)
    for c, r in zip(calls, results):
        msgs.append({"role": "assistant", "content": json.dumps({"name": c["name"], "arguments": c["arguments"]}, ensure_ascii=False)})
        if c["name"] == "sys.ask_missing_info":
            return msgs
        msgs.append(tool_result_message(r))
    msgs.append({"role": "assistant", "content": DONE})
    return msgs


def with_apps(sample: dict, keep) -> dict:
    s = copy.deepcopy(sample)
    s["context"]["installed_apps"] = {k: v for k, v in sample["context"]["installed_apps"].items() if keep(k)}
    return s


def shuffled_apps(sample: dict, rng: random.Random) -> dict:
    items = list(sample["context"]["installed_apps"].items())
    rng.shuffle(items)
    s = dict(sample)
    s["context"] = {**sample["context"], "installed_apps": dict(items)}
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--ids", default="train")
    ap.add_argument("--split-file", default="splits/train_split.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--drop-app", type=float, default=0.1)
    ap.add_argument("--asr", default=None, help="JSON {audio path: text}")
    ap.add_argument("--asr-ratio", type=float, default=0.5)
    ap.add_argument("--keep-held-out", action="store_true", help="final model: train on every app")
    args = ap.parse_args()

    rng = random.Random(0)
    split = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
    held = set() if args.keep_held_out else set(split["held_out_apps"])
    samples = select(load_samples("train", args.root), args.ids, args.split_file, None)
    tools, apps = catalog_index(load_catalog("train", args.root))
    asr = json.loads(Path(args.asr).read_text(encoding="utf-8")) if args.asr else {}

    rows = []
    stats = {"samples": 0, "dropped_app": 0, "asr_text": 0}
    for s in samples:
        s = shuffled_apps(with_apps(s, lambda a: a not in held), rng)
        users = [h for h in s["history"] if h["role"] == "user"]
        req, hist = s["audio_text"], [h.get("audio_text", "") for h in users]
        if s["audio"] in asr and rng.random() < args.asr_ratio:
            req, hist = asr[s["audio"]], [asr.get(h["audio"], t) for h, t in zip(users, hist)]
            stats["asr_text"] += 1
        lab = s["label"]
        rows.append({"id": s["id"], "messages": trajectory_messages(s, tools, apps, req, hist, lab["tool_calls"], lab["tool_results"])})
        stats["samples"] += 1

        main_apps = {c["name"].split(".")[0] for c in lab["tool_calls"]} - HELPER_APPS
        if len(main_apps) == 1 and rng.random() < args.drop_app:
            gone = next(iter(main_apps))
            if gone not in apps_used({"label": {"tool_calls": []}, "history": s["history"]}):
                d = with_apps(s, lambda a: a != gone)
                rows.append({"id": f"{s['id']}#drop_{gone}", "messages": trajectory_messages(d, tools, apps, req, hist, [], [])})
                stats["dropped_app"] += 1

    rng.shuffle(rows)
    write_jsonl(args.out, rows)
    print(f"{len(rows)} examples -> {args.out}  {stats}  held-out apps hidden: {sorted(held)}")


if __name__ == "__main__":
    main()
