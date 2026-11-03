"""Error analysis of a train run: where the score is lost.

    python -m lifemate.analyze runs/val_seen_qwen3_4b [--show 5]
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from .data import catalog_index, fuzzy_params, load_catalog, load_samples, read_jsonl
from .scorer import calls_match


def diagnose(sub: list[dict], gold: list[dict], fuzzy) -> list[str]:
    """Error tags of one sample (empty list = perfect)."""
    if not gold:
        return [] if not sub else ["acted_on_out_of_scope"]
    if not sub:
        return ["no_call"]
    tags = []
    gold_ask = any(c["name"] == "sys.ask_missing_info" for c in gold)
    sub_ask = any(c["name"] == "sys.ask_missing_info" for c in sub)
    if gold_ask and not sub_ask:
        tags.append("missed_ask")
    if sub_ask and not gold_ask:
        tags.append("spurious_ask")
    unmatched_g = list(gold)
    unmatched_s = []
    for s in sub:
        j = next((j for j, g in enumerate(unmatched_g) if calls_match(s, g, fuzzy)), None)
        if j is None:
            unmatched_s.append(s)
        else:
            unmatched_g.pop(j)
    for s in unmatched_s:
        same = [g for g in unmatched_g if g["name"] == s["name"]]
        if same:
            g = same[0]
            keys = sorted(k for k in set(s["arguments"]) | set(g["arguments"]) if s["arguments"].get(k) != g["arguments"].get(k))
            tags.append(f"wrong_args:{s['name']}:{','.join(keys)}")
            unmatched_g.remove(g)
        else:
            kind = "read" if any(w in s["name"] for w in (".get", ".list", ".search", "status", ".find", "_status", "guide")) else "call"
            tags.append(f"extra_{kind}:{s['name']}")
    for g in unmatched_g:
        tags.append(f"missing:{g['name']}")
    return tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--root", default=None)
    ap.add_argument("--show", type=int, default=3, help="examples printed per top error tag")
    args = ap.parse_args()
    run = Path(args.run_dir)
    tools, _ = catalog_index(load_catalog("train", args.root))
    fuzzy = fuzzy_params(tools)
    samples = {s["id"]: s for s in load_samples("train", args.root)}
    sub = {r["id"]: r for r in read_jsonl(run / "submission.jsonl")}
    scores = {r["id"]: r for r in read_jsonl(run / "scores.jsonl")}
    traces = {r["id"]: r["trace"] for r in read_jsonl(run / "traces.jsonl")} if (run / "traces.jsonl").exists() else {}

    print(json.dumps(json.loads((run / "summary.json").read_text()), indent=1))
    tag_count, tag_family, examples = collections.Counter(), collections.Counter(), collections.defaultdict(list)
    finish = collections.Counter()
    invalid = collections.Counter()
    for sid, sc in scores.items():
        tags = diagnose(sub[sid]["tool_calls"], samples[sid]["label"]["tool_calls"], fuzzy)
        for t in tags:
            tag_count[t] += 1
            tag_family[t.split(":")[0]] += 1
            examples[t.split(":")[0]].append(sid)
        for e in traces.get(sid, []):
            if e["event"] == "finish":
                finish[e["why"]] += 1
            elif e["event"] == "invalid":
                invalid[e["error"].split(":")[0]] += 1
        if not sc["hash_ok"] and not tags:
            tag_family["hash_only"] += 1

    n = len(scores)
    print(f"\nsamples with errors: {sum(1 for s in scores.values() if s['score'] < 1)}/{n}")
    print("\nerror families (count of samples x tags):")
    for k, v in tag_family.most_common():
        print(f"  {k:24s} {v}")
    print("\ntop specific errors:")
    for k, v in tag_count.most_common(25):
        print(f"  {v:4d}  {k}")
    print("\nfinish reasons:", dict(finish))
    print("schema-rejected attempts (not executed):", dict(invalid))

    for fam, _ in tag_family.most_common(6):
        for sid in examples.get(fam, [])[: args.show]:
            s = samples[sid]
            print(f"\n=== [{fam}] {sid}: {s['audio_text']}")
            print("  gold:", json.dumps(s["label"]["tool_calls"], ensure_ascii=False)[:600])
            print("  pred:", json.dumps(sub[sid]["tool_calls"], ensure_ascii=False)[:600])


if __name__ == "__main__":
    main()
