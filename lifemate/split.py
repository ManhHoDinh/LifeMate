"""Leave-apps-out validation split of train.

val_unseen: samples whose gold calls or history calls touch a held-out app (simulates the new apps of public/private test).
val_seen:   a random slice of the remaining samples (in-distribution).
train:      the rest; when building SFT data, the held-out apps must also be removed from installed_apps of these samples.

    python -m lifemate.split --out splits/train_split.json
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from .data import load_samples

# spread over groups, 25-65 gold calls each; none is a cross-app helper like contacts
HELD_OUT_APPS = ["door_lock", "learning", "household_chores", "loyalty", "flights", "podcast_radio", "medication", "library"]


def apps_used(sample: dict) -> set[str]:
    calls = list(sample["label"]["tool_calls"])
    for h in sample["history"]:
        calls += h.get("tool_calls") or []
    return {c["name"].split(".")[0] for c in calls} - {"sys"}


def make_split(samples, held_out=HELD_OUT_APPS, n_seen=200, seed=0) -> dict:
    held = set(held_out)
    unseen = [s["id"] for s in samples if apps_used(s) & held]
    rest = [s["id"] for s in samples if not apps_used(s) & held]
    rng = random.Random(seed)
    seen = sorted(rng.sample(rest, n_seen))
    train = sorted(set(rest) - set(seen))
    return {"held_out_apps": sorted(held), "train": train, "val_seen": seen, "val_unseen": sorted(unseen)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default="splits/train_split.json")
    ap.add_argument("--n-seen", type=int, default=200)
    args = ap.parse_args()
    samples = load_samples("train", args.root)
    split = make_split(samples, n_seen=args.n_seen)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(split, ensure_ascii=False, indent=1), encoding="utf-8")
    by_id = {s["id"]: s for s in samples}
    for k in ("train", "val_seen", "val_unseen"):
        cats = collections.Counter("empty" if not by_id[i]["label"]["tool_calls"] else
                                   "ask" if any(c["name"] == "sys.ask_missing_info" for c in by_id[i]["label"]["tool_calls"]) else "act"
                                   for i in split[k])
        print(f"{k:11s} {len(split[k]):5d}  {dict(cats)}")
    print("held-out apps:", split["held_out_apps"], "->", args.out)


if __name__ == "__main__":
    main()
