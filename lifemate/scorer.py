"""Local re-implementation of the LifeMate metric (train split, where gold labels exist).

Per sample:
  gold == []            -> 1 if submission == [] else 0
  state_hash != gold    -> 0
  otherwise             -> F1 with P = M / (M + U), R = M / |G|
A submitted call matches a gold call when the name is equal, every non-fuzzy argument is identical and every fuzzy
argument (catalog x-match: fuzzy) has token-F1 > 0.6 after lower-casing + NFC. One-to-one matching, order ignored.

Assumptions (the official formula image was not in the brief): sample score = F1 when the hash matches, final = mean;
arguments are compared over the union of keys (an extra non-fuzzy argument breaks the match); lists of scalars are
compared as multisets (e.g. sys.ask_missing_info.params).
"""
from __future__ import annotations

import collections
import unicodedata

FUZZY_THRESHOLD = 0.6


def _tok(s) -> list[str]:
    return unicodedata.normalize("NFC", str(s)).lower().split()


def token_f1(pred, gold) -> float:
    p, g = _tok(pred), _tok(gold)
    if not p and not g:
        return 1.0
    common = sum((collections.Counter(p) & collections.Counter(g)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def _norm_value(v):
    if isinstance(v, list) and all(not isinstance(x, (dict, list)) for x in v):
        return sorted(v, key=lambda x: (type(x).__name__, str(x)))
    return v


def calls_match(sub: dict, gold: dict, fuzzy: dict[str, set[str]]) -> bool:
    if sub.get("name") != gold.get("name"):
        return False
    sa, ga = sub.get("arguments") or {}, gold.get("arguments") or {}
    fz = fuzzy.get(gold["name"], set())
    for k in set(sa) | set(ga):
        if k in fz:
            if k not in ga and k not in sa:
                continue
            if k not in ga or k not in sa or token_f1(sa[k], ga[k]) <= FUZZY_THRESHOLD:
                return False
        elif _norm_value(sa.get(k)) != _norm_value(ga.get(k)):
            return False
    return True


def max_matching(subs: list[dict], golds: list[dict], fuzzy) -> int:
    """Maximum bipartite matching (Kuhn's algorithm); lists are tiny."""
    adj = [[j for j, g in enumerate(golds) if calls_match(s, g, fuzzy)] for s in subs]
    owner = [-1] * len(golds)

    def try_assign(i, seen):
        for j in adj[i]:
            if j in seen:
                continue
            seen.add(j)
            if owner[j] == -1 or try_assign(owner[j], seen):
                owner[j] = i
                return True
        return False

    return sum(try_assign(i, set()) for i in range(len(subs)))


def score_sample(sub_calls: list[dict], sub_hash: str, gold_calls: list[dict], gold_hash: str, fuzzy) -> dict:
    if not gold_calls:
        s = 1.0 if not sub_calls else 0.0
        return {"score": s, "f1": s, "hash_ok": not sub_calls, "matched": 0}
    m = max_matching(sub_calls, gold_calls, fuzzy)
    p = m / len(sub_calls) if sub_calls else 0.0
    r = m / len(gold_calls)
    f1 = 2 * p * r / (p + r) if m else 0.0
    hash_ok = sub_hash == gold_hash
    return {"score": f1 if hash_ok else 0.0, "f1": f1, "hash_ok": hash_ok, "matched": m}


def gold_hashes(samples: list[dict], split: str, root) -> dict[str, str]:
    """Replay each gold call list on a fresh environment (evaluation only, never in a submission run)."""
    from .data import load_zaic_tools

    zt = load_zaic_tools(split, root)
    out = {}
    for s in samples:
        env = zt.ToolEnv(split, s["id"], root=root)
        for c in s["label"]["tool_calls"]:
            env.call(c["name"], c["arguments"])
        out[s["id"]] = env.get_state_hash()
    return out


def category(sample: dict) -> str:
    calls = sample["label"]["tool_calls"]
    if not calls:
        return "out_of_scope"
    if any(c["name"] == "sys.ask_missing_info" for c in calls):
        return "ask_missing"
    return "single" if len(calls) == 1 else "multi"


def score_submission(submission: list[dict], samples: list[dict], fuzzy, gold_hash: dict[str, str]) -> dict:
    """submission rows: {id, tool_calls, state_hash}; samples: labelled samples to score against (others ignored)."""
    by_id = {r["id"]: r for r in submission}
    rows = []
    for s in samples:
        r = by_id.get(s["id"], {"tool_calls": [], "state_hash": ""})
        res = score_sample(r["tool_calls"], r["state_hash"], s["label"]["tool_calls"], gold_hash[s["id"]], fuzzy)
        res.update(id=s["id"], category=category(s), n_sub=len(r["tool_calls"]), n_gold=len(s["label"]["tool_calls"]))
        rows.append(res)

    def agg(rs):
        n = len(rs) or 1
        return {"n": len(rs), "score": sum(x["score"] for x in rs) / n, "f1": sum(x["f1"] for x in rs) / n,
                "hash_acc": sum(x["hash_ok"] for x in rs) / n}

    by_cat = collections.defaultdict(list)
    for x in rows:
        by_cat[x["category"]].append(x)
    return {"overall": agg(rows), "by_category": {k: agg(v) for k, v in sorted(by_cat.items())}, "rows": rows}
