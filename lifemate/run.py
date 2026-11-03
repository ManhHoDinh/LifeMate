"""Run the agent on a split and write submission.jsonl (+ traces, + score on train).

    # end-to-end check of loop + scorer (must print score 1.0)
    python -m lifemate.run --split train --ids val_seen --backend oracle
    # LLM baseline on gold transcripts (GPU)
    python -m lifemate.run --split train --ids val_unseen --backend vllm --model Qwen/Qwen3-4B-Instruct-2507
    # public test needs ASR transcripts: {"audio/<id>/<id>.wav": "text", ...}
    python -m lifemate.run --split public_test --transcripts asr/public_test.json --backend vllm --model ...
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from .agent import Agent, run_agents
from .backends import make_backend
from .data import catalog_index, data_root, fuzzy_params, load_catalog, load_samples, load_zaic_tools, write_jsonl


def select(samples, ids_arg: str | None, split_file: str, limit: int | None):
    if ids_arg and ids_arg != "all":
        if Path(ids_arg).is_file():
            keep = set(json.loads(Path(ids_arg).read_text(encoding="utf-8")))
        else:
            keep = set(json.loads(Path(split_file).read_text(encoding="utf-8"))[ids_arg])
        samples = [s for s in samples if s["id"] in keep]
    if limit:
        samples = random.Random(0).sample(samples, min(limit, len(samples)))
    return samples


def texts_for(sample: dict, transcripts: dict | None) -> tuple[str, list[str]]:
    """(request transcript, transcripts of the user turns in history): ASR file if given, else train's audio_text."""
    users = [h for h in sample["history"] if h["role"] == "user"]
    if transcripts is not None:
        return transcripts[sample["audio"]], [transcripts[h["audio"]] for h in users]
    if "audio_text" not in sample:
        raise SystemExit(f"{sample['id']} has no audio_text: pass --transcripts (ASR output) for this split")
    return sample["audio_text"], [h.get("audio_text", "") for h in users]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--root", default=None, help="folder holding the split folders")
    ap.add_argument("--ids", default=None, help="key of --split-file (train/val_seen/val_unseen), a JSON id list file, or all")
    ap.add_argument("--split-file", default="splits/train_split.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--backend", default="oracle", choices=["oracle", "empty", "openai", "vllm"])
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--no-think", action="store_true", help="openai backend: pass enable_thinking=False (Qwen3 hybrid models)")
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--gpu-mem", type=float, default=0.85, help="vllm gpu_memory_utilization")
    ap.add_argument("--wave", type=int, default=0, help="run agents in groups of N to completion (0 = all at once)")
    ap.add_argument("--lora", default=None, help="vllm: LoRA adapter dir from lifemate.train")
    ap.add_argument("--transcripts", default=None, help="JSON {audio path: text}; default = train audio_text")
    ap.add_argument("--out", default=None, help="default runs/<split>_<ids>_<backend>")
    args = ap.parse_args()

    root = data_root(args.root)
    zt = load_zaic_tools(args.split, root)
    samples = select(load_samples(args.split, args.root), args.ids, args.split_file, args.limit)
    catalog = load_catalog(args.split, args.root)
    tools, apps = catalog_index(catalog)
    transcripts = json.loads(Path(args.transcripts).read_text(encoding="utf-8")) if args.transcripts else None
    out_dir = Path(args.out or f"runs/{args.split}_{args.ids or 'all'}_{args.backend}")
    out_dir.mkdir(parents=True, exist_ok=True)

    backend = make_backend(args.backend, model=args.model, base_url=args.base_url, api_key=args.api_key,
                           no_think=args.no_think, max_model_len=args.max_model_len, gpu_mem=args.gpu_mem, lora=args.lora)
    agents = []
    for s in samples:
        req, hist = texts_for(s, transcripts)
        env = zt.ToolEnv(args.split, s["id"], root=root)
        agents.append(Agent(s, env, tools, apps, req, hist))

    t0 = time.time()
    run_agents(agents, backend, log_every=0 if args.wave else 1 if args.backend in ("vllm", "openai") else 0, wave=args.wave)
    elapsed = time.time() - t0

    sub = [{"id": a.sample["id"], "tool_calls": a.env.get_tool_calls(), "state_hash": a.env.get_state_hash()} for a in agents]
    write_jsonl(out_dir / "submission.jsonl", sub)
    write_jsonl(out_dir / "traces.jsonl", [{"id": a.sample["id"], "messages": a.messages, "trace": a.trace} for a in agents])
    print(f"{len(sub)} samples in {elapsed:.1f}s ({elapsed / max(len(sub), 1):.2f}s/sample) -> {out_dir}")

    if args.split == "train":
        from .scorer import gold_hashes, score_submission

        res = score_submission(sub, samples, fuzzy_params(tools), gold_hashes(samples, args.split, root))
        write_jsonl(out_dir / "scores.jsonl", res["rows"])
        o = res["overall"]
        print(f"SCORE {o['score']:.4f}  (F1 {o['f1']:.4f}, hash acc {o['hash_acc']:.4f}, n={o['n']})")
        for k, v in res["by_category"].items():
            print(f"  {k:13s} n={v['n']:4d}  score {v['score']:.4f}  F1 {v['f1']:.4f}  hash {v['hash_acc']:.4f}")
        (out_dir / "summary.json").write_text(json.dumps({k: res[k] for k in ("overall", "by_category")}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
