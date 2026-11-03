"""Final inference: <split folder> -> result/submission.jsonl. This is the Docker entrypoint.

    python -m lifemate.infer --data-dir /data --asr-model models/asr --llm-model models/llm --out result/submission.jsonl

--data-dir is a split folder as released (public_test/, private_test/): <split>.jsonl, audio/, <split>_environment/,
<split>_tools_catalog.json, zaic_tools/. The organisers mount it at /data, so the split name comes from the jsonl
file name, not from the folder name.

Pipeline (fully offline, deterministic: greedy decoding, fixed seeds):
1. ASR on every user turn (final request + history turns), then the ASR model is freed;
2. one ToolEnv per sample (the split's own zaic_tools) and the LLM agent loop, batched in waves on vLLM;
3. one submission line per sample: every call made on its env and the env's state hash.
--backend oracle / --transcripts let the whole path be tested without a GPU (train split).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
import time
from pathlib import Path

from .agent import Agent, run_agents
from .asr import user_turns
from .backends import make_backend
from .data import catalog_index, load_catalog, load_samples, load_zaic_tools, write_jsonl


def resolve_split(data_dir: Path) -> tuple[str, Path]:
    """(split name, root holding <split>/) for a split folder mounted under any name."""
    cands = [p for p in data_dir.glob("*.jsonl") if (data_dir / f"{p.stem}_tools_catalog.json").is_file()]
    if len(cands) != 1:
        raise SystemExit(f"cannot tell the split of {data_dir}: expected one <split>.jsonl next to <split>_tools_catalog.json")
    split = cands[0].stem
    if data_dir.name == split:
        return split, data_dir.parent
    root = Path(tempfile.mkdtemp(prefix="lifemate_root_"))
    os.symlink(data_dir.resolve(), root / split, target_is_directory=True)
    return split, root


def seed_everything(seed: int = 0):
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/data")
    ap.add_argument("--out", default="result/submission.jsonl")
    ap.add_argument("--asr-model", default="models/asr")
    ap.add_argument("--llm-model", default="models/llm")
    ap.add_argument("--lora", default=None, help="LoRA adapter dir when --llm-model is the base model")
    ap.add_argument("--backend", default="vllm", choices=["vllm", "oracle", "empty", "openai"])
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--transcripts", default=None, help="JSON {audio path: text}: skip ASR (tests)")
    ap.add_argument("--wave", type=int, default=8)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--asr-batch", type=int, default=16)
    ap.add_argument("--no-cudnn", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="first N samples (tests)")
    args = ap.parse_args()

    seed_everything(0)
    t_start = time.time()
    split, root = resolve_split(Path(args.data_dir))
    zt = load_zaic_tools(split, root)
    samples = load_samples(split, root)[: args.limit]
    tools, apps = catalog_index(load_catalog(split, root))
    print(f"split {split}: {len(samples)} samples, root {root}", flush=True)

    # 1. ASR
    if args.transcripts:
        texts = json.loads(Path(args.transcripts).read_text(encoding="utf-8"))
    else:
        from .asr import transcribe

        paths = sorted({p for s in samples for p, _ in user_turns(s)})
        t0 = time.time()
        texts = transcribe(paths, root / split, args.asr_model, args.asr_batch, args.no_cudnn)
        print(f"ASR: {len(paths)} clips in {time.time() - t0:.1f}s", flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).with_name("transcripts.json").write_text(json.dumps(texts, ensure_ascii=False, indent=0), encoding="utf-8")

    # 2. agents: exactly one ToolEnv per sample
    backend = make_backend(args.backend, model=args.llm_model, base_url=args.base_url, gpu_mem=args.gpu_mem, lora=args.lora)
    agents, envs = [], {}
    for s in samples:
        env = zt.ToolEnv(split, s["id"], root=root)
        envs[s["id"]] = env
        users = [h for h in s["history"] if h["role"] == "user"]
        req = texts.get(s["audio"], "")
        hist = [texts.get(h["audio"], "") for h in users]
        agents.append(Agent(s, env, tools, apps, req, hist))
    t0 = time.time()
    run_agents(agents, backend, wave=args.wave)
    print(f"agent: {len(agents)} samples in {time.time() - t0:.1f}s", flush=True)

    # 3. submission: one line per sample, in the input order
    rows = [{"id": s["id"], "tool_calls": envs[s["id"]].get_tool_calls(), "state_hash": envs[s["id"]].get_state_hash()} for s in samples]
    write_jsonl(args.out, rows)
    total = time.time() - t_start
    print(f"{len(rows)} lines -> {args.out}  total {total:.1f}s ({total / max(len(rows), 1):.2f}s/sample)", flush=True)


if __name__ == "__main__":
    main()
