"""Build the models/ folder the Docker image ships: merged LLM (base + LoRA) and the fine-tuned ASR.

    python -m lifemate.export --adapter ckpt/adapter --asr ckpt/asr_ft --out models
    -> models/llm (Qwen3-4B-Instruct-2507 + LoRA merged, fp16 safetensors), models/asr (Whisper), models/MANIFEST.json

Budget check (rules: whole system < 6B params, LLM < 5B, context <= 16K) is computed from the saved weights.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def legacy_preprocessor(d) -> None:
    """transformers v5 nests the feature extractor in processor_config.json; v4 (e.g. the vLLM image) reads
    preprocessor_config.json. Write the latter so the Whisper folder loads under both."""
    d = Path(d)
    pre, proc = d / "preprocessor_config.json", d / "processor_config.json"
    if not pre.is_file() and proc.is_file():
        fe = json.loads(proc.read_text(encoding="utf-8")).get("feature_extractor")
        if fe:
            fe.setdefault("processor_class", "WhisperProcessor")
            pre.write_text(json.dumps(fe, indent=2), encoding="utf-8")


def shard_safetensors(d, n: int) -> None:
    """Split d/model.safetensors into n shards + model.safetensors.index.json (the layout save_pretrained writes for big
    models), copying the raw tensor bytes. Lets the Dockerfile put each ~1 GB shard in its own layer, so a dropped
    connection during `docker push` re-sends one shard instead of the whole 8 GB file."""
    d = Path(d)
    src = d / "model.safetensors"
    with open(src, "rb") as f:
        hlen = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(hlen))
        base = 8 + hlen
        meta = header.pop("__metadata__", None)
        names = sorted(header, key=lambda k: header[k]["data_offsets"][0])
        total = sum(header[k]["data_offsets"][1] - header[k]["data_offsets"][0] for k in names)
        groups, cur, size = [], [], 0
        for k in names:  # contiguous groups of ~total/n bytes
            cur.append(k)
            size += header[k]["data_offsets"][1] - header[k]["data_offsets"][0]
            if size >= total * (len(groups) + 1) / n and len(groups) < n - 1:
                groups.append(cur)
                cur = []
        groups.append(cur)
        weight_map = {}
        for i, g in enumerate(groups, 1):
            name = f"model-{i:05d}-of-{len(groups):05d}.safetensors"
            h, off = {}, 0
            for k in g:
                b, e = header[k]["data_offsets"]
                h[k] = {**header[k], "data_offsets": [off, off + e - b]}
                off += e - b
                weight_map[k] = name
            if meta:
                h["__metadata__"] = meta
            hb = json.dumps(h, separators=(",", ":")).encode()
            hb += b" " * (-len(hb) % 8)
            with open(d / name, "wb") as out:
                out.write(len(hb).to_bytes(8, "little") + hb)
                for k in g:
                    b, e = header[k]["data_offsets"]
                    f.seek(base + b)
                    out.write(f.read(e - b))
    (d / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2), encoding="utf-8")
    src.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--adapter", default=None, help="LoRA dir; without it the base model is exported as is")
    ap.add_argument("--asr", required=True, help="Whisper model dir or hub id")
    ap.add_argument("--out", default="models")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, WhisperForConditionalGeneration, WhisperProcessor

    out = Path(args.out)
    llm = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float16)
    if args.adapter:
        from peft import PeftModel

        llm = PeftModel.from_pretrained(llm, args.adapter).merge_and_unload()
    n_llm = count_params(llm)
    llm.save_pretrained(out / "llm", safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(out / "llm")
    del llm

    asr = WhisperForConditionalGeneration.from_pretrained(args.asr, dtype=torch.float16)
    n_asr = count_params(asr)
    asr.save_pretrained(out / "asr", safe_serialization=True)
    WhisperProcessor.from_pretrained(args.asr).save_pretrained(out / "asr")
    gen = Path(args.asr) / "generation_config.json"
    if gen.is_file():
        shutil.copy(gen, out / "asr" / "generation_config.json")
    legacy_preprocessor(out / "asr")

    manifest = {"llm": {"base": args.base, "adapter": args.adapter, "params": n_llm}, "asr": {"source": args.asr, "params": n_asr},
                "total_params": n_llm + n_asr, "max_model_len": 16384}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps(manifest, indent=1))
    assert n_llm < 5e9 and n_llm + n_asr < 6e9, "parameter budget exceeded"


if __name__ == "__main__":
    main()
