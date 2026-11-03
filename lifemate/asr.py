"""ASR: transcribe every user turn of a split (final request + history turns) -> {audio path: text} for run.py --transcripts.

    python -m lifemate.asr --split train --ids val_seen --model vinai/PhoWhisper-medium --out asr/train_val_seen.json
    python -m lifemate.asr --split public_test --model vinai/PhoWhisper-medium --out asr/public_test.json
On train the word error rate against audio_text is printed (overall and on number / time words, the costly errors).
Greedy decoding, fixed seed: deterministic as the rules require.
"""
from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from pathlib import Path

from .data import data_root, load_samples
from .run import select


def user_turns(sample: dict) -> list[tuple[str, str | None]]:
    """[(audio path, reference text or None)] for the final request and every user turn of history."""
    out = [(h["audio"], h.get("audio_text")) for h in sample["history"] if h["role"] == "user"]
    out.append((sample["audio"], sample.get("audio_text")))
    return out


def norm_words(s: str) -> list[str]:
    s = unicodedata.normalize("NFC", s.lower())
    return re.sub(r"[^\w\s]", " ", s).split()


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


NUMBER_WORDS = {"một", "hai", "ba", "bốn", "tư", "năm", "lăm", "sáu", "bảy", "tám", "chín", "mười", "mươi", "trăm", "nghìn",
                "ngàn", "triệu", "tỷ", "tỉ", "rưỡi", "giờ", "phút", "kém", "linh", "lẻ", "mốt"}


def number_recall(hyp: list[str], ref: list[str]) -> tuple[int, int]:
    """(found, total) of reference number/time words (digits included) present in the hypothesis."""
    hs = set(hyp)
    keys = [w for w in ref if w in NUMBER_WORDS or any(c.isdigit() for c in w)]
    return sum(w in hs for w in keys), len(keys)


def load_wav(path, sr: int = 16000):
    """Mono float32 at `sr`, decoded here so the pipeline needs neither ffmpeg nor torchaudio (the competition wavs are 16 kHz mono)."""
    import numpy as np

    try:
        import soundfile as sf

        x, rate = sf.read(str(path), dtype="float32", always_2d=True)
    except ImportError:
        import wave

        with wave.open(str(path)) as w:
            rate, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
            assert width == 2, f"{path}: {8 * width}-bit wav needs soundfile"
            x = np.frombuffer(w.readframes(w.getnframes()), "<i2").reshape(-1, ch).astype(np.float32) / 32768
    x = x.mean(axis=1)
    if rate != sr:
        x = np.interp(np.arange(0, len(x), rate / sr), np.arange(len(x)), x).astype(np.float32)
    return x


def transcribe(paths: list[str], audio_root: Path, model: str, batch_size: int = 16, no_cudnn: bool = False) -> dict[str, str]:
    """{relative audio path: transcript}; greedy decoding, fp16 on GPU. Frees the model's GPU memory before returning."""
    import gc

    import torch
    from transformers import pipeline, set_seed

    torch.backends.cudnn.enabled = not no_cudnn
    set_seed(0)
    cuda = torch.cuda.is_available()
    asr = pipeline("automatic-speech-recognition", model=model, device=0 if cuda else -1,
                   torch_dtype=torch.float16 if cuda else torch.float32)
    clips = ({"raw": load_wav(audio_root / p), "sampling_rate": 16000} for p in paths)
    outs = asr(clips, batch_size=batch_size,
               generate_kwargs={"num_beams": 1, "do_sample": False, "language": "vi", "task": "transcribe"})
    result = {p: o["text"].strip() for p, o in zip(paths, outs)}
    del asr
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--root", default=None)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--split-file", default="splits/train_split.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default="vinai/PhoWhisper-medium")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-cudnn", action="store_true",
                    help="disable cuDNN (Colab after `pip install vllm`: CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH in the conv encoder)")
    args = ap.parse_args()

    root = data_root(args.root)
    samples = select(load_samples(args.split, root), args.ids, args.split_file, args.limit)
    turns = {p: ref for s in samples for p, ref in user_turns(s)}
    paths = sorted(turns)
    t0 = time.time()
    result = transcribe(paths, root / args.split, args.model, args.batch_size, args.no_cudnn)
    elapsed = time.time() - t0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"{len(paths)} clips in {elapsed:.1f}s ({elapsed / max(len(paths), 1):.3f}s/clip) -> {args.out}")

    refs = [(p, r) for p, r in turns.items() if r]
    if refs:
        errs = words = nf = nt = 0
        for p, r in refs:
            h, rw = norm_words(result[p]), norm_words(r)
            errs += edit_distance(h, rw)
            words += len(rw)
            f, t = number_recall(h, rw)
            nf, nt = nf + f, nt + t
        print(f"WER {errs / max(words, 1):.4f} on {len(refs)} clips; number/time word recall {nf / max(nt, 1):.4f} ({nt} words)")


if __name__ == "__main__":
    main()
