"""Fine-tune a Whisper-family ASR (PhoWhisper) on the competition audio: every user turn of the chosen samples.

    python -m lifemate.asr_train --ids train val_unseen --model vinai/PhoWhisper-medium --out ckpt/phowhisper_medium_ft --no-cudnn
    python -m lifemate.asr --split train --ids val_seen --model ckpt/phowhisper_medium_ft --no-cudnn --out asr/val_seen_ft.json

ASR has no notion of apps, so the LLM's val_unseen samples may train it; val_seen stays out for the WER check.
Targets are the transcripts as written in train (digits, brand names like VTV3 / RON95): the LLM is trained on the
same spelling. Adafactor + fp16 keeps a full fine-tune of the 0.77B medium model inside a 16 GB T4.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .asr import user_turns
from .data import data_root, load_samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--ids", nargs="+", default=["train", "val_unseen"], help="keys of --split-file")
    ap.add_argument("--split-file", default="splits/train_split.json")
    ap.add_argument("--model", default="vinai/PhoWhisper-medium")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--no-cudnn", action="store_true")
    args = ap.parse_args()

    import soundfile as sf
    import torch
    from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration, WhisperProcessor,
                              set_seed)

    torch.backends.cudnn.enabled = not args.no_cudnn
    set_seed(0)
    root = data_root(args.root)
    keep = set()
    split = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
    for k in args.ids:
        keep |= set(split[k])
    clips = [(p, t) for s in load_samples("train", root) if s["id"] in keep for p, t in user_turns(s)
             if t and (root / "train" / p).is_file()]
    random.Random(0).shuffle(clips)
    print(f"{len(clips)} clips for training")

    proc = WhisperProcessor.from_pretrained(args.model, language="vi", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.generation_config.language = "vi"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False

    class Clips(torch.utils.data.Dataset):
        def __len__(self):
            return len(clips)

        def __getitem__(self, i):
            path, text = clips[i]
            audio, sr = sf.read(str(root / "train" / path), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            feats = proc.feature_extractor(audio, sampling_rate=sr).input_features[0]
            return {"input_features": feats, "labels": proc.tokenizer(text).input_ids}

    def collate(batch):
        feats = proc.feature_extractor.pad([{"input_features": b["input_features"]} for b in batch], return_tensors="pt")
        lab = proc.tokenizer.pad([{"input_ids": b["labels"]} for b in batch], return_tensors="pt")
        labels = lab["input_ids"].masked_fill(lab["attention_mask"].ne(1), -100)
        if (labels[:, 0] == model.config.decoder_start_token_id).all():  # the model prepends it itself
            labels = labels[:, 1:]
        return {"input_features": feats["input_features"], "labels": labels}

    targs = Seq2SeqTrainingArguments(output_dir=args.out, per_device_train_batch_size=args.batch_size,
                                     gradient_accumulation_steps=args.grad_accum, num_train_epochs=args.epochs,
                                     learning_rate=args.lr, warmup_steps=50, lr_scheduler_type="linear", optim="adafactor",
                                     fp16=torch.cuda.is_available(), gradient_checkpointing=True, logging_steps=20,
                                     save_strategy="no", report_to=[], seed=0, remove_unused_columns=False,
                                     dataloader_num_workers=2)
    Seq2SeqTrainer(model=model, args=targs, train_dataset=Clips(), data_collator=collate).train()
    model.config.use_cache = True
    model.to(torch.float16).save_pretrained(args.out)  # inference runs in fp16; halves the file
    proc.save_pretrained(args.out)
    print("saved ->", args.out)


if __name__ == "__main__":
    main()
