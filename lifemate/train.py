"""LoRA SFT of the agent LLM on lifemate.sft data (GPU: Colab / Kaggle).

    python -m lifemate.train --data sft/train.jsonl --model Qwen/Qwen3-4B-Instruct-2507 --out ckpt/qwen3_4b_lora
    python -m lifemate.train --data sft/train.jsonl --check      # tokenization / masking / lengths only (CPU)

Loss on assistant turns only: each assistant message's token span is found from the token length of the chat
template rendered up to that message (with and without the generation prompt), which requires a template whose
renderings are prefixes of each other (true for Qwen3-Instruct-2507, verified by --check).
After training, serve with vLLM: --merge writes merged full weights to <out>/merged (what the Docker image ships).
"""
from __future__ import annotations

import argparse
import json
import random


def encode(tok, messages: list[dict], max_len: int) -> dict | None:
    """input_ids + labels (-100 outside assistant turns); None if longer than max_len."""
    ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if hasattr(ids, "input_ids") or isinstance(ids, dict):  # transformers >= 5 returns a BatchEncoding
        ids = ids["input_ids"]
    if len(ids) > max_len:
        return None
    labels = [-100] * len(ids)
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        start = _len(tok, messages[:i], gen=True)
        end = _len(tok, messages[:i + 1], gen=False)
        labels[start:end] = ids[start:end]
    return {"input_ids": ids, "labels": labels}


def _len(tok, messages, gen: bool) -> int:
    ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=gen)
    if hasattr(ids, "input_ids") or isinstance(ids, dict):
        ids = ids["input_ids"]
    return len(ids)


def load_rows(paths: str) -> list[dict]:
    """Comma-separated jsonl files; `file:0.3` keeps a fixed random 30% of that file (replay of older data)."""
    rows = []
    for spec in paths.split(","):
        path, _, frac = spec.rpartition(":")
        if not path or not frac.replace(".", "", 1).isdigit():  # no ratio (a colon may belong to a drive letter)
            path, frac = spec, ""
        with open(path, encoding="utf-8") as f:
            part = [json.loads(line) for line in f if line.strip()]
        if frac:
            part = random.Random(0).sample(part, round(len(part) * float(frac)))
        rows += part
    from .prompt import refresh_system

    for r in rows:  # data written by an older prompt version trains the current one
        if r["messages"] and r["messages"][0]["role"] == "system":
            r["messages"][0]["content"] = refresh_system(r["messages"][0]["content"])
    return rows


def check(tok, rows, max_len: int):
    lens, dropped = [], 0
    for r in rows:
        e = encode(tok, r["messages"], max_len)
        if e is None:
            dropped += 1
            continue
        lens.append(len(e["input_ids"]))
    lens.sort()
    print(f"{len(rows)} examples, dropped >{max_len}: {dropped}; tokens median {lens[len(lens) // 2]}, "
          f"p95 {lens[int(.95 * len(lens))]}, max {lens[-1]}, total {sum(lens) / 1e6:.2f}M")
    e = encode(tok, rows[0]["messages"], 10 ** 9)
    spans, cur = [], None
    for i, l in enumerate(e["labels"]):
        if l != -100 and cur is None:
            cur = i
        if l == -100 and cur is not None:
            spans.append((cur, i))
            cur = None
    if cur is not None:
        spans.append((cur, len(e["labels"])))
    print("trained spans of example 0:")
    for a, b in spans:
        print("  ", repr(tok.decode(e["input_ids"][a:b])))
    n_asst = sum(m["role"] == "assistant" for m in rows[0]["messages"])
    assert len(spans) == n_asst, f"{len(spans)} spans for {n_asst} assistant turns: template is not prefix-consistent"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="jsonl file(s), comma-separated; file:0.3 keeps 30%% of a file")
    ap.add_argument("--init-adapter", default=None, help="LoRA adapter dir to continue training from")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--out", default="ckpt/qwen3_4b_lora")
    ap.add_argument("--max-len", type=int, default=12288)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--qlora", action="store_true", help="4-bit base (T4 / P100); default bf16/fp16 base")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--max-steps", type=int, default=-1, help="smoke test: stop after N optimizer steps")
    ap.add_argument("--longest-first", action="store_true", help="smoke test: put the longest examples first (peak memory)")
    ap.add_argument("--merge", action="store_true", help="also write merged weights to <out>/merged")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    rows = load_rows(args.data)
    if args.check:
        return check(tok, rows, args.max_len)

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, set_seed

    attn_impl = "sdpa"
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
        # On a T4 (sm75) only SDPA's memory-efficient kernel avoids materialising the L x L scores, and the way
        # transformers v5 calls SDPA (enable_gqa / 4-D masks) disqualifies it: the math kernel then needed 3.2 GB for
        # one 5.2K-token layer and OOMed. Plain q/k/v with is_causal=True runs the efficient kernel (0.33 GB at 4K,
        # measured on Kaggle's T4, torch 2.11), so training uses exactly that call. Valid only without padding:
        # batches are a single example. The math kernel is disabled so a regression raises instead of OOMing.
        from transformers import AttentionInterface

        def causal_sdpa(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kw):
            # q/k leave q_norm/k_norm + RoPE in fp32 while v comes from v_proj in fp16: SDPA kernels need one dtype
            out_dtype = query.dtype
            if not getattr(causal_sdpa, "logged", False):
                causal_sdpa.logged = True
                print(f"  causal_sdpa in: q {tuple(query.shape)} {query.dtype} k {key.dtype} v {value.dtype}", flush=True)
            query, key, value = (t.to(dtype).contiguous() for t in (query, key, value))
            n_rep = module.num_key_value_groups
            if n_rep > 1:
                b, h, l, d = key.shape
                key = key[:, :, None].expand(b, h, n_rep, l, d).reshape(b, h * n_rep, l, d)
                value = value[:, :, None].expand(b, h, n_rep, l, d).reshape(b, h * n_rep, l, d)
            out = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, dropout_p=dropout if module.training else 0.0, scale=scaling, is_causal=query.shape[2] > 1)
            return out.to(out_dtype).transpose(1, 2).contiguous(), None

        AttentionInterface.register("causal_sdpa", causal_sdpa)
        attn_impl = "causal_sdpa"
        torch.backends.cuda.enable_math_sdp(False)

    set_seed(0)
    random.seed(0)
    data = [e for r in rows if (e := encode(tok, r["messages"], args.max_len)) is not None]
    print(f"{len(data)}/{len(rows)} examples within {args.max_len} tokens")
    if args.longest_first:
        data.sort(key=lambda e: -len(e["input_ids"]))
    # is_bf16_supported() is True on a T4 too (emulated), but SDPA's memory-efficient kernel needs native bf16 (sm80+):
    # on sm75 bf16 silently sent attention to the math kernel (OOM), so a T4 trains in fp16
    bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    kw = {"dtype": dtype}
    if args.qlora:
        from transformers import BitsAndBytesConfig

        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map={"": 0}, attn_implementation=attn_impl, **kw)
    model.config.use_cache = False
    # no prepare_model_for_kbit_training: it upcasts every non-quantized weight (the 389M-param embedding too) to fp32,
    # which on a T4 pushed attention off the memory-efficient SDPA kernel and ran out of memory in backward
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if args.init_adapter:  # continue training an earlier adapter (new data + replay) instead of starting from zero
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    def collate(batch):
        assert len(batch) == 1 or attn_impl == "sdpa", "causal_sdpa ignores padding masks: keep per_device batch size 1"
        n = max(len(b["input_ids"]) for b in batch)
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        return {"input_ids": torch.tensor([b["input_ids"] + [pad] * (n - len(b["input_ids"])) for b in batch]),
                "labels": torch.tensor([b["labels"] + [-100] * (n - len(b["labels"])) for b in batch]),
                "attention_mask": torch.tensor([[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch])}

    import inspect
    import math

    import torch.nn.functional as F

    class MaskedLMTrainer(Trainer):
        """Loss from lm_head applied to the labelled positions only. The stock loss materialises fp32 logits for the
        whole sequence (5.7K tokens x 152K vocab = 3.4 GB: OOM on a 16 GB T4) although ~2% of tokens carry labels."""

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            lm = getattr(model, "module", model).get_base_model()  # CausalLM inside PeftModel; LoRA layers live in its modules
            mask = inputs["attention_mask"]
            # without padding the mask is dropped so SDPA takes the causal memory-efficient path (no L x L matrix)
            # explicit autocast: calling the inner model skips accelerate's autocast wrapper around model.forward, and
            # fp32 LoRA outputs would otherwise turn q/k/v into fp32
            with torch.autocast("cuda", dtype=dtype):
                hidden = lm.model(input_ids=inputs["input_ids"], attention_mask=None if bool(mask.all()) else mask).last_hidden_state
                tgt = labels[:, 1:]
                sel = tgt != -100
                logits = lm.lm_head(hidden[:, :-1][sel]).float()
            self._n_steps = getattr(self, "_n_steps", 0) + 1
            if self._n_steps <= 4 or self._n_steps % 200 == 0:
                print(f"  micro-step {self._n_steps}: {inputs['input_ids'].shape[1]} tokens, "
                      f"peak {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GB", flush=True)
            loss = F.cross_entropy(logits, tgt[sel], reduction="sum")
            # Trainer passes the label count of the whole accumulated batch when the model takes loss kwargs
            denom = num_items_in_batch if num_items_in_batch is not None else sel.sum()
            loss = loss / denom.to(loss.device) if torch.is_tensor(denom) else loss / denom
            return (loss, None) if return_outputs else loss

    total_steps = math.ceil(len(data) / args.grad_accum * args.epochs)
    targ_kw = dict(output_dir=args.out, per_device_train_batch_size=1, gradient_accumulation_steps=args.grad_accum,
                   num_train_epochs=args.epochs, learning_rate=args.lr, lr_scheduler_type="cosine",
                   warmup_steps=max(1, round(0.05 * total_steps)), logging_steps=5, save_strategy="epoch", save_total_limit=1,
                   bf16=bf16, fp16=not bf16, gradient_checkpointing=True,
                   gradient_checkpointing_kwargs={"use_reentrant": False}, report_to=[], seed=0, data_seed=0,
                   remove_unused_columns=False, dataloader_num_workers=0, max_steps=args.max_steps)
    if args.longest_first:  # keep the given order (Trainer shuffles by default)
        targ_kw["train_sampling_strategy"] = "sequential"
    # TrainingArguments drops options between major versions (v5 removed warmup_ratio): pass only the ones it knows
    known = set(inspect.signature(TrainingArguments.__init__).parameters)
    skipped = sorted(set(targ_kw) - known)
    if skipped:
        print("TrainingArguments: unsupported, skipped:", skipped)
    targs = TrainingArguments(**{k: v for k, v in targ_kw.items() if k in known})
    MaskedLMTrainer(model=model, args=targs, train_dataset=data, data_collator=collate).train()
    model.save_pretrained(f"{args.out}/adapter")
    tok.save_pretrained(f"{args.out}/adapter")
    print("adapter ->", f"{args.out}/adapter")
    if args.merge:
        del model
        torch.cuda.empty_cache()
        from peft import PeftModel

        base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
        merged = PeftModel.from_pretrained(base, f"{args.out}/adapter").merge_and_unload()
        merged.save_pretrained(f"{args.out}/merged", safe_serialization=True)
        tok.save_pretrained(f"{args.out}/merged")
        print("merged ->", f"{args.out}/merged")


if __name__ == "__main__":
    main()
