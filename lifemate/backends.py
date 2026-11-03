"""LLM backends. Interface: generate(agents) -> one output string (or None) per agent, from agent.messages.

- oracle: replays the gold calls of a train sample (end-to-end test of agent loop + scorer, score must be 1.0)
- empty:  always answers done (lower bound)
- openai: any OpenAI-compatible server (vLLM `vllm serve`, llama.cpp server, ...), requests in parallel threads
- vllm:   in-process vLLM, the whole step batched in one LLM.chat call (Colab / Kaggle / final Docker)
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor

DONE = '{"name": "done"}'


class OracleBackend:
    def generate(self, agents):
        outs = []
        for a in agents:
            gold = a.sample["label"]["tool_calls"]
            i = a.n_executed
            outs.append(json.dumps(gold[i], ensure_ascii=False) if i < len(gold) else DONE)
        return outs


class EmptyBackend:
    def generate(self, agents):
        return [DONE] * len(agents)


class OpenAIBackend:
    def __init__(self, model: str, base_url: str, api_key: str = "EMPTY", max_tokens: int = 512, workers: int = 16,
                 extra_body: dict | None = None):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model, self.max_tokens, self.workers = model, max_tokens, workers
        self.extra_body = extra_body or {}

    def _one(self, messages):
        try:
            r = self.client.chat.completions.create(model=self.model, messages=messages, temperature=0, seed=0,
                                                    max_tokens=self.max_tokens, extra_body=self.extra_body)
            return r.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001 - a failed request ends that sample, not the run
            print(f"  request failed: {type(e).__name__}: {str(e)[:200]}")
            return None

    def generate(self, agents):
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(self._one, [a.messages for a in agents]))


class VLLMBackend:
    def __init__(self, model: str, max_model_len: int = 16384, max_tokens: int = 512, gpu_memory_utilization: float = 0.85,
                 dtype: str = "auto", enable_thinking: bool = False, seed: int = 0, lora: str | None = None):
        # engine core in this process: every llm.chat() batch is scheduled whole, not as requests reach a separate engine
        # process, so batch composition (and the fp results that depend on it) is the same on every run (vLLM reproducibility docs)
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        from vllm import LLM, SamplingParams

        extra = {"enable_lora": True, "max_lora_rank": 64} if lora else {}
        import torch

        if torch.cuda.is_available():
            # KV cache sized from the card's total memory, not from what is free at startup: the free memory depends on what
            # ran before in the process (the ASR leaves ~0.1 GiB), and a different block count changes preemption and with it
            # the outputs (seen on Kaggle: 23,984 vs 24,976 KV tokens -> 1/634 samples differed). 9.25 GiB covers the fp16
            # 4B weights, activations and CUDA graphs; rounded down to 256 MiB.
            total = torch.cuda.get_device_properties(0).total_memory
            extra["kv_cache_memory_bytes"] = int((total * gpu_memory_utilization - 9.25 * 2**30) // 2**28) * 2**28
        self.llm = LLM(model=model, max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
                       dtype=dtype, seed=seed, enable_prefix_caching=True, **extra)
        self.lora = None
        if lora:
            from vllm.lora.request import LoRARequest

            self.lora = LoRARequest("sft", 1, lora)
        self.tok = self.llm.get_tokenizer()
        self.params = SamplingParams(temperature=0, max_tokens=max_tokens, seed=seed)
        self.max_prompt = max_model_len - max_tokens
        self.template_kwargs = {"enable_thinking": enable_thinking}

    def n_tokens(self, messages) -> int:
        return len(self.tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **self.template_kwargs))

    def generate(self, agents):
        outs: list[str | None] = [None] * len(agents)
        ok = [i for i, a in enumerate(agents) if self.n_tokens(a.messages) <= self.max_prompt]
        if ok:
            res = self.llm.chat([agents[i].messages for i in ok], self.params, use_tqdm=False,
                                chat_template_kwargs=self.template_kwargs, lora_request=self.lora)
            for i, r in zip(ok, res):
                outs[i] = r.outputs[0].text
        return outs


def make_backend(name: str, **kw):
    if name == "oracle":
        return OracleBackend()
    if name == "empty":
        return EmptyBackend()
    if name == "openai":
        return OpenAIBackend(kw["model"], kw["base_url"], kw.get("api_key") or "EMPTY",
                             extra_body={"chat_template_kwargs": {"enable_thinking": False}} if kw.get("no_think") else None)
    if name == "vllm":
        return VLLMBackend(kw["model"], max_model_len=kw.get("max_model_len", 16384), gpu_memory_utilization=kw.get("gpu_mem", 0.85),
                           lora=kw.get("lora"))
    raise ValueError(f"unknown backend {name!r}")
