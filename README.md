# LifeMate (Zalo AI Challenge 2026)

Vietnamese audio → transcript → LLM agent calling tools on `zaic_tools` → `submission.jsonl`.

**This project is focused on Vietnamese.** Users speak Vietnamese requests; the ASR (PhoWhisper, fine-tuned on Vietnamese speech) and the LLM work in Vietnamese end to end. The repo's docs, comments and commit messages are in English, but the functional strings stay Vietnamese on purpose and must not be translated:
- the system prompt, context labels and agent feedback in `lifemate/prompt.py` and `lifemate/agent.py` (the LoRA was trained on exactly these strings, so changing them changes the model's behaviour);
- the Vietnamese number words in `lifemate/asr.py` (used to measure number/time recall);
- the Vietnamese inputs in `tests/test_basic.py` (they test fuzzy matching on Vietnamese text).

## Layout

| File | Role |
|---|---|
| `lifemate/data.py` | Data paths (`LIFEMATE_DATA`), jsonl/catalog readers, imports each split's own `zaic_tools` |
| `lifemate/scorer.py` | Local scorer: hash gate + F1 (fuzzy token-F1 > 0.6, 1-1 matching, order ignored) |
| `lifemate/split.py` | Leave-apps-out split of train → `splits/train_split.json` (train / val_seen / val_unseen) |
| `lifemate/prompt.py` | Renders compact tool schemas, context (time, weekday, tomorrow, profile) and history |
| `lifemate/agent.py` | Agent loop, one tool call per turn; `env.validate()` before `env.call()`, cached repeat reads, stops after `ask_missing_info` |
| `lifemate/backends.py` | `oracle` (replays gold), `empty`, `openai` (OpenAI-compatible server), `vllm` (in-process, batched per step) |
| `lifemate/run.py` | Runs a split → `runs/<...>/submission.jsonl`, `traces.jsonl`, plus `scores.jsonl` + `summary.json` on train |
| `lifemate/analyze.py` | Error analysis of a train run (missing/extra calls, wrong arguments, misplaced asks...) |
| `lifemate/asr.py` / `asr_train.py` | Whisper ASR (PhoWhisper) + WER; fine-tuning on the competition audio |
| `lifemate/sft.py` / `train.py` | SFT data from gold call sequences; LoRA/QLoRA training (loss on assistant turns only) |
| `lifemate/export.py` | Merges the LoRA into the base model + ASR → `models/` (checks the parameter budget); shards the LLM for the image |
| `lifemate/infer.py` | **Final entrypoint**: split folder → ASR → agent → `result/submission.jsonl` |
| `Dockerfile` | Offline image for the private round (models baked into the image) |
| `pack.py` | Packs code + data (no audio) / audio subsets for upload |
| `kaggle/` | Kaggle kernels: `train_kernel` (LoRA), `asr_kernel` (ASR fine-tune), `final_kernel` (export + evaluation + public_test), `docker_parity` / `shard_check` (GPU checks of the image setup); `orchestrate.sh` chains them and downloads the results |
| `notebooks/baseline_colab.ipynb` | Runs Qwen3-4B with vLLM on a GPU |

The organiser data is not in this repo (it may not be redistributed); point `LIFEMATE_DATA` at your own copy.

## Kaggle resources

All GPU work runs on Kaggle under the account `manhhodinh`. The code of every kernel is in `kaggle/` here; the Kaggle pages hold the runs, logs and outputs.

**Access:** these datasets and kernels are **private and must stay private**: they contain the organiser data (audio, `state.json`, labels) or outputs derived from it (transcripts, submissions), which may only be used by registered teams and not redistributed. Teammates get access by being added as collaborators: on each page, **Share → Add collaborators** → their Kaggle username (Viewer to read, Editor to run new versions). Never switch them to Public.

| Kind | Link | What it holds / does |
|---|---|---|
| Dataset | [lifemate-sft](https://www.kaggle.com/datasets/manhhodinh/lifemate-sft) | `lifemate_code` (this package), `sft_train.jsonl`, organiser data without audio |
| Dataset | [lifemate-audio](https://www.kaggle.com/datasets/manhhodinh/lifemate-audio) | train audio (ASR fine-tune) |
| Dataset | [lifemate-public-audio](https://www.kaggle.com/datasets/manhhodinh/lifemate-public-audio) | public_test audio |
| Kernel | [lifemate-train-qlora](https://www.kaggle.com/code/manhhodinh/lifemate-train-qlora) | LoRA training of Qwen3-4B (`kaggle/train_kernel`) → adapter |
| Kernel | [lifemate-asr-finetune](https://www.kaggle.com/code/manhhodinh/lifemate-asr-finetune) | PhoWhisper-medium fine-tune (`kaggle/asr_kernel`) |
| Kernel | [lifemate-final](https://www.kaggle.com/code/manhhodinh/lifemate-final) | merge + export `models/`, val scores, public_test submission (`kaggle/final_kernel`) |
| Kernel | [lifemate-docker-parity](https://www.kaggle.com/code/manhhodinh/lifemate-docker-parity) | GPU run of the Docker image setup, run twice for determinism (`kaggle/docker_parity`) |
| Kernel | [lifemate-shard-check](https://www.kaggle.com/code/manhhodinh/lifemate-shard-check) | GPU check of the 8-shard LLM layout (`kaggle/shard_check`) |
| Kernel | [lifemate-sdpa-diag](https://www.kaggle.com/code/manhhodinh/lifemate-sdpa-diag) | T4 attention-kernel diagnostics (`kaggle/sdpa_diag`) |

To run a kernel from your own account: copy it on Kaggle ("Copy & Edit") once you have access, or push the folder with the Kaggle CLI after changing `id` in its `kernel-metadata.json` to your username (the dataset sources must be shared with you too).

## Final pipeline

1. `python -m lifemate.sft --ids train --out sft/train.jsonl` → private Kaggle dataset `manhhodinh/lifemate-sft` (together with the code and the audio-free data).
2. `bash kaggle/orchestrate.sh` → LoRA training (`lifemate-train-qlora`) → ASR fine-tune (`lifemate-asr-finetune`) → `lifemate-final` (exports `models/`, scores val_seen / val_unseen / val_seen through ASR, runs `lifemate.infer` on public_test + validates) → downloads `runs/kaggle_final/`.
3. `powershell -ExecutionPolicy Bypass -File build_docker.ps1` (stages in `C:\lifemate_build` by default, needs ~25 GB): copies the code, downloads `models/` (~9.6 GB) from the `lifemate-final` output, shards the LLM, `docker build -t lifemate:final`, then a CPU smoke test (loads the ASR processor and LLM tokenizer with the image's transformers). `-Save` also writes `lifemate_final.tar`. `VLLM_IMAGE` is pinned to `vllm/vllm-openai:v0.32.0-cu129`, the vLLM version validated in the kernels.
4. Run it as the organisers do: `docker run --gpus all --network none -v <private_test>:/data -v $PWD/result:/app/result lifemate:final`
5. Public image on Docker Hub: `docker pull dinhmanh11/lifemate:final`. Digest `sha256:252ea79ed8bb18e656c14d781351f2925ffecd47f8e7a5805cfe0c13a56cb457` (2026-11-03, both determinism fixes and the 8-shard LLM), ~19.9 GB compressed.

Budget (checked at export): Qwen3-4B (~4.0B) + PhoWhisper-medium (~0.77B) < 6B; LLM < 5B; `max_model_len` 16384. Greedy decoding, fixed seed; exactly one `ToolEnv` per sample, `state.json` is never read.

## Commands

```bash
python -X utf8 -m lifemate.split                                              # build the split
python -X utf8 -m lifemate.run --split train --ids val_seen --backend oracle  # must print SCORE 1.0000
python -X utf8 tests/test_basic.py
python -X utf8 -m lifemate.run --split train --ids val_unseen --backend vllm --model Qwen/Qwen3-4B-Instruct-2507
python -X utf8 -m lifemate.run --split train --ids val_seen --backend openai --base-url http://<host>:8000/v1 --model <name>
```

Public test has no transcripts: pass `--transcripts asr.json` (`{"audio/<id>/<id>.wav": "text", ...}`, including the `_u1`, `_u2`... history turns).

## Reference points (train, 1545 samples)

| Backend | Score | Notes |
|---|---|---|
| oracle | 1.0000 | checks the agent loop + scorer (0.9999 after adding the made-up-ID guard) |
| empty | 0.0718 | only the out-of-scope samples are right (7.2%) |

## Colab T4 results (2026-11-01)

| Experiment | val_seen (200) | val_unseen (316) |
|---|---|---|
| Qwen3-4B-Instruct-2507 zero-shot, gold transcript | 0.239 (F1 0.345, hash 0.44) | 0.212 (F1 0.318, hash 0.36) |
| + made-up-ID guard (x-ref) | 0.238 (F1 0.352, hash 0.445) | |
| + ID guard, PhoWhisper-medium ASR transcript | **0.092** (F1 0.165, hash 0.255) | |

- Main errors: wrong/extra optional arguments on read tools (`list_bills(type=…)`, `get_status(room=…)`), missing read before write, misplaced asks for missing info (spurious/missed ask), and **made-up IDs** (52 samples, now blocked by the x-ref rule in `agent.py`).
- ASR before fine-tuning, 645 val_seen clips (T4): PhoWhisper-medium WER 0.304 / number-time recall 0.69 / 0.33 s/clip; PhoWhisper-large 0.284 / 0.70 / 0.49 s; whisper-large-v3-turbo 0.365 / 0.35 / 0.34 s. Worst on English words/names (VTV3, RON95) and numbers → fine-tune (`lifemate/asr_train.py`).
- LoRA training on T4 (sm75): SDPA's memory-efficient kernel only takes **fp16** q/k/v of one dtype, without GQA (`enable_gqa`). Two traps hit: (1) transformers v5 calls SDPA with `enable_gqa=True` → math kernel → OOM at 6.4K tokens (13.3 GB, measured on Colab); (2) torch 2.11 reports `is_bf16_supported()=True` on T4, so the model loads in bf16 → no efficient kernel accepts it. `train.py` uses its own `causal_sdpa` attention (repeat_kv + is_causal) and fp16 on GPUs below sm80; fp16 + repeat_kv peaks at 5.1 GB at 6.4K tokens (4-bit).
- Speed: running all 200 agents at once on a T4 overflows the KV cache (13–15 min/step); `--wave 6` gives ~7–9 s/sample.
- Colab: after `pip install vllm`, ASR needs `--no-cudnn` (CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH).

## Kaggle T4 results (2026-11-01/02)

| Step | Result |
|---|---|
| LoRA r32 on Qwen3-4B-Instruct-2507 (fp16, not 4-bit), 1065 trajectories × 2 epochs | 5.7 h; loss 0.65 → 0.03 (train_loss 0.086); peak 11.6 GB |
| PhoWhisper-medium fine-tune (3 epochs, train+val_unseen audio) | val_seen 645 clips: **WER 0.304 → 0.104**, number/time recall 0.69 → **0.888**, 0.69 s/clip |

Final pipeline (`lifemate-final`: vLLM 0.32.0, torch 2.13+cu130, transformers 5.16.1, T4; LLM with the LoRA merged, `--wave 8`):

| Set | Score | F1 | Hash | Zero-shot before |
|---|---|---|---|---|
| val_seen (200), gold transcript | **0.658** | 0.737 | 0.765 | 0.239 |
| val_unseen (316, apps unseen in training), gold transcript | **0.456** | 0.587 | 0.557 | 0.212 |
| val_seen, fine-tuned ASR transcript | **0.496** | 0.639 | 0.635 | 0.092 |

- val_seen by category: single 0.763, multi 0.651, out_of_scope 0.727, ask_missing 0.548; on val_unseen ask_missing is weakest (0.246).
- public_test (634 samples) through the Docker entrypoint `lifemate.infer` (ASR → agent): 634 lines, 0 schema errors from the organisers' validator; 9.7 s/sample on T4 including ASR (`runs/kaggle_final/result/submission.jsonl`).
- Budget: LLM 4.02B + ASR 0.76B = 4.79B parameters.
- Kaggle after `pip install vllm`: uninstall `torchaudio` (cu128 build, mismatched with torch cu130 → Whisper import fails; audio is decoded with soundfile) and `torchao` 0.10 (breaks peft).
- transformers v5 saves the Whisper processor as `processor_config.json` (feature extractor nested inside). v4 (e.g. in the vLLM image) needs `preprocessor_config.json`, so `export.legacy_preprocessor` writes it too.
- Docker (2026-11-02, `build_docker.ps1 -SkipDownload`): `lifemate:final` 36.4 GB. CPU smoke test with `--network none` passes: vllm 0.32.0, torch 2.13.0+cu129, transformers 5.19.0; the image loads the ASR processor and LLM tokenizer. No end-to-end GPU run inside the image, since the dev machine has no NVIDIA GPU.
- GPU check (`kaggle/docker_parity`, T4): same vllm 0.32.0, transformers 5.19.0, same `models/`, environment variables and `ENTRYPOINT` arguments as the image, on public_test. Result: exit 0, 634 lines, 0 schema errors, 11.4 s/sample including model loading. ASR transcripts identical on 1933/1933 clips; 629/634 lines identical to `lifemate-final`. The other 5 differ in the LLM's call sequence (2 in hash), because vLLM greedy decoding is not bit-exact when batch order changes.
- Two changes make runs reproducible (`backends.VLLMBackend`):
  - `VLLM_ENABLE_V1_MULTIPROCESSING=0`: the engine runs in the same process, so each batch is scheduled whole.
  - KV cache sized from total GPU memory: `0.85·total − 9.25 GiB`, ~3 GiB on T4, ~10.5 GiB on A30. Before, it depended on the free memory: 23,984 tokens with ASR in the process, 24,976 without, and 1/634 samples differed.
  - Rechecked on Kaggle T4 (`runs/kaggle_determinism3`): entrypoint run twice, KV cache 21,840 tokens both times, 634/634 lines identical, 0 schema errors, 11.7 s/sample.
- Image: the LLM weights are split into 8 shards of ~1 GB (`export.shard_safetensors`, 398 tensors checked byte for byte), one layer each, so a push over a flaky connection re-sends ~1 GB instead of 9.6 GB.
  - GPU check (`kaggle/shard_check`, T4, `runs/kaggle_shard_check`): vLLM loads all 8/8 shards through `model.safetensors.index.json`; the entrypoint on the first 160 public_test samples gives 160/160 lines identical to both runs of the determinism check.

## Scorer assumptions

The organisers' formula is only given as an image. The local scorer assumes: sample score = F1 if the hash matches (else 0), with P = M/(M+U) and R = M/|G|; final score = mean. Arguments are compared over the union of keys, so one extra non-fuzzy argument makes the call not match. Scalar lists are compared as multisets.
