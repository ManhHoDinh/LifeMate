# LifeMate baseline (Zalo AI Challenge 2026)

Audio tiếng Việt → transcript → LLM agent gọi tool trên `zaic_tools` → `submission.jsonl`.

## Cấu trúc

| File | Vai trò |
|---|---|
| `lifemate/data.py` | Đường dẫn data (`LIFEMATE_DATA`), đọc jsonl/catalog, import `zaic_tools` của từng split |
| `lifemate/scorer.py` | Bộ chấm local: hash gate + F1 (fuzzy token-F1 > 0.6, matching 1-1, không xét thứ tự) |
| `lifemate/split.py` | Chia train leave-apps-out → `splits/train_split.json` (train / val_seen / val_unseen) |
| `lifemate/prompt.py` | Render schema rút gọn, ngữ cảnh (giờ, thứ, ngày mai, profile), history |
| `lifemate/agent.py` | Vòng lặp 1 lượt = 1 tool call; `env.validate()` trước `env.call()`, cache tool đọc trùng, dừng sau `ask_missing_info` |
| `lifemate/backends.py` | `oracle` (phát lại gold), `empty`, `openai` (server OpenAI-compatible), `vllm` (in-process, batch theo bước) |
| `lifemate/run.py` | Chạy một split → `runs/<...>/submission.jsonl`, `traces.jsonl`, và `scores.jsonl` + `summary.json` khi split là train |
| `lifemate/analyze.py` | Phân tích lỗi một run train (thiếu/thừa call, sai tham số, hỏi sai chỗ…) |
| `lifemate/asr.py` / `asr_train.py` | ASR Whisper (PhoWhisper) + đo WER; fine-tune trên audio của cuộc thi |
| `lifemate/sft.py` / `train.py` | Dữ liệu SFT từ chuỗi call gold; train LoRA/QLoRA (loss chỉ trên lượt assistant) |
| `lifemate/export.py` | Gộp LoRA vào model gốc + ASR → `models/` (kiểm ngân sách tham số) |
| `lifemate/infer.py` | **Entrypoint cuối**: thư mục split → ASR → agent → `result/submission.jsonl` |
| `Dockerfile` | Image offline cho vòng private (models nằm sẵn trong image) |
| `pack.py` | Đóng gói code + data (không audio) / audio theo tập con để upload |
| `kaggle/` | Kernel Kaggle: `train_kernel` (LoRA), `asr_kernel` (fine-tune ASR), `final_kernel` (export + đánh giá + public_test); `orchestrate.sh` chạy nối tiếp và tải kết quả |
| `notebooks/baseline_colab.ipynb` | Chạy Qwen3-4B bằng vLLM trên GPU |

## Pipeline cuối

1. `python -m lifemate.sft --ids train --out sft/train.jsonl` → dataset private `manhhodinh/lifemate-sft` (cùng code + data không audio).
2. `bash kaggle/orchestrate.sh` → train LoRA (`lifemate-train-qlora`) → fine-tune ASR (`lifemate-asr-finetune`) → `lifemate-final` (xuất `models/`, chấm val_seen / val_unseen / val_seen-qua-ASR, chạy `lifemate.infer` trên public_test + validate) → tải `runs/kaggle_final/`.
3. `powershell -ExecutionPolicy Bypass -File build_docker.ps1` (mặc định dựng ở `C:\lifemate_build`, cần ~25 GB): chép code, tải `models/` (~9.6 GB) từ output của `lifemate-final`, `docker build -t lifemate:final`, rồi smoke test trên CPU (nạp processor ASR + tokenizer LLM bằng transformers trong image). `-Save` xuất thêm `lifemate_final.tar`. `VLLM_IMAGE` đã ghim `vllm/vllm-openai:v0.32.0-cu129`, cùng bản vLLM đã chạy trong kernel.
4. Chạy như BTC: `docker run --gpus all --network none -v <private_test>:/data -v $PWD/result:/app/result lifemate:final`
5. Image public trên Docker Hub: `docker pull dinhmanh11/lifemate:final`. Digest `sha256:252ea79ed8bb18e656c14d781351f2925ffecd47f8e7a5805cfe0c13a56cb457` (2026-11-03, có hai sửa determinism và LLM chia 8 shard), khoảng 19.9 GB nén.

Ngân sách (export kiểm tra): Qwen3-4B (~4.0B) + PhoWhisper-medium (~0.77B) < 6B; LLM < 5B; `max_model_len` 16384. Giải mã greedy, seed cố định; mỗi mẫu đúng một `ToolEnv`, không đọc `state.json`.

## Lệnh

```bash
python -X utf8 -m lifemate.split                                              # tạo split
python -X utf8 -m lifemate.run --split train --ids val_seen --backend oracle  # phải ra SCORE 1.0000
python -X utf8 tests/test_basic.py
python -X utf8 -m lifemate.run --split train --ids val_unseen --backend vllm --model Qwen/Qwen3-4B-Instruct-2507
python -X utf8 -m lifemate.run --split train --ids val_seen --backend openai --base-url http://<host>:8000/v1 --model <name>
```

Public test chưa có transcript: cần `--transcripts asr.json` (`{"audio/<id>/<id>.wav": "text", ...}`, gồm cả các lượt `_u1`, `_u2`… trong history).

## Mốc tham chiếu (train, 1545 mẫu)

| Backend | Score | Ghi chú |
|---|---|---|
| oracle | 1.0000 | kiểm tra agent loop + bộ chấm (0.9999 sau khi thêm luật chặn ID bịa) |
| empty | 0.0718 | chỉ đúng các mẫu ngoài phạm vi (7.2%) |

## Kết quả Colab T4 (2026-11-01)

| Thí nghiệm | val_seen (200) | val_unseen (316) |
|---|---|---|
| Qwen3-4B-Instruct-2507 zero-shot, transcript gốc | 0.239 (F1 0.345, hash 0.44) | 0.212 (F1 0.318, hash 0.36) |
| + luật chặn ID bịa (x-ref) | 0.238 (F1 0.352, hash 0.445) | |
| + luật chặn ID, transcript ASR PhoWhisper-medium | **0.092** (F1 0.165, hash 0.255) | |

- Lỗi chính: sai/thừa tham số tùy chọn ở tool đọc (`list_bills(type=…)`, `get_status(room=…)`), thiếu bước đọc trước khi ghi, hỏi thiếu thông tin sai chỗ (spurious/missed ask), và **bịa ID** (52 mẫu, đã chặn bằng luật x-ref trong `agent.py`).
- ASR chưa fine-tune trên 645 clip val_seen (T4): PhoWhisper-medium WER 0.304 / recall số-giờ 0.69 / 0.33 s/clip; PhoWhisper-large 0.284 / 0.70 / 0.49 s; whisper-large-v3-turbo 0.365 / 0.35 / 0.34 s. Sai nặng ở từ tiếng Anh/tên riêng (VTV3, RON95) và số → cần fine-tune (`lifemate/asr_train.py`).
- Train LoRA trên T4 (sm75): kernel memory-efficient của SDPA chỉ nhận q/k/v **fp16**, không GQA (`enable_gqa`), cùng dtype. Hai bẫy đã gặp: (1) transformers v5 gọi SDPA với `enable_gqa=True` → kernel math → OOM ở 6.4K token (13.3 GB, đo trên Colab); (2) torch 2.11 báo `is_bf16_supported()=True` trên T4 nên model nạp bf16 → không kernel efficient nào nhận. `train.py` dùng attention riêng `causal_sdpa` (repeat_kv + is_causal) và fp16 trên GPU < sm80; với fp16 + repeat_kv đo được 5.1 GB đỉnh ở 6.4K token (4-bit).
- Tốc độ: chạy cả 200 agent cùng lúc trên T4 làm tràn KV cache (13–15 phút/bước); `--wave 6` còn ~7–9 s/mẫu.
- Colab: sau `pip install vllm`, chạy ASR cần `--no-cudnn` (lỗi CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH).

## Kết quả Kaggle T4 (2026-11-01/02)

| Bước | Kết quả |
|---|---|
| LoRA r32 trên Qwen3-4B-Instruct-2507 (fp16, không 4-bit), 1065 quỹ đạo × 2 epoch | 5.7 h; loss 0.65 → 0.03 (train_loss 0.086); đỉnh 11.6 GB |
| Fine-tune PhoWhisper-medium (3 epoch, audio train+val_unseen) | val_seen 645 clip: **WER 0.304 → 0.104**, recall số/giờ 0.69 → **0.888**, 0.69 s/clip |


Pipeline cuối (`lifemate-final`: vLLM 0.32.0, torch 2.13+cu130, transformers 5.16.1, T4; LLM đã gộp LoRA, `--wave 8`):

| Tập | Score | F1 | Hash | Zero-shot trước đó |
|---|---|---|---|---|
| val_seen (200), transcript gốc | **0.658** | 0.737 | 0.765 | 0.239 |
| val_unseen (316, app chưa thấy khi train), transcript gốc | **0.456** | 0.587 | 0.557 | 0.212 |
| val_seen, transcript ASR đã fine-tune | **0.496** | 0.639 | 0.635 | 0.092 |

- val_seen theo loại: single 0.763, multi 0.651, out_of_scope 0.727, ask_missing 0.548; val_unseen: ask_missing yếu nhất (0.246).
- public_test (634 mẫu) chạy đúng entrypoint Docker `lifemate.infer` (ASR → agent): 634 dòng, validator của BTC báo 0 lỗi schema; 9.7 s/mẫu trên T4 tính cả ASR (`runs/kaggle_final/result/submission.jsonl`).
- Ngân sách: LLM 4.02B + ASR 0.76B = 4.79B tham số.
- Kaggle sau `pip install vllm`: phải gỡ `torchaudio` (bản cu128, lệch torch cu130 → không import được Whisper; audio tự đọc bằng soundfile) và `torchao` 0.10 (làm hỏng peft).
- transformers v5 lưu processor Whisper thành `processor_config.json` (feature extractor lồng bên trong). Bản v4 (ví dụ trong image vLLM) cần `preprocessor_config.json`, nên `export.legacy_preprocessor` ghi thêm file này.
- Docker (2026-11-02, `build_docker.ps1 -SkipDownload`): `lifemate:final` 36.4 GB. Smoke test CPU, `--network none`, đạt: vllm 0.32.0, torch 2.13.0+cu129, transformers 5.19.0; image đọc được processor ASR và tokenizer LLM. Chưa chạy end-to-end trên GPU trong image vì máy dev không có GPU NVIDIA.
- Kiểm tra trên GPU (`kaggle/docker_parity`, T4): dùng cùng vllm 0.32.0, transformers 5.19.0, cùng `models/`, biến môi trường và tham số `ENTRYPOINT` như image, chạy trên public_test. Kết quả: exit 0, 634 dòng, 0 lỗi schema, 11.4 s/mẫu tính cả nạp mô hình. Transcript ASR trùng 1933/1933 clip; 629/634 dòng trùng khớp với `lifemate-final`. 5 mẫu còn lại lệch chuỗi call ở LLM (2 mẫu lệch hash), do vLLM greedy không cho kết quả giống hệt bit khi thứ tự batch thay đổi.
- Để chạy ra kết quả lặp lại được (`backends.VLLMBackend`), có hai thay đổi:
  - `VLLM_ENABLE_V1_MULTIPROCESSING=0`: engine chạy trong cùng tiến trình, nên mỗi batch được lập lịch nguyên khối.
  - KV cache cố định theo tổng bộ nhớ GPU: `0.85·tổng − 9.25 GiB`, ~3 GiB trên T4, ~10.5 GiB trên A30. Trước đây KV cache phụ thuộc phần bộ nhớ còn trống: chạy có ASR được 23,984 token, không ASR được 24,976 token, và 1/634 mẫu ra khác.
  - Kiểm tra lại trên Kaggle T4 (`runs/kaggle_determinism3`): chạy entrypoint hai lần, KV cache 21,840 token ở cả hai lần, 634/634 dòng giống hệt nhau, 0 lỗi schema, 11.7 s/mẫu.
- Image: trọng số LLM chia thành 8 shard ~1 GB (`export.shard_safetensors`, kiểm tra 398 tensor giống từng byte), mỗi shard một layer. Push qua mạng chập chờn khi đó chỉ phải gửi lại ~1 GB thay vì cả 9.6 GB.
  - Kiểm tra trên GPU (`kaggle/shard_check`, T4, `runs/kaggle_shard_check`): vLLM nạp đủ 8/8 shard qua `model.safetensors.index.json`; entrypoint chạy 160 mẫu đầu public_test, 160/160 dòng giống hệt cả hai lần chạy của kiểm tra determinism.

## Giả định của bộ chấm

Công thức gốc của BTC chỉ có dạng ảnh. Bộ chấm local giả định: điểm mẫu = F1 nếu hash khớp (ngược lại 0), với P = M/(M+U) và R = M/|G|; điểm cuối = trung bình. Tham số được so trên hợp các khóa, nên thừa một tham số không-fuzzy thì call đó không khớp. List scalar so như multiset.
