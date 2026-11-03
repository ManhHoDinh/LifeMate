# LifeMate final solution (ZAC 2026). Fully offline: models are baked into the image.
#
# Build (needs models/ from `python -m lifemate.export`, i.e. the output of the Kaggle kernel lifemate-final):
#   docker build -t lifemate:final .
# Run as the organisers do (split folder mounted at /data, result collected from /app/result):
#   docker run --gpus all --network none -v /path/to/private_test:/data -v $PWD/result:/app/result lifemate:final
#
# vLLM 0.32.0 is the version validated by the lifemate-final kernel (final.log). The cu129 build runs on any driver
# that supports CUDA 12 (minor-version compatibility); the plain v0.32.0 tag is CUDA 13 and needs driver >= 580.
ARG VLLM_IMAGE=vllm/vllm-openai:v0.32.0-cu129
FROM ${VLLM_IMAGE}

WORKDIR /app
# audio is decoded with soundfile (lifemate.asr.load_wav); a torchaudio built for another CUDA breaks the Whisper import
RUN pip install --no-cache-dir soundfile && (pip uninstall -y torchaudio torchcodec || true)
# models first: a code or ENV change then leaves these layers cached (no rebuild, no re-push). The LLM weights come as
# 8 shards of ~1 GB (export.shard_safetensors, run by build_docker.ps1), one layer each: a dropped connection during
# `docker push` then re-sends ~1 GB, not 8 GB.
COPY models/MANIFEST.json /app/models/
COPY models/asr/ /app/models/asr/
COPY models/llm/model-00001-of-00008.safetensors /app/models/llm/
COPY models/llm/model-00002-of-00008.safetensors /app/models/llm/
COPY models/llm/model-00003-of-00008.safetensors /app/models/llm/
COPY models/llm/model-00004-of-00008.safetensors /app/models/llm/
COPY models/llm/model-00005-of-00008.safetensors /app/models/llm/
COPY models/llm/model-00006-of-00008.safetensors /app/models/llm/
COPY models/llm/model-00007-of-00008.safetensors /app/models/llm/
COPY models/llm/model-00008-of-00008.safetensors /app/models/llm/
COPY models/llm/*.json models/llm/*.jinja /app/models/llm/

# VLLM_ENABLE_V1_MULTIPROCESSING=0: deterministic scheduling of offline batches (vLLM reproducibility docs)
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    PYTHONHASHSEED=0 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTHONUNBUFFERED=1
COPY lifemate/ /app/lifemate/
COPY splits/ /app/splits/

ENTRYPOINT ["python3", "-m", "lifemate.infer", "--data-dir", "/data", "--out", "result/submission.jsonl", \
            "--asr-model", "models/asr", "--llm-model", "models/llm", "--wave", "8", "--no-cudnn"]
