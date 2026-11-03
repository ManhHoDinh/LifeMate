"""Kaggle kernel: export the final models, evaluate them, and run the Docker entrypoint on public_test.

Inputs: manhhodinh/lifemate-sft (code + zaic data without audio), manhhodinh/lifemate-public-audio,
outputs of manhhodinh/lifemate-train-qlora (LoRA adapter) and manhhodinh/lifemate-asr-finetune (ASR + val_seen transcripts).
Output (/kaggle/working): models/ (what the Docker image ships), runs/ (scores), result/submission.jsonl, final.log.
"""
import os
import shutil
import subprocess
import sys

BASE = "Qwen/Qwen3-4B-Instruct-2507"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
W = "/kaggle/working"


def sh(cmd, check=True):
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=check)


def find(name, want_dir=False, parent=None):
    for d, dirs, files in os.walk("/kaggle/input", followlinks=True):
        if name in (dirs if want_dir else files) and (parent is None or os.path.basename(d) == parent):
            return os.path.join(d, name)
    return None


code = f"{W}/w"
shutil.copytree(os.path.dirname(os.path.dirname(find("agent.py"))), code, dirs_exist_ok=True)
zaic = os.path.dirname(os.path.dirname(find("train_tools_catalog.json")))  # folder holding train/ and public_test/
adapter = os.path.dirname(find("adapter_config.json"))
asr_model = os.path.dirname(find("model.safetensors", parent="asr_ft") or find("preprocessor_config.json"))
asr_val_seen = find("val_seen_ft.json")
pub_audio = find("audio", want_dir=True, parent="public_test")
print("zaic:", zaic, "\nadapter:", adapter, "\nasr:", asr_model, "\nasr transcripts:", asr_val_seen, "\npublic audio:", pub_audio, flush=True)

# public_test assembled as released: the data without audio (small copy) + the audio folder
pub = f"{W}/data/public_test"
shutil.copytree(f"{zaic}/public_test", pub, dirs_exist_ok=True)
os.symlink(pub_audio, f"{pub}/audio")
os.environ["LIFEMATE_DATA"] = zaic
os.chdir(code)

# the ASR was saved by transformers v5 (processor_config.json only); vllm may pin v4, which needs preprocessor_config.json
sys.path.insert(0, code)
from lifemate.export import legacy_preprocessor  # noqa: E402

shutil.copytree(asr_model, f"{W}/asr_src")
asr_model = f"{W}/asr_src"
legacy_preprocessor(asr_model)
print(sorted(os.listdir(asr_model)), flush=True)

sh(f"{sys.executable} -m pip install -q vllm peft soundfile")
# vllm pulls a torch built for another CUDA than Kaggle's torchaudio; transformers then fails to import Whisper.
# Audio is decoded with soundfile (lifemate.asr.load_wav), so torchaudio is not needed; torchao 0.10 breaks peft (as in training).
sh(f"{sys.executable} -m pip uninstall -y -q torchaudio torchcodec torchao", check=False)
sh(f"{sys.executable} -c \"import vllm, torch, transformers; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda, "
   f"'transformers', transformers.__version__); x = torch.ones(2, 2, device='cuda'); print('gpu ok', torch.cuda.get_device_name(0), (x @ x).sum().item())\"")
sh("nvidia-smi --query-gpu=driver_version --format=csv", check=False)

# 1. models/ for the Docker image (merged LLM + ASR), parameter budget asserted
sh(f"{sys.executable} -u -m lifemate.export --base {BASE} --adapter {adapter} --asr {asr_model} --out {W}/models")
llm = f"{W}/models/llm"

# 2-3. scores on the held-out splits (gold text, then ASR transcripts)
for ids, extra in (("val_seen", ""), ("val_unseen", ""), ("val_seen", f"--transcripts {asr_val_seen}")):
    tag = ids + ("_asr" if extra else "")
    sh(f"{sys.executable} -u -m lifemate.run --split train --ids {ids} --backend vllm --model {llm} --wave 8 --gpu-mem 0.9 "
       f"{extra} --out {W}/runs/{tag} 2>&1 | grep -vE 'INFO|WARNING|it/s|Loading' | tee -a {W}/final.log", check=False)

# 4. the Docker entrypoint on public_test, then the organisers' schema check
sh(f"{sys.executable} -u -m lifemate.infer --data-dir {pub} --asr-model {W}/models/asr --llm-model {llm} --wave 8 --no-cudnn "
   f"--out {W}/result/submission.jsonl 2>&1 | grep -vE 'INFO|WARNING|it/s|Loading' | tee -a {W}/final.log")
sh(f"cd {pub} && {sys.executable} -m zaic_tools --split public_test --root .. --validate {W}/result/submission.jsonl | tee -a {W}/final.log")
shutil.rmtree(code)
shutil.rmtree(f"{W}/data")
