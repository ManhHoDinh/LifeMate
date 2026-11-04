"""Kaggle kernel: does synthetic data for unseen apps help? (experiment on the held-out train apps)

Continues the lifemate-train-qlora adapter for one epoch on the synthetic data of the 8 held-out apps
(lifemate-synth output synth/held_out/sft.jsonl, built without any held-out sample or state) mixed with a replay of
30% of the original SFT data, merges it, and scores val_unseen (target: beat 0.456) and val_seen (must not drop below 0.658)
with gold transcripts, exactly as lifemate-final scored the current model.
"""
import glob
import os
import shutil
import subprocess
import sys

BASE = "Qwen/Qwen3-4B-Instruct-2507"
REPLAY = "0.3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
W = "/kaggle/working"


def sh(cmd, check=True):
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=check)


def find(name, want_dir=False, parent=None, under=""):
    for d, dirs, files in os.walk("/kaggle/input", followlinks=True):
        if under in d and name in (dirs if want_dir else files) and (parent is None or os.path.basename(d) == parent):
            return os.path.join(d, name)
    return None


code = f"{W}/w"
shutil.copytree(os.path.dirname(os.path.dirname(find("agent.py"))), code, dirs_exist_ok=True)
assert "refresh_system" in open(f"{code}/lifemate/train.py").read(), "stale lifemate code in lifemate-sft"
zaic = os.path.dirname(os.path.dirname(find("train_tools_catalog.json")))
sft = find("sft_train.jsonl")
synth = find("sft.jsonl", parent="held_out", under="lifemate-synth")
adapter = os.path.dirname(find("adapter_config.json", under="lifemate-train-qlora"))
print("sft:", sft, "\nsynth:", synth, "\nadapter:", adapter, flush=True)
sh(f"wc -l {sft} {synth}")
os.environ["LIFEMATE_DATA"] = zaic
os.chdir(code)

# 1. continue the adapter: synthetic held-out-app data + replay (same fp16 LoRA setup as lifemate-train-qlora)
sh(f"{sys.executable} -m pip install -q -U peft bitsandbytes accelerate")
subprocess.run(f"{sys.executable} -m pip uninstall -y -q torchao", shell=True)
data = f"{synth},{sft}:{REPLAY}"
sh(f"{sys.executable} -u -m lifemate.train --data {data} --check")
sh(f"{sys.executable} -u -m lifemate.train --data {data} --init-adapter {adapter} --epochs 1 --out {W}/ckpt --merge")
for d in glob.glob(f"{W}/ckpt/checkpoint-*"):
    shutil.rmtree(d)
merged = f"{W}/ckpt/merged"

# 2. scores with gold transcripts (as lifemate-final: vLLM, --wave 8)
sh(f"{sys.executable} -m pip install -q vllm==0.32.0")
sh(f"{sys.executable} -m pip uninstall -y -q torchaudio torchcodec torchao", check=False)
for ids in ("val_unseen", "val_seen"):
    sh(f"{sys.executable} -u -m lifemate.run --split train --ids {ids} --backend vllm --model {merged} --wave 8 --gpu-mem 0.9 "
       f"--out {W}/runs/{ids} 2>&1 | grep -vE 'INFO|WARNING|it/s|Loading' | tee -a {W}/exp.log", check=False)
shutil.rmtree(merged)  # 8 GB; the adapter in ckpt/adapter is enough to rebuild it
shutil.rmtree(code)
sh(f"cat {W}/runs/*/summary.json", check=False)
