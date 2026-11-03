"""Kaggle kernel: QLoRA SFT of Qwen3-4B-Instruct-2507 on lifemate SFT data (dataset manhhodinh/lifemate-sft).

Pushed with `kaggle kernels push -p kaggle/train_kernel`; output (/kaggle/working/ckpt/adapter) pulled with
`kaggle kernels output manhhodinh/lifemate-train-qlora`.
"""
import glob
import os
import shutil
import subprocess
import sys
import zipfile

EPOCHS = os.environ.get("EPOCHS", "2")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Kaggle T4 x2: one GPU, no DataParallel around the 4-bit model


def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def find(name):
    """Kaggle may unpack uploaded zips and mounts inputs under symlinks, so walk with followlinks."""
    for d, dirs, files in os.walk("/kaggle/input", followlinks=True):
        if name in files or name in dirs:
            return os.path.join(d, name)
    return None


sh("find -L /kaggle/input -maxdepth 4 | head -40")
work = "/kaggle/working/w"
os.makedirs(work, exist_ok=True)
code_zip = find("lifemate_code.zip")
if code_zip and os.path.isfile(code_zip):
    zipfile.ZipFile(code_zip).extractall(work)
else:  # unpacked by Kaggle: the folder holding lifemate/ and splits/
    src = os.path.dirname(find("agent.py") or "")
    shutil.copytree(os.path.dirname(src), work, dirs_exist_ok=True)
data = find("sft_train.jsonl")
os.chdir(work)
sh("ls -R lifemate splits | head -30")

sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
sh(f"{sys.executable} -m pip install -q -U peft bitsandbytes accelerate")
# Kaggle ships torchao 0.10; peft >= 0.21 refuses it while dispatching LoRA layers on a non-quantized model
subprocess.run(f"{sys.executable} -m pip uninstall -y -q torchao", shell=True)
sh(f"{sys.executable} -c \"import torch, transformers, peft; print(torch.__version__, transformers.__version__, peft.__version__)\"")
sh(f"{sys.executable} -u -m lifemate.train --data {data} --check")
# fp16 LoRA (no 4-bit dequant: faster, no quantization error) if the longest examples fit, else QLoRA (5.1 GB peak
# at 6.4K tokens measured on a T4)
mode = None
for cand in ("", "--qlora"):
    smoke = subprocess.run(f"{sys.executable} -u -m lifemate.train --data {data} {cand} --grad-accum 1 --max-steps 2 "
                           f"--longest-first --out /kaggle/working/smoke", shell=True)
    shutil.rmtree("/kaggle/working/smoke", ignore_errors=True)
    print(f"smoke {cand or 'fp16 LoRA'}: exit {smoke.returncode}", flush=True)
    if smoke.returncode == 0:
        mode = cand
        break
if mode is None:
    raise SystemExit("both fp16 LoRA and QLoRA failed on the longest examples; see the smoke logs above")
print("training mode:", mode or "fp16 LoRA", flush=True)
sh(f"{sys.executable} -u -m lifemate.train --data {data} {mode} --epochs {EPOCHS} --out /kaggle/working/ckpt")

# keep the output small: the adapter only, no optimizer checkpoints
for d in glob.glob("/kaggle/working/ckpt/checkpoint-*"):
    shutil.rmtree(d)
shutil.rmtree(work)
sh("ls -la /kaggle/working/ckpt/adapter")
