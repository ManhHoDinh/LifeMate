"""Kaggle kernel: fine-tune PhoWhisper on the competition audio (train + val_unseen user turns), then WER on val_seen.

Inputs: manhhodinh/lifemate-sft (code), manhhodinh/lifemate-audio (train audio + train.jsonl).
Output: /kaggle/working/asr_ft (fp16 model), /kaggle/working/asr/val_seen_ft.json (transcripts), asr_eval.log.
"""
import os
import shutil
import subprocess
import sys

MODEL = os.environ.get("ASR_MODEL", "vinai/PhoWhisper-medium")
EPOCHS = os.environ.get("ASR_EPOCHS", "3")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def find(name, want_dir=False, parent=None):
    """Kaggle unpacks uploaded zips and mounts inputs under symlinks: walk with followlinks."""
    for d, dirs, files in os.walk("/kaggle/input", followlinks=True):
        pool = dirs if want_dir else files
        if name in pool and (parent is None or os.path.basename(d) == parent):
            return os.path.join(d, name)
    return None


work = "/kaggle/working/w"
src = os.path.dirname(os.path.dirname(find("agent.py")))  # folder holding lifemate/ and splits/
shutil.copytree(src, work, dirs_exist_ok=True)
data = "/kaggle/working/data/train"
os.makedirs(data, exist_ok=True)
shutil.copy(find("train.jsonl"), data)
os.symlink(find("audio", want_dir=True, parent="train"), f"{data}/audio")
os.environ["LIFEMATE_DATA"] = "/kaggle/working/data"
os.chdir(work)

sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
sh("ls /kaggle/working/data/train/audio | wc -l")
sh(f"{sys.executable} -u -m lifemate.asr_train --ids train val_unseen --model {MODEL} --epochs {EPOCHS} --out /kaggle/working/asr_ft")
sh(f"{sys.executable} -u -m lifemate.asr --split train --ids val_seen --model /kaggle/working/asr_ft "
   f"--out /kaggle/working/asr/val_seen_ft.json 2>&1 | tee /kaggle/working/asr_eval.log")
shutil.rmtree(work)
