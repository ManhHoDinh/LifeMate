"""Kaggle kernel: synthetic SFT data with lifemate.synth on 2x T4 (vLLM tensor parallel).

MODE = "smoke": each candidate generator on 2 held-out apps x 3 states (load check, speed, acceptance, sample output).
MODE = "full":  MODEL on (a) the held-out train apps (experiment, split train) and (b) the apps new in public_test.
Inputs: manhhodinh/lifemate-sft (code + zaic data without audio). Only train_environment states are read; the target
apps' states are generated (lifemate.synth). Output: /kaggle/working/synth/<run>/{sft,val,states}.jsonl + summary.json.
"""
import os
import shutil
import subprocess
import sys
import time

MODE = "full"
# full: which parts to generate and how many states per target app. Teammates can split the parts across accounts
# (one kernel per part); a part stops starting new rounds after its minute budget so the 12 h session ends with outputs.
PARTS = {"held_out": ("train", "held_out", 25, 300), "new_apps": ("public_test", "new", 20, 300)}
CANDIDATES = ["QuantTrio/Qwen3-30B-A3B-Instruct-2507-AWQ", "Qwen/Qwen3-32B-AWQ", "Qwen/Qwen3-14B-AWQ"]
# full: the MoE 30B-A3B (3B active, several times faster) if its 4-bit build runs on a T4, else Qwen3-32B-AWQ, which matched
# more often than 14B in the smoke run (9 vs 8 of 24 kept, 7 vs 13 agent mismatches)
FULL_CANDIDATES = ["cyankiwi/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit", "Qwen/Qwen3-32B-AWQ"]
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
assert "max_minutes" in open(f"{code}/lifemate/synth.py").read(), "stale lifemate code in lifemate-sft"
zaic = os.path.dirname(os.path.dirname(find("train_tools_catalog.json")))
os.environ["LIFEMATE_DATA"] = zaic
os.chdir(code)

sh(f"{sys.executable} -m pip install -q vllm==0.32.0")
sh(f"{sys.executable} -m pip uninstall -y -q torchaudio torchcodec torchao", check=False)
sh(f"{sys.executable} -c \"import torch, vllm; print('vllm', vllm.__version__, 'gpus', torch.cuda.device_count(), torch.cuda.get_device_name(0))\"")
import torch  # noqa: E402

tp = torch.cuda.device_count()


def synth(model, split, targets, out, states, extra="", minutes=0):
    t0 = time.time()
    with open(f"{W}/synth.log", "a") as log:
        log.write(f"\n===== {model} {split} {targets} -> {out}\n")
        log.flush()
        rc = subprocess.run(f"{sys.executable} -u -m lifemate.synth --split {split} --targets {targets} --model {model} --tp {tp} "
                            f"--max-model-len 12288 --gpu-mem 0.9 --states-per-app {states} --chunk 16 --max-minutes {minutes} --out {W}/synth/{out} {extra}",
                            shell=True, stdout=log, stderr=subprocess.STDOUT).returncode
    print(f"{model} {out}: rc {rc}, {time.time() - t0:.0f}s", flush=True)
    sh(f"cat {W}/synth/{out}/summary.json", check=False)


if MODE == "smoke":
    for i, m in enumerate(CANDIDATES):
        synth(m, "train", "flights,medication", f"smoke{i}", 3, "--val-ratio 0")
else:
    model = None
    for m in FULL_CANDIDATES:  # load + one generation, in a child process so a failed load frees the GPUs
        probe = (f"from lifemate.synth import Chat; c = Chat('{m}', tp={tp}, max_model_len=12288); "
                 f"print('probe:', c([[{{'role': 'user', 'content': 'Xin chào, bạn là ai?'}}]], 0.0, 32))")
        t0 = time.time()
        with open(f"{W}/probe.log", "a") as log:
            rc = subprocess.run([sys.executable, "-c", probe], stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"probe {m}: rc {rc}, {time.time() - t0:.0f}s", flush=True)
        sh(f"grep -E 'probe:|Error' {W}/probe.log | tail -n 3", check=False)
        if rc == 0:
            model = m
            break
    assert model, "no generator model loads"
    print("generator:", model, flush=True)
    for out, (split, targets, states, minutes) in PARTS.items():
        synth(model, split, targets, out, states, minutes=minutes)
sh(f"grep -vE ' INFO |WARNING|it/s\\]' {W}/synth.log | tail -n 80", check=False)
shutil.rmtree(code)
