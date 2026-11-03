"""Kaggle kernel: GPU test of what the Docker image lifemate:final ships (Kaggle cannot run Docker itself).

Same package versions as the image (vllm 0.32.0, transformers 5.19.0, soundfile; no torchaudio), same models/ (output of
manhhodinh/lifemate-final, copied into the image), same environment variables and ENTRYPOINT arguments, run from an /app-like
folder on public_test. Then: organisers' schema check, time per sample, and agreement with the earlier lifemate-final submission.
"""
import json
import os
import shutil
import subprocess
import sys
import time

W = "/kaggle/working"


def sh(cmd, check=True):
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=check)


def find(name, want_dir=False, parent=None):
    for d, dirs, files in os.walk("/kaggle/input", followlinks=True):
        if name in (dirs if want_dir else files) and (parent is None or os.path.basename(d) == parent):
            return os.path.join(d, name)
    return None


src = os.path.dirname(os.path.dirname(find("agent.py")))
zaic = os.path.dirname(os.path.dirname(find("train_tools_catalog.json")))
models = find("models", want_dir=True)
prev = find("submission.jsonl", parent="result")
pub_audio = find("audio", want_dir=True, parent="public_test")
print("code:", src, "\nzaic:", zaic, "\nmodels:", models, "\nprev:", prev, "\naudio:", pub_audio, flush=True)
print(open(f"{models}/MANIFEST.json").read(), flush=True)

# /data as the organisers mount it: the released public_test folder with its audio
data = f"{W}/data"
shutil.copytree(f"{zaic}/public_test", data, dirs_exist_ok=True)
os.symlink(pub_audio, f"{data}/audio")

# /app as in the Dockerfile: lifemate/, splits/, models/
app = f"{W}/app"
os.makedirs(app, exist_ok=True)
shutil.copytree(f"{src}/lifemate", f"{app}/lifemate")
# the dataset version attached at push time can lag the upload: fail fast on code without the fixed KV cache size
assert "kv_cache_memory_bytes" in open(f"{app}/lifemate/backends.py").read(), "stale lifemate code in lifemate-sft"
os.symlink(models, f"{app}/models")

sh(f"{sys.executable} -m pip install -q vllm==0.32.0 transformers==5.19.0 soundfile")
sh(f"{sys.executable} -m pip uninstall -y -q torchaudio torchcodec torchao", check=False)
sh(f"{sys.executable} -c \"import vllm, torch, transformers; print('vllm', vllm.__version__, 'torch', torch.__version__, "
   f"'transformers', transformers.__version__, 'gpu', torch.cuda.get_device_name(0))\"")

env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", VLLM_NO_USAGE_STATS="1", VLLM_ENABLE_V1_MULTIPROCESSING="0",
           PYTHONHASHSEED="0", CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES="0")
entry = [sys.executable, "-m", "lifemate.infer", "--data-dir", data, "--out", "result/submission.jsonl",
         "--asr-model", "models/asr", "--llm-model", "models/llm", "--wave", "8", "--no-cudnn"]
t0 = time.time()
with open(f"{W}/parity.log", "w") as log:
    rc = subprocess.run(entry, cwd=app, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
dt = time.time() - t0
out = f"{app}/result/submission.jsonl"
shutil.copytree(f"{app}/result", f"{W}/result")

# determinism: a second process on the same GPU (ASR output reused: it already matched clip for clip across runs)
with open(f"{W}/parity_run2.log", "w") as log:
    rc2 = subprocess.run(entry[:6] + ["result2/submission.jsonl"] + entry[7:] + ["--transcripts", "result/transcripts.json"],
                         cwd=app, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
shutil.copytree(f"{app}/result2", f"{W}/result2")
run2 = {r["id"]: r for r in (json.loads(line) for line in open(f"{app}/result2/submission.jsonl", encoding="utf-8"))}
# the validator wants <root>/public_test; /data keeps its own name above so infer's split detection is exercised as in Docker
os.makedirs(f"{W}/vroot", exist_ok=True)
os.symlink(data, f"{W}/vroot/public_test")
sh(f"cd {data} && {sys.executable} -m zaic_tools --split public_test --root {W}/vroot --validate {W}/result/submission.jsonl "
   f"| tee {W}/validate.log", check=False)

new = [json.loads(line) for line in open(out, encoding="utf-8")]
old = {r["id"]: r for r in (json.loads(line) for line in open(prev, encoding="utf-8"))}
same = sum(r == old.get(r["id"]) for r in new)
summary = {"returncode": rc, "lines": len(new), "seconds": round(dt, 1), "s_per_sample": round(dt / max(len(new), 1), 2),
           "identical_to_lifemate_final": same, "returncode_run2": rc2,
           "identical_run1_run2": sum(r == run2.get(r["id"]) for r in new),
           "differing_run1_run2": [r["id"] for r in new if r != run2.get(r["id"])]}
print(json.dumps(summary), flush=True)
json.dump(summary, open(f"{W}/parity_summary.json", "w"), indent=1)
shutil.rmtree(app)
shutil.rmtree(data)
shutil.rmtree(f"{W}/vroot")
