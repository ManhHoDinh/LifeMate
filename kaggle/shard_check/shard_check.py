"""Kaggle kernel: GPU check of the 8-shard LLM layout the Docker image ships (export.shard_safetensors, run by build_docker.ps1).

The parity kernel loads the single-file models/llm of lifemate-final; the image has 8 shards + model.safetensors.index.json
instead. Here the same file is sharded as build_docker.ps1 does, the image entrypoint runs on it over the first LIMIT
public_test samples (ASR skipped: transcripts of lifemate-docker-parity v4), and the output must equal that run's lines.
Waves run in order, so the first LIMIT samples see the same batches as in the full run.
"""
import json
import os
import shutil
import subprocess
import sys
import time

W = "/kaggle/working"
LIMIT = 160  # a multiple of --wave 8


def sh(cmd, check=True):
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=check)


def find(name, want_dir=False, parent=None, under=""):
    for d, dirs, files in os.walk("/kaggle/input", followlinks=True):
        if under in d and name in (dirs if want_dir else files) and (parent is None or os.path.basename(d) == parent):
            return os.path.join(d, name)
    return None


src = os.path.dirname(os.path.dirname(find("agent.py")))
zaic = os.path.dirname(os.path.dirname(find("train_tools_catalog.json")))
models = find("models", want_dir=True, under="lifemate-final")
ref_dir = os.path.dirname(find("transcripts.json", parent="result", under="docker-parity"))
pub_audio = find("audio", want_dir=True, parent="public_test")
print("code:", src, "\nzaic:", zaic, "\nmodels:", models, "\nref:", ref_dir, "\naudio:", pub_audio, flush=True)

# /data and /app as in the parity kernel (same paths, so the reference transcripts' keys resolve)
data = f"{W}/data"
shutil.copytree(f"{zaic}/public_test", data, dirs_exist_ok=True)
os.symlink(pub_audio, f"{data}/audio")
app = f"{W}/app"
os.makedirs(app, exist_ok=True)
shutil.copytree(f"{src}/lifemate", f"{app}/lifemate")
assert "def shard_safetensors" in open(f"{app}/lifemate/export.py").read(), "stale lifemate code in lifemate-sft"

# models/ as in the image: asr as is, llm = every small file + 8 shards of model.safetensors (written to /tmp, not output)
m = "/tmp/models"
os.makedirs(f"{m}/llm", exist_ok=True)
os.symlink(f"{models}/asr", f"{m}/asr")
shutil.copy(f"{models}/MANIFEST.json", m)
for f in os.listdir(f"{models}/llm"):
    if f != "model.safetensors":
        shutil.copy(f"{models}/llm/{f}", f"{m}/llm")
os.symlink(f"{models}/llm/model.safetensors", f"{m}/llm/model.safetensors")  # read, then only the link is unlinked
sh("df -h /tmp")
sys.path.insert(0, app)
from lifemate.export import shard_safetensors  # noqa: E402

shard_safetensors(f"{m}/llm", 8)
print(sorted(os.listdir(f"{m}/llm")), flush=True)
os.symlink(m, f"{app}/models")

sh(f"{sys.executable} -m pip install -q vllm==0.32.0 transformers==5.19.0 soundfile")
sh(f"{sys.executable} -m pip uninstall -y -q torchaudio torchcodec torchao", check=False)

env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", VLLM_NO_USAGE_STATS="1", VLLM_ENABLE_V1_MULTIPROCESSING="0",
           PYTHONHASHSEED="0", CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES="0")
entry = [sys.executable, "-m", "lifemate.infer", "--data-dir", data, "--out", "result/submission.jsonl",
         "--asr-model", "models/asr", "--llm-model", "models/llm", "--wave", "8", "--no-cudnn",
         "--transcripts", f"{ref_dir}/transcripts.json", "--limit", str(LIMIT)]
t0 = time.time()
with open(f"{W}/shard.log", "w") as log:
    rc = subprocess.run(entry, cwd=app, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
dt = time.time() - t0
shutil.copytree(f"{app}/result", f"{W}/result")

new = [json.loads(line) for line in open(f"{W}/result/submission.jsonl", encoding="utf-8")]
summary = {"returncode": rc, "lines": len(new), "seconds": round(dt, 1), "shards": sorted(os.listdir(f"{m}/llm"))}
for name in ("result", "result2"):  # v4 run 1 (with ASR) and run 2 (same transcripts): both must match
    ref = {r["id"]: r for r in (json.loads(line) for line in open(f"{os.path.dirname(ref_dir)}/{name}/submission.jsonl",
                                                                   encoding="utf-8"))}
    summary[f"identical_to_v4_{name}"] = sum(r == ref.get(r["id"]) for r in new)
    summary[f"differing_v4_{name}"] = [r["id"] for r in new if r != ref.get(r["id"])]
print(json.dumps(summary), flush=True)
json.dump(summary, open(f"{W}/shard_summary.json", "w"), indent=1)
shutil.rmtree(app)
shutil.rmtree(data)
shutil.rmtree(m)
