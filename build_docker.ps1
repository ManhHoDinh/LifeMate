# Build the offline Docker image (lifemate:final) in a staging folder on a drive with space (~25 GB).
#   powershell -ExecutionPolicy Bypass -File build_docker.ps1 [-Stage C:\lifemate_build] [-SkipDownload] [-Save]
# Steps: copy code -> download models/ from the Kaggle kernel lifemate-final -> docker build -> CPU smoke test
# (loads the ASR processor and LLM tokenizer inside the image; the full run needs an NVIDIA GPU) -> optional docker save.
param(
    [string]$Stage = "C:\lifemate_build",
    [switch]$SkipDownload,
    [switch]$Save
)
$ErrorActionPreference = "Stop"
$Repo = $PSScriptRoot
$env:PYTHONUTF8 = "1"
$env:PATH = "$HOME\.local\bin;$env:PATH"

docker info --format "{{.ServerVersion}}" | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Docker daemon is not running: start Docker Desktop first." }

New-Item -ItemType Directory -Force $Stage | Out-Null
foreach ($d in "lifemate", "splits") {
    robocopy "$Repo\$d" "$Stage\$d" /MIR /XD __pycache__ /NFL /NDL /NJH /NJS | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy $d failed" }
}
Copy-Item "$Repo\Dockerfile", "$Repo\.dockerignore" $Stage -Force

if (-not $SkipDownload) {
    # merged Qwen3-4B + LoRA (models/llm, ~8 GB fp16) and fine-tuned PhoWhisper-medium (models/asr, ~1.5 GB)
    kaggle kernels output manhhodinh/lifemate-final -p $Stage --file-pattern '^models/'
    if ($LASTEXITCODE -ne 0) { throw "kaggle download failed" }
}
# the Dockerfile copies the LLM as 8 shards (one ~1 GB layer each, so an interrupted push resumes per shard)
if (Test-Path "$Stage\models\llm\model.safetensors") {
    Push-Location $Repo
    python -c "from lifemate.export import shard_safetensors; shard_safetensors(r'$Stage\models\llm', 8)"
    $rc = $LASTEXITCODE
    Pop-Location
    if ($rc -ne 0) { throw "sharding the LLM failed" }
}
foreach ($f in "models\MANIFEST.json", "models\llm\config.json", "models\llm\tokenizer.json", "models\llm\model.safetensors.index.json",
               "models\asr\model.safetensors", "models\asr\preprocessor_config.json") {
    if (-not (Test-Path "$Stage\$f")) { throw "missing $Stage\$f" }
}
Get-Content "$Stage\models\MANIFEST.json"

docker build -t lifemate:final $Stage
if ($LASTEXITCODE -ne 0) { throw "docker build failed" }

# CPU smoke test: the image's own transformers must read both model folders offline
docker run --rm --network none --entrypoint python3 lifemate:final -c @"
import vllm, torch, transformers
from transformers import AutoTokenizer, WhisperProcessor
import lifemate.infer
WhisperProcessor.from_pretrained('models/asr')
AutoTokenizer.from_pretrained('models/llm')
import os, json; assert os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] == '0'
from safetensors import safe_open
wm = json.load(open('models/llm/model.safetensors.index.json'))['weight_map']
assert sum(len(list(safe_open('models/llm/' + s, 'pt').keys())) for s in set(wm.values())) == len(wm) == 398
print('smoke ok: vllm', vllm.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__)
"@
if ($LASTEXITCODE -ne 0) { throw "smoke test failed" }

if ($Save) {
    docker save lifemate:final -o "$Stage\lifemate_final.tar"
    Get-Item "$Stage\lifemate_final.tar" | Select-Object FullName, Length
}
Write-Host "Built lifemate:final. GPU run: docker run --gpus all --network none -v <split folder>:/data -v <out>:/app/result lifemate:final"
