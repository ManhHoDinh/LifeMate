#!/usr/bin/env bash
# Chain the Kaggle kernels: LoRA training -> ASR fine-tune -> final (export + eval + public_test), then pull outputs.
#   bash kaggle/orchestrate.sh [start-stage]   stage: train | asr | final | pull
# Stops at the first kernel that does not COMPLETE and prints the tail of its log.
set -u
export PATH="$HOME/.local/bin:$PATH" PYTHONUTF8=1 PYTHONIOENCODING=utf-8
HERE="$(cd "$(dirname "$0")" && pwd)"
STAGE="${1:-train}"

tok() { export KAGGLE_API_TOKEN=$(kaggle auth print-access-token 2>/dev/null); }

wait_kernel() {  # $1 = owner/slug ; returns 0 when COMPLETE
  while true; do
    tok
    s=$(kaggle kernels status "$1" 2>&1)
    echo "$(date -u +%T) $1 $s"
    if echo "$s" | grep -q COMPLETE; then return 0; fi
    if echo "$s" | grep -qiE "error|cancel"; then
      kaggle kernels logs "$1" 2>&1 | python -X utf8 -c "
import sys, json
t = sys.stdin.read()
try: s = ''.join(e['data'] for e in json.loads(t))
except Exception: s = t
i = s.rfind('Traceback'); print(s[i:i + 3000] if i >= 0 else s[-3000:])"
      return 1
    fi
    sleep 240
  done
}

push() {  # $1 = kernel folder ; retries while Kaggle's GPU session limit is reached
  while true; do
    tok
    out=$(cd "$HERE/$1" && kaggle kernels push -p . --accelerator NvidiaTeslaT4 2>&1)
    echo "$out" | tail -1
    if echo "$out" | grep -q "successfully pushed"; then sleep 60; return 0; fi
    if echo "$out" | grep -q "session count"; then sleep 300; continue; fi
    return 1
  done
}

case "$STAGE" in
  train) wait_kernel manhhodinh/lifemate-train-qlora || exit 1 ;&
  asr)   push asr_kernel || exit 1; wait_kernel manhhodinh/lifemate-asr-finetune || exit 1 ;&
  final) push final_kernel || exit 1; wait_kernel manhhodinh/lifemate-final || exit 1 ;&
  pull)
    tok
    dest="$HERE/../runs/kaggle_final"
    mkdir -p "$dest"
    kaggle kernels output manhhodinh/lifemate-final -p "$dest" --file-pattern '^(runs/|result/|final\.log|models/MANIFEST\.json)' 2>&1 | tail -3
    kaggle kernels output manhhodinh/lifemate-asr-finetune -p "$HERE/../runs/kaggle_asr" --file-pattern '^(asr/|asr_eval\.log)' 2>&1 | tail -2
    echo "PULLED -> $dest"
    ;;
esac
