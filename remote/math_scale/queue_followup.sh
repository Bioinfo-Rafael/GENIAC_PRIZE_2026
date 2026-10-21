#!/usr/bin/env bash
set -euo pipefail
umask 077
MF_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MF_WORKSPACE="${GENIAC_WORKSPACE:-$HOME/geniac_math}"
MF_PREDECESSOR="${1:-$MF_WORKSPACE/math_scale/math_scale_20261021_v1}"
MF_ID="${2:-math_scale_followup_20261021_v1}"
[[ "$MF_ID" =~ ^math_scale_followup_[a-zA-Z0-9_-]+$ ]] || exit 2
MF_ROOT="$MF_WORKSPACE/math_scale/$MF_ID"
mkdir -p "$MF_ROOT"
# Queue-specific lock: do not take the GPU locks until the predecessor completes.
exec 7>"$MF_ROOT/.queue.lock"
flock -n 7 || { echo 'Followup queue already running'; exit 1; }
MF_TRAIN="${GENIAC_TRAIN_PYTHON:-$MF_WORKSPACE/.venv-rank-ablation/bin/python}"
MF_EVAL="${GENIAC_EVAL_PYTHON:-$MF_WORKSPACE/.venv-eval/bin/python}"
MF_DATA="${GENIAC_DATA_PYTHON:-$MF_WORKSPACE/.venv-submission-data/bin/python}"
for MF_PY in "$MF_TRAIN" "$MF_EVAL" "$MF_DATA"; do test -x "$MF_PY"; done
MF_REMAINING="$("$MF_TRAIN" -c 'import json,sys,time; print(max(0,int(json.load(open(sys.argv[1]))["deadline_epoch"]-time.time())-5))' "$MF_PREDECESSOR/budget.json")"
if (( MF_REMAINING <= 0 )); then echo 'Original deadline exhausted'; exit 75; fi
if [[ ! -d "$MF_ROOT/code" ]]; then
  mkdir -p "$MF_ROOT/code.partial"
  for MF_DIR in math_scale math_curriculum math_training; do cp -R "$MF_HERE/../$MF_DIR" "$MF_ROOT/code.partial/$MF_DIR"; done
  mkdir -p "$MF_ROOT/code.partial/rank64_12k"
  cp "$MF_HERE/../rank64_12k/official_prompt.py" "$MF_ROOT/code.partial/rank64_12k/"
  mv "$MF_ROOT/code.partial" "$MF_ROOT/code"
fi
# Recompute after snapshot so startup never extends the deadline.
MF_REMAINING="$("$MF_TRAIN" -c 'import json,sys,time; print(max(0,int(json.load(open(sys.argv[1]))["deadline_epoch"]-time.time())-5))' "$MF_PREDECESSOR/budget.json")"
if (( MF_REMAINING <= 0 )); then exit 75; fi
export HF_HOME="${HF_HOME:-$MF_WORKSPACE/hf_cache}"
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
exec timeout --signal=TERM --kill-after=5s "${MF_REMAINING}s" "$MF_TRAIN" -u "$MF_ROOT/code/math_scale/followup.py" \
  --root "$MF_ROOT" --source "$MF_PREDECESSOR" --eval-python "$MF_EVAL" --data-python "$MF_DATA"
