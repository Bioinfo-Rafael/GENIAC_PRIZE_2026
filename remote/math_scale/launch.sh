#!/usr/bin/env bash
set -euo pipefail
umask 077
MS_STARTED="$(date +%s)"
MS_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MS_WORKSPACE="${GENIAC_WORKSPACE:-$HOME/geniac_math}"
MS_ID="${1:-math_scale_20261021_v1}"
[[ "$MS_ID" =~ ^math_scale_[a-zA-Z0-9_-]+$ ]] || exit 2
MS_SOURCE="${2:-$MS_WORKSPACE/math_curriculum/math_curriculum_20261021_3000_v1}"
mkdir -p "$MS_WORKSPACE"
exec 9>"$MS_WORKSPACE/.launch.lock"
flock -n 9 || { echo 'Another experiment holds .launch.lock'; exit 1; }
exec 8>"$MS_WORKSPACE/.math_training.lock"
flock -n 8 || { echo 'Another experiment holds .math_training.lock'; exit 1; }
MS_ROOT="$MS_WORKSPACE/math_scale/$MS_ID"
mkdir -p "$MS_ROOT"
MS_TRAIN="${GENIAC_TRAIN_PYTHON:-$MS_WORKSPACE/.venv-rank-ablation/bin/python}"
MS_EVAL="${GENIAC_EVAL_PYTHON:-$MS_WORKSPACE/.venv-eval/bin/python}"
MS_DATA="${GENIAC_DATA_PYTHON:-$MS_WORKSPACE/.venv-submission-data/bin/python}"
for MS_PY in "$MS_TRAIN" "$MS_EVAL" "$MS_DATA"; do test -x "$MS_PY"; done
# Start the nonrenewable wall clock before snapshots, environment checks or data work.
export MATH_SCALE_STARTED="$MS_STARTED"
export MS_BUDGET_PATH="$MS_ROOT/budget.json"
"$MS_TRAIN" -c 'import json,os,pathlib; p=pathlib.Path(os.environ["MS_BUDGET_PATH"]); start=float(os.environ["MATH_SCALE_STARTED"]); p.exists() or p.write_text(json.dumps(dict(started_epoch=start,deadline_epoch=start+36000,max_seconds=36000)))' 
if [[ ! -d "$MS_ROOT/code" ]]; then
  mkdir -p "$MS_ROOT/code.partial"
  for MS_DIR in math_scale math_curriculum math_training; do cp -R "$MS_HERE/../$MS_DIR" "$MS_ROOT/code.partial/$MS_DIR"; done
  mkdir -p "$MS_ROOT/code.partial/rank64_12k"
  cp "$MS_HERE/../rank64_12k/official_prompt.py" "$MS_ROOT/code.partial/rank64_12k/"
  mv "$MS_ROOT/code.partial" "$MS_ROOT/code"
fi
export HF_HOME="${HF_HOME:-$MS_WORKSPACE/hf_cache}"
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
# Outer watchdog includes startup and enforces the same absolute deadline on resume.
MS_REMAINING="$("$MS_TRAIN" -c 'import json,sys,time; print(max(0,int(json.load(open(sys.argv[1]))["deadline_epoch"]-time.time())-5))' "$MS_ROOT/budget.json")"
if (( MS_REMAINING <= 0 )); then echo 'Original 600-minute budget exhausted'; exit 75; fi
exec timeout --signal=TERM --kill-after=5s "${MS_REMAINING}s" "$MS_TRAIN" -u "$MS_ROOT/code/math_scale/run.py" \
  --root "$MS_ROOT" --source "$MS_SOURCE" --eval-python "$MS_EVAL" --data-python "$MS_DATA"
