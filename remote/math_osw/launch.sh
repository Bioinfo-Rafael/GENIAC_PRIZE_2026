#!/usr/bin/env bash
set -euo pipefail
umask 077
OSW_STARTED="$(date +%s)"
OSW_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OSW_WORKSPACE="${GENIAC_WORKSPACE:-$HOME/geniac_math}"
OSW_ID="${1:-math_osw_$(date +%Y%m%d_%H%M%S)}"
[[ "$OSW_ID" =~ ^math_osw_[a-zA-Z0-9_-]+$ ]] || exit 2
OSW_ROOT="$OSW_WORKSPACE/math_osw/$OSW_ID"
OSW_PREVIOUS="${2:-$OSW_WORKSPACE/math_scale/math_scale_20261021_v1}"
mkdir -p "$OSW_ROOT"
exec 9>"$OSW_WORKSPACE/.launch.lock"
flock -n 9 || { echo 'Another experiment holds .launch.lock'; exit 1; }
exec 8>"$OSW_WORKSPACE/.math_training.lock"
flock -n 8 || { echo 'Another experiment holds .math_training.lock'; exit 1; }
OSW_TRAIN="${GENIAC_TRAIN_PYTHON:-$OSW_WORKSPACE/.venv-rank-ablation/bin/python}"
OSW_EVAL="${GENIAC_EVAL_PYTHON:-$OSW_WORKSPACE/.venv-eval/bin/python}"
OSW_DATA="${GENIAC_DATA_PYTHON:-$OSW_WORKSPACE/.venv-submission-data/bin/python}"
for OSW_PY in "$OSW_TRAIN" "$OSW_EVAL" "$OSW_DATA"; do test -x "$OSW_PY"; done
export HF_HOME="${HF_HOME:-$OSW_WORKSPACE/hf_cache}"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
"$OSW_TRAIN" "$OSW_HERE/budget_plan.py" "$OSW_ROOT" "$OSW_STARTED" "${OSW_PROFILE:-60m}"
# Snapshot the established helpers, plus this new isolated runner. No pip installs.
if [[ ! -d "$OSW_ROOT/code" ]]; then
  mkdir -p "$OSW_ROOT/code.partial"
  for OSW_DIR in math_osw math_scale math_curriculum math_training; do
    cp -R "$OSW_HERE/../$OSW_DIR" "$OSW_ROOT/code.partial/$OSW_DIR"
  done
  mkdir -p "$OSW_ROOT/code.partial/rank64_12k"
  cp "$OSW_HERE/../rank64_12k/official_prompt.py" "$OSW_ROOT/code.partial/rank64_12k/"
  mv "$OSW_ROOT/code.partial" "$OSW_ROOT/code"
fi
OSW_REMAINING="$("$OSW_TRAIN" -c 'import json,sys,time; print(max(0,int(json.load(open(sys.argv[1]))["deadline_epoch"]-time.time())-5))' "$OSW_ROOT/budget.json")"
if (( OSW_REMAINING <= 0 )); then echo 'Original deadline exhausted'; exit 75; fi
OSW_COMMAND=("$OSW_TRAIN" -u "$OSW_ROOT/code/math_osw/run.py" \
  --root "$OSW_ROOT" --predecessor "$OSW_PREVIOUS" \
  --train-python "$OSW_TRAIN" --eval-python "$OSW_EVAL" --data-python "$OSW_DATA")
if [[ "${OSW_PROFILE:-60m}" == 15m ]]; then
  # Runner bounds prepare/train/merge; submission is allowed to finish.
  exec "${OSW_COMMAND[@]}"
else
  exec timeout --signal=TERM --kill-after=5s "${OSW_REMAINING}s" "${OSW_COMMAND[@]}"
fi
