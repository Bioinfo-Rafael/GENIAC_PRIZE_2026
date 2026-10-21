#!/usr/bin/env bash
set -euo pipefail
umask 077
MC_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MC_WORKSPACE="${GENIAC_WORKSPACE:-$HOME/geniac_math}"
MC_ID="${1:?Usage: launch.sh math_curriculum_RUN_ID [SOURCE_RUN_ID]}"
MC_SOURCE_ID="${2:-}"
if [[ -n "$MC_SOURCE_ID" ]]; then
  [[ "$MC_SOURCE_ID" =~ ^math_curriculum_[a-zA-Z0-9_-]+$ && "$MC_SOURCE_ID" != "$MC_ID" ]] || { echo "Use distinct safe source and new run IDs"; exit 2; }
fi
[[ "$MC_ID" =~ ^math_curriculum_[a-zA-Z0-9_-]+$ ]] || { echo 'Expected math_curriculum_ prefix and safe run ID'; exit 2; }
mkdir -p "$MC_WORKSPACE"
exec 9>"$MC_WORKSPACE/.launch.lock"
flock -n 9 || { echo 'Another experiment holds .launch.lock'; exit 1; }
exec 8>"$MC_WORKSPACE/.math_training.lock"
flock -n 8 || { echo 'Another experiment holds .math_training.lock'; exit 1; }
MC_ROOT="$MC_WORKSPACE/math_curriculum/$MC_ID"
mkdir -p "$MC_ROOT"
if [[ ! -d "$MC_ROOT/code" ]]; then
  rm -rf "$MC_ROOT/code.partial"
  mkdir -p "$MC_ROOT/code.partial"
  for MC_DIR in math_curriculum math_training; do
    cp -R "$MC_HERE/../$MC_DIR" "$MC_ROOT/code.partial/$MC_DIR"
  done
  mkdir -p "$MC_ROOT/code.partial/public_eval_diagnostics" "$MC_ROOT/code.partial/rank64_12k"
  cp "$MC_HERE/../public_eval_diagnostics/data.py" "$MC_ROOT/code.partial/public_eval_diagnostics/"
  cp "$MC_HERE/../rank64_12k/official_prompt.py" "$MC_ROOT/code.partial/rank64_12k/"
  mv "$MC_ROOT/code.partial" "$MC_ROOT/code"
fi
MC_TRAIN="${GENIAC_TRAIN_PYTHON:-$MC_WORKSPACE/.venv-rank-ablation/bin/python}"
MC_EVAL="${GENIAC_EVAL_PYTHON:-$MC_WORKSPACE/.venv-eval/bin/python}"
MC_DATA="${GENIAC_DATA_PYTHON:-$MC_WORKSPACE/.venv-submission-data/bin/python}"
for MC_PY in "$MC_TRAIN" "$MC_EVAL" "$MC_DATA"; do
  test -x "$MC_PY" || { echo "Existing venv missing: $MC_PY"; exit 1; }
done
export HF_HOME="${HF_HOME:-$MC_WORKSPACE/hf_cache}"
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
if [[ -n "$MC_SOURCE_ID" ]]; then
  exec "$MC_TRAIN" -u "$MC_ROOT/code/math_curriculum/run.py" --root "$MC_ROOT" \
    --continue-from "$MC_WORKSPACE/math_curriculum/$MC_SOURCE_ID" \
    --eval-python "$MC_EVAL" --data-python "$MC_DATA"
fi
# The small submission-first controller is separate from the frozen training
# snapshot. Existing optimizer/checkpoint signatures therefore remain unchanged.
mkdir -p "$MC_ROOT/control_submission_first"
if [[ ! -f "$MC_ROOT/control_submission_first/run.py" ]]; then
  cp "$MC_HERE/run.py" "$MC_ROOT/control_submission_first/run.py.partial"
  mv "$MC_ROOT/control_submission_first/run.py.partial" "$MC_ROOT/control_submission_first/run.py"
fi
export PYTHONPATH="$MC_ROOT/code/math_curriculum${PYTHONPATH:+:$PYTHONPATH}"
exec "$MC_TRAIN" -u "$MC_ROOT/control_submission_first/run.py" --root "$MC_ROOT" \
  --eval-python "$MC_EVAL" --data-python "$MC_DATA"
