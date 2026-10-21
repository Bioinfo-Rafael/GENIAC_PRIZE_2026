#!/usr/bin/env bash
set -euo pipefail
umask 077
NEW_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEW_WORKSPACE="${GENIAC_WORKSPACE:-$HOME/geniac_math}"
NEW_ID="${1:?Usage: launch.sh RUN_ID --source-rank PATH}"
shift
[[ "$NEW_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
mkdir -p "$NEW_WORKSPACE"
exec 9>"$NEW_WORKSPACE/.launch.lock"
flock -n 9 || { echo 'Another experiment is active'; exit 1; }
NEW_ROOT="$NEW_WORKSPACE/rank64_12k/$NEW_ID"
mkdir -p "$NEW_ROOT"
if [[ ! -d "$NEW_ROOT/code" ]]; then
  mkdir -p "$NEW_ROOT/code.partial"
  for NEW_DIR in rank64_12k stage1_rank_ablation math_training; do
    mkdir -p "$NEW_ROOT/code.partial/$NEW_DIR"
    find "$NEW_HERE/../$NEW_DIR" -maxdepth 1 -type f \( -name '*.py' -o -name '*.json' -o -name '*.txt' -o -name '*.sh' -o -name '*.jinja' \) \
      -exec cp '{}' "$NEW_ROOT/code.partial/$NEW_DIR/" \;
  done
  cp -R "$NEW_HERE/../math_training/submission_assets" "$NEW_ROOT/code.partial/math_training/"
  mv "$NEW_ROOT/code.partial" "$NEW_ROOT/code"
fi
NEW_CODE="$NEW_ROOT/code/rank64_12k"
NEW_TRAIN="$NEW_WORKSPACE/.venv-rank-ablation/bin/python"
NEW_EVAL="${GENIAC_EVAL_PYTHON:-$NEW_WORKSPACE/.venv-eval/bin/python}"
NEW_DATA="$NEW_WORKSPACE/.venv-submission-data/bin/python"
test -x "$NEW_TRAIN"
test -x "$NEW_EVAL"
# Existing D training environment is read-only; do not reinstall packages or benchmark.
"$NEW_TRAIN" -c 'import torch, peft, datasets; assert torch.cuda.is_available(); print("Existing D training environment ready")'
if [[ ! -x "$NEW_DATA" ]]; then
  "$NEW_TRAIN" -m venv "$NEW_WORKSPACE/.venv-submission-data"
  "$NEW_DATA" -m pip install -r "$NEW_ROOT/code/math_training/requirements-submission-data.txt"
fi
"$NEW_DATA" -c 'import datasets'
export HF_HOME="${HF_HOME:-$NEW_WORKSPACE/hf_cache}"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
exec "$NEW_TRAIN" -u "$NEW_CODE/run.py" --root "$NEW_ROOT" --eval-python "$NEW_EVAL" --data-python "$NEW_DATA" "$@"
