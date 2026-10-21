#!/usr/bin/env bash
# CPU-only: no GPU lock, model load, training, generation or package installation.
set -euo pipefail
umask 077
DIAG_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIAG_SOURCE="${1:?Usage: launch.sh R12_RUN_PATH}"
DIAG_WORKSPACE="${GENIAC_WORKSPACE:-$HOME/geniac_math}"
DIAG_PYTHON="${GENIAC_DIAG_PYTHON:-$DIAG_WORKSPACE/.venv-rank-ablation/bin/python}"
DIAG_ROOT="$DIAG_WORKSPACE/diagnostics/$(basename "$DIAG_SOURCE")_saved_v1"
mkdir -p "$DIAG_ROOT"
exec 9>"$DIAG_ROOT/.lock"
flock -n 9 || { echo 'This diagnostic is already running'; exit 1; }
if [[ ! -d "$DIAG_ROOT/code" ]]; then
  mkdir -p "$DIAG_ROOT/code.partial/rank64_diagnostics" "$DIAG_ROOT/code.partial/math_training"
  cp "$DIAG_HERE/analyze.py" "$DIAG_ROOT/code.partial/rank64_diagnostics/"
  for DIAG_FILE in core.py suite_utils.py worker.py grading.py chat_template.jinja; do
    cp "$DIAG_HERE/../math_training/$DIAG_FILE" "$DIAG_ROOT/code.partial/math_training/"
  done
  mv "$DIAG_ROOT/code.partial" "$DIAG_ROOT/code"
fi
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec "$DIAG_PYTHON" -u "$DIAG_ROOT/code/rank64_diagnostics/analyze.py" \
  --r12 "$DIAG_SOURCE" --output "$DIAG_ROOT"
