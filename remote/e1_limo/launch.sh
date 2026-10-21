#!/usr/bin/env bash
set -euo pipefail
umask 077
E1_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E1_WORKSPACE="${GENIAC_WORKSPACE:-$HOME/geniac_math}"
E1_ID="${1:?Usage: launch.sh RUN_ID [--source-run PATH] [--independent PATH]}"
shift
[[ "$E1_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
mkdir -p "$E1_WORKSPACE"
exec 9>"$E1_WORKSPACE/.launch.lock"
flock -n 9 || { echo 'Another experiment is active'; exit 1; }
E1_RUNTIME="$E1_WORKSPACE/e1_runtime/$E1_ID"
if [[ ! -d "$E1_RUNTIME/code" ]]; then
  rm -rf "$E1_RUNTIME/code.partial"
  mkdir -p "$E1_RUNTIME/code.partial"
  for E1_DIR in e1_limo math_training public_eval_diagnostics rank64_12k; do
    mkdir -p "$E1_RUNTIME/code.partial/$E1_DIR"
  done
  cp "$E1_HERE/"*.py "$E1_HERE/launch.sh" "$E1_RUNTIME/code.partial/e1_limo/"
  cp -R "$E1_HERE/../math_training/." "$E1_RUNTIME/code.partial/math_training/"
  cp "$E1_HERE/../public_eval_diagnostics/data.py" "$E1_RUNTIME/code.partial/public_eval_diagnostics/"
  cp "$E1_HERE/../rank64_12k/official_prompt.py" "$E1_RUNTIME/code.partial/rank64_12k/"
  mv "$E1_RUNTIME/code.partial" "$E1_RUNTIME/code"
fi
E1_TRAIN="${GENIAC_TRAIN_PYTHON:-$E1_WORKSPACE/.venv-train/bin/python}"
E1_EVAL="${GENIAC_EVAL_PYTHON:-$E1_WORKSPACE/.venv-eval/bin/python}"
E1_DATA="${GENIAC_DATA_PYTHON:-$E1_WORKSPACE/.venv-submission-data/bin/python}"
for E1_PY in "$E1_TRAIN" "$E1_EVAL" "$E1_DATA"; do test -x "$E1_PY"; done
export HF_HOME="${HF_HOME:-$E1_WORKSPACE/hf_cache}"
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
E1_RESUME=()
if [[ -f "$E1_WORKSPACE/suites/$E1_ID/suite_context.json" ]]; then E1_RESUME=(--resume); fi
exec "$E1_TRAIN" -u "$E1_RUNTIME/code/e1_limo/pipeline.py" \
  --workspace "$E1_WORKSPACE" --run-id "$E1_ID" \
  --source-run "$E1_WORKSPACE/runs/ada_20261016_221122" \
  --independent "$E1_WORKSPACE/diagnostics/rank64_12k_20261021_183439_public_eval_v1/independent_dev400.json" \
  --train-python "$E1_TRAIN" --eval-python "$E1_EVAL" --data-python "$E1_DATA" \
  "${E1_RESUME[@]}" "$@"
