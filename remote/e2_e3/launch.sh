#!/usr/bin/env bash
set -euo pipefail
umask 077
E23_STARTED="$(date +%s)"
E23_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E23_WORKSPACE="${GENIAC_WORKSPACE:-$HOME/geniac_math}"
E23_ID="${1:?Usage: launch.sh e2e3_RUN_ID [--baseline-submission DIR]}"
shift
[[ "$E23_ID" =~ ^e2e3_[a-zA-Z0-9_-]+$ ]] || exit 2
mkdir -p "$E23_WORKSPACE"
exec 9>"$E23_WORKSPACE/.launch.lock"
flock -n 9 || { echo 'Another experiment is active; no job stopped'; exit 1; }
E23_RUNTIME="$E23_WORKSPACE/e2_e3_runtime/$E23_ID"
if [[ ! -d "$E23_RUNTIME/code" ]]; then
  mkdir -p "$E23_RUNTIME/code.partial/e2_e3" "$E23_RUNTIME/code.partial/public_eval_diagnostics" "$E23_RUNTIME/code.partial/rank64_12k"
  cp "$E23_HERE/"*.py "$E23_HERE/launch.sh" "$E23_RUNTIME/code.partial/e2_e3/"
  cp -R "$E23_HERE/../math_training" "$E23_RUNTIME/code.partial/"
  cp "$E23_HERE/../public_eval_diagnostics/data.py" "$E23_RUNTIME/code.partial/public_eval_diagnostics/"
  cp "$E23_HERE/../rank64_12k/official_prompt.py" "$E23_RUNTIME/code.partial/rank64_12k/"
  mv "$E23_RUNTIME/code.partial" "$E23_RUNTIME/code"
fi
E23_TRAIN="${GENIAC_TRAIN_PYTHON:-$E23_WORKSPACE/.venv-train/bin/python}"
E23_EVAL="${GENIAC_EVAL_PYTHON:-$E23_WORKSPACE/.venv-eval/bin/python}"
E23_DATA="${GENIAC_DATA_PYTHON:-$E23_WORKSPACE/.venv-submission-data/bin/python}"
for E23_PY in "$E23_TRAIN" "$E23_EVAL" "$E23_DATA"; do test -x "$E23_PY"; done
export HF_HOME="${HF_HOME:-$E23_WORKSPACE/hf_cache}"
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
E23_RESUME=()
if [[ -f "$E23_WORKSPACE/suites/$E23_ID/suite_context.json" ]]; then E23_RESUME=(--resume); fi
exec "$E23_TRAIN" -u "$E23_RUNTIME/code/e2_e3/pipeline.py" \
  --workspace "$E23_WORKSPACE" --run-id "$E23_ID" --started-at "$E23_STARTED" \
  --source-run "$E23_WORKSPACE/runs/ada_20261016_221122" \
  --independent "$E23_WORKSPACE/diagnostics/rank64_12k_20261021_183439_public_eval_v1/independent_dev400.json" \
  --train-python "$E23_TRAIN" --eval-python "$E23_EVAL" --data-python "$E23_DATA" \
  "${E23_RESUME[@]}" "$@"
