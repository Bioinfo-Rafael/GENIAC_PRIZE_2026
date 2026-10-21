#!/usr/bin/env bash
set -euo pipefail
RANK_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RANK_WORKSPACE="${GENIAC_WORKSPACE:-$HOME/geniac_math}"
RANK_ID="${1:?Usage: bash launch.sh RUN_ID [--reference EVALUATION_A_DIR]}"
shift
[[ "$RANK_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid run ID'; exit 2; }
mkdir -p "$RANK_WORKSPACE"
exec 9>"$RANK_WORKSPACE/.launch.lock"
flock -n 9 || { echo 'Another launch is active in this workspace'; exit 1; }
RANK_ROOT="$RANK_WORKSPACE/rank_ablation/$RANK_ID"
mkdir -p "$RANK_ROOT"
# Reuse the immutable code snapshot on resume. Never overwrite old experiment code.
if [[ ! -d "$RANK_ROOT/code" ]]; then
  mkdir -p "$RANK_ROOT/code.partial/stage1_rank_ablation" "$RANK_ROOT/code.partial/math_training"
  cp "$RANK_HERE/"*.py "$RANK_HERE/"*.json "$RANK_HERE/"*.txt "$RANK_HERE/"*.sh "$RANK_ROOT/code.partial/stage1_rank_ablation/"
  for RANK_FILE in core.py worker.py grading.py suite_utils.py input_manifest.json chat_template.jinja bootstrap_python.sh requirements-eval.txt; do
    cp "$RANK_HERE/../math_training/$RANK_FILE" "$RANK_ROOT/code.partial/math_training/"
  done
  mv "$RANK_ROOT/code.partial" "$RANK_ROOT/code"
fi
RANK_CODE="$RANK_ROOT/code/stage1_rank_ablation"
RANK_STARTED=$(date +%s)
bash "$RANK_CODE/setup.sh" "$RANK_WORKSPACE"
printf '{"environment_setup_seconds":%s}\n' "$(( $(date +%s) - RANK_STARTED ))" > "$RANK_ROOT/setup_time.json"
RANK_ENV="$RANK_WORKSPACE/.venv-rank-ablation"
exec "$RANK_ENV/bin/python" -u "$RANK_CODE/run.py" --root "$RANK_ROOT" \
  --cache "$RANK_WORKSPACE/inputs" --eval-python "$(cat "$RANK_ENV/eval-python.txt")" "$@"
