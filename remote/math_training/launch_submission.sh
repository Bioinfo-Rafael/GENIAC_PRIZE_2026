#!/usr/bin/env bash
set -euo pipefail
TASK_CODE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_WORKSPACE="${GENIAC_WORKSPACE:-$HOME/geniac_math}"
mkdir -p "$TASK_WORKSPACE"
exec 9>"$TASK_WORKSPACE/.launch.lock"
flock -n 9 || { echo 'Another launch is active in this workspace'; exit 1; }
export HF_HOME="${HF_HOME:-$TASK_WORKSPACE/hf_cache}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
TASK_TRAIN_PYTHON="$TASK_WORKSPACE/.venv-train/bin/python"
TASK_DATA_ENV="$TASK_WORKSPACE/.venv-submission-data"
test -x "$TASK_TRAIN_PYTHON"
test -x "$TASK_WORKSPACE/.venv-eval/bin/python"
if [[ ! -x "$TASK_DATA_ENV/bin/python" ]]; then
  "$TASK_TRAIN_PYTHON" -m venv "$TASK_DATA_ENV"
fi
TASK_EXPECTED="$(sha256sum "$TASK_CODE_DIR/requirements-submission-data.txt" | cut -d ' ' -f 1)"
if [[ ! -f "$TASK_DATA_ENV/.requirements.sha256" ]] || [[ "$(cat "$TASK_DATA_ENV/.requirements.sha256")" != "$TASK_EXPECTED" ]]; then
  "$TASK_DATA_ENV/bin/python" -m pip install -r "$TASK_CODE_DIR/requirements-submission-data.txt"
  "$TASK_DATA_ENV/bin/python" -m pip check
  printf '%s\n' "$TASK_EXPECTED" > "$TASK_DATA_ENV/.requirements.sha256"
fi
exec "$TASK_TRAIN_PYTHON" -u "$TASK_CODE_DIR/submission.py" run --workspace "$TASK_WORKSPACE" "$@"
