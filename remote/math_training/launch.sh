#!/usr/bin/env bash
# Intended for: nohup bash remote/math_training/launch.sh ... > console.log 2>&1 &
set -euo pipefail
TASK_CODE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_WORKSPACE="${GENIAC_WORKSPACE:-$HOME/geniac_math}"
TASK_STARTED_AT="$(date +%s)"
TASK_BUDGET=180
TASK_ARGS=("$@")
for ((i=0; i<${#TASK_ARGS[@]}; i++)); do
  case "${TASK_ARGS[$i]}" in
    --workspace) TASK_WORKSPACE="${TASK_ARGS[$((i+1))]}" ;;
    --budget-minutes) TASK_BUDGET="${TASK_ARGS[$((i+1))]}" ;;
  esac
done
mkdir -p "$TASK_WORKSPACE"
# Lock also covers installation; two background launches must not edit the same venv.
exec 9>"$TASK_WORKSPACE/.launch.lock"
flock -n 9 || { echo 'Another launch is active in this workspace'; exit 1; }
TASK_PYTHON="${GENIAC_BOOTSTRAP_PYTHON:-python3}"
"$TASK_PYTHON" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "Use Python 3.10–3.12 (recommended 3.11)"'
TASK_TRAIN_ENV="$TASK_WORKSPACE/.venv-train"
TASK_EVAL_ENV="$TASK_WORKSPACE/.venv-eval"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME="${HF_HOME:-$TASK_WORKSPACE/hf_cache}"

install_env() {
  local env_dir="$1" requirements="$2" marker expected remaining
  marker="$env_dir/.requirements.sha256"
  expected="$(sha256sum "$requirements" | cut -d ' ' -f 1)"
  if [[ -f "$marker" ]] && [[ "$(cat "$marker")" == "$expected" ]]; then return; fi
  if [[ ! -x "$env_dir/bin/python" ]]; then "$TASK_PYTHON" -m venv "$env_dir"; fi
  remaining=$((TASK_BUDGET * 60 - $(date +%s) + TASK_STARTED_AT - 60))
  if ((remaining <= 0)); then echo 'Budget exhausted during environment setup'; exit 75; fi
  timeout "$remaining" "$env_dir/bin/python" -m pip install 'pip==25.2'
  remaining=$((TASK_BUDGET * 60 - $(date +%s) + TASK_STARTED_AT - 60))
  if ((remaining <= 0)); then exit 75; fi
  timeout "$remaining" "$env_dir/bin/python" -m pip install 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cu128
  remaining=$((TASK_BUDGET * 60 - $(date +%s) + TASK_STARTED_AT - 60))
  if ((remaining <= 0)); then exit 75; fi
  timeout "$remaining" "$env_dir/bin/python" -m pip install -r "$requirements"
  "$env_dir/bin/python" -m pip check
  printf '%s\n' "$expected" > "$marker"
}
install_env "$TASK_TRAIN_ENV" "$TASK_CODE_DIR/requirements-train.txt"
install_env "$TASK_EVAL_ENV" "$TASK_CODE_DIR/requirements-eval.txt"
exec "$TASK_TRAIN_ENV/bin/python" -u "$TASK_CODE_DIR/run.py" \
  --workspace "$TASK_WORKSPACE" --eval-python "$TASK_EVAL_ENV/bin/python" \
  --started-at "$TASK_STARTED_AT" "${TASK_ARGS[@]}"
