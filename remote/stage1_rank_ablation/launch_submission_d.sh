#!/usr/bin/env bash
set -euo pipefail
umask 077
D_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D_WORKSPACE="${GENIAC_WORKSPACE:-$HOME/geniac_math}"
D_SOURCE="${1:?Usage: launch_submission_d.sh SOURCE_RANK}"
D_SOURCE="$(cd "$D_SOURCE" && pwd)"
mkdir -p "$D_WORKSPACE"
exec 9>"$D_WORKSPACE/.launch.lock"
flock -n 9 || { echo 'Another experiment is active'; exit 1; }
D_ROOT="$D_WORKSPACE/submissions/$(basename "$D_SOURCE")_D"
mkdir -p "$D_ROOT/private_logs"
D_EVAL="${GENIAC_EVAL_PYTHON:-$D_WORKSPACE/.venv-eval/bin/python}"
D_DATA="$D_WORKSPACE/.venv-submission-data/bin/python"
test -x "$D_EVAL"
test -x "$D_DATA"
if [[ ! -d "$D_ROOT/code" ]]; then
  rm -rf "$D_ROOT/code.partial"
  mkdir -p "$D_ROOT/code.partial"
  for D_DIR in stage1_rank_ablation math_training rank64_12k; do
    mkdir -p "$D_ROOT/code.partial/$D_DIR"
    find "$D_HERE/../$D_DIR" -maxdepth 1 -type f \( -name '*.py' -o -name '*.json' -o -name '*.txt' -o -name '*.jinja' \) \
      -exec cp '{}' "$D_ROOT/code.partial/$D_DIR/" \;
  done
  cp -R "$D_HERE/../math_training/submission_assets" "$D_ROOT/code.partial/math_training/"
  mv "$D_ROOT/code.partial" "$D_ROOT/code"
fi
export HF_HOME="${HF_HOME:-$D_WORKSPACE/hf_cache}"
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
D_SCRIPT="$D_ROOT/code/stage1_rank_ablation/submit_d.py"
for D_PHASE in prepare fetch generate; do
  echo "PHASE D_submission_$D_PHASE"
  D_PYTHON="$D_EVAL"
  [[ "$D_PHASE" != fetch ]] || D_PYTHON="$D_DATA"
  if ! "$D_PYTHON" -u "$D_SCRIPT" "$D_PHASE" --source-rank "$D_SOURCE" --root "$D_ROOT" \
      >> "$D_ROOT/private_logs/$D_PHASE.log" 2>&1; then
    echo "FAILED: $D_PHASE; inspect locally: $D_ROOT/private_logs/$D_PHASE.log"
    exit 1
  fi
done
echo "SUBMISSION_READY $D_ROOT/submission.json"
echo "FORMAT_CHECK $D_ROOT/format_check.json"
