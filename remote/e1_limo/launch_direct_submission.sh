#!/usr/bin/env bash
set -euo pipefail
umask 077
E1_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E1_SOURCE="${1:?Usage: launch_direct_submission.sh /absolute/path/to/source-suite}"
E1_SOURCE="$(cd "$E1_SOURCE" && pwd)"
E1_WORKSPACE="$(cd "$E1_SOURCE/../.." && pwd)"
exec 9>"$E1_WORKSPACE/.launch.lock"
flock -n 9 || { echo 'Another experiment is active'; exit 1; }
E1_OUT="$E1_SOURCE/direct_submission"
mkdir -p "$E1_OUT"
if [[ ! -d "$E1_OUT/code" ]]; then
  rm -rf "$E1_OUT/code.partial"
  mkdir -p "$E1_OUT/code.partial/e1_limo" "$E1_OUT/code.partial/rank64_12k"
  cp "$E1_HERE/direct_submit.py" "$E1_HERE/e1_common.py" "$E1_HERE/launch_direct_submission.sh" "$E1_OUT/code.partial/e1_limo/"
  cp -R "$E1_HERE/../math_training" "$E1_OUT/code.partial/math_training"
  cp "$E1_HERE/../rank64_12k/official_prompt.py" "$E1_OUT/code.partial/rank64_12k/"
  mv "$E1_OUT/code.partial" "$E1_OUT/code"
fi
export HF_HOME="${HF_HOME:-$E1_WORKSPACE/hf_cache}"
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
exec "${GENIAC_TRAIN_PYTHON:-$E1_WORKSPACE/.venv-train/bin/python}" -u \
  "$E1_OUT/code/e1_limo/direct_submit.py" run --source-suite "$E1_SOURCE"
