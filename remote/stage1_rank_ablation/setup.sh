#!/usr/bin/env bash
set -euo pipefail
RANK_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RANK_WORKSPACE="$1"
source "$RANK_HERE/../math_training/bootstrap_python.sh"
select_bootstrap_python "$RANK_WORKSPACE" "$(( $(date +%s) + 7200 ))"
RANK_ENV="$RANK_WORKSPACE/.venv-rank-ablation"
if [[ ! -x "$RANK_ENV/bin/python" ]]; then
  if [[ "$TASK_MANAGED" == 1 ]]; then
    "$TASK_UV" venv --python 3.11 "$RANK_ENV"
  else
    "$TASK_PYTHON" -m venv "$RANK_ENV"
  fi
fi
RANK_HASH="$(sha256sum "$RANK_HERE/requirements-train.txt" | cut -d ' ' -f 1)"
if [[ ! -f "$RANK_ENV/.ready-$RANK_HASH" ]]; then
  "$RANK_ENV/bin/python" -m pip install --upgrade pip
  "$RANK_ENV/bin/python" -m pip install -r "$RANK_HERE/requirements-train.txt"
  # Optional accelerator, constrained so it cannot replace the common training stack.
  if ! "$RANK_ENV/bin/python" -m pip install -c "$RANK_HERE/requirements-train.txt" \
       'unsloth==2026.10.5' 'unsloth_zoo==2026.10.5' 'trl==1.15.0' 'datasets==4.7.0' \
       'xformers==0.0.32.post2' 'torchvision==0.23.0'; then
    echo 'Unsloth install unavailable; measured HF QLoRA candidates remain available.'
  fi
  # Only a binary wheel: do not spend time compiling FlashAttention.
  RANK_WHEEL="$("$RANK_ENV/bin/python" - <<'PY'
import sys,torch
cp=f'cp{sys.version_info.major}{sys.version_info.minor}'
abi=str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()
print(f'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abi{abi}-{cp}-{cp}-linux_x86_64.whl')
PY
)"
  timeout 180 "$RANK_ENV/bin/python" -m pip install --no-deps "$RANK_WHEEL" || echo 'FlashAttention binary unavailable; SDPA remains available.'
  "$RANK_ENV/bin/python" -m pip check
  # pip check cannot detect undeclared Python API incompatibilities.
  "$RANK_ENV/bin/python" - <<'PYIMPORT'
import torch, torchao
from transformers import AutoModelForCausalLM, PreTrainedModel
from peft import PeftModel
from importlib.metadata import version
assert torch.__version__.split('+')[0] == '2.8.0'
assert version('torchao') == '0.13.0'
print('Training model imports OK', flush=True)
PYIMPORT
  touch "$RANK_ENV/.ready-$RANK_HASH"
fi
RANK_EVAL="${GENIAC_EVAL_PYTHON:-$RANK_WORKSPACE/.venv-eval/bin/python}"
if [[ ! -x "$RANK_EVAL" ]]; then
  echo 'Existing evaluation venv not found; creating a separate evaluation environment.'
  RANK_EVAL="$RANK_WORKSPACE/.venv-rank-eval/bin/python"
  "$RANK_ENV/bin/python" -m venv "$RANK_WORKSPACE/.venv-rank-eval"
  "$RANK_EVAL" -m pip install -r "$RANK_HERE/../math_training/requirements-eval.txt"
fi
"$RANK_EVAL" - <<'PY'
from importlib.metadata import version
for p,v in [('vllm','0.10.2'),('torch','2.8.0'),('transformers','4.56.2'),('math-verify','0.9.0')]:
    assert version(p).split('+')[0]==v,(p,version(p),v)
PY
printf '%s\n' "$RANK_EVAL" > "$RANK_ENV/eval-python.txt"
