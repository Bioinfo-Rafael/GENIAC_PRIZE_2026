#!/usr/bin/env bash
# Sourced by launch.sh. TASK_PYTHON/TASK_UV/TASK_MANAGED are outputs.

supported_python() {
  "$1" -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)' \
    >/dev/null 2>&1
}

select_bootstrap_python() {
  local task_workspace="$1" task_deadline="$2" candidate remaining uv_path
  TASK_PYTHON=''
  TASK_UV=''
  TASK_MANAGED=0

  if [[ -n "${GENIAC_BOOTSTRAP_PYTHON:-}" ]]; then
    if ! supported_python "$GENIAC_BOOTSTRAP_PYTHON"; then
      echo "GENIAC_BOOTSTRAP_PYTHON=$GENIAC_BOOTSTRAP_PYTHON is unavailable or not Python 3.10–3.12." >&2
      return 2
    fi
    TASK_PYTHON="$GENIAC_BOOTSTRAP_PYTHON"
  else
    for candidate in python3.11 python3.12 python3.10 python3; do
      if command -v "$candidate" >/dev/null 2>&1 && supported_python "$candidate"; then
        TASK_PYTHON="$(command -v "$candidate")"
        break
      fi
    done
  fi
  if [[ -n "$TASK_PYTHON" ]]; then
    echo "Using bootstrap Python: $TASK_PYTHON ($("$TASK_PYTHON" --version 2>&1))"
    return 0
  fi

  # An existing managed Python can be reused even when the host's python3 is 3.13+.
  if [[ -n "${GENIAC_UV:-}" ]]; then
    uv_path="$GENIAC_UV"
  elif [[ -x "$task_workspace/.uv/bin/uv" ]]; then
    uv_path="$task_workspace/.uv/bin/uv"
  else
    uv_path="$(command -v uv || true)"
  fi
  if [[ -z "$uv_path" ]]; then
    if ! command -v curl >/dev/null 2>&1; then
      echo 'No supported Python or uv was found; install Python 3.11 or provide curl for local uv bootstrap.' >&2
      return 2
    fi
    echo 'No Python 3.10–3.12 found; installing uv locally to provision Python 3.11.'
    mkdir -p "$task_workspace/.uv/bin"
    remaining=$((task_deadline - $(date +%s) - 60))
    if (( remaining <= 0 )); then echo 'Budget exhausted before uv setup.' >&2; return 75; fi
    timeout "$remaining" curl -LsSf https://astral.sh/uv/0.13.1/install.sh \
      -o "$task_workspace/.uv/install.sh"
    remaining=$((task_deadline - $(date +%s) - 60))
    if (( remaining <= 0 )); then echo 'Budget exhausted during uv setup.' >&2; return 75; fi
    timeout "$remaining" env UV_UNMANAGED_INSTALL="$task_workspace/.uv/bin" \
      sh "$task_workspace/.uv/install.sh"
    uv_path="$task_workspace/.uv/bin/uv"
  fi
  if [[ ! -x "$uv_path" ]]; then echo "uv executable is unavailable: $uv_path" >&2; return 2; fi
  TASK_UV="$uv_path"
  TASK_MANAGED=1
  export UV_PYTHON_INSTALL_DIR="$task_workspace/.uv/python"
  export UV_CACHE_DIR="$task_workspace/.uv/cache"
  echo "Using uv-managed Python 3.11 via $TASK_UV"
}
